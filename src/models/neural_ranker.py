"""
Neural Ranking / Re-ranking (Phase 7)
=====================================

Neural Re-ranker that takes a candidate pool from FAISS and scores them
using a rich set of features (user embedding, song embedding, preferences, UPS).

Architecture:
    Candidate Features -> Dense -> ReLU -> Dense -> ReLU -> Score (Logits)
"""

import os
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Any, Optional

from src.interactions.models import UserInteraction, InteractionStore
from src.interactions.ups import calculate_user_song_preference
from src.interactions.preferences import PreferenceStore, SongMetadataLookup

class CandidateFeatureExtractor:
    """Extracts ranking features for a (user, song) candidate pair without leakage."""
    
    def __init__(
        self,
        metadata_lookup: SongMetadataLookup,
        song_embeddings: np.ndarray,
        song_id_to_row: Dict[str, int],
    ):
        self.metadata = metadata_lookup
        self.song_embeddings = song_embeddings
        self.song_id_to_row = song_id_to_row
        
    def extract_features(
        self,
        user_id: str,
        user_embedding: np.ndarray,
        song_id: str,
        user_history: List[UserInteraction],
    ) -> Optional[np.ndarray]:
        """Extract a 133-dimensional feature vector for ranking.
        
        Features:
        - User Embedding (64)
        - Song Embedding (64)
        - Two-Tower Similarity (1)
        - Artist Preference (1)
        - Genre Preference (1)
        - Interaction Count (1)
        - Historical UPS (1)
        """
        row = self.song_id_to_row.get(song_id)
        if row is None:
            return None
            
        song_emb = self.song_embeddings[row]
        
        # 1. Similarity
        similarity = np.dot(user_embedding, song_emb)
        
        # 2. Preferences (built purely from history to prevent leakage)
        istore = InteractionStore()
        istore._events.extend(user_history)
        pref_store = PreferenceStore(istore, self.metadata)
        pref_store.rebuild_all()
        
        artist_pref = 0.0
        artist_info = self.metadata.get_artist(song_id)
        if artist_info:
            artist_id = artist_info[0]
            artist_pref = pref_store.get_artist_preference(user_id, artist_id)
            if artist_pref is not None:
                artist_pref = artist_pref.preference_score
            else:
                artist_pref = 0.0
            
        genre_pref = 0.0
        genres = self.metadata.get_genres(song_id)
        if genres:
            genre_prefs_list = [pref_store.get_genre_preference(user_id, g) for g in genres]
            valid_prefs = [p.preference_score for p in genre_prefs_list if p is not None]
            genre_pref = max(valid_prefs, default=0.0)
            
        # 3. Direct song history
        song_history = [e for e in user_history if e.song_id == song_id]
        interaction_count = len(song_history)
        historical_ups = calculate_user_song_preference(user_id, song_id, song_history)
        
        features = np.concatenate([
            user_embedding.flatten(),
            song_emb.flatten(),
            np.array([
                similarity,
                artist_pref,
                genre_pref,
                float(interaction_count),
                float(historical_ups.preference_score)
            ], dtype=np.float32)
        ])
        
        return features.astype(np.float32)

class NeuralRanker(nn.Module):
    """Feed-forward neural network for scoring candidates."""
    
    def __init__(self, input_dim: int = 133, hidden_dims: List[int] = [128, 64], dropout: float = 0.1):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
            
        # Final scoring layer (1 output, no activation because we use BCEWithLogitsLoss)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logits for ranking."""
        return self.network(x).squeeze(-1)
        
    def predict_score(self, x: np.ndarray) -> np.ndarray:
        """Inference helper."""
        was_training = self.training
        self.eval()
        with torch.no_grad():
            tensor_x = torch.from_numpy(x).float()
            if tensor_x.ndim == 1:
                tensor_x = tensor_x.unsqueeze(0)
            scores = self.forward(tensor_x).numpy()
        if was_training:
            self.train()
        return scores

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)
        
    @classmethod
    def load(cls, path: str, input_dim: int = 133) -> "NeuralRanker":
        model = cls(input_dim=input_dim)
        model.load_state_dict(torch.load(path))
        return model

class RankerTrainer:
    """Trainer for the NeuralRanker using BCEWithLogitsLoss."""
    
    def __init__(self, model: NeuralRanker, lr: float = 1e-3):
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.BCEWithLogitsLoss()
        
    def train_epoch(self, features: np.ndarray, labels: np.ndarray, batch_size: int = 256) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        indices = np.random.permutation(len(features))
        
        for start_idx in range(0, len(features), batch_size):
            batch_idx = indices[start_idx:start_idx + batch_size]
            
            x = torch.from_numpy(features[batch_idx]).float()
            y = torch.from_numpy(labels[batch_idx]).float()
            
            self.optimizer.zero_grad()
            logits = self.model(x)
            loss = self.criterion(logits, y)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            
        return total_loss / max(1, n_batches)
        
    def evaluate(self, features: np.ndarray, labels: np.ndarray, batch_size: int = 256) -> float:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for start_idx in range(0, len(features), batch_size):
                end_idx = min(start_idx + batch_size, len(features))
                
                x = torch.from_numpy(features[start_idx:end_idx]).float()
                y = torch.from_numpy(labels[start_idx:end_idx]).float()
                
                logits = self.model(x)
                loss = self.criterion(logits, y)
                
                total_loss += loss.item()
                n_batches += 1
                
        return total_loss / max(1, n_batches)
