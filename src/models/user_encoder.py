"""
Neural User Encoder (Phase 3)
==============================

Represents each user as a learned embedding using interaction history
and preference information.

Architecture:
    Interaction history (song embeddings)
        ↓
    Weighted aggregation using UPS event scores
        ↓
    Concatenate: [aggregated_song_emb | artist_pref_features | genre_pref_features]
        ↓
    MLP
        ↓
    User embedding (same dimensionality as song embedding)

The user representation captures:
- Interacted songs (via song embeddings)
- Positive/negative behavior (via UPS-weighted aggregation)
- Repeated engagement (via interaction scores)
- Artist preference (top-K artist preference scores)
- Genre preference (top-K genre preference scores)
- Recency (via decay-weighted scores)

Cold-start users get a safe zero-vector fallback.
"""

import os
import json
from typing import Dict, List, Optional, Any, Tuple

import torch
import torch.nn as nn
import numpy as np

from src.interactions.ups import calculate_event_score


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_TOP_K_ARTISTS = 10
DEFAULT_TOP_K_GENRES = 10


# ---------------------------------------------------------------------------
# User History Representation
# ---------------------------------------------------------------------------

class UserHistoryBuilder:
    """Constructs a feature vector for the user tower input.
    
    The input is composed of three parts:
    1. Aggregated song embedding (weighted by UPS scores)
    2. Artist preference feature vector (top-K preference scores)
    3. Genre preference feature vector (top-K preference scores)
    
    Total input dimension = song_embedding_dim + top_k_artists + top_k_genres
    """

    def __init__(
        self,
        song_embedding_dim: int = 64,
        top_k_artists: int = DEFAULT_TOP_K_ARTISTS,
        top_k_genres: int = DEFAULT_TOP_K_GENRES,
    ):
        self.song_embedding_dim = song_embedding_dim
        self.top_k_artists = top_k_artists
        self.top_k_genres = top_k_genres

    @property
    def input_dim(self) -> int:
        """Total input dimension for the user encoder MLP."""
        return self.song_embedding_dim + self.top_k_artists + self.top_k_genres

    def build_user_features(
        self,
        song_embeddings: np.ndarray,
        interaction_scores: np.ndarray,
        artist_pref_scores: List[float],
        genre_pref_scores: List[float],
    ) -> np.ndarray:
        """Build the user feature vector.
        
        Parameters
        ----------
        song_embeddings : ndarray of shape (n_interactions, song_embedding_dim)
            Embeddings of songs the user has interacted with.
        interaction_scores : ndarray of shape (n_interactions,)
            UPS event score for each interaction (can be positive or negative).
        artist_pref_scores : list of floats
            Top-K artist preference scores (normalized, in [-1, 1]).
        genre_pref_scores : list of floats
            Top-K genre preference scores (normalized, in [-1, 1]).
        
        Returns
        -------
        ndarray of shape (input_dim,)
        """
        # 1. Weighted aggregation of song embeddings
        agg_song_emb = self._weighted_aggregate(
            song_embeddings, interaction_scores
        )

        # 2. Pad/truncate artist and genre preference scores to fixed size
        artist_features = self._pad_or_truncate(
            artist_pref_scores, self.top_k_artists
        )
        genre_features = self._pad_or_truncate(
            genre_pref_scores, self.top_k_genres
        )

        # 3. Concatenate
        return np.concatenate([agg_song_emb, artist_features, genre_features])

    def build_cold_start_features(self) -> np.ndarray:
        """Return a safe zero-vector for cold-start (new) users.
        
        This prevents crashing when a user has no interaction history.
        """
        return np.zeros(self.input_dim, dtype=np.float32)

    def _weighted_aggregate(
        self,
        embeddings: np.ndarray,
        scores: np.ndarray,
    ) -> np.ndarray:
        """Compute weighted average of embeddings using interaction scores.
        
        Uses softmax-like weighting:
            weight_i = exp(score_i) / sum(exp(score_j))
        
        This ensures:
        - Positive interactions get higher weight
        - Negative interactions get lower weight
        - All weights are positive (no sign issues)
        - Numerically stable via max subtraction
        """
        if len(embeddings) == 0:
            return np.zeros(self.song_embedding_dim, dtype=np.float32)

        # Softmax-like weighting for stability
        scores_shifted = scores - np.max(scores)  # numerical stability
        exp_scores = np.exp(scores_shifted)
        weights = exp_scores / (exp_scores.sum() + 1e-8)

        # Weighted sum
        weighted = (embeddings.T * weights).T
        return weighted.sum(axis=0).astype(np.float32)

    @staticmethod
    def _pad_or_truncate(values: List[float], target_length: int) -> np.ndarray:
        """Pad with zeros or truncate to fixed length."""
        arr = np.zeros(target_length, dtype=np.float32)
        n = min(len(values), target_length)
        if n > 0:
            arr[:n] = values[:n]
        return arr


# ---------------------------------------------------------------------------
# User Encoder Network
# ---------------------------------------------------------------------------

class UserEncoder(nn.Module):
    """Neural network that maps user features to a learned embedding.
    
    The input dimension is:
        song_embedding_dim + top_k_artists + top_k_genres
    
    Parameters
    ----------
    input_dim : int
        Input feature dimension (from UserHistoryBuilder.input_dim).
    hidden_dim : int
        Width of hidden layers (default: 128).
    embedding_dim : int
        Dimensionality of the output user embedding (default: 64).
        Should match song embedding dim for similarity computation.
    normalize : bool
        If True, L2-normalize the output embedding (default: True).
    dropout : float
        Dropout probability (default: 0.1).
    """

    def __init__(
        self,
        input_dim: int = 84,  # 64 + 10 + 10
        hidden_dim: int = 128,
        embedding_dim: int = 64,
        normalize: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.do_normalize = normalize
        self.dropout_rate = dropout

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Parameters
        ----------
        x : Tensor of shape (batch_size, input_dim)
        
        Returns
        -------
        Tensor of shape (batch_size, embedding_dim)
        """
        emb = self.network(x)
        if self.do_normalize:
            emb = nn.functional.normalize(emb, p=2, dim=1)
        return emb

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def encode_user(self, user_features: np.ndarray) -> np.ndarray:
        """Encode a single user.
        
        Parameters
        ----------
        user_features : ndarray of shape (input_dim,) or (1, input_dim)
        
        Returns
        -------
        ndarray of shape (embedding_dim,)
        """
        was_training = self.training
        self.eval()

        if user_features.ndim == 1:
            user_features = user_features.reshape(1, -1)

        with torch.no_grad():
            x = torch.from_numpy(user_features.astype(np.float32))
            emb = self.forward(x)

        if was_training:
            self.train()

        return emb.numpy().squeeze(0)

    def encode_users(self, user_features_batch: np.ndarray) -> np.ndarray:
        """Encode a batch of users.
        
        Parameters
        ----------
        user_features_batch : ndarray of shape (n_users, input_dim)
        
        Returns
        -------
        ndarray of shape (n_users, embedding_dim)
        """
        was_training = self.training
        self.eval()

        with torch.no_grad():
            x = torch.from_numpy(user_features_batch.astype(np.float32))
            emb = self.forward(x)

        if was_training:
            self.train()

        return emb.numpy()

    # ------------------------------------------------------------------
    # Config / checkpoint management
    # ------------------------------------------------------------------

    def get_config(self) -> Dict[str, Any]:
        """Return the model configuration as a JSON-serializable dict."""
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "embedding_dim": self.embedding_dim,
            "normalize": self.do_normalize,
            "dropout": self.dropout_rate,
        }

    def save(self, directory: str) -> None:
        """Save model checkpoint and configuration."""
        os.makedirs(directory, exist_ok=True)

        weights_path = os.path.join(directory, "user_encoder.pt")
        config_path = os.path.join(directory, "user_encoder_config.json")

        torch.save(self.state_dict(), weights_path)
        with open(config_path, "w") as f:
            json.dump(self.get_config(), f, indent=2)

    @classmethod
    def load(cls, directory: str) -> "UserEncoder":
        """Load model from checkpoint and configuration."""
        config_path = os.path.join(directory, "user_encoder_config.json")
        weights_path = os.path.join(directory, "user_encoder.pt")

        with open(config_path, "r") as f:
            config = json.load(f)

        model = cls(
            input_dim=config["input_dim"],
            hidden_dim=config["hidden_dim"],
            embedding_dim=config["embedding_dim"],
            normalize=config["normalize"],
            dropout=config["dropout"],
        )

        model.load_state_dict(torch.load(weights_path, weights_only=True))
        model.eval()
        return model
