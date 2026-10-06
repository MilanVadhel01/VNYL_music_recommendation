"""
Two-Tower Neural Recommender (Phase 4)
=======================================

Trains a two-tower architecture where the user tower produces user embeddings
and the song tower produces song embeddings, such that compatible (user, song)
pairs have higher dot-product similarity.

Architecture:
                USER TOWER
                     │
    User History ────┤
    Artist Prefs ────┤
    Genre Prefs  ────┤
                     ↓
               User Embedding
                     │
                     │ dot-product similarity
                     │
               Song Embedding
                     ↑
    13 Features  ────┤
                     │
                SONG TOWER

Training Objective:
    Contrastive loss (InfoNCE / in-batch negatives):
        For each (user, positive_song) pair, treat all other songs in the
        batch as negatives. Maximize similarity to the positive song relative
        to negatives.

Training Data Construction:
    Positive signals: LIKE, REPLAY, COMPLETE, ADD_TO_PLAYLIST, high completion
    Negative signals: DISLIKE, SKIP <10s, SKIP <30s
    Neutral signals (excluded): PLAY, PAUSE, RESUME, SEARCH

The training pipeline avoids data leakage by ensuring the user representation
for each training example uses only interactions that occurred BEFORE the
target interaction.
"""

import os
import json
import time
import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

from src.interactions.models import ActionType, UserInteraction, InteractionStore
from src.interactions.ups import calculate_event_score
from src.interactions.preferences import (
    SongMetadataLookup,
    PreferenceStore,
    normalize_preference,
    NORM_SCALE,
)
from src.models.song_encoder import SongEncoder, INPUT_DIM, CONTENT_FEATURE_ORDER
from src.models.user_encoder import UserEncoder, UserHistoryBuilder


# ---------------------------------------------------------------------------
# Training Data Configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """Configuration for the two-tower training pipeline.
    
    All parameters are configurable for reproducibility and experimentation.
    """
    # Architecture
    song_embedding_dim: int = 64
    user_embedding_dim: int = 64
    song_hidden_dim: int = 128
    user_hidden_dim: int = 128
    top_k_artists: int = 10
    top_k_genres: int = 10

    # Training
    batch_size: int = 256
    learning_rate: float = 1e-3
    epochs: int = 10
    weight_decay: float = 1e-5
    temperature: float = 0.07  # InfoNCE temperature

    # Data
    negative_samples: int = 0  # 0 = in-batch negatives only
    val_fraction: float = 0.2
    min_interactions_per_user: int = 2  # need at least 1 history + 1 target

    # Reproducibility
    random_seed: int = 42

    # Dropout
    dropout: float = 0.1


# ---------------------------------------------------------------------------
# Positive / Negative Signal Classification
# ---------------------------------------------------------------------------

POSITIVE_ACTIONS = {
    ActionType.LIKE,
    ActionType.REPLAY,
    ActionType.COMPLETE,
    ActionType.ADD_TO_PLAYLIST,
    ActionType.SHARE,
    ActionType.MOVE_SONG_TOP,
}

NEGATIVE_ACTIONS = {
    ActionType.DISLIKE,
    ActionType.SKIP,
    ActionType.REMOVE_LIKE,
    ActionType.REMOVE_FROM_PLAYLIST,
    ActionType.MOVE_SONG_BOTTOM,
}

# Skip with completion > 50% is mildly negative (-2), not strong signal
# Skip with completion > 90% is neutral (0)

def is_positive_interaction(event: UserInteraction) -> bool:
    """Determine if an interaction is a positive training signal."""
    if event.action in POSITIVE_ACTIONS:
        return True
    # PLAY with high completion is positive
    if event.action == ActionType.PLAY and event.completion_percentage is not None:
        if event.completion_percentage >= 75.0:
            return True
    return False


def is_negative_interaction(event: UserInteraction) -> bool:
    """Determine if an interaction is a negative training signal."""
    if event.action == ActionType.DISLIKE:
        return True
    if event.action == ActionType.SKIP:
        # Only strong skips are negative signals
        if event.listen_duration is not None and event.listen_duration < 30.0:
            return True
        if event.completion_percentage is not None and event.completion_percentage < 50.0:
            return True
        return False
    if event.action in {ActionType.REMOVE_LIKE, ActionType.REMOVE_FROM_PLAYLIST, ActionType.MOVE_SONG_BOTTOM}:
        return True
    return False


# ---------------------------------------------------------------------------
# Training Example
# ---------------------------------------------------------------------------

@dataclass
class TrainingExample:
    """A single training example for the two-tower model.
    
    Contains:
    - User features built from history BEFORE the target interaction
    - Song features for the target song
    - Label: 1.0 for positive, 0.0 for negative
    """
    user_features: np.ndarray   # shape: (user_input_dim,)
    song_features: np.ndarray   # shape: (13,)
    label: float                # 1.0 = positive, 0.0 = negative
    user_id: str
    song_id: str


# ---------------------------------------------------------------------------
# Training Dataset
# ---------------------------------------------------------------------------

class TwoTowerDataset(Dataset):
    """PyTorch Dataset for two-tower training examples."""

    def __init__(self, examples: List[TrainingExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, float]:
        ex = self.examples[idx]
        return (
            torch.from_numpy(ex.user_features.astype(np.float32)),
            torch.from_numpy(ex.song_features.astype(np.float32)),
            ex.label,
        )


# ---------------------------------------------------------------------------
# Training Data Builder
# ---------------------------------------------------------------------------

class TrainingDataBuilder:
    """Constructs training examples from interaction history.
    
    For each positive/negative target interaction:
    1. Build user representation from interactions BEFORE the target
    2. Extract song features for the target song
    3. Create a TrainingExample with the appropriate label
    
    This prevents data leakage — the user representation never sees
    future interactions.
    """

    def __init__(
        self,
        interaction_store: InteractionStore,
        metadata_lookup: SongMetadataLookup,
        song_features: np.ndarray,
        song_id_to_row: Dict[str, int],
        song_encoder: SongEncoder,
        config: TrainingConfig,
    ):
        self.interaction_store = interaction_store
        self.metadata = metadata_lookup
        self.song_features = song_features
        self.song_id_to_row = song_id_to_row
        self.song_encoder = song_encoder
        self.config = config

        self.history_builder = UserHistoryBuilder(
            song_embedding_dim=config.song_embedding_dim,
            top_k_artists=config.top_k_artists,
            top_k_genres=config.top_k_genres,
        )

    def build_examples(self) -> List[TrainingExample]:
        """Build training examples from all interaction history.
        
        Returns a list of TrainingExample sorted chronologically.
        """
        all_events = self.interaction_store.get_all_events()

        # Group events by user, sorted by timestamp
        events_by_user: Dict[str, List[UserInteraction]] = {}
        for e in all_events:
            events_by_user.setdefault(e.user_id, []).append(e)

        for uid in events_by_user:
            events_by_user[uid].sort(key=lambda e: e.timestamp)

        examples = []

        for user_id, user_events in events_by_user.items():
            if len(user_events) < self.config.min_interactions_per_user:
                continue

            for i, target_event in enumerate(user_events):
                # Determine if this is a positive or negative signal
                if is_positive_interaction(target_event):
                    label = 1.0
                elif is_negative_interaction(target_event):
                    label = 0.0
                else:
                    continue  # Skip neutral events

                # Get song features for the target
                song_row = self.song_id_to_row.get(target_event.song_id)
                if song_row is None:
                    continue  # Song not in feature matrix

                song_feats = self.song_features[song_row]

                # Build user representation from history BEFORE this event
                history = user_events[:i]  # strictly before target
                user_feats = self._build_user_features(user_id, history)

                examples.append(TrainingExample(
                    user_features=user_feats,
                    song_features=song_feats.astype(np.float32),
                    label=label,
                    user_id=user_id,
                    song_id=target_event.song_id,
                ))

        return examples

    def _build_user_features(
        self,
        user_id: str,
        history: List[UserInteraction],
    ) -> np.ndarray:
        """Build user feature vector from historical interactions."""
        if len(history) == 0:
            return self.history_builder.build_cold_start_features()

        # Get song embeddings for history
        song_embs = []
        scores = []
        artist_score_acc: Dict[str, float] = {}
        genre_score_acc: Dict[str, float] = {}

        for event in history:
            row = self.song_id_to_row.get(event.song_id)
            if row is None:
                continue

            # Get song embedding
            song_feat = self.song_features[row].astype(np.float32)
            song_emb = self.song_encoder.encode_song(song_feat)
            song_embs.append(song_emb)

            score = calculate_event_score(event)
            scores.append(score)

            # Accumulate artist preferences
            artist_info = self.metadata.get_artist(event.song_id)
            if artist_info:
                artist_id = artist_info[0]
                artist_score_acc[artist_id] = artist_score_acc.get(artist_id, 0) + score

            # Accumulate genre preferences
            genres = self.metadata.get_genres(event.song_id)
            for g in genres:
                genre_score_acc[g] = genre_score_acc.get(g, 0) + score

        if len(song_embs) == 0:
            return self.history_builder.build_cold_start_features()

        song_embs_arr = np.stack(song_embs)
        scores_arr = np.array(scores, dtype=np.float32)

        # Normalize artist/genre scores
        artist_prefs = sorted(artist_score_acc.values(), reverse=True)
        artist_prefs = [normalize_preference(s, NORM_SCALE) for s in artist_prefs]

        genre_prefs = sorted(genre_score_acc.values(), reverse=True)
        genre_prefs = [normalize_preference(s, NORM_SCALE) for s in genre_prefs]

        return self.history_builder.build_user_features(
            song_embs_arr, scores_arr, artist_prefs, genre_prefs
        )


# ---------------------------------------------------------------------------
# Two-Tower Model
# ---------------------------------------------------------------------------

class TwoTowerModel(nn.Module):
    """Two-tower neural recommender combining user and song encoders.
    
    Computes similarity(user_embedding, song_embedding) as the dot product.
    Trained with InfoNCE contrastive loss using in-batch negatives.
    """

    def __init__(
        self,
        song_encoder: SongEncoder,
        user_encoder: UserEncoder,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.song_encoder = song_encoder
        self.user_encoder = user_encoder
        self.temperature = temperature

    def forward(
        self,
        user_features: torch.Tensor,
        song_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute user and song embeddings.
        
        Returns
        -------
        user_emb : Tensor (batch_size, embedding_dim)
        song_emb : Tensor (batch_size, embedding_dim)
        """
        user_emb = self.user_encoder(user_features)
        song_emb = self.song_encoder(song_features)
        return user_emb, song_emb

    def compute_similarity(
        self,
        user_emb: torch.Tensor,
        song_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Compute similarity scores between user and song embeddings.
        
        Returns a (batch_size, batch_size) matrix of similarities.
        """
        return torch.mm(user_emb, song_emb.t()) / self.temperature

    def score_pairs(
        self,
        user_features: np.ndarray,
        song_features: np.ndarray,
    ) -> np.ndarray:
        """Score (user, song) pairs for inference.
        
        Parameters
        ----------
        user_features : ndarray (n_users, user_input_dim)
        song_features : ndarray (n_songs, 13)
        
        Returns
        -------
        ndarray (n_users, n_songs) of similarity scores
        """
        was_training = self.training
        self.eval()

        with torch.no_grad():
            u = torch.from_numpy(user_features.astype(np.float32))
            s = torch.from_numpy(song_features.astype(np.float32))

            user_emb = self.user_encoder(u)
            song_emb = self.song_encoder(s)

            # Dot product (without temperature for raw similarity)
            scores = torch.mm(user_emb, song_emb.t())

        if was_training:
            self.train()

        return scores.numpy()


# ---------------------------------------------------------------------------
# Loss Function
# ---------------------------------------------------------------------------

def infonce_loss(
    user_emb: torch.Tensor,
    song_emb: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """InfoNCE contrastive loss with in-batch negatives.
    
    For positive pairs, maximize similarity relative to all other songs in batch.
    For negative pairs, minimize similarity.
    
    Parameters
    ----------
    user_emb : Tensor (batch_size, embedding_dim)
    song_emb : Tensor (batch_size, embedding_dim)
    labels : Tensor (batch_size,) — 1.0 for positive, 0.0 for negative
    temperature : float
    
    Returns
    -------
    Scalar loss tensor
    """
    batch_size = user_emb.shape[0]

    # Similarity matrix: (batch_size, batch_size)
    logits = torch.mm(user_emb, song_emb.t()) / temperature

    # For each user, the positive is the diagonal element
    # Use only positive examples as anchors
    positive_mask = labels > 0.5
    if positive_mask.sum() == 0:
        return torch.tensor(0.0, requires_grad=True)

    # Cross-entropy where the target is the diagonal
    targets = torch.arange(batch_size, device=logits.device)

    # Only compute loss for positive anchors
    pos_logits = logits[positive_mask]
    pos_targets = targets[positive_mask]

    loss = F.cross_entropy(pos_logits, pos_targets)
    return loss


# ---------------------------------------------------------------------------
# Training Pipeline
# ---------------------------------------------------------------------------

class TwoTowerTrainer:
    """End-to-end training pipeline for the two-tower recommender.
    
    Usage:
        trainer = TwoTowerTrainer(config, interaction_store, metadata, ...)
        trainer.prepare_data()
        results = trainer.train()
        trainer.save_checkpoint(directory)
    """

    def __init__(
        self,
        config: TrainingConfig,
        interaction_store: InteractionStore,
        metadata_lookup: SongMetadataLookup,
        song_features: np.ndarray,
        song_id_to_row: Dict[str, int],
    ):
        self.config = config
        self.interaction_store = interaction_store
        self.metadata = metadata_lookup
        self.song_features = song_features
        self.song_id_to_row = song_id_to_row

        # Set random seeds
        torch.manual_seed(config.random_seed)
        np.random.seed(config.random_seed)

        # Initialize models
        self.song_encoder = SongEncoder(
            input_dim=INPUT_DIM,
            hidden_dim=config.song_hidden_dim,
            embedding_dim=config.song_embedding_dim,
            dropout=config.dropout,
        )

        user_input_dim = (
            config.song_embedding_dim +
            config.top_k_artists +
            config.top_k_genres
        )

        self.user_encoder = UserEncoder(
            input_dim=user_input_dim,
            hidden_dim=config.user_hidden_dim,
            embedding_dim=config.user_embedding_dim,
            dropout=config.dropout,
        )

        self.model = TwoTowerModel(
            self.song_encoder,
            self.user_encoder,
            temperature=config.temperature,
        )

        self.train_examples: List[TrainingExample] = []
        self.val_examples: List[TrainingExample] = []
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None

        self.train_losses: List[float] = []
        self.val_losses: List[float] = []

    def prepare_data(self) -> Dict[str, int]:
        """Build training examples and split into train/val.
        
        Returns
        -------
        Dictionary with data statistics.
        """
        # Build examples using frozen song encoder (pre-training embeddings)
        self.song_encoder.eval()

        builder = TrainingDataBuilder(
            self.interaction_store,
            self.metadata,
            self.song_features,
            self.song_id_to_row,
            self.song_encoder,
            self.config,
        )

        all_examples = builder.build_examples()

        # Sort by user_id then chronologically for reproducible split
        # Use chronological split: earlier examples → train, later → val
        n_val = max(1, int(len(all_examples) * self.config.val_fraction))
        n_train = len(all_examples) - n_val

        self.train_examples = all_examples[:n_train]
        self.val_examples = all_examples[n_train:]

        # Count positive/negative
        n_pos_train = sum(1 for e in self.train_examples if e.label > 0.5)
        n_neg_train = sum(1 for e in self.train_examples if e.label <= 0.5)
        n_pos_val = sum(1 for e in self.val_examples if e.label > 0.5)
        n_neg_val = sum(1 for e in self.val_examples if e.label <= 0.5)

        # Create DataLoaders
        self.train_loader = DataLoader(
            TwoTowerDataset(self.train_examples),
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True if len(self.train_examples) > self.config.batch_size else False,
        )

        self.val_loader = DataLoader(
            TwoTowerDataset(self.val_examples),
            batch_size=self.config.batch_size,
            shuffle=False,
        )

        stats = {
            "total_examples": len(all_examples),
            "train_examples": len(self.train_examples),
            "val_examples": len(self.val_examples),
            "train_positive": n_pos_train,
            "train_negative": n_neg_train,
            "val_positive": n_pos_val,
            "val_negative": n_neg_val,
        }

        return stats

    def train(self) -> Dict[str, Any]:
        """Run the full training loop.
        
        Returns
        -------
        Dictionary with training results and metrics.
        """
        if self.train_loader is None:
            raise RuntimeError("Call prepare_data() before train()")

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        self.train_losses = []
        self.val_losses = []

        start_time = time.time()

        for epoch in range(self.config.epochs):
            # Training
            train_loss = self._train_epoch(optimizer)
            self.train_losses.append(train_loss)

            # Validation
            val_loss = self._validate_epoch()
            self.val_losses.append(val_loss)

        elapsed = time.time() - start_time

        return {
            "epochs": self.config.epochs,
            "final_train_loss": self.train_losses[-1] if self.train_losses else None,
            "final_val_loss": self.val_losses[-1] if self.val_losses else None,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "elapsed_seconds": elapsed,
        }

    def _train_epoch(self, optimizer: torch.optim.Optimizer) -> float:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for user_feats, song_feats, labels in self.train_loader:
            optimizer.zero_grad()

            user_emb, song_emb = self.model(user_feats, song_feats)

            loss = infonce_loss(
                user_emb, song_emb, labels,
                temperature=self.config.temperature,
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def _validate_epoch(self) -> float:
        """Run one validation epoch."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for user_feats, song_feats, labels in self.val_loader:
                user_emb, song_emb = self.model(user_feats, song_feats)

                loss = infonce_loss(
                    user_emb, song_emb, labels,
                    temperature=self.config.temperature,
                )

                total_loss += loss.item()
                n_batches += 1

        return total_loss / max(n_batches, 1)

    # ------------------------------------------------------------------
    # Evaluation Metrics
    # ------------------------------------------------------------------

    def evaluate(self, top_k: int = 10) -> Dict[str, float]:
        """Evaluate the model using Recall@K, HitRate@K, NDCG@K.
        
        Uses the validation set. For each positive validation example,
        checks if the target song appears in the top-K ranked candidates
        among all songs that the user interacted with in the validation set.
        
        CAVEAT: Metrics are computed on a small synthetic dataset.
        Results should not be taken as statistically meaningful until
        evaluated on a real, large-scale dataset.
        """
        self.model.eval()

        if not self.val_examples:
            return {"recall@k": 0.0, "hit_rate@k": 0.0, "ndcg@k": 0.0, "top_k": top_k}

        # Group val examples by user
        user_examples: Dict[str, List[TrainingExample]] = {}
        for ex in self.val_examples:
            user_examples.setdefault(ex.user_id, []).append(ex)

        hits = 0
        total_positive = 0
        ndcg_sum = 0.0
        users_with_hits = 0
        users_evaluated = 0

        for user_id, examples in user_examples.items():
            positive_songs = {ex.song_id for ex in examples if ex.label > 0.5}
            if not positive_songs:
                continue

            # Get all candidate songs for this user
            all_song_ids = list({ex.song_id for ex in examples})
            if len(all_song_ids) < 2:
                continue

            users_evaluated += 1

            # Build user features from the first example
            user_feats = examples[0].user_features

            # Get song features for all candidates
            song_feats = np.stack([
                self.song_features[self.song_id_to_row[sid]].astype(np.float32)
                for sid in all_song_ids
                if sid in self.song_id_to_row
            ])
            valid_song_ids = [
                sid for sid in all_song_ids if sid in self.song_id_to_row
            ]

            if len(valid_song_ids) < 2:
                continue

            # Score all candidates
            scores = self.model.score_pairs(
                user_feats.reshape(1, -1), song_feats
            ).squeeze(0)

            # Rank
            ranked_indices = np.argsort(-scores)[:top_k]
            ranked_songs = [valid_song_ids[i] for i in ranked_indices]

            # Compute metrics
            user_had_hit = False
            for pos_song in positive_songs:
                total_positive += 1
                if pos_song in ranked_songs:
                    hits += 1
                    user_had_hit = True
                    rank = ranked_songs.index(pos_song)
                    ndcg_sum += 1.0 / math.log2(rank + 2)

            if user_had_hit:
                users_with_hits += 1

        recall = hits / max(total_positive, 1)
        hit_rate = users_with_hits / max(users_evaluated, 1)
        ndcg = ndcg_sum / max(total_positive, 1)

        return {
            "recall@k": recall,
            "hit_rate@k": hit_rate,
            "ndcg@k": ndcg,
            "top_k": top_k,
            "total_positive_val": total_positive,
            "hits": hits,
        }

    # ------------------------------------------------------------------
    # Checkpoint Management
    # ------------------------------------------------------------------

    def save_checkpoint(self, directory: str) -> None:
        """Save both encoders, the two-tower model config, and training results."""
        os.makedirs(directory, exist_ok=True)

        # Save individual encoders
        song_dir = os.path.join(directory, "song_encoder")
        user_dir = os.path.join(directory, "user_encoder")
        self.song_encoder.save(song_dir)
        self.user_encoder.save(user_dir)

        # Save training config and results
        results = {
            "config": {
                "song_embedding_dim": self.config.song_embedding_dim,
                "user_embedding_dim": self.config.user_embedding_dim,
                "song_hidden_dim": self.config.song_hidden_dim,
                "user_hidden_dim": self.config.user_hidden_dim,
                "top_k_artists": self.config.top_k_artists,
                "top_k_genres": self.config.top_k_genres,
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
                "epochs": self.config.epochs,
                "temperature": self.config.temperature,
                "random_seed": self.config.random_seed,
                "dropout": self.config.dropout,
            },
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
        }

        with open(os.path.join(directory, "training_results.json"), "w") as f:
            json.dump(results, f, indent=2)

    @classmethod
    def load_model(cls, directory: str) -> TwoTowerModel:
        """Load a trained two-tower model from checkpoint."""
        song_dir = os.path.join(directory, "song_encoder")
        user_dir = os.path.join(directory, "user_encoder")

        song_encoder = SongEncoder.load(song_dir)
        user_encoder = UserEncoder.load(user_dir)

        # Load config for temperature
        config_path = os.path.join(directory, "training_results.json")
        temperature = 0.07
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                data = json.load(f)
                temperature = data.get("config", {}).get("temperature", 0.07)

        return TwoTowerModel(song_encoder, user_encoder, temperature)


# ---------------------------------------------------------------------------
# Inference Utilities
# ---------------------------------------------------------------------------

class TwoTowerInference:
    """Inference utilities for the two-tower recommender.
    
    Provides a simple flow:
        user_id → user representation → user embedding
        song_id → song features → song embedding
        similarity(user_emb, song_emb) → relevance score
    """

    def __init__(
        self,
        model: TwoTowerModel,
        song_features: np.ndarray,
        song_id_to_row: Dict[str, int],
        interaction_store: InteractionStore,
        metadata_lookup: SongMetadataLookup,
        config: TrainingConfig,
    ):
        self.model = model
        self.song_features = song_features
        self.song_id_to_row = song_id_to_row
        self.interaction_store = interaction_store
        self.metadata = metadata_lookup
        self.config = config

        self.history_builder = UserHistoryBuilder(
            song_embedding_dim=config.song_embedding_dim,
            top_k_artists=config.top_k_artists,
            top_k_genres=config.top_k_genres,
        )

        self.model.eval()

    def score_user_songs(
        self,
        user_id: str,
        candidate_song_ids: List[str],
    ) -> List[Tuple[str, float]]:
        """Score multiple candidate songs for a user.
        
        Returns list of (song_id, score) sorted by score descending.
        """
        # Build user representation
        user_feats = self._build_user_features(user_id)

        # Get song features
        valid_songs = []
        song_feats_list = []
        for sid in candidate_song_ids:
            row = self.song_id_to_row.get(sid)
            if row is not None:
                valid_songs.append(sid)
                song_feats_list.append(self.song_features[row].astype(np.float32))

        if not valid_songs:
            return []

        song_feats = np.stack(song_feats_list)

        # Score
        scores = self.model.score_pairs(
            user_feats.reshape(1, -1), song_feats
        ).squeeze(0)

        # Sort by score descending
        results = list(zip(valid_songs, scores.tolist()))
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def _build_user_features(self, user_id: str) -> np.ndarray:
        """Build user feature vector from all available interactions."""
        events = self.interaction_store.get_user_interactions(user_id)
        events.sort(key=lambda e: e.timestamp)

        if not events:
            return self.history_builder.build_cold_start_features()

        song_embs = []
        scores = []
        artist_acc: Dict[str, float] = {}
        genre_acc: Dict[str, float] = {}

        self.model.song_encoder.eval()

        for event in events:
            row = self.song_id_to_row.get(event.song_id)
            if row is None:
                continue

            feat = self.song_features[row].astype(np.float32)
            emb = self.model.song_encoder.encode_song(feat)
            song_embs.append(emb)

            score = calculate_event_score(event)
            scores.append(score)

            artist_info = self.metadata.get_artist(event.song_id)
            if artist_info:
                artist_acc[artist_info[0]] = artist_acc.get(artist_info[0], 0) + score

            for g in self.metadata.get_genres(event.song_id):
                genre_acc[g] = genre_acc.get(g, 0) + score

        if not song_embs:
            return self.history_builder.build_cold_start_features()

        artist_prefs = sorted(artist_acc.values(), reverse=True)
        artist_prefs = [normalize_preference(s, NORM_SCALE) for s in artist_prefs]

        genre_prefs = sorted(genre_acc.values(), reverse=True)
        genre_prefs = [normalize_preference(s, NORM_SCALE) for s in genre_prefs]

        return self.history_builder.build_user_features(
            np.stack(song_embs),
            np.array(scores, dtype=np.float32),
            artist_prefs,
            genre_prefs,
        )
