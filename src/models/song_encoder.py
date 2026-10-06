"""
Neural Song Encoder (Phase 2)
==============================

A neural network that converts the existing 13 song content features
into a learned song embedding.

Architecture:
    13 input features (StandardScaler-normalized)
        ↓
    Dense(13 → hidden_dim)
        ↓
    ReLU + BatchNorm
        ↓
    Dense(hidden_dim → hidden_dim)
        ↓
    ReLU + BatchNorm
        ↓
    Dense(hidden_dim → embedding_dim)
        ↓
    L2 normalization (optional)
        ↓
    Song embedding (64 or 128 dimensions)

Note: At this phase the encoder produces learned song representations.
These are NOT yet trained recommendation embeddings — that happens in Phase 4.

Existing pipeline is preserved:
    raw features → StandardScaler → L2 norm → existing content vector
    
The neural encoder operates on StandardScaler features (pre-L2):
    raw features → StandardScaler → Song Encoder → learned embedding
"""

import os
import json
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import numpy as np


# ---------------------------------------------------------------------------
# Feature ordering (matches prepare_features.py CANDIDATE_FEATURES)
# ---------------------------------------------------------------------------

CONTENT_FEATURE_ORDER = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "duration",
    "key",
    "mode",
    "time_signature",
]

INPUT_DIM = len(CONTENT_FEATURE_ORDER)  # 13


# ---------------------------------------------------------------------------
# Song Encoder Network
# ---------------------------------------------------------------------------

class SongEncoder(nn.Module):
    """Neural network that maps 13 content features to a learned embedding.
    
    Parameters
    ----------
    input_dim : int
        Number of input features (default: 13).
    hidden_dim : int
        Width of hidden layers (default: 128).
    embedding_dim : int
        Dimensionality of the output embedding (default: 64).
    normalize : bool
        If True, L2-normalize the output embedding (default: True).
    dropout : float
        Dropout probability (default: 0.1).
    """

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
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

    def encode_song(self, features: np.ndarray) -> np.ndarray:
        """Encode a single song.
        
        Parameters
        ----------
        features : ndarray of shape (input_dim,) or (1, input_dim)
        
        Returns
        -------
        ndarray of shape (embedding_dim,)
        """
        was_training = self.training
        self.eval()

        if features.ndim == 1:
            features = features.reshape(1, -1)

        with torch.no_grad():
            x = torch.from_numpy(features.astype(np.float32))
            emb = self.forward(x)

        if was_training:
            self.train()

        return emb.numpy().squeeze(0)

    def encode_songs(self, feature_matrix: np.ndarray) -> np.ndarray:
        """Encode a batch of songs.
        
        Parameters
        ----------
        feature_matrix : ndarray of shape (n_songs, input_dim)
        
        Returns
        -------
        ndarray of shape (n_songs, embedding_dim)
        """
        was_training = self.training
        self.eval()

        with torch.no_grad():
            x = torch.from_numpy(feature_matrix.astype(np.float32))
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
            "feature_order": CONTENT_FEATURE_ORDER,
        }

    def save(self, directory: str) -> None:
        """Save model checkpoint and configuration.
        
        Saves:
            {directory}/song_encoder.pt   — model weights
            {directory}/song_encoder_config.json — architecture config
        """
        os.makedirs(directory, exist_ok=True)

        weights_path = os.path.join(directory, "song_encoder.pt")
        config_path = os.path.join(directory, "song_encoder_config.json")

        torch.save(self.state_dict(), weights_path)
        with open(config_path, "w") as f:
            json.dump(self.get_config(), f, indent=2)

    @classmethod
    def load(cls, directory: str) -> "SongEncoder":
        """Load model from checkpoint and configuration.
        
        Parameters
        ----------
        directory : str
            Directory containing song_encoder.pt and song_encoder_config.json
        
        Returns
        -------
        SongEncoder with loaded weights
        """
        config_path = os.path.join(directory, "song_encoder_config.json")
        weights_path = os.path.join(directory, "song_encoder.pt")

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
