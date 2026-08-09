"""
FAISS Content-Based Recommender
================================

Fast similarity search using a pre-built FAISS IndexFlatIP index.

The index stores L2-normalized content feature vectors so that inner
product equals cosine similarity.

Architecture:
    track_id -> row_index -> L2-normalize feature vector
        -> FAISS search -> top-K row indices -> song metadata

Usage:
    from src.models.faiss_recommender import FAISSContentRecommender

    rec = FAISSContentRecommender()
    results = rec.recommend_by_track_id("0pqnGHJpmpxLKifKRmU6WP", top_n=10)
"""

import os
from typing import Optional

import faiss
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Default artifact paths
# ---------------------------------------------------------------------------
DEFAULT_INDEX_PATH: str = os.path.join(
    "data", "processed", "faiss", "content.index"
)
DEFAULT_FEATURES_PATH: str = os.path.join(
    "data", "processed", "content_features.npy"
)
DEFAULT_SONG_INDEX_PATH: str = os.path.join(
    "data", "processed", "song_index.parquet"
)

# Buffer: search for extra candidates so we can exclude the query song
_SEARCH_BUFFER: int = 10


class FAISSContentRecommender:
    """Content-based recommender backed by a FAISS inner-product index.

    Parameters
    ----------
    index_path : str
        Path to the saved FAISS index.
    features_path : str
        Path to the original (StandardScaler) feature matrix.
        Used to retrieve the raw vector for a given row index.
    song_index_path : str
        Path to the song-index mapping parquet.
    """

    def __init__(
        self,
        index_path: str = DEFAULT_INDEX_PATH,
        features_path: str = DEFAULT_FEATURES_PATH,
        song_index_path: str = DEFAULT_SONG_INDEX_PATH,
    ) -> None:
        for path, label in [
            (index_path, "FAISS index"),
            (features_path, "Feature matrix"),
            (song_index_path, "Song index"),
        ]:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"{label} not found at: {path}\n"
                    f"Run the appropriate build script first."
                )

        self.index: faiss.Index = faiss.read_index(index_path)
        self.feature_matrix: np.ndarray = np.load(features_path)
        self.song_index: pd.DataFrame = pd.read_parquet(song_index_path)

        # Fast lookup
        self._id_to_row: dict[str, int] = dict(
            zip(self.song_index["track_id"], self.song_index["row_index"])
        )

        # Validate consistency
        if self.index.ntotal != len(self.song_index):
            raise ValueError(
                f"FAISS index vectors ({self.index.ntotal:,d}) do not match "
                f"song index rows ({len(self.song_index):,d})."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def recommend_by_track_id(
        self,
        track_id: str,
        top_n: int = 10,
    ) -> list[dict]:
        """Return the top-N most similar songs for a given track_id."""
        if top_n < 1:
            raise ValueError(f"top_n must be >= 1, got {top_n}")

        row_idx = self._id_to_row.get(track_id)
        if row_idx is None:
            raise ValueError(f"Track ID not found in catalog: {track_id}")

        return self._recommend_by_row(row_idx, top_n)

    def recommend_by_song_name(
        self,
        song_name: str,
        top_n: int = 10,
        artist_name: Optional[str] = None,
    ) -> list[dict]:
        """Return the top-N most similar songs looked up by name."""
        if top_n < 1:
            raise ValueError(f"top_n must be >= 1, got {top_n}")

        matches = self.song_index[
            self.song_index["track_name"].str.lower() == song_name.lower()
        ]
        if artist_name is not None:
            matches = matches[
                matches["artist_name"].str.lower() == artist_name.lower()
            ]
        if matches.empty:
            extra = f" by '{artist_name}'" if artist_name else ""
            raise ValueError(f"Song not found: '{song_name}'{extra}")

        row_idx = int(matches.iloc[0]["row_index"])
        return self._recommend_by_row(row_idx, top_n)

    def search_songs(self, query: str, max_results: int = 10) -> pd.DataFrame:
        """Search the catalogue by partial song name."""
        mask = self.song_index["track_name"].str.contains(
            query, case=False, na=False
        )
        return self.song_index[mask].head(max_results)

    def get_song_info(self, track_id: str) -> dict:
        """Return metadata for a single track."""
        row_idx = self._id_to_row.get(track_id)
        if row_idx is None:
            raise ValueError(f"Track ID not found in catalog: {track_id}")
        row = self.song_index[self.song_index["row_index"] == row_idx].iloc[0]
        return row.to_dict()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _recommend_by_row(self, row_idx: int, top_n: int) -> list[dict]:
        """L2-normalize the query vector, search FAISS, return results."""

        # Get the raw StandardScaler vector and L2-normalize
        raw_vector = self.feature_matrix[row_idx].astype(np.float32)
        norm = np.linalg.norm(raw_vector)
        if norm > 0:
            raw_vector = raw_vector / norm
        query = raw_vector.reshape(1, -1)

        # Search with buffer to allow excluding the query song
        k = min(top_n + _SEARCH_BUFFER, self.index.ntotal)
        scores, indices = self.index.search(query, k)

        scores = scores[0]
        indices = indices[0]

        # Build results, excluding the query row
        results: list[dict] = []
        seen_ids: set[str] = set()

        for idx, score in zip(indices, scores):
            if idx == row_idx:
                continue  # Exclude query song
            if len(results) >= top_n:
                break

            row = self.song_index.iloc[idx]
            tid = row["track_id"]

            if tid in seen_ids:
                continue
            seen_ids.add(tid)

            results.append(
                {
                    "track_id": tid,
                    "track_name": row["track_name"],
                    "artist_name": row["artist_name"],
                    "similarity_score": float(score),
                }
            )

        return results
