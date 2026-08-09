"""
Content-Based Music Recommender
================================

Baseline recommendation engine that finds similar songs using cosine
similarity over normalised audio / content features.

Architecture:
    songs_1m.parquet -> StandardScaler -> feature matrix -> cosine similarity -> Top-N

This module does NOT use FAISS or any approximate nearest-neighbour index.
Brute-force similarity is acceptable for the baseline but will not scale
efficiently to production-level request volumes over 1M songs.

Usage:
    from src.models.content_model import ContentRecommender

    rec = ContentRecommender()
    results = rec.recommend_by_track_id("0pqnGHJpmpxLKifKRmU6WP", top_n=10)
"""

import os
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Default artifact paths
# ---------------------------------------------------------------------------
DEFAULT_FEATURES_PATH: str = os.path.join("data", "processed", "content_features.npy")
DEFAULT_INDEX_PATH: str = os.path.join("data", "processed", "song_index.parquet")
DEFAULT_SCALER_PATH: str = os.path.join("data", "processed", "content_scaler.joblib")


class ContentRecommender:
    """Baseline content-based recommender using cosine similarity.

    Parameters
    ----------
    features_path : str
        Path to the normalised feature matrix (``.npy``).
    index_path : str
        Path to the song-index mapping (``.parquet``).
    scaler_path : str
        Path to the fitted ``StandardScaler`` (``.joblib``).
    """

    def __init__(
        self,
        features_path: str = DEFAULT_FEATURES_PATH,
        index_path: str = DEFAULT_INDEX_PATH,
        scaler_path: str = DEFAULT_SCALER_PATH,
    ) -> None:
        # Validate that all required files exist
        for path, label in [
            (features_path, "Feature matrix"),
            (index_path, "Song index"),
            (scaler_path, "Scaler"),
        ]:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"{label} not found at: {path}\n"
                    f"Run  python src/data/prepare_features.py  first."
                )

        self.feature_matrix: np.ndarray = np.load(features_path)
        self.song_index: pd.DataFrame = pd.read_parquet(index_path)
        self.scaler = joblib.load(scaler_path)

        # Build a fast track_id → row_index lookup
        self._id_to_row: dict[str, int] = dict(
            zip(self.song_index["track_id"], self.song_index["row_index"])
        )

        # Validate consistency
        n_features, n_index = len(self.feature_matrix), len(self.song_index)
        if n_features != n_index:
            raise ValueError(
                f"Feature matrix rows ({n_features:,d}) do not match "
                f"song index rows ({n_index:,d})."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def recommend_by_track_id(
        self,
        track_id: str,
        top_n: int = 10,
    ) -> list[dict]:
        """Return the top-N most similar songs for a given ``track_id``.

        Parameters
        ----------
        track_id : str
            The Spotify track ID to query.
        top_n : int
            Number of recommendations to return.  Must be >= 1.

        Returns
        -------
        list[dict]
            Each dict contains ``track_id``, ``track_name``,
            ``artist_name``, and ``similarity_score``.

        Raises
        ------
        ValueError
            If *track_id* is not in the catalogue or *top_n* < 1.
        """
        if top_n < 1:
            raise ValueError(f"top_n must be >= 1, got {top_n}")

        row_idx = self._id_to_row.get(track_id)
        if row_idx is None:
            raise ValueError(
                f"Track ID not found in catalog: {track_id}"
            )

        return self._recommend_by_row(row_idx, top_n)

    def recommend_by_song_name(
        self,
        song_name: str,
        top_n: int = 10,
        artist_name: Optional[str] = None,
    ) -> list[dict]:
        """Return the top-N most similar songs for a song looked up by name.

        If multiple tracks match *song_name*, the first exact
        case-insensitive match is used.  When *artist_name* is provided
        the results are further filtered.

        Parameters
        ----------
        song_name : str
            The song title to search for (case-insensitive).
        top_n : int
            Number of recommendations to return.
        artist_name : str, optional
            Filter matches to a specific artist.

        Returns
        -------
        list[dict]
            Recommendation results (same format as
            :meth:`recommend_by_track_id`).

        Raises
        ------
        ValueError
            If no matching track is found.
        """
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
            extra = (
                f" by '{artist_name}'" if artist_name else ""
            )
            raise ValueError(
                f"Song not found: '{song_name}'{extra}"
            )

        # Use the first match
        row_idx = int(matches.iloc[0]["row_index"])
        return self._recommend_by_row(row_idx, top_n)

    def search_songs(
        self,
        query: str,
        max_results: int = 10,
    ) -> pd.DataFrame:
        """Search the catalogue by partial song name (case-insensitive).

        Returns a DataFrame of matching songs (useful for interactive
        exploration and finding valid track IDs).
        """
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
        """Compute cosine similarity for a single row and return top-N."""

        query_vector = self.feature_matrix[row_idx].reshape(1, -1)

        # Compute similarity against all songs
        similarities = cosine_similarity(query_vector, self.feature_matrix)[0]

        # Exclude the query song itself by setting its similarity to -inf
        similarities[row_idx] = -np.inf

        # Get top-N indices (sorted descending)
        top_indices = np.argsort(similarities)[::-1][:top_n]

        # Build results
        results: list[dict] = []
        seen_ids: set[str] = set()

        for idx in top_indices:
            row = self.song_index.iloc[idx]
            tid = row["track_id"]

            # Guard against duplicates (should not happen, but defensive)
            if tid in seen_ids:
                continue
            seen_ids.add(tid)

            results.append(
                {
                    "track_id": tid,
                    "track_name": row["track_name"],
                    "artist_name": row["artist_name"],
                    "similarity_score": float(similarities[idx]),
                }
            )

        return results
