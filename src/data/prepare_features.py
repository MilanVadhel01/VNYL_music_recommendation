"""
Feature Preparation Pipeline
=============================

Steps 9–12 of the content-based recommendation system.

This script:
    1. Loads ``songs_1m.parquet``
    2. Validates that the expected content features exist
    3. Reports missing values and suspicious / invalid values
    4. Fills missing values with column medians (defensive -- currently a no-op)
    5. Normalises features with ``StandardScaler``
    6. Saves three artifacts:
       - ``data/processed/content_features.npy``   -- normalised feature matrix
       - ``data/processed/song_index.parquet``      -- row-index → song metadata
       - ``data/processed/content_scaler.joblib``   -- fitted scaler

Usage:
    python src/data/prepare_features.py
"""

import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INPUT_PATH: str = os.path.join("data", "processed", "songs_1m.parquet")

OUTPUT_FEATURES: str = os.path.join("data", "processed", "content_features.npy")
OUTPUT_INDEX: str = os.path.join("data", "processed", "song_index.parquet")
OUTPUT_SCALER: str = os.path.join("data", "processed", "content_scaler.joblib")

# Candidate content features -- only those actually present in the dataset
# will be used.
CANDIDATE_FEATURES: list[str] = [
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

# Columns expected in the song-index mapping
INDEX_COLUMNS: list[str] = [
    "track_id",
    "track_name",
    "artist_name",
]

# Features expected to lie in [0, 1]
BOUNDED_FEATURES: list[str] = [
    "danceability",
    "energy",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
]


# ---------------------------------------------------------------------------
# Step 9 -- Dataset quality check
# ---------------------------------------------------------------------------
def _validate_features(df: pd.DataFrame) -> list[str]:
    """Return the subset of CANDIDATE_FEATURES that exist in *df*.

    Prints which features are available and which are missing.
    """
    available = [f for f in CANDIDATE_FEATURES if f in df.columns]
    missing = [f for f in CANDIDATE_FEATURES if f not in df.columns]

    print("Available recommendation features:")
    for f in available:
        print(f"  [OK] {f}")
    print()

    if missing:
        print("Missing expected features:")
        for f in missing:
            print(f"  [MISSING] {f}")
        print()
    else:
        print("All 13 candidate features are present.\n")

    if not available:
        print("ERROR: No content features found in the dataset.")
        sys.exit(1)

    return available


def _report_missing_values(df: pd.DataFrame, features: list[str]) -> None:
    """Print per-feature missing-value counts."""
    missing = df[features].isnull().sum()
    total_missing = missing.sum()

    print(f"Missing values in content features: {total_missing:,d}")
    if total_missing > 0:
        for feat, count in missing.items():
            if count > 0:
                print(f"  {feat}: {count:,d}")
    print()


def _report_invalid_values(df: pd.DataFrame, features: list[str]) -> None:
    """Report suspicious / invalid numerical values."""
    print("Invalid / suspicious values:")

    issues_found = False

    if "duration" in features:
        n = (df["duration"] <= 0).sum()
        if n > 0:
            print(f"  duration <= 0:        {n:,d}")
            issues_found = True

    if "tempo" in features:
        n = (df["tempo"] <= 0).sum()
        if n > 0:
            print(f"  tempo <= 0:           {n:,d}  (Spotify API: 0 = unknown)")
            issues_found = True

    if "time_signature" in features:
        n = (df["time_signature"] == 0).sum()
        if n > 0:
            print(f"  time_signature == 0:  {n:,d}  (Spotify API: 0 = unknown)")
            issues_found = True

    if "duration" in features:
        n = (df["duration"] > 3600).sum()
        if n > 0:
            print(f"  duration > 3600s:     {n:,d}  (very long tracks)")
            issues_found = True

    for feat in BOUNDED_FEATURES:
        if feat in features:
            n = ((df[feat] < 0) | (df[feat] > 1)).sum()
            if n > 0:
                print(f"  {feat} outside [0,1]: {n:,d}")
                issues_found = True

    if not issues_found:
        print("  None.")

    print()


# ---------------------------------------------------------------------------
# Step 10 -- Handle missing values
# ---------------------------------------------------------------------------
def _handle_missing_values(
    df: pd.DataFrame, features: list[str]
) -> pd.DataFrame:
    """Fill missing values in content features with the column median.

    Currently a no-op because the dataset has zero missing values in content
    features, but coded defensively for robustness.
    """
    missing_before = df[features].isnull().sum().sum()

    if missing_before > 0:
        print(f"Filling {missing_before:,d} missing values with column medians …")
        df[features] = df[features].fillna(df[features].median())
    else:
        print("No missing values to fill in content features.")

    # Assert post-condition
    missing_after = df[features].isnull().sum().sum()
    assert missing_after == 0, (
        f"ASSERTION FAILED: {missing_after:,d} missing values remain after "
        f"imputation. Cannot proceed."
    )
    print("[OK] Zero missing values after imputation.\n")

    return df


# ---------------------------------------------------------------------------
# Step 12 -- Normalise features
# ---------------------------------------------------------------------------
def _normalise_features(
    df: pd.DataFrame, features: list[str]
) -> tuple[np.ndarray, StandardScaler]:
    """Fit a StandardScaler and return the normalised feature matrix."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features].values)

    print(f"Feature matrix shape: {X_scaled.shape}")
    print(f"Feature matrix dtype: {X_scaled.dtype}")
    print(f"Feature matrix size:  {X_scaled.nbytes / (1024 * 1024):.1f} MB")
    print()

    return X_scaled, scaler


# ---------------------------------------------------------------------------
# Save artifacts
# ---------------------------------------------------------------------------
def _save_artifacts(
    X_scaled: np.ndarray,
    scaler: StandardScaler,
    df: pd.DataFrame,
    features: list[str],
) -> None:
    """Save the normalised feature matrix, song index, and scaler."""

    # --- Feature matrix ---
    np.save(OUTPUT_FEATURES, X_scaled)
    size_mb = os.path.getsize(OUTPUT_FEATURES) / (1024 * 1024)
    print(f"[OK] Saved feature matrix:  {OUTPUT_FEATURES}  ({size_mb:.1f} MB)")

    # --- Song index ---
    available_index_cols = [c for c in INDEX_COLUMNS if c in df.columns]
    song_index = df[available_index_cols].copy()
    song_index.insert(0, "row_index", range(len(song_index)))
    song_index = song_index.reset_index(drop=True)
    song_index.to_parquet(OUTPUT_INDEX, index=False)
    size_mb = os.path.getsize(OUTPUT_INDEX) / (1024 * 1024)
    print(f"[OK] Saved song index:      {OUTPUT_INDEX}  ({size_mb:.1f} MB)")

    # --- Scaler ---
    joblib.dump(scaler, OUTPUT_SCALER)
    size_kb = os.path.getsize(OUTPUT_SCALER) / 1024
    print(f"[OK] Saved scaler:          {OUTPUT_SCALER}  ({size_kb:.1f} KB)")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Run the full feature preparation pipeline."""

    print("=" * 60)
    print("VNYL Music Recommendation -- Feature Preparation")
    print("=" * 60)
    print()

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    if not os.path.exists(INPUT_PATH):
        print(f"ERROR: Input file not found: {INPUT_PATH}")
        print("Run  python src/data/sample.py  first.")
        sys.exit(1)

    start = time.time()
    print(f"Loading dataset: {INPUT_PATH}")
    df = pd.read_parquet(INPUT_PATH)
    print(f"  Rows:    {len(df):,d}")
    print(f"  Columns: {len(df.columns)}")
    print()

    if len(df) == 0:
        print("ERROR: Dataset is empty.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 9 -- Validate features
    # ------------------------------------------------------------------
    available_features = _validate_features(df)

    # ------------------------------------------------------------------
    # Step 9 -- Report data quality
    # ------------------------------------------------------------------
    _report_missing_values(df, available_features)
    _report_invalid_values(df, available_features)

    # ------------------------------------------------------------------
    # Step 10 -- Handle missing values
    # ------------------------------------------------------------------
    df = _handle_missing_values(df, available_features)

    # ------------------------------------------------------------------
    # Step 11 -- Feature selection (already done via available_features)
    # ------------------------------------------------------------------
    print(f"Selected {len(available_features)} features for content model:")
    for f in available_features:
        print(f"  - {f}")
    print()

    # ------------------------------------------------------------------
    # Step 12 -- Normalise
    # ------------------------------------------------------------------
    X_scaled, scaler = _normalise_features(df, available_features)

    # ------------------------------------------------------------------
    # Save artifacts
    # ------------------------------------------------------------------
    _save_artifacts(X_scaled, scaler, df, available_features)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    elapsed = time.time() - start
    print("=" * 60)
    print("Feature preparation complete.")
    print(f"  Features used:  {len(available_features)}")
    print(f"  Matrix shape:   {X_scaled.shape}")
    print(f"  Time elapsed:   {elapsed:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
