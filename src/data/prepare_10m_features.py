"""
Phase 8: 10M-Song Streaming Feature Pipeline
==========================================

Streams up to 10M songs from the HF dataset, avoiding RAM limits.
Key features:
1. Streaming download (chunked).
2. Streaming normalization via StandardScaler.partial_fit().
3. De-duplication of (track_name, artist_name) pairs.
4. Output saved in chunked Parquet/Numpy format.

Duplicate Investigation & Documentation:
----------------------------------------
In the Spotify 45M dataset, there are often multiple releases of the same song 
(e.g., single release, album release, explicit/clean versions, remasters).
These share the exact same track_name and artist_name, but have different track_ids.
If left unchecked, FAISS nearest neighbors will be dominated by identical duplicate
audio tracks, hurting discovery. We track seen (lower_track_name, lower_artist_name)
and discard duplicates on-the-fly.
"""

import os
import time
import numpy as np
import pandas as pd
from typing import Any, Iterator
from sklearn.preprocessing import StandardScaler
import joblib

DATASET_NAME = "GD-Studio/embeat_45m_spotify_tracks"
SPLIT = "train"
TARGET_ROWS = 10_000_000
CHUNK_SIZE = 100_000

CANDIDATE_FEATURES = [
    "danceability", "energy", "loudness", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo", "duration",
    "key", "mode", "time_signature",
]

INDEX_COLUMNS = ["track_id", "track_name", "artist_name"]

def process_chunk(chunk_df: pd.DataFrame, scaler: StandardScaler, is_first: bool, features_file: str):
    """Processes a chunk and saves data."""
    # Handle missing
    chunk_df[CANDIDATE_FEATURES] = chunk_df[CANDIDATE_FEATURES].fillna(0.0)
    
    # Partial fit and transform
    if is_first:
        scaler.partial_fit(chunk_df[CANDIDATE_FEATURES])
    else:
        scaler.partial_fit(chunk_df[CANDIDATE_FEATURES])
        
    X_scaled = scaler.transform(chunk_df[CANDIDATE_FEATURES]).astype(np.float32)
    
    # Append to npy-like or just parquet. For scale, we save features to a growing file.
    with open(features_file, "ab") as f:
        f.write(X_scaled.tobytes())
        
    return chunk_df[INDEX_COLUMNS]

def build_10m_dataset(target_rows: int = TARGET_ROWS):
    from datasets import load_dataset
    
    os.makedirs("data/processed/10m", exist_ok=True)
    features_file = "data/processed/10m/content_features_10m.raw"
    scaler_file = "data/processed/10m/content_scaler_10m.joblib"
    index_file = "data/processed/10m/song_index_10m.parquet"
    
    if os.path.exists(features_file):
        os.remove(features_file)

    print(f"Connecting to dataset: {DATASET_NAME} …")
    dataset = load_dataset(DATASET_NAME, split=SPLIT, streaming=True)
    
    scaler = StandardScaler()
    seen_tracks = set()
    collected = 0
    duplicates_skipped = 0
    
    chunk = []
    metadata_chunks = []
    
    start_time = time.time()
    
    for record in dataset:
        if collected >= target_rows:
            break
            
        t_name = str(record.get("track_name", "")).strip().lower()
        a_name = str(record.get("artist_name", "")).strip().lower()
        
        # Deduplication check
        dup_key = (t_name, a_name)
        if not t_name or not a_name or dup_key in seen_tracks:
            duplicates_skipped += 1
            continue
            
        seen_tracks.add(dup_key)
        chunk.append(record)
        collected += 1
        
        if len(chunk) >= CHUNK_SIZE:
            df = pd.DataFrame(chunk)
            meta = process_chunk(df, scaler, collected <= CHUNK_SIZE, features_file)
            metadata_chunks.append(meta)
            chunk = []
            print(f"Collected {collected:,} unique songs... (skipped {duplicates_skipped:,} duplicates)")
            
    if chunk:
        df = pd.DataFrame(chunk)
        meta = process_chunk(df, scaler, collected <= CHUNK_SIZE, features_file)
        metadata_chunks.append(meta)
        
    # Save Metadata and Scaler
    print("Saving 10M Index and Scaler...")
    full_meta = pd.concat(metadata_chunks, ignore_index=True)
    full_meta.insert(0, "row_index", range(len(full_meta)))
    full_meta.to_parquet(index_file, index=False)
    
    joblib.dump(scaler, scaler_file)
    
    print(f"Done! Pipeline took {time.time() - start_time:.2f}s")
    print(f"Total Unique: {collected:,} | Duplicates Skipped: {duplicates_skipped:,}")
    
if __name__ == "__main__":
    # Example test usage: python src/data/prepare_10m_features.py --test
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        build_10m_dataset(target_rows=5_000)
    else:
        build_10m_dataset()
