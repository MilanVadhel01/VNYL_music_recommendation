"""
Phase 5: Learned Song Embeddings Generation Pipeline
=====================================================

Loads preprocessed content features and generates learned embeddings
using the SongEncoder.

Because we do not have a full dataset of real interactions to train on yet,
this pipeline instantiates the encoder, optionally saves the checkpoint,
and encodes the entire 1M-song feature matrix in batches to prevent OOM.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import argparse
import time
import numpy as np
import pandas as pd
import torch

from src.models.song_encoder import SongEncoder, INPUT_DIM

def generate_embeddings(
    features_path: str,
    index_path: str,
    model_save_dir: str,
    embeddings_save_path: str,
    index_save_path: str,
    batch_size: int = 10000,
    embedding_dim: int = 64,
):
    print("=" * 60)
    print("PHASE 5: GENERATING LEARNED SONG EMBEDDINGS")
    print("=" * 60)

    # 1. Initialize and save Song Encoder
    print(f"1. Initializing SongEncoder (input_dim={INPUT_DIM}, emb_dim={embedding_dim})")
    torch.manual_seed(42)
    encoder = SongEncoder(input_dim=INPUT_DIM, embedding_dim=embedding_dim)
    
    # Save the encoder for traceability
    print(f"   Saving encoder to {model_save_dir}...")
    encoder.save(model_save_dir)
    encoder.eval()
    
    # 2. Load feature matrix and index
    print(f"2. Loading features from {features_path}...")
    # Map memory so we don't necessarily load everything if we don't want to,
    # though for 1M x 13 (float64) it's ~100MB, easily fits in RAM.
    features = np.load(features_path)
    num_songs = features.shape[0]
    print(f"   Loaded {num_songs} feature vectors of dimension {features.shape[1]}")

    print(f"3. Loading index from {index_path}...")
    index_df = pd.read_parquet(index_path)
    assert len(index_df) == num_songs, "Feature rows and index rows mismatch!"

    # 3. Batch processing
    print(f"4. Generating embeddings in batches of {batch_size}...")
    embeddings_list = []
    
    start_time = time.time()
    
    with torch.no_grad():
        for start_idx in range(0, num_songs, batch_size):
            end_idx = min(start_idx + batch_size, num_songs)
            batch = features[start_idx:end_idx].astype(np.float32)
            
            # Encode
            emb_batch = encoder.encode_songs(batch)
            embeddings_list.append(emb_batch)
            
            if (start_idx // batch_size) % 10 == 0:
                print(f"   Processed {end_idx}/{num_songs} songs...")

    embeddings = np.concatenate(embeddings_list, axis=0)
    elapsed = time.time() - start_time
    
    print(f"   Completed in {elapsed:.2f}s ({num_songs / elapsed:.0f} songs/sec)")
    
    # 4. Validation
    assert embeddings.shape == (num_songs, embedding_dim), "Embedding shape mismatch!"
    assert not np.any(np.isnan(embeddings)), "NaNs detected in embeddings!"
    assert not np.any(np.isinf(embeddings)), "Infs detected in embeddings!"
    
    # Norm check (should be L2 normalized)
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4), "Embeddings are not L2 normalized!"

    # 5. Save Artifacts
    print(f"5. Saving artifacts...")
    os.makedirs(os.path.dirname(embeddings_save_path), exist_ok=True)
    
    # Save embeddings
    np.save(embeddings_save_path, embeddings)
    print(f"   Saved embeddings to {embeddings_save_path} (shape: {embeddings.shape})")
    
    # Save corresponding index
    index_df.to_parquet(index_save_path)
    print(f"   Saved index to {index_save_path}")

    print("=" * 60)
    print("SUCCESS: Phase 5 completed.")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate learned song embeddings.")
    parser.add_argument("--features", type=str, default="data/processed/content_features.npy")
    parser.add_argument("--index", type=str, default="data/processed/song_index.parquet")
    parser.add_argument("--model_dir", type=str, default="models/song_encoder_v1")
    parser.add_argument("--out_embeddings", type=str, default="data/processed/song_embeddings_v1.npy")
    parser.add_argument("--out_index", type=str, default="data/processed/song_embedding_index_v1.parquet")
    parser.add_argument("--batch_size", type=int, default=10000)
    parser.add_argument("--dim", type=int, default=64)

    args = parser.parse_args()
    
    generate_embeddings(
        features_path=args.features,
        index_path=args.index,
        model_save_dir=args.model_dir,
        embeddings_save_path=args.out_embeddings,
        index_save_path=args.out_index,
        batch_size=args.batch_size,
        embedding_dim=args.dim,
    )
