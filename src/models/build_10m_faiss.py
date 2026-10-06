"""
Phase 8: Scale to 10M Songs (FAISS Indexing)
============================================

Builds a FAISS index from the 10M embeddings.

Since 10M vectors is large but fits in RAM (~2.5GB), we could use FlatIP,
but query time would be ~150ms per query. To scale efficiently, we use
an IVF index (e.g. IVF16384,Flat or IVF16384_HNSW32) trained on a sample of
the dataset (e.g. 500k vectors).

We use IVF-Flat here as the baseline for 10M as it provides
good speed/recall trade-off and avoids the 5x RAM overhead of HNSW.
"""

import os
import argparse
import time
import numpy as np
import faiss
import pandas as pd

def build_10m_faiss(
    embeddings_path: str,
    index_meta_path: str,
    out_index_path: str,
    nlist: int = 16384,
    train_sample_size: int = 500_000
):
    print("=" * 60)
    print("PHASE 8: BUILDING 10M FAISS INDEX")
    print("=" * 60)
    
    # 1. Load Metadata to get count
    meta_df = pd.read_parquet(index_meta_path)
    num_songs = len(meta_df)
    print(f"Total songs to index: {num_songs:,}")
    
    if num_songs == 0:
        print("No songs to index.")
        return
        
    dim = 64
    
    # 2. Open Embeddings Memmap
    print(f"Loading embeddings from {embeddings_path}...")
    embeddings_mmap = np.memmap(embeddings_path, dtype=np.float32, mode='r', shape=(num_songs, dim))
    
    # 3. Train IVF Index
    quantizer = faiss.IndexFlatIP(dim)
    # nlist = roughly sqrt(num_songs) to 4*sqrt(num_songs)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    
    # Subsample for training
    actual_sample_size = min(train_sample_size, num_songs)
    print(f"Training IVF Index (nlist={nlist}) on {actual_sample_size:,} samples...")
    start_train = time.time()
    
    # Random sample for training
    rng = np.random.default_rng(seed=42)
    train_indices = rng.choice(num_songs, actual_sample_size, replace=False)
    # Note: Fancy indexing on memmap can be slow, but for 500k it's acceptable.
    # We sort to optimize disk reads.
    train_indices.sort()
    train_data = embeddings_mmap[train_indices]
    
    index.train(train_data)
    print(f"Training completed in {time.time() - start_train:.2f}s")
    
    # 4. Add vectors in chunks to respect RAM (even though 2.5GB fits, chunking is safer)
    chunk_size = 1_000_000
    print(f"Adding vectors to index in chunks of {chunk_size:,}...")
    start_add = time.time()
    
    for start_idx in range(0, num_songs, chunk_size):
        end_idx = min(start_idx + chunk_size, num_songs)
        
        # Read from memmap (fast sequential read)
        chunk_data = embeddings_mmap[start_idx:end_idx]
        
        # Add to index
        index.add(chunk_data)
        print(f"  Added {end_idx:,} / {num_songs:,} ...")
        
    print(f"Adding completed in {time.time() - start_add:.2f}s")
    
    # 5. Save index
    os.makedirs(os.path.dirname(out_index_path), exist_ok=True)
    faiss.write_index(index, out_index_path)
    
    size_mb = os.path.getsize(out_index_path) / (1024 * 1024)
    print(f"Index saved to {out_index_path} ({size_mb:.2f} MB)")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", type=str, default="data/processed/10m/song_embeddings_10m.npy")
    parser.add_argument("--index_meta", type=str, default="data/processed/10m/song_index_10m.parquet")
    parser.add_argument("--out_index", type=str, default="data/processed/faiss/learned_10m.index")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.embeddings):
        print(f"Skip building: {args.embeddings} not found.")
    else:
        build_10m_faiss(
            args.embeddings,
            args.index_meta,
            args.out_index
        )
