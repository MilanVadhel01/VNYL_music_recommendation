"""
Build FAISS Index for Learned Embeddings (Phase 6)
==================================================

Builds a FAISS IndexFlatIP from the generated learned song embeddings.
IndexFlatIP is exact and fast enough for 1M embeddings (~15ms latency).
"""

import os
import argparse
import time
import faiss
import numpy as np

def build_index(embeddings_path: str, index_save_path: str):
    print("=" * 60)
    print("PHASE 6: BUILDING LEARNED FAISS INDEX")
    print("=" * 60)
    
    print(f"Loading embeddings from {embeddings_path}...")
    embeddings = np.load(embeddings_path).astype(np.float32)
    num_songs, dim = embeddings.shape
    print(f"Loaded {num_songs} embeddings of dimension {dim}")
    
    print("Building FAISS IndexFlatIP (Exact Inner Product Search)...")
    start_time = time.time()
    
    # We use IndexFlatIP because the embeddings are L2 normalized from the encoder
    # Inner product on L2-normalized vectors is exact cosine similarity.
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    
    elapsed = time.time() - start_time
    print(f"Index built in {elapsed:.2f} seconds.")
    
    os.makedirs(os.path.dirname(index_save_path), exist_ok=True)
    faiss.write_index(index, index_save_path)
    
    size_mb = os.path.getsize(index_save_path) / (1024 * 1024)
    print(f"Saved index to {index_save_path} ({size_mb:.2f} MB)")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", type=str, default="data/processed/song_embeddings_v1.npy")
    parser.add_argument("--out_index", type=str, default="data/processed/faiss/learned_v1.index")
    args = parser.parse_args()
    
    build_index(args.embeddings, args.out_index)
