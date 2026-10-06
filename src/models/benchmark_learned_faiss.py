"""
Phase 6: Benchmarking FAISS indices for Learned Embeddings
==========================================================

This script benchmarks different FAISS index types on the 1M learned song embeddings.
It compares IndexFlatIP (exact search), IVF, and HNSW strategies.

Metrics recorded:
- Build time
- Index size (on disk)
- Query latency
- Recall@K against the exact search baseline
"""

import os
import time
import psutil
import faiss
import numpy as np
import tempfile

def measure_index_stats(index, name, embeddings, queries, k=10, exact_I=None):
    print(f"\n--- Benchmarking {name} ---")
    
    # Measure build time
    start_time = time.time()
    
    if name.startswith("IVF"):
        index.train(embeddings)
        
    index.add(embeddings)
    build_time = time.time() - start_time
    print(f"Build time: {build_time:.4f}s")
    
    # Save to disk to measure size
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_name = tmp.name
    
    try:
        faiss.write_index(index, tmp_name)
        size_mb = os.path.getsize(tmp_name) / (1024 * 1024)
        print(f"Index size: {size_mb:.2f} MB")
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        
    # Measure query latency
    start_time = time.time()
    D, I = index.search(queries, k)
    query_time = (time.time() - start_time) / len(queries)
    print(f"Avg Query latency: {query_time * 1000:.3f} ms")
    
    # Measure Recall@K if not exact
    if exact_I is not None:
        hits = 0
        for i in range(len(queries)):
            hits += len(set(I[i]).intersection(set(exact_I[i])))
        recall = hits / (len(queries) * k)
        print(f"Recall@{k}: {recall:.4f}")
    else:
        print(f"Recall@{k}: 1.0000 (Baseline)")
        
    return I

def main():
    print("Loading embeddings...")
    embeddings_path = "data/processed/song_embeddings_v1.npy"
    
    if not os.path.exists(embeddings_path):
        print(f"Error: {embeddings_path} not found.")
        return
        
    embeddings = np.load(embeddings_path).astype(np.float32)
    num_songs, dim = embeddings.shape
    
    print(f"Loaded {num_songs} embeddings of dimension {dim}")
    
    # Create 1000 random queries using same distribution
    np.random.seed(42)
    queries = np.random.randn(1000, dim).astype(np.float32)
    # L2 normalize queries since we expect inner product on L2-normalized vectors (cosine sim)
    faiss.normalize_L2(queries)
    
    # 1. IndexFlatIP (Exact Search Baseline)
    index_flat = faiss.IndexFlatIP(dim)
    exact_I = measure_index_stats(index_flat, "IndexFlatIP", embeddings, queries)
    
    # 2. HNSW (Hierarchical Navigable Small World)
    # M=32 (number of connections), efConstruction=40
    index_hnsw = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
    measure_index_stats(index_hnsw, "HNSW (M=32)", embeddings, queries, exact_I=exact_I)
    
    # 3. IVF (Inverted File Index)
    nlist = 1000 # number of clusters
    quantizer = faiss.IndexFlatIP(dim)
    index_ivf = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    # nprobe=10 means search 10/1000 clusters
    index_ivf.nprobe = 10
    measure_index_stats(index_ivf, "IVF (nlist=1000, nprobe=10)", embeddings, queries, exact_I=exact_I)
    
    # 4. IVF + PQ (Product Quantization)
    m = 8 # number of subquantizers (dim must be multiple of m)
    index_ivfpq = faiss.IndexIVFPQ(quantizer, dim, nlist, m, 8, faiss.METRIC_INNER_PRODUCT)
    index_ivfpq.nprobe = 10
    measure_index_stats(index_ivfpq, "IVFPQ (nlist=1000, nprobe=10, m=8)", embeddings, queries, exact_I=exact_I)

if __name__ == "__main__":
    main()
