"""
FAISS Index Builder
====================

Builds a FAISS IndexFlatIP index from the normalised content feature matrix.

Pipeline:
    content_features.npy (StandardScaler output)
        -> L2-normalize each vector
        -> convert to float32
        -> faiss.IndexFlatIP
        -> save to data/processed/faiss/content.index

With L2-normalized vectors, inner product == cosine similarity.

Usage:
    python src/models/build_faiss_index.py
"""

import os
import sys
import time

import faiss
import numpy as np

# Force UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INPUT_FEATURES: str = os.path.join("data", "processed", "content_features.npy")
OUTPUT_DIR: str = os.path.join("data", "processed", "faiss")
OUTPUT_INDEX: str = os.path.join(OUTPUT_DIR, "content.index")


def main() -> None:
    """Build and save the FAISS index."""

    print("=" * 60)
    print("VNYL Music Recommendation -- FAISS Index Builder")
    print("=" * 60)
    print()

    # ------------------------------------------------------------------
    # 1. Load feature matrix
    # ------------------------------------------------------------------
    if not os.path.exists(INPUT_FEATURES):
        print(f"ERROR: Feature matrix not found: {INPUT_FEATURES}")
        print("Run  python src/data/prepare_features.py  first.")
        sys.exit(1)

    print(f"Loading feature matrix: {INPUT_FEATURES}")
    X = np.load(INPUT_FEATURES)
    print(f"  Shape: {X.shape}")
    print(f"  Dtype: {X.dtype}")
    print()

    n_vectors, n_dims = X.shape

    if n_vectors == 0:
        print("ERROR: Feature matrix is empty.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. L2-normalize for cosine similarity via inner product
    # ------------------------------------------------------------------
    print("L2-normalizing vectors ...")
    start = time.perf_counter()

    # Compute L2 norms
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    # Avoid division by zero (should not happen with StandardScaler output)
    norms[norms == 0] = 1.0
    X_normalized = X / norms

    # Convert to float32 (FAISS requirement)
    X_normalized = X_normalized.astype(np.float32)

    norm_time = time.perf_counter() - start
    print(f"  Done in {norm_time:.2f}s")
    print(f"  Output dtype: {X_normalized.dtype}")
    print()

    # ------------------------------------------------------------------
    # 3. Build FAISS index
    # ------------------------------------------------------------------
    print(f"Building FAISS IndexFlatIP (d={n_dims}) ...")
    start = time.perf_counter()

    index = faiss.IndexFlatIP(n_dims)
    index.add(X_normalized)

    build_time = time.perf_counter() - start
    print(f"  Done in {build_time:.2f}s")
    print(f"  Vectors indexed: {index.ntotal:,d}")
    print()

    # ------------------------------------------------------------------
    # 4. Save index
    # ------------------------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Saving index: {OUTPUT_INDEX}")
    faiss.write_index(index, OUTPUT_INDEX)

    index_size_mb = os.path.getsize(OUTPUT_INDEX) / (1024 * 1024)
    print(f"  Index size: {index_size_mb:.1f} MB")
    print()

    # ------------------------------------------------------------------
    # 5. Quick verification
    # ------------------------------------------------------------------
    print("Verification:")
    loaded_index = faiss.read_index(OUTPUT_INDEX)
    print(f"  Loaded vectors: {loaded_index.ntotal:,d}")
    print(f"  Dimensions:     {loaded_index.d}")

    # Test a single query
    query = X_normalized[0:1]
    scores, indices = loaded_index.search(query, 5)
    print(f"  Test query top-5 indices: {indices[0].tolist()}")
    print(f"  Test query top-5 scores:  {[f'{s:.4f}' for s in scores[0]]}")
    print()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total_time = norm_time + build_time
    print("=" * 60)
    print("FAISS index build complete.")
    print(f"  Vectors:    {n_vectors:,d}")
    print(f"  Dimensions: {n_dims}")
    print(f"  Index type: IndexFlatIP (exact cosine via L2-norm + IP)")
    print(f"  Index file: {OUTPUT_INDEX}")
    print(f"  Index size: {index_size_mb:.1f} MB")
    print(f"  Total time: {total_time:.2f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
