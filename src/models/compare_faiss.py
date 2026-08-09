"""
FAISS vs Brute-Force Comparison
================================

Validates that the FAISS index produces the same results as the brute-force
cosine similarity baseline, and measures latency differences.

Usage:
    python src/models/compare_faiss.py
"""

import sys
import time

# Force UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, ".")

from src.models.content_model import ContentRecommender  # noqa: E402
from src.models.faiss_recommender import FAISSContentRecommender  # noqa: E402

# ---------------------------------------------------------------------------
# Test songs (verified to exist in the dataset)
# ---------------------------------------------------------------------------
TEST_SONGS = [
    ("0pqnGHJpmpxLKifKRmU6WP", "Believer", "Imagine Dragons"),
    ("2plbrEY59IikOBgBGLjaoe", "Die With A Smile", "Bruno Mars"),
    ("6dOtVTDdiauQNBQEDOtlAB", "BIRDS OF A FEATHER", "Billie Eilish"),
]

TOP_N = 10


def main() -> None:
    print("=" * 60)
    print("FAISS vs BRUTE-FORCE COMPARISON")
    print("=" * 60)
    print()

    # Load both recommenders
    print("Loading brute-force recommender ...")
    bf = ContentRecommender()
    print("Loading FAISS recommender ...")
    faiss_rec = FAISSContentRecommender()
    print()

    bf_latencies: list[float] = []
    faiss_latencies: list[float] = []
    all_pass = True

    for track_id, track_name, artist_name in TEST_SONGS:
        print("-" * 60)
        print(f"Query: {track_name} -- {artist_name}")
        print(f"       {track_id}")
        print()

        # Brute-force
        t0 = time.perf_counter()
        bf_results = bf.recommend_by_track_id(track_id, top_n=TOP_N)
        bf_time = (time.perf_counter() - t0) * 1000
        bf_latencies.append(bf_time)

        # FAISS
        t0 = time.perf_counter()
        faiss_results = faiss_rec.recommend_by_track_id(track_id, top_n=TOP_N)
        faiss_time = (time.perf_counter() - t0) * 1000
        faiss_latencies.append(faiss_time)

        # Compare
        bf_ids = [r["track_id"] for r in bf_results]
        faiss_ids = [r["track_id"] for r in faiss_results]

        overlap = len(set(bf_ids) & set(faiss_ids))
        match = overlap == TOP_N

        print(f"  Brute-force:  {bf_time:>7.2f} ms")
        print(f"  FAISS:        {faiss_time:>7.2f} ms")
        print(f"  Overlap:      {overlap} / {TOP_N}", end="")
        if match:
            print("  [OK]")
        else:
            print("  [MISMATCH]")
            all_pass = False

        # Show side-by-side top 3
        print()
        print(f"  {'Rank':<5} {'Brute-Force':<35} {'FAISS':<35}")
        for i in range(min(3, len(bf_results), len(faiss_results))):
            bf_name = f"{bf_results[i]['track_name'][:30]}"
            fa_name = f"{faiss_results[i]['track_name'][:30]}"
            bf_score = bf_results[i]["similarity_score"]
            fa_score = faiss_results[i]["similarity_score"]
            print(f"  {i+1:<5} {bf_name:<30} {bf_score:.4f}   {fa_name:<30} {fa_score:.4f}")
        print()

    # Summary
    print("=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    avg_bf = sum(bf_latencies) / len(bf_latencies)
    avg_faiss = sum(faiss_latencies) / len(faiss_latencies)

    print(f"  Average brute-force latency: {avg_bf:.2f} ms")
    print(f"  Average FAISS latency:       {avg_faiss:.2f} ms")

    if avg_bf > 0:
        speedup = avg_bf / avg_faiss if avg_faiss > 0 else float("inf")
        print(f"  Speed improvement:           {speedup:.1f}x")

    print()
    if all_pass:
        print("  [OK] ALL QUERIES MATCH -- FAISS results are identical to brute-force")
    else:
        print("  [WARNING] Some queries do not fully match -- investigate before continuing")
    print()


if __name__ == "__main__":
    main()
