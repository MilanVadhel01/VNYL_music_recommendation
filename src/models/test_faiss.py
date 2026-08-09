"""
FAISS Recommendation -- Test Script
=====================================

Loads the FAISS index, runs a recommendation query, and validates the results.

Usage:
    python src/models/test_faiss.py
    python src/models/test_faiss.py --track-id 0pqnGHJpmpxLKifKRmU6WP
    python src/models/test_faiss.py --song-name "Believer" --artist "Imagine Dragons"
"""

import argparse
import sys
import time

# Force UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, ".")

from src.models.faiss_recommender import FAISSContentRecommender  # noqa: E402


def _print_results(input_info: dict, results: list[dict]) -> None:
    print()
    print("=" * 60)
    print("FAISS RECOMMENDATION TEST")
    print("=" * 60)
    print()
    print("Input:")
    print(f"  Track:     {input_info.get('track_name', 'N/A')}")
    print(f"  Artist:    {input_info.get('artist_name', 'N/A')}")
    print(f"  Track ID:  {input_info.get('track_id', 'N/A')}")
    print()
    print(f"Top {len(results)} Similar Songs:")
    print("-" * 60)

    for i, rec in enumerate(results, start=1):
        print(f"  {i:>2}. {rec['track_name']}")
        print(f"      Artist:     {rec['artist_name']}")
        print(f"      Similarity: {rec['similarity_score']:.4f}")
        print()


def _validate_results(
    results: list[dict], input_track_id: str, top_n: int
) -> bool:
    print("=" * 60)
    print("VALIDATION")
    print("=" * 60)
    all_ok = True

    # 1. Correct output count
    if len(results) <= top_n:
        print(f"  [OK] Output count:     {len(results)} <= {top_n}")
    else:
        print(f"  [FAIL] Output count:     {len(results)} > {top_n}")
        all_ok = False

    # 2. No duplicate results
    track_ids = [r["track_id"] for r in results]
    unique_ids = set(track_ids)
    if len(track_ids) == len(unique_ids):
        print(f"  [OK] No duplicates:    {len(unique_ids)} unique results")
    else:
        print(f"  [FAIL] Duplicates found")
        all_ok = False

    # 3. Query song excluded
    if input_track_id not in unique_ids:
        print(f"  [OK] Query excluded:   Input song not in results")
    else:
        print(f"  [FAIL] Query included")
        all_ok = False

    # 4. Similarity ordering
    scores = [r["similarity_score"] for r in results]
    is_sorted = all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))
    if is_sorted:
        print(f"  [OK] Sorted:           Descending by similarity")
    else:
        print(f"  [FAIL] Not sorted")
        all_ok = False

    # 5. Score range (cosine via IP on L2-normed vectors: -1 to 1, with fp tolerance)
    eps = 1e-6
    min_s = min(scores) if scores else 0
    max_s = max(scores) if scores else 0
    if -1.0 - eps <= min_s and max_s <= 1.0 + eps:
        print(f"  [OK] Scores valid:     range [{min_s:.4f}, {max_s:.4f}]")
    else:
        print(f"  [FAIL] Scores invalid: range [{min_s:.4f}, {max_s:.4f}]")
        all_ok = False

    print()
    if all_ok:
        print("  [OK] ALL CHECKS PASSED")
    else:
        print("  [FAIL] SOME CHECKS FAILED")
    print()
    return all_ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Test the FAISS recommender.")
    parser.add_argument("--track-id", type=str, default=None)
    parser.add_argument("--song-name", type=str, default=None)
    parser.add_argument("--artist", type=str, default=None)
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args()

    print("Loading FAISSContentRecommender ...")
    start = time.perf_counter()
    try:
        rec = FAISSContentRecommender()
    except FileNotFoundError as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)

    load_time = time.perf_counter() - start
    print(
        f"  Loaded in {load_time:.1f}s  "
        f"({rec.index.ntotal:,d} vectors, {rec.index.d} dims)"
    )
    print()

    # Determine query
    if args.track_id:
        input_track_id = args.track_id
        info = rec.get_song_info(input_track_id)
    elif args.song_name:
        matches = rec.search_songs(args.song_name)
        if args.artist:
            matches = matches[matches["artist_name"].str.lower() == args.artist.lower()]
        if matches.empty:
            extra = f" by '{args.artist}'" if args.artist else ""
            print(f"ERROR: Song not found: '{args.song_name}'{extra}")
            sys.exit(1)
        input_track_id = matches.iloc[0]["track_id"]
        info = rec.get_song_info(input_track_id)
    else:
        first = rec.song_index.iloc[0]
        input_track_id = first["track_id"]
        info = rec.get_song_info(input_track_id)
        print("No query specified -- using first track in dataset.\n")

    # Generate recommendations
    start = time.perf_counter()
    results = rec.recommend_by_track_id(input_track_id, top_n=args.top_n)
    rec_time = (time.perf_counter() - start) * 1000

    _print_results(info, results)
    print(f"  Retrieval time: {rec_time:.2f} ms\n")

    _validate_results(results, input_track_id, args.top_n)


if __name__ == "__main__":
    main()
