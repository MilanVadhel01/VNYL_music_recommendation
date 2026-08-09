"""
Content-Based Recommendation -- Test Script
============================================

Loads the processed artifacts, initialises the ContentRecommender, and
runs a battery of validation checks.

Usage:
    python src/models/test_content_model.py
    python src/models/test_content_model.py --track-id 0pqnGHJpmpxLKifKRmU6WP
    python src/models/test_content_model.py --song-name "Believer"
    python src/models/test_content_model.py --song-name "Believer" --artist "Imagine Dragons"
"""

import argparse
import sys
import time

# Force UTF-8 output on Windows to handle international song names
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

# Ensure the project root is on sys.path so relative imports work
# regardless of the working directory.
sys.path.insert(0, ".")

from src.models.content_model import ContentRecommender  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _print_results(
    input_info: dict,
    results: list[dict],
    top_n: int,
) -> None:
    """Pretty-print recommendation results."""

    print()
    print("=" * 60)
    print("CONTENT-BASED RECOMMENDATION TEST")
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
        print(
            f"  {i:>2}. {rec['track_name']}"
        )
        print(
            f"      Artist:     {rec['artist_name']}"
        )
        print(
            f"      Similarity: {rec['similarity_score']:.4f}"
        )
        print()


def _validate_results(
    results: list[dict],
    input_track_id: str,
    top_n: int,
) -> bool:
    """Run validation checks and return True if all pass."""

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
        print(f"  [FAIL] Duplicates found: {len(track_ids)} results, {len(unique_ids)} unique")
        all_ok = False

    # 3. Query song excluded
    if input_track_id not in unique_ids:
        print(f"  [OK] Query excluded:   Input song not in results")
    else:
        print(f"  [FAIL] Query included:   Input song appears in results")
        all_ok = False

    # 4. Similarity ordering (descending)
    scores = [r["similarity_score"] for r in results]
    is_sorted = all(
        scores[i] >= scores[i + 1] for i in range(len(scores) - 1)
    )
    if is_sorted:
        print(f"  [OK] Sorted:           Descending by similarity")
    else:
        print(f"  [FAIL] Not sorted:       Results are not in descending order")
        all_ok = False

    # 5. Similarity values valid (cosine: -1 to 1, with fp tolerance)
    eps = 1e-6
    min_score = min(scores) if scores else 0
    max_score = max(scores) if scores else 0
    if -1.0 - eps <= min_score and max_score <= 1.0 + eps:
        print(
            f"  [OK] Scores valid:     range [{min_score:.4f}, {max_score:.4f}]"
        )
    else:
        print(
            f"  [FAIL] Scores invalid:   range [{min_score:.4f}, {max_score:.4f}] "
            f"(expected [-1, 1])"
        )
        all_ok = False

    print()
    if all_ok:
        print("  [OK] ALL CHECKS PASSED")
    else:
        print("  [FAIL] SOME CHECKS FAILED")
    print()

    return all_ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test the content-based music recommender."
    )
    parser.add_argument(
        "--track-id",
        type=str,
        default=None,
        help="Spotify track ID to query.",
    )
    parser.add_argument(
        "--song-name",
        type=str,
        default=None,
        help="Song name to search for.",
    )
    parser.add_argument(
        "--artist",
        type=str,
        default=None,
        help="Artist name (used with --song-name for disambiguation).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of recommendations (default: 10).",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load recommender
    # ------------------------------------------------------------------
    print("Loading ContentRecommender …")
    start = time.time()

    try:
        rec = ContentRecommender()
    except FileNotFoundError as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)

    load_time = time.time() - start
    print(
        f"  Loaded in {load_time:.1f}s  "
        f"({len(rec.feature_matrix):,d} songs, "
        f"{rec.feature_matrix.shape[1]} features)"
    )
    print()

    # ------------------------------------------------------------------
    # Determine query
    # ------------------------------------------------------------------
    input_track_id: str

    if args.track_id:
        input_track_id = args.track_id
        try:
            info = rec.get_song_info(input_track_id)
        except ValueError as exc:
            print(f"ERROR: {exc}")
            sys.exit(1)

    elif args.song_name:
        # Search and use the first result
        matches = rec.search_songs(args.song_name)
        if args.artist:
            matches = matches[
                matches["artist_name"].str.lower() == args.artist.lower()
            ]
        if matches.empty:
            extra = f" by '{args.artist}'" if args.artist else ""
            print(f"ERROR: Song not found: '{args.song_name}'{extra}")
            sys.exit(1)

        first = matches.iloc[0]
        input_track_id = first["track_id"]
        info = rec.get_song_info(input_track_id)

        if len(matches) > 1:
            print(f"Found {len(matches)} matches for '{args.song_name}':")
            for _, row in matches.iterrows():
                marker = " <- selected" if row["track_id"] == input_track_id else ""
                print(
                    f"  {row['track_id']}  {row['track_name']}  "
                    f"-- {row['artist_name']}{marker}"
                )
            print()

    else:
        # Default: use the first track in the dataset
        first_row = rec.song_index.iloc[0]
        input_track_id = first_row["track_id"]
        info = rec.get_song_info(input_track_id)
        print(f"No query specified — using first track in dataset.\n")

    # ------------------------------------------------------------------
    # Generate recommendations
    # ------------------------------------------------------------------
    start = time.time()
    results = rec.recommend_by_track_id(input_track_id, top_n=args.top_n)
    rec_time = time.time() - start

    _print_results(info, results, args.top_n)
    print(f"  Recommendation time: {rec_time:.2f}s\n")

    # ------------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------------
    _validate_results(results, input_track_id, args.top_n)


if __name__ == "__main__":
    main()
