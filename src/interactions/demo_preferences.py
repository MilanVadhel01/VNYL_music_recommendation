"""
Phase 1 Demo — Artist & Genre Preferences
==========================================

Demonstrates deriving user-level artist and genre preferences
from raw interaction events.

Usage:
    python src/interactions/demo_preferences.py
"""

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, ".")

from datetime import datetime, timezone, timedelta

from src.interactions.models import InteractionStore, ActionType
from src.interactions.preferences import (
    SongMetadataLookup,
    PreferenceStore,
)


def main():
    print("=" * 60)
    print("PHASE 1 DEMO — Artist & Genre Preferences")
    print("=" * 60)
    print()

    # 1. Setup
    store = InteractionStore()
    meta = SongMetadataLookup()

    # Register some songs with artist/genre metadata
    meta.register_song("song_A", "art_1", "Imagine Dragons", ["rock", "alternative", "pop rock"])
    meta.register_song("song_B", "art_1", "Imagine Dragons", ["rock", "alternative", "pop rock"])
    meta.register_song("song_C", "art_2", "The Weeknd", ["r&b", "pop", "synth-pop"])
    meta.register_song("song_D", "art_3", "Daft Punk", ["electronic", "house", "dance"])
    meta.register_song("song_E", "art_4", "Unknown Artist", [])  # no genres

    pref_store = PreferenceStore(store, meta)

    user_id = "demo_user"

    # 2. Simulate interactions
    t0 = datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)

    interactions = [
        # User loves Imagine Dragons
        ("song_A", ActionType.PLAY, dict(timestamp=t0)),
        ("song_A", ActionType.COMPLETE, dict(timestamp=t0 + timedelta(minutes=4))),
        ("song_A", ActionType.LIKE, dict(timestamp=t0 + timedelta(minutes=5))),
        ("song_A", ActionType.REPLAY, dict(timestamp=t0 + timedelta(minutes=10), replay_count=2)),
        ("song_B", ActionType.PLAY, dict(timestamp=t0 + timedelta(hours=1))),
        ("song_B", ActionType.ADD_TO_PLAYLIST, dict(timestamp=t0 + timedelta(hours=1, minutes=5))),

        # User likes The Weeknd
        ("song_C", ActionType.PLAY, dict(timestamp=t0 + timedelta(hours=2))),
        ("song_C", ActionType.COMPLETE, dict(timestamp=t0 + timedelta(hours=2, minutes=4))),

        # User skips Daft Punk
        ("song_D", ActionType.PLAY, dict(timestamp=t0 + timedelta(hours=3))),
        ("song_D", ActionType.SKIP, dict(timestamp=t0 + timedelta(hours=3, minutes=0, seconds=8), listen_duration=8.0)),

        # User plays unknown artist (no genres)
        ("song_E", ActionType.PLAY, dict(timestamp=t0 + timedelta(hours=4))),
        ("song_E", ActionType.COMPLETE, dict(timestamp=t0 + timedelta(hours=4, minutes=3))),
    ]

    print("Simulated Interactions:")
    print("-" * 50)
    for song_id, action, kwargs in interactions:
        store.record_interaction(user_id, song_id, action, **kwargs)
        print(f"  {action.value:20s}  →  {song_id}")
    print()

    # 3. Build preferences
    pref_store.rebuild_all(user_id)

    # 4. Display top artists
    print("Top Artists for demo_user:")
    print("-" * 50)
    top_artists = pref_store.get_top_artists(user_id, top_k=5)
    for i, a in enumerate(top_artists, 1):
        print(
            f"  {i}. {a.artist_name:20s}  "
            f"score={a.preference_score:+.4f}  "
            f"raw={a.raw_score:+.1f}  "
            f"count={a.interaction_count}  "
            f"pos={a.positive_interactions}  "
            f"neg={a.negative_interactions}"
        )
    print()

    # 5. Display top genres
    print("Top Genres for demo_user:")
    print("-" * 50)
    top_genres = pref_store.get_top_genres(user_id, top_k=10)
    for i, g in enumerate(top_genres, 1):
        print(
            f"  {i}. {g.genre:20s}  "
            f"score={g.preference_score:+.4f}  "
            f"raw={g.raw_score:+.1f}  "
            f"count={g.interaction_count}  "
            f"pos={g.positive_interactions}  "
            f"neg={g.negative_interactions}"
        )
    print()

    # 6. Verify raw events unchanged
    all_events = store.get_all_events()
    print(f"Raw events preserved: {len(all_events)} events (immutable)")
    print()

    print("=" * 60)
    print("Phase 1 Demo Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
