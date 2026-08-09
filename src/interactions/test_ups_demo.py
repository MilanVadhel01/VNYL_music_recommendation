"""
UPS Demonstration
=================

Simulates a user interacting with a song and calculates the resulting UPS.

Usage:
    python src/interactions/test_ups_demo.py
"""

import sys

# Force UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, ".")

from src.interactions.models import InteractionStore, ActionType
from src.interactions.ups import (
    calculate_event_score,
    calculate_user_song_preference,
)


def main():
    store = InteractionStore()
    user_id = "demo_user"
    song_id = "Believer"

    # Simulate events
    events = [
        store.record_interaction(user_id, song_id, ActionType.PLAY),
        store.record_interaction(user_id, song_id, ActionType.COMPLETE),
        store.record_interaction(user_id, song_id, ActionType.REPLAY, replay_count=1),
        store.record_interaction(user_id, song_id, ActionType.LIKE),
        store.record_interaction(user_id, song_id, ActionType.ADD_TO_PLAYLIST),
    ]

    print("=" * 50)
    print("UPS DEMONSTRATION")
    print("=" * 50)
    print()
    print(f"User:\n{user_id}")
    print()
    print(f"Song:\n{song_id}")
    print()
    print("Interactions:")
    print()

    # We manually show the score per event to match the requested output
    for event in events:
        score = calculate_event_score(event)
        # Pad action name for alignment
        action_str = event.action.value.ljust(15)
        print(f"{action_str}  +{score:g}")

    # Calculate final aggregate score
    pref = calculate_user_song_preference(user_id, song_id, events)

    print()
    print("Total UPS:")
    print(f"{pref.preference_score:g}")
    print()


if __name__ == "__main__":
    main()
