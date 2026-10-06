"""
Tests for Artist & Genre Preferences (Phase 1)
================================================

Validates that:
- Positive interactions increase preference
- Negative interactions decrease preference
- Replay/completion contribute positively
- Early skip contributes negatively
- Playlist addition contributes positively
- Artist search behaves per existing scoring rules
- Multiple genres aggregate correctly
- Missing genres do not crash
- Normalization stays within [-1, +1]
- Decay works consistently
- Rebuild is deterministic
- Raw interaction events are unchanged
"""

import pytest
import math
from datetime import datetime, timezone, timedelta

from src.interactions.models import ActionType, UserInteraction, InteractionStore
from src.interactions.ups import calculate_event_score, apply_decay
from src.interactions.preferences import (
    UserArtistPreference,
    UserGenrePreference,
    SongMetadataLookup,
    PreferenceStore,
    normalize_preference,
    _parse_genres,
    NORM_SCALE,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def store_and_meta():
    """Create a fresh InteractionStore + SongMetadataLookup + PreferenceStore."""
    istore = InteractionStore()
    meta = SongMetadataLookup()

    # Register test songs
    meta.register_song("s1", "artist_1", "Artist One", ["rock", "alternative"])
    meta.register_song("s2", "artist_1", "Artist One", ["rock"])
    meta.register_song("s3", "artist_2", "Artist Two", ["pop", "dance"])
    meta.register_song("s4", "artist_3", "Artist Three", [])  # no genres
    meta.register_song("s5", "artist_4", "Artist Four", ["rock", "pop", "indie"])

    pstore = PreferenceStore(istore, meta)
    return istore, meta, pstore


# ---------------------------------------------------------------------------
# Test: Positive interaction increases preference
# ---------------------------------------------------------------------------

def test_positive_interaction_increases_preference(store_and_meta):
    istore, meta, pstore = store_and_meta

    istore.record_interaction("u1", "s1", ActionType.LIKE)
    pstore.rebuild_artist_preferences("u1")

    pref = pstore.get_artist_preference("u1", "artist_1")
    assert pref is not None
    assert pref.raw_score > 0
    assert pref.preference_score > 0
    assert pref.positive_interactions == 1
    assert pref.negative_interactions == 0


# ---------------------------------------------------------------------------
# Test: Negative interaction decreases preference
# ---------------------------------------------------------------------------

def test_negative_interaction_decreases_preference(store_and_meta):
    istore, meta, pstore = store_and_meta

    istore.record_interaction("u1", "s1", ActionType.DISLIKE)
    pstore.rebuild_artist_preferences("u1")

    pref = pstore.get_artist_preference("u1", "artist_1")
    assert pref is not None
    assert pref.raw_score < 0
    assert pref.preference_score < 0
    assert pref.positive_interactions == 0
    assert pref.negative_interactions == 1


# ---------------------------------------------------------------------------
# Test: Replay contributes positively
# ---------------------------------------------------------------------------

def test_replay_contributes_positively(store_and_meta):
    istore, meta, pstore = store_and_meta

    istore.record_interaction("u1", "s1", ActionType.REPLAY, replay_count=3)
    pstore.rebuild_artist_preferences("u1")

    pref = pstore.get_artist_preference("u1", "artist_1")
    assert pref is not None
    assert pref.raw_score == 10.0  # REPLAY_MULTIPLE
    assert pref.preference_score > 0


# ---------------------------------------------------------------------------
# Test: Completion contributes positively
# ---------------------------------------------------------------------------

def test_completion_contributes_positively(store_and_meta):
    istore, meta, pstore = store_and_meta

    istore.record_interaction("u1", "s1", ActionType.COMPLETE)
    pstore.rebuild_artist_preferences("u1")

    pref = pstore.get_artist_preference("u1", "artist_1")
    assert pref is not None
    assert pref.raw_score == 5.0  # COMPLETE
    assert pref.preference_score > 0


# ---------------------------------------------------------------------------
# Test: Early skip contributes negatively
# ---------------------------------------------------------------------------

def test_early_skip_contributes_negatively(store_and_meta):
    istore, meta, pstore = store_and_meta

    istore.record_interaction(
        "u1", "s1", ActionType.SKIP, listen_duration=5.0
    )
    pstore.rebuild_artist_preferences("u1")

    pref = pstore.get_artist_preference("u1", "artist_1")
    assert pref is not None
    assert pref.raw_score == -8.0  # SKIP < 10 sec
    assert pref.preference_score < 0
    assert pref.negative_interactions == 1


# ---------------------------------------------------------------------------
# Test: Playlist addition contributes positively
# ---------------------------------------------------------------------------

def test_playlist_addition_contributes_positively(store_and_meta):
    istore, meta, pstore = store_and_meta

    istore.record_interaction("u1", "s1", ActionType.ADD_TO_PLAYLIST)
    pstore.rebuild_artist_preferences("u1")

    pref = pstore.get_artist_preference("u1", "artist_1")
    assert pref is not None
    assert pref.raw_score == 8.0
    assert pref.preference_score > 0


# ---------------------------------------------------------------------------
# Test: Artist search behaves per existing scoring rules
# ---------------------------------------------------------------------------

def test_artist_search_scoring(store_and_meta):
    istore, meta, pstore = store_and_meta

    # SEARCH_ARTIST has weight +4 in UPS
    istore.record_interaction("u1", "s1", ActionType.SEARCH_ARTIST)
    pstore.rebuild_artist_preferences("u1")

    pref = pstore.get_artist_preference("u1", "artist_1")
    assert pref is not None
    assert pref.raw_score == 4.0


# ---------------------------------------------------------------------------
# Test: Multiple genres aggregate correctly
# ---------------------------------------------------------------------------

def test_multiple_genres_aggregate(store_and_meta):
    istore, meta, pstore = store_and_meta

    # s1 has genres: ["rock", "alternative"]
    istore.record_interaction("u1", "s1", ActionType.LIKE)  # +10
    pstore.rebuild_genre_preferences("u1")

    rock = pstore.get_genre_preference("u1", "rock")
    alt = pstore.get_genre_preference("u1", "alternative")

    assert rock is not None
    assert alt is not None
    assert rock.raw_score == 10.0
    assert alt.raw_score == 10.0


def test_genre_aggregation_across_songs(store_and_meta):
    istore, meta, pstore = store_and_meta

    # s1 genres: ["rock", "alternative"], s2 genres: ["rock"]
    istore.record_interaction("u1", "s1", ActionType.LIKE)     # +10 to rock, alt
    istore.record_interaction("u1", "s2", ActionType.COMPLETE)  # +5 to rock
    pstore.rebuild_genre_preferences("u1")

    rock = pstore.get_genre_preference("u1", "rock")
    alt = pstore.get_genre_preference("u1", "alternative")

    assert rock is not None
    assert alt is not None
    assert rock.raw_score == 15.0  # 10 + 5
    assert alt.raw_score == 10.0   # only from s1
    assert rock.interaction_count == 2
    assert alt.interaction_count == 1


# ---------------------------------------------------------------------------
# Test: Missing genres do not crash
# ---------------------------------------------------------------------------

def test_missing_genres_no_crash(store_and_meta):
    istore, meta, pstore = store_and_meta

    # s4 has no genres
    istore.record_interaction("u1", "s4", ActionType.LIKE)
    pstore.rebuild_genre_preferences("u1")

    # No genres should be created for u1
    top = pstore.get_top_genres("u1", top_k=10)
    assert len(top) == 0


def test_unknown_song_no_crash(store_and_meta):
    istore, meta, pstore = store_and_meta

    # "unknown_song" is not registered in metadata
    istore.record_interaction("u1", "unknown_song", ActionType.LIKE)
    pstore.rebuild_artist_preferences("u1")
    pstore.rebuild_genre_preferences("u1")

    assert pstore.get_top_artists("u1") == []
    assert pstore.get_top_genres("u1") == []


# ---------------------------------------------------------------------------
# Test: Normalization stays within [-1, +1]
# ---------------------------------------------------------------------------

def test_normalization_bounds():
    """Verify that normalize_preference always returns values in [-1, +1]."""
    test_values = [-1000, -100, -50, -10, -1, 0, 1, 10, 50, 100, 1000]
    for v in test_values:
        result = normalize_preference(v)
        assert -1.0 <= result <= 1.0, f"normalize_preference({v}) = {result}"


def test_normalization_monotonic():
    """Verify monotonicity."""
    prev = normalize_preference(-1000)
    for v in range(-999, 1001):
        curr = normalize_preference(v)
        assert curr >= prev, f"Not monotonic at {v}"
        prev = curr


def test_normalization_symmetry():
    """Verify symmetry around zero."""
    for v in [1, 5, 10, 30, 100]:
        assert abs(normalize_preference(v) + normalize_preference(-v)) < 1e-10


def test_normalization_zero():
    assert normalize_preference(0.0) == 0.0


def test_preference_scores_bounded(store_and_meta):
    """Verify computed preferences are bounded."""
    istore, meta, pstore = store_and_meta

    # Generate many interactions
    for _ in range(50):
        istore.record_interaction("u1", "s1", ActionType.LIKE)
    pstore.rebuild_artist_preferences("u1")

    pref = pstore.get_artist_preference("u1", "artist_1")
    assert -1.0 <= pref.preference_score <= 1.0


# ---------------------------------------------------------------------------
# Test: Decay works consistently
# ---------------------------------------------------------------------------

def test_decay_reduces_preference(store_and_meta):
    istore, meta, pstore = store_and_meta

    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    istore.record_interaction("u1", "s1", ActionType.LIKE, timestamp=t0)
    pstore.rebuild_artist_preferences("u1")

    # Reference time 30 days later
    ref = t0 + timedelta(days=30)
    pref_no_decay = pstore.get_artist_preference("u1", "artist_1")
    pref_with_decay = pstore.get_artist_preference_with_decay(
        "u1", "artist_1", reference_time=ref
    )

    assert pref_with_decay is not None
    assert pref_with_decay.raw_score < pref_no_decay.raw_score
    assert pref_with_decay.preference_score < pref_no_decay.preference_score


def test_decay_consistent_with_ups(store_and_meta):
    """Verify decay uses the same apply_decay function from ups.py."""
    istore, meta, pstore = store_and_meta

    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    istore.record_interaction("u1", "s1", ActionType.LIKE, timestamp=t0)
    pstore.rebuild_artist_preferences("u1")

    ref = t0 + timedelta(days=10)
    pref = pstore.get_artist_preference_with_decay("u1", "artist_1", reference_time=ref)

    expected_raw = apply_decay(10.0, 10.0)  # LIKE = +10, 10 days
    assert abs(pref.raw_score - expected_raw) < 1e-10


def test_genre_decay(store_and_meta):
    istore, meta, pstore = store_and_meta

    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    istore.record_interaction("u1", "s1", ActionType.LIKE, timestamp=t0)
    pstore.rebuild_genre_preferences("u1")

    ref = t0 + timedelta(days=30)
    pref = pstore.get_genre_preference_with_decay("u1", "rock", reference_time=ref)
    assert pref is not None
    assert pref.raw_score < 10.0  # decayed from 10.0


# ---------------------------------------------------------------------------
# Test: Rebuild is deterministic
# ---------------------------------------------------------------------------

def test_rebuild_deterministic(store_and_meta):
    istore, meta, pstore = store_and_meta

    istore.record_interaction("u1", "s1", ActionType.LIKE)
    istore.record_interaction("u1", "s2", ActionType.COMPLETE)
    istore.record_interaction("u1", "s3", ActionType.DISLIKE)

    pstore.rebuild_all("u1")
    artists_1 = [
        (p.artist_id, p.raw_score, p.preference_score)
        for p in pstore.get_top_artists("u1", top_k=10)
    ]
    genres_1 = [
        (p.genre, p.raw_score, p.preference_score)
        for p in pstore.get_top_genres("u1", top_k=10)
    ]

    # Rebuild again
    pstore.rebuild_all("u1")
    artists_2 = [
        (p.artist_id, p.raw_score, p.preference_score)
        for p in pstore.get_top_artists("u1", top_k=10)
    ]
    genres_2 = [
        (p.genre, p.raw_score, p.preference_score)
        for p in pstore.get_top_genres("u1", top_k=10)
    ]

    assert artists_1 == artists_2
    assert genres_1 == genres_2


# ---------------------------------------------------------------------------
# Test: Raw interaction events are unchanged
# ---------------------------------------------------------------------------

def test_raw_events_unchanged(store_and_meta):
    istore, meta, pstore = store_and_meta

    e1 = istore.record_interaction("u1", "s1", ActionType.LIKE)
    e2 = istore.record_interaction("u1", "s2", ActionType.DISLIKE)

    events_before = list(istore.get_all_events())
    pstore.rebuild_all("u1")
    events_after = list(istore.get_all_events())

    assert len(events_before) == len(events_after)
    for eb, ea in zip(events_before, events_after):
        assert eb.id == ea.id
        assert eb.user_id == ea.user_id
        assert eb.song_id == ea.song_id
        assert eb.action == ea.action
        assert eb.timestamp == ea.timestamp


# ---------------------------------------------------------------------------
# Test: Top-K ordering
# ---------------------------------------------------------------------------

def test_top_artists_ordering(store_and_meta):
    istore, meta, pstore = store_and_meta

    # artist_1 gets +10, artist_2 gets -10
    istore.record_interaction("u1", "s1", ActionType.LIKE)      # artist_1: +10
    istore.record_interaction("u1", "s3", ActionType.DISLIKE)    # artist_2: -10
    pstore.rebuild_artist_preferences("u1")

    top = pstore.get_top_artists("u1", top_k=2)
    assert len(top) == 2
    assert top[0].artist_id == "artist_1"
    assert top[1].artist_id == "artist_2"


def test_top_genres_ordering(store_and_meta):
    istore, meta, pstore = store_and_meta

    istore.record_interaction("u1", "s1", ActionType.LIKE)      # rock +10, alt +10
    istore.record_interaction("u1", "s3", ActionType.DISLIKE)    # pop -10, dance -10
    pstore.rebuild_genre_preferences("u1")

    top = pstore.get_top_genres("u1", top_k=10)
    assert len(top) == 4
    # rock and alternative should be first (both +10)
    top_genres = [g.genre for g in top]
    assert "rock" in top_genres[:2]
    assert "alternative" in top_genres[:2]


# ---------------------------------------------------------------------------
# Test: Genre parsing edge cases
# ---------------------------------------------------------------------------

def test_parse_genres_list_literal():
    assert _parse_genres("['rock', 'pop']") == ["rock", "pop"]


def test_parse_genres_actual_list():
    assert _parse_genres(["Rock", " Pop "]) == ["rock", "pop"]


def test_parse_genres_empty():
    assert _parse_genres("") == []
    assert _parse_genres(None) == []
    assert _parse_genres(float("nan")) == []


def test_parse_genres_comma_separated():
    result = _parse_genres("rock, indie, folk")
    assert result == ["rock", "indie", "folk"]


# ---------------------------------------------------------------------------
# Test: Multiple users isolation
# ---------------------------------------------------------------------------

def test_user_isolation(store_and_meta):
    istore, meta, pstore = store_and_meta

    istore.record_interaction("u1", "s1", ActionType.LIKE)     # u1 likes artist_1
    istore.record_interaction("u2", "s1", ActionType.DISLIKE)  # u2 dislikes artist_1
    pstore.rebuild_artist_preferences()

    p1 = pstore.get_artist_preference("u1", "artist_1")
    p2 = pstore.get_artist_preference("u2", "artist_1")

    assert p1.raw_score > 0
    assert p2.raw_score < 0


# ---------------------------------------------------------------------------
# Test: Rebuild for specific user doesn't affect others
# ---------------------------------------------------------------------------

def test_rebuild_single_user(store_and_meta):
    istore, meta, pstore = store_and_meta

    istore.record_interaction("u1", "s1", ActionType.LIKE)
    istore.record_interaction("u2", "s3", ActionType.COMPLETE)
    pstore.rebuild_all()

    p_u2_before = pstore.get_artist_preference("u2", "artist_2")

    # Add more interactions for u1 and rebuild only u1
    istore.record_interaction("u1", "s2", ActionType.REPLAY, replay_count=3)
    pstore.rebuild_artist_preferences("u1")

    # u2's preference should be unchanged
    p_u2_after = pstore.get_artist_preference("u2", "artist_2")
    assert p_u2_before.raw_score == p_u2_after.raw_score


# ---------------------------------------------------------------------------
# Test: Song contributing to artist (two songs same artist)
# ---------------------------------------------------------------------------

def test_two_songs_same_artist(store_and_meta):
    istore, meta, pstore = store_and_meta

    # s1 and s2 both belong to artist_1
    istore.record_interaction("u1", "s1", ActionType.LIKE)      # +10
    istore.record_interaction("u1", "s2", ActionType.COMPLETE)   # +5
    pstore.rebuild_artist_preferences("u1")

    pref = pstore.get_artist_preference("u1", "artist_1")
    assert pref.raw_score == 15.0
    assert pref.interaction_count == 2
