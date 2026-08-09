import pytest
from datetime import datetime, timezone, timedelta

from src.interactions.models import ActionType, UserInteraction, InteractionStore
from src.interactions.ups import (
    calculate_event_score,
    calculate_user_song_preference,
    apply_decay,
)

# ---------------------------------------------------------------------------
# Test calculate_event_score (Single Events)
# ---------------------------------------------------------------------------

def test_explicit_weights():
    assert calculate_event_score(UserInteraction("u1", "s1", ActionType.LIKE)) == 10.0
    assert calculate_event_score(UserInteraction("u1", "s1", ActionType.DISLIKE)) == -10.0
    assert calculate_event_score(UserInteraction("u1", "s1", ActionType.ADD_TO_PLAYLIST)) == 8.0

def test_listening_base_weights():
    assert calculate_event_score(UserInteraction("u1", "s1", ActionType.PLAY)) == 1.0
    assert calculate_event_score(UserInteraction("u1", "s1", ActionType.COMPLETE)) == 5.0

def test_replay_logic():
    # REPLAY_ONCE -> +6
    e1 = UserInteraction("u1", "s1", ActionType.REPLAY, replay_count=1)
    assert calculate_event_score(e1) == 6.0
    
    # REPLAY_MULTIPLE -> +10
    e2 = UserInteraction("u1", "s1", ActionType.REPLAY, replay_count=3)
    assert calculate_event_score(e2) == 10.0
    
    # Default (no count) -> assume 1
    e3 = UserInteraction("u1", "s1", ActionType.REPLAY)
    assert calculate_event_score(e3) == 6.0

def test_skip_precedence():
    # SKIP < 10 sec -> -8
    e1 = UserInteraction("u1", "s1", ActionType.SKIP, listen_duration=5.0)
    assert calculate_event_score(e1) == -8.0
    
    # SKIP < 30 sec -> -6
    e2 = UserInteraction("u1", "s1", ActionType.SKIP, listen_duration=25.0)
    assert calculate_event_score(e2) == -6.0
    
    # SKIP after 50% -> -2
    e3 = UserInteraction("u1", "s1", ActionType.SKIP, listen_duration=60.0, completion_percentage=60.0)
    assert calculate_event_score(e3) == -2.0
    
    # SKIP after 90% -> 0
    e4 = UserInteraction("u1", "s1", ActionType.SKIP, listen_duration=120.0, completion_percentage=95.0)
    assert calculate_event_score(e4) == 0.0

# ---------------------------------------------------------------------------
# Test Aggregation (Multiple Events)
# ---------------------------------------------------------------------------

def test_user_song_preference_aggregation():
    interactions = [
        UserInteraction("u1", "s1", ActionType.PLAY),                # +1
        UserInteraction("u1", "s1", ActionType.COMPLETE),            # +5
        UserInteraction("u1", "s1", ActionType.LIKE),                # +10
        UserInteraction("u1", "s1", ActionType.ADD_TO_PLAYLIST)      # +8
    ]
    
    pref = calculate_user_song_preference("u1", "s1", interactions)
    assert pref.preference_score == 24.0
    assert pref.interaction_count == 4

def test_completion_tiers_during_aggregation():
    # Play event that also reached 80% completion (should add +4 for tier >= 75)
    # Total: PLAY(1.0) + COMPLETION(4.0) = 5.0
    interactions = [
        UserInteraction(
            "u1", "s1", ActionType.PLAY, 
            completion_percentage=80.0
        )
    ]
    pref = calculate_user_song_preference("u1", "s1", interactions)
    assert pref.preference_score == 5.0

def test_aggregation_filters_correctly():
    interactions = [
        UserInteraction("u1", "s1", ActionType.LIKE),    # +10 (u1, s1)
        UserInteraction("u1", "s2", ActionType.LIKE),    # other song
        UserInteraction("u2", "s1", ActionType.LIKE),    # other user
    ]
    
    pref = calculate_user_song_preference("u1", "s1", interactions)
    assert pref.preference_score == 10.0
    assert pref.interaction_count == 1

# ---------------------------------------------------------------------------
# Test Decay
# ---------------------------------------------------------------------------

def test_apply_decay():
    score = 10.0
    
    # 0 days -> no decay
    assert apply_decay(score, 0) == 10.0
    
    # 1 day -> 10 * 0.99 = 9.9
    assert abs(apply_decay(score, 1) - 9.9) < 1e-5
    
    # 30 days -> 10 * (0.99^30)
    expected_30 = 10.0 * (0.99 ** 30)
    assert abs(apply_decay(score, 30) - expected_30) < 1e-5

# ---------------------------------------------------------------------------
# Test InteractionStore
# ---------------------------------------------------------------------------

def test_interaction_store_validation():
    store = InteractionStore()
    
    # Valid
    event = store.record_interaction("u1", "s1", ActionType.PLAY)
    assert event.user_id == "u1"
    assert event.action == ActionType.PLAY
    
    # Invalid: completion > 100
    with pytest.raises(ValueError):
        store.record_interaction("u1", "s1", ActionType.PLAY, completion_percentage=150.0)
        
    # Invalid: negative duration
    with pytest.raises(ValueError):
        store.record_interaction("u1", "s1", ActionType.PLAY, listen_duration=-5.0)
