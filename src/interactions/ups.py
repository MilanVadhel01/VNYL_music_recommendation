"""
User Preference Score (UPS) Calculator
======================================

Calculates preference scores based on raw user interactions.

Scores are derived heuristically based on the project specification.
They are NOT machine-learned parameters.

Raw event scores remain immutable.
"""

from typing import List
from datetime import datetime, timezone
import math

from src.interactions.models import ActionType, UserInteraction, UserSongPreference

# ---------------------------------------------------------------------------
# Initial Heuristic Weights
# ---------------------------------------------------------------------------

EXPLICIT_WEIGHTS = {
    ActionType.LIKE: 10.0,
    ActionType.DISLIKE: -10.0,
    ActionType.ADD_TO_PLAYLIST: 8.0,
    ActionType.REMOVE_LIKE: -8.0,
    ActionType.REMOVE_FROM_PLAYLIST: -5.0,
    ActionType.SHARE: 7.0,
}

# General listening base points
LISTEN_BASE_WEIGHT = {
    ActionType.PLAY: 1.0,
    ActionType.PAUSE: 0.0,  # Defined in rules as +1 for PAUSE_RESUME, handled separately
    ActionType.RESUME: 1.0,
}

# Search / Discovery Weights
SEARCH_WEIGHTS = {
    ActionType.SEARCH_ARTIST: 4.0,
    ActionType.SEARCH_GENRE: 3.0,
    ActionType.SEARCH_SONG: 2.0,
    ActionType.OPEN_ARTIST: 2.0,
    ActionType.VIEW_ALBUM: 2.0,
}

# Playlist Weights
PLAYLIST_WEIGHTS = {
    ActionType.CREATE_PLAYLIST: 2.0,
    ActionType.MOVE_SONG_TOP: 5.0,
    ActionType.MOVE_SONG_BOTTOM: -2.0,
    ActionType.RENAME_PLAYLIST_AROUND_ARTIST: 3.0,
}

# Note: FAVORITE_ARTIST and FAVORITE_GENRE are +15 and +12, 
# but they aren't song-level events.


def calculate_event_score(interaction: UserInteraction) -> float:
    """
    Calculate the UPS score for a single raw event.
    
    This function implements the precedence logic for completions,
    skips, and replays.
    """
    score = 0.0
    action = interaction.action

    # 1. Explicit Feedback
    if action in EXPLICIT_WEIGHTS:
        return EXPLICIT_WEIGHTS[action]

    # 2. Search / Discovery
    if action in SEARCH_WEIGHTS:
        return SEARCH_WEIGHTS[action]

    # 3. Playlist Behavior
    if action in PLAYLIST_WEIGHTS:
        return PLAYLIST_WEIGHTS[action]

    # 4. Listening / Playback
    if action in LISTEN_BASE_WEIGHT:
        score += LISTEN_BASE_WEIGHT[action]
        return score
        
    if action == ActionType.COMPLETE:
        # If it's explicitly marked as COMPLETE
        return 5.0

    if action == ActionType.REPLAY:
        # Replay logic
        count = interaction.replay_count if interaction.replay_count is not None else 1
        if count == 1:
            return 6.0
        elif count > 1:
            return 10.0
        return 0.0

    if action == ActionType.SKIP:
        # Skip logic precedence: strongest applicable rule
        # Strongest negative first: < 10s
        duration = interaction.listen_duration
        percent = interaction.completion_percentage
        
        if duration is not None:
            if duration < 10.0:
                return -8.0
            if duration < 30.0:
                return -6.0
                
        if percent is not None:
            # If we don't have duration, or duration >= 30, check percentages
            if percent > 90.0:
                return 0.0
            if percent > 50.0:
                return -2.0
                
        # Fallback if no contextual data was provided with SKIP
        return -2.0

    # Fallback for unknown actions
    return 0.0


def calculate_completion_score(percentage: float) -> float:
    """Calculate score based purely on completion percentage."""
    if percentage >= 100.0:
        return 5.0
    if percentage >= 75.0:
        return 4.0
    if percentage >= 50.0:
        return 3.0
    if percentage >= 25.0:
        return 2.0
    return 0.0


def calculate_user_song_preference(
    user_id: str, 
    song_id: str, 
    interactions: List[UserInteraction]
) -> UserSongPreference:
    """
    Aggregate a list of raw interactions into a single preference score.
    """
    total_score = 0.0
    last_interaction = None
    
    # Filter for the specific user/song just in case
    filtered = [
        i for i in interactions 
        if i.user_id == user_id and i.song_id == song_id
    ]
    
    for event in filtered:
        # 1. Base event score
        event_score = calculate_event_score(event)
        
        # 2. Add completion percentage score if present, EXCEPT for COMPLETE/SKIP actions
        # which have their own defined logic in calculate_event_score.
        if (event.action not in (ActionType.COMPLETE, ActionType.SKIP) 
            and event.completion_percentage is not None):
            event_score += calculate_completion_score(event.completion_percentage)
            
        total_score += event_score
        
        # Track latest interaction time
        if last_interaction is None or event.timestamp > last_interaction:
            last_interaction = event.timestamp

    return UserSongPreference(
        user_id=user_id,
        song_id=song_id,
        preference_score=total_score,
        interaction_count=len(filtered),
        last_interaction_at=last_interaction
    )


def apply_decay(score: float, days_since_last_interaction: float) -> float:
    """
    Apply time decay to a score.
    
    Formula: score * (0.99 ^ days_since_last_interaction)
    Does not mutate historical events.
    """
    if days_since_last_interaction <= 0:
        return score
        
    decay_factor = math.pow(0.99, days_since_last_interaction)
    return score * decay_factor
