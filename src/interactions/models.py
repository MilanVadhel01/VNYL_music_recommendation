"""
Interaction Models
===================

Data structures for tracking raw user interactions (events) and
aggregated user preferences.

The raw events are immutable historical records.
UserSongPreference is a derived aggregation.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, List
import uuid


class ActionType(str, Enum):
    """Controlled vocabulary for user actions."""
    
    # Listening behavior
    PLAY = "PLAY"
    PAUSE = "PAUSE"
    RESUME = "RESUME"
    SKIP = "SKIP"
    COMPLETE = "COMPLETE"
    REPLAY = "REPLAY"
    
    # Explicit feedback
    LIKE = "LIKE"
    DISLIKE = "DISLIKE"
    REMOVE_LIKE = "REMOVE_LIKE"
    SHARE = "SHARE"
    
    # Playlist behavior
    CREATE_PLAYLIST = "CREATE_PLAYLIST"
    ADD_TO_PLAYLIST = "ADD_TO_PLAYLIST"
    REMOVE_FROM_PLAYLIST = "REMOVE_FROM_PLAYLIST"
    MOVE_SONG_TOP = "MOVE_SONG_TOP"
    MOVE_SONG_BOTTOM = "MOVE_SONG_BOTTOM"
    RENAME_PLAYLIST_AROUND_ARTIST = "RENAME_PLAYLIST_AROUND_ARTIST"
    
    # Search / Discovery behavior
    SEARCH_SONG = "SEARCH_SONG"
    SEARCH_ARTIST = "SEARCH_ARTIST"
    SEARCH_GENRE = "SEARCH_GENRE"
    OPEN_ARTIST = "OPEN_ARTIST"
    VIEW_ALBUM = "VIEW_ALBUM"


@dataclass(frozen=True)
class UserInteraction:
    """A single, immutable user interaction event."""
    
    user_id: str
    song_id: str
    action: ActionType
    
    # Auto-generated ID and timestamp if not provided
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    
    # Contextual data
    completion_percentage: Optional[float] = None  # 0.0 to 100.0
    listen_duration: Optional[float] = None        # in seconds
    replay_count: Optional[int] = None             # >= 0
    device_type: Optional[str] = None
    context: Optional[str] = None
    
    def __post_init__(self):
        """Validate constraints on the event data."""
        if not self.user_id:
            raise ValueError("user_id is required")
        if not self.song_id:
            raise ValueError("song_id is required")
        if not isinstance(self.action, ActionType):
            raise ValueError(f"action must be a valid ActionType, got {self.action}")
            
        if self.completion_percentage is not None:
            if not (0.0 <= self.completion_percentage <= 100.0):
                raise ValueError("completion_percentage must be between 0 and 100")
                
        if self.listen_duration is not None:
            if self.listen_duration < 0:
                raise ValueError("listen_duration cannot be negative")
                
        if self.replay_count is not None:
            if self.replay_count < 0:
                raise ValueError("replay_count cannot be negative")


@dataclass
class UserSongPreference:
    """Aggregated preference representation for a (User, Song) pair."""
    
    user_id: str
    song_id: str
    preference_score: float = 0.0
    interaction_count: int = 0
    last_interaction_at: Optional[datetime] = None


class InteractionStore:
    """
    In-memory storage for raw interaction events.
    
    Designed with a clear interface so it can be replaced with 
    a PostgreSQL-backed implementation in the future.
    """
    
    def __init__(self):
        # List of all raw immutable events
        self._events: List[UserInteraction] = []
        
    def record_interaction(
        self,
        user_id: str,
        song_id: str,
        action: ActionType,
        timestamp: Optional[datetime] = None,
        completion_percentage: Optional[float] = None,
        listen_duration: Optional[float] = None,
        replay_count: Optional[int] = None,
        device_type: Optional[str] = None,
        context: Optional[str] = None
    ) -> UserInteraction:
        """Create, validate, and store a new interaction event."""
        
        event_args = {
            "user_id": user_id,
            "song_id": song_id,
            "action": action,
            "completion_percentage": completion_percentage,
            "listen_duration": listen_duration,
            "replay_count": replay_count,
            "device_type": device_type,
            "context": context
        }
        
        if timestamp is not None:
            event_args["timestamp"] = timestamp
            
        # This will run validation in __post_init__
        event = UserInteraction(**event_args)
        
        self._events.append(event)
        return event

    def get_user_interactions(self, user_id: str) -> List[UserInteraction]:
        """Retrieve all historical interactions for a user."""
        return [e for e in self._events if e.user_id == user_id]
        
    def get_user_song_interactions(
        self, user_id: str, song_id: str
    ) -> List[UserInteraction]:
        """Retrieve all historical interactions for a specific (User, Song) pair."""
        return [
            e for e in self._events 
            if e.user_id == user_id and e.song_id == song_id
        ]
        
    def get_all_events(self) -> List[UserInteraction]:
        """Retrieve all events across all users."""
        return list(self._events)
    
    def clear(self):
        """Clear all events (mainly for testing)."""
        self._events.clear()
