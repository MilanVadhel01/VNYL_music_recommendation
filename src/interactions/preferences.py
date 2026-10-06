"""
Artist & Genre Preference Aggregation
======================================

Derives user-level artist and genre preference representations from
the existing immutable interaction events and UPS scoring logic.

Preferences are derived data — raw interactions remain the source of truth.

Preference scores are normalized to [-1.0, +1.0] using tanh scaling.

Architecture:
    Raw Interaction Events
        ↓
    UPS calculate_event_score (reused)
        ↓
    Song → Artist mapping
    Song → Genre mapping
        ↓
    Aggregated artist/genre preference scores
        ↓
    tanh normalization to [-1, +1]
        ↓
    apply_decay (reused, optional)
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
import math

from src.interactions.models import ActionType, UserInteraction, InteractionStore
from src.interactions.ups import calculate_event_score, apply_decay


# ---------------------------------------------------------------------------
# Preference Data Structures
# ---------------------------------------------------------------------------

@dataclass
class UserArtistPreference:
    """Aggregated preference for a (user, artist) pair.
    
    Derived from raw interaction events — not a primary record.
    
    Attributes:
        preference_score: Normalized score in [-1.0, +1.0] via tanh scaling.
        raw_score: Unnormalized sum of UPS event scores.
    """
    user_id: str
    artist_id: str
    artist_name: str
    preference_score: float = 0.0       # normalized to [-1, +1]
    raw_score: float = 0.0              # unnormalized sum
    interaction_count: int = 0
    positive_interactions: int = 0
    negative_interactions: int = 0
    last_interaction_timestamp: Optional[datetime] = None


@dataclass
class UserGenrePreference:
    """Aggregated preference for a (user, genre) pair.
    
    Derived from raw interaction events — not a primary record.
    
    Attributes:
        preference_score: Normalized score in [-1.0, +1.0] via tanh scaling.
        raw_score: Unnormalized sum of UPS event scores.
    """
    user_id: str
    genre: str
    preference_score: float = 0.0       # normalized to [-1, +1]
    raw_score: float = 0.0              # unnormalized sum
    interaction_count: int = 0
    positive_interactions: int = 0
    negative_interactions: int = 0
    last_interaction_timestamp: Optional[datetime] = None


# ---------------------------------------------------------------------------
# Song Metadata Lookup
# ---------------------------------------------------------------------------

class SongMetadataLookup:
    """Maps song_id → (artist_id, artist_name, genres).
    
    This is the bridge between interaction events (which reference song_id)
    and artist/genre aggregation. Must be populated from the dataset before
    preference computation.
    """

    def __init__(self):
        # song_id -> (artist_id, artist_name)
        self._song_to_artist: Dict[str, Tuple[str, str]] = {}
        # song_id -> list of genre strings
        self._song_to_genres: Dict[str, List[str]] = {}

    def register_song(
        self,
        song_id: str,
        artist_id: str,
        artist_name: str,
        genres: Optional[List[str]] = None,
    ) -> None:
        """Register a song's artist and genre metadata."""
        self._song_to_artist[song_id] = (artist_id, artist_name)
        self._song_to_genres[song_id] = genres if genres else []

    def get_artist(self, song_id: str) -> Optional[Tuple[str, str]]:
        """Return (artist_id, artist_name) for a song, or None."""
        return self._song_to_artist.get(song_id)

    def get_genres(self, song_id: str) -> List[str]:
        """Return list of genre strings for a song. Empty list if missing."""
        return self._song_to_genres.get(song_id, [])

    def load_from_dataframe(self, df) -> None:
        """Bulk-load song metadata from a pandas DataFrame.
        
        Expected columns: track_id, artist_id, artist_name, artist_genres.
        artist_genres is expected as a string representation of a list
        (e.g. "['rock', 'pop']") or empty/NaN for missing genres.
        """
        import ast

        for _, row in df.iterrows():
            song_id = str(row.get("track_id", ""))
            artist_id = str(row.get("artist_id", ""))
            artist_name = str(row.get("artist_name", ""))

            # Parse genres safely
            raw_genres = row.get("artist_genres", "")
            genres = _parse_genres(raw_genres)

            self.register_song(song_id, artist_id, artist_name, genres)


def _parse_genres(raw_genres) -> List[str]:
    """Safely parse genre data from various formats.
    
    Handles:
        - Python list literal strings: "['rock', 'pop']"
        - Actual Python lists
        - Comma-separated strings
        - NaN / None / empty
    """
    import ast

    if raw_genres is None:
        return []

    # Handle NaN (float)
    if isinstance(raw_genres, float) and math.isnan(raw_genres):
        return []

    if isinstance(raw_genres, list):
        return [g.strip().lower() for g in raw_genres if isinstance(g, str) and g.strip()]

    if isinstance(raw_genres, str):
        s = raw_genres.strip()
        if not s or s.lower() == "nan":
            return []

        # Try parsing as Python list literal
        if s.startswith("["):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, list):
                    return [g.strip().lower() for g in parsed if isinstance(g, str) and g.strip()]
            except (ValueError, SyntaxError):
                pass

        # Fallback: comma-separated
        return [g.strip().lower() for g in s.split(",") if g.strip()]

    return []


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

# Normalization scale factor for tanh.
# tanh(raw_score / NORM_SCALE) maps raw scores to [-1, +1].
# With NORM_SCALE=30, a raw score of ~30 maps to ~0.76, ~60 to ~0.96.
# This is tuned to the UPS weight magnitudes (max single event = ±15).
NORM_SCALE: float = 30.0


def normalize_preference(raw_score: float, scale: float = NORM_SCALE) -> float:
    """Normalize a raw preference score to [-1.0, +1.0] using tanh.
    
    Properties:
        - Monotonically increasing
        - Symmetric around 0
        - Bounded: output ∈ [-1, +1]
        - Smooth / differentiable
        - Deterministic
    """
    return math.tanh(raw_score / scale)


# ---------------------------------------------------------------------------
# Preference Store
# ---------------------------------------------------------------------------

class PreferenceStore:
    """Computes and caches derived artist and genre preferences.
    
    All preferences are derived from raw interaction events using the
    existing UPS calculate_event_score() function. No contradictory
    hardcoded weights are introduced.
    
    The store requires a SongMetadataLookup to map songs to artists/genres.
    """

    def __init__(
        self,
        interaction_store: InteractionStore,
        metadata_lookup: SongMetadataLookup,
        norm_scale: float = NORM_SCALE,
    ):
        self._interaction_store = interaction_store
        self._metadata = metadata_lookup
        self._norm_scale = norm_scale

        # Caches: keyed by (user_id, artist_id) or (user_id, genre)
        self._artist_prefs: Dict[Tuple[str, str], UserArtistPreference] = {}
        self._genre_prefs: Dict[Tuple[str, str], UserGenrePreference] = {}

    # ------------------------------------------------------------------
    # Public API — Artist Preferences
    # ------------------------------------------------------------------

    def get_artist_preference(
        self, user_id: str, artist_id: str
    ) -> Optional[UserArtistPreference]:
        """Get the preference for a specific (user, artist) pair."""
        key = (user_id, artist_id)
        return self._artist_prefs.get(key)

    def get_top_artists(
        self, user_id: str, top_k: int = 10
    ) -> List[UserArtistPreference]:
        """Return the top-K artists by preference score for a user."""
        user_prefs = [
            p for (uid, _), p in self._artist_prefs.items()
            if uid == user_id
        ]
        user_prefs.sort(key=lambda p: p.preference_score, reverse=True)
        return user_prefs[:top_k]

    # ------------------------------------------------------------------
    # Public API — Genre Preferences
    # ------------------------------------------------------------------

    def get_genre_preference(
        self, user_id: str, genre: str
    ) -> Optional[UserGenrePreference]:
        """Get the preference for a specific (user, genre) pair."""
        key = (user_id, genre.lower())
        return self._genre_prefs.get(key)

    def get_top_genres(
        self, user_id: str, top_k: int = 10
    ) -> List[UserGenrePreference]:
        """Return the top-K genres by preference score for a user."""
        user_prefs = [
            p for (uid, _), p in self._genre_prefs.items()
            if uid == user_id
        ]
        user_prefs.sort(key=lambda p: p.preference_score, reverse=True)
        return user_prefs[:top_k]

    # ------------------------------------------------------------------
    # Rebuild — Artist Preferences
    # ------------------------------------------------------------------

    def rebuild_artist_preferences(
        self, user_id: Optional[str] = None
    ) -> None:
        """Rebuild artist preferences from raw interaction events.
        
        If user_id is provided, only that user's preferences are rebuilt.
        If user_id is None, all users' preferences are rebuilt.
        
        This is deterministic and reproducible for the same input events.
        """
        if user_id is not None:
            # Clear existing prefs for this user
            keys_to_remove = [
                k for k in self._artist_prefs if k[0] == user_id
            ]
            for k in keys_to_remove:
                del self._artist_prefs[k]

            events = self._interaction_store.get_user_interactions(user_id)
            self._build_artist_prefs_from_events(user_id, events)
        else:
            # Rebuild all
            self._artist_prefs.clear()
            all_events = self._interaction_store.get_all_events()
            events_by_user: Dict[str, List[UserInteraction]] = {}
            for e in all_events:
                events_by_user.setdefault(e.user_id, []).append(e)

            for uid, events in events_by_user.items():
                self._build_artist_prefs_from_events(uid, events)

    def _build_artist_prefs_from_events(
        self, user_id: str, events: List[UserInteraction]
    ) -> None:
        """Aggregate events into per-artist preferences for one user."""
        # Accumulator: artist_id -> {artist_name, raw_score, counts, timestamps}
        acc: Dict[str, dict] = {}

        for event in events:
            artist_info = self._metadata.get_artist(event.song_id)
            if artist_info is None:
                continue  # Unknown song — skip safely

            artist_id, artist_name = artist_info
            score = calculate_event_score(event)

            if artist_id not in acc:
                acc[artist_id] = {
                    "artist_name": artist_name,
                    "raw_score": 0.0,
                    "interaction_count": 0,
                    "positive_interactions": 0,
                    "negative_interactions": 0,
                    "last_timestamp": None,
                }

            entry = acc[artist_id]
            entry["raw_score"] += score
            entry["interaction_count"] += 1

            if score > 0:
                entry["positive_interactions"] += 1
            elif score < 0:
                entry["negative_interactions"] += 1

            if entry["last_timestamp"] is None or event.timestamp > entry["last_timestamp"]:
                entry["last_timestamp"] = event.timestamp

        # Convert to UserArtistPreference objects
        for artist_id, data in acc.items():
            pref = UserArtistPreference(
                user_id=user_id,
                artist_id=artist_id,
                artist_name=data["artist_name"],
                preference_score=normalize_preference(data["raw_score"], self._norm_scale),
                raw_score=data["raw_score"],
                interaction_count=data["interaction_count"],
                positive_interactions=data["positive_interactions"],
                negative_interactions=data["negative_interactions"],
                last_interaction_timestamp=data["last_timestamp"],
            )
            self._artist_prefs[(user_id, artist_id)] = pref

    # ------------------------------------------------------------------
    # Rebuild — Genre Preferences
    # ------------------------------------------------------------------

    def rebuild_genre_preferences(
        self, user_id: Optional[str] = None
    ) -> None:
        """Rebuild genre preferences from raw interaction events.
        
        If user_id is provided, only that user's preferences are rebuilt.
        If user_id is None, all users' preferences are rebuilt.
        
        Each song interaction contributes to every valid genre associated
        with that song. Missing genre metadata is handled safely (skipped).
        """
        if user_id is not None:
            keys_to_remove = [
                k for k in self._genre_prefs if k[0] == user_id
            ]
            for k in keys_to_remove:
                del self._genre_prefs[k]

            events = self._interaction_store.get_user_interactions(user_id)
            self._build_genre_prefs_from_events(user_id, events)
        else:
            self._genre_prefs.clear()
            all_events = self._interaction_store.get_all_events()
            events_by_user: Dict[str, List[UserInteraction]] = {}
            for e in all_events:
                events_by_user.setdefault(e.user_id, []).append(e)

            for uid, events in events_by_user.items():
                self._build_genre_prefs_from_events(uid, events)

    def _build_genre_prefs_from_events(
        self, user_id: str, events: List[UserInteraction]
    ) -> None:
        """Aggregate events into per-genre preferences for one user."""
        acc: Dict[str, dict] = {}

        for event in events:
            genres = self._metadata.get_genres(event.song_id)
            if not genres:
                continue  # No genre metadata — skip safely

            score = calculate_event_score(event)

            for genre in genres:
                genre_key = genre.lower()
                if genre_key not in acc:
                    acc[genre_key] = {
                        "raw_score": 0.0,
                        "interaction_count": 0,
                        "positive_interactions": 0,
                        "negative_interactions": 0,
                        "last_timestamp": None,
                    }

                entry = acc[genre_key]
                entry["raw_score"] += score
                entry["interaction_count"] += 1

                if score > 0:
                    entry["positive_interactions"] += 1
                elif score < 0:
                    entry["negative_interactions"] += 1

                if entry["last_timestamp"] is None or event.timestamp > entry["last_timestamp"]:
                    entry["last_timestamp"] = event.timestamp

        for genre_key, data in acc.items():
            pref = UserGenrePreference(
                user_id=user_id,
                genre=genre_key,
                preference_score=normalize_preference(data["raw_score"], self._norm_scale),
                raw_score=data["raw_score"],
                interaction_count=data["interaction_count"],
                positive_interactions=data["positive_interactions"],
                negative_interactions=data["negative_interactions"],
                last_interaction_timestamp=data["last_timestamp"],
            )
            self._genre_prefs[(user_id, genre_key)] = pref

    # ------------------------------------------------------------------
    # Convenience: rebuild all
    # ------------------------------------------------------------------

    def rebuild_all(self, user_id: Optional[str] = None) -> None:
        """Rebuild both artist and genre preferences."""
        self.rebuild_artist_preferences(user_id)
        self.rebuild_genre_preferences(user_id)

    # ------------------------------------------------------------------
    # Decay support
    # ------------------------------------------------------------------

    def get_artist_preference_with_decay(
        self,
        user_id: str,
        artist_id: str,
        reference_time: Optional[datetime] = None,
    ) -> Optional[UserArtistPreference]:
        """Get artist preference with time decay applied.
        
        Uses the existing apply_decay() function from ups.py.
        Returns a new object — does not mutate the cached preference.
        """
        pref = self.get_artist_preference(user_id, artist_id)
        if pref is None:
            return None

        if reference_time is None:
            reference_time = datetime.now(timezone.utc)

        if pref.last_interaction_timestamp is None:
            return pref

        days = (reference_time - pref.last_interaction_timestamp).total_seconds() / 86400.0
        decayed_raw = apply_decay(pref.raw_score, days)

        return UserArtistPreference(
            user_id=pref.user_id,
            artist_id=pref.artist_id,
            artist_name=pref.artist_name,
            preference_score=normalize_preference(decayed_raw, self._norm_scale),
            raw_score=decayed_raw,
            interaction_count=pref.interaction_count,
            positive_interactions=pref.positive_interactions,
            negative_interactions=pref.negative_interactions,
            last_interaction_timestamp=pref.last_interaction_timestamp,
        )

    def get_genre_preference_with_decay(
        self,
        user_id: str,
        genre: str,
        reference_time: Optional[datetime] = None,
    ) -> Optional[UserGenrePreference]:
        """Get genre preference with time decay applied.
        
        Uses the existing apply_decay() function from ups.py.
        Returns a new object — does not mutate the cached preference.
        """
        pref = self.get_genre_preference(user_id, genre)
        if pref is None:
            return None

        if reference_time is None:
            reference_time = datetime.now(timezone.utc)

        if pref.last_interaction_timestamp is None:
            return pref

        days = (reference_time - pref.last_interaction_timestamp).total_seconds() / 86400.0
        decayed_raw = apply_decay(pref.raw_score, days)

        return UserGenrePreference(
            user_id=pref.user_id,
            genre=pref.genre,
            preference_score=normalize_preference(decayed_raw, self._norm_scale),
            raw_score=decayed_raw,
            interaction_count=pref.interaction_count,
            positive_interactions=pref.positive_interactions,
            negative_interactions=pref.negative_interactions,
            last_interaction_timestamp=pref.last_interaction_timestamp,
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def get_all_artist_preferences(
        self, user_id: Optional[str] = None
    ) -> List[UserArtistPreference]:
        """Return all cached artist preferences, optionally filtered by user."""
        prefs = list(self._artist_prefs.values())
        if user_id is not None:
            prefs = [p for p in prefs if p.user_id == user_id]
        return prefs

    def get_all_genre_preferences(
        self, user_id: Optional[str] = None
    ) -> List[UserGenrePreference]:
        """Return all cached genre preferences, optionally filtered by user."""
        prefs = list(self._genre_prefs.values())
        if user_id is not None:
            prefs = [p for p in prefs if p.user_id == user_id]
        return prefs
