# Project Identity

Project Name:
Music Recommendation System

Project Type:
Personalized Music Recommendation Platform

Primary Goal:
Build a personalized music recommendation engine using a large music dataset, audio/content features, user interactions, and eventually hybrid recommendation techniques.

Current Development Stage:
Fast Vector Similarity Search & UPS Foundation

Current Phase:
Phase 8 -- User Interaction Tracking & Phase 9 -- UPS Personalization (Initial Foundation)

Music Search
      |
Music Catalog
      |
Song Metadata
      |
Song Representation / Embeddings
      |
Recommendation Engine
      |
User Interaction Tracking
      |
Personalization
      |
Hybrid Recommendation

---

# Long-Term Project Vision

1. Music Search [PLANNED]
2. Music Catalog [PLANNED]
3. External Music Metadata Retrieval [PLANNED]
4. Audio/Content Feature Processing [COMPLETED]
5. Song Embeddings [PLANNED]
6. Vector Similarity Search [COMPLETED]
7. Content-Based Recommendation [COMPLETED -- FAISS]
8. Collaborative Filtering [PLANNED]
9. User Preference Scoring [COMPLETED -- Foundation]
10. Artist Preferences [PLANNED]
11. Genre Preferences [PLANNED]
12. Time-Based Preferences [PLANNED]
13. Context-Based Preferences [PLANNED]
14. Score Decay [COMPLETED -- Foundation]
15. Hybrid Recommendation Ranking [PLANNED]
16. Live Catalog Updates [PLANNED]
17. New Song Handling [PLANNED]
18. Lyrics Synchronization [PLANNED]
19. Audio Caching/Playback [PLANNED]

---

# Current Development Phase

CURRENT PHASE

Phase 7 -- FAISS / Vector Search
Status: COMPLETED

Phase 8 -- User Interaction Tracking
Status: COMPLETED

Phase 9 -- UPS Personalization
Status: COMPLETED (Foundation)

Objective:
Replace brute-force search with FAISS for fast similarity retrieval.
Build a reliable event logging layer for user interactions.
Implement the first version of the User Preference Score (UPS).

Features used:
FAISS IndexFlatIP (exact cosine similarity via L2-normalization + inner product).
Raw event history separate from aggregated user preferences.
Heuristic initial weights for UPS, not learned parameters.

---

# Phase Roadmap

[x] Phase 1 -- Project Setup
[x] Phase 2 -- Dataset Acquisition
[x] Phase 3 -- Dataset Validation & Cleaning
[x] Phase 4 -- Feature Engineering
[x] Phase 5 -- Initial Content-Based Recommendation
[ ] Phase 6 -- Song Embeddings
[x] Phase 7 -- FAISS / Vector Search
[x] Phase 8 -- User Interaction Tracking
[x] Phase 9 -- UPS Personalization
[ ] Phase 10 -- Artist & Genre Preferences
[ ] Phase 11 -- Time / Context Signals
[ ] Phase 12 -- Collaborative Filtering
[ ] Phase 13 -- Hybrid Recommendation
[ ] Phase 14 -- Live Song Catalog
[ ] Phase 15 -- New Song Embedding Pipeline
[ ] Phase 16 -- Recommendation API
[ ] Phase 17 -- Lyrics Integration
[ ] Phase 18 -- Audio / Playback Integration
[ ] Phase 19 -- Evaluation
[ ] Phase 20 -- Deployment

---

# Architecture Decisions

ARCHITECTURE DECISIONS

### Dataset
Source:
GD-Studio/embeat_45m_spotify_tracks

Initial working dataset:
1M tracks

Do not download all 45M into memory.

Use streaming where appropriate.

### Storage
Initial dataset:
Parquet

Future application database:
PostgreSQL

Vector search:
FAISS (IndexFlatIP used for the initial exact similarity baseline).

### Recommendation Architecture
Content Similarity
        +
Collaborative Filtering
        +
User Preference Score (UPS)
        +
Artist Preference
        +
Genre Preference
        +
Context
        +
Freshness / Popularity
        |
Final Ranking

### Content-Based FAISS Recommender (COMPLETED)

Similarity metric:
Cosine similarity (implemented through L2-normalized vectors + inner product in FAISS IndexFlatIP).

Feature normalisation:
StandardScaler (sklearn) followed by L2-normalization.

FAISS IndexFlatIP is used for the initial exact similarity baseline.

### User Interaction & UPS Foundation (COMPLETED)

Raw user interaction events are kept separate from aggregated user preferences.

UPS weights are heuristic initial values and are not learned parameters.

Historical interaction scores remain immutable.

Decay is applied as a derived transformation, not by modifying historical events.

---

# Dataset Context

Dataset:
GD-Studio/embeat_45m_spotify_tracks

Source:
Hugging Face

Approximate source size:
45M tracks

Initial selected subset:
1M tracks

Format:
Parquet after processing

Confirmed fields (27 columns):
track_id (str)
track_name (str)
isrc (str)
popularity (float64)
explicit (int64)

artist_idx (int64)
artist_id (str)
artist_name (str)
artist_popularity (float64)
artist_genres (str)
artist_genre_idx (int64)

album_id (str)
album_name (str)
release_year (int64)

duration (int64)
time_signature (int64)
tempo (int64)
key (int64)
mode (int64)

danceability (float64)
energy (float64)
loudness (float64)
speechiness (float64)
acousticness (float64)
instrumentalness (float64)
liveness (float64)
valence (float64)

---

# Current Repository Structure

/
├── context.md
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   ├── processed/
│   │   ├── .gitkeep
│   │   ├── songs_1m.parquet
│   │   ├── content_features.npy
│   │   ├── song_index.parquet
│   │   ├── content_scaler.joblib
│   │   └── faiss/
│   │       └── content.index
│   └── embeddings/
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   └── 02_content_based_recommendation.ipynb
│
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── inspect_dataset.py
│   │   ├── sample.py
│   │   └── prepare_features.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── content_model.py
│   │   ├── test_content_model.py
│   │   ├── build_faiss_index.py
│   │   ├── faiss_recommender.py
│   │   ├── compare_faiss.py
│   │   └── test_faiss.py
│   │
│   └── interactions/
│       ├── __init__.py
│       ├── models.py
│       ├── ups.py
│       └── test_ups_demo.py
│
└── tests/
    └── test_ups.py

---

# Current Implementation State

| Component                   | Status      | Location                                | Notes |
| --------------------------- | ----------- | --------------------------------------- | ----- |
| Python environment          | COMPLETED   |                                         | Verified in `venv` |
| Requirements                | COMPLETED   | `requirements.txt`                      | Added faiss-cpu |
| Dataset validation          | COMPLETED   | `src/data/prepare_features.py`          | 0 missing in features |
| Feature normalisation       | COMPLETED   | `src/data/prepare_features.py`          | StandardScaler |
| Feature matrix              | COMPLETED   | `data/processed/content_features.npy`   | (1000000, 13) float64 |
| FAISS Index Builder         | COMPLETED   | `src/models/build_faiss_index.py`       | L2-normalized IndexFlatIP |
| FAISS Index Artifact        | COMPLETED   | `data/processed/faiss/content.index`    | 49.6 MB |
| FAISS Recommender           | COMPLETED   | `src/models/faiss_recommender.py`       | |
| Interaction Tracking Models | COMPLETED   | `src/interactions/models.py`            | Raw immutable events |
| UPS Scoring                 | COMPLETED   | `src/interactions/ups.py`               | Event heuristics & decay |

---

# Current Working State

CURRENT STATE

Current phase:
Phases 7-9: FAISS Search, User Interaction Tracking, UPS Foundation (COMPLETED)

Current task:
None -- phase completed.

Last completed action:
Built FAISS index, faiss_recommender, interaction models, and UPS scoring.

Current blocker:
None.

Next action:
Implement artist and genre preference aggregation and begin the personalized candidate-ranking layer.

---

# Session Log

## Session 2026-08-09 -- Session 3

### Objective
Implement Steps 14-16: FAISS / Fast Similarity Search, User Interaction Tracking, and User Preference Score (UPS) Foundation.

### Completed
- Added `faiss-cpu` and `pytest` to requirements.
- Built FAISS `IndexFlatIP` from L2-normalized standardized feature vectors.
- Created `FAISSContentRecommender` to replace brute-force baseline.
- Validated FAISS output against brute-force (10/10 overlap).
- Measured FAISS latency (FAISS ~12ms vs Brute-force ~121ms, speedup ~9.7x).
- Designed interaction tracking schema with `ActionType` enum and `UserInteraction` dataclass (raw immutable events).
- Created `InteractionStore` as an in-memory database prep for PostgreSQL.
- Implemented UPS heuristic scoring rules (`calculate_event_score`, `calculate_user_song_preference`).
- Addressed skip precedence and completion tier logic correctly (no double counting).
- Added `apply_decay` function for time-based score decay without modifying raw events.
- Created `test_ups.py` and `test_ups_demo.py` and passed all tests.

### Files Changed
- `requirements.txt` (added `faiss-cpu`)
- `src/models/build_faiss_index.py` [NEW]
- `src/models/faiss_recommender.py` [NEW]
- `src/models/compare_faiss.py` [NEW]
- `src/models/test_faiss.py` [NEW]
- `src/interactions/__init__.py` [NEW]
- `src/interactions/models.py` [NEW]
- `src/interactions/ups.py` [NEW]
- `src/interactions/test_ups_demo.py` [NEW]
- `tests/test_ups.py` [NEW]
- `context.md` (updated)

### Tests Performed
- `test_faiss.py`: Confirmed correct Top-N output, query exclusion, duplicates, sorting, score range.
- `compare_faiss.py`: Tested exactness of FAISS vs brute-force (100% overlap).
- `pytest tests/test_ups.py`: Passed 9/9 tests verifying logic for explicit weights, replay logic, skip precedence, aggregations, filters, and decay.

### Benchmark Results
- Average brute-force latency: 121.73 ms
- Average FAISS latency: 12.59 ms
- Speed improvement: 9.7x
- FAISS indexing time: 0.14s (including L2 normalization)

### Problems Encountered
- Missing `pytest` module.

### Solutions
- Installed `pytest` and successfully ran the tests.

### Current State
FAISS indexing, querying, and benchmark comparison are operational. The event schema and UPS heuristic foundation are tested and working.

### Next Action
Implement artist and genre preference aggregation and begin the personalized candidate-ranking layer.

---

# Decision Log

## Decision: Streaming Dataset Processing

Date:
2026-08-09

Decision:
Use Hugging Face streaming instead of loading the complete 45M dataset into memory.

Reason:
The source dataset is very large and full in-memory loading is unnecessary.

Consequence:
Data extraction must be incremental/chunked.

## Decision: FAISS IndexFlatIP with L2 Normalization

Date:
2026-08-09

Decision:
Use FAISS IndexFlatIP (inner product) for exact similarity search, coupled with L2 normalization of vectors before insertion and query.

Reason:
The baseline recommender uses cosine similarity. FAISS inner product on L2-normalized vectors is mathematically equivalent to cosine similarity. IndexFlatIP provides exact nearest neighbors, avoiding quantization loss for the baseline comparison.

## Decision: Separation of Events and Preferences

Date:
2026-08-09

Decision:
Raw interaction events are logged and remain immutable. UserSongPreference is an aggregation calculated from these events.

Reason:
Preserves historical interaction data. Prevents "double counting" errors when re-evaluating scores in the future. Allows decay functions to operate purely on the aggregated output without tampering with the original event records.

## Decision: UPS Heuristics Are Not Learned

Date:
2026-08-09

Decision:
UPS calculation relies on static heuristic weights based on project specifications. 

Reason:
There is currently no objective ground truth to train these weights on. These serve as a structural foundation for the personalization ranking layer.

---

# Technical Constraints

1. Do not load the complete 45M dataset into RAM.

2. Do not commit datasets to Git.

3. Never hardcode API keys or Hugging Face tokens.

4. Do not build recommendation models before the dataset pipeline is verified.

5. Do not assume dataset columns without checking the actual schema.

6. Keep data-processing code separate from recommendation-model code.

7. Prefer reproducible processing.

8. Do not silently delete data during cleaning.

9. Record important data-quality statistics before cleaning.

10. Do not introduce heavy infrastructure unless the current phase requires it.

11. Do not overwrite songs_1m.parquet during feature engineering -- derived artifacts are stored separately.

---

# Data Quality Tracking

DATA QUALITY

Source rows: 45,000,000 (Approx)
Selected rows: 1,000,000
Valid rows: 1,000,000 (all rows usable for content features)
Duplicate track IDs: 0
Duplicate track+artist pairs: 93,389
Missing track names: 0
Missing artist names: 0
Missing audio features: 0 (all 13 features have zero nulls)
Invalid numerical values:
  tempo == 0: 465 (Spotify API "unknown")
  time_signature == 0: 487 (Spotify API "unknown")
  duration > 3600s: 83 (very long tracks)
  Bounded features outside [0,1]: 0
Parquet size: ~165 MB
Feature matrix size: 99.2 MB (1000000 x 13 float64)
Song index size: 50.6 MB
FAISS Index size: 49.6 MB
Last validation date: 2026-08-09

---

# ML Design Notes

### Stage 1
Content-based similarity.
Status: COMPLETED (using FAISS)

Features used (13):
danceability
energy
loudness
speechiness
acousticness
instrumentalness
liveness
valence
tempo
duration
key
mode
time_signature

Normalisation: StandardScaler + L2-Normalization
Similarity: Cosine (via FAISS Inner Product)
Performance: ~12.5ms per query (FAISS IndexFlatIP)

### Stage 2
Song representations / embeddings.

### Stage 3
Vector search.

### Stage 4
User preference modeling.
Status: COMPLETED (Foundation)

### Stage 5
Collaborative filtering.

### Stage 6
Hybrid ranking.

---

# UPS DESIGN

Explicit feedback:
Like                 +10
Dislike              -10
Add to Playlist       +8
Favorite Artist      +15
Favorite Genre       +12
Remove Like           -8
Remove Playlist       -5
Share                 +7

Implicit feedback:
Play                  +1
Listen >25%           +2
Listen >50%           +3
Listen >75%           +4
Listen 100%           +5
Replay Once            +6
Replay Multiple       +10
Skip <10 sec           -8
Skip <30 sec           -6
Skip after 50%        -2
Skip after 90%          0
Pause & Resume         +1

Search:
Search Artist          +4
Search Genre           +3
Search Song            +2
Open Artist             +2
View Album              +2

Playlist:
Create Playlist        +2
Add Song                +8
Remove Song             -5
Move Song Top           +5
Move Song Bottom        -2

---

# Planned Live Song Flow

User searches song
        |
Check local catalog
        |
Song exists?
   +----+----+
   YES       NO
    |         |
    |    External API
    |         |
    |    Fetch metadata
    |         |
    |    Create song record
    |         |
    |    Generate representation
    |         |
    +----+----+
         |
     Song available
         |
      Playback
         |
 User interaction logging

PLANNED -- NOT IMPLEMENTED

---

# Things NOT To Do Yet

DO NOT IMPLEMENT YET

- embeddings
- collaborative filtering
- full database for UPS
- user interaction database
- hybrid ranking
- external music APIs
- Discogs integration
- ReccoBeats integration
- Telegram audio storage
- lyrics API
- frontend recommendation UI
- production deployment

---

# Known Problems / Blockers

KNOWN ISSUES

### Issue: Duplicate Track+Artist Pairs
Description:
Duplicate track+artist pairs present in the 1M sample (93,389 pairs).

Impact:
Data redundancy -- same-name songs appear in recommendation results because they have different track_ids (different album releases, remasters, etc.).

Status:
OPEN

Solution:
Need to analyze differences in duplicates (e.g., different album releases, remasters) and implement a deduplication strategy during a future cleaning phase.

Date:
2026-08-09

---

# NEXT ACTION

Current next task:
Implement artist and genre preference aggregation and begin the personalized candidate-ranking layer.

Expected output:
Logic to calculate user preferences towards specific artists and genres based on raw interaction events, feeding into a candidate ranker.

Do not proceed beyond:
Collaborative filtering or any downstream system.
