# Project Identity

Project Name:
Music Recommendation System

Project Type:
Personalized Music Recommendation Platform

Primary Goal:
Build a personalized music recommendation engine using a large music dataset, audio/content features, user interactions, and eventually hybrid recommendation techniques.

Current Development Stage:
Content-Based Recommendation Baseline

Current Phase:
Phase 5 -- Initial Content-Based Recommendation (COMPLETED)

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
6. Vector Similarity Search [PLANNED]
7. Content-Based Recommendation [COMPLETED -- baseline]
8. Collaborative Filtering [PLANNED]
9. User Preference Scoring [PLANNED]
10. Artist Preferences [PLANNED]
11. Genre Preferences [PLANNED]
12. Time-Based Preferences [PLANNED]
13. Context-Based Preferences [PLANNED]
14. Score Decay [PLANNED]
15. Hybrid Recommendation Ranking [PLANNED]
16. Live Catalog Updates [PLANNED]
17. New Song Handling [PLANNED]
18. Lyrics Synchronization [PLANNED]
19. Audio Caching/Playback [PLANNED]

---

# Current Development Phase

CURRENT PHASE

Phase 5 -- Initial Content-Based Recommendation
Status: COMPLETED

Objective:
Build a working baseline content-based recommender using cosine similarity
over normalised audio features from the 1M-track dataset.

Features used:
13 audio/content features (danceability, energy, loudness, speechiness,
acousticness, instrumentalness, liveness, valence, tempo, duration,
key, mode, time_signature)

Similarity metric:
Cosine similarity (brute-force, no ANN index)

---

# Phase Roadmap

[x] Phase 1 -- Project Setup
[x] Phase 2 -- Dataset Acquisition
[x] Phase 3 -- Dataset Validation & Cleaning
[x] Phase 4 -- Feature Engineering
[x] Phase 5 -- Initial Content-Based Recommendation
[ ] Phase 6 -- Song Embeddings
[ ] Phase 7 -- FAISS / Vector Search
[ ] Phase 8 -- User Interaction Tracking
[ ] Phase 9 -- UPS Personalization
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

Future vector search:
FAISS or another vector database depending on scale/performance evaluation.

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

### Content-Based Baseline (COMPLETED)

Similarity metric:
Cosine similarity

Feature normalisation:
StandardScaler (sklearn)

Feature weighting:
Equal treatment -- no arbitrary weights in the baseline.

FAISS:
Intentionally deferred to Phase 7.

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
│   │   └── content_scaler.joblib
│   └── embeddings/
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   └── 02_content_based_recommendation.ipynb
│
└── src/
    ├── data/
    │   ├── __init__.py
    │   ├── inspect_dataset.py
    │   ├── sample.py
    │   └── prepare_features.py
    │
    └── models/
        ├── __init__.py
        ├── content_model.py
        └── test_content_model.py

---

# Current Implementation State

| Component                   | Status      | Location                                | Notes |
| --------------------------- | ----------- | --------------------------------------- | ----- |
| Python environment          | COMPLETED   |                                         | Verified in `venv` |
| Requirements                | COMPLETED   | `requirements.txt`                      | Added joblib |
| Hugging Face authentication | COMPLETED   |                                         |       |
| Dataset inspection          | COMPLETED   | `src/data/inspect_dataset.py`           |       |
| 1M extraction               | COMPLETED   | `src/data/sample.py`                    |       |
| Parquet generation          | COMPLETED   | `data/processed/songs_1m.parquet`       |       |
| Dataset validation          | COMPLETED   | `src/data/prepare_features.py`          | 0 missing in features, 465 tempo=0, 487 time_sig=0 |
| EDA notebook                | COMPLETED   | `notebooks/01_dataset_exploration.ipynb` |       |
| Feature selection           | COMPLETED   | `src/data/prepare_features.py`          | 13 features dynamically validated |
| Feature normalisation       | COMPLETED   | `src/data/prepare_features.py`          | StandardScaler |
| Feature matrix              | COMPLETED   | `data/processed/content_features.npy`   | (1000000, 13) float64, 99.2 MB |
| Scaler                      | COMPLETED   | `data/processed/content_scaler.joblib`  | 0.9 KB |
| Song index                  | COMPLETED   | `data/processed/song_index.parquet`     | 50.6 MB |
| Content-based recommender   | COMPLETED   | `src/models/content_model.py`           | Cosine similarity baseline |
| Recommender test script     | COMPLETED   | `src/models/test_content_model.py`      | All validations pass |
| Recommendation notebook     | COMPLETED   | `notebooks/02_content_based_recommendation.ipynb` |       |
| Embeddings                  | NOT STARTED |                                         |       |
| FAISS                       | NOT STARTED |                                         |       |
| UPS                         | NOT STARTED |                                         |       |

---

# Current Working State

CURRENT STATE

Current phase:
Phase 5 -- Initial Content-Based Recommendation (COMPLETED)

Current task:
None -- phase completed.

Last completed action:
Built and tested the baseline content-based recommendation system.

Current blocker:
None.

Next action:
Implement FAISS / approximate nearest-neighbour search for efficient
similarity retrieval over the 1M-song feature matrix.

---

# Session Log

## Session 2026-08-09 -- Session 1

### Objective
Initial project setup, dataset inspection, and extraction of a 1M track sample.

### Completed
- Set up Python environment and requirements.
- Configured `.gitignore` and `README.md`.
- Wrote `inspect_dataset.py` to check the Hugging Face dataset schema.
- Wrote `sample.py` to stream and extract 1M tracks.
- Saved extracted data to `data/processed/songs_1m.parquet`.
- Ran a basic duplicate check showing 93,389 duplicate track/artist pairs.
- Committed and pushed initial project structure to origin.

### Files Changed
- `.gitignore`
- `README.md`
- `data/processed/.gitkeep`
- `notebooks/01_dataset_exploration.ipynb`
- `requirements.txt`
- `src/data/__init__.py`
- `src/data/inspect_dataset.py`
- `src/data/sample.py`

### Important Decisions
- Proceed with `songs_1m.parquet` extraction as baseline.

### Problems Encountered
- Found duplicate track_name + artist_name pairs (93,389).

### Solutions
- Documented duplicates, keeping them in the parquet for now to address in a dedicated cleaning phase.

### Current State
1M dataset extracted and available in Parquet format. Next steps involve validating the schema, evaluating missing values, and executing data cleaning.

### Next Step
Execute deep dataset validation and decide on the cleaning strategy.

## Session 2026-08-09 -- Session 2

### Objective
Implement Steps 9-13: dataset quality check, feature engineering, normalisation, and baseline content-based recommendation system.

### Completed
- Validated all 13 candidate content features exist with zero missing values.
- Reported suspicious values: 465 tempo=0, 487 time_signature=0, 83 duration>3600s.
- Built feature preparation pipeline (`src/data/prepare_features.py`).
- Normalised 13 features using StandardScaler (no arbitrary weighting).
- Saved artifacts: `content_features.npy` (99.2 MB), `song_index.parquet` (50.6 MB), `content_scaler.joblib` (0.9 KB).
- Implemented `ContentRecommender` class with `recommend_by_track_id`, `recommend_by_song_name`, and `search_songs`.
- Created CLI test script with `--track-id`, `--song-name`, and `--artist` flags.
- Tested with multiple songs (Believer, Die With A Smile, Bohemian Rhapsody) -- all validations pass.
- Created `notebooks/02_content_based_recommendation.ipynb`.
- Fixed Windows cp1252 encoding issues across all scripts.

### Files Changed
- `requirements.txt` (added joblib)
- `src/data/prepare_features.py` [NEW]
- `src/data/inspect_dataset.py` (UTF-8 fix)
- `src/data/sample.py` (UTF-8 fix)
- `src/models/__init__.py` [NEW]
- `src/models/content_model.py` [NEW]
- `src/models/test_content_model.py` [NEW]
- `notebooks/02_content_based_recommendation.ipynb` [NEW]
- `context.md` (updated)

### Important Decisions
- Content-based baseline uses standardised numerical audio features.
- Cosine similarity is the initial similarity metric.
- FAISS is intentionally deferred to a later phase.
- Arbitrary feature weighting is not used in the baseline.
- tempo=0 and time_signature=0 are kept (Spotify API "unknown" values).
- Duplicate track_name+artist_name pairs are kept in the baseline.

### Problems Encountered
- Windows cp1252 terminal cannot render Unicode characters (check marks, arrows, bullets).
- Cosine similarity produces exact 1.0000 for duplicate tracks, causing floating-point boundary check to fail.

### Solutions
- Added `sys.stdout.reconfigure(encoding="utf-8")` to all scripts.
- Added epsilon tolerance (1e-6) to similarity score validation bounds.

### Current State
Baseline content-based recommendation system is fully operational. All validation checks pass for multiple test songs.

### Next Step
Implement FAISS / approximate nearest-neighbour search for efficient similarity retrieval.

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

## Decision: Initial 1M Sample

Date:
2026-08-09

Decision:
Extract only 1M rows into a Parquet file for initial EDA and modeling.

Reason:
To enable faster iteration and model development before scaling up to 45M.

## Decision: Do Not Retrain For Every New Song

Decision:
When a new song appears in the live catalog, generate its representation and add it to the vector index instead of retraining the complete recommendation model.

Reason:
Full model retraining for every new song is computationally inefficient.

Status:
PLANNED

## Decision: Content-Based Baseline Uses StandardScaler

Date:
2026-08-09

Decision:
Use StandardScaler (zero mean, unit variance) for feature normalisation in the content-based baseline.

Reason:
Features have very different scales (e.g., tempo 0-250, danceability 0-1, loudness negative dB). Without normalisation, large-scale variables dominate cosine similarity.

Consequence:
The fitted scaler must be saved and reloaded for inference.

## Decision: Cosine Similarity as Baseline Metric

Date:
2026-08-09

Decision:
Use cosine similarity for the initial content-based recommender.

Reason:
Well-understood, widely used for content-based filtering. Provides a correct mathematical baseline before optimising with ANN indices.

## Decision: FAISS Deferred to Phase 7

Date:
2026-08-09

Decision:
Do not implement FAISS or any approximate nearest-neighbour index in the baseline phase.

Reason:
The current goal is a correct baseline. Brute-force cosine similarity over 1M songs completes in ~0.1s per query, which is acceptable for development. FAISS will be needed for production-scale request volumes.

## Decision: No Arbitrary Feature Weighting

Date:
2026-08-09

Decision:
All 13 content features receive equal treatment in the baseline model (no manual weight assignment).

Reason:
There is no empirical evidence yet to justify specific feature weights. Arbitrary weighting could bias the model without improving quality.

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
Last validation date: 2026-08-09

---

# ML Design Notes

### Stage 1
Content-based similarity.
Status: COMPLETED

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

Normalisation: StandardScaler
Similarity: Cosine
Performance: ~0.1s per query (brute-force)

### Stage 2
Song representations / embeddings.

### Stage 3
Vector search.

### Stage 4
User preference modeling.

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

- FAISS
- embeddings
- collaborative filtering
- UPS implementation
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

### Issue: Windows cp1252 Encoding
Description:
Windows PowerShell default encoding (cp1252) cannot render Unicode characters used in print statements.

Impact:
Scripts crash with UnicodeEncodeError when outputting special characters.

Status:
RESOLVED

Solution:
Added `sys.stdout.reconfigure(encoding="utf-8")` to all scripts that produce terminal output.

Date:
2026-08-09

---

# Commands That Currently Work

```bash
python -m venv venv

venv\Scripts\activate

pip install -r requirements.txt

hf auth login

python src/data/inspect_dataset.py

python src/data/sample.py

python src/data/prepare_features.py

python src/models/test_content_model.py

python src/models/test_content_model.py --track-id 0pqnGHJpmpxLKifKRmU6WP

python src/models/test_content_model.py --song-name "Believer" --artist "Imagine Dragons"
```

---

# NEXT ACTION

Current next task:
Implement FAISS / approximate nearest-neighbour search for efficient
similarity retrieval over the 1M-song feature matrix.

Expected files:
FAISS index and related search code.

Expected output:
Sub-millisecond similarity search replacing brute-force cosine similarity.

Do not proceed beyond:
Embeddings, collaborative filtering, UPS, or any downstream system.
