"""
Tests for Learned FAISS Recommender (Phase 6)
=============================================

Validates:
- index construction
- save/load
- query shape handling
- valid song IDs returned
- no duplicate output IDs
- self-match / exclude filtering
- deterministic behavior
"""

import os
import pytest
import numpy as np
import pandas as pd
import faiss

from src.models.learned_recommender import LearnedFaissRecommender

@pytest.fixture
def synthetic_index(tmp_path):
    """Creates a small FAISS index and metadata parquet."""
    dim = 16
    num_songs = 100
    
    # Generate random L2-normalized embeddings
    embs = np.random.randn(num_songs, dim).astype(np.float32)
    faiss.normalize_L2(embs)
    
    index_path = os.path.join(tmp_path, "test.index")
    index = faiss.IndexFlatIP(dim)
    index.add(embs)
    faiss.write_index(index, index_path)
    
    # Metadata
    meta_df = pd.DataFrame({
        "row_index": range(num_songs),
        "track_id": [f"track_{i}" for i in range(num_songs)],
        "track_name": [f"Song {i}" for i in range(num_songs)],
        "artist_name": [f"Artist {i}" for i in range(num_songs)]
    })
    meta_path = os.path.join(tmp_path, "test_meta.parquet")
    meta_df.to_parquet(meta_path)
    
    return index_path, meta_path, embs, meta_df

def test_recommender_initialization(synthetic_index):
    idx_path, meta_path, _, _ = synthetic_index
    rec = LearnedFaissRecommender(idx_path, meta_path)
    assert rec.dim == 16
    assert rec.index.ntotal == 100
    assert len(rec.meta_df) == 100

def test_query_shape_handling(synthetic_index):
    idx_path, meta_path, embs, _ = synthetic_index
    rec = LearnedFaissRecommender(idx_path, meta_path)
    
    # 1D array
    query_1d = embs[0].flatten()
    res_1d = rec.recommend_by_embedding(query_1d, top_k=5)
    
    # 2D array
    query_2d = embs[0].reshape(1, -1)
    res_2d = rec.recommend_by_embedding(query_2d, top_k=5)
    
    assert [r["track_id"] for r in res_1d] == [r["track_id"] for r in res_2d]

def test_valid_ids_and_no_duplicates(synthetic_index):
    idx_path, meta_path, embs, meta_df = synthetic_index
    rec = LearnedFaissRecommender(idx_path, meta_path)
    
    query = embs[50]
    results = rec.recommend_by_embedding(query, top_k=10)
    
    assert len(results) == 10
    
    ids = [r["track_id"] for r in results]
    # No duplicates
    assert len(ids) == len(set(ids))
    
    # Valid IDs
    valid_ids = set(meta_df["track_id"])
    for tid in ids:
        assert tid in valid_ids

def test_exclude_filtering(synthetic_index):
    idx_path, meta_path, embs, _ = synthetic_index
    rec = LearnedFaissRecommender(idx_path, meta_path)
    
    query = embs[10]
    # Without exclude, track_10 should be top because it's exact match
    res1 = rec.recommend_by_embedding(query, top_k=5)
    assert res1[0]["track_id"] == "track_10"
    
    # With exclude
    res2 = rec.recommend_by_embedding(query, top_k=5, exclude_song_ids=["track_10", "track_99"])
    ids2 = [r["track_id"] for r in res2]
    
    assert "track_10" not in ids2
    assert "track_99" not in ids2
    assert len(res2) == 5

def test_deterministic_behavior(synthetic_index):
    idx_path, meta_path, embs, _ = synthetic_index
    rec = LearnedFaissRecommender(idx_path, meta_path)
    
    query = embs[25]
    res1 = rec.recommend_by_embedding(query, top_k=10)
    res2 = rec.recommend_by_embedding(query, top_k=10)
    
    assert [r["track_id"] for r in res1] == [r["track_id"] for r in res2]
    for r1, r2 in zip(res1, res2):
        assert np.isclose(r1["score"], r2["score"])
