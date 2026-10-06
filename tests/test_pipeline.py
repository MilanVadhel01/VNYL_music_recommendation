"""
Phase 10 Tests: Unified Recommendation Pipeline
===============================================
"""

import os
import pytest
import numpy as np
import pandas as pd
import faiss
from datetime import datetime, timezone

from src.interactions.models import InteractionStore, UserInteraction, ActionType
from src.interactions.preferences import SongMetadataLookup
from src.models.two_tower import UserEncoder
from src.models.learned_recommender import LearnedFaissRecommender
from src.models.neural_ranker import NeuralRanker
from src.models.pipeline import VNYLRecommenderPipeline

@pytest.fixture
def mock_pipeline_components(tmp_path):
    # Data
    num_songs = 10
    dim = 64
    song_embeddings = np.random.randn(num_songs, dim).astype(np.float32)
    faiss.normalize_L2(song_embeddings)
    
    song_id_to_row = {f"track_{i}": i for i in range(num_songs)}
    
    metadata_lookup = SongMetadataLookup()
    for i in range(num_songs):
        metadata_lookup.register_song(f"track_{i}", f"artist_{i%3}", f"Artist {i%3}", [f"genre_{i%2}"])
        
    df = pd.DataFrame({
        "row_index": range(num_songs),
        "track_id": [f"track_{i}" for i in range(num_songs)],
        "track_name": [f"Song {i}" for i in range(num_songs)],
        "artist_name": [f"Artist {i%3}" for i in range(num_songs)]
    })
    meta_path = os.path.join(tmp_path, "meta.parquet")
    df.to_parquet(meta_path)
    
    idx_path = os.path.join(tmp_path, "index.faiss")
    index = faiss.IndexFlatIP(dim)
    index.add(song_embeddings)
    faiss.write_index(index, idx_path)
    
    faiss_rec = LearnedFaissRecommender(idx_path, meta_path)
    
    # Models
    user_encoder = UserEncoder(input_dim=84, hidden_dim=32, embedding_dim=64)
    ranker = NeuralRanker(input_dim=133, hidden_dims=[32])
    
    istore = InteractionStore()
    
    return istore, metadata_lookup, user_encoder, ranker, faiss_rec, song_embeddings, song_id_to_row

def test_pipeline(mock_pipeline_components):
    istore, meta, u_enc, ranker, faiss_rec, embs, row_map = mock_pipeline_components
    
    pipeline = VNYLRecommenderPipeline(
        interaction_store=istore,
        metadata_lookup=meta,
        user_encoder=u_enc,
        ranker=ranker,
        faiss_recommender=faiss_rec,
        song_embeddings=embs,
        song_id_to_row=row_map
    )
    
    # Empty history
    res = pipeline.recommend("u1", top_k=3, candidate_pool_size=5)
    assert len(res) == 3
    assert "ranker_score" in res[0]
    
    # Add history
    t = datetime(2026, 1, 1, tzinfo=timezone.utc)
    istore.record_interaction("u1", "track_0", ActionType.LIKE, timestamp=t)
    istore.record_interaction("u1", "track_1", ActionType.PLAY, listen_duration=120, timestamp=t)
    
    res2 = pipeline.recommend("u1", top_k=3, candidate_pool_size=10)
    
    assert len(res2) == 3
    ids = [r["track_id"] for r in res2]
    # Should exclude interacted items
    assert "track_0" not in ids
    assert "track_1" not in ids
