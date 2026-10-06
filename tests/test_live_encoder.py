"""
Phase 9 Tests: Live New-Song Encoding
=====================================
"""

import os
import pytest
import numpy as np
import pandas as pd
import faiss
import joblib
import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler
from src.models.song_encoder import SongEncoder
from src.models.live_encoder import LiveSongEncoder
from src.data.prepare_features import CANDIDATE_FEATURES

@pytest.fixture
def mock_system(tmp_path):
    # 1. Mock Model
    model_dir = os.path.join(tmp_path, "model")
    model = SongEncoder(input_dim=13, hidden_dim=32, embedding_dim=64)
    model.save(model_dir)
    
    # 2. Mock Scaler
    scaler = StandardScaler()
    scaler.fit(np.random.randn(10, 13))
    scaler_path = os.path.join(tmp_path, "scaler.joblib")
    joblib.dump(scaler, scaler_path)
    
    # 3. Mock FAISS Index
    dim = 64
    index = faiss.IndexFlatIP(dim)
    index.add(np.random.randn(5, dim).astype(np.float32))
    faiss_path = os.path.join(tmp_path, "index.faiss")
    faiss.write_index(index, faiss_path)
    
    # 4. Mock Metadata
    df = pd.DataFrame({
        "row_index": range(5),
        "track_id": [f"t{i}" for i in range(5)],
        "track_name": [f"Song {i}" for i in range(5)],
        "artist_name": [f"Artist {i}" for i in range(5)]
    })
    meta_path = os.path.join(tmp_path, "meta.parquet")
    df.to_parquet(meta_path)
    
    return model_dir, scaler_path, faiss_path, meta_path

def test_live_encoder(mock_system):
    model_dir, scaler_path, faiss_path, meta_path = mock_system
    
    encoder = LiveSongEncoder(model_dir, scaler_path, faiss_path, meta_path)
    
    assert encoder.index.ntotal == 5
    assert len(encoder.meta_df) == 5
    
    song_data = {
        "track_id": "new_track_001",
        "track_name": "New Song",
        "artist_name": "New Artist",
    }
    # Fill with some features
    for f in CANDIDATE_FEATURES:
        song_data[f] = 0.5
        
    inserted_id = encoder.encode_and_insert(song_data)
    
    assert inserted_id == "new_track_001"
    
    # Check that index grew
    assert encoder.index.ntotal == 6
    assert len(encoder.meta_df) == 6
    
    # Check that disk is updated
    index2 = faiss.read_index(faiss_path)
    assert index2.ntotal == 6
    
    df2 = pd.read_parquet(meta_path)
    assert len(df2) == 6
    assert df2.iloc[-1]["track_id"] == "new_track_001"
