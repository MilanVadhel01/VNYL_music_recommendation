"""
Tests for Learned Song Embeddings Generation (Phase 5)
======================================================

Validates:
- Embedding dimension
- Number of songs encoded matches input
- Row-to-song mapping preservation
- NaN/Inf count (should be 0)
- Embedding norm statistics (L2 normalized)
- Checkpoint reproducibility
"""

import os
import tempfile
import pytest
import numpy as np
import pandas as pd
import torch

from src.models.generate_embeddings import generate_embeddings
from src.models.song_encoder import SongEncoder, INPUT_DIM

@pytest.fixture
def synthetic_data(tmp_path):
    """Create a small synthetic feature matrix and index for testing."""
    num_songs = 1000
    features = np.random.randn(num_songs, INPUT_DIM).astype(np.float32)
    
    # Create corresponding index
    index_df = pd.DataFrame({
        "row_index": range(num_songs),
        "track_id": [f"track_{i}" for i in range(num_songs)],
        "track_name": [f"Song {i}" for i in range(num_songs)],
        "artist_name": [f"Artist {i}" for i in range(num_songs)]
    })
    
    features_path = os.path.join(tmp_path, "test_features.npy")
    index_path = os.path.join(tmp_path, "test_index.parquet")
    
    np.save(features_path, features)
    index_df.to_parquet(index_path)
    
    return {
        "features_path": features_path,
        "index_path": index_path,
        "num_songs": num_songs,
        "tmp_path": tmp_path
    }

def test_generate_embeddings(synthetic_data):
    """Test the embedding generation pipeline end-to-end."""
    env = synthetic_data
    
    model_save_dir = os.path.join(env["tmp_path"], "model_v1")
    embeddings_save_path = os.path.join(env["tmp_path"], "embeddings_v1.npy")
    index_save_path = os.path.join(env["tmp_path"], "embeddings_index_v1.parquet")
    
    dim = 32
    
    # Run generation
    generate_embeddings(
        features_path=env["features_path"],
        index_path=env["index_path"],
        model_save_dir=model_save_dir,
        embeddings_save_path=embeddings_save_path,
        index_save_path=index_save_path,
        batch_size=100,
        embedding_dim=dim,
    )
    
    # 1. Verify artifacts exist
    assert os.path.exists(os.path.join(model_save_dir, "song_encoder.pt"))
    assert os.path.exists(embeddings_save_path)
    assert os.path.exists(index_save_path)
    
    # 2. Check dimensions and row counts
    embs = np.load(embeddings_save_path)
    assert embs.shape == (env["num_songs"], dim)
    
    idx_df = pd.read_parquet(index_save_path)
    assert len(idx_df) == env["num_songs"]
    
    # 3. Check mapping preservation
    # The order of the output embeddings should match the output index exactly
    # We can't verify individual embedding correctness trivially without re-running,
    # but we can verify the index matches the input index exactly
    orig_idx_df = pd.read_parquet(env["index_path"])
    pd.testing.assert_frame_equal(idx_df, orig_idx_df)
    
    # 4. No NaN/Inf
    assert not np.any(np.isnan(embs))
    assert not np.any(np.isinf(embs))
    
    # 5. Embedding Norm Statistics (L2 normalized)
    norms = np.linalg.norm(embs, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-4)

def test_checkpoint_reproducibility(synthetic_data):
    """Ensure the saved checkpoint produces the exact same embeddings."""
    env = synthetic_data
    
    model_save_dir = os.path.join(env["tmp_path"], "model_v1")
    embeddings_save_path = os.path.join(env["tmp_path"], "embeddings_v1.npy")
    index_save_path = os.path.join(env["tmp_path"], "embeddings_index_v1.parquet")
    
    generate_embeddings(
        features_path=env["features_path"],
        index_path=env["index_path"],
        model_save_dir=model_save_dir,
        embeddings_save_path=embeddings_save_path,
        index_save_path=index_save_path,
        batch_size=100,
        embedding_dim=16,
    )
    
    # Load embeddings produced by the script
    generated_embs = np.load(embeddings_save_path)
    
    # Load model and run inference manually
    loaded_model = SongEncoder.load(model_save_dir)
    loaded_model.eval()
    
    features = np.load(env["features_path"]).astype(np.float32)
    with torch.no_grad():
        manual_embs = loaded_model.encode_songs(features)
        
    # Should be identical
    np.testing.assert_allclose(generated_embs, manual_embs, atol=1e-5)
