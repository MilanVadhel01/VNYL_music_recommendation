"""
Tests for Neural Ranker (Phase 7)
=================================

Validates:
- Feature extraction logic (dimension 133, no leakage, proper UPS/preferences)
- Neural Ranker forward pass and inference
- Model serialization
"""

import os
import pytest
import numpy as np
import torch
from datetime import datetime, timezone

from src.interactions.models import ActionType, UserInteraction
from src.interactions.preferences import SongMetadataLookup
from src.models.neural_ranker import CandidateFeatureExtractor, NeuralRanker

@pytest.fixture
def dummy_data():
    dim = 64
    num_songs = 10
    song_embeddings = np.random.randn(num_songs, dim).astype(np.float32)
    song_id_to_row = {f"track_{i}": i for i in range(num_songs)}
    
    metadata = SongMetadataLookup()
    for i in range(num_songs):
        metadata.register_song(f"track_{i}", f"artist_{i % 3}", f"Artist {i % 3}", [f"genre_{i % 2}"])
        
    return metadata, song_embeddings, song_id_to_row

def test_feature_extractor(dummy_data):
    metadata, song_embeddings, song_id_to_row = dummy_data
    extractor = CandidateFeatureExtractor(metadata, song_embeddings, song_id_to_row)
    
    user_emb = np.random.randn(64).astype(np.float32)
    
    # Create some history
    t1 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    t2 = datetime(2026, 1, 2, tzinfo=timezone.utc)
    
    history = [
        UserInteraction(user_id="u1", song_id="track_1", action=ActionType.LIKE, timestamp=t1),
        UserInteraction(user_id="u1", song_id="track_1", action=ActionType.REPLAY, timestamp=t2),
        UserInteraction(user_id="u1", song_id="track_2", action=ActionType.DISLIKE, timestamp=t1),
    ]
    
    # Extract features for track_1
    features = extractor.extract_features("u1", user_emb, "track_1", history)
    
    assert features is not None
    assert features.shape == (133,)
    assert features.dtype == np.float32
    
    # Values check
    # similarity = dot product
    expected_sim = np.dot(user_emb, song_embeddings[1])
    assert np.isclose(features[128], expected_sim)
    
    # Interaction count for track_1 is 2
    assert features[131] == 2.0
    
    # Historical UPS for track_1: LIKE(+10) + REPLAY(+6) = 16.0 (before decay)
    # But since it's historical UPS, decay is applied relative to now. We just check it's > 0.
    assert features[132] > 0.0
    
    # Extract features for unknown track should return None
    assert extractor.extract_features("u1", user_emb, "track_99", history) is None

def test_neural_ranker():
    model = NeuralRanker(input_dim=133, hidden_dims=[64, 32], dropout=0.0)
    
    # Forward pass
    x = torch.randn(10, 133)
    logits = model(x)
    assert logits.shape == (10,)
    
    # Inference
    x_np = np.random.randn(5, 133).astype(np.float32)
    scores = model.predict_score(x_np)
    assert scores.shape == (5,)
    
    # Single item
    score1 = model.predict_score(x_np[0])
    assert score1.shape == (1,)
    
def test_ranker_serialization(tmp_path):
    model = NeuralRanker(input_dim=133)
    path = os.path.join(tmp_path, "ranker.pt")
    model.save(path)
    
    model2 = NeuralRanker.load(path, input_dim=133)
    
    # Compare weights
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2)

def test_ranker_trainer():
    from src.models.neural_ranker import RankerTrainer
    
    model = NeuralRanker(input_dim=133, hidden_dims=[32])
    trainer = RankerTrainer(model)
    
    features = np.random.randn(100, 133).astype(np.float32)
    labels = np.random.randint(0, 2, size=(100,)).astype(np.float32)
    
    train_loss = trainer.train_epoch(features, labels, batch_size=32)
    assert train_loss > 0
    
    val_loss = trainer.evaluate(features, labels, batch_size=32)
    assert val_loss > 0
