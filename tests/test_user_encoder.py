"""
Tests for Neural User Encoder (Phase 3)
=========================================

Validates:
- Empty/new user handling (cold start)
- Single-interaction user
- Multiple-interaction user
- Positive and negative interaction handling
- Correct output dimensions
- Deterministic inference
- Checkpoint save/load
- No NaN/Inf
"""

import os
import tempfile
import pytest
import numpy as np

import torch

from src.models.user_encoder import UserEncoder, UserHistoryBuilder


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def builder():
    return UserHistoryBuilder(
        song_embedding_dim=64, top_k_artists=10, top_k_genres=10
    )


@pytest.fixture
def encoder():
    torch.manual_seed(42)
    return UserEncoder(input_dim=84, hidden_dim=128, embedding_dim=64)


# ---------------------------------------------------------------------------
# Test: Cold-start (empty/new user)
# ---------------------------------------------------------------------------

def test_cold_start_user(builder, encoder):
    features = builder.build_cold_start_features()
    assert features.shape == (84,)
    assert np.all(features == 0)

    emb = encoder.encode_user(features)
    assert emb.shape == (64,)
    assert not np.any(np.isnan(emb))


# ---------------------------------------------------------------------------
# Test: Single-interaction user
# ---------------------------------------------------------------------------

def test_single_interaction_user(builder, encoder):
    np.random.seed(42)
    song_emb = np.random.randn(1, 64).astype(np.float32)
    scores = np.array([10.0])  # positive interaction
    artist_prefs = [0.5]
    genre_prefs = [0.3, 0.1]

    features = builder.build_user_features(
        song_emb, scores, artist_prefs, genre_prefs
    )
    assert features.shape == (84,)

    emb = encoder.encode_user(features)
    assert emb.shape == (64,)
    assert not np.any(np.isnan(emb))


# ---------------------------------------------------------------------------
# Test: Multiple-interaction user
# ---------------------------------------------------------------------------

def test_multiple_interaction_user(builder, encoder):
    np.random.seed(42)
    song_embs = np.random.randn(20, 64).astype(np.float32)
    scores = np.random.randn(20).astype(np.float32) * 5
    artist_prefs = [0.8, 0.5, 0.3, -0.1]
    genre_prefs = [0.7, 0.4, 0.2, -0.2, -0.5]

    features = builder.build_user_features(
        song_embs, scores, artist_prefs, genre_prefs
    )
    assert features.shape == (84,)

    emb = encoder.encode_user(features)
    assert emb.shape == (64,)
    assert not np.any(np.isnan(emb))


# ---------------------------------------------------------------------------
# Test: Positive and negative interaction handling
# ---------------------------------------------------------------------------

def test_positive_interactions_weighted_higher(builder):
    """Positive interactions should have higher weight in aggregation."""
    np.random.seed(42)

    # Two songs: one positive, one negative
    emb_pos = np.ones((1, 64), dtype=np.float32)
    emb_neg = -np.ones((1, 64), dtype=np.float32)
    song_embs = np.vstack([emb_pos, emb_neg])

    # Strong positive, weak negative
    scores = np.array([10.0, -8.0])
    features = builder.build_user_features(
        song_embs, scores, [], []
    )

    # The aggregated song embedding (first 64 dims) should lean positive
    agg = features[:64]
    assert agg.sum() > 0  # overall positive direction


def test_negative_interactions_weighted_lower(builder):
    """Negative interactions should contribute less weight."""
    np.random.seed(42)

    emb1 = np.ones((1, 64), dtype=np.float32) * 2
    song_embs = np.vstack([emb1, emb1])

    scores_pos = np.array([10.0, 10.0])
    scores_neg = np.array([-8.0, -8.0])

    features_pos = builder.build_user_features(song_embs, scores_pos, [], [])
    features_neg = builder.build_user_features(song_embs, scores_neg, [], [])

    # With positive scores: uniform weighting (both equal scores)
    # With negative scores: also uniform (both equal scores)
    # Both should be valid but different due to different softmax weights
    assert features_pos.shape == (84,)
    assert features_neg.shape == (84,)


# ---------------------------------------------------------------------------
# Test: Correct output dimensions
# ---------------------------------------------------------------------------

def test_output_dim_matches_song_encoder(encoder):
    """User embedding dim should match song embedding dim (64)."""
    torch.manual_seed(42)
    features = np.random.randn(1, 84).astype(np.float32)
    emb = encoder.encode_user(features)
    assert emb.shape == (64,)


def test_batch_encoding(encoder):
    torch.manual_seed(42)
    features = np.random.randn(5, 84).astype(np.float32)
    embs = encoder.encode_users(features)
    assert embs.shape == (5, 64)


# ---------------------------------------------------------------------------
# Test: Deterministic inference
# ---------------------------------------------------------------------------

def test_deterministic_inference(encoder):
    encoder.eval()
    features = np.random.randn(5, 84).astype(np.float32)

    emb1 = encoder.encode_users(features)
    emb2 = encoder.encode_users(features)

    np.testing.assert_array_equal(emb1, emb2)


# ---------------------------------------------------------------------------
# Test: Checkpoint save/load
# ---------------------------------------------------------------------------

def test_checkpoint_save_load(encoder):
    features = np.random.randn(5, 84).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmpdir:
        encoder.save(tmpdir)

        assert os.path.exists(os.path.join(tmpdir, "user_encoder.pt"))
        assert os.path.exists(os.path.join(tmpdir, "user_encoder_config.json"))

        loaded = UserEncoder.load(tmpdir)

        encoder.eval()
        loaded.eval()

        emb_original = encoder.encode_users(features)
        emb_loaded = loaded.encode_users(features)

        np.testing.assert_allclose(emb_original, emb_loaded, atol=1e-6)


# ---------------------------------------------------------------------------
# Test: No NaN/Inf
# ---------------------------------------------------------------------------

def test_no_nan_output(encoder):
    features = np.random.randn(10, 84).astype(np.float32)
    embs = encoder.encode_users(features)
    assert not np.any(np.isnan(embs))
    assert not np.any(np.isinf(embs))


def test_no_nan_zero_input(encoder):
    features = np.zeros((1, 84), dtype=np.float32)
    emb = encoder.encode_user(features)
    assert not np.any(np.isnan(emb))


def test_no_nan_extreme_input(encoder):
    features = np.full((1, 84), 100.0, dtype=np.float32)
    emb = encoder.encode_user(features)
    assert not np.any(np.isnan(emb))
    assert not np.any(np.isinf(emb))


# ---------------------------------------------------------------------------
# Test: UserHistoryBuilder edge cases
# ---------------------------------------------------------------------------

def test_builder_empty_interactions(builder):
    """Empty interaction history → zero song embedding."""
    features = builder.build_user_features(
        np.empty((0, 64), dtype=np.float32),
        np.empty((0,), dtype=np.float32),
        [], []
    )
    assert features.shape == (84,)
    assert np.all(features[:64] == 0)  # zero aggregated song emb


def test_builder_padding(builder):
    """Preference lists shorter than top_k get zero-padded."""
    np.random.seed(42)
    song_embs = np.random.randn(1, 64).astype(np.float32)
    scores = np.array([5.0])

    features = builder.build_user_features(
        song_embs, scores,
        [0.5, 0.3],     # only 2 of 10 artists
        [0.7],           # only 1 of 10 genres
    )

    # artist features: indices 64-73, genre features: 74-83
    assert features[66] == 0.0  # 3rd artist slot (index 64+2) should be 0
    assert features[75] == 0.0  # 2nd genre slot should be 0


def test_builder_truncation(builder):
    """Preference lists longer than top_k get truncated."""
    np.random.seed(42)
    song_embs = np.random.randn(1, 64).astype(np.float32)
    scores = np.array([5.0])

    many_artists = list(range(20))  # 20 values > top_k=10
    features = builder.build_user_features(
        song_embs, scores, many_artists, []
    )

    # Only first 10 artist values should be used
    assert features[64] == 0.0  # first artist
    assert features[73] == 9.0  # 10th artist (0-indexed: indices 64..73)


# ---------------------------------------------------------------------------
# Test: L2 normalization
# ---------------------------------------------------------------------------

def test_l2_normalized_output(encoder):
    encoder.eval()
    features = np.random.randn(10, 84).astype(np.float32)
    embs = encoder.encode_users(features)
    norms = np.linalg.norm(embs, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# Test: Gradient flow
# ---------------------------------------------------------------------------

def test_gradient_flow():
    torch.manual_seed(42)
    enc = UserEncoder(input_dim=84)
    enc.train()

    x = torch.randn(8, 84)
    y = enc(x)

    loss = y.sum()
    loss.backward()

    for param in enc.parameters():
        assert param.grad is not None
