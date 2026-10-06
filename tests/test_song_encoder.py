"""
Tests for Neural Song Encoder (Phase 2)
========================================

Validates:
- Correct input shape
- Correct output shape
- Batch inference
- No NaN/Inf output
- Repeatability in evaluation mode
- Checkpoint save/load
- Compatibility with existing processed feature matrix shape
"""

import os
import tempfile
import pytest
import numpy as np

import torch

from src.models.song_encoder import SongEncoder, INPUT_DIM, CONTENT_FEATURE_ORDER


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def encoder():
    """Create a default SongEncoder."""
    torch.manual_seed(42)
    return SongEncoder(input_dim=13, hidden_dim=128, embedding_dim=64)


@pytest.fixture
def sample_features():
    """Create sample feature vectors matching the 13-feature layout."""
    np.random.seed(42)
    return np.random.randn(10, 13).astype(np.float32)


# ---------------------------------------------------------------------------
# Test: Input dimension
# ---------------------------------------------------------------------------

def test_input_dim():
    assert INPUT_DIM == 13
    assert len(CONTENT_FEATURE_ORDER) == 13


# ---------------------------------------------------------------------------
# Test: Correct output shape — single song
# ---------------------------------------------------------------------------

def test_encode_single_song(encoder, sample_features):
    single = sample_features[0]
    emb = encoder.encode_song(single)
    assert emb.shape == (64,), f"Expected (64,), got {emb.shape}"


def test_encode_single_song_2d(encoder, sample_features):
    single = sample_features[0:1]  # shape (1, 13)
    emb = encoder.encode_song(single)
    assert emb.shape == (64,)


# ---------------------------------------------------------------------------
# Test: Correct output shape — batch
# ---------------------------------------------------------------------------

def test_encode_songs_batch(encoder, sample_features):
    embs = encoder.encode_songs(sample_features)
    assert embs.shape == (10, 64), f"Expected (10, 64), got {embs.shape}"


def test_encode_songs_single_batch(encoder, sample_features):
    embs = encoder.encode_songs(sample_features[0:1])
    assert embs.shape == (1, 64)


# ---------------------------------------------------------------------------
# Test: No NaN/Inf
# ---------------------------------------------------------------------------

def test_no_nan_output(encoder, sample_features):
    embs = encoder.encode_songs(sample_features)
    assert not np.any(np.isnan(embs)), "Output contains NaN"


def test_no_inf_output(encoder, sample_features):
    embs = encoder.encode_songs(sample_features)
    assert not np.any(np.isinf(embs)), "Output contains Inf"


def test_no_nan_extreme_input(encoder):
    """Test with extreme but valid input values."""
    extreme = np.array([[100, -100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32)
    emb = encoder.encode_song(extreme)
    assert not np.any(np.isnan(emb))
    assert not np.any(np.isinf(emb))


def test_no_nan_zero_input(encoder):
    """Test with all-zero input."""
    zeros = np.zeros((1, 13), dtype=np.float32)
    emb = encoder.encode_song(zeros)
    assert not np.any(np.isnan(emb))


# ---------------------------------------------------------------------------
# Test: Repeatability in eval mode
# ---------------------------------------------------------------------------

def test_repeatability_eval_mode(encoder, sample_features):
    encoder.eval()
    emb1 = encoder.encode_songs(sample_features)
    emb2 = encoder.encode_songs(sample_features)
    np.testing.assert_array_equal(emb1, emb2)


def test_repeatability_single(encoder, sample_features):
    encoder.eval()
    e1 = encoder.encode_song(sample_features[0])
    e2 = encoder.encode_song(sample_features[0])
    np.testing.assert_array_equal(e1, e2)


# ---------------------------------------------------------------------------
# Test: Checkpoint save/load
# ---------------------------------------------------------------------------

def test_checkpoint_save_load(encoder, sample_features):
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save
        encoder.save(tmpdir)

        # Verify files exist
        assert os.path.exists(os.path.join(tmpdir, "song_encoder.pt"))
        assert os.path.exists(os.path.join(tmpdir, "song_encoder_config.json"))

        # Load
        loaded = SongEncoder.load(tmpdir)

        # Compare outputs
        encoder.eval()
        loaded.eval()
        emb_original = encoder.encode_songs(sample_features)
        emb_loaded = loaded.encode_songs(sample_features)

        np.testing.assert_allclose(emb_original, emb_loaded, atol=1e-6)


def test_config_roundtrip(encoder):
    config = encoder.get_config()
    assert config["input_dim"] == 13
    assert config["hidden_dim"] == 128
    assert config["embedding_dim"] == 64
    assert config["normalize"] is True
    assert config["feature_order"] == CONTENT_FEATURE_ORDER


# ---------------------------------------------------------------------------
# Test: L2 normalization
# ---------------------------------------------------------------------------

def test_l2_normalized_output(encoder, sample_features):
    encoder.eval()
    embs = encoder.encode_songs(sample_features)
    norms = np.linalg.norm(embs, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_unnormalized_encoder(sample_features):
    """Verify that setting normalize=False skips L2 norm."""
    torch.manual_seed(42)
    enc = SongEncoder(normalize=False)
    enc.eval()
    embs = enc.encode_songs(sample_features)
    norms = np.linalg.norm(embs, axis=1)
    # At least some norms should differ from 1.0
    assert not np.allclose(norms, 1.0, atol=0.01)


# ---------------------------------------------------------------------------
# Test: Configurable dimensions
# ---------------------------------------------------------------------------

def test_custom_embedding_dim():
    torch.manual_seed(42)
    enc = SongEncoder(embedding_dim=128)
    features = np.random.randn(5, 13).astype(np.float32)
    embs = enc.encode_songs(features)
    assert embs.shape == (5, 128)


def test_custom_hidden_dim():
    torch.manual_seed(42)
    enc = SongEncoder(hidden_dim=256, embedding_dim=32)
    features = np.random.randn(5, 13).astype(np.float32)
    embs = enc.encode_songs(features)
    assert embs.shape == (5, 32)


# ---------------------------------------------------------------------------
# Test: Compatibility with existing feature matrix shape
# ---------------------------------------------------------------------------

def test_compatible_with_feature_matrix_shape():
    """Verify encoder handles the (1000000, 13) feature matrix shape.
    
    We test with a small representative batch rather than loading the
    full 1M matrix.
    """
    torch.manual_seed(42)
    enc = SongEncoder()
    # Simulate a batch from the 1M feature matrix
    batch = np.random.randn(100, 13).astype(np.float32)
    embs = enc.encode_songs(batch)
    assert embs.shape == (100, 64)


# ---------------------------------------------------------------------------
# Test: Gradient flow (training mode)
# ---------------------------------------------------------------------------

def test_gradient_flow():
    """Verify gradients flow through the encoder during training."""
    torch.manual_seed(42)
    enc = SongEncoder()
    enc.train()

    x = torch.randn(8, 13)
    y = enc(x)

    # Create a dummy loss and backpropagate
    loss = y.sum()
    loss.backward()

    # Check that gradients exist
    for param in enc.parameters():
        assert param.grad is not None
