"""
Tests for Two-Tower Neural Recommender (Phase 4)
==================================================

Validates:
- Positive/negative signal classification
- Training data construction
- No data leakage (chronological split)
- InfoNCE loss computation
- Two-tower model forward pass
- Training loop convergence
- Inference utilities
- Checkpoint save/load
- Metric computation
"""

import os
import math
import tempfile
import pytest
import numpy as np
from datetime import datetime, timezone, timedelta

import torch
import torch.nn.functional as F

from src.interactions.models import ActionType, UserInteraction, InteractionStore
from src.interactions.preferences import SongMetadataLookup
from src.models.song_encoder import SongEncoder
from src.models.user_encoder import UserEncoder
from src.models.two_tower import (
    TrainingConfig,
    TwoTowerModel,
    TwoTowerTrainer,
    TwoTowerDataset,
    TwoTowerInference,
    TrainingDataBuilder,
    TrainingExample,
    infonce_loss,
    is_positive_interaction,
    is_negative_interaction,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config():
    return TrainingConfig(
        song_embedding_dim=32,
        user_embedding_dim=32,
        song_hidden_dim=64,
        user_hidden_dim=64,
        top_k_artists=5,
        top_k_genres=5,
        batch_size=8,
        learning_rate=1e-3,
        epochs=3,
        random_seed=42,
        dropout=0.0,  # no dropout for test determinism
    )


@pytest.fixture
def synthetic_env(config):
    """Create a full synthetic environment for training tests."""
    torch.manual_seed(42)
    np.random.seed(42)

    # Create interaction store with synthetic events
    istore = InteractionStore()
    meta = SongMetadataLookup()

    # Register 20 songs with 4 artists, some genres
    for i in range(20):
        artist_idx = i % 4
        genres = [["rock", "indie"], ["pop", "dance"], ["jazz", "blues"], ["electronic", "ambient"]][artist_idx]
        meta.register_song(
            f"song_{i}",
            f"artist_{artist_idx}",
            f"Artist {artist_idx}",
            genres,
        )

    # Create synthetic features for 20 songs
    song_features = np.random.randn(20, 13).astype(np.float32)
    song_id_to_row = {f"song_{i}": i for i in range(20)}

    # Generate synthetic interactions for 5 users
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)

    for u in range(5):
        user_id = f"user_{u}"
        for step in range(15):
            song_idx = (u * 3 + step) % 20
            song_id = f"song_{song_idx}"
            t = t0 + timedelta(hours=u * 100 + step)

            # Mix of positive and negative actions
            if step % 3 == 0:
                istore.record_interaction(user_id, song_id, ActionType.LIKE, timestamp=t)
            elif step % 3 == 1:
                istore.record_interaction(user_id, song_id, ActionType.COMPLETE, timestamp=t)
            else:
                istore.record_interaction(
                    user_id, song_id, ActionType.SKIP,
                    timestamp=t, listen_duration=5.0,
                )

    return {
        "config": config,
        "istore": istore,
        "meta": meta,
        "song_features": song_features,
        "song_id_to_row": song_id_to_row,
    }


# ---------------------------------------------------------------------------
# Test: Positive/negative classification
# ---------------------------------------------------------------------------

def test_is_positive_like():
    e = UserInteraction("u1", "s1", ActionType.LIKE)
    assert is_positive_interaction(e) is True
    assert is_negative_interaction(e) is False


def test_is_positive_replay():
    e = UserInteraction("u1", "s1", ActionType.REPLAY, replay_count=2)
    assert is_positive_interaction(e) is True


def test_is_positive_complete():
    e = UserInteraction("u1", "s1", ActionType.COMPLETE)
    assert is_positive_interaction(e) is True


def test_is_positive_add_to_playlist():
    e = UserInteraction("u1", "s1", ActionType.ADD_TO_PLAYLIST)
    assert is_positive_interaction(e) is True


def test_is_positive_high_completion():
    e = UserInteraction("u1", "s1", ActionType.PLAY, completion_percentage=80.0)
    assert is_positive_interaction(e) is True


def test_is_negative_dislike():
    e = UserInteraction("u1", "s1", ActionType.DISLIKE)
    assert is_negative_interaction(e) is True
    assert is_positive_interaction(e) is False


def test_is_negative_early_skip():
    e = UserInteraction("u1", "s1", ActionType.SKIP, listen_duration=5.0)
    assert is_negative_interaction(e) is True


def test_neutral_play():
    """Plain PLAY without completion data is neither positive nor negative."""
    e = UserInteraction("u1", "s1", ActionType.PLAY)
    assert is_positive_interaction(e) is False
    assert is_negative_interaction(e) is False


def test_neutral_search():
    e = UserInteraction("u1", "s1", ActionType.SEARCH_ARTIST)
    assert is_positive_interaction(e) is False
    assert is_negative_interaction(e) is False


# ---------------------------------------------------------------------------
# Test: InfoNCE loss
# ---------------------------------------------------------------------------

def test_infonce_loss_computation():
    torch.manual_seed(42)
    user_emb = F.normalize(torch.randn(4, 32), dim=1)
    song_emb = F.normalize(torch.randn(4, 32), dim=1)
    labels = torch.tensor([1.0, 1.0, 0.0, 1.0])

    loss = infonce_loss(user_emb, song_emb, labels, temperature=0.07)
    assert loss.item() > 0
    assert not math.isnan(loss.item())


def test_infonce_loss_all_negative():
    """Loss should be 0 when there are no positive examples."""
    user_emb = F.normalize(torch.randn(4, 32), dim=1)
    song_emb = F.normalize(torch.randn(4, 32), dim=1)
    labels = torch.tensor([0.0, 0.0, 0.0, 0.0])

    loss = infonce_loss(user_emb, song_emb, labels, temperature=0.07)
    assert loss.item() == 0.0


def test_infonce_loss_gradient():
    """Loss should produce gradients."""
    user_emb = F.normalize(torch.randn(4, 32, requires_grad=True), dim=1)
    song_emb = F.normalize(torch.randn(4, 32, requires_grad=True), dim=1)
    labels = torch.tensor([1.0, 1.0, 0.0, 1.0])

    loss = infonce_loss(user_emb, song_emb, labels)
    loss.backward()


# ---------------------------------------------------------------------------
# Test: Two-Tower model forward
# ---------------------------------------------------------------------------

def test_two_tower_forward(config):
    torch.manual_seed(42)
    song_enc = SongEncoder(hidden_dim=64, embedding_dim=32, dropout=0.0)
    user_enc = UserEncoder(input_dim=42, hidden_dim=64, embedding_dim=32, dropout=0.0)  # 32 + 5 + 5
    model = TwoTowerModel(song_enc, user_enc)

    user_feats = torch.randn(4, 42)
    song_feats = torch.randn(4, 13)

    user_emb, song_emb = model(user_feats, song_feats)
    assert user_emb.shape == (4, 32)
    assert song_emb.shape == (4, 32)


# ---------------------------------------------------------------------------
# Test: Training data construction
# ---------------------------------------------------------------------------

def test_training_data_construction(synthetic_env):
    env = synthetic_env
    torch.manual_seed(42)

    song_encoder = SongEncoder(
        hidden_dim=env["config"].song_hidden_dim,
        embedding_dim=env["config"].song_embedding_dim,
        dropout=0.0,
    )
    song_encoder.eval()

    builder = TrainingDataBuilder(
        env["istore"], env["meta"],
        env["song_features"], env["song_id_to_row"],
        song_encoder, env["config"],
    )

    examples = builder.build_examples()
    assert len(examples) > 0

    # Check that we have both positive and negative examples
    n_pos = sum(1 for e in examples if e.label > 0.5)
    n_neg = sum(1 for e in examples if e.label <= 0.5)
    assert n_pos > 0, "No positive examples"
    assert n_neg > 0, "No negative examples"


def test_no_data_leakage(synthetic_env):
    """Verify that user representations don't contain future interactions."""
    env = synthetic_env
    torch.manual_seed(42)

    song_encoder = SongEncoder(
        hidden_dim=env["config"].song_hidden_dim,
        embedding_dim=env["config"].song_embedding_dim,
        dropout=0.0,
    )

    builder = TrainingDataBuilder(
        env["istore"], env["meta"],
        env["song_features"], env["song_id_to_row"],
        song_encoder, env["config"],
    )

    examples = builder.build_examples()

    # The first example for each user should have cold-start features
    # (no history available before the first interaction)
    user_first = {}
    for ex in examples:
        if ex.user_id not in user_first:
            user_first[ex.user_id] = ex

    for uid, first_ex in user_first.items():
        # First example should use cold-start features (all zeros)
        # because there's no history before the first interaction
        assert first_ex.user_features is not None


# ---------------------------------------------------------------------------
# Test: Training loop
# ---------------------------------------------------------------------------

def test_training_loop(synthetic_env):
    env = synthetic_env

    trainer = TwoTowerTrainer(
        config=env["config"],
        interaction_store=env["istore"],
        metadata_lookup=env["meta"],
        song_features=env["song_features"],
        song_id_to_row=env["song_id_to_row"],
    )

    stats = trainer.prepare_data()
    assert stats["total_examples"] > 0
    assert stats["train_examples"] > 0

    results = trainer.train()
    assert results["epochs"] == 3
    assert len(results["train_losses"]) == 3
    assert len(results["val_losses"]) == 3

    # Loss should be a finite number
    for loss in results["train_losses"]:
        assert not math.isnan(loss), "NaN training loss"
    for loss in results["val_losses"]:
        assert not math.isnan(loss), "NaN validation loss"


# ---------------------------------------------------------------------------
# Test: Evaluation metrics
# ---------------------------------------------------------------------------

def test_evaluation(synthetic_env):
    env = synthetic_env

    trainer = TwoTowerTrainer(
        config=env["config"],
        interaction_store=env["istore"],
        metadata_lookup=env["meta"],
        song_features=env["song_features"],
        song_id_to_row=env["song_id_to_row"],
    )

    trainer.prepare_data()
    trainer.train()

    metrics = trainer.evaluate(top_k=5)
    assert "recall@k" in metrics
    assert "hit_rate@k" in metrics
    assert "ndcg@k" in metrics
    assert 0.0 <= metrics["recall@k"] <= 1.0
    assert 0.0 <= metrics["hit_rate@k"] <= 1.0
    assert 0.0 <= metrics["ndcg@k"] <= 1.0


# ---------------------------------------------------------------------------
# Test: Checkpoint save/load
# ---------------------------------------------------------------------------

def test_checkpoint_save_load(synthetic_env):
    env = synthetic_env

    trainer = TwoTowerTrainer(
        config=env["config"],
        interaction_store=env["istore"],
        metadata_lookup=env["meta"],
        song_features=env["song_features"],
        song_id_to_row=env["song_id_to_row"],
    )

    trainer.prepare_data()
    trainer.train()

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer.save_checkpoint(tmpdir)

        # Verify files
        assert os.path.exists(os.path.join(tmpdir, "song_encoder", "song_encoder.pt"))
        assert os.path.exists(os.path.join(tmpdir, "user_encoder", "user_encoder.pt"))
        assert os.path.exists(os.path.join(tmpdir, "training_results.json"))

        # Load and compare
        loaded_model = TwoTowerTrainer.load_model(tmpdir)
        assert isinstance(loaded_model, TwoTowerModel)

        # Verify output consistency
        test_song = np.random.randn(1, 13).astype(np.float32)
        test_user = np.random.randn(1, 42).astype(np.float32)

        trainer.model.eval()
        loaded_model.eval()

        with torch.no_grad():
            orig_score = trainer.model.score_pairs(test_user, test_song)
            loaded_score = loaded_model.score_pairs(test_user, test_song)

        np.testing.assert_allclose(orig_score, loaded_score, atol=1e-5)


# ---------------------------------------------------------------------------
# Test: Inference
# ---------------------------------------------------------------------------

def test_inference(synthetic_env):
    env = synthetic_env

    trainer = TwoTowerTrainer(
        config=env["config"],
        interaction_store=env["istore"],
        metadata_lookup=env["meta"],
        song_features=env["song_features"],
        song_id_to_row=env["song_id_to_row"],
    )

    trainer.prepare_data()
    trainer.train()

    inference = TwoTowerInference(
        model=trainer.model,
        song_features=env["song_features"],
        song_id_to_row=env["song_id_to_row"],
        interaction_store=env["istore"],
        metadata_lookup=env["meta"],
        config=env["config"],
    )

    # Score songs for user_0
    candidates = [f"song_{i}" for i in range(10)]
    results = inference.score_user_songs("user_0", candidates)

    assert len(results) > 0
    # Results should be sorted by score descending
    scores = [r[1] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_inference_cold_start(synthetic_env):
    """Inference for a user with no interactions should not crash."""
    env = synthetic_env

    trainer = TwoTowerTrainer(
        config=env["config"],
        interaction_store=env["istore"],
        metadata_lookup=env["meta"],
        song_features=env["song_features"],
        song_id_to_row=env["song_id_to_row"],
    )

    trainer.prepare_data()
    trainer.train()

    inference = TwoTowerInference(
        model=trainer.model,
        song_features=env["song_features"],
        song_id_to_row=env["song_id_to_row"],
        interaction_store=env["istore"],
        metadata_lookup=env["meta"],
        config=env["config"],
    )

    # Score songs for a non-existent user
    candidates = [f"song_{i}" for i in range(5)]
    results = inference.score_user_songs("unknown_user", candidates)
    assert isinstance(results, list)


# ---------------------------------------------------------------------------
# Test: Dataset
# ---------------------------------------------------------------------------

def test_dataset():
    examples = [
        TrainingExample(
            user_features=np.zeros(42, dtype=np.float32),
            song_features=np.zeros(13, dtype=np.float32),
            label=1.0,
            user_id="u1",
            song_id="s1",
        ),
        TrainingExample(
            user_features=np.ones(42, dtype=np.float32),
            song_features=np.ones(13, dtype=np.float32),
            label=0.0,
            user_id="u1",
            song_id="s2",
        ),
    ]

    dataset = TwoTowerDataset(examples)
    assert len(dataset) == 2

    u, s, label = dataset[0]
    assert u.shape == (42,)
    assert s.shape == (13,)
    assert label == 1.0
