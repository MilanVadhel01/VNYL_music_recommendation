"""
Phase 4 Demo — Two-Tower Neural Recommender
===========================================

Demonstrates the two-tower neural recommender pipeline:
1. Synthetic dataset creation
2. TwoTowerTrainer data preparation (Chronological Split)
3. Model training (InfoNCE Loss)
4. Evaluation (Recall, Hit Rate, NDCG)
5. Inference candidate scoring

Usage:
    python src/models/demo_two_tower.py
"""

import sys
import os

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, ".")

import numpy as np
from datetime import datetime, timezone, timedelta

from src.interactions.models import InteractionStore, ActionType
from src.interactions.preferences import SongMetadataLookup
from src.models.two_tower import (
    TrainingConfig,
    TwoTowerTrainer,
    TwoTowerInference,
)


def main():
    print("=" * 60)
    print("PHASE 4 DEMO — Two-Tower Neural Recommender")
    print("=" * 60)
    print()

    # 1. Setup minimal synthetic environment
    print("1. Initializing synthetic environment...")
    store = InteractionStore()
    meta = SongMetadataLookup()

    num_songs = 100
    num_users = 20
    
    # Create random 13-dim song features
    np.random.seed(42)
    song_features = np.random.randn(num_songs, 13).astype(np.float32)
    song_id_to_row = {}

    for i in range(num_songs):
        song_id = f"song_{i}"
        artist_id = f"artist_{i % 10}"
        artist_name = f"Artist {i % 10}"
        
        # Pick some genres
        genres_list = [["rock", "pop"], ["jazz"], ["classical"], ["electronic", "dance"]]
        genres = genres_list[i % 4]
        
        meta.register_song(song_id, artist_id, artist_name, genres)
        song_id_to_row[song_id] = i

    # Generate synthetic interactions
    t0 = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    for u in range(num_users):
        user_id = f"user_{u}"
        
        # Each user interacts with 15 songs
        for step in range(15):
            # Introduce some patterns: user_0 likes artist_0, user_1 likes artist_1, etc.
            preferred_artist = u % 10
            
            # 70% chance to pick a song from preferred artist, 30% random
            if np.random.rand() < 0.7:
                candidate_songs = [i for i in range(num_songs) if i % 10 == preferred_artist]
                song_idx = int(np.random.choice(candidate_songs))
            else:
                song_idx = np.random.randint(0, num_songs)
                
            song_id = f"song_{song_idx}"
            t = t0 + timedelta(hours=u * 10 + step)
            
            import random
            
            # Actions: mostly positive for preferred, negative/neutral for others
            if song_idx % 10 == preferred_artist:
                action = random.choice([ActionType.LIKE, ActionType.COMPLETE, ActionType.REPLAY])
                store.record_interaction(user_id, song_id, action, timestamp=t)
            else:
                action = random.choice([ActionType.SKIP, ActionType.DISLIKE, ActionType.PLAY])
                kwargs = {"timestamp": t}
                if action == ActionType.SKIP:
                    kwargs["listen_duration"] = 5.0  # strong skip
                store.record_interaction(user_id, song_id, action, **kwargs)

    print(f"   Created {num_songs} songs, {num_users} users, {len(store.get_all_events())} interactions.")
    print()

    # 2. Config & Trainer
    print("2. Configuring Two-Tower Trainer...")
    config = TrainingConfig(
        song_embedding_dim=16,
        user_embedding_dim=16,
        song_hidden_dim=32,
        user_hidden_dim=32,
        batch_size=32,
        epochs=15,
        learning_rate=0.005,
    )

    trainer = TwoTowerTrainer(
        config=config,
        interaction_store=store,
        metadata_lookup=meta,
        song_features=song_features,
        song_id_to_row=song_id_to_row,
    )

    print("   Preparing data (Chronological Train/Val Split)...")
    stats = trainer.prepare_data()
    for k, v in stats.items():
        print(f"     {k}: {v}")
    print()

    # 3. Training
    print("3. Training Model (InfoNCE Loss)...")
    results = trainer.train()
    
    print(f"   Completed {results['epochs']} epochs in {results['elapsed_seconds']:.2f}s")
    print(f"   Final Train Loss: {results['final_train_loss']:.4f}")
    print(f"   Final Val Loss:   {results['final_val_loss']:.4f}")
    print()

    # 4. Evaluation
    print("4. Evaluating Validation Set Metrics...")
    metrics = trainer.evaluate(top_k=5)
    print(f"   Recall@5:   {metrics['recall@k']:.4f}")
    print(f"   HitRate@5:  {metrics['hit_rate@k']:.4f}")
    print(f"   NDCG@5:     {metrics['ndcg@k']:.4f}")
    print("   (Note: These metrics are on a tiny synthetic dataset)")
    print()

    # 5. Inference
    print("5. Running Inference Scoring...")
    inference = TwoTowerInference(
        model=trainer.model,
        song_features=song_features,
        song_id_to_row=song_id_to_row,
        interaction_store=store,
        metadata_lookup=meta,
        config=config,
    )

    # Score some songs for user_0 (who prefers artist_0)
    # We'll score a mix of artist_0 songs and others
    candidates = ["song_0", "song_10", "song_20", "song_1", "song_2", "song_3"]
    scores = inference.score_user_songs("user_0", candidates)
    
    print("   Scores for user_0 (prefers artist_0):")
    for song_id, score in scores:
        artist_id = meta.get_artist(song_id)[0] if meta.get_artist(song_id) else "Unknown"
        marker = " (MATCH)" if artist_id == "artist_0" else ""
        print(f"     {song_id:10s} [Artist: {artist_id:10s}] -> Score: {score:+.4f}{marker}")
    
    print()
    print("=" * 60)
    print("Phase 4 Demo Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
