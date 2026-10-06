"""
Phase 10: Final Evaluation & Recommendation Pipeline
===================================================

Consolidated pipeline for recommending songs to users using the full system.
Pipeline flow:
1. `UserEncoder` generates the user embedding.
2. `LearnedFaissRecommender` retrieves candidate pool.
3. `CandidateFeatureExtractor` builds ranking features.
4. `NeuralRanker` re-ranks candidates and returns the Top K.
"""

import numpy as np
from typing import List, Dict, Any, Optional

from src.interactions.models import InteractionStore, UserInteraction
from src.interactions.preferences import SongMetadataLookup, PreferenceStore
from src.models.user_encoder import UserHistoryBuilder
from src.models.two_tower import UserEncoder
from src.models.learned_recommender import LearnedFaissRecommender
from src.models.neural_ranker import CandidateFeatureExtractor, NeuralRanker

class VNYLRecommenderPipeline:
    def __init__(
        self,
        interaction_store: InteractionStore,
        metadata_lookup: SongMetadataLookup,
        user_encoder: UserEncoder,
        ranker: NeuralRanker,
        faiss_recommender: LearnedFaissRecommender,
        song_embeddings: np.ndarray,
        song_id_to_row: Dict[str, int],
        device: str = "cpu"
    ):
        self.interaction_store = interaction_store
        self.metadata_lookup = metadata_lookup
        
        self.user_encoder = user_encoder.to(device)
        self.user_encoder.eval()
        
        self.ranker = ranker.to(device)
        self.ranker.eval()
        
        self.faiss_recommender = faiss_recommender
        self.device = device
        
        self.history_builder = UserHistoryBuilder()
        self.feature_extractor = CandidateFeatureExtractor(
            metadata_lookup, song_embeddings, song_id_to_row
        )
        
    def recommend(self, user_id: str, top_k: int = 10, candidate_pool_size: int = 200) -> List[Dict[str, Any]]:
        """Unified recommendation pipeline."""
        
        # 1. Gather History
        user_history = self.interaction_store.get_user_interactions(user_id)
        
        # We need a PreferenceStore for the user_encoder features
        # (Though in production, this might be cached)
        pref_store = PreferenceStore(self.interaction_store, self.metadata_lookup)
        pref_store.rebuild_all(user_id)
        
        # 2. Build User Features & Encode
        user_features = self._build_user_features(user_id, user_history, pref_store)
        
        import torch
        with torch.no_grad():
            tensor = torch.from_numpy(user_features).float().unsqueeze(0).to(self.device)
            user_embedding = self.user_encoder(tensor).cpu().numpy()[0]
            
        # 3. Retrieve Candidate Pool from FAISS
        # Exclude songs already interacted with
        interacted_ids = list({e.song_id for e in user_history})
        candidates = self.faiss_recommender.recommend_by_embedding(
            user_embedding, top_k=candidate_pool_size, exclude_song_ids=interacted_ids
        )
        
        if not candidates:
            return []
            
        # 4. Extract Features and Re-rank
        ranked_results = []
        features_list = []
        valid_candidates = []
        
        for cand in candidates:
            song_id = cand["track_id"]
            feats = self.feature_extractor.extract_features(
                user_id, user_embedding, song_id, user_history
            )
            if feats is not None:
                features_list.append(feats)
                valid_candidates.append(cand)
                
        if not features_list:
            return []
            
        features_np = np.vstack(features_list)
        
        # 5. Score with NeuralRanker
        scores = self.ranker.predict_score(features_np)
        
        for i, cand in enumerate(valid_candidates):
            cand["ranker_score"] = float(scores[i])
            ranked_results.append(cand)
            
        # Sort descending by ranker_score
        ranked_results.sort(key=lambda x: x["ranker_score"], reverse=True)
        
        return ranked_results[:top_k]

    def _build_user_features(
        self, user_id: str, history: List[UserInteraction], pref_store: PreferenceStore
    ) -> np.ndarray:
        if len(history) == 0:
            return self.history_builder.build_cold_start_features()
            
        song_embs = []
        scores = []
        from src.interactions.ups import calculate_event_score
        
        for event in history:
            row = self.feature_extractor.song_id_to_row.get(event.song_id)
            if row is None:
                continue
                
            song_embs.append(self.feature_extractor.song_embeddings[row])
            scores.append(calculate_event_score(event))
            
        if len(song_embs) == 0:
            return self.history_builder.build_cold_start_features()
            
        song_embs_arr = np.stack(song_embs)
        scores_arr = np.array(scores, dtype=np.float32)
        
        # Get preferences from pref_store
        # We need the top K artist and genre scores. We can extract from _artist_prefs and _genre_prefs.
        artist_prefs_list = [p.preference_score for p in pref_store._artist_prefs.values() if p.user_id == user_id]
        genre_prefs_list = [p.preference_score for p in pref_store._genre_prefs.values() if p.user_id == user_id]
        
        artist_prefs_list.sort(reverse=True)
        genre_prefs_list.sort(reverse=True)
        
        return self.history_builder.build_user_features(
            song_embs_arr, scores_arr, artist_prefs_list, genre_prefs_list
        )
