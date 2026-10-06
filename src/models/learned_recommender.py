"""
Learned FAISS Recommender (Phase 6)
===================================

Provides a clean interface for recommending songs using a learned user embedding
against the learned FAISS index.
"""

import faiss
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional

class LearnedFaissRecommender:
    """FAISS-based recommender using learned embeddings.
    
    Loads a FAISS index and a metadata index (parquet).
    Given a user embedding, retrieves the Top-K nearest neighbors.
    """
    
    def __init__(self, index_path: str, meta_path: str):
        self.index = faiss.read_index(index_path)
        self.meta_df = pd.read_parquet(meta_path)
        self.dim = self.index.d
        
        # Fast lookup
        self.id_to_meta = self.meta_df.set_index("track_id").to_dict(orient="index")
        
    def recommend_by_embedding(
        self, 
        user_embedding: np.ndarray, 
        top_k: int = 10,
        exclude_song_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve top_k candidates for a given user embedding.
        
        Parameters
        ----------
        user_embedding : ndarray of shape (dim,)
        top_k : int
        exclude_song_ids : list of str, optional
            IDs to exclude from results (e.g. songs already interacted with)
            
        Returns
        -------
        list of dicts containing track_id, track_name, artist_name, and score
        """
        if user_embedding.ndim == 1:
            user_embedding = user_embedding.reshape(1, -1)
            
        user_embedding = user_embedding.astype(np.float32)
        
        # We query more if we need to exclude some
        k_query = top_k
        if exclude_song_ids:
            k_query = min(top_k + len(exclude_song_ids), self.index.ntotal)
            
        scores, I = self.index.search(user_embedding, k_query)
        
        scores = scores[0]
        indices = I[0]
        
        exclude_set = set(exclude_song_ids) if exclude_song_ids else set()
        
        results = []
        for idx, score in zip(indices, scores):
            if idx < 0 or idx >= len(self.meta_df):
                continue
                
            track_id = self.meta_df.iloc[idx]["track_id"]
            if track_id in exclude_set:
                continue
                
            results.append({
                "track_id": track_id,
                "track_name": self.meta_df.iloc[idx]["track_name"],
                "artist_name": self.meta_df.iloc[idx]["artist_name"],
                "score": float(score),
                "row_index": idx
            })
            
            if len(results) == top_k:
                break
                
        return results
