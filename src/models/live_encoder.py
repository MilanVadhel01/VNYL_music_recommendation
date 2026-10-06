"""
Phase 9: Live New-Song Encoding
===============================

Provides functionality to encode a new song on-the-fly and insert it 
into the FAISS index and metadata store without a full system rebuild.
"""

import faiss
import joblib
import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, List

from src.models.song_encoder import SongEncoder
from src.data.prepare_features import CANDIDATE_FEATURES

class LiveSongEncoder:
    """Handles on-the-fly encoding and indexing of new songs."""
    
    def __init__(
        self,
        model_dir: str,
        scaler_path: str,
        faiss_index_path: str,
        metadata_path: str,
        device: str = "cpu"
    ):
        self.device = device
        self.model = SongEncoder.load(model_dir).to(self.device)
        self.model.eval()
        
        self.scaler = joblib.load(scaler_path)
        self.faiss_index_path = faiss_index_path
        self.metadata_path = metadata_path
        
        self.index = faiss.read_index(faiss_index_path)
        self.meta_df = pd.read_parquet(metadata_path)
        
    def encode_and_insert(self, song_data: Dict[str, Any]) -> str:
        """
        Encode a raw song dict and insert it into the FAISS index.
        
        Expected keys in song_data:
        - track_id, track_name, artist_name
        - all 13 features in CANDIDATE_FEATURES
        
        Returns the track_id.
        """
        # Extract features in exact order
        raw_features = []
        for feat in CANDIDATE_FEATURES:
            raw_features.append(song_data.get(feat, 0.0))
            
        raw_features_np = np.array([raw_features], dtype=np.float32)
        
        # Scale
        scaled_features = self.scale_features(raw_features_np)
        
        # Encode
        embedding = self.encode_features(scaled_features)
        
        # Insert into Metadata
        new_row_idx = len(self.meta_df)
        new_meta = {
            "row_index": new_row_idx,
            "track_id": song_data.get("track_id", f"live_{new_row_idx}"),
            "track_name": song_data.get("track_name", "Unknown"),
            "artist_name": song_data.get("artist_name", "Unknown")
        }
        
        # We append using pd.concat
        new_df = pd.DataFrame([new_meta])
        self.meta_df = pd.concat([self.meta_df, new_df], ignore_index=True)
        self.meta_df.to_parquet(self.metadata_path, index=False)
        
        # Insert into FAISS
        self.index.add(embedding)
        faiss.write_index(self.index, self.faiss_index_path)
        
        return new_meta["track_id"]
        
    def scale_features(self, features: np.ndarray) -> np.ndarray:
        return self.scaler.transform(features).astype(np.float32)
        
    def encode_features(self, scaled_features: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            tensor = torch.from_numpy(scaled_features).to(self.device)
            embedding = self.model(tensor).cpu().numpy()
        return embedding
