"""Embedding and FAISS Indexing Module for Telangana Organic Farming Advisory System.

Uses:
- Sentence-Transformers: 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
- Embedding text: crop + problem_name + symptoms + farmer_query_example (preprocessed)
- FAISS Index: Inner Product (IndexFlatIP) on L2-normalized vectors (Cosine Similarity)
- Retrieval: top-k matches with similarity scores and metadata mapping
"""

import os
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

from src.preprocessor import TextPreprocessor, preprocess_text
from src.utils import load_telangana_dataset


# Ensure HuggingFace loads from local cache without blocking on network queries
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
INDEX_DEFAULT_PATH = Path("data/faiss_index.bin")
METADATA_DEFAULT_PATH = Path("data/index_metadata.pkl")


def load_sentence_transformer(model_name: str = MODEL_NAME) -> SentenceTransformer:
    """Load SentenceTransformer prioritizing cached local weights."""
    try:
        return SentenceTransformer(model_name, local_files_only=True)
    except Exception:
        # Fallback if local_files_only flag causes issues
        return SentenceTransformer(model_name)


class EmbeddingIndexer:
    """Handles text embedding, FAISS index construction, persistence, and semantic retrieval."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        index_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
    ):
        self.model_name = model_name
        self.index_path = Path(index_path) if index_path else INDEX_DEFAULT_PATH
        self.metadata_path = Path(metadata_path) if metadata_path else METADATA_DEFAULT_PATH
        self.preprocessor = TextPreprocessor()
        self.model = load_sentence_transformer(self.model_name)
        self.index: Optional[faiss.IndexFlatIP] = None
        self.metadata: Optional[List[Dict[str, Any]]] = None
        self.df: Optional[pd.DataFrame] = None
        self.embedding_dimension = 384

    @staticmethod
    def construct_composite_text(row: pd.Series) -> str:
        """Construct composite text from crop, problem_name, symptoms, farmer_query_example."""
        crop = str(row.get("crop", "")).strip()
        crop_telugu = str(row.get("crop_telugu", "")).strip()
        problem = str(row.get("problem_name", "")).strip()
        symptoms = str(row.get("symptoms", "")).strip()
        query = str(row.get("farmer_query_example", "")).strip()

        crop_info = f"{crop} {crop_telugu}".strip()
        text = f"{crop_info} {problem} {symptoms} {query}"
        return text

    def build_and_save_index(self, df: Optional[pd.DataFrame] = None, force_rebuild: bool = False):
        """Build FAISS index from dataframe and persist to disk."""
        if df is None:
            df = load_telangana_dataset()
        self.df = df.reset_index(drop=True)

        # Prepare composite texts and preprocess
        composite_texts = []
        for _, row in self.df.iterrows():
            raw_text = self.construct_composite_text(row)
            cleaned_text = self.preprocessor.preprocess(raw_text)
            composite_texts.append(cleaned_text)

        # Compute normalized embeddings
        embeddings = self.model.encode(
            composite_texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True
        ).astype("float32")

        dim = embeddings.shape[1]
        self.embedding_dimension = dim
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

        # Save metadata
        self.metadata = self.df.to_dict(orient="records")

        # Ensure directory exists and save
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))

        with open(self.metadata_path, "wb") as f:
            pickle.dump({
                "metadata": self.metadata,
                "composite_texts": composite_texts,
                "dimension": dim
            }, f)

        # Also copy index to root if building in data/
        try:
            root_index = Path("faiss_index.bin")
            root_meta = Path("index_metadata.pkl")
            if str(self.index_path.resolve()) != str(root_index.resolve()):
                faiss.write_index(self.index, str(root_index))
                with open(root_meta, "wb") as f:
                    pickle.dump({
                        "metadata": self.metadata,
                        "composite_texts": composite_texts,
                        "dimension": dim
                    }, f)
        except Exception:
            pass

    def load_or_build(self, force_rebuild: bool = False):
        """Load index from disk if present, else build it."""
        if not force_rebuild:
            # Check standard paths
            candidate_indexes = [self.index_path, Path("faiss_index.bin"), Path("data/faiss_index.bin")]
            candidate_metas = [self.metadata_path, Path("index_metadata.pkl"), Path("data/index_metadata.pkl")]

            idx_file = next((p for p in candidate_indexes if p.exists()), None)
            meta_file = next((p for p in candidate_metas if p.exists()), None)

            if idx_file and meta_file:
                try:
                    self.index = faiss.read_index(str(idx_file))
                    with open(meta_file, "rb") as f:
                        meta_obj = pickle.load(f)
                        self.metadata = meta_obj["metadata"]
                        self.df = pd.DataFrame(self.metadata)
                    return
                except Exception:
                    pass

        self.build_and_save_index()

    def search(
        self,
        query: str,
        top_k: int = 3,
        crop_filter: Optional[str] = None,
        district_filter: Optional[str] = None,
        season_filter: Optional[str] = None
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Retrieve top_k matching records for query, with robust multi-pass filtering and fallbacks."""
        if self.index is None or self.metadata is None:
            self.load_or_build()

        preprocessed_query = self.preprocessor.preprocess(query)
        if not preprocessed_query.strip():
            # If query stripped to empty, fallback to raw query
            preprocessed_query = query.lower().strip() or "organic farming crop protection"

        query_vector = self.model.encode(
            [preprocessed_query],
            normalize_embeddings=True,
            convert_to_numpy=True
        ).astype("float32")

        # Scan all available metadata so no crop candidate is ever missed
        sample_k = len(self.metadata)
        distances, indices = self.index.search(query_vector, sample_k)

        has_crop_filter = bool(crop_filter and crop_filter != "All Crops")
        has_district_filter = bool(district_filter and "all" not in district_filter.lower())
        has_season_filter = bool(season_filter and "all" not in season_filter.lower())
        
        filter_crop_low = crop_filter.lower().strip() if has_crop_filter else ""
        raw_dist_low = district_filter.split("(")[0].strip().lower() if has_district_filter else ""
        filter_season_low = season_filter.lower().strip() if has_season_filter else ""

        def matches_crop(item: Dict[str, Any]) -> bool:
            if not has_crop_filter:
                return True
            crop_val = str(item.get("crop", "")).lower()
            crop_tel = str(item.get("crop_telugu", "")).lower()
            return crop_val == filter_crop_low or filter_crop_low in crop_val or filter_crop_low in crop_tel

        def matches_district(item: Dict[str, Any]) -> bool:
            if not has_district_filter:
                return True
            districts_str = (str(item.get("major_districts_telangana", "")) + " " + str(item.get("districts_list", ""))).lower()
            if "all districts" in districts_str or "telangana" in districts_str:
                return True
            if raw_dist_low and raw_dist_low in districts_str:
                return True
            # Match tokens (e.g. Asifabad from Kumuram Bheem Asifabad, Gadwal, Sircilla, Kothagudem, etc.)
            for token in raw_dist_low.replace("-", " ").split():
                if len(token) > 3 and token in districts_str:
                    return True
            return False

        def matches_season(item: Dict[str, Any]) -> bool:
            if not has_season_filter:
                return True
            item_season = str(item.get("season", "")).lower()
            if "kharif" in filter_season_low:
                return "kharif" in item_season or "all seasons" in item_season or "perennial" in item_season or "annual" in item_season or "year-round" in item_season
            if "rabi" in filter_season_low:
                return "rabi" in item_season or "all seasons" in item_season or "perennial" in item_season or "annual" in item_season or "year-round" in item_season
            if "summer" in filter_season_low:
                return "summer" in item_season or "all seasons" in item_season or "perennial" in item_season or "annual" in item_season or "year-round" in item_season
            if "perennial" in filter_season_low or "annual" in filter_season_low:
                return "perennial" in item_season or "annual" in item_season or "year-round" in item_season or "all seasons" in item_season
            return filter_season_low in item_season or "all seasons" in item_season

        results: List[Tuple[Dict[str, Any], float]] = []
        seen_ids = set()

        # Pass 1: Strict match (Semantic relevance + Crop + District + Season)
        for score, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            item = self.metadata[idx]
            item_id = item.get("id", idx)
            if item_id in seen_ids:
                continue

            if matches_crop(item) and matches_district(item) and matches_season(item):
                results.append((item, float(score)))
                seen_ids.add(item_id)
                if len(results) >= top_k:
                    break

        # Pass 2: Relax district filter (Crop + Season)
        if len(results) < top_k and (has_district_filter or has_season_filter):
            for score, idx in zip(distances[0], indices[0]):
                if idx < 0 or idx >= len(self.metadata):
                    continue
                item = self.metadata[idx]
                item_id = item.get("id", idx)
                if item_id in seen_ids:
                    continue

                if matches_crop(item) and matches_season(item):
                    results.append((item, float(score)))
                    seen_ids.add(item_id)
                    if len(results) >= top_k:
                        break

        # Pass 3: Relax season filter if needed (Crop only)
        if len(results) < top_k and has_crop_filter:
            for score, idx in zip(distances[0], indices[0]):
                if idx < 0 or idx >= len(self.metadata):
                    continue
                item = self.metadata[idx]
                item_id = item.get("id", idx)
                if item_id in seen_ids:
                    continue

                if matches_crop(item):
                    results.append((item, float(score)))
                    seen_ids.add(item_id)
                    if len(results) >= top_k:
                        break

        # Pass 4: Prevalent issues for this specific crop (ensures crop search never returns empty)
        if len(results) < top_k and has_crop_filter:
            for item in self.metadata:
                item_id = item.get("id")
                if item_id in seen_ids:
                    continue
                if matches_crop(item):
                    results.append((item, 0.72))
                    seen_ids.add(item_id)
                    if len(results) >= top_k:
                        break

        # Pass 5: Universal Organic Practices fallback ("All Crops") if still empty or fewer than top_k
        if len(results) < top_k and (not has_crop_filter or len(results) == 0):
            for item in self.metadata:
                item_id = item.get("id")
                if item_id in seen_ids:
                    continue
                if str(item.get("crop", "")).lower() == "all crops":
                    results.append((item, 0.65))
                    seen_ids.add(item_id)
                    if len(results) >= top_k:
                        break

        # Pass 6: Top FAISS matches fallback if still needed
        if len(results) < top_k and not has_crop_filter:
            for score, idx in zip(distances[0], indices[0]):
                if idx < 0 or idx >= len(self.metadata):
                    continue
                item = self.metadata[idx]
                item_id = item.get("id", idx)
                if item_id in seen_ids:
                    continue
                results.append((item, float(score)))
                seen_ids.add(item_id)
                if len(results) >= top_k:
                    break

        return results
