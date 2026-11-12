"""Organic Advisory Engine for Telangana Farmers.

Coordinates:
- Semantic NLP retrieval of top-3 matching crop issues via FAISS & SentenceTransformers
- Extraction of organic_solution, how_to_apply, preventive_practice, and escalation guidelines
- Mapping and inclusion of detailed preparation recipes from organic_inputs_recipes.csv
- Related agronomic practices linking
"""

from typing import List, Dict, Any, Optional
import pandas as pd
from src.embedding_indexer import EmbeddingIndexer
from src.recipe_matcher import RecipeMatcher
from src.utils import load_agronomic_practices


class OrganicAdvisoryEngine:
    """End-to-end organic farming advisory coordinator."""

    def __init__(
        self,
        indexer: Optional[EmbeddingIndexer] = None,
        recipe_matcher: Optional[RecipeMatcher] = None
    ):
        self.indexer = indexer or EmbeddingIndexer()
        self.indexer.load_or_build()
        self.recipe_matcher = recipe_matcher or RecipeMatcher()
        try:
            self.practices_df = load_agronomic_practices()
        except Exception:
            self.practices_df = pd.DataFrame()

    def _format_practice_row(self, row: Any) -> Dict[str, Any]:
        """Format a practice row into a standard dictionary."""
        return {
            "name": row.get("name", ""),
            "category": row.get("category", ""),
            "purpose": row.get("purpose_or_description", ""),
            "method": row.get("method_or_preparation", ""),
            "benefits": row.get("benefits", ""),
            "suitable_crops": row.get("suitable_crops", ""),
            "cautions": row.get("cautions_or_remarks", "")
        }

    def get_crop_agronomic_practices(
        self,
        crop: str,
        max_practices: int = 4
    ) -> List[Dict[str, Any]]:
        """Retrieve suitable agronomic practices for any crop from agronomic_practices.csv."""
        if self.practices_df.empty:
            return []

        crop_low = (crop or "").lower().strip()
        matched = []
        seen_names = set()

        # Crop category groupings in Telangana
        is_veg = crop_low in ["tomato", "chilli", "brinjal", "okra", "cabbage and cauliflower", "onion", "cucurbits", "vegetables"]
        is_orchard = crop_low in ["mango", "guava", "citrus", "banana", "papaya", "pomegranate", "oil palm", "orchards"]
        is_pulse = crop_low in ["red gram", "bengal gram", "green gram", "black gram", "pulses", "chickpea"]
        is_cereal = crop_low in ["paddy", "maize", "sorghum", "pearl millet", "cereals"]
        is_oilseed = crop_low in ["groundnut", "soybean", "sesame", "sunflower", "castor", "safflower", "oilseeds"]

        # Pass 1: Direct crop mention in suitable_crops
        for _, row in self.practices_df.iterrows():
            suitable = str(row.get("suitable_crops", "")).lower()
            name = str(row.get("name", ""))
            if name in seen_names:
                continue

            if crop_low and crop_low != "all crops" and crop_low in suitable:
                matched.append(self._format_practice_row(row))
                seen_names.add(name)
                if len(matched) >= max_practices:
                    break

        # Pass 2: Category matching (vegetables, orchards, pulses, cereals, oilseeds)
        if len(matched) < max_practices:
            for _, row in self.practices_df.iterrows():
                suitable = str(row.get("suitable_crops", "")).lower()
                name = str(row.get("name", ""))
                if name in seen_names:
                    continue

                cat_match = (
                    (is_veg and "vegetable" in suitable) or
                    (is_orchard and "orchard" in suitable) or
                    (is_pulse and "pulse" in suitable) or
                    (is_cereal and "cereal" in suitable) or
                    (is_oilseed and "oilseed" in suitable)
                )
                if cat_match:
                    matched.append(self._format_practice_row(row))
                    seen_names.add(name)
                    if len(matched) >= max_practices:
                        break

        # Pass 3: Universal organic practices ('all crops', 'all')
        if len(matched) < max_practices:
            for _, row in self.practices_df.iterrows():
                suitable = str(row.get("suitable_crops", "")).lower()
                name = str(row.get("name", ""))
                if name in seen_names:
                    continue

                if "all crops" in suitable or suitable == "all" or "all major crops" in suitable:
                    matched.append(self._format_practice_row(row))
                    seen_names.add(name)
                    if len(matched) >= max_practices:
                        break

        return matched

    def get_related_agronomic_practices(
        self,
        crop: str,
        category: str,
        max_practices: int = 2
    ) -> List[Dict[str, Any]]:
        """Find relevant agronomic practices from agronomic_practices.csv."""
        if self.practices_df.empty:
            return []

        matched = []
        crop_low = crop.lower()
        cat_low = category.lower()

        for _, row in self.practices_df.iterrows():
            suitable = str(row.get("suitable_crops", "")).lower()
            p_cat = str(row.get("category", "")).lower()

            is_crop_match = crop_low in suitable or "all crops" in suitable or suitable == "all"
            is_cat_match = cat_low in p_cat or (category == "Pest" and "pest" in p_cat) or (category == "Disease" and "disease" in p_cat)

            if is_crop_match or is_cat_match:
                matched.append(self._format_practice_row(row))
            if len(matched) >= max_practices:
                break

        # Fallback to crop practices if no category match
        if not matched:
            matched = self.get_crop_agronomic_practices(crop, max_practices=max_practices)

        return matched

    def get_advisory(
        self,
        farmer_query: str,
        top_k: int = 3,
        crop_filter: Optional[str] = None,
        district_filter: Optional[str] = None,
        season_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate structured advisory for a farmer's natural language query."""
        search_results = self.indexer.search(
            query=farmer_query,
            top_k=top_k,
            crop_filter=crop_filter,
            district_filter=district_filter,
            season_filter=season_filter
        )

        matches = []
        for item, score in search_results:
            sol = item.get("organic_solution", "")
            app = item.get("how_to_apply", "")
            prev = item.get("preventive_practice", "")
            crop = item.get("crop", "")
            cat = item.get("problem_category", "")

            # Match preparation recipes from organic_inputs_recipes.csv
            recipes = self.recipe_matcher.match_recipes(
                organic_solution=sol,
                how_to_apply=app,
                max_recipes=2
            )

            # Match agronomic preventive practices
            practices = self.get_related_agronomic_practices(crop, cat, max_practices=1)

            matches.append({
                "id": item.get("id", ""),
                "crop": crop,
                "crop_telugu": item.get("crop_telugu", ""),
                "season": item.get("season", ""),
                "problem_name": item.get("problem_name", ""),
                "causal_agent": item.get("causal_agent", ""),
                "problem_category": cat,
                "symptoms": item.get("symptoms", ""),
                "farmer_query_example": item.get("farmer_query_example", ""),
                "organic_solution": sol,
                "how_to_apply": app,
                "preventive_practice": prev,
                "when_to_seek_help": item.get("when_to_seek_help", ""),
                "major_districts": item.get("major_districts_telangana", ""),
                "similarity_score": score,
                "confidence_percentage": round(min(max(score, 0.0), 1.0) * 100, 1),
                "related_recipes": recipes,
                "related_agronomic_practices": practices
            })

        crop_for_practices = crop_filter if crop_filter and crop_filter != "All Crops" else "All Crops"
        crop_agronomic_practices = self.get_crop_agronomic_practices(crop_for_practices, max_practices=4)

        return {
            "query": farmer_query,
            "crop_filter": crop_filter,
            "district_filter": district_filter,
            "season_filter": season_filter,
            "top_k": top_k,
            "results_count": len(matches),
            "matches": matches,
            "crop_agronomic_practices": crop_agronomic_practices
        }

