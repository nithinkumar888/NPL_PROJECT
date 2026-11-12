"""Unit and Integration Tests for Telangana Organic Farming Advisory System."""

import unittest
from pathlib import Path
import pandas as pd

from src.preprocessor import TextPreprocessor, preprocess_text, tokenize_and_lemmatize
from src.recipe_matcher import RecipeMatcher
from src.utils import (
    load_telangana_dataset,
    load_organic_inputs_recipes,
    load_agronomic_practices
)
from src.embedding_indexer import EmbeddingIndexer
from src.advisory_engine import OrganicAdvisoryEngine


class TestPreprocessor(unittest.TestCase):
    """Test text preprocessing, tokenization, and lemmatization."""

    def setUp(self):
        self.preprocessor = TextPreprocessor()

    def test_lowercasing(self):
        text = "Brown Planthopper In PADDY"
        self.assertEqual(self.preprocessor.lowercase(text), "brown planthopper in paddy")

    def test_tokenization_and_units(self):
        text = "Spray NSKE 5% at early damage stage."
        tokens = self.preprocessor.tokenize(text)
        self.assertIn("spray", tokens)
        self.assertIn("nske", tokens)
        self.assertIn("5%", tokens)

    def test_lemmatization(self):
        text = "leaves spots and caterpillars attacking roots"
        cleaned = self.preprocessor.preprocess(text)
        # 'leaves' -> 'leaf', 'spots' -> 'spot', 'caterpillars' -> 'caterpillar', 'roots' -> 'root'
        self.assertIn("leaf", cleaned)
        self.assertIn("spot", cleaned)
        self.assertIn("caterpillar", cleaned)
        self.assertIn("root", cleaned)

    def test_telugu_unicode_preservation(self):
        text = "వరి ఆకులు పసుపుగా మారుతున్నాయి"
        tokens = self.preprocessor.tokenize(text)
        self.assertTrue(len(tokens) >= 4)
        self.assertIn("వరి", tokens)
        self.assertIn("ఆకులు", tokens)


class TestDatasets(unittest.TestCase):
    """Test datasets integrity and schema."""

    def test_telangana_dataset_loaded(self):
        df = load_telangana_dataset()
        self.assertFalse(df.empty)
        self.assertGreaterEqual(len(df), 140)
        expected_cols = [
            "id", "crop", "problem_name", "symptoms",
            "farmer_query_example", "organic_solution",
            "how_to_apply", "preventive_practice"
        ]
        for col in expected_cols:
            self.assertIn(col, df.columns)

    def test_organic_recipes_loaded(self):
        df = load_organic_inputs_recipes()
        self.assertFalse(df.empty)
        self.assertGreaterEqual(len(df), 40)
        self.assertIn("name", df.columns)
        self.assertIn("method_or_preparation", df.columns)

    def test_agronomic_practices_loaded(self):
        df = load_agronomic_practices()
        self.assertFalse(df.empty)
        self.assertGreaterEqual(len(df), 25)
        self.assertIn("name", df.columns)
        self.assertIn("purpose_or_description", df.columns)


class TestRecipeMatcher(unittest.TestCase):
    """Test automated recipe matching logic."""

    def setUp(self):
        self.matcher = RecipeMatcher()

    def test_nske_matching(self):
        solution = "Spray NSKE 5% directed at plant base"
        recipes = self.matcher.match_recipes(solution)
        names = [r["name"] for r in recipes]
        self.assertTrue(any("NSKE" in n for n in names))

    def test_jeevamrutham_matching(self):
        solution = "Apply Jeevamrutham with irrigation water"
        recipes = self.matcher.match_recipes(solution)
        names = [r["name"] for r in recipes]
        self.assertIn("Jeevamrutham", names)

    def test_biopesticide_matching(self):
        solution = "Beauveria bassiana 5 g per litre + Trichoderma"
        recipes = self.matcher.match_recipes(solution)
        names = [r["name"] for r in recipes]
        self.assertTrue(any("Beauveria" in n for n in names))


class TestAdvisoryEngine(unittest.TestCase):
    """Integration test for end-to-end advisory engine."""

    @classmethod
    def setUpClass(cls):
        cls.engine = OrganicAdvisoryEngine()

    def test_top3_advisory_retrieval(self):
        query = "my paddy is drying in circles and small brown insects at the bottom of plants"
        result = self.engine.get_advisory(farmer_query=query, top_k=3)

        self.assertEqual(result["query"], query)
        self.assertEqual(result["results_count"], 3)
        self.assertEqual(len(result["matches"]), 3)

        top_match = result["matches"][0]
        self.assertIn("crop", top_match)
        self.assertIn("problem_name", top_match)
        self.assertIn("organic_solution", top_match)
        self.assertIn("how_to_apply", top_match)
        self.assertIn("preventive_practice", top_match)
        self.assertIn("related_recipes", top_match)
        self.assertIn("similarity_score", top_match)

        # Ground truth check: Brown planthopper in Paddy
        self.assertEqual(top_match["crop"], "Paddy")
        self.assertIn("Brown planthopper", top_match["problem_name"])

    def test_crop_filter(self):
        query = "yellow leaves and fungal spots"
        result = self.engine.get_advisory(farmer_query=query, top_k=3, crop_filter="Cotton")
        for match in result["matches"]:
            self.assertEqual(match["crop"], "Cotton")


if __name__ == "__main__":
    unittest.main()

