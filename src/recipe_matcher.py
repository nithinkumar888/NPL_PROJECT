"""Recipe Matcher Module for Telangana Organic Farming Advisory System.

Links organic solutions and applications recommended in problem diagnostics
to detailed preparation recipes from organic_inputs_recipes.csv.
"""

import re
from typing import List, Dict, Any, Optional
import pandas as pd
from src.utils import load_organic_inputs_recipes


class RecipeMatcher:
    """Matches organic treatments from crop diagnosis to exact input preparation recipes."""

    RECIPE_KEYWORDS = [
        ("Neem Seed Kernel Extract (NSKE) 5%", [r"\bnske\b", r"neem seed kernel", r"kernel extract"]),
        ("Neem oil spray", [r"neem oil"]),
        ("Jeevamrutham", [r"jeevamruth", r"jeevamrut"]),
        ("Beejamrutham", [r"beejamruth", r"beejamrut"]),
        ("Panchagavya", [r"panchagavya", r"panchakavya"]),
        ("Agniastram", [r"agniastr", r"agni astram"]),
        ("Trichoderma viride", [r"trichoderma"]),
        ("Pseudomonas fluorescens", [r"pseudomonas"]),
        ("HaNPV / SlNPV", [r"\bhanpv\b", r"\bslnpv\b", r"\bnpv\b"]),
        ("Vermicompost", [r"vermicompost", r"earthworm"]),
        ("Green manure (dhaincha / sunhemp)", [r"green manure", r"dhaincha", r"sunhemp", r"sunnhemp"]),
        ("Azolla", [r"azolla"]),
        ("Yellow and blue sticky traps", [r"sticky trap", r"yellow trap", r"blue trap", r"sticky sheet"]),
        ("Pheromone traps", [r"pheromone", r"lure"]),
        ("Neemastram", [r"neemastr", r"neem astram"]),
        ("Brahmastram", [r"brahmastr", r"brahma astram"]),
        ("Dashaparni ark", [r"dashaparni", r"dashparni"]),
        ("Sour buttermilk spray", [r"buttermilk", r"sour milk", r"curd spray"]),
        ("Fish amino acid", [r"fish amino"]),
        ("Vermiwash", [r"vermiwash"]),
        ("Ghanajeevamrutham", [r"ghanajeevamruth", r"ghana jeevamrut"]),
        ("NADEP / pit compost", [r"nadep", r"pit compost", r"\bcompost\b", r"\bfym\b", r"farmyard manure"]),
        ("Biogas slurry", [r"biogas slurry", r"bio-gas slurry", r"\bslurry\b"]),
        ("Neem cake", [r"neem cake"]),
        ("Karanj (pongamia) cake", [r"karanj", r"pongamia"]),
        ("Bacillus thuringiensis (Bt)", [r"\bbt\b", r"bacillus thuringiensis", r"dipel"]),
        ("Metarhizium anisopliae", [r"metarhizium"]),
        ("Beauveria bassiana", [r"beauveria"]),
        ("Verticillium (Lecanicillium) lecanii", [r"verticillium", r"lecanicillium"]),
        ("Paecilomyces lilacinus", [r"paecilomyces"]),
        ("Trichogramma egg parasitoid cards", [r"trichogramma", r"parasitoid", r"tricho card"]),
        ("Rhizobium / Bradyrhizobium", [r"rhizobium", r"bradyrhizobium"]),
        ("Azotobacter / Azospirillum", [r"azotobacter", r"azospirillum"]),
        ("Phosphate solubilizing bacteria (PSB)", [r"\bpsb\b", r"phosphate solubiliz"]),
        ("Potassium mobilizing bacteria (KMB)", [r"\bkmb\b", r"potassium mobiliz"]),
        ("VAM mycorrhiza", [r"\bvam\b", r"mycorrhiza"]),
        ("Cow urine spray", [r"cow urine"]),
        ("Cow dung extract 20%", [r"cow dung extract", r"fresh cow dung"]),
        ("Wood ash", [r"wood ash", r"\bash\b"]),
        ("Gypsum", [r"gypsum"]),
        ("Rock phosphate", [r"rock phosphate"]),
        ("Wettable sulphur", [r"wettable sulphur", r"wettable sulfur", r"sulphur", r"sulfur"]),
        ("Bordeaux mixture 1%", [r"bordeaux"]),
    ]

    def __init__(self, recipes_df: Optional[pd.DataFrame] = None):
        if recipes_df is None:
            recipes_df = load_organic_inputs_recipes()
        self.recipes_df = recipes_df
        # Index dataframe rows by standardized name
        self.recipe_by_name: Dict[str, Dict[str, Any]] = {}
        for _, row in self.recipes_df.iterrows():
            name = str(row.get("name", "")).strip()
            if name:
                self.recipe_by_name[name.lower()] = row.to_dict()

    def get_recipe_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Find a recipe by exact or normalized name."""
        return self.recipe_by_name.get(name.lower().strip())

    def match_recipes(
        self,
        organic_solution: str,
        how_to_apply: str = "",
        max_recipes: int = 3
    ) -> List[Dict[str, Any]]:
        """Identify matching organic recipes from solution and application text."""
        combined_text = f"{organic_solution or ''} {how_to_apply or ''}".lower()
        matched_recipes: List[Dict[str, Any]] = []
        matched_names = set()

        # 1. Regex pattern matching
        for recipe_name, patterns in self.RECIPE_KEYWORDS:
            for pattern in patterns:
                if re.search(pattern, combined_text, re.IGNORECASE):
                    recipe_dict = self.get_recipe_by_name(recipe_name)
                    if recipe_dict and recipe_name not in matched_names:
                        matched_recipes.append(recipe_dict)
                        matched_names.add(recipe_name)
                    break

        # 2. Token overlap fallback if direct regex didn't catch anything
        if not matched_recipes:
            for _, row in self.recipes_df.iterrows():
                r_name = str(row.get("name", "")).lower()
                r_tokens = [t for t in re.findall(r"\w+", r_name) if len(t) > 3]
                if r_tokens and any(t in combined_text for t in r_tokens):
                    r_dict = row.to_dict()
                    matched_recipes.append(r_dict)
                    matched_names.add(row.get("name"))
                    if len(matched_recipes) >= max_recipes:
                        break

        # 3. Default fallback to Jeevamrutham (the universal organic microbial booster)
        if not matched_recipes:
            jeeva = self.get_recipe_by_name("Jeevamrutham")
            if jeeva:
                matched_recipes.append(jeeva)

        return matched_recipes[:max_recipes]

