"""Utility functions and dataset loaders for Telangana Organic Farming Advisory System."""

import os
from pathlib import Path
from typing import Optional
import pandas as pd


# Potential root directories to search for datasets
DATA_SEARCH_PATHS = [
    Path("data"),
    Path("."),
    Path(__file__).resolve().parent.parent / "data",
    Path(__file__).resolve().parent.parent,
]


def resolve_file_path(filename: str, fallback_filenames: Optional[list] = None) -> str:
    """Locate a dataset file across standard project directories."""
    all_names = [filename] + (fallback_filenames or [])
    for base_dir in DATA_SEARCH_PATHS:
        for name in all_names:
            candidate = base_dir / name
            if candidate.exists() and candidate.is_file():
                return str(candidate.resolve())
    raise FileNotFoundError(f"Could not locate file '{filename}' in search paths: {DATA_SEARCH_PATHS}")


def load_telangana_dataset(filepath: Optional[str] = None) -> pd.DataFrame:
    """Load the Telangana organic farming crop problem & solution dataset."""
    path = filepath or resolve_file_path(
        "telangana_organic_farming_dataset.csv",
        ["telangana_organic_cleaned.csv"]
    )
    df = pd.read_csv(path, encoding="utf-8")
    return df


def load_organic_inputs_recipes(filepath: Optional[str] = None) -> pd.DataFrame:
    """Load organic input preparation recipes (botanicals, fermented concoctions, biopesticides)."""
    try:
        path = filepath or resolve_file_path("organic_inputs_recipes.csv")
        df = pd.read_csv(path, encoding="utf-8")
    except FileNotFoundError:
        # Fallback: extract from agronomy_cleaned.csv if present
        agro_path = resolve_file_path("agronomy_cleaned.csv", ["agronomic_practices.csv"])
        agro_df = pd.read_csv(agro_path, encoding="utf-8")
        df = agro_df[agro_df["record_type"] == "Organic Input"].copy()
    return df


def load_agronomic_practices(filepath: Optional[str] = None) -> pd.DataFrame:
    """Load preventive agronomic practices."""
    try:
        path = filepath or resolve_file_path("agronomic_practices.csv")
        df = pd.read_csv(path, encoding="utf-8")
    except FileNotFoundError:
        agro_path = resolve_file_path("agronomy_cleaned.csv")
        agro_df = pd.read_csv(agro_path, encoding="utf-8")
        df = agro_df[agro_df["record_type"] == "Agronomic Practice"].copy()
    return df

