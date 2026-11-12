"""Build / rebuild FAISS index and metadata cache from datasets."""

import time
from src.embedding_indexer import EmbeddingIndexer
from src.utils import load_telangana_dataset


def main():
    print("=" * 60)
    print(" Telangana Organic Farming Advisory System - FAISS Index Builder")
    print("=" * 60)

    start_time = time.time()
    df = load_telangana_dataset()
    print(f"[1/3] Loaded dataset with {len(df)} crop problem records.")

    print("[2/3] Generating SentenceTransformer embeddings and building FAISS index...")
    indexer = EmbeddingIndexer()
    indexer.build_and_save_index(df=df, force_rebuild=True)

    elapsed = time.time() - start_time
    print(f"[3/3] FAISS index built successfully in {elapsed:.2f} seconds!")
    print(f"      - Index Dimension: {indexer.embedding_dimension}")
    print(f"      - Total Indexed Items: {len(df)}")
    print(f"      - Saved to: {indexer.index_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

