"""Evaluation Script for Telangana Organic Farming Advisory System.

Evaluates semantic retrieval accuracy on farmer_query_example queries:
- Top-1 Accuracy: Ground truth problem is the #1 retrieved match
- Top-3 Accuracy: Ground truth problem is within the top-3 retrieved matches
- Mean Reciprocal Rank (MRR@3): Precision-weighted ranking metric
- Category-wise and crop-wise accuracy breakdowns
"""

import time
import json
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
# Standard clean formatter


from src.embedding_indexer import EmbeddingIndexer
from src.advisory_engine import OrganicAdvisoryEngine
from src.utils import load_telangana_dataset


def run_evaluation(output_json: str = "evaluation_results.json") -> Dict[str, Any]:
    print("=" * 70)
    print("  TELANGANA ORGANIC FARMING ADVISORY SYSTEM - EVALUATION BENCHMARK")
    print("=" * 70)

    # 1. Load dataset & initialize engine
    df = load_telangana_dataset()
    total_samples = len(df)
    print(f"Loaded {total_samples} test cases from dataset.")
    print("Initializing Organic Advisory Engine & loading FAISS index...")

    start_init = time.time()
    engine = OrganicAdvisoryEngine()
    print(f"Advisory engine initialized in {time.time() - start_init:.2f}s.\n")

    top1_hits = 0
    top2_hits = 0
    top3_hits = 0
    reciprocal_ranks = []

    category_stats: Dict[str, Dict[str, int]] = {}
    crop_stats: Dict[str, Dict[str, int]] = {}
    detailed_results = []

    print(f"Evaluating {total_samples} farmer queries against FAISS semantic index (top_k=3)...")
    eval_start = time.time()

    for idx, row in df.iterrows():
        expected_id = str(row["id"]).strip()
        expected_problem = str(row["problem_name"]).strip()
        expected_crop = str(row["crop"]).strip()
        category = str(row.get("problem_category", "Unknown")).strip()
        query = str(row["farmer_query_example"]).strip()

        # Initialize category breakdown
        if category not in category_stats:
            category_stats[category] = {"total": 0, "top1": 0, "top3": 0}
        category_stats[category]["total"] += 1

        # Initialize crop breakdown
        if expected_crop not in crop_stats:
            crop_stats[expected_crop] = {"total": 0, "top1": 0, "top3": 0}
        crop_stats[expected_crop]["total"] += 1

        # Query advisory engine
        advisory = engine.get_advisory(farmer_query=query, top_k=3)
        matches = advisory.get("matches", [])
        retrieved_ids = [str(m.get("id")).strip() for m in matches]

        rank = -1
        if expected_id in retrieved_ids:
            rank = retrieved_ids.index(expected_id) + 1

        is_top1 = (rank == 1)
        is_top2 = (rank in [1, 2])
        is_top3 = (rank in [1, 2, 3])

        if is_top1:
            top1_hits += 1
            category_stats[category]["top1"] += 1
            crop_stats[expected_crop]["top1"] += 1
        if is_top2:
            top2_hits += 1
        if is_top3:
            top3_hits += 1
            category_stats[category]["top3"] += 1
            crop_stats[expected_crop]["top3"] += 1

        rr = 1.0 / rank if rank > 0 else 0.0
        reciprocal_ranks.append(rr)

        top_match = matches[0] if matches else {}
        detailed_results.append({
            "index": idx,
            "id": expected_id,
            "crop": expected_crop,
            "problem": expected_problem,
            "category": category,
            "query": query,
            "rank": rank,
            "top1_match": is_top1,
            "top3_match": is_top3,
            "top1_predicted_problem": top_match.get("problem_name", ""),
            "top1_predicted_crop": top_match.get("crop", ""),
            "top1_score": round(top_match.get("similarity_score", 0.0), 4),
            "matched_recipes": [r.get("name") for r in top_match.get("related_recipes", [])]
        })

    eval_duration = time.time() - eval_start
    queries_per_sec = total_samples / eval_duration if eval_duration > 0 else 0

    top1_acc = (top1_hits / total_samples) * 100
    top2_acc = (top2_hits / total_samples) * 100
    top3_acc = (top3_hits / total_samples) * 100
    mrr = sum(reciprocal_ranks) / total_samples

    # Print Summary Metrics
    print("\n" + "=" * 70)
    print("                     OVERALL BENCHMARK RESULTS")
    print("=" * 70)
    print(f" Total Test Queries Evaluated : {total_samples}")
    print(f" Total Evaluation Time        : {eval_duration:.2f} seconds ({queries_per_sec:.1f} queries/sec)")
    print(f" Top-1 Accuracy               : {top1_acc:.2f}% ({top1_hits}/{total_samples})")
    print(f" Top-2 Accuracy               : {top2_acc:.2f}% ({top2_hits}/{total_samples})")
    print(f" Top-3 Accuracy               : {top3_acc:.2f}% ({top3_hits}/{total_samples})")
    print(f" Mean Reciprocal Rank (MRR@3) : {mrr:.4f}")
    print("=" * 70)

    # Print Category-wise Breakdown Table
    print("\n---------------- CATEGORY-WISE PERFORMANCE BREAKDOWN ----------------")
    print(f"{'Category':<24} | {'Total':<6} | {'Top-1 Acc':<11} | {'Top-3 Acc':<11}")
    print("-" * 62)
    for cat, stats in sorted(category_stats.items()):
        c_tot = stats["total"]
        c_t1 = (stats["top1"] / c_tot) * 100 if c_tot else 0
        c_t3 = (stats["top3"] / c_tot) * 100 if c_tot else 0
        print(f"{cat:<24} | {c_tot:<6} | {c_t1:>8.1f}%   | {c_t3:>8.1f}%")
    print("-" * 62)

    # Print Top Crops Breakdown Table
    print("\n---------------- MAJOR CROPS PERFORMANCE (TOP 10) ----------------")
    print(f"{'Crop':<20} | {'Total':<6} | {'Top-1 Acc':<11} | {'Top-3 Acc':<11}")
    print("-" * 58)
    top_crops = sorted(crop_stats.items(), key=lambda x: x[1]["total"], reverse=True)[:10]
    for crop, stats in top_crops:
        cr_tot = stats["total"]
        cr_t1 = (stats["top1"] / cr_tot) * 100 if cr_tot else 0
        cr_t3 = (stats["top3"] / cr_tot) * 100 if cr_tot else 0
        print(f"{crop:<20} | {cr_tot:<6} | {cr_t1:>8.1f}%   | {cr_t3:>8.1f}%")
    print("-" * 58)

    # Print Sample Retrievals
    print("\n---------------- SAMPLE RETRIEVAL DEMONSTRATIONS ----------------")
    for sample in detailed_results[:4]:
        status = "PASSED (Top-1)" if sample["top1_match"] else ("PASSED (Top-3)" if sample["top3_match"] else "MISS")
        print(f"\n[Query] \"{sample['query']}\"")
        print(f"  Target:     {sample['crop']} - {sample['problem']} ({sample['id']})")
        print(f"  Predicted:  {sample['top1_predicted_crop']} - {sample['top1_predicted_problem']} (Score: {sample['top1_score']:.3f})")
        print(f"  Recipes:    {', '.join(sample['matched_recipes'])}")
        print(f"  Result:     {status}")

    # Save to JSON
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_samples": total_samples,
        "evaluation_time_seconds": round(eval_duration, 2),
        "queries_per_second": round(queries_per_sec, 2),
        "metrics": {
            "top1_accuracy": round(top1_acc, 2),
            "top2_accuracy": round(top2_acc, 2),
            "top3_accuracy": round(top3_acc, 2),
            "mean_reciprocal_rank": round(mrr, 4)
        },
        "category_breakdown": {
            cat: {
                "total": s["total"],
                "top1_accuracy": round((s["top1"] / s["total"]) * 100, 2),
                "top3_accuracy": round((s["top3"] / s["total"]) * 100, 2)
            }
            for cat, s in category_stats.items()
        },
        "detailed_results": detailed_results
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[Report saved to {output_json}]")
    print("=" * 70)
    return report


if __name__ == "__main__":
    run_evaluation()
