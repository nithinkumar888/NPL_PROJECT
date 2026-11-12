"""CLI Query Tool for Telangana Organic Farming Advisory System.

Usage:
    python query.py "my paddy is drying in circles and small brown insects at the bottom of plants"
    python query.py "వరి ఆకులు పసుపు రంగులోకి మారుతున్నాయి" --crop Paddy
    python query.py --interactive
"""

import sys
import argparse
from typing import Optional
from src.advisory_engine import OrganicAdvisoryEngine


def display_advisory(result: dict):
    print("\n" + "=" * 75)
    print(f"🌾 FARMER QUERY: \"{result['query']}\"")
    if result.get("crop_filter"):
        print(f"🔍 Crop Filter: {result['crop_filter']}")
    if result.get("district_filter"):
        print(f"📍 District Filter: {result['district_filter']}")
    print("=" * 75)

    matches = result.get("matches", [])
    if not matches:
        print("No matching organic farming advice found.")
        return

    for rank, m in enumerate(matches, 1):
        crop_disp = f"{m['crop']}"
        if m.get("crop_telugu"):
            crop_disp += f" ({m['crop_telugu']})"

        print(f"\n[{rank}] {crop_disp} — {m['problem_name']} (Category: {m['problem_category']})")
        print(f"    Match Confidence: {m['confidence_percentage']}% (Score: {m['similarity_score']:.3f})")
        if m.get("causal_agent") and m["causal_agent"] != "not specified":
            print(f"    Causal Agent:     {m['causal_agent']}")
        print(f"    Symptoms:         {m['symptoms']}")
        print(f"\n    🌿 ORGANIC SOLUTION:")
        print(f"       {m['organic_solution']}")
        print(f"\n    📋 HOW TO PREPARE & APPLY:")
        print(f"       {m['how_to_apply']}")
        print(f"\n    🛡️ PREVENTIVE PRACTICES:")
        print(f"       {m['preventive_practice']}")

        if m.get("when_to_seek_help"):
            print(f"\n    ⚠️ WHEN TO SEEK HELP / CONTACT:")
            print(f"       {m['when_to_seek_help']}")

        recipes = m.get("related_recipes", [])
        if recipes:
            print(f"\n    🧪 PREPARATION RECIPES:")
            for r in recipes:
                print(f"       • {r.get('name')} ({r.get('category', 'Bio-Input')})")
                print(f"         - Composition: {r.get('composition_or_product', 'N/A')}")
                print(f"         - Preparation: {r.get('method_or_preparation', 'N/A')}")
                print(f"         - Application: {r.get('application', 'N/A')}")
                print(f"         - Cautions:    {r.get('cautions_or_remarks', 'N/A')}")

        practices = m.get("related_agronomic_practices", [])
        if practices:
            print(f"\n    🚜 SUPPORTING AGRONOMIC PRACTICES:")
            for p in practices:
                print(f"       • {p.get('name')} ({p.get('category')}): {p.get('purpose')}")

        print("-" * 75)


def main():
    parser = argparse.ArgumentParser(description="Query the Telangana Organic Farming Advisory System via CLI")
    parser.add_argument("query", nargs="?", default=None, help="Farmer symptom description or question")
    parser.add_argument("--crop", default=None, help="Filter by specific crop (e.g. Paddy, Cotton, Chilli)")
    parser.add_argument("--district", default=None, help="Filter by Telangana district")
    parser.add_argument("--top-k", type=int, default=3, help="Number of recommendations to retrieve (default: 3)")
    parser.add_argument("--interactive", "-i", action="store_true", help="Launch interactive CLI query loop")

    args = parser.parse_args()

    print("Initializing Organic Advisory Engine...")
    engine = OrganicAdvisoryEngine()

    if args.interactive or not args.query:
        print("\n" + "=" * 65)
        print("  Rythu Mitra CLI: Interactive Organic Advisory Terminal")
        print("  Type your crop problem (or 'exit' / 'quit' to stop)")
        print("=" * 65)
        while True:
            try:
                user_q = input("\nEnter crop problem: ").strip()
                if user_q.lower() in ["exit", "quit", "q"]:
                    print("Exiting advisory CLI. Happy organic farming!")
                    break
                if not user_q:
                    continue
                result = engine.get_advisory(user_q, top_k=args.top_k, crop_filter=args.crop, district_filter=args.district)
                display_advisory(result)
            except (KeyboardInterrupt, EOFError):
                break
    else:
        result = engine.get_advisory(args.query, top_k=args.top_k, crop_filter=args.crop, district_filter=args.district)
        display_advisory(result)


if __name__ == "__main__":
    main()

