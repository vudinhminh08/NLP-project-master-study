"""
run_augmentation.py — Entry point cho Week 3 Part 2: Data Augmentation.
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "week1"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "week3"))
sys.path.insert(0, os.path.dirname(__file__))

from utils.helpers import set_seed  # noqa: E402
from llm_client import LLMClient  # noqa: E402
from augment_filter import filter_augmented_reviews  # noqa: E402
from augmentor import run_augmentation_all_aspects  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM Data Augmentation for rare aspects")
    parser.add_argument("--provider", type=str, default="openai", choices=["openai", "gemini"])
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--n_per_aspect", type=int, default=30)
    parser.add_argument("--verify_rate", type=float, default=0.3)
    parser.add_argument("--skip_filter", action="store_true")
    args = parser.parse_args()

    set_seed(42)
    api_key = args.api_key or os.environ.get(
        "OPENAI_API_KEY" if args.provider == "openai" else "GEMINI_API_KEY"
    )
    if not api_key:
        print(f"Missing API key for {args.provider}.")
        return

    train_df = pd.read_csv("data/train_preprocessed.csv")
    client = LLMClient(provider=args.provider, api_key=api_key)

    print("\n" + "=" * 70)
    print("WEEK 3 PART 2: Generating augmented reviews")
    print("=" * 70)
    raw_path = "data/augmented_reviews.json"
    run_augmentation_all_aspects(
        train_df=train_df,
        llm_client=client,
        n_per_aspect=args.n_per_aspect,
        output_path=raw_path,
    )

    if not args.skip_filter:
        print("\n" + "=" * 70)
        print("WEEK 3 PART 2: Filtering generated reviews")
        print("=" * 70)
        filter_augmented_reviews(
            raw_path=raw_path,
            filtered_path="data/augmented_reviews_filtered.json",
            train_df=train_df,
            llm_client=client if args.verify_rate > 0 else None,
            verify_rate=args.verify_rate,
        )

    print("\n" + "=" * 70)
    print("Data augmentation complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()

