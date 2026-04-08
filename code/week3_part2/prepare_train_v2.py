"""
prepare_train_v2.py — Segment augmented_reviews_v2_filtered.json + merge thành train_augmented_v2.csv.

Chạy từ PROJECT ROOT:
    python3 code/week3_part2/prepare_train_v2.py

Yêu cầu:
    - data/augmented_reviews_v2_filtered.json đã tồn tại
    - vncorenlp/ thư mục có VnCoreNLP-1.2.jar + models/
"""

import json
import os
import sys

# Project root = thư mục chứa script này, hai cấp lên
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code", "week1"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code", "week3_part2"))

import pandas as pd
from step3_preprocessing import VnCoreNLPSegmenter, preprocess_text
from utils.constants import ASPECT_COLUMNS, LABEL_TO_IDX

FILTERED_PATH = os.path.join(PROJECT_ROOT, "data", "augmented_reviews_v2_filtered.json")
TRAIN_ORIG    = os.path.join(PROJECT_ROOT, "data", "train_preprocessed.csv")
OUTPUT_PATH   = os.path.join(PROJECT_ROOT, "data", "train_augmented_v2.csv")
VNCORENLP_DIR = os.path.join(PROJECT_ROOT, "vncorenlp")


def main():
    # Load filtered reviews
    with open(FILTERED_PATH, "r", encoding="utf-8") as f:
        filtered = json.load(f)
    print(f"Filtered reviews: {len(filtered)}")

    # Init VnCoreNLP với absolute path — KHÔNG dùng fallback
    segmenter = VnCoreNLPSegmenter(
        vncorenlp_dir=VNCORENLP_DIR,
        use_fallback=False,
    )

    # Segment + build DataFrame
    rows = []
    for i, item in enumerate(filtered):
        if (i + 1) % 20 == 0:
            print(f"  Segmenting {i+1}/{len(filtered)}...")
        processed = preprocess_text(item["review"], segmenter=segmenter, do_segment=True)
        row = {
            "Review": item["review"],
            "processed_review": processed,
        }
        for asp in ASPECT_COLUMNS:
            row[asp] = LABEL_TO_IDX.get(item["labels"].get(asp, "absent"), 0)
        rows.append(row)

    segmenter.close()

    aug_df  = pd.DataFrame(rows)
    orig_df = pd.read_csv(TRAIN_ORIG)

    # Align columns
    aug_df = aug_df.reindex(columns=orig_df.columns, fill_value=0)

    merged = pd.concat([orig_df, aug_df], ignore_index=True)
    merged = merged.sample(frac=1, random_state=42).reset_index(drop=True)
    merged.to_csv(OUTPUT_PATH, index=False)

    print(f"\ntrain_augmented_v2.csv: {merged.shape}")
    print(f"  Original : {len(orig_df)}")
    print(f"  Augmented: {len(aug_df)}")
    print(f"  Saved to : {OUTPUT_PATH}")

    # Quick sanity check
    sample = merged.sample(3, random_state=0)
    print("\nSample processed_review:")
    for _, row in sample.iterrows():
        print(f"  {row['processed_review'][:80]}")


if __name__ == "__main__":
    main()
