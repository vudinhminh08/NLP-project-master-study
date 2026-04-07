"""
augment_filter.py — Quality filter cho LLM-generated reviews.
"""

import json
import os
import random
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "week3"))
sys.path.insert(0, os.path.dirname(__file__))

from llm_client import LLMClient  # noqa: E402
from augmentor import ASPECT_DESCRIPTIONS  # noqa: E402


def heuristic_filter(reviews: list[dict], train_df: pd.DataFrame) -> list[dict]:
    train_texts = set(train_df["Review"].astype(str).str.strip().str.lower().tolist())
    vietnamese_chars = set(
        "àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ"
    )

    passed = []
    stats = {"too_short": 0, "too_long": 0, "duplicate": 0, "no_vietnamese": 0}

    for review in reviews:
        text = str(review.get("review", ""))
        lowered = text.strip().lower()
        if len(text) < 15:
            stats["too_short"] += 1
            continue
        if len(text) > 500:
            stats["too_long"] += 1
            continue
        if lowered in train_texts:
            stats["duplicate"] += 1
            continue
        if not any(char in vietnamese_chars for char in lowered):
            stats["no_vietnamese"] += 1
            continue
        passed.append(review)

    print(f"[Heuristic filter] {len(reviews)} -> {len(passed)} (removed: {stats})")
    return passed


VERIFY_PROMPT = """Đọc review khách sạn sau và trả lời: review này có ĐỀ CẬP đến "{aspect}" ({description}) không?

Review: {review}

Trả lời CHỈ "YES" hoặc "NO", không giải thích.
"""


def llm_verify_filter(
    reviews: list[dict],
    llm_client: LLMClient,
    sample_rate: float = 0.3,
) -> list[dict]:
    if not reviews:
        return reviews

    n_verify = max(1, int(len(reviews) * sample_rate))
    verify_indices = set(random.sample(range(len(reviews)), min(n_verify, len(reviews))))

    passed = []
    removed = 0
    for idx, review in enumerate(reviews):
        if idx not in verify_indices:
            passed.append(review)
            continue

        target = review.get("source_aspect", "")
        description = ASPECT_DESCRIPTIONS.get(target, target)
        messages = [
            {
                "role": "user",
                "content": VERIFY_PROMPT.format(
                    aspect=target,
                    description=description,
                    review=review["review"],
                ),
            }
        ]
        try:
            response = llm_client.complete(messages).strip().upper()
            if "NO" in response:
                removed += 1
                continue
        except Exception:
            pass

        passed.append(review)
        time.sleep(0.5)

    print(
        f"[LLM verify filter] Verified {len(verify_indices)}, "
        f"removed {removed} -> {len(passed)} reviews"
    )
    return passed


def dedup_filter(reviews: list[dict], threshold: float = 0.95) -> list[dict]:
    def char_ngrams(text: str, n: int = 3) -> set[str]:
        normalized = text.lower().strip()
        if len(normalized) < n:
            return {normalized} if normalized else set()
        return {normalized[i:i + n] for i in range(len(normalized) - n + 1)}

    passed = []
    existing_ngrams = []
    for review in reviews:
        ngrams = char_ngrams(review.get("review", ""))
        is_dup = False
        for existing in existing_ngrams:
            if not ngrams or not existing:
                continue
            jaccard = len(ngrams & existing) / len(ngrams | existing)
            if jaccard > threshold:
                is_dup = True
                break
        if not is_dup:
            passed.append(review)
            existing_ngrams.append(ngrams)

    print(
        f"[Dedup filter] {len(reviews)} -> {len(passed)} "
        f"(removed {len(reviews) - len(passed)} duplicates)"
    )
    return passed


def filter_augmented_reviews(
    raw_path: str = "data/augmented_reviews.json",
    filtered_path: str = "data/augmented_reviews_filtered.json",
    train_df: pd.DataFrame | None = None,
    llm_client: LLMClient | None = None,
    verify_rate: float = 0.3,
) -> list[dict]:
    with open(raw_path, "r", encoding="utf-8") as handle:
        reviews = json.load(handle)

    print(f"\n{'=' * 60}")
    print(f"Filtering {len(reviews)} generated reviews")
    print(f"{'=' * 60}")

    if train_df is not None:
        reviews = heuristic_filter(reviews, train_df)
    if llm_client is not None and verify_rate > 0:
        reviews = llm_verify_filter(reviews, llm_client, sample_rate=verify_rate)
    reviews = dedup_filter(reviews)

    os.makedirs(os.path.dirname(filtered_path) or ".", exist_ok=True)
    with open(filtered_path, "w", encoding="utf-8") as handle:
        json.dump(reviews, handle, ensure_ascii=False, indent=2)

    print(f"\nFiltered: {len(reviews)} reviews -> {filtered_path}")
    return reviews

