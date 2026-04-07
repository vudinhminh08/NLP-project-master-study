"""
augmentor.py — LLM-based Data Augmentation cho rare aspects.

FIX v2 (bugs gây ra 240 generated → 40 filtered):
- BUG 1: get_real_examples gọi với random_state=42 cố định ngoài vòng lặp
  → mọi batch dùng cùng examples → LLM sinh cùng 5 reviews lặp đi lặp lại
- FIX 1: Gọi get_real_examples BÊN TRONG batch loop với random_state=batch_idx*17+3
- BUG 2: llm_client.complete(use_cache=True) mặc định → same prompt = same cache key
  → response cached từ batch đầu được trả về cho TẤT CẢ batch sau
- FIX 2: use_cache=False khi generate (cần diverse output, không cần cache)
- FIX 3: Thêm danh sách reviews đã sinh vào prompt → LLM tránh lặp lại
"""

import json
import os
import sys
import time
from typing import Optional

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "week1"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "week3"))

from utils.constants import (  # noqa: E402
    ASPECT_COLUMNS,
    IDX_TO_LABEL,
    RARE_ASPECTS,
    ZERO_TRAIN_ASPECTS,
)
from llm_client import LLMClient  # noqa: E402


ASPECT_DESCRIPTIONS = {
    "FACILITIES#MISCELLANEOUS": "Tiện nghi chung của khách sạn (hồ bơi, gym, spa, sảnh, thang máy, v.v.)",
    "ROOM_AMENITIES#PRICES": "Giá cả các tiện nghi trong phòng (minibar, wifi tính phí, dịch vụ phòng có phí, v.v.)",
    "ROOM_AMENITIES#MISCELLANEOUS": "Các tiện nghi khác trong phòng (remote TV, ổ cắm, móc áo, bàn là, v.v.)",
    "ROOM_AMENITIES#CLEANLINESS": "Vệ sinh tiện nghi phòng (khăn tắm, ga giường, ly cốc, v.v.)",
    "ROOM_AMENITIES#DESIGN&FEATURES": "Thiết kế/tính năng tiện nghi phòng (bồn tắm đẹp, tủ lạnh hiện đại, v.v.)",
    "HOTEL#DESIGN&FEATURES": "Thiết kế/kiến trúc/nội thất tổng thể khách sạn",
    "ROOMS#MISCELLANEOUS": "Các vấn đề khác về phòng (tiếng ồn, côn trùng, mùi, v.v.)",
    "FOOD&DRINKS#MISCELLANEOUS": "Các vấn đề khác về đồ ăn thức uống (thời gian phục vụ, đa dạng, v.v.)",
}


AUGMENTATION_SYSTEM_PROMPT = """Bạn là chuyên gia tạo dữ liệu huấn luyện cho bài toán phân tích đánh giá khách sạn tiếng Việt.

Nhiệm vụ: Viết các review khách sạn tiếng Việt TỰ NHIÊN, giống review thật của khách hàng Việt Nam.

## Quy tắc quan trọng:
1. Review phải tự nhiên, đa dạng phong cách (ngắn gọn / chi tiết / có teencode nhẹ)
2. Mỗi review phải ĐỀ CẬP rõ ràng đến aspect mục tiêu
3. Mỗi review CÓ THỂ đề cập thêm các aspects KHÁC (realistic — review thật thường đề cập nhiều aspect)
4. Sentiment cho target aspect phải rõ ràng (positive / negative / neutral)
5. Độ dài: 1–4 câu, tương đương review thật
6. KHÔNG viết review quá formal hoặc quá giống AI
7. CÓ THỂ dùng teencode nhẹ: "ko" = không, "nv" = nhân viên, "ks" = khách sạn (tùy review)

## 34 Aspect Categories hợp lệ:
{aspect_list}

## Sentiment labels:
- positive: khen, hài lòng
- negative: chê, không hài lòng
- neutral: đề cập nhưng không rõ tốt/xấu
- absent: KHÔNG đề cập → KHÔNG đưa vào output
"""


AUGMENTATION_USER_PROMPT = """Hãy viết {batch_size} review khách sạn tiếng Việt. Mỗi review PHẢI đề cập đến aspect:

**{target_aspect}** — {aspect_description}

Yêu cầu đa dạng:
- {n_positive} review có sentiment POSITIVE cho {target_aspect}
- {n_negative} review có sentiment NEGATIVE cho {target_aspect}
- {n_neutral} review có sentiment NEUTRAL cho {target_aspect}
- Mỗi review có thể đề cập thêm 1-3 aspects KHÁC với sentiment phù hợp

## Ví dụ reviews THẬT từ dataset (để tham khảo phong cách):
{real_examples}

{already_generated_section}

## Format output — JSON array, MỖI phần tử gồm:
```json
[
  {{
    "review": "text review tiếng Việt (chưa qua word segmentation)",
    "labels": {{
      "TARGET_ASPECT#NAME": "positive/negative/neutral",
      "OTHER_ASPECT#NAME": "positive/negative/neutral"
    }}
  }}
]
```

CHỈ trả về JSON array, không giải thích gì thêm.
"""

ALREADY_GENERATED_SECTION = """## Reviews ĐÃ ĐƯỢC VIẾT — KHÔNG lặp lại, tạo nội dung HOÀN TOÀN KHÁC:
{reviews}
"""


def get_real_examples(
    train_df: pd.DataFrame,
    target_aspect: str,
    n_examples: int = 3,
    random_state: Optional[int] = None,  # FIX: caller truyền seed khác nhau mỗi batch
) -> str:
    if target_aspect in ZERO_TRAIN_ASPECTS:
        entity = target_aspect.split("#")[0]
        sibling_aspects = [
            aspect for aspect in ASPECT_COLUMNS
            if aspect.startswith(entity + "#") and aspect != target_aspect
        ]
        mask = train_df[sibling_aspects].max(axis=1) > 0 if sibling_aspects else train_df.index >= 0
    else:
        mask = train_df[target_aspect] > 0

    candidates = train_df[mask]
    if len(candidates) == 0:
        candidates = train_df.sample(min(n_examples, len(train_df)), random_state=random_state)
    else:
        candidates = candidates.sample(min(n_examples, len(candidates)), random_state=random_state)

    examples = []
    for _, row in candidates.iterrows():
        review = row.get("Review", row.get("processed_review", ""))
        labels = {}
        for aspect in ASPECT_COLUMNS:
            value = int(row[aspect])
            if value > 0:
                labels[aspect] = IDX_TO_LABEL[value]
        examples.append(
            f"Review: {review}\nLabels: {json.dumps(labels, ensure_ascii=False)}"
        )
    return "\n\n".join(examples)


def _parse_augmentation_response(response: str, target_aspect: str) -> list[dict]:
    text = response.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > start:
            try:
                data = json.loads(text[start:end])
            except json.JSONDecodeError:
                return []
        else:
            return []

    if not isinstance(data, list):
        return []

    valid = []
    for item in data:
        if not isinstance(item, dict):
            continue
        review = item.get("review")
        labels = item.get("labels")
        if not isinstance(review, str) or len(review.strip()) < 10:
            continue
        if not isinstance(labels, dict):
            continue

        clean_labels = {}
        for aspect, sentiment in labels.items():
            if aspect in ASPECT_COLUMNS and sentiment in {"positive", "negative", "neutral"}:
                clean_labels[aspect] = sentiment
        if target_aspect not in clean_labels:
            continue

        valid.append(
            {
                "review": review.strip(),
                "labels": clean_labels,
                "source_aspect": target_aspect,
            }
        )
    return valid


def generate_augmented_reviews(
    train_df: pd.DataFrame,
    target_aspect: str,
    llm_client: LLMClient,
    n_total: int = 30,
    batch_size: int = 5,
) -> list[dict]:
    """
    FIX v2: mỗi batch dùng random_state khác nhau + use_cache=False + already_generated prompt
    để tránh LLM sinh cùng 5 reviews lặp đi lặp lại qua 6 batch.
    """
    all_reviews = []
    n_batches = (n_total + batch_size - 1) // batch_size
    description = ASPECT_DESCRIPTIONS.get(target_aspect, target_aspect)

    system_prompt = AUGMENTATION_SYSTEM_PROMPT.format(
        aspect_list="\n".join(f"  - {aspect}" for aspect in ASPECT_COLUMNS)
    )

    for batch_idx in range(n_batches):
        remaining = n_total - len(all_reviews)
        current_batch = min(batch_size, remaining)
        n_negative = current_batch // 3
        n_neutral = max(1, current_batch // 6)
        n_positive = max(0, current_batch - n_negative - n_neutral)

        # FIX 1: random_state khác nhau mỗi batch → examples khác nhau
        real_examples = get_real_examples(
            train_df, target_aspect, n_examples=3,
            random_state=batch_idx * 17 + 3,
        )

        # FIX 3: liệt kê reviews đã sinh để LLM tránh lặp lại
        if all_reviews:
            already_texts = "\n".join(
                f"- {r['review'][:120]}" for r in all_reviews[-10:]
            )
            already_section = ALREADY_GENERATED_SECTION.format(reviews=already_texts)
        else:
            already_section = ""

        user_prompt = AUGMENTATION_USER_PROMPT.format(
            batch_size=current_batch,
            target_aspect=target_aspect,
            aspect_description=description,
            n_positive=n_positive,
            n_negative=n_negative,
            n_neutral=n_neutral,
            real_examples=real_examples,
            already_generated_section=already_section,
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            # FIX 2: use_cache=False → tránh cached duplicate response
            response = llm_client.complete(messages, temperature=0.8, use_cache=False)
            parsed = _parse_augmentation_response(response, target_aspect)
            all_reviews.extend(parsed)
            print(
                f"  [{target_aspect}] Batch {batch_idx + 1}/{n_batches}: "
                f"got {len(parsed)} reviews (total: {len(all_reviews)})"
            )
        except Exception as exc:
            print(f"  [{target_aspect}] Batch {batch_idx + 1} FAILED: {exc}")
        time.sleep(1.5)

    return all_reviews[:n_total]


def run_augmentation_all_aspects(
    train_df: pd.DataFrame,
    llm_client: LLMClient,
    n_per_aspect: int = 30,
    output_path: str = "data/augmented_reviews.json",
) -> list[dict]:
    all_augmented = []

    for aspect in RARE_ASPECTS:
        print(f"\n{'=' * 60}")
        print(f"Generating {n_per_aspect} reviews for: {aspect}")
        print(f"{'=' * 60}")
        reviews = generate_augmented_reviews(
            train_df=train_df,
            target_aspect=aspect,
            llm_client=llm_client,
            n_total=n_per_aspect,
        )
        all_augmented.extend(reviews)
        print(f"  -> Got {len(reviews)} valid reviews")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(all_augmented, handle, ensure_ascii=False, indent=2)

    print(f"\nTotal generated: {len(all_augmented)} reviews -> {output_path}")
    return all_augmented

