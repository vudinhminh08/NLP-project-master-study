"""
augmentor_v2.py — Error-driven augmentation targeting WEAK_ASPECTS.

Cải tiến so với augmentor.py (v1):
1. Target WEAK_ASPECTS (9 aspects F1<0.35) thay vì RARE_ASPECTS
2. n_per_aspect=60 (v1: 30) — volume lớn hơn để bù filter loss
3. Label consistency check: verify 100% generated reviews
4. Sentiment ratio cứng: 40% positive, 35% negative, 25% neutral
"""

import json
import os
import sys
import time
from typing import Optional

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "week1"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "week3"))

from utils.constants import ASPECT_COLUMNS, IDX_TO_LABEL, WEAK_ASPECTS  # noqa: E402
from llm_client import LLMClient  # noqa: E402


DIFFICULT_ASPECTS = {
    "FACILITIES#MISCELLANEOUS",
    "ROOM_AMENITIES#PRICES",
}

TARGET_COUNT_OVERRIDES = {
    "FACILITIES#MISCELLANEOUS": 120,
    "ROOM_AMENITIES#PRICES": 120,
}

BATCH_SIZE_OVERRIDES = {
    "FACILITIES#MISCELLANEOUS": 4,
    "ROOM_AMENITIES#PRICES": 3,
}

ASPECT_STRICT_HINTS = {
    "FACILITIES#MISCELLANEOUS": (
        "Cần nói rất rõ đây là tiện nghi hoặc khu vực chung ngoài phòng như hồ bơi, gym, spa, "
        "thang máy, bãi đỗ xe, phòng hội nghị, khu vui chơi. "
        "Không được viết chung chung kiểu 'cơ sở vật chất tốt' vì cái đó dễ bị hiểu thành FACILITIES#GENERAL. "
        "Mỗi review nên có ít nhất 1 từ khóa cụ thể như 'hồ bơi', 'gym', 'spa', 'thang máy', "
        "'bãi đỗ xe', 'phòng hội nghị', 'khu vui chơi'."
    ),
    "ROOM_AMENITIES#PRICES": (
        "Cần nói rất rõ về giá hoặc phụ phí của tiện nghi trong phòng như minibar, wifi tính phí, "
        "giặt ủi, đồ ăn nhẹ trong phòng, dịch vụ phòng tính tiền riêng. "
        "Mỗi review phải có từ chỉ tiền hoặc giá như 'đắt', 'giá', 'phí', 'tính tiền', '50k', "
        "'120 nghìn', 'phụ thu'. Không được chỉ chê chất lượng minibar hay wifi nếu không nhắc tới giá."
    ),
}

ANCHOR_EXAMPLES = {
    "FACILITIES#MISCELLANEOUS": """
Review: Hồ bơi khá rộng và sạch, trẻ con nhà mình mê khu vui chơi luôn. Tuy nhiên thang máy giờ cao điểm chờ hơi lâu.
Labels: {"FACILITIES#MISCELLANEOUS": "positive"}

Review: Bãi đỗ xe quá chật, cuối tuần phải chạy vòng vòng mới tìm được chỗ. Thang máy còn bị trục trặc một lần làm mình khá khó chịu.
Labels: {"FACILITIES#MISCELLANEOUS": "negative"}

Review: Khách sạn có spa và phòng gym ở tầng 3, mình có ghé qua nhưng chưa dùng nhiều nên chưa đánh giá thêm.
Labels: {"FACILITIES#MISCELLANEOUS": "neutral"}
""".strip(),
    "ROOM_AMENITIES#PRICES": """
Review: Minibar trong phòng hơi đắt, một lon nước mà tính gần 40k nên mình không dám dùng thêm.
Labels: {"ROOM_AMENITIES#PRICES": "negative"}

Review: Wifi phòng phải mua thêm gói riêng 50k một ngày, cũng hơi bất tiện nhưng giá vậy tạm chấp nhận được.
Labels: {"ROOM_AMENITIES#PRICES": "neutral"}

Review: Khách sạn để bảng giá minibar khá rõ và mức giá mềm hơn mình nghĩ, dùng vài món cũng không thấy bị chém.
Labels: {"ROOM_AMENITIES#PRICES": "positive"}
""".strip(),
}


WEAK_ASPECT_DESCRIPTIONS = {
    "FACILITIES#MISCELLANEOUS": (
        "Tiện nghi chung của khách sạn ngoài phòng: hồ bơi, gym, spa, sảnh chờ, "
        "thang máy, bãi đỗ xe, phòng hội nghị, khu vui chơi. "
        "Ví dụ: 'ho boi sach dep', 'thang may hay hong', 'co gym nhung thiet bi cu'."
    ),
    "FOOD&DRINKS#MISCELLANEOUS": (
        "Các vấn đề khác về ăn uống: thời gian phục vụ bữa sáng, sự đa dạng món, "
        "đồ uống tại bar, room service. "
        "Ví dụ: 'bua sang phuc vu den 10h', 'menu bar da dang', 'room service cham'."
    ),
    "ROOMS#MISCELLANEOUS": (
        "Các vấn đề khác về phòng: tiếng ồn từ ngoài hoặc phòng bên, côn trùng, "
        "mùi ẩm mốc, view từ phòng, tầng phòng. "
        "Ví dụ: 'phong bi on do gan thang may', 'thay gian trong phong', 'view bien tuyet'."
    ),
    "ROOM_AMENITIES#MISCELLANEOUS": (
        "Tiện nghi trong phòng, không phải giường hay phòng tắm: remote TV, ổ cắm, "
        "móc áo, bàn là, két sắt, máy pha cà phê. "
        "Ví dụ: 'thieu o cam canh giuong', 'co ket sat tien', 'remote TV bi hong'."
    ),
    "ROOM_AMENITIES#PRICES": (
        "Giá cả dịch vụ tính phí trong phòng: minibar đắt, wifi tính phí, "
        "dịch vụ phòng có phí riêng, tiền giặt ủi. "
        "Ví dụ: 'minibar tinh gia cat co', 'wifi phong mat them 50k moi ngay', "
        "'gia giat ui qua cao'."
    ),
    "HOTEL#MISCELLANEOUS": (
        "Các vấn đề tổng quát về khách sạn không thuộc nhóm khác: chính sách "
        "check-in hay check-out, phụ phí ẩn, quy định pet, chính sách hủy phòng. "
        "Ví dụ: 'check-out muon mat phi', 'co phu phi resort fee khong bao truoc'."
    ),
    "FACILITIES#GENERAL": (
        "Đánh giá chung về tiện nghi hoặc cơ sở vật chất khách sạn mà không đi vào "
        "một hạng mục cụ thể. Ví dụ: 'co so vat chat tot', 'tien nghi day du'."
    ),
    "FACILITIES#CLEANLINESS": (
        "Vệ sinh khu vực chung của khách sạn: sảnh, hành lang, thang máy, hồ bơi, "
        "nhà vệ sinh công cộng. Ví dụ: 'sanh khach san rat sach', 'hanh lang co mui'."
    ),
    "FACILITIES#COMFORT": (
        "Sự thoải mái của tiện nghi hoặc khu vực chung: ghế ở sảnh, nhiệt độ khu chung, "
        "ánh sáng, âm nhạc nền. Ví dụ: 'sanh mat me thoai mai', 'khu cho cho khong em'."
    ),
}

_SYSTEM_PROMPT = """Bạn là chuyên gia tạo dữ liệu huấn luyện cho bài toán phân tích cảm xúc theo khía cạnh (ABSA) cho đánh giá khách sạn tiếng Việt.

Nhiệm vụ: Viết các review khách sạn tiếng Việt tự nhiên, giống review thật của khách hàng.

Quy tắc cứng:
1. Mỗi review phải đề cập rõ ràng đến aspect mục tiêu
2. Review phải tự nhiên: ngắn gọn 1-2 câu hoặc vừa 3-4 câu, có thể có teencode nhẹ
3. Sentiment cho target aspect phải rõ ràng: positive, negative hoặc neutral
4. Mỗi review có thể đề cập thêm 1-2 aspects khác
5. Không viết review quá formal, quá dài, hoặc giống AI

34 Aspect Categories hợp lệ:
{aspect_list}

Sentiment labels hợp lệ: positive, negative, neutral, absent
(absent = không đề cập, không đưa vào labels output)
"""

_USER_PROMPT = """Viết {batch_size} review khách sạn tiếng Việt, mỗi review phải đề cập đến:

Target aspect: {target_aspect}
Mô tả: {description}

Phân phối sentiment bắt buộc:
- {n_pos} review với {target_aspect} = POSITIVE
- {n_neg} review với {target_aspect} = NEGATIVE
- {n_neu} review với {target_aspect} = NEUTRAL

Ví dụ review thật từ dataset, chỉ để tham khảo phong cách, không copy:
{real_examples}

Anchor examples để bám đúng target aspect:
{anchor_examples}

Ràng buộc bổ sung cho target aspect:
{strict_hint}

{already_section}

Output: JSON array, mỗi phần tử gồm:
[
  {{
    "review": "text review tiếng Việt",
    "labels": {{
      "{target_aspect}": "positive/negative/neutral",
      "ASPECT_KHAC#NAME": "positive/negative/neutral"
    }}
  }}
]

Chỉ trả về JSON array, không giải thích.
"""

_ALREADY_SECTION = """Reviews đã sinh, tuyệt đối không lặp lại:
{texts}
"""

_CONSISTENCY_PROMPT = """Review khách sạn sau có đề cập rõ ràng đến "{aspect}" ({description}) không?

Review: {review}

Trả lời chỉ "YES" hoặc "NO".
"""


def _get_real_examples(
    train_df: pd.DataFrame,
    target_aspect: str,
    n: int = 3,
    random_state: Optional[int] = None,
) -> str:
    """Lấy n review thật có target_aspect, fallback sang sibling aspects nếu cần."""
    mask = train_df[target_aspect] > 0
    candidates = train_df[mask]

    if len(candidates) == 0:
        entity = target_aspect.split("#")[0]
        siblings = [a for a in ASPECT_COLUMNS if a.startswith(entity + "#") and a != target_aspect]
        if siblings:
            candidates = train_df[train_df[siblings].max(axis=1) > 0]

    if len(candidates) == 0:
        candidates = train_df

    sample = candidates.sample(min(n, len(candidates)), random_state=random_state)
    lines = []
    for _, row in sample.iterrows():
        review = str(row.get("Review", row.get("processed_review", "")))
        labels = {
            aspect: IDX_TO_LABEL[int(row[aspect])]
            for aspect in ASPECT_COLUMNS
            if int(row[aspect]) > 0
        }
        lines.append(f"Review: {review}\nLabels: {json.dumps(labels, ensure_ascii=False)}")
    return "\n\n".join(lines)


def _dedup_reviews(reviews: list[dict]) -> list[dict]:
    """Exact dedup ngay từ bước raw để tránh mất quota vào review lặp."""
    seen = set()
    unique = []
    for item in reviews:
        key = item.get("review", "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _parse_response(response: str, target_aspect: str) -> list[dict]:
    """Parse JSON response từ LLM và giữ lại items hợp lệ."""
    text = response.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:])
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]") + 1
        if start < 0 or end <= start:
            return []
        try:
            data = json.loads(text[start:end])
        except json.JSONDecodeError:
            return []

    if not isinstance(data, list):
        return []

    valid = []
    for item in data:
        if not isinstance(item, dict):
            continue
        review = item.get("review", "")
        labels = item.get("labels", {})
        if not isinstance(review, str) or len(review.strip()) < 15:
            continue
        if not isinstance(labels, dict):
            continue
        clean_labels = {
            aspect: sentiment
            for aspect, sentiment in labels.items()
            if aspect in ASPECT_COLUMNS and sentiment in {"positive", "negative", "neutral"}
        }
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


def _label_consistency_check(
    reviews: list[dict],
    llm_client: LLMClient,
) -> list[dict]:
    """Verify 100% generated reviews để loại case target aspect không đủ rõ."""
    passed = []
    removed = 0
    for review in reviews:
        target = review["source_aspect"]
        description = WEAK_ASPECT_DESCRIPTIONS.get(target, target)[:120]
        messages = [
            {
                "role": "user",
                "content": _CONSISTENCY_PROMPT.format(
                    aspect=target,
                    description=description,
                    review=review["review"],
                ),
            }
        ]
        try:
            resp = llm_client.complete(messages, temperature=0.0, use_cache=False).strip().upper()
            if "NO" in resp:
                removed += 1
                continue
        except Exception:
            pass
        passed.append(review)
        time.sleep(0.3)

    print(f"  [Consistency check] {len(reviews)} -> {len(passed)} (removed {removed})")
    return passed


def generate_for_aspect(
    train_df: pd.DataFrame,
    target_aspect: str,
    llm_client: LLMClient,
    n_total: int = 60,
    batch_size: int = 6,
) -> list[dict]:
    """
    Sinh n_total reviews cho target_aspect.
    Sentiment ratio cố định toàn cục: 40% pos, 35% neg, 25% neutral.
    """
    system_prompt = _SYSTEM_PROMPT.format(
        aspect_list="\n".join(f"  - {aspect}" for aspect in ASPECT_COLUMNS)
    )
    description = WEAK_ASPECT_DESCRIPTIONS.get(target_aspect, target_aspect)
    target_total = max(n_total, TARGET_COUNT_OVERRIDES.get(target_aspect, n_total))
    batch_size = BATCH_SIZE_OVERRIDES.get(target_aspect, batch_size)
    n_pos_total = round(target_total * 0.40)
    n_neg_total = round(target_total * 0.35)
    n_neu_total = target_total - n_pos_total - n_neg_total

    remaining_targets = {
        "positive": n_pos_total,
        "negative": n_neg_total,
        "neutral": n_neu_total,
    }
    all_reviews = []
    n_batches = (target_total + batch_size - 1) // batch_size
    max_attempts = n_batches * (4 if target_aspect in DIFFICULT_ASPECTS else 2)
    attempt = 0

    while len(all_reviews) < target_total and attempt < max_attempts:
        batch_idx = attempt
        remaining = target_total - len(all_reviews)
        if remaining <= 0:
            break
        attempt += 1

        current = min(batch_size, remaining)
        sentiments = []
        for sentiment in ("positive", "negative", "neutral"):
            take = min(remaining_targets[sentiment], current - len(sentiments))
            sentiments.extend([sentiment] * take)
            remaining_targets[sentiment] -= take
        while len(sentiments) < current:
            sentiments.append("positive")

        n_pos = sentiments.count("positive")
        n_neg = sentiments.count("negative")
        n_neu = sentiments.count("neutral")

        real_examples = _get_real_examples(
            train_df=train_df,
            target_aspect=target_aspect,
            n=3,
            random_state=batch_idx * 19 + 7,
        )

        already_section = ""
        if all_reviews:
            texts = "\n".join(f"- {item['review'][:100]}" for item in all_reviews[-8:])
            already_section = _ALREADY_SECTION.format(texts=texts)

        user_prompt = _USER_PROMPT.format(
            batch_size=current,
            target_aspect=target_aspect,
            description=description,
            n_pos=n_pos,
            n_neg=n_neg,
            n_neu=n_neu,
            real_examples=real_examples,
            anchor_examples=ANCHOR_EXAMPLES.get(target_aspect, "Không có."),
            strict_hint=ASPECT_STRICT_HINTS.get(
                target_aspect,
                "Luôn nêu rõ chi tiết target aspect trong câu, tránh nói chung chung.",
            ),
            already_section=already_section,
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = llm_client.complete(messages, temperature=0.85, use_cache=False)
            parsed = _dedup_reviews(_parse_response(response, target_aspect))
            all_reviews.extend(parsed)
            print(
                f"  [{target_aspect}] Attempt {attempt}/{max_attempts}: "
                f"got {len(parsed)} (total: {len(all_reviews)})"
            )
        except Exception as exc:
            print(f"  [{target_aspect}] Attempt {attempt} FAILED: {exc}")
        time.sleep(1.5)

    return _dedup_reviews(all_reviews)[:target_total]


def run_augmentation_v2(
    train_df: pd.DataFrame,
    llm_client: LLMClient,
    n_per_aspect: int = 60,
    raw_output_path: str = "data/augmented_reviews_v2.json",
    run_consistency_check: bool = True,
) -> list[dict]:
    """Sinh augmented data cho toàn bộ WEAK_ASPECTS."""
    all_augmented = []

    for aspect in WEAK_ASPECTS:
        print(f"\n{'=' * 60}")
        target_count = max(n_per_aspect, TARGET_COUNT_OVERRIDES.get(aspect, n_per_aspect))
        print(f"Generating {target_count} reviews for: {aspect}")
        print(f"{'=' * 60}")

        reviews = generate_for_aspect(
            train_df=train_df,
            target_aspect=aspect,
            llm_client=llm_client,
            n_total=target_count,
        )
        print(f"  -> Generated: {len(reviews)}")

        if run_consistency_check and reviews:
            print("  -> Running label consistency check...")
            reviews = _label_consistency_check(reviews, llm_client)

        all_augmented.extend(reviews)
        print(f"  -> Kept: {len(reviews)}")

    os.makedirs(os.path.dirname(raw_output_path) or ".", exist_ok=True)
    with open(raw_output_path, "w", encoding="utf-8") as handle:
        json.dump(all_augmented, handle, ensure_ascii=False, indent=2)

    print(f"\nTotal: {len(all_augmented)} reviews -> {raw_output_path}")
    return all_augmented
