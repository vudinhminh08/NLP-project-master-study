"""
explainer.py — LLM Explainability Layer cho PhoBERT predictions.

Pipeline:
1. PhoBERT predict → 34 nhãn
2. Filter chỉ lấy aspects PRESENT (label != 0)
3. Gửi review + predicted labels cho LLM
4. LLM sinh giải thích cho TỪNG aspect
5. Output: structured JSON dùng cho ứng dụng thực tế

Ứng dụng thực tế:
- Quản lý khách sạn nhận review → thấy ngay:
  + Aspect nào được đề cập
  + Sentiment là gì
  + TẠI SAO (evidence từ review + giải thích)
  + Đề xuất hành động
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "week1"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "week3"))

from utils.constants import ASPECT_COLUMNS, IDX_TO_LABEL  # noqa: E402
from llm_client import LLMClient  # noqa: E402


# ─── Prompts ─────────────────────────────────────────────────────────────────

EXPLAINER_SYSTEM_PROMPT = """Bạn là hệ thống phân tích đánh giá khách sạn. Bạn nhận review của khách hàng cùng kết quả phân tích từ AI, và nhiệm vụ của bạn là GIẢI THÍCH kết quả bằng tiếng Việt tự nhiên.

## Quy tắc:
1. Với mỗi aspect được phát hiện, trích dẫn ĐÚNG phần review liên quan (evidence)
2. Giải thích ngắn gọn tại sao sentiment là positive/negative/neutral
3. Đề xuất hành động cụ thể cho quản lý khách sạn (nếu negative)
4. Trả về JSON, KHÔNG giải thích thêm
"""

EXPLAINER_USER_PROMPT = """Review khách hàng:
"{review}"

Kết quả phân tích AI:
{predictions}

Hãy giải thích từng aspect. Format JSON:
```json
[
  {{
    "aspect": "ENTITY#ATTRIBUTE",
    "sentiment": "positive/negative/neutral",
    "evidence": "trích dẫn phần review liên quan",
    "explanation": "giải thích ngắn gọn tiếng Việt",
    "action": "đề xuất hành động (chỉ với negative, để null nếu positive/neutral)"
  }}
]
```
"""


def explain_predictions(
    review: str,
    predictions: dict,
    llm_client: LLMClient,
) -> list[dict]:
    """
    Sinh giải thích cho PhoBERT predictions trên 1 review.

    Args:
        review: text review gốc (chưa segment)
        predictions: dict {aspect: label_idx} hoặc {aspect: sentiment_str}
                     chỉ chứa aspects PRESENT (label != 0 / != "absent")
        llm_client: LLMClient instance

    Returns:
        list of dicts: giải thích cho từng aspect
    """
    if not predictions:
        return []

    pred_lines = []
    for aspect, value in predictions.items():
        if isinstance(value, int):
            if value == 0:
                continue
            sentiment = IDX_TO_LABEL.get(value, "unknown")
        else:
            if value == "absent":
                continue
            sentiment = value
        pred_lines.append(f"- {aspect}: {sentiment}")

    if not pred_lines:
        return []

    predictions_text = "\n".join(pred_lines)
    messages = [
        {"role": "system", "content": EXPLAINER_SYSTEM_PROMPT},
        {"role": "user", "content": EXPLAINER_USER_PROMPT.format(
            review=review,
            predictions=predictions_text,
        )},
    ]

    try:
        response = llm_client.complete(messages)
        return _parse_explanation(response)
    except Exception as exc:
        print(f"  Explainer error: {exc}")
        return [{"aspect": a, "sentiment": s, "error": str(exc)}
                for a, s in predictions.items()]


def _parse_explanation(response: str) -> list[dict]:
    """Parse JSON response từ LLM."""
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

    return [item for item in data if isinstance(item, dict) and "aspect" in item]


def explain_batch(
    reviews: list[str],
    predictions_list: list[dict],
    llm_client: LLMClient,
    delay: float = 1.0,
) -> list[list[dict]]:
    """
    Giải thích batch reviews.

    Args:
        reviews: list of review texts
        predictions_list: list of dicts, mỗi dict là {aspect: sentiment}
        delay: seconds giữa API calls

    Returns:
        list of explanations (mỗi review = 1 list of aspect explanations)
    """
    all_explanations = []

    for i, (review, preds) in enumerate(zip(reviews, predictions_list)):
        print(f"  Explaining review {i + 1}/{len(reviews)}...")
        expl = explain_predictions(review, preds, llm_client)
        all_explanations.append(expl)

        if delay > 0 and i < len(reviews) - 1:
            time.sleep(delay)

    return all_explanations
