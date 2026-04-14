"""
explanation_prompts.py — Prompts for LLM explanation, not prediction.
"""

from __future__ import annotations

import json


EXPLANATION_SYSTEM_PROMPT = """Bạn là trợ lý giải thích kết quả phân tích cảm xúc theo khía cạnh cho review khách sạn tiếng Việt.

Bạn KHÔNG phải là classifier. Aspect và sentiment đã được PhoBERT dự đoán.
Nhiệm vụ của bạn:
1. Trích bằng chứng trong review cho từng prediction.
2. Giải thích ngắn gọn vì sao prediction hợp lý.
3. Không thêm aspect ngoài danh sách prediction.
4. Không sửa sentiment.
5. Nếu không tìm thấy bằng chứng rõ, đặt evidence_uncertain=true.

Chỉ trả JSON hợp lệ, không trả markdown."""


OUTPUT_SCHEMA = {
    "items": [
        {
            "aspect": "ENTITY#ATTRIBUTE",
            "sentiment": "positive|negative|neutral",
            "evidence": "cụm từ trong review, hoặc rỗng nếu không chắc",
            "explanation": "giải thích ngắn gọn bằng tiếng Việt",
            "evidence_confidence": "low|medium|high",
            "evidence_uncertain": False,
        }
    ],
    "overall_summary": "tóm tắt ngắn nội dung review",
    "recommended_action": "gợi ý hành động nếu có negative, hoặc null",
}


def build_explanation_prompt(
    review: str,
    predictions: list[dict],
    rag_examples: list[dict] | None = None,
) -> list[dict]:
    """
    Build messages for LLM explanation.

    predictions must contain present PhoBERT predictions only.
    """
    user_payload = {
        "review": review,
        "phobert_predictions": [
            {
                "aspect": p["aspect"],
                "sentiment": p["sentiment"],
                **({"confidence": p["confidence"]} if "confidence" in p else {}),
            }
            for p in predictions
        ],
        "rag_examples": rag_examples or [],
        "rules": [
            "Giữ nguyên aspect và sentiment từ phobert_predictions.",
            "Không thêm aspect mới.",
            "Evidence phải là cụm từ nằm trong review nếu có thể.",
            "Nếu prediction có vẻ khó tìm bằng chứng, vẫn giữ prediction nhưng đánh dấu evidence_uncertain=true.",
        ],
        "output_schema": OUTPUT_SCHEMA,
    }
    return [
        {"role": "system", "content": EXPLANATION_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]
