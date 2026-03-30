"""
prompts.py — Prompt templates tiếng Việt cho ABSA VLSP 2018 Hotel.

Nguyên tắc thiết kế prompt:
1. System prompt: định nghĩa rõ vai trò + 34 aspects hợp lệ
2. Liệt kê RARE_ASPECTS riêng: nhắc model chú ý aspects hiếm
3. Chain-of-Thought: yêu cầu model giải thích trước khi ra output
4. Output format: JSON chặt chẽ, dễ parse
5. Few-shot examples: minh họa cả trường hợp có và không có aspect
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'week1'))
from utils.constants import ASPECT_COLUMNS, RARE_ASPECTS, LABEL_TO_IDX


# ─── System Prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Bạn là chuyên gia phân tích đánh giá khách sạn tiếng Việt.
Nhiệm vụ: Phân tích review và xác định Aspect Category Sentiment Analysis (ABSA).

## 34 Aspect Categories hợp lệ:
{aspect_list}

## Lưu ý đặc biệt — Các aspect hiếm gặp (cần chú ý kỹ):
{rare_aspects_list}

## Quy tắc phân loại:
- **positive**: review khen, hài lòng, tốt về aspect đó
- **negative**: review chê, không hài lòng, tệ về aspect đó
- **neutral**: đề cập nhưng không rõ tích cực hay tiêu cực (ví dụ: "bình thường", "tạm được")
- **absent**: KHÔNG đề cập aspect này trong review → KHÔNG đưa vào output

## Quy tắc output:
- Chỉ đưa vào output những aspects ĐƯỢC ĐỀ CẬP trong review
- Không bịa thêm aspect nếu review không đề cập
- Format: JSON object, key là "ENTITY#ATTRIBUTE", value là "positive"/"negative"/"neutral"
- Nếu không có aspect nào: trả về {{}}
""".format(
    aspect_list="\n".join(f"  - {a}" for a in ASPECT_COLUMNS),
    rare_aspects_list="\n".join(f"  - {a} (rất hiếm, chỉ đưa vào nếu chắc chắn)" for a in RARE_ASPECTS),
)


# ─── Chain-of-Thought Template ────────────────────────────────────────────────

COT_INSTRUCTION = """Hãy phân tích theo các bước:
1. Đọc review và xác định các từ/cụm từ quan trọng
2. Với mỗi aspect, tự hỏi: "Review này có đề cập đến [aspect] không?"
3. Nếu có, xác định cảm xúc: positive / negative / neutral
4. Chỉ đưa vào output các aspects ĐƯỢC ĐỀ CẬP

Sau phần phân tích, trả về kết quả theo đúng format JSON.
"""


# ─── Few-shot Example Template ────────────────────────────────────────────────

def format_example(review: str, labels: dict, include_cot: bool = True) -> str:
    """
    Format 1 few-shot example.

    Args:
        review: text review gốc
        labels: dict {aspect: sentiment} chỉ chứa aspects PRESENT
        include_cot: có thêm chain-of-thought không

    Returns:
        formatted string dùng trong prompt
    """
    output_json = {asp: sent for asp, sent in labels.items()}

    if include_cot:
        # Tạo CoT tự động từ labels
        cot_lines = ["Phân tích:"]
        for asp, sent in labels.items():
            entity, attr = asp.split("#")
            cot_lines.append(f"- {asp}: review đề cập đến {entity.lower()} / {attr.lower()} → {sent}")
        if not labels:
            cot_lines.append("- Review không đề cập cụ thể aspect nào trong danh sách.")

        return f"""Review: {review}
{chr(10).join(cot_lines)}
Output: {str(output_json).replace("'", '"')}"""
    else:
        return f"""Review: {review}
Output: {str(output_json).replace("'", '"')}"""


# ─── Build Full Prompt ─────────────────────────────────────────────────────────

def build_prompt(
    test_review: str,
    examples: list[dict],
    include_cot: bool = True,
) -> list[dict]:
    """
    Build full prompt cho 1 test review.

    Args:
        test_review: review cần phân tích
        examples: list of dicts, mỗi dict có keys 'review' và 'labels'
                  labels là dict {aspect: sentiment} chỉ gồm present aspects
        include_cot: có dùng chain-of-thought không

    Returns:
        list of message dicts cho OpenAI / Gemini format
        [{"role": "system", "content": ...},
         {"role": "user", "content": ...},
         {"role": "assistant", "content": ...},  # few-shot
         {"role": "user", "content": ...}]        # test
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Few-shot examples
    for ex in examples:
        formatted = format_example(
            ex["review"], ex["labels"], include_cot=include_cot
        )
        messages.append({"role": "user", "content": formatted.split("\nOutput:")[0]})
        messages.append({
            "role": "assistant",
            "content": formatted.split("\nOutput:")[-1].strip()
        })

    # Test review
    test_msg = f"""Review: {test_review}
{COT_INSTRUCTION if include_cot else ""}
Trả về JSON output:"""
    messages.append({"role": "user", "content": test_msg})

    return messages


# ─── Parse LLM Output ─────────────────────────────────────────────────────────

def parse_llm_output(raw_output: str) -> dict:
    """
    Parse JSON từ LLM output. Robust với các format khác nhau.

    Xử lý các trường hợp:
    - Output chuẩn: {"SERVICE#GENERAL": "positive"}
    - Có markdown fence: ```json {...} ```
    - Có text thừa trước/sau JSON
    - JSON không hợp lệ → trả về {} và log warning

    Returns:
        dict {aspect: sentiment} chỉ gồm valid aspects và valid sentiments
    """
    import json, re

    valid_sentiments = {"positive", "negative", "neutral"}
    valid_aspects    = set(ASPECT_COLUMNS)

    # Strip markdown fences
    raw = raw_output.strip()
    raw = re.sub(r'```(?:json)?\s*', '', raw)
    raw = re.sub(r'```\s*$', '', raw)

    # Tìm JSON object trong output — thử từ outermost đến innermost
    parsed = None
    # Tìm tất cả cặp {} và thử parse từ ngoài vào trong
    brace_matches = list(re.finditer(r'\{', raw))
    for start_match in brace_matches:
        start = start_match.start()
        # Tìm closing brace tương ứng (đếm depth)
        depth = 0
        for i, ch in enumerate(raw[start:]):
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    candidate = raw[start: start + i + 1]
                    try:
                        parsed = json.loads(candidate)
                        if isinstance(parsed, dict):
                            break
                    except json.JSONDecodeError:
                        pass
        if parsed is not None:
            break

    # Fallback: ast.literal_eval cho Python-style single-quote dict
    if parsed is None:
        import ast
        sq_match = re.search(r'\{[^{}]*\}', raw, re.DOTALL)
        if sq_match:
            try:
                parsed = ast.literal_eval(sq_match.group())
            except (ValueError, SyntaxError):
                pass

    if parsed is None:
        if '{}' in raw:
            return {}
        print(f"[WARN] Không tìm thấy JSON trong output: {raw[:100]}")
        return {}

    if not isinstance(parsed, dict):
        print(f"[WARN] Parsed value không phải dict: {type(parsed)}")
        return {}

    # Safeguard: nếu values là dicts (nested JSON như {"result": {...}}),
    # tìm nested dict chứa valid aspect-sentiment pairs
    if any(isinstance(v, dict) for v in parsed.values()):
        for v in parsed.values():
            if isinstance(v, dict):
                candidate = {
                    k2: str(v2).lower().strip()
                    for k2, v2 in v.items()
                    if k2 in valid_aspects and str(v2).lower().strip() in valid_sentiments
                }
                if candidate:
                    parsed = v
                    break

    # Validate và filter
    result = {}
    for asp, sent in parsed.items():
        if asp not in valid_aspects:
            print(f"[WARN] Invalid aspect: {asp}")
            continue
        sent = str(sent).lower().strip()
        if sent not in valid_sentiments:
            print(f"[WARN] Invalid sentiment '{sent}' for {asp}")
            continue
        result[asp] = sent

    return result


def labels_dict_to_array(labels_dict: dict) -> list[int]:
    """
    Chuyển dict {aspect: sentiment} → list 34 integers (0/1/2/3).
    Dùng để tính F1 với step4_eval.
    """
    result = [0] * len(ASPECT_COLUMNS)
    for asp, sent in labels_dict.items():
        if asp in ASPECT_COLUMNS:
            idx = ASPECT_COLUMNS.index(asp)
            result[idx] = LABEL_TO_IDX.get(sent, 0)
    return result
