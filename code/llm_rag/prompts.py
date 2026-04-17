
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data_processing'))
from utils.constants import ASPECT_COLUMNS, RARE_ASPECTS, LABEL_TO_IDX




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




COT_INSTRUCTION = """Hãy phân tích theo các bước:
1. Đọc review và xác định các từ/cụm từ quan trọng
2. Với mỗi aspect, tự hỏi: "Review này có đề cập đến [aspect] không?"
3. Nếu có, xác định cảm xúc: positive / negative / neutral
4. Chỉ đưa vào output các aspects ĐƯỢC ĐỀ CẬP

Sau phần phân tích, trả về kết quả theo đúng format JSON.
"""




def format_example(review: str, labels: dict, include_cot: bool = True) -> str:
    output_json = {asp: sent for asp, sent in labels.items()}

    if include_cot:

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




def build_prompt(
    test_review: str,
    examples: list[dict],
    include_cot: bool = True,
) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]


    for ex in examples:
        formatted = format_example(
            ex["review"], ex["labels"], include_cot=include_cot
        )
        messages.append({"role": "user", "content": formatted.split("\nOutput:")[0]})
        messages.append({
            "role": "assistant",
            "content": formatted.split("\nOutput:")[-1].strip()
        })


    test_msg = f"""Review: {test_review}
{COT_INSTRUCTION if include_cot else ""}
Trả về JSON output (kết thúc bằng JSON, ví dụ: {{"SERVICE#GENERAL": "positive"}} hoặc {{}} nếu không có aspect nào):"""
    messages.append({"role": "user", "content": test_msg})

    return messages




def parse_llm_output(raw_output: str) -> dict:
    import json, re

    valid_sentiments = {"positive", "negative", "neutral"}
    valid_aspects    = set(ASPECT_COLUMNS)


    raw = raw_output.strip()
    raw = re.sub(r'```(?:json)?\s*', '', raw)
    raw = re.sub(r'```\s*$', '', raw)


    parsed = None

    brace_matches = list(re.finditer(r'\{', raw))
    for start_match in brace_matches:
        start = start_match.start()

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
        return {}

    if not isinstance(parsed, dict):
        return {}

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


    result = {}
    for asp, sent in parsed.items():
        if asp not in valid_aspects:
            continue
        sent = str(sent).lower().strip()
        if sent not in valid_sentiments:
            continue
        result[asp] = sent

    return result


def labels_dict_to_array(labels_dict: dict) -> list[int]:
    result = [0] * len(ASPECT_COLUMNS)
    for asp, sent in labels_dict.items():
        if asp in ASPECT_COLUMNS:
            idx = ASPECT_COLUMNS.index(asp)
            result[idx] = LABEL_TO_IDX.get(sent, 0)
    return result
