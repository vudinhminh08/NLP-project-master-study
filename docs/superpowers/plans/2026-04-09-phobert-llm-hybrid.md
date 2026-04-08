# PhoBERT + LLM Hybrid Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cải thiện Combined F1 từ 0.5543 bằng cách (A) sinh data mới targeted theo error analysis + retrain PhoBERT, rồi (C) thêm LLM cascade ở inference time cho những aspect PhoBERT uncertain.

**Architecture:**
- Phase A: Phân tích per-aspect F1 → xác định 9 WEAK_ASPECTS (F1 < 0.35 hoặc = 0) → LLM sinh 60 samples/aspect (540 total) → filter → VnCoreNLP segment local → `train_augmented_v2.csv` → retrain trên Kaggle.
- Phase C: Sau khi có model mới, thêm `cascade_predictor.py`: PhoBERT predict + lấy confidence (max softmax per head) → nếu bất kỳ WEAK_ASPECT nào uncertain (confidence < threshold) → gọi LLM RAG k=4 → override prediction cho aspect đó.

**Tech Stack:** Python 3.11+, PyTorch, HuggingFace Transformers, GPT-4o-mini (OpenAI), FAISS (RAG retriever đã có), VnCoreNLP (local only, đã có), Kaggle GPU T4.

---

## Bối cảnh dự án (đọc trước khi làm)

```
absa-vlsp2018-hotel/
├── code/
│   ├── week1/
│   │   ├── utils/constants.py      # ASPECT_COLUMNS, RARE_ASPECTS, LABEL_TO_IDX, IDX_TO_LABEL
│   │   ├── utils/helpers.py        # set_seed, save_json
│   │   ├── step2_dataloader.py     # create_dataloaders(train_path, dev_path, test_path, tokenizer, batch_size, max_len, num_workers, use_preprocessed)
│   │   └── step4_eval.py           # evaluate_predictions(y_true, y_pred) -> metrics dict
│   ├── week2/
│   │   ├── model.py                # ABSAPhoBERT(model_name, encoder_option, dropout, focal_gamma)
│   │   │                           # forward(input_ids, attention_mask, labels=None, class_weights=None, token_type_ids=None) -> {loss, logits, preds}
│   │   │                           # logits = list of 34 tensors [batch, 4]
│   │   │                           # preds = [batch, 34] argmax
│   │   └── train.py                # train(model, train_loader, dev_loader, class_weights, device, config, save_dir, results_dir)
│   │                               # load_class_weights(weights_path, weight_clip, device)
│   ├── week3/
│   │   ├── llm_client.py           # LLMClient(provider, api_key); complete(messages, temperature, use_cache)
│   │   ├── rag_retriever.py        # ABSARetriever; retrieve(query, k, aspect_aware) -> list[int]
│   │   ├── icl_predictor.py        # df_to_examples(train_df, indices) -> list[dict]
│   │   └── prompts.py              # build_prompt(query, examples) -> messages; parse_llm_output(raw) -> dict; labels_dict_to_array(d) -> np.ndarray shape [34]
│   └── week3_part2/
│       ├── augmentor.py            # generate_augmented_reviews(), run_augmentation_all_aspects()
│       └── augment_filter.py       # filter_augmented_reviews(raw_path, filtered_path, train_df, llm_client, verify_rate)
├── data/
│   ├── train_preprocessed.csv      # 3000 samples, cols: Review, processed_review, + 34 aspect cols (0-3)
│   ├── dev_preprocessed.csv        # 2000 samples
│   ├── test_preprocessed.csv       # 600 samples
│   └── train_augmented.csv         # v1: 3231 samples (deprecated sau plan này)
├── outputs/
│   ├── eda/class_weights.json      # per-aspect class weights (34 aspects × 4 classes)
│   └── results/week2_results_VNcoreNLP/results_cls_only/week2_test_metrics.json
│                                   # per_aspect dict: {aspect: {acd_f1, spc_f1}}
└── notebooks/
    └── week4_demo.ipynb            # Kaggle notebook để train + eval
```

**Per-aspect F1 baseline (PhoBERT cls_only + VNcoreNLP, test set):**
| Aspect | ACD F1 | SPC F1 | Combined |
|---|---|---|---|
| FACILITIES#MISCELLANEOUS | 0.000 | 0.000 | 0.000 |
| FOOD&DRINKS#MISCELLANEOUS | 0.000 | 0.000 | 0.000 |
| ROOMS#MISCELLANEOUS | 0.000 | 0.000 | 0.000 |
| ROOM_AMENITIES#MISCELLANEOUS | 0.000 | 0.000 | 0.000 |
| ROOM_AMENITIES#PRICES | 0.000 | 0.000 | 0.000 |
| HOTEL#MISCELLANEOUS | 0.315 | 0.181 | 0.248 |
| FACILITIES#GENERAL | 0.415 | 0.230 | 0.322 |
| FACILITIES#CLEANLINESS | 0.400 | 0.267 | 0.333 |
| FACILITIES#COMFORT | 0.488 | 0.190 | 0.339 |

**Checkpoint format (quan trọng khi load model):**
```python
ckpt = torch.load('best_model.pt', map_location=device)
model.load_state_dict(ckpt['model_state_dict'])  # KHÔNG load cả ckpt
```

---

## File Structure — Những file sẽ tạo / sửa

| File | Action | Mục đích |
|---|---|---|
| `code/week1/utils/constants.py` | Modify | Thêm `WEAK_ASPECTS` list (9 aspects có F1 thấp) |
| `code/week3_part2/augmentor_v2.py` | **Create** | Augmentor mới: target WEAK_ASPECTS, 60/aspect, label consistency check |
| `data/augmented_reviews_v2.json` | Generated | Raw output của augmentor_v2 (không commit, tự sinh) |
| `data/augmented_reviews_v2_filtered.json` | Generated | Sau filter (không commit, tự sinh) |
| `data/train_augmented_v2.csv` | Generated | Merge gốc + augmented v2, VnCoreNLP segment (commit lên git) |
| `code/week3_part2/cascade_predictor.py` | **Create** | Inference cascade: PhoBERT + LLM override cho uncertain aspects |
| `notebooks/week4_demo.ipynb` | Modify | Dùng train_augmented_v2.csv + thêm Cell 11 chạy cascade eval |
| `CHANGELOG.md` | Modify | Cập nhật kết quả Phase A và Phase C |

---

## PHASE A — Error-Driven Augmentation

### Task 1: Thêm WEAK_ASPECTS vào constants.py

**Files:**
- Modify: `code/week1/utils/constants.py`

- [ ] **Step 1: Đọc file hiện tại để biết cấu trúc**

```bash
cat code/week1/utils/constants.py
```

- [ ] **Step 2: Thêm WEAK_ASPECTS list sau RARE_ASPECTS**

Mở `code/week1/utils/constants.py`, tìm dòng có `RARE_ASPECTS` và thêm sau đó:

```python
# Aspects có Combined F1 = 0 hoặc < 0.35 trên test set (PhoBERT cls_only baseline)
# Dùng làm target cho error-driven augmentation (Phase A, Week 4 v2)
WEAK_ASPECTS = [
    "FACILITIES#MISCELLANEOUS",       # F1 = 0.000
    "FOOD&DRINKS#MISCELLANEOUS",      # F1 = 0.000
    "ROOMS#MISCELLANEOUS",            # F1 = 0.000
    "ROOM_AMENITIES#MISCELLANEOUS",   # F1 = 0.000
    "ROOM_AMENITIES#PRICES",          # F1 = 0.000
    "HOTEL#MISCELLANEOUS",            # F1 = 0.248
    "FACILITIES#GENERAL",             # F1 = 0.322
    "FACILITIES#CLEANLINESS",         # F1 = 0.333
    "FACILITIES#COMFORT",             # F1 = 0.339
]
```

- [ ] **Step 3: Verify import**

```bash
cd /path/to/absa-vlsp2018-hotel
python3 -c "
import sys; sys.path.insert(0, 'code/week1')
from utils.constants import WEAK_ASPECTS
print('WEAK_ASPECTS:', len(WEAK_ASPECTS), 'aspects')
assert len(WEAK_ASPECTS) == 9
print('OK')
"
```

Expected output:
```
WEAK_ASPECTS: 9 aspects
OK
```

- [ ] **Step 4: Commit**

```bash
git add code/week1/utils/constants.py
git commit -m "feat: add WEAK_ASPECTS constant (9 aspects with F1<0.35 on test set)"
```

---

### Task 2: Tạo augmentor_v2.py — target WEAK_ASPECTS với label consistency check

**Files:**
- Create: `code/week3_part2/augmentor_v2.py`

Cải tiến so với augmentor.py:
1. Dùng `WEAK_ASPECTS` thay `RARE_ASPECTS`
2. n_per_aspect=60 (gấp đôi)
3. Thêm `label_consistency_check`: sau khi parse, filter những review mà LLM tự sinh nhưng labels không khớp với target aspect (verify 100% thay vì 30% sample)
4. Better diversity: sentiment ratio cứng 40:35:25 pos:neg:neutral thay vì tính động

- [ ] **Step 1: Tạo file augmentor_v2.py**

```python
# code/week3_part2/augmentor_v2.py
"""
augmentor_v2.py — Error-driven augmentation targeting WEAK_ASPECTS.

Cải tiến so với augmentor.py (v1):
1. Target WEAK_ASPECTS (9 aspects F1<0.35) thay vì RARE_ASPECTS (8 aspects by frequency)
2. n_per_aspect=60 (v1: 30) — volume lớn hơn để bù filter loss
3. Label consistency check: sau parse, dùng LLM verify 100% reviews
   (v1 chỉ verify 30% sample sau khi đã merge tất cả aspects)
4. Sentiment ratio cứng: 24 positive, 21 negative, 15 neutral mỗi aspect
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


# ─── Mô tả chi tiết cho từng WEAK_ASPECT ──────────────────────────────────────

WEAK_ASPECT_DESCRIPTIONS = {
    "FACILITIES#MISCELLANEOUS": (
        "Tiện nghi chung của khách sạn NGOÀI phòng: hồ bơi, gym, spa, sảnh chờ, "
        "thang máy, bãi đỗ xe, phòng hội nghị, khu vui chơi. "
        "Ví dụ: 'hồ bơi sạch đẹp', 'thang máy hay hỏng', 'có gym nhưng thiết bị cũ'."
    ),
    "FOOD&DRINKS#MISCELLANEOUS": (
        "Các vấn đề KHÁC về ăn uống: thời gian phục vụ bữa sáng, sự đa dạng món, "
        "đồ uống tại bar, room service. "
        "Ví dụ: 'bữa sáng phục vụ đến 10h', 'menu bar đa dạng', 'room service chậm'."
    ),
    "ROOMS#MISCELLANEOUS": (
        "Các vấn đề KHÁC về phòng: tiếng ồn từ ngoài/phòng bên, côn trùng, "
        "mùi ẩm mốc, view từ phòng, tầng phòng. "
        "Ví dụ: 'phòng bị ồn do gần thang máy', 'thấy gián trong phòng', 'view biển tuyệt'."
    ),
    "ROOM_AMENITIES#MISCELLANEOUS": (
        "Tiện nghi TRONG phòng (không phải giường/tắm): remote TV, số lượng ổ cắm, "
        "móc áo, bàn là, két sắt, máy pha cà phê. "
        "Ví dụ: 'thiếu ổ cắm cạnh giường', 'có két sắt tiện', 'remote TV bị hỏng'."
    ),
    "ROOM_AMENITIES#PRICES": (
        "Giá cả dịch vụ tính phí TRONG phòng: minibar đắt, wifi tính phí, "
        "dịch vụ phòng có phí riêng, tiền giặt ủi. "
        "Ví dụ: 'minibar tính giá cắt cổ', 'wifi phòng mất thêm 50k/ngày', "
        "'giá giặt ủi quá cao'."
    ),
    "HOTEL#MISCELLANEOUS": (
        "Các vấn đề TỔNG QUÁT về khách sạn không thuộc category khác: "
        "chính sách check-in/out, phụ phí ẩn, quy định pet, chính sách hủy phòng. "
        "Ví dụ: 'check-out muộn mất phí', 'có phụ phí resort fee không báo trước'."
    ),
    "FACILITIES#GENERAL": (
        "Đánh giá CHUNG về tiện nghi/cơ sở vật chất khách sạn (không cụ thể). "
        "Ví dụ: 'cơ sở vật chất tốt', 'tiện nghi đầy đủ', 'facilities cũ kỹ cần nâng cấp'."
    ),
    "FACILITIES#CLEANLINESS": (
        "Vệ sinh khu vực CHUNG của khách sạn: sảnh, hành lang, thang máy, hồ bơi, nhà vệ sinh công cộng. "
        "Ví dụ: 'sảnh khách sạn rất sạch', 'hành lang có mùi', 'hồ bơi nước xanh trong'."
    ),
    "FACILITIES#COMFORT": (
        "Sự thoải mái của tiện nghi/khu vực chung: ghế ở sảnh, nhiệt độ điều hòa khu chung, "
        "ánh sáng, âm nhạc nền. "
        "Ví dụ: 'sảnh mát mẻ thoải mái', 'khu vực ngồi chờ không thoải mái'."
    ),
}

# ─── Prompts ──────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """Bạn là chuyên gia tạo dữ liệu huấn luyện cho bài toán phân tích cảm xúc theo khía cạnh (ABSA) cho đánh giá khách sạn tiếng Việt.

Nhiệm vụ: Viết các review khách sạn tiếng Việt TỰ NHIÊN, giống review thật của khách hàng.

Quy tắc CỨNG:
1. Mỗi review PHẢI đề cập rõ ràng đến aspect mục tiêu — đủ rõ để người đọc nhận ra
2. Review phải tự nhiên: ngắn gọn (1-2 câu) hoặc vừa (3-4 câu), đôi khi có teencode nhẹ (ko, nv, ks)
3. Sentiment cho target aspect phải rõ ràng: POSITIVE (khen/hài lòng), NEGATIVE (chê/không hài lòng), NEUTRAL (đề cập trung tính)
4. Mỗi review CÓ THỂ đề cập thêm 1-2 aspects khác (realistic)
5. KHÔNG viết review quá formal, quá dài (>4 câu), hoặc giống AI

34 Aspect Categories hợp lệ:
{aspect_list}

Sentiment labels hợp lệ: positive, negative, neutral, absent
(absent = không đề cập → KHÔNG đưa vào labels output)
"""

_USER_PROMPT = """Viết {batch_size} review khách sạn tiếng Việt, mỗi review PHẢI đề cập đến:

**Target aspect: {target_aspect}**
Mô tả: {description}

Phân phối sentiment BẮT BUỘC:
- {n_pos} review với {target_aspect} = POSITIVE
- {n_neg} review với {target_aspect} = NEGATIVE
- {n_neu} review với {target_aspect} = NEUTRAL

Ví dụ review THẬT từ dataset (tham khảo phong cách, KHÔNG copy):
{real_examples}

{already_section}

Output: JSON array, mỗi phần tử gồm:
```json
[
  {{
    "review": "text review tiếng Việt",
    "labels": {{
      "{target_aspect}": "positive/negative/neutral",
      "ASPECT_KHAC#NAME": "positive/negative/neutral"
    }}
  }}
]
```

CHỈ trả về JSON array, không giải thích.
"""

_ALREADY_SECTION = """Reviews ĐÃ SINH — TUYỆT ĐỐI không lặp lại, tạo nội dung HOÀN TOÀN KHÁC:
{texts}
"""

_CONSISTENCY_PROMPT = """Review khách sạn sau có ĐỀ CẬP rõ ràng đến "{aspect}" ({description}) không?

Review: {review}

Trả lời CHỈ "YES" hoặc "NO".
"""


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _get_real_examples(
    train_df: pd.DataFrame,
    target_aspect: str,
    n: int = 3,
    random_state: Optional[int] = None,
) -> str:
    """Lấy n review thật có target_aspect từ train set (hoặc cùng entity nếu không có)."""
    mask = train_df[target_aspect] > 0
    candidates = train_df[mask]

    if len(candidates) == 0:
        entity = target_aspect.split("#")[0]
        siblings = [a for a in ASPECT_COLUMNS if a.startswith(entity + "#") and a != target_aspect]
        if siblings:
            mask = train_df[siblings].max(axis=1) > 0
            candidates = train_df[mask]

    if len(candidates) == 0:
        candidates = train_df

    sample = candidates.sample(min(n, len(candidates)), random_state=random_state)
    lines = []
    for _, row in sample.iterrows():
        review = row.get("Review", row.get("processed_review", ""))
        labels = {a: IDX_TO_LABEL[int(row[a])] for a in ASPECT_COLUMNS if int(row[a]) > 0}
        lines.append(f"Review: {review}\nLabels: {json.dumps(labels, ensure_ascii=False)}")
    return "\n\n".join(lines)


def _parse_response(response: str, target_aspect: str) -> list[dict]:
    """Parse JSON response từ LLM, filter những item không hợp lệ."""
    text = response.strip()
    # Strip markdown code fences nếu có
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:])
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("["), text.rfind("]") + 1
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
        review = item.get("review", "")
        labels = item.get("labels", {})
        if not isinstance(review, str) or len(review.strip()) < 15:
            continue
        if not isinstance(labels, dict):
            continue
        # Clean labels — chỉ giữ aspects hợp lệ
        clean_labels = {
            a: s for a, s in labels.items()
            if a in ASPECT_COLUMNS and s in {"positive", "negative", "neutral"}
        }
        # Phải có target_aspect trong labels
        if target_aspect not in clean_labels:
            continue
        valid.append({
            "review": review.strip(),
            "labels": clean_labels,
            "source_aspect": target_aspect,
        })
    return valid


def _label_consistency_check(
    reviews: list[dict],
    llm_client: LLMClient,
) -> list[dict]:
    """
    Verify 100% generated reviews: LLM kiểm tra xem review có thực sự đề cập
    target aspect không. Loại bỏ những review không rõ ràng.
    
    Chi phí: ~len(reviews) API calls, mỗi call rất ngắn (~50 tokens).
    """
    passed = []
    removed = 0
    for review in reviews:
        target = review["source_aspect"]
        desc = WEAK_ASPECT_DESCRIPTIONS.get(target, target)
        messages = [{
            "role": "user",
            "content": _CONSISTENCY_PROMPT.format(
                aspect=target,
                description=desc[:100],
                review=review["review"],
            ),
        }]
        try:
            resp = llm_client.complete(messages, temperature=0.0, use_cache=False).strip().upper()
            if "NO" in resp:
                removed += 1
                continue
        except Exception:
            pass  # Nếu lỗi → giữ lại (conservative)
        passed.append(review)
        time.sleep(0.3)  # Tránh rate limit

    print(f"  [Consistency check] {len(reviews)} → {len(passed)} (removed {removed})")
    return passed


# ─── Core generation ──────────────────────────────────────────────────────────

def generate_for_aspect(
    train_df: pd.DataFrame,
    target_aspect: str,
    llm_client: LLMClient,
    n_total: int = 60,
    batch_size: int = 6,
) -> list[dict]:
    """
    Sinh n_total reviews cho target_aspect.
    Sentiment ratio: 40% pos, 35% neg, 25% neu (tổng = 100%).
    """
    system_prompt = _SYSTEM_PROMPT.format(
        aspect_list="\n".join(f"  - {a}" for a in ASPECT_COLUMNS)
    )
    description = WEAK_ASPECT_DESCRIPTIONS.get(target_aspect, target_aspect)

    # Kế hoạch sentiment: n_pos + n_neg + n_neu = n_total
    n_pos_total = round(n_total * 0.40)
    n_neg_total = round(n_total * 0.35)
    n_neu_total = n_total - n_pos_total - n_neg_total  # phần còn lại

    all_reviews = []
    n_batches = (n_total + batch_size - 1) // batch_size

    for batch_idx in range(n_batches):
        remaining = n_total - len(all_reviews)
        if remaining <= 0:
            break
        current = min(batch_size, remaining)

        # Phân chia sentiment cho batch này
        n_pos = round(current * 0.40)
        n_neg = round(current * 0.35)
        n_neu = current - n_pos - n_neg

        real_examples = _get_real_examples(
            train_df, target_aspect, n=3,
            random_state=batch_idx * 19 + 7,
        )

        already_section = ""
        if all_reviews:
            texts = "\n".join(f"- {r['review'][:100]}" for r in all_reviews[-8:])
            already_section = _ALREADY_SECTION.format(texts=texts)

        user_prompt = _USER_PROMPT.format(
            batch_size=current,
            target_aspect=target_aspect,
            description=description,
            n_pos=n_pos,
            n_neg=n_neg,
            n_neu=n_neu,
            real_examples=real_examples,
            already_section=already_section,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            resp = llm_client.complete(messages, temperature=0.85, use_cache=False)
            parsed = _parse_response(resp, target_aspect)
            all_reviews.extend(parsed)
            print(
                f"  [{target_aspect}] Batch {batch_idx+1}/{n_batches}: "
                f"got {len(parsed)} (total: {len(all_reviews)})"
            )
        except Exception as exc:
            print(f"  [{target_aspect}] Batch {batch_idx+1} FAILED: {exc}")
        time.sleep(1.5)

    return all_reviews[:n_total]


# ─── Main entry point ─────────────────────────────────────────────────────────

def run_augmentation_v2(
    train_df: pd.DataFrame,
    llm_client: LLMClient,
    n_per_aspect: int = 60,
    raw_output_path: str = "data/augmented_reviews_v2.json",
    run_consistency_check: bool = True,
) -> list[dict]:
    """
    Sinh augmented data cho tất cả WEAK_ASPECTS.

    Args:
        train_df: DataFrame gốc (train_preprocessed.csv)
        llm_client: LLMClient đã khởi tạo với API key
        n_per_aspect: số reviews sinh mỗi aspect (default 60)
        raw_output_path: nơi lưu kết quả raw (trước filter)
        run_consistency_check: có chạy label consistency check không

    Returns:
        list of dicts với keys: review, labels, source_aspect
    """
    all_augmented = []

    for aspect in WEAK_ASPECTS:
        print(f"\n{'='*60}")
        print(f"Generating {n_per_aspect} reviews for: {aspect}")
        print(f"{'='*60}")

        reviews = generate_for_aspect(
            train_df=train_df,
            target_aspect=aspect,
            llm_client=llm_client,
            n_total=n_per_aspect,
        )
        print(f"  -> Generated: {len(reviews)}")

        if run_consistency_check and reviews:
            print(f"  -> Running label consistency check...")
            reviews = _label_consistency_check(reviews, llm_client)

        all_augmented.extend(reviews)
        print(f"  -> Kept: {len(reviews)}")

    os.makedirs(os.path.dirname(raw_output_path) or ".", exist_ok=True)
    with open(raw_output_path, "w", encoding="utf-8") as f:
        json.dump(all_augmented, f, ensure_ascii=False, indent=2)

    print(f"\nTotal: {len(all_augmented)} reviews -> {raw_output_path}")
    return all_augmented
```

- [ ] **Step 2: Verify file syntax**

```bash
python3 -c "import ast; ast.parse(open('code/week3_part2/augmentor_v2.py').read()); print('Syntax OK')"
```

Expected: `Syntax OK`

- [ ] **Step 3: Verify imports (không cần API key)**

```bash
cd /path/to/absa-vlsp2018-hotel
python3 -c "
import sys
sys.path.insert(0, 'code/week1')
sys.path.insert(0, 'code/week3')
sys.path.insert(0, 'code/week3_part2')
from augmentor_v2 import WEAK_ASPECT_DESCRIPTIONS, run_augmentation_v2, WEAK_ASPECTS
from utils.constants import WEAK_ASPECTS as WA
assert set(WEAK_ASPECTS) == set(WA), 'WEAK_ASPECTS mismatch'
print('Imports OK, WEAK_ASPECTS:', len(WEAK_ASPECTS))
"
```

Expected: `Imports OK, WEAK_ASPECTS: 9`

- [ ] **Step 4: Commit**

```bash
git add code/week3_part2/augmentor_v2.py
git commit -m "feat: augmentor_v2 — target WEAK_ASPECTS with label consistency check"
```

---

### Task 3: Sinh augmented data (chạy local với notebook hoặc script)

**Files:**
- Modify: `notebooks/week3_part2_data_augmentation.ipynb` — thêm cells mới ở cuối
- Generated: `data/augmented_reviews_v2.json`, `data/augmented_reviews_v2_filtered.json`

> ⚠️ Task này cần OPENAI_API_KEY. Ước tính chi phí: 9 aspects × 60 samples × ~300 tokens/sample + consistency check ~540 × 80 tokens ≈ ~200K tokens input ≈ **~$0.03 với GPT-4o-mini**. Tổng chi phí thực tế < $0.10.

- [ ] **Step 1: Mở notebook week3_part2_data_augmentation.ipynb**

Cuộn xuống dưới cùng. Thêm cell mới:

```python
# Cell NEW-A — Sinh augmented data v2 (targeting WEAK_ASPECTS)
import os, sys
sys.path.insert(0, '../code/week1')
sys.path.insert(0, '../code/week3')
sys.path.insert(0, '../code/week3_part2')

import pandas as pd
from llm_client import LLMClient
from augmentor_v2 import run_augmentation_v2

# Load train data gốc
train_df = pd.read_csv('../data/train_preprocessed.csv')
print(f"Train: {len(train_df)} samples")

# Init LLM client
api_key = os.environ.get("OPENAI_API_KEY", "")
assert api_key, "Set OPENAI_API_KEY env var"
llm = LLMClient(provider="openai", api_key=api_key)

# Sinh data — ~15-25 phút
augmented_v2 = run_augmentation_v2(
    train_df=train_df,
    llm_client=llm,
    n_per_aspect=60,
    raw_output_path="../data/augmented_reviews_v2.json",
    run_consistency_check=True,
)
print(f"\nDone: {len(augmented_v2)} reviews saved")
```

- [ ] **Step 2: Thêm cell filter (dùng augment_filter.py đã có)**

```python
# Cell NEW-B — Filter augmented_reviews_v2.json
from augment_filter import heuristic_filter, dedup_filter
import json

with open('../data/augmented_reviews_v2.json') as f:
    raw = json.load(f)
print(f"Raw: {len(raw)}")

# Heuristic filter (length, Vietnamese chars, exact dup với train)
after_heuristic = heuristic_filter(raw, train_df)

# Dedup filter (Jaccard char 3-gram > 0.95)
after_dedup = dedup_filter(after_heuristic, threshold=0.90)  # threshold chặt hơn v1

with open('../data/augmented_reviews_v2_filtered.json', 'w') as f:
    json.dump(after_dedup, f, ensure_ascii=False, indent=2)

print(f"After filter: {len(after_dedup)} reviews")

# Distribution per aspect
from collections import Counter
aspect_counts = Counter(r['source_aspect'] for r in after_dedup)
for asp, cnt in sorted(aspect_counts.items()):
    print(f"  {asp}: {cnt}")
```

- [ ] **Step 3: Thêm cell VnCoreNLP segment + merge**

```python
# Cell NEW-C — VnCoreNLP segment + merge thành train_augmented_v2.csv
# (VnCoreNLP chỉ chạy được local, không trên Kaggle)
import json
import pandas as pd
from utils.constants import ASPECT_COLUMNS, LABEL_TO_IDX

# Load VnCoreNLP (đường dẫn local)
from py_vncorenlp import VnCoreNLP
rdrsegmenter = VnCoreNLP(
    save_dir='../vncorenlp',
    annotators=["wseg"],
    max_heap_size='-Xmx2g'
)

def segment_text(text: str) -> str:
    try:
        sents = rdrsegmenter.word_segment(text)
        return " ".join(sents) if sents else text
    except Exception:
        return text

with open('../data/augmented_reviews_v2_filtered.json') as f:
    filtered = json.load(f)

# Build DataFrame từ filtered reviews
rows = []
for item in filtered:
    row = {
        "Review": item["review"],
        "processed_review": segment_text(item["review"]),
    }
    for asp in ASPECT_COLUMNS:
        sentiment = item["labels"].get(asp, "absent")
        row[asp] = LABEL_TO_IDX.get(sentiment, 0)
    rows.append(row)

aug_df = pd.DataFrame(rows)
orig_df = pd.read_csv('../data/train_preprocessed.csv')

# Đảm bảo cùng cột với orig_df
aug_df = aug_df.reindex(columns=orig_df.columns, fill_value=0)

merged = pd.concat([orig_df, aug_df], ignore_index=True)
merged = merged.sample(frac=1, random_state=42).reset_index(drop=True)
merged.to_csv('../data/train_augmented_v2.csv', index=False)

print(f"train_augmented_v2.csv: {len(merged)} samples")
print(f"  Original: {len(orig_df)}, Augmented: {len(aug_df)}")
```

- [ ] **Step 4: Verify file tồn tại và đúng format**

```bash
python3 -c "
import pandas as pd
df = pd.read_csv('data/train_augmented_v2.csv')
print(f'Shape: {df.shape}')
assert 'Review' in df.columns
assert 'processed_review' in df.columns
assert len(df) > 3000, f'Expected >3000, got {len(df)}'
print('OK')
"
```

Expected: `Shape: (3XXX, 36)` (số hàng > 3000)

- [ ] **Step 5: Commit data file**

```bash
git add data/train_augmented_v2.csv data/augmented_reviews_v2_filtered.json
git commit -m "data: train_augmented_v2.csv — 9 WEAK_ASPECTS targeted, VnCoreNLP segmented"
```

---

### Task 4: Cập nhật Kaggle notebook để train với train_augmented_v2.csv

**Files:**
- Modify: `notebooks/week4_demo.ipynb` — sửa Cell 4 và Cell 6

- [ ] **Step 1: Mở week4_demo.ipynb, tìm Cell 4 (Verify data files)**

Cell 4 hiện tại verify `train_augmented.csv`. Thêm `train_augmented_v2.csv` vào danh sách:

```python
# Cell 4 — Verify data files (sau khi sửa)
import os

required_files = [
    'data/train_augmented_v2.csv',    # <-- dùng v2 thay v1
    'data/dev_preprocessed.csv',
    'data/test_preprocessed.csv',
    'outputs/eda/class_weights.json',
    'outputs/eda/encoder_config.json',
]

for f in required_files:
    exists = os.path.exists(f)
    size = os.path.getsize(f) if exists else 0
    status = "OK" if exists else "MISSING"
    print(f"[{status}] {f} ({size:,} bytes)")

assert all(os.path.exists(f) for f in required_files), "Missing files!"
print("\nAll files verified.")
```

- [ ] **Step 2: Tìm Cell 6 (Train PhoBERT), sửa train_path**

Tìm dòng có `train_path` hoặc `train_augmented.csv` trong Cell 6, đổi thành:

```python
TRAIN_PATH = 'data/train_augmented_v2.csv'   # <-- đổi từ train_augmented.csv
SAVE_DIR   = 'outputs/results/week4_augmented_v2'
```

Và update `save_dir` tương ứng trong lời gọi `retrain_phobert()` hoặc `train_fn()`.

- [ ] **Step 3: Verify notebook cells đã sửa đúng**

```bash
python3 -c "
import json
with open('notebooks/week4_demo.ipynb') as f:
    nb = json.load(f)
cells = nb['cells']
found_v2 = False
for i, cell in enumerate(cells):
    src = ''.join(cell.get('source', []))
    if 'train_augmented_v2' in src:
        print(f'Cell {i}: found train_augmented_v2')
        found_v2 = True
assert found_v2, 'train_augmented_v2 not found in notebook!'
print('OK')
"
```

- [ ] **Step 4: Commit notebook**

```bash
git add notebooks/week4_demo.ipynb
git commit -m "feat: update week4_demo to use train_augmented_v2.csv for retraining"
```

---

### Task 5: Train trên Kaggle + evaluate

> ⚠️ Task này thực hiện trên Kaggle, không phải local.

- [ ] **Step 1: Lên Kaggle, mở notebook `week4_demo.ipynb`**
  - Settings → Accelerator → **GPU T4 x2**
  - Add-ons → Secrets → Đảm bảo có `OPENAI_API_KEY`

- [ ] **Step 2: Cell 2 — git pull để lấy code mới nhất**

```python
# Cell 2 chạy lệnh:
!git pull origin master
```

Verify output có các file mới: `augmentor_v2.py`, `train_augmented_v2.csv`.

- [ ] **Step 3: Chạy Cell 1 → Cell 6 theo thứ tự**

Cell 6 (train) sẽ mất ~60-90 phút. Kết quả save tại `outputs/results/week4_augmented_v2/`.

- [ ] **Step 4: Cell 7 — so sánh baseline vs augmented v2**

Cell 7 tự động đọc từ JSON files. Cần đảm bảo `w4_candidates` trong Cell 7 include đường dẫn mới:

```python
w4_candidates = [
    'outputs/results/week4_augmented_v2/week4_test_metrics.json',
    'outputs/results/week4_augmented_v2/test_metrics.json',
    'outputs/results/week4_augmented/week4_test_metrics.json',  # fallback v1
]
```

- [ ] **Step 5: Ghi lại kết quả — điền vào bảng CHANGELOG.md**

Sau khi train xong, copy kết quả ACD F1, SPC F1, Combined F1 từ output Cell 7.

Mở `CHANGELOG.md`, cập nhật bảng kết quả:

```markdown
| PhoBERT + augmented v2 (WEAK_ASPECTS) | X.XXXX | X.XXXX | X.XXXX | Error-driven targeting |
```

---

## PHASE C — Inference Cascade (PhoBERT + LLM)

> Phase C bắt đầu SAU KHI Phase A hoàn thành và có checkpoint `week4_augmented_v2/models/best_model.pt`.

### Task 6: Tạo cascade_predictor.py

**Files:**
- Create: `code/week3_part2/cascade_predictor.py`

Logic cascade:
1. PhoBERT predict → lấy softmax confidence per head
2. Check: bất kỳ aspect nào trong `WEAK_ASPECTS` có confidence < threshold?
3. Nếu có → gọi LLM RAG k=4 → lấy prediction của aspect đó từ LLM
4. Kết hợp: confident aspects từ PhoBERT, uncertain aspects từ LLM

- [ ] **Step 1: Tạo file cascade_predictor.py**

```python
# code/week3_part2/cascade_predictor.py
"""
cascade_predictor.py — Hybrid inference: PhoBERT + LLM cascade.

Luồng:
    review
      → PhoBERT predict 34 aspects + confidence scores (max softmax per head)
      → Nếu bất kỳ WEAK_ASPECT nào có confidence < threshold
          → LLM RAG k=4 predict
          → Override PhoBERT prediction cho aspect đó với LLM prediction
      → Output: final 34-class prediction array

Chi phí LLM: chỉ gọi khi model uncertain trên WEAK_ASPECTS.
Ước tính: ~30-50% reviews trên test set sẽ trigger LLM call
(vì model yếu trên WEAK_ASPECTS → thường uncertain cho các aspect này).
"""

import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "week1"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "week3"))

from utils.constants import ASPECT_COLUMNS, WEAK_ASPECTS  # noqa: E402
from prompts import build_prompt, parse_llm_output, labels_dict_to_array  # noqa: E402
from llm_client import LLMClient  # noqa: E402
from rag_retriever import ABSARetriever  # noqa: E402
from icl_predictor import df_to_examples  # noqa: E402

import pandas as pd
import time

# Index của WEAK_ASPECTS trong ASPECT_COLUMNS (dùng để kiểm tra uncertain)
WEAK_ASPECT_INDICES = [ASPECT_COLUMNS.index(a) for a in WEAK_ASPECTS if a in ASPECT_COLUMNS]


def predict_single_phobert(
    processed_text: str,
    model: "ABSAPhoBERT",
    tokenizer,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Predict 1 review với PhoBERT.

    Returns:
        preds: shape [34] — argmax per head (int, 0-3)
        confidences: shape [34] — max softmax score per head (float, 0-1)
    """
    inputs = tokenizer(
        processed_text,
        max_length=256,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    # Lọc token_type_ids vì ABSAPhoBERT không dùng (RoBERTa-based)
    inputs = {k: v.to(device) for k, v in inputs.items() if k in ("input_ids", "attention_mask")}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs["logits"]  # list of 34 tensors [1, 4]
    preds = np.array([l.argmax(dim=-1)[0].item() for l in logits])  # [34]
    confidences = np.array([
        F.softmax(l, dim=-1)[0].max().item() for l in logits
    ])  # [34]

    return preds, confidences


def predict_with_cascade(
    review_text: str,
    processed_text: str,
    model: "ABSAPhoBERT",
    tokenizer,
    device: torch.device,
    train_df: pd.DataFrame,
    retriever: ABSARetriever,
    llm_client: LLMClient,
    threshold: float = 0.60,
    k: int = 4,
    sleep_sec: float = 5.0,
) -> tuple[np.ndarray, bool]:
    """
    Cascade prediction cho 1 review.

    Args:
        review_text: review gốc (chưa segment) — dùng cho LLM
        processed_text: review đã VnCoreNLP segment — dùng cho PhoBERT + RAG retrieval
        threshold: confidence < threshold → uncertain → trigger LLM
        k: số examples cho RAG

    Returns:
        final_preds: shape [34] — prediction array
        used_llm: True nếu LLM được gọi
    """
    # Step 1: PhoBERT predict
    phobert_preds, confidences = predict_single_phobert(
        processed_text, model, tokenizer, device
    )

    # Step 2: Check uncertain weak aspects
    uncertain_weak = [
        i for i in WEAK_ASPECT_INDICES
        if confidences[i] < threshold
    ]

    if not uncertain_weak:
        return phobert_preds, False

    # Step 3: LLM RAG predict
    try:
        indices = retriever.retrieve(query=processed_text, k=k, aspect_aware=True)
        examples = df_to_examples(train_df, indices)
        messages = build_prompt(processed_text, examples)
        raw = llm_client.complete(messages, temperature=0.0, use_cache=True)
        llm_pred_dict = parse_llm_output(raw)
        llm_preds = labels_dict_to_array(llm_pred_dict)  # shape [34]
        time.sleep(sleep_sec)
    except Exception as exc:
        print(f"  [Cascade] LLM call failed: {exc} — using PhoBERT only")
        return phobert_preds, False

    # Step 4: Combine — override PhoBERT với LLM chỉ cho uncertain weak aspects
    final_preds = phobert_preds.copy()
    for i in uncertain_weak:
        # Chỉ override nếu LLM predict non-absent (tránh LLM hallucinate absent)
        if llm_preds[i] != 0:
            final_preds[i] = llm_preds[i]

    return final_preds, True


def run_cascade_on_test(
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    model: "ABSAPhoBERT",
    tokenizer,
    device: torch.device,
    retriever: ABSARetriever,
    llm_client: LLMClient,
    threshold: float = 0.60,
    k: int = 4,
    sleep_sec: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Chạy cascade prediction trên toàn bộ test set.

    Returns:
        y_true: shape [N, 34]
        y_pred: shape [N, 34]
        stats: dict với llm_call_count, total, trigger_rate
    """
    y_true_list, y_pred_list = [], []
    llm_call_count = 0

    for idx, (_, row) in enumerate(test_df.iterrows()):
        review_text = str(row.get("Review", ""))
        processed = str(row.get("processed_review", review_text))

        final_preds, used_llm = predict_with_cascade(
            review_text=review_text,
            processed_text=processed,
            model=model,
            tokenizer=tokenizer,
            device=device,
            train_df=train_df,
            retriever=retriever,
            llm_client=llm_client,
            threshold=threshold,
            k=k,
            sleep_sec=sleep_sec,
        )

        # Ground truth
        y_true = np.array([int(row[a]) for a in ASPECT_COLUMNS])

        y_true_list.append(y_true)
        y_pred_list.append(final_preds)
        if used_llm:
            llm_call_count += 1

        if (idx + 1) % 50 == 0:
            print(f"  [{idx+1}/{len(test_df)}] LLM calls: {llm_call_count}")

    y_true = np.array(y_true_list)   # [N, 34]
    y_pred = np.array(y_pred_list)   # [N, 34]
    stats = {
        "llm_call_count": llm_call_count,
        "total": len(test_df),
        "trigger_rate": llm_call_count / len(test_df),
    }
    return y_true, y_pred, stats
```

- [ ] **Step 2: Verify syntax**

```bash
python3 -c "import ast; ast.parse(open('code/week3_part2/cascade_predictor.py').read()); print('Syntax OK')"
```

Expected: `Syntax OK`

- [ ] **Step 3: Verify imports (không cần model)**

```bash
python3 -c "
import sys
sys.path.insert(0, 'code/week1')
sys.path.insert(0, 'code/week3')
sys.path.insert(0, 'code/week3_part2')
from cascade_predictor import WEAK_ASPECT_INDICES, predict_single_phobert, run_cascade_on_test
print('WEAK_ASPECT_INDICES:', WEAK_ASPECT_INDICES)
assert len(WEAK_ASPECT_INDICES) == 9
print('Imports OK')
"
```

Expected: 9 indices, không có ImportError.

- [ ] **Step 4: Commit**

```bash
git add code/week3_part2/cascade_predictor.py
git commit -m "feat: cascade_predictor — PhoBERT + LLM override for uncertain WEAK_ASPECTS"
```

---

### Task 7: Thêm Cell cascade evaluation vào week4_demo.ipynb

**Files:**
- Modify: `notebooks/week4_demo.ipynb` — thêm Cell 11

- [ ] **Step 1: Thêm cell mới sau Cell 10 (Save results)**

```python
# Cell 11 — Cascade evaluation: PhoBERT + LLM hybrid
# ⚠️ Chạy sau Cell 8 (model đã load) và Cell 3 (API key đã set)
# ⚠️ Chi phí API: ~$0.30-0.50 cho 600 test reviews nếu trigger rate ~40%

import sys, os
sys.path.insert(0, f'{REPO_ROOT}/code/week3')
sys.path.insert(0, f'{REPO_ROOT}/code/week3_part2')

from rag_retriever import ABSARetriever
from llm_client import LLMClient
from cascade_predictor import run_cascade_on_test
from step4_eval import evaluate_predictions

# Init RAG retriever (dùng train_augmented_v2.csv để retrieve)
train_df = pd.read_csv(f'{REPO_ROOT}/data/train_augmented_v2.csv')
retriever = ABSARetriever()
retriever.build_index(train_df)

# Init LLM client
llm_client = LLMClient(provider="openai", api_key=os.environ["OPENAI_API_KEY"])

# Load test data
test_df = pd.read_csv(f'{REPO_ROOT}/data/test_preprocessed.csv')

# Chạy cascade (threshold=0.60, k=4)
# sleep_sec=5.0 để tránh rate limit GPT-4o-mini
print("Running cascade on test set (this will take ~60-90 mins)...")
y_true, y_pred, stats = run_cascade_on_test(
    test_df=test_df,
    train_df=train_df,
    model=model_inf,       # từ Cell 8
    tokenizer=tokenizer,   # từ Cell 8
    device=device,
    retriever=retriever,
    llm_client=llm_client,
    threshold=0.60,
    k=4,
    sleep_sec=5.0,
)

print(f"\nLLM trigger rate: {stats['trigger_rate']:.1%} ({stats['llm_call_count']}/{stats['total']})")

# Evaluate
cascade_metrics = evaluate_predictions(y_true, y_pred)
print(f"\nCascade Results:")
print(f"  ACD F1:      {cascade_metrics['macro_acd_f1']:.4f}")
print(f"  SPC F1:      {cascade_metrics['macro_spc_f1']:.4f}")
print(f"  Combined F1: {cascade_metrics['macro_combined_f1']:.4f}")

# So sánh với PhoBERT alone (từ Cell 7)
print(f"\nComparison:")
print(f"  {'Metric':<20} {'PhoBERT':>10} {'Cascade':>10} {'Delta':>8}")
print(f"  {'-'*50}")
for key, label in [('macro_acd_f1','ACD F1'), ('macro_spc_f1','SPC F1'), ('macro_combined_f1','Combined F1')]:
    v_phobert = w4_m.get(key, 0) if w4_m else 0
    v_cascade = cascade_metrics.get(key, 0)
    print(f"  {label:<20} {v_phobert:>10.4f} {v_cascade:>10.4f} {v_cascade-v_phobert:>+8.4f}")

# Save cascade metrics
import json
cascade_save_path = f'outputs/results/week4_augmented_v2/cascade_metrics.json'
os.makedirs(os.path.dirname(cascade_save_path), exist_ok=True)
with open(cascade_save_path, 'w') as f:
    json.dump({**cascade_metrics, **stats}, f, indent=2)
print(f"\nSaved to {cascade_save_path}")
```

- [ ] **Step 2: Commit notebook**

```bash
git add notebooks/week4_demo.ipynb
git commit -m "feat: add Cell 11 — cascade evaluation (PhoBERT + LLM hybrid)"
```

- [ ] **Step 3: Push lên GitHub**

```bash
git push origin master
```

---

### Task 8: Chạy cascade trên Kaggle + cập nhật CHANGELOG

> ⚠️ Chạy Cell 11 trên Kaggle sau khi Cell 8 đã complete (model đã load).
> Cascade trên 600 reviews × ~40% trigger × 5s/call ≈ ~20 phút.
> Chi phí: ước tính $0.20-0.40.

- [ ] **Step 1: Chạy Cell 11 trên Kaggle**

Đợi output. Ghi lại:
- Combined F1 của Cascade
- LLM trigger rate (% reviews trigger LLM call)

- [ ] **Step 2: Cập nhật CHANGELOG.md — bảng kết quả**

Mở `CHANGELOG.md`, điền vào bảng cuối:

```markdown
| PhoBERT + augmented v2 | X.XXXX | X.XXXX | X.XXXX | Error-driven, 9 WEAK_ASPECTS |
| PhoBERT+LLM Cascade    | X.XXXX | X.XXXX | X.XXXX | Hybrid inference, ~X% LLM trigger |
```

- [ ] **Step 3: Cập nhật section "Trạng thái hiện tại" trong CHANGELOG.md**

```markdown
## Trạng thái hiện tại (2026-04-09 — updated)

### Đã hoàn thành (Phase A + C)
- [x] WEAK_ASPECTS constant (9 aspects F1<0.35)
- [x] augmentor_v2.py — error-driven targeting + label consistency check
- [x] train_augmented_v2.csv — 9 WEAK_ASPECTS × 60 samples
- [x] PhoBERT retrained với augmented v2
- [x] cascade_predictor.py — PhoBERT + LLM cascade
- [x] Evaluation: PhoBERT v2 vs Cascade vs Baseline

### Kết quả cuối cùng
Best system: [điền sau khi có kết quả]
```

- [ ] **Step 4: Commit final**

```bash
git add CHANGELOG.md
git commit -m "docs: update CHANGELOG with Phase A+C results"
git push origin master
```

---

## Tóm tắt thứ tự thực hiện

```
Task 1 (local, 5 phút)   → Add WEAK_ASPECTS constant
Task 2 (local, 10 phút)  → Tạo augmentor_v2.py
Task 3 (local, 20-30 phút, cần API key + VnCoreNLP) → Sinh data
Task 4 (local, 5 phút)   → Update notebook
Task 5 (Kaggle, 60-90 phút train) → Retrain + evaluate Phase A
Task 6 (local, 10 phút)  → Tạo cascade_predictor.py
Task 7 (local, 10 phút)  → Add Cell 11 vào notebook
Task 8 (Kaggle, 20-30 phút) → Chạy cascade + ghi kết quả
```

**Ước tính tổng chi phí API:** < $0.50 (GPT-4o-mini)
**Ước tính tổng thời gian GPU:** 1 lần train Kaggle (~90 phút)
