# CODE REVIEW DIRECTIVE — Tuần 3: LLM Few-shot + Aspect-aware RAG
## Dự án: ABSA VLSP 2018 Hotel | NLP Course — HUST

> **Dành cho Claude Code.**
> Đọc TOÀN BỘ file này trước khi làm bất cứ điều gì.
> Nhiệm vụ: **REVIEW và FIX** code tuần 3 đã có — không viết lại từ đầu.

---

## 0. Bối cảnh

Code tuần 3 đã được implement sẵn trong `code/week3/`. Nhiệm vụ của bạn là:
1. Đọc từng file
2. Kiểm tra đối chiếu với spec bên dưới
3. Phát hiện bugs / thiếu sót / code không chuyên nghiệp
4. Fix tất cả vấn đề tìm được
5. Chạy verification tests để xác nhận mọi thứ hoạt động

---

## 1. Danh sách file cần review

```
code/week3/
├── __init__.py
├── prompts.py
├── llm_client.py
├── icl_predictor.py
├── rag_retriever.py
├── rag_predictor.py
├── run_tier2.py
├── run_tier3.py
└── compare_results.py
```

Với mỗi file, thực hiện theo thứ tự:
```
view file → đối chiếu spec → ghi nhận vấn đề → fix
```

---

## 2. Checklist review từng file

### 2.1 `prompts.py`

**Kiểm tra:**
- [ ] `SYSTEM_PROMPT` có đủ 34 aspects từ `ASPECT_COLUMNS` không?
- [ ] `RARE_ASPECTS` được liệt kê riêng với ghi chú "chỉ đưa vào nếu chắc chắn"?
- [ ] `format_example()` sinh CoT đúng format không? Khi `labels={}` có xử lý gracefully không?
- [ ] `build_prompt()` tạo đúng cấu trúc `[system, user, assistant, ..., user]` không?
  - System prompt phải là message đầu tiên
  - Mỗi few-shot example phải là cặp `user/assistant`
  - Test review phải là message cuối cùng với role `user`
- [ ] `parse_llm_output()` xử lý đủ các edge cases:
  - JSON chuẩn: `{"SERVICE#GENERAL": "positive"}`
  - Có markdown fence: ` ```json {...} ``` `
  - Có CoT text trước JSON
  - JSON rỗng: `{}`
  - Output hoàn toàn không có JSON → trả về `{}`
  - Aspect không hợp lệ → bị filter ra
  - Sentiment không hợp lệ → bị filter ra
- [ ] `labels_dict_to_array()` map đúng index trong `ASPECT_COLUMNS` và dùng `LABEL_TO_IDX`?

**Fix bắt buộc nếu phát hiện:**
- parse_llm_output không handle nested JSON `{"a": {"b": "c"}}` → thêm safeguard
- Thiếu validation sentiment value trước khi map
- CoT format trong `format_example` bị lỗi với aspects có tên phức tạp

---

### 2.2 `llm_client.py`

**Kiểm tra:**
- [ ] Cache dùng `MD5(json.dumps(messages))` — đảm bảo `sort_keys=True` để cache hit nhất quán?
- [ ] Retry logic: exponential backoff `delay * 2^attempt` đúng không?
- [ ] Khi API fail hết retries: trả về `"{}"` (string) hay `{}` (dict)?
  - **Phải trả về string `"{}"` vì `complete()` luôn trả về str**
- [ ] Gemini message conversion: system prompt có được ghép vào đúng chỗ không?
  - Nếu có history: system phải được prepend vào `history[0]["parts"][0]`
  - Nếu không có history: system + last_user message
- [ ] `get_usage_stats()` có tính đúng cost estimate không?
  - GPT-4o-mini: `$0.15/1M input tokens` (kiểm tra xem có dùng giá input hay output không)
- [ ] Rate limiting: có `time.sleep()` giữa các requests không? Hoặc để caller tự handle?

**Fix bắt buộc nếu phát hiện:**
- Gemini không handle system prompt trong multi-turn conversation
- Cache key không deterministic (thiếu `sort_keys`)
- Missing `time.sleep` hoặc rate limit hoàn toàn

---

### 2.3 `icl_predictor.py`

**Kiểm tra:**
- [ ] `df_to_examples()` filter đúng `val != 0` để chỉ lấy PRESENT aspects?
- [ ] `IDX_TO_LABEL` được import và map đúng `{1: "positive", 2: "negative", 3: "neutral"}`?
- [ ] `predict_icl()` có `set_seed()` trước `random.sample()` không? Seed phải được set mỗi lần gọi để đảm bảo reproducibility.
- [ ] Ground truth array: `[int(test_row[asp]) for asp in ASPECT_COLUMNS]` — đúng thứ tự?
- [ ] Rate limiting: `time.sleep(1.0)` sau mỗi 10 requests — có ổn không? Nên sleep sau **mỗi** request với Gemini free tier.
- [ ] `run_icl_ablation()` lưu metrics JSON đúng path `tier2_{provider}_k{k}_metrics.json`?
- [ ] In usage stats sau mỗi provider xong?

**Fix bắt buộc:**
- Nếu `set_seed` chỉ được gọi 1 lần ở đầu function thay vì reset mỗi experiment → fix để mỗi `(provider, k)` combo có seed riêng nhất quán

---

### 2.4 `rag_retriever.py`

**Kiểm tra:**
- [ ] Model mặc định: `keepitreal/vietnamese-sbert` với fallback `paraphrase-multilingual-MiniLM-L12-v2`?
- [ ] `fit()` có kiểm tra cache size match với `len(train_df)` trước khi dùng không?
- [ ] Embeddings được `normalize_embeddings=True` để dot product = cosine similarity?
- [ ] FAISS index dùng `IndexFlatIP` (Inner Product) — đúng vì đã normalize?
- [ ] `retrieve()` trả về `list[int]` là indices của `train_df` (sau `reset_index(drop=True)`)?
- [ ] `_aspect_aware_select()` greedy algorithm:
  - Lấy top-1 candidate trước tiên (giống nhất semantic)
  - Sau đó ưu tiên candidates tăng aspect coverage
  - Fill đủ k nếu cần
  - Không trả về ít hơn k (trừ khi candidates < k)
- [ ] Numpy fallback khi không có FAISS hoạt động đúng không?

**Fix bắt buộc:**
- Nếu `_aspect_aware_select` có thể trả về ít hơn k examples → thêm assertion hoặc fallback
- Nếu cache path directory không tồn tại → `os.makedirs` trước `np.save`

---

### 2.5 `rag_predictor.py`

**Kiểm tra:**
- [ ] `predict_rag()` gọi `retriever.retrieve()` với đúng text column (`processed_review`)?
- [ ] Rate limiting sau mỗi request?
- [ ] `run_rag_ablation()` tạo **1 retriever duy nhất** rồi dùng lại cho tất cả `(provider, k)` — không tạo lại mỗi lần (tốn thời gian embed lại)?
- [ ] Output file tên đúng `tier3_{provider}_k{k}_metrics.json`?

**Fix bắt buộc:**
- Nếu retriever được tạo mới bên trong vòng lặp `for k in k_values` → move ra ngoài

---

### 2.6 `compare_results.py`

**Kiểm tra:**
- [ ] Load được tất cả `tier2_*` và `tier3_*` JSON từ `outputs/results/`?
- [ ] Handle gracefully khi một số file chưa tồn tại (chỉ in kết quả của files có sẵn)?
- [ ] Bảng so sánh có đủ cột: `Method | Provider | k | ACD F1 | SPC F1 | Combined F1`?
- [ ] Có so sánh với PhoBERT baseline từ `week2_test_metrics.json` không?
- [ ] Bảng được sort theo `Combined F1` giảm dần?
- [ ] Output được lưu vào `outputs/results/week3_comparison.md`?

---

### 2.7 `run_tier2.py` và `run_tier3.py`

**Kiểm tra:**
- [ ] Import paths đúng (`sys.path.insert` trước khi import từ week1)?
- [ ] `main()` có `max_samples` parameter để test nhanh không?
- [ ] Error handling khi thiếu API key: in message rõ ràng thay vì crash?
- [ ] `run_tier3.py` gọi `generate_comparison_table()` sau khi chạy xong?

---

## 3. Verification Tests — Chạy sau khi fix xong

Chạy từng test block dưới đây. Tất cả phải **PASS** trước khi coi review xong.

### Test Block 1: Imports và constants

```python
# Chạy từ root directory của project
import sys
sys.path.insert(0, "code/week1")
sys.path.insert(0, "code/week3")

from utils.constants import ASPECT_COLUMNS, RARE_ASPECTS, LABEL_TO_IDX, IDX_TO_LABEL
assert len(ASPECT_COLUMNS) == 34, f"Cần 34 aspects, có {len(ASPECT_COLUMNS)}"
assert set(RARE_ASPECTS).issubset(set(ASPECT_COLUMNS)), "RARE_ASPECTS phải nằm trong ASPECT_COLUMNS"
assert LABEL_TO_IDX == {"absent": 0, "positive": 1, "negative": 2, "neutral": 3}
assert IDX_TO_LABEL == {0: "absent", 1: "positive", 2: "negative", 3: "neutral"}
print("✅ Test 1 PASSED: Imports OK")
```

### Test Block 2: parse_llm_output

```python
from prompts import parse_llm_output

# Case 1: JSON chuẩn
assert parse_llm_output('{"SERVICE#GENERAL": "positive"}') == {"SERVICE#GENERAL": "positive"}

# Case 2: Có markdown fence
assert parse_llm_output('```json\n{"HOTEL#GENERAL": "negative"}\n```') == {"HOTEL#GENERAL": "negative"}

# Case 3: Rỗng
assert parse_llm_output('{}') == {}

# Case 4: Output không có JSON
assert parse_llm_output('Tôi không hiểu yêu cầu.') == {}

# Case 5: Aspect không hợp lệ bị filter
result = parse_llm_output('{"INVALID#ASPECT": "positive", "SERVICE#GENERAL": "positive"}')
assert result == {"SERVICE#GENERAL": "positive"}, f"Phải filter invalid aspect, got: {result}"

# Case 6: Sentiment không hợp lệ bị filter
result = parse_llm_output('{"SERVICE#GENERAL": "very_positive"}')
assert result == {}, f"Phải filter invalid sentiment, got: {result}"

# Case 7: CoT trước JSON
cot_output = """Phân tích:
- SERVICE#GENERAL: đề cập dịch vụ tốt → positive
Output: {"SERVICE#GENERAL": "positive"}"""
assert parse_llm_output(cot_output) == {"SERVICE#GENERAL": "positive"}

print("✅ Test 2 PASSED: parse_llm_output OK")
```

### Test Block 3: labels_dict_to_array

```python
from prompts import labels_dict_to_array
from utils.constants import ASPECT_COLUMNS

# Tất cả absent
arr = labels_dict_to_array({})
assert arr == [0] * 34, "Dict rỗng phải cho array toàn 0"

# 1 aspect positive
asp = ASPECT_COLUMNS[0]  # aspect đầu tiên
arr = labels_dict_to_array({asp: "positive"})
assert arr[0] == 1, f"positive phải map sang 1, got {arr[0]}"
assert sum(arr) == 1, "Chỉ có 1 aspect được set"

# Kiểm tra index cuối
last_asp = ASPECT_COLUMNS[-1]
arr = labels_dict_to_array({last_asp: "negative"})
assert arr[-1] == 2, f"negative phải map sang 2, got {arr[-1]}"

print("✅ Test 3 PASSED: labels_dict_to_array OK")
```

### Test Block 4: build_prompt structure

```python
from prompts import build_prompt

examples = [
    {"review": "Phòng sạch, view đẹp", "labels": {"ROOM#CLEANLINESS": "positive"}},
    {"review": "Giá hơi cao", "labels": {"ROOM#PRICES": "negative"}},
]
messages = build_prompt("Khách sạn ổn nhưng service chậm", examples)

# Kiểm tra cấu trúc
assert messages[0]["role"] == "system", "Message đầu phải là system"
assert messages[-1]["role"] == "user", "Message cuối phải là user (test review)"

# Đếm số cặp user/assistant = số examples
user_msgs = [m for m in messages if m["role"] == "user"]
assistant_msgs = [m for m in messages if m["role"] == "assistant"]
assert len(assistant_msgs) == len(examples), \
    f"Phải có {len(examples)} assistant messages, có {len(assistant_msgs)}"

print("✅ Test 4 PASSED: build_prompt structure OK")
```

### Test Block 5: LLMClient cache

```python
import os, json, tempfile
from llm_client import LLMClient

# Test cache key deterministic
class MockClient(LLMClient):
    def __init__(self):
        self.cache_dir = tempfile.mkdtemp()
        self.provider = "openai"
        self.model = "gpt-4o-mini"
        self.max_retries = 3
        self.retry_delay = 2.0
        self.total_tokens = 0
        os.makedirs(self.cache_dir, exist_ok=True)

mock = MockClient()
msgs1 = [{"role": "user", "content": "Hello"}, {"role": "system", "content": "Be helpful"}]
msgs2 = [{"role": "system", "content": "Be helpful"}, {"role": "user", "content": "Hello"}]

key1 = mock._cache_key(msgs1)
key2 = mock._cache_key(msgs2)
# Note: cùng content nhưng thứ tự list khác thì key PHẢI khác (thứ tự messages quan trọng)
assert key1 != key2, "Messages với thứ tự khác phải có cache key khác"

# Test save/load cache
mock._save_cache("testkey123", "test response")
loaded = mock._load_cache("testkey123")
assert loaded == "test response", f"Cache load/save thất bại: {loaded}"

print("✅ Test 5 PASSED: LLMClient cache OK")
```

### Test Block 6: ABSARetriever

```python
import pandas as pd
import numpy as np
import tempfile
from rag_retriever import ABSARetriever
from utils.constants import ASPECT_COLUMNS

# Tạo dummy train data
n = 20
dummy_data = {"processed_review": [f"review số {i}" for i in range(n)]}
for asp in ASPECT_COLUMNS:
    dummy_data[asp] = [0] * n
# Thêm vài aspects khác nhau
dummy_data[ASPECT_COLUMNS[0]][0] = 1   # positive
dummy_data[ASPECT_COLUMNS[1]][1] = 2   # negative
dummy_data[ASPECT_COLUMNS[2]][2] = 3   # neutral

train_df = pd.DataFrame(dummy_data)
cache_path = os.path.join(tempfile.mkdtemp(), "test_cache.npy")

retriever = ABSARetriever(cache_path=cache_path)
retriever.fit(train_df)

# Test retrieve trả về đúng số lượng
indices = retriever.retrieve("phòng sạch đẹp", k=3)
assert len(indices) == 3, f"retrieve(k=3) phải trả về 3 indices, got {len(indices)}"
assert all(0 <= i < n for i in indices), "Indices phải nằm trong range train_df"
assert len(set(indices)) == 3, "Không được trùng lặp indices"

# Test cache được tạo
assert os.path.exists(cache_path), "Cache file phải được tạo sau fit()"

# Test load từ cache
retriever2 = ABSARetriever(cache_path=cache_path)
retriever2.fit(train_df)  # Phải load từ cache, không embed lại
assert retriever2.train_embeds is not None

print("✅ Test 6 PASSED: ABSARetriever OK")
```

### Test Block 7: df_to_examples

```python
from icl_predictor import df_to_examples
from utils.constants import ASPECT_COLUMNS, IDX_TO_LABEL

# Tạo dummy train row
dummy_data = {"processed_review": ["review A", "review B"]}
for asp in ASPECT_COLUMNS:
    dummy_data[asp] = [0, 0]
dummy_data[ASPECT_COLUMNS[0]][0] = 1  # positive
dummy_data[ASPECT_COLUMNS[1]][0] = 2  # negative

train_df = pd.DataFrame(dummy_data)
examples = df_to_examples(train_df, [0, 1])

# Row 0: có 2 aspects
assert len(examples[0]["labels"]) == 2, \
    f"Row 0 phải có 2 aspects, got {len(examples[0]['labels'])}"
assert examples[0]["labels"][ASPECT_COLUMNS[0]] == "positive"
assert examples[0]["labels"][ASPECT_COLUMNS[1]] == "negative"

# Row 1: không có aspect nào
assert examples[1]["labels"] == {}, \
    f"Row 1 phải có labels rỗng, got {examples[1]['labels']}"

print("✅ Test 7 PASSED: df_to_examples OK")
```

### Test Block 8: End-to-end mock (không cần API key)

```python
from unittest.mock import patch, MagicMock
from icl_predictor import predict_icl
from llm_client import LLMClient
import pandas as pd

# Mock LLMClient.complete để không gọi API thật
mock_response = '{"SERVICE#GENERAL": "positive"}'

# Tạo mini test data (5 samples)
n_test, n_train = 5, 20
test_data = {"processed_review": [f"test {i}" for i in range(n_test)]}
train_data = {"processed_review": [f"train {i}" for i in range(n_train)]}
for asp in ASPECT_COLUMNS:
    test_data[asp] = [0] * n_test
    train_data[asp] = [0] * n_train
test_data[ASPECT_COLUMNS[-1]] = [1] * n_test  # SERVICE#GENERAL positive

test_df = pd.DataFrame(test_data)
train_df = pd.DataFrame(train_data)

# Tạo mock client
mock_client = MagicMock(spec=LLMClient)
mock_client.complete.return_value = mock_response

y_true, y_pred = predict_icl(
    test_df=test_df,
    train_df=train_df,
    client=mock_client,
    k=2,
    max_samples=5,
)

assert y_true.shape == (5, 34), f"y_true shape sai: {y_true.shape}"
assert y_pred.shape == (5, 34), f"y_pred shape sai: {y_pred.shape}"
assert mock_client.complete.call_count == 5, \
    f"complete() phải được gọi 5 lần, gọi {mock_client.complete.call_count}"

print("✅ Test 8 PASSED: End-to-end mock OK")
```

---

## 4. Các vấn đề thường gặp cần chú ý đặc biệt

### 4.1 Gemini system prompt handling
Gemini API không có native system role như OpenAI. Code phải convert:
```python
# ĐÚng: ghép system vào user message
if system_content:
    if history:
        # Prepend vào history[0]
        history[0]["parts"][0] = system_content + "\n\n" + history[0]["parts"][0]
    else:
        last_user = system_content + "\n\n" + last_user
```
Nếu không làm vậy, system prompt bị bỏ qua hoàn toàn → model không biết mình đang làm gì.

### 4.2 Rate limiting đúng chỗ
```python
# ĐÚNG: sleep sau mỗi request
for i, row in enumerate(test_df.iterrows()):
    response = client.complete(messages)
    time.sleep(1.0)  # sau mỗi request

# SAI: sleep không đều
if i % 10 == 0:
    time.sleep(1.0)  # 10 requests liên tiếp → bị rate limit
```

### 4.3 Cache key và reproducibility
Seed phải được set **trước mỗi experiment** để đảm bảo cùng `(provider, k)` luôn chọn cùng random examples:
```python
for k in k_values:
    set_seed(42)  # Reset seed mỗi experiment
    y_true, y_pred = predict_icl(..., k=k)
```

### 4.4 ABSARetriever phải được khởi tạo 1 lần
```python
# ĐÚNG: fit 1 lần, dùng lại
retriever = ABSARetriever(cache_path=...)
retriever.fit(train_df)  # tốn ~30 giây

for provider in providers:
    for k in k_values:
        predict_rag(..., retriever=retriever, k=k)  # reuse

# SAI: tạo lại mỗi lần
for k in k_values:
    retriever = ABSARetriever(...)
    retriever.fit(train_df)  # embed lại 3 lần không cần thiết
```

### 4.5 JSON parse phải robust
LLM output không bao giờ guaranteed clean. `parse_llm_output` phải handle:
- Có text "Tôi phân tích review này..." trước JSON
- `{'key': 'value'}` với single quotes (Python dict format) → dùng `ast.literal_eval` fallback
- JSON ở trong nested object `{"analysis": "...", "result": {...}}`

---

## 5. Sau khi review và fix xong

Báo cáo theo format sau:

```
## Báo cáo Review Tuần 3

### Files đã review:
- [ ] prompts.py
- [ ] llm_client.py
- [ ] icl_predictor.py
- [ ] rag_retriever.py
- [ ] rag_predictor.py
- [ ] compare_results.py
- [ ] run_tier2.py / run_tier3.py

### Bugs tìm được và đã fix:
1. [file] [mô tả bug] → [fix áp dụng]
2. ...

### Verification tests:
- Test 1: ✅/❌
- Test 2: ✅/❌
- ...
- Test 8: ✅/❌

### Nhận xét tổng thể:
[Code đã sẵn sàng chạy / Cần thêm X trước khi chạy]
```

---

## 6. KHÔNG làm trong quá trình review

- ❌ Không viết lại toàn bộ file nếu chỉ có bug nhỏ — dùng `str_replace` để fix từng chỗ
- ❌ Không thay đổi function signatures (break compatibility với code khác)
- ❌ Không thêm dependencies mới ngoài danh sách requirements.txt
- ❌ Không xóa logging / print statements hữu ích
- ❌ Không hardcode API keys dù chỉ để test

---

*Review Directive version: 1.0*
*Mục tiêu: Code week3 sẵn sàng chạy trên Colab/Kaggle với kết quả reproducible*
