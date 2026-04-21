# ABSA VLSP 2018 Hotel - Detailed Flow Guide

## Tổng quan

Project này giải bài toán Aspect-Based Sentiment Analysis (ABSA) cho review khách sạn tiếng Việt. Mục tiêu là xác định review đang nói về khía cạnh nào (aspect) và cảm xúc tương ứng (sentiment).

---

# Tổng Pipeline

```text
Raw Dataset
   ↓
1. Data Preprocessing
   ↓
Clean Dataset
   ↓
2. Baseline SVM
3. PhoBERT
4. LLM + RAG
5. PhoBERT + LLM Explanation
   ↓
6. Compare Results
```

---

# 1. Data Preprocessing

## Mục tiêu

Biến dữ liệu thô thành dữ liệu sạch, chuẩn format để model học hiệu quả.

## Input

Dataset gốc VLSP gồm:

* review text
* label aspect + sentiment

Ví dụ:

```csv
review,label
"Phòng đẹp !!! nhân viên tốt ","ROOM#Positive;STAFF#Positive"
"Wifi yếu, giá cao :(","WIFI#Negative;PRICE#Negative"
```

## Flow xử lý chi tiết

### Step 1: Load dữ liệu

Đọc file dữ liệu bằng pandas.

```python
df = pd.read_csv(...)
```

### Step 2: Làm sạch text

* Lowercase
* Xóa khoảng trắng dư
* Xóa ký tự rác
* Chuẩn hóa unicode tiếng Việt

Ví dụ:

* `Phòng Đẹp` → `phòng đẹp`
* `hoàng` → `hoàng`

### Step 3: Tokenization (nếu dùng)

Tách từ tiếng Việt:

```text
nhân viên thân thiện
→ nhân_viên thân_thiện
```

Có thể dùng VnCoreNLP.

### Step 4: Parse labels

Từ:

```text
ROOM#Positive;STAFF#Negative
```

Thành:

```json
[
 {"aspect":"ROOM","sentiment":"Positive"},
 {"aspect":"STAFF","sentiment":"Negative"}
]
```

### Step 5: Chia tập dữ liệu

* train
* dev
* test

## Output

```text
data/train_preprocessed.csv
data/dev_preprocessed.csv
data/test_preprocessed.csv
```

## Ý nghĩa

Preprocessing tốt giúp model tăng accuracy đáng kể.

---

# 2. Baseline SVM

## Mục tiêu

Tạo baseline bằng Machine Learning truyền thống để so sánh.

## Input

* train_preprocessed.csv
* test_preprocessed.csv

## Flow xử lý chi tiết

### Step 1: Text → TF-IDF Vector

Ví dụ:

```text
phòng đẹp sạch
```

Thành vector số:

```text
[0.1, 0.6, 0.2, 0.0, ...]
```

### Step 2: Train SVM

Model học mối liên hệ giữa từ và nhãn.

Ví dụ:

* đẹp → Positive
* bẩn → Negative

### Step 3: Predict

Dự đoán trên test set.

### Step 4: Evaluate

Tính:

* Precision
* Recall
* F1-score

## Output

```text
Combined F1 ≈ 0.3173
```

## Ý nghĩa

Là mốc tối thiểu để đánh giá các model khác.

---

# 3. PhoBERT

## Mục tiêu

Dùng Transformer tiếng Việt để đạt hiệu quả cao.

## Input

* train_preprocessed.csv
* dev_preprocessed.csv
* test_preprocessed.csv

## Flow xử lý chi tiết

### Step 1: Tokenization

Text → token ids.

```text
phòng sạch đẹp
→ [101, 2345, 889, ...]
```

### Step 2: Embedding

Mỗi token thành vector nhiều chiều.

### Step 3: Transformer Encoder

Self-attention giúp hiểu context.

Ví dụ:

* `phòng không tệ` khác `phòng tệ`

### Step 4: CLS Vector

Lấy vector `[CLS]` đại diện toàn câu.

### Step 5: Classification Heads

* ACD: aspect nào xuất hiện
* SPC: sentiment là gì

### Step 6: Backpropagation

Cập nhật weight qua nhiều epoch.

### Step 7: Evaluate

Đánh giá trên test set.

## Output

```text
ACD F1      ≈ 0.6360
SPC F1      ≈ 0.4727
Combined F1 ≈ 0.5543
```

## Ý nghĩa

Là model dự đoán mạnh nhất trong project.

---

# 4. LLM + RAG

## Mục tiêu

Dùng LLM với Retrieval-Augmented Generation để dự đoán.

## Input

* Review text
* Example dataset / vector store

## Flow xử lý chi tiết

### Step 1: Embedding query

Review → vector semantic.

### Step 2: Retrieve examples gần giống

Lấy top-k reviews tương tự.

Ví dụ:

* phòng sạch, staff tệ
* nhân viên chậm, giá cao

### Step 3: Build Prompt

```text
Ví dụ 1: ...
Label ...

Ví dụ 2: ...
Label ...

Hãy dự đoán review mới...
```

### Step 4: Call LLM

Gửi prompt đến model LLM.

### Step 5: Parse Output

```json
{
 "ROOM":"Positive",
 "STAFF":"Negative"
}
```

### Step 6: Evaluate

So sánh với ground truth.

## Output

Prediction files + metrics.

## Ý nghĩa

Thể hiện hướng hiện đại dùng prompt thay vì train sâu.

---

# 5. PhoBERT + LLM Explanation

## Mục tiêu

Kết hợp sức mạnh:

* PhoBERT dự đoán tốt
* LLM giải thích tốt

## Input

### Review

```text
phòng sạch nhưng nhân viên chậm
```

### Prediction từ PhoBERT

```json
{
 "ROOM":"Positive",
 "STAFF":"Negative"
}
```

## Flow xử lý chi tiết

### Step 1: PhoBERT predict

Sinh labels trước.

### Step 2: Build explanation prompt

```text
Review: ...
Labels: ...
Hãy tìm evidence và giải thích.
```

### Step 3: LLM extract evidence

```json
{
 "ROOM": {
   "evidence":"phòng sạch",
   "explanation":"Review khen phòng sạch"
 },
 "STAFF": {
   "evidence":"nhân viên chậm",
   "explanation":"Review chê tốc độ phục vụ"
 }
}
```

### Step 4: Quality Check

* evidence có nằm trong text không
* explanation có đúng label không

### Step 5: Save reports

## Output

```text
outputs/results/llm_explainability/explanation_samples.json
outputs/results/llm_explainability/explanation_quality_report.json
```

## Ý nghĩa

Rất phù hợp dashboard phân tích review thực tế.

---

# 6. Compare Results

## Mục tiêu

So sánh các phương pháp.

## Input

Metrics từ:

* SVM
* PhoBERT
* LLM + RAG
* Explainability pipeline

## Flow xử lý chi tiết

### Step 1: Read metrics

Load score từng model.

### Step 2: Build comparison table

| Method            | F1             |
| ----------------- | -------------- |
| SVM               | 0.31           |
| PhoBERT           | 0.55           |
| LLM + RAG         | thấp hơn       |
| PhoBERT + Explain | practical best |

### Step 3: Export markdown

## Output

```text
outputs/results/final_four_direction_comparison.md
```

## Ý nghĩa

Dùng cho báo cáo, thesis, slide trình bày.

---

# Toàn bộ Data Flow thực tế

```text
Raw Review
 ↓
Preprocessing
 ↓
Clean Text + Labels
 ├──> SVM
 ├──> PhoBERT
 ├──> LLM + RAG
 └──> PhoBERT + LLM Explain
        ↓
   Final Comparison
```

---

# Tóm tắt cho người mới

* SVM = học bằng thống kê từ khóa
* PhoBERT = hiểu ngữ cảnh sâu bằng Transformer
* LLM + RAG = dùng AI lớn + ví dụ gần giống
* Explainability = giải thích lý do dự đoán

---

# Thứ tự nên chạy project

```bash
pip install -r requirements.txt
python code/data_processing/step1_eda.py
python code/data_processing/step3_preprocessing.py
python code/svm_baseline/svm_baseline.py
python code/phobert/run_experiment.py --encoder cls_only
OPENAI_API_KEY=... python code/llm_rag/run_llm_rag.py
OPENAI_API_KEY=... python code/llm_explainability/run_explainability.py ...
python code/llm_rag/compare_results.py
```
