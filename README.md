# ABSA VLSP 2018 Hotel

Aspect-Based Sentiment Analysis cho review khách sạn tiếng Việt.

Final scope được giữ gọn còn 4 hướng chính:

1. **SVM + TF-IDF** — baseline truyền thống.
2. **PhoBERT** — model dự đoán chính, dùng bản single model tốt nhất, không ensemble.
3. **LLM + RAG** — thử nghiệm advanced method; RAG cải thiện LLM nhưng không vượt PhoBERT.
4. **PhoBERT + LLM explanation** — hướng ứng dụng thực tế: PhoBERT dự đoán, LLM giải thích/evidence/error analysis.

Các thử nghiệm phụ như cascade, verifier, augmentation, retrain v1, nhiều notebook cũ đã được đưa vào `draft/` để báo cáo chính không bị loãng.

## Cài Đặt

```bash
pip install -r requirements.txt
```

## Data

Dataset gốc: `ds4v/absa-vlsp-2018`.

Các script chính dùng cache đã preprocess:

```text
data/train_preprocessed.csv
data/dev_preprocessed.csv
data/test_preprocessed.csv
```

Nếu cần tạo lại:

```bash
python code/data_processing/step1_eda.py
python code/data_processing/step3_preprocessing.py
```

## 1. SVM Baseline

```bash
python code/svm_baseline/svm_baseline.py
```

Kết quả tham chiếu:

```text
Combined F1 ≈ 0.3173
```

## 2. PhoBERT Predictor

Train/evaluate single PhoBERT:

```bash
python code/phobert/run_experiment.py --encoder cls_only
```

Notebook minh chứng đã chạy có output:

```text
notebooks/phase_phobert_vncorenlp_executed.ipynb
```

Notebook ablation không dùng VnCoreNLP:

```text
notebooks/phase_phobert_no_vncorenlp_executed.ipynb
```

So sánh chi tiết:

```text
docs/phobert_vncorenlp_ablation.md
```

Kết quả report chính dùng:

```text
PhoBERT cls_only + VnCoreNLP
ACD F1      ≈ 0.6360
SPC F1      ≈ 0.4727
Combined F1 ≈ 0.5543
```

Ghi chú: không dùng ensemble trong phần chính để giữ báo cáo gọn.

## 3. LLM + RAG

Chạy RAG few-shot predictor:

```bash
OPENAI_API_KEY=... python code/llm_rag/run_llm_rag.py
```

Notebook minh chứng đã chạy có output:

```text
notebooks/phase_llm_rag_executed.ipynb
```

Các file chính:

```text
code/llm_rag/llm_client.py
code/llm_rag/prompts.py
code/llm_rag/rag_retriever.py
code/llm_rag/rag_predictor.py
code/llm_rag/run_llm_rag.py
```

Kết luận report:

```text
RAG cải thiện LLM few-shot, nhưng LLM prediction vẫn thấp hơn PhoBERT nhiều.
```

## 4. PhoBERT + LLM Explanation

Hướng final ứng dụng:

```text
Review
  -> PhoBERT predicts aspect-sentiment labels
  -> LLM receives only present predictions
  -> LLM extracts evidence and writes explanation
  -> Output supports practical hotel review analysis
```

Các file chính:

```text
code/llm_explainability/prediction_formatter.py
code/llm_explainability/explanation_prompts.py
code/llm_explainability/evidence_checker.py
code/llm_explainability/llm_explainer.py
code/llm_explainability/error_analyzer.py
code/llm_explainability/run_explainability.py
```

Ví dụ chạy nếu đã có saved prediction matrix:

```bash
OPENAI_API_KEY=... python code/llm_explainability/run_explainability.py \
  --predictions_json outputs/results/final_predictions.json \
  --provider openai \
  --max_samples 20
```

Output:

```text
outputs/results/llm_explainability/explanation_samples.json
outputs/results/llm_explainability/explanation_quality_report.json
```

## Final Report Table

Tạo bảng gọn cho 4 hướng:

```bash
python code/llm_rag/compare_results.py
```

Output:

```text
outputs/results/final_four_direction_comparison.md
```

## Draft

Các thử nghiệm không dùng trong báo cáo chính được lưu tại:

```text
draft/
```

Không xóa các file này; chỉ không dùng chúng trong flow chính.
