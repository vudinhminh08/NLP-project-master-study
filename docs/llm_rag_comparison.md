# LLM + RAG Comparison

Tài liệu này chỉ giữ phần cần thiết cho hướng LLM + RAG trong báo cáo chính.

## Main Result

| Method | k | ACD F1 | SPC F1 | Combined F1 | Note |
|---|---:|---:|---:|---:|---|
| ICL GPT-4o-mini | 8 | 0.3504 | 0.2567 | 0.3035 | Few-shot random examples |
| RAG GPT-4o-mini | 2 | 0.3793 | 0.2866 | 0.3330 | Retrieved examples |
| RAG GPT-4o-mini | 4 | 0.3866 | 0.2883 | 0.3375 | Retrieved examples |
| **RAG GPT-4o-mini** | **8** | **0.4034** | **0.3031** | **0.3532** | Best LLM predictor |
| RAG GPT-4o-mini | 16 | 0.3988 | 0.2879 | 0.3433 | Context overload / diminishing returns |

## Giải thích tham số và cách lựa chọn

| Tham số | Cách chọn | Lý do |
|---------|----------|-------|
| `model=gpt-4o-mini` | Chọn model LLM chi phí thấp, tốc độ ổn | Thí nghiệm cần gọi nhiều request trên 600 test reviews; model quá đắt không phù hợp bài tập lớn. |
| `k=2,4,8,16` | Chạy ablation theo số ví dụ retrieved | Dùng nhiều mức k để kiểm tra trade-off: ít ví dụ thì thiếu ngữ cảnh, quá nhiều ví dụ thì prompt nhiễu. |
| `k=8` | Chọn vì Combined F1 cao nhất trong các run RAG | Đây là điểm cân bằng giữa đủ ví dụ domain và prompt chưa quá dài. |
| Sentence embedding retriever | Dùng embedding similarity thay vì random examples | Review tương tự thường có aspect/sentiment cue tương tự, giúp LLM hiểu schema nhanh hơn ICL ngẫu nhiên. |
| JSON output schema | Ép LLM trả về cấu trúc parse được | ABSA cần evaluate tự động; output tự do sẽ khó map về 34 aspect labels. |
| Cache response | Lưu output LLM theo prompt | Tránh tốn API khi rerun, đồng thời giúp notebook executed giữ kết quả ổn định. |
| `sleep_sec=1.0` | Thêm khoảng nghỉ giữa request | Giảm rủi ro rate limit khi chạy nhiều sample trên Kaggle. |
| `max_samples=None` | Chạy full test set khi lấy kết quả chính | Dùng 600 test reviews để metric so sánh công bằng với SVM và PhoBERT. |

Các tham số của LLM+RAG không nhằm cạnh tranh bằng mọi giá với PhoBERT. Mục
tiêu là kiểm tra một hướng predictor không train supervised model, từ đó chứng
minh retrieval có ích nhưng vẫn chưa đủ thay thế PhoBERT.

## Quy trình xử lý

LLM + RAG được thử như một hướng predictor không train thêm model supervised.
Mục tiêu là kiểm tra xem LLM có thể dựa vào ví dụ tương tự trong train set để
dự đoán aspect/sentiment tốt hơn few-shot ngẫu nhiên hay không.

Pipeline:

1. Dùng train set đã preprocess và có label để xây kho ví dụ.
2. Encode review bằng sentence-transformer.
3. Với mỗi review test, retrieve `k` review train gần nhất.
4. Format các ví dụ retrieved thành prompt gồm review và nhãn aspect present.
5. Gọi GPT-4o-mini để dự đoán 34 aspect labels.
6. Parse JSON output của LLM về vector `[34]`.
7. Evaluate bằng cùng metric ACD F1, SPC F1, Combined F1.

Các giá trị `k = 2, 4, 8, 16` được thử để xem retrieval context ảnh hưởng thế
nào đến LLM. `k=8` là điểm tốt nhất trong các run được giữ lại.

## Code và notebook liên quan

Các file code chính:

- `code/llm_rag/rag_retriever.py`: xây embedding index và retrieve examples.
- `code/llm_rag/prompts.py`: format prompt, schema output và parser.
- `code/llm_rag/rag_predictor.py`: chạy prediction bằng LLM + retrieved
  examples.
- `code/llm_rag/run_llm_rag.py`: entrypoint chạy experiment.
- `code/llm_rag/llm_client.py`: wrapper OpenAI/Gemini, retry và cache.

Notebook báo cáo:

- `notebooks/phase_llm_rag_executed.ipynb`: notebook executed có output thật
  cho kết quả RAG k=8.

## Interpretation

RAG improves LLM prediction compared with random in-context examples:

```text
ICL k=8 Combined F1 = 0.3035
RAG k=8 Combined F1 = 0.3532
Delta = +0.0497
```

However, even the best LLM + RAG run remains far below supervised PhoBERT:

```text
PhoBERT cls_only + VnCoreNLP Combined F1 = 0.5543
LLM + RAG k=8 Combined F1 = 0.3532
Gap = -0.2011
```

## Phân tích kết quả chi tiết

RAG giúp LLM tăng từ Combined F1 `0.3035` lên `0.3532`, tức tăng `+0.0497`.
Điều này chứng minh retrieval có ích: khi LLM được xem các review tương tự đã
gán nhãn, nó hiểu domain khách sạn và schema aspect tốt hơn so với few-shot
ngẫu nhiên.

Tuy nhiên, kết quả vẫn thấp hơn PhoBERT `0.2011` Combined F1. Nguyên nhân chính:

- LLM phải vừa detect aspect, vừa classify sentiment, vừa giữ đúng schema 34
  aspects trong một prompt ngắn.
- Các aspect hiếm gần như không đủ ví dụ retrieved ổn định.
- Một review có thể chứa nhiều aspect và sentiment trái chiều, dễ làm LLM bỏ
  sót hoặc hallucinate aspect.
- Khi `k=16`, context dài hơn nhưng điểm giảm, cho thấy thêm nhiều ví dụ không
  luôn tốt; prompt có thể bị nhiễu và LLM khó tập trung.

Vì vậy, RAG được giữ trong báo cáo như một negative/contrastive experiment:
nó tốt hơn ICL ngẫu nhiên, nhưng không đủ mạnh để thay supervised PhoBERT.

## Report Claim

LLM + RAG is useful as an experimental direction because retrieval gives a clear
improvement over plain few-shot prompting. It should not be positioned as the
main classifier because supervised PhoBERT is much stronger for this dataset.

The final system therefore uses:

```text
PhoBERT = prediction
LLM + RAG = explanation and evidence extraction
```
