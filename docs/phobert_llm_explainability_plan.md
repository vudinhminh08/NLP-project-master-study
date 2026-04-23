# PhoBERT + LLM Explainability Plan

Đây là plan hiện hành cho hướng ứng dụng trong báo cáo v10. LLM không được dùng
làm classifier và không thay đổi nhãn PhoBERT. Vai trò của LLM là giải thích
prediction, trích evidence và đề xuất action cho chủ khách sạn.

## Pipeline

```text
Review
-> VnCoreNLP preprocessing
-> PhoBERT v3 cls_only prediction
-> Present aspect/sentiment items
-> LLM explanation
-> Evidence + reason + recommended action
```

## Thành phần

| Thành phần | Giá trị hiện hành | Ghi chú |
|---|---|---|
| PhoBERT checkpoint | `outputs/results/phobert_results_version3/models_cls_only/best_model.pt` | Best single model hiện tại |
| Encoder | `cls_only` | Test Combined F1 `0.6218` |
| LLM provider | OpenAI GPT-4o-mini | Dùng cho explanation |
| App | `app/streamlit_app.py` | Nhập review hoặc CSV/XLSX |
| VnCoreNLP | `vncorenlp/` | Bắt buộc, không fallback rule-based |

## Thiết kế trách nhiệm

- PhoBERT chịu trách nhiệm nhãn aspect/sentiment.
- LLM chỉ nhận danh sách prediction đã có và review gốc.
- LLM output phải giữ nguyên aspect và sentiment PhoBERT.
- Evidence nên là substring hoặc gần-substring trong review gốc.
- Nếu thiếu checkpoint hoặc VnCoreNLP, app báo lỗi thay vì fake label.

## Chất lượng explanation

Các metric đã dùng trong notebook validation:

| Metric | Kết quả | Ý nghĩa |
|---|---:|---|
| Label Consistency | 100.0% | LLM không đổi aspect/sentiment PhoBERT |
| BERTScore Evidence F1 | 0.5535 | Evidence có groundedness ở mức chấp nhận được |
| Human Faithfulness | 1.867 / 2.0 | Ít hallucination trong diễn giải |
| Human Usefulness | 1.667 / 2.0 | Action phần lớn có ích, còn phụ thuộc domain context |

## Lý do không dùng LLM làm predictor

ABSA VLSP 2018 Hotel có schema 34 aspect và class imbalance nặng. LLM có thể
giải thích tốt nhưng không nên thay supervised PhoBERT trong phần sinh nhãn vì:

- dễ drift schema khi output nhiều aspect;
- khó đảm bảo nhất quán trên aspect hiếm;
- chi phí inference cao hơn;
- khó đạt F1 ổn định bằng model supervised đã fine-tune.

Vì vậy report v10 chỉ giữ LLM trong vai trò explainability layer.

## Cách chạy app

```bash
pip install -r app/requirements-app.txt
streamlit run app/streamlit_app.py
```

Config mặc định nên trỏ tới:

```text
outputs/results/phobert_results_version3/models_cls_only/best_model.pt
```

## Cách chạy batch explanation

`code/llm_explainability/run_explainability.py` không load checkpoint trực tiếp;
nó nhận prediction matrix đã sinh từ PhoBERT. Khi chạy batch, cần đảm bảo
`--predictions_json` được tạo từ checkpoint v3 `cls_only`.
