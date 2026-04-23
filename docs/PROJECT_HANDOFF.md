# Project Handoff

Tài liệu này là bản handoff rút gọn sau khi cập nhật báo cáo
`Group16_Report_v10.docx`.

## Trạng thái hiện tại

- Branch đang làm việc: `feature/change-phobert`.
- Báo cáo chính: `Group16_Report_v10.docx`.
- Model chính: PhoBERT v3 `cls_only`.
- App chính: PhoBERT + LLM Explanation.
- Các thí nghiệm predictor phụ không còn nằm trong báo cáo v10; phần cần trình
  bày chỉ gồm SVM baseline, PhoBERT v3 và PhoBERT + LLM Explanation.

## Kết quả chính

| Method | Vai trò | Test ACD F1 | Test SPC F1 | Test Combined F1 |
|---|---|---:|---:|---:|
| SVM + TF-IDF | Baseline | 0.4074 | 0.2272 | 0.3173 |
| PhoBERT v3 `concat_4_layers` | Ablation | 0.6827 | 0.5374 | 0.6101 |
| PhoBERT v3 `cls_only` | Best single | 0.6849 | 0.5587 | 0.6218 |
| SOTA DS4V/Dang et al. 2022 | Tham chiếu | 0.8255 | - | 0.7732 |

Best single hiện tại là `cls_only` vì Test Combined F1 `0.6218` cao hơn
`concat_4_layers`.

## Checkpoint và output quan trọng

| Artifact | Path |
|---|---|
| PhoBERT v3 best checkpoint | `outputs/results/phobert_results_version3/models_cls_only/best_model.pt` |
| PhoBERT v3 cls metrics | `outputs/results/phobert_results_version3/results_cls_only/` |
| PhoBERT v3 concat metrics | `outputs/results/phobert_results_version3/results/` |
| No-VnCoreNLP ablation | `outputs/results/phobert_results_version3/results/phobert_no_vncorenlp/` |
| Attention maps | `outputs/results/phobert_results_version3/attention_maps_cls_only/` |
| Notebook kết quả | `notebooks/phobert_vnccorenlp_version3_clean_done(best_results).ipynb` |

## Kỹ thuật đã áp dụng cho PhoBERT v3

- VnCoreNLP word segmentation.
- PhoBERT base v2 encoder.
- Aspect-Aware Attention Pooling.
- Dual Head: Presence Head cho ACD và Sentiment Head cho SPC.
- Split loss: `presence_loss + sentiment_loss`.
- AdamW, weight decay, head LR multiplier, layer-wise LR decay.
- Focal loss, class weighting, rare aspect oversampling.
- Threshold tuning trên dev.
- Gradient checkpointing để chạy ổn định trên Kaggle T4.

## LLM Explanation

LLM không sinh nhãn mới. Pipeline ứng dụng:

```text
Review -> PhoBERT prediction -> LLM explanation -> evidence/reason/action
```

Metric validation:

| Metric | Kết quả |
|---|---:|
| Label Consistency | 100.0% |
| BERTScore Evidence F1 | 0.5535 |
| Human Faithfulness | 1.867 / 2.0 |
| Human Usefulness | 1.667 / 2.0 |

## Tài liệu trong `docs/`

| File | Vai trò |
|---|---|
| `phobert_best_single_summary.md` | Summary model v3 và luồng xử lý |
| `phobert_vncorenlp_ablation.md` | Ablation VnCoreNLP |
| `svm_baseline_summary.md` | Baseline SVM |
| `eda_summary_report.md` | EDA rút gọn |
| `phobert_llm_explainability_plan.md` | Plan app PhoBERT + LLM Explanation |
| `llm_explanation_validation_summary.md` | Validation chất lượng explanation |

## Lưu ý còn lại

- `outputs/results/phobert_results_version3/models_cls_only/best_model.pt` là
  checkpoint nên dùng cho app và demo.
- `models/best_model.pt` là bản `concat_4_layers`, nặng hơn và score thấp hơn.
- Nếu app vẫn trỏ checkpoint cũ, cần đổi sang path version3.
- Không thêm lại các thí nghiệm predictor phụ vào báo cáo v10 trừ khi có yêu cầu
  mới.
