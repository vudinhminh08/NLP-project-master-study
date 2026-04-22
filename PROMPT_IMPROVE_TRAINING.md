# Execution Plan: Hybrid Explainable PhoBERT v3

## Summary

Tạo nhánh `feature/improve-training-v3-explainable` từ `feature/improve-training-v2` và triển khai hướng "Hybrid giải thích được": giữ pipeline v2 đã có early stopping theo Combined F1, top-k checkpoint, ensemble và run tag; thêm các kỹ thuật hiệu quả từ notebook tham khảo nhưng giữ cấu trúc dễ trình bày trong báo cáo.

Mục tiêu:
- Vượt baseline cũ `0.5543`.
- Cố gắng vượt notebook tham khảo tốt nhất `0.6218`.
- Không claim SOTA; báo cáo rõ gap với SOTA `0.7732`.

## Why This Is Explainable

1. **Metric selection đúng bài toán**
   - Chọn checkpoint theo `macro_combined_f1`, không theo `dev_loss`.
   - Combined F1 là metric chính của đề bài.

2. **Tách bài toán theo bản chất ABSA**
   - ACD: aspect có xuất hiện hay không.
   - SPC: nếu aspect xuất hiện thì sentiment là `positive/negative/neutral`.
   - Split loss giúp giảm ảnh hưởng lớp `absent` áp đảo.

3. **Xử lý mất cân bằng dữ liệu**
   - Rare aspect oversampling.
   - Class weights/focal loss.
   - Tune threshold trên dev để tối ưu macro Combined F1.

## Implementation Changes

### Modeling

- Thêm `presence_heads` cho 34 aspects.
- Giữ sentiment classifier 4 logits để tương thích code cũ, nhưng khi `use_split_loss=True` chỉ dùng logits `1..3` cho sentiment.
- Thêm config:
  - `use_split_loss=True`
  - `lambda_presence=1.0`
  - `lambda_sentiment=1.0`
  - `presence_threshold=0.5`
  - `tune_presence_threshold=True`
  - `presence_threshold_mode=per_aspect_combined`
  - `presence_threshold_grid=[0.35, 0.4, 0.45, 0.5, 0.55, 0.6]`
- Mặc định dùng CLS/concat CLS để dễ giải thích.
- `use_attention_pooling=False` mặc định; attention pooling chỉ là ablation optional.

### Training/Data

- Đổi optimizer sang `AdamW`.
- Thêm:
  - `weight_decay=0.01`
  - `head_lr_mult=5.0`
  - `layerwise_lr_decay=0.92`
  - `use_gradient_checkpointing=True`
  - `max_grad_norm=1.0`
- Thêm rare aspect oversampling bằng `WeightedRandomSampler`.
- Thêm rare loss multipliers:
  - `rare_aspect_mult=1.5`
  - `rare_presence_pos_mult=1.2`
  - `rare_sentiment_mult=1.1`
- Giữ:
  - early stopping theo `macro_combined_f1`
  - top-k checkpoint tracking
  - ensemble top-3
  - run tag chống ghi đè

### Prediction/Report

- Thêm `tune_presence_threshold()` trên dev.
- Hỗ trợ global threshold và per-aspect threshold.
- Lưu threshold values vào `presence_thresholds.json`.
- Report thêm:
  - threshold mode/grid
  - split loss on/off
  - primary result giữa single và ensemble
  - baseline cũ `0.5543`
  - notebook tham khảo `0.6101`/`0.6218`
  - SOTA `0.7732`

### Kaggle Notebook

Notebook chạy các run chính:

1. `v3_cls_split_lr1e4`
   - encoder: `cls_only`
   - lr: `1e-4`
   - max_epochs: `20`
   - early_stop_patience: `7`

2. `v3_concat_split_lr1e4`
   - encoder: `concat_4_layers`
   - lr: `1e-4`
   - max_epochs: `20`
   - early_stop_patience: `7`

3. `v3_cls_split_lr2e5`
   - encoder: `cls_only`
   - lr: `2e-5`
   - max_epochs: `40`
   - early_stop_patience: `10`

Optional:
- `v3_cls_attention_lr1e4` với `use_attention_pooling=True`.

Notebook phải:
- clone/pull branch `feature/improve-training-v3-explainable`
- chạy các run trên
- vẽ learning curve
- in comparison table
- chọn primary result tự động
- zip outputs

## Test Plan

- `ast.parse` cho:
  - `code/phobert/model.py`
  - `code/phobert/train.py`
  - `code/phobert/predict.py`
  - `code/phobert/run_experiment.py`
  - `code/phobert/ensemble.py`
  - `code/data_processing/step2_dataloader.py`
- Validate notebook JSON và parse từng code cell.
- Chạy `code/phobert/smoke_test.py`.
- Smoke checks cần pass:
  - forward pass có `presence_logits`
  - split loss chạy được
  - threshold tuning không crash
  - run tag không ghi đè output cũ

## Acceptance Criteria

- Branch mới được commit và push.
- Không ghi đè output cũ.
- Full local smoke test pass.
- Notebook Kaggle chạy được từ đầu đến cuối.
- Nếu kết quả không vượt `0.6218`, report vẫn phải có ablation và giải thích rõ.
