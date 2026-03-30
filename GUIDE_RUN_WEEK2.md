# Hướng dẫn chạy Week 2 — PhoBERT ds4v Sync

## Tóm tắt thay đổi so với v1

| | Version 1 (cũ) | Version 2 (mới) |
|---|---|---|
| Word segmentation | underthesea | **VnCoreNLP** |
| Model | phobert-base | **phobert-base-v2** |
| Learning rate | 2e-5 | **1e-4** |
| Optimizer | AdamW | **Adam** |
| Warmup | 10% | **15%** |
| Batch size | 8 (eff. 32) | **16** |
| Early stop monitor | combined_f1 | **dev_loss** |
| Kết quả kỳ vọng | ~0.30 | **~0.60-0.75** |

---

## Bước 1 — Cài VnCoreNLP (local, chạy 1 lần)

```bash
pip install py_vncorenlp
```

Chạy từ thư mục root project để download model (~250MB):

```bash
python -c "
import py_vncorenlp
m = py_vncorenlp.VnCoreNLP(annotators=['wseg'], save_dir='./vncorenlp')
result = m.word_segment('Khách sạn rất tốt và nhân viên thân thiện')
print(result)
m.close()
"
```

Expected output:
```
['Khách_sạn', 'rất', 'tốt', 'và', 'nhân_viên', 'thân_thiện']
```

Nếu thấy từ ghép được nối bằng `_` → VnCoreNLP đã hoạt động đúng.

---

## Bước 2 — Re-preprocess data với VnCoreNLP

Xóa cache cũ (underthesea) và chạy lại:

```bash
rm data/train_preprocessed.csv data/dev_preprocessed.csv data/test_preprocessed.csv
python code/week1/step3_preprocessing.py
```

Expected output (phải thấy VnCoreNLP, không phải fallback):
```
[Segmenter] VnCoreNLP loaded successfully.
[Preprocessing] train...
  train: 3000 samples preprocessed
```

Nếu thấy `[WARN] Fallback: underthesea` → VnCoreNLP chưa cài đúng, quay lại Bước 1.

Verify kết quả (phải có dấu `_`):

```bash
python -c "
import pandas as pd
s = pd.read_csv('data/train_preprocessed.csv').iloc[0]['processed_review']
print(s[:200])
assert '_' in s, 'FAIL: Không có underscore!'
print('OK: VnCoreNLP segmentation confirmed')
"
```

---

## Bước 3 — Commit và push lên GitHub

```bash
git add data/train_preprocessed.csv data/dev_preprocessed.csv data/test_preprocessed.csv
git add code/week1/utils/constants.py code/week2/train.py notebooks/finetune-notebook-week2.ipynb
git commit -m "feat: week2 ds4v sync — VnCoreNLP + LR=1e-4 + Adam + phobert-base-v2"
git push
```

---

## Bước 4 — Chạy trên Kaggle

1. Mở notebook `finetune-notebook-week2.ipynb` trên Kaggle
2. **Cell 2:** Clone/pull repo mới nhất
3. **Cell 3:** Download data (nếu chưa có)
4. **Cell 4:** Sẽ thấy `✅ Cache đã có` — dùng CSV VnCoreNLP vừa push
5. **Cell 5:** Verify config — phải thấy `✅ Config OK — ds4v sync verified`
6. **Cell 6:** Train `concat_4_layers` — kỳ vọng Combined F1 > 0.50
7. **Cell 7:** Ablation `cls_only`
8. **Cell 8:** Learning curve
9. **Cell 9:** Download kết quả

---

## Nếu gặp lỗi OOM trên Kaggle

Giảm batch_size trong Cell 5 (thêm 2 dòng sau phần import):

```python
TRAIN_CONFIG['batch_size'] = 8
TRAIN_CONFIG['grad_accumulation_steps'] = 2
```

Effective batch vẫn là 16, nhưng dùng ít VRAM hơn.

---

## So sánh kết quả

| Model | Combined F1 (Test) |
|---|---|
| SVM + TF-IDF | 0.317 |
| PhoBERT v1 (underthesea) | 0.298 |
| **PhoBERT v2 (VnCoreNLP, ds4v sync)** | **~0.60-0.75** |
| SOTA (ds4v) | 0.773 |
