# CHANGELOG — ABSA VLSP 2018 Hotel

> Dự án NLP môn học HUST — Phân tích cảm xúc theo khía cạnh (ABSA) trên dataset đánh giá khách sạn VLSP 2018.

---

## Bối cảnh dự án

**Bài toán:** Multi-label classification — mỗi review tiếng Việt được gán nhãn cho 34 aspect (entity#attribute + sentiment).
- 34 classes = 17 aspects × {positive, negative, neutral} + absent
- Dataset: 3000 train / 2000 dev / 600 test
- SOTA (Huynh 2022): ACD F1 = 0.8255, Combined F1 = 0.7732

**Cấu trúc thư mục:**
```
absa-vlsp2018-hotel/
├── data/
│   ├── train.csv / dev.csv / test.csv          # raw data gốc
│   ├── train_preprocessed.csv                  # sau VnCoreNLP segment
│   ├── dev_preprocessed.csv
│   ├── test_preprocessed.csv
│   ├── train_augmented.csv                     # train + 231 samples LLM (Week 4)
│   ├── augmented_reviews.json                  # raw generated (240 reviews)
│   └── augmented_reviews_filtered.json         # sau filter (231 reviews)
├── code/
│   ├── baseline/                               # SVM + TF-IDF
│   ├── week1/                                  # EDA, preprocessing, dataloader, eval
│   ├── week2/                                  # PhoBERT training pipeline
│   ├── week3/                                  # LLM ICL/RAG (GPT-4o-mini)
│   ├── week3_part2/                            # Data augmentation + Explainability
│   └── week4/                                  # (deprecated, dùng week3_part2 thay)
├── notebooks/
│   ├── phobert-vncorenlp.ipynb                 # Kaggle: train PhoBERT Week 2 (CHUẨN)
│   ├── week3_part2_data_augmentation.ipynb     # Local: sinh augmented data
│   └── week4_demo.ipynb                        # Kaggle: train augmented + LLM explain
├── outputs/
│   ├── eda/                                    # class_weights.json, encoder_config.json
│   └── results/
│       ├── week2_results_VNcoreNLP/            # kết quả PhoBERT Week 2
│       ├── ensemble/                           # ensemble + threshold tuning
│       └── week4_augmented/                    # kết quả PhoBERT Week 4 (sau khi train)
└── vncorenlp/                                  # VnCoreNLP jar + models (local only)
```

---

## Kết quả tất cả các tuần

| Phương pháp | ACD F1 | SPC F1 | Combined F1 | Ghi chú |
|---|---|---|---|---|
| SOTA (Huynh 2022) | 0.8255 | — | 0.7732 | Mục tiêu |
| **PhoBERT cls_only + VNcoreNLP** | **0.6360** | **0.4727** | **0.5543** | Best single model |
| PhoBERT cls_only + threshold tuning | — | — | 0.5527 | Nhỉnh hơn argmax |
| Ensemble + threshold tuning | — | — | **0.5644** | Best overall |
| PhoBERT concat_4_layers + VNcoreNLP | 0.5836 | 0.4658 | 0.5247 | concat_4 kém cls_only |
| PhoBERT concat_4 (không VNcoreNLP) | 0.3592 | 0.2371 | 0.2981 | VNcoreNLP quan trọng |
| RAG GPT-4o-mini k=8 | 0.4034 | 0.3031 | 0.3532 | Best LLM |
| ICL GPT-4o-mini k=8 | 0.3504 | 0.2567 | 0.3035 | RAG > ICL +5% |
| SVM + TF-IDF | 0.4074 | 0.2272 | 0.3173 | Baseline |
| PhoBERT + augmented data (Week 4) | 0.5953 | 0.4370 | 0.5162 | Kém baseline -3.8% |

---

## Week 1 — EDA, Preprocessing, DataLoader

### Những gì đã làm
- EDA phân tích phân phối label, class imbalance (85% absent)
- Phát hiện 8 rare aspects (0–90 mẫu), đặc biệt ROOM_AMENITIES#PRICES = 0 mẫu
- Preprocessing pipeline: unicode → whitespace → teencode → special chars → VnCoreNLP segment
- VnCoreNLP segment từ ghép thành `khách_sạn`, `nhân_viên` — bắt buộc để PhoBERT hoạt động đúng
- Tạo `train/dev/test_preprocessed.csv` với cột `processed_review`

### File quan trọng
- `code/week1/step3_preprocessing.py` — `VnCoreNLPSegmenter`, `preprocess_text()`
- `code/week1/step2_dataloader.py` — `create_dataloaders(train_path, dev_path, test_path, tokenizer, batch_size, max_len, num_workers, use_preprocessed)`
- `code/week1/step4_eval.py` — `evaluate_predictions()` tính ACD F1, SPC F1, Combined F1
- `outputs/eda/class_weights.json` — per-aspect class weights (neutral weight=154 → clip=10)
- `outputs/eda/encoder_config.json` — encoder_option, recommended_max_seq_len=256

---

## Week 2 — PhoBERT Multi-task Fine-tuning

### Những gì đã làm
- Model: PhoBERT (vinai/phobert-base-v2) + 34 classification heads song song
- 2 variants: `cls_only` (768 dim) và `concat_4_layers` (concat 4 last hidden layers, 3072 dim)
- Focal Loss (γ=2) + class weights + LR warmup-peak-decay
- Early stopping theo dev_loss (patience=7)
- **Phát hiện:** cls_only (0.5543) tốt hơn concat_4_layers (0.5247) — ngược với kỳ vọng
- Ensemble + threshold tuning trên dev set → Combined F1 = 0.5644

### File quan trọng
- `code/week2/model.py` — `ABSAPhoBERT(model_name, num_aspects=34, num_labels=4, dropout=0.2, encoder_option, focal_gamma=2.0)`
- `code/week2/train.py` — `train(model, train_loader, dev_loader, class_weights, device, config, save_dir, results_dir, use_amp=False) → history`; `load_class_weights(weights_path, weight_clip=10.0, device=None)`
- `code/week2/predict.py` — `load_best_model(checkpoint_path, model, device) → model`; `predict_and_evaluate(model, dataloader, class_weights, device, split_name, save_path) → (metrics, y_true, y_pred)`
- `code/week2/run_experiment.py` — pipeline đầy đủ, đây là chuẩn để tham chiếu
- `notebooks/phobert-vncorenlp.ipynb` — Kaggle notebook đã chạy thành công

### Checkpoint format
```python
# Checkpoint lưu dạng dict:
ckpt = {
    'epoch': int,
    'model_state_dict': OrderedDict,   # <-- dùng cái này khi load
    'dev_loss': float,
    'combined_f1': float,
    'acd_f1': float,
    'spc_f1': float,
    'config': dict,
}
# Load đúng cách:
ckpt = torch.load('best_model.pt', map_location=device)
model.load_state_dict(ckpt['model_state_dict'])   # KHÔNG được load cả ckpt
```

---

## Week 3 — LLM Few-shot ICL & RAG

### Những gì đã làm
- GPT-4o-mini dự đoán 34-class ABSA bằng few-shot (ICL và RAG)
- RAG: embedding reviews → FAISS → retrieve k similar examples
- Thử k=2, 4, 8, 16 → k=8 optimal (k=16 giảm do context overload)
- **Kết luận: LLM kém PhoBERT ~20%** — few-shot không đủ cho 34-class multi-label

### Kết quả (test set)
- Best LLM: RAG k=8 → Combined F1 = 0.3532 (vs PhoBERT 0.5543)
- RAG > ICL cùng k=8: +4.97% (semantic retrieval có đóng góp)
- Ensemble ICL+RAG kém RAG đơn thuần: khi 2 models không ngang sức thì ensemble kéo xuống

### File quan trọng
- `code/week3/llm_client.py` — `LLMClient(provider, api_key)`; `complete(messages, temperature=0.0, use_cache=True)`
- `outputs/results/week3_comparison.md` — bảng so sánh đầy đủ

---

## Week 4 — LLM Data Augmentation + Explainability

### Định hướng
Thay vì dùng LLM để predict (kém), dùng LLM ở 2 vai trò phù hợp hơn:
1. **Data Augmentation:** LLM sinh training samples cho 8 rare aspects
2. **Explainability:** PhoBERT predict → LLM giải thích kết quả bằng tiếng Việt

### Phase 1 — Data Augmentation (ĐÃ HOÀN THÀNH)

#### Quy trình
```
train_preprocessed.csv
  → [LLM GPT-4o-mini] sinh 30 reviews/aspect × 8 aspects = 240 reviews
  → [Filter] heuristic + LLM verify + dedup (Jaccard char 3-gram)
  → 231 reviews sau filter
  → [VnCoreNLP] segment processed_review
  → train_augmented.csv (3000 + 231 = 3231 samples)
```

#### 8 rare aspects được augment
| Aspect | Trước | Sau |
|---|---|---|
| FACILITIES#MISCELLANEOUS | 33 | 63 |
| ROOM_AMENITIES#PRICES | 0 | 27 |
| ROOM_AMENITIES#MISCELLANEOUS | 3 | 31 |
| ROOM_AMENITIES#CLEANLINESS | 89 | 115 |
| ROOM_AMENITIES#DESIGN&FEATURES | 354 | 384 |
| HOTEL#DESIGN&FEATURES | 877 | 907 |
| ROOMS#MISCELLANEOUS | 5 | 35 |
| FOOD&DRINKS#MISCELLANEOUS | 13 | 43 |

#### Bug đã fix trong augmentor.py
- **Bug gốc:** `get_real_examples(random_state=42)` ngoài vòng lặp + `use_cache=True` → cùng 3 examples mọi batch → cùng 5 reviews → 240 generated nhưng chỉ 5 unique → 40 sau filter
- **Fix:** Move vào trong loop với `random_state=batch_idx*17+3`; `use_cache=False, temperature=0.8`; thêm "already generated" section vào prompt

#### File quan trọng
- `code/week3_part2/augmentor.py` — `run_augmentation_all_aspects(train_df, llm_client, n_per_aspect, output_path)`
- `code/week3_part2/augment_filter.py` — `filter_augmented_reviews(raw_path, filtered_path, train_df, llm_client, verify_rate)`
- `notebooks/week3_part2_data_augmentation.ipynb` — chạy local, Cell 15 tạo train_augmented.csv với VnCoreNLP

### Phase 2 — Explainability (ĐÃ HOÀN THÀNH, chưa chạy trên Kaggle)

#### Quy trình
```
review text
  → PhoBERT (checkpoint best_model.pt)
  → 34 predictions → filter aspects present (label != 0)
  → LLM GPT-4o-mini explain
  → JSON: {aspect, sentiment, evidence, explanation, action}
```

#### File quan trọng
- `code/week3_part2/explainer.py` — `explain_predictions(review, predictions, llm_client) → list[dict]`
- `code/week3_part2/run_demo.py` — `format_output(result) → str`

### Phase 1c — Re-train PhoBERT (ĐANG THỰC HIỆN trên Kaggle)

#### Cách chạy trên Kaggle
1. Mở `notebooks/week4_demo.ipynb` trên Kaggle
2. Đảm bảo GPU T4 được bật (Settings → Accelerator)
3. Chạy theo thứ tự Cell 1 → Cell 10

#### Luồng các cell
| Cell | Việc làm | Ghi chú |
|---|---|---|
| 1 | Check GPU, pip install | |
| 2 | git clone/pull repo, setup paths | pull latest để có code mới nhất |
| 3 | Load API key từ Kaggle Secrets | Cần OPENAI_API_KEY trong Secrets |
| 4 | Verify data files | Kiểm tra train_augmented.csv tồn tại |
| 5 | Verify EDA config | class_weights.json, encoder_config.json |
| 5b | Pre-flight: in signatures + verify files | Chạy cell này để kiểm tra import trước khi train |
| 6 | **Train PhoBERT** (~60 phút) | Lưu best_model.pt, dev/test metrics |
| 7 | So sánh baseline vs augmented | Đọc từ JSON files |
| 8 | Load checkpoint, inference test set | 600 reviews |
| 9 | LLM explain 5 reviews | Cần API key |
| 10 | Save results, tạo zip download | |

#### Lưu ý quan trọng
- **VnCoreNLP preprocessing:** `train_augmented.csv` đã được segment bằng VnCoreNLP local trước khi commit — Kaggle chỉ cần dùng `processed_review` column, không cần chạy VnCoreNLP lại
- **Checkpoint format:** Khi load thủ công phải dùng `ckpt['model_state_dict']`, không load cả dict
- **class_weights trên CPU:** `load_class_weights(..., device=torch.device('cpu'))` — `run_epoch` tự move sang GPU

---

## Vấn đề đã gặp & cách fix

| Vấn đề | Nguyên nhân | Fix |
|---|---|---|
| 240 generated → 40 sau filter | `get_real_examples` ngoài loop + `use_cache=True` | Move vào loop, `use_cache=False` |
| `create_dataloaders()` unexpected keyword | Dùng `max_seq_len` thay vì `max_len` | Đổi sang `max_len=256` |
| `KeyError: 'macro_acd_f1'` | `train()` trả về `history` không phải `test_metrics` | Dùng `predict_and_evaluate()` riêng |
| `load_best_model()` unexpected keyword | Dùng `model_class`, `checkpoint_dir` sai | Dùng đúng: `(checkpoint_path, model, device)` |
| `RuntimeError: Unexpected key 'model_state_dict'` | `model.load_state_dict(ckpt)` thay vì `ckpt['model_state_dict']` | Extract `ckpt['model_state_dict']` trước |
| `No module named 'py_vncorenlp'` | Notebook dùng kernel `.venv` nhưng package cài trong system Python | `pip install py_vncorenlp` vào đúng `.venv` |
| API key lộ trong notebook | Hardcode fallback `sk-proj-XL2...` trong Cell 3 | Xóa hardcode, dùng `os.environ` + `raise ValueError` |

---

## Định hướng tiếp theo — Phase A+C (2026-04-09)

**Kế hoạch chi tiết:** `docs/superpowers/plans/2026-04-09-phobert-llm-hybrid.md`

### Phase A — Error-Driven Augmentation (re-train)
- Target **9 WEAK_ASPECTS** (F1=0 hoặc <0.35 trên test set) thay vì rare aspects by frequency
- Sinh 60 samples/aspect (v1: 30) + label consistency check (LLM verify 100%)
- Tạo `train_augmented_v2.csv` → retrain PhoBERT

### Phase C — Inference Cascade
- PhoBERT predict → lấy confidence (max softmax per head)
- Nếu confidence < 0.60 cho bất kỳ WEAK_ASPECT nào → gọi LLM RAG k=4 → override
- `cascade_predictor.py` trong `code/week3_part2/`

---

## Trạng thái hiện tại (2026-04-09)

### Đã hoàn thành
- [x] `data/train_augmented.csv` — 3231 samples, processed_review đã VnCoreNLP segment
- [x] `data/augmented_reviews_filtered.json` — 231 reviews đã filter
- [x] `code/week3_part2/augmentor.py` — sinh augmented data, đã fix bug duplicate
- [x] `code/week3_part2/augment_filter.py` — filter heuristic + LLM verify + dedup
- [x] `code/week3_part2/explainer.py` — LLM explain PhoBERT predictions
- [x] `code/week3_part2/run_demo.py` — demo end-to-end, `format_output()`
- [x] `notebooks/week3_part2_data_augmentation.ipynb` — pipeline local hoàn chỉnh
- [x] `notebooks/week4_demo.ipynb` — Kaggle notebook sẵn sàng chạy
- [x] Tất cả code đã được push lên GitHub

### Đang làm / Chưa xong
- [ ] Chạy `week4_demo.ipynb` trên Kaggle → lấy kết quả PhoBERT + augmented data
- [ ] Điền kết quả vào bảng so sánh ở trên
- [ ] Đánh giá xem augmentation có cải thiện rare aspects không

### Bước tiếp theo
1. Trên Kaggle: git pull → Save Version and Run All `week4_demo.ipynb`
2. Sau khi có kết quả: điền vào bảng so sánh, viết báo cáo

---

## Cách chạy lại từ đầu (nếu cần)

### Local — sinh augmented data
```bash
# 1. Cài dependencies
pip install openai py_vncorenlp underthesea

# 2. Set API key
export OPENAI_API_KEY="sk-..."

# 3. Mở notebook
jupyter notebook notebooks/week3_part2_data_augmentation.ipynb
# Chạy Cell 1 → Cell 16
```

### Kaggle — train PhoBERT
```
1. Vào kaggle.com → Code → your notebook week4_demo.ipynb
2. Settings → Accelerator → GPU T4 x2
3. Add-ons → Secrets → Add OPENAI_API_KEY
4. Edit → (Cell 2 chạy git pull nếu cần update code)
5. Run All / Save Version and Run All
```

### Local — chạy demo inference
```bash
python code/week3_part2/run_demo.py \
    --provider openai \
    --checkpoint_path outputs/results/week4_augmented/models/best_model.pt \
    --interactive
```
