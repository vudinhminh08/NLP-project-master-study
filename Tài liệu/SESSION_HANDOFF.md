# SESSION HANDOFF — ABSA VLSP 2018 Hotel
## Paste file này vào ĐẦU chat mới để tiếp tục làm việc

> Tóm tắt toàn bộ context, quyết định, kết quả và trạng thái hiện tại.
> Claude mới đọc file này = hiểu ngay, không cần giải thích lại.

---

## 1. DỰ ÁN LÀ GÌ

**Bài toán:** Aspect-Based Sentiment Analysis (ABSA) cho review khách sạn tiếng Việt.
- Input: 1 review tiếng Việt
- Output: 34 nhãn ∈ {0=absent, 1=positive, 2=negative, 3=neutral}
- 2 subtask đồng thời: ACD (phát hiện aspect) + SPC (phân loại cảm xúc)
- Ví dụ: "Phòng sạch nhưng nhân viên kém" → {ROOMS#CLEANLINESS: positive, SERVICE#GENERAL: negative}

**Môn:** Xử lý Ngôn ngữ Tự nhiên — HUST | **Deadline:** 21/04/2026 | **Nhóm:** 3 người

**Tư tưởng làm việc:**
- Claude Projects = kiến trúc sư, nghiên cứu lý thuyết, viết Coding Directives
- Claude Code (CLI) = implement code trực tiếp vào máy theo directive
- Mỗi directive là 1 file `.md` với spec đủ chi tiết để Claude Code không cần hỏi thêm

---

## 2. YÊU CẦU CỦA THẦY

### Mức độ đánh giá
| Mức | Yêu cầu | Project này |
|-----|---------|-------------|
| Cơ bản | Học máy truyền thống (SVM, TF-IDF) | Không làm |
| Khá | Transformer/BERT cơ bản | PhoBERT alone = tối đa Khá |
| **Giỏi** | **Transformer + ICL/RAG tiên tiến** | **PhoBERT + RAG ← mục tiêu** |
| Xuất sắc | Đề xuất cải tiến mới | Nếu có thêm thời gian |

> ⚠️ Thầy cảnh báo: "Chỉ dùng BERT mà đạt kết quả cao = tối đa mức Khá"
> → **Bắt buộc phải có ICL/RAG mới lên Giỏi**

### Yêu cầu bắt buộc trong báo cáo cuối
1. **Tham số mô hình:** LR, batch size, MAX_SEQ_LEN, dropout, optimizer...
2. **Thiết lập thí nghiệm:** GPU, seed, train/dev/test split, metric
3. **Đường học của mô hình (learning curve):** train loss, dev loss, F1 theo epoch
4. **Giải thích tại sao dừng tại epoch đó:** Early stopping vì dev F1 không cải thiện = overfit

### Cấu trúc thầy yêu cầu
```
Tuần 2: Chạy BASELINE (PhoBERT)
    ↓
Tuần 3: Chạy PROPOSED METHOD (LLM + RAG)
    ↓
So sánh: Proposed có tốt hơn Baseline không? Tại sao?
```

### Deadlines
- Proposal: 22/03/2026 ✅ Đã nộp
- Slide + Báo cáo + Mã nguồn: **21/04/2026** (nộp muộn 1 ngày trừ 1 điểm)

---

## 3. THIẾT KẾ — BASELINE + PROPOSED METHOD

### Baseline — Tuần 2 (PhoBERT)
- Fine-tune `vinai/phobert-base` trên 3000 reviews
- Multi-task: 34 classification heads song song
- Kiến trúc concat 4 layers (theo SOTA ds4v IEEE 2022)
- **Mục đích:** Điểm tham chiếu để so sánh với proposed method

### Proposed Method — Tuần 3 (LLM + RAG)
- Không train, chỉ gọi GPT-4o-mini / Gemini 1.5 Flash với k examples trong prompt
- Gồm 2 chiến lược chọn examples (ablation study):
  - **ICL** (In-Context Learning): chọn examples **ngẫu nhiên** — điểm tham chiếu trong tuần 3
  - **RAG** (Retrieval-Augmented Generation): chọn examples **thông minh** bằng semantic similarity — đề xuất chính
- ICL và RAG KHÔNG phải 2 hướng riêng — chúng là 2 cách chọn examples trong cùng 1 paradigm LLM
- Mọi cải thiện F1 từ ICL → RAG = đóng góp thuần túy của retrieval

### Bảng so sánh đầy đủ (điền số sau khi chạy xong)
| Phương pháp | Loại | ACD F1 | Combined F1 |
|-------------|------|--------|-------------|
| SVM + TF-IDF | Traditional baseline | 69% | 61% |
| **PhoBERT ours (concat 4 layers)** | **Baseline** | __ | __ |
| PhoBERT ours (cls_only, ablation) | Baseline ablation | __ | __ |
| LLM + ICL k=2 | Proposed | __ | __ |
| LLM + ICL k=4 | Proposed | __ | __ |
| LLM + ICL k=8 | Proposed | __ | __ |
| **LLM + RAG k=2** | **Proposed (đề xuất chính)** | __ | __ |
| **LLM + RAG k=4** | **Proposed (đề xuất chính)** | __ | __ |
| **LLM + RAG k=8** | **Proposed (đề xuất chính)** | __ | __ |
| SOTA Huynh 2022 | Upper bound | 82.55% | 77.32% |

### 3 câu hỏi nghiên cứu cần trả lời trong báo cáo
1. PhoBERT (supervised) vs LLM few-shot: supervised có tốt hơn không và tại sao?
2. RAG vs ICL (cùng k): chọn examples thông minh có cải thiện F1 không?
3. k tăng (2→4→8): thêm examples có luôn giúp ích không?

---

## 4. DATASET

- **Nguồn:** VLSP 2018 Hotel từ `ds4v/absa-vlsp-2018` (CSV format)
- **Size:** train=3000, dev=2000, test=600
- **Format:** cột `Review` + 34 cột aspect (0/1/2/3)
- **Download:**
```bash
git clone https://github.com/ds4v/absa-vlsp-2018.git /tmp/ds4v --depth=1
cp /tmp/ds4v/datasets/vlsp2018_hotel/*.csv data/
```

---

## 5. KẾT QUẢ ĐÃ CÓ — TUẦN 1 (DONE)

### EDA thực tế (từ EDA_Summary_Report.md)
| Phát hiện | Giá trị | Hệ quả đã xử lý |
|-----------|---------|----------------|
| p99 review length | 243 words | MAX_SEQ_LEN=384 (không phải 256) |
| Weight neutral (global) | 154× | Clip tại 10.0 |
| ROOM_AMENITIES#PRICES | 0 mẫu train | Exclude khỏi Macro-F1 |
| Segmenter dùng | underthesea (fallback) | Gap ~1-2% vs SOTA, ghi báo cáo |
| SERVICE#GENERAL | 63% reviews | Aspect dominant, dễ học nhất |
| Absent class | 85% tổng nhãn | Mất cân bằng nặng → weighted loss |
| ROOMS#MISCELLANEOUS | Chỉ có nhãn negative | Thêm vào RARE_ASPECTS |
| FOOD&DRINKS#MISCELLANEOUS | weight pos=426 | Thêm vào RARE_ASPECTS |

### RARE_ASPECTS (8, cập nhật từ EDA)
```python
RARE_ASPECTS = [
    "FACILITIES#MISCELLANEOUS",
    "ROOM_AMENITIES#PRICES",         # 0 mẫu train!
    "ROOM_AMENITIES#MISCELLANEOUS",
    "ROOM_AMENITIES#CLEANLINESS",
    "ROOM_AMENITIES#DESIGN&FEATURES",
    "HOTEL#DESIGN&FEATURES",
    "ROOMS#MISCELLANEOUS",           # thêm từ EDA
    "FOOD&DRINKS#MISCELLANEOUS",     # thêm từ EDA
]
ZERO_TRAIN_ASPECTS = ["ROOM_AMENITIES#PRICES"]
```

### Files đã tạo (Tuần 1)
```
outputs/eda/class_weights.json    ← per-aspect weights, tuần 2 load
outputs/eda/encoder_config.json   ← MAX_SEQ_LEN=384, concat_4_layers
data/train_preprocessed.csv       ← 5-bước pipeline, underthesea
data/dev_preprocessed.csv
data/test_preprocessed.csv
```

---

## 6. QUYẾT ĐỊNH KỸ THUẬT ĐÃ CHỐT

### Framework
- PyTorch (không phải TensorFlow — ds4v dùng TF)
- HuggingFace Transformers, underthesea word segmentation

### Kiến trúc PhoBERT Baseline (Tuần 2)
```
vinai/phobert-base (output_hidden_states=True)
→ concat(hidden[-4], hidden[-3], hidden[-2], hidden[-1])[:, CLS, :]
→ [batch, 3072]  ← KHÔNG phải 768 (insight từ SOTA ds4v)
→ Dropout(0.2)
→ 34 × Linear(3072, 4) song song
→ 34 × CrossEntropyLoss với per-aspect weights (clip=10.0)
```

### Hyperparameters Baseline (từ EDA thực tế)
```python
MAX_SEQ_LEN            = 384    # p99=243 × 1.5 → 384
batch_size             = 8      # T4 16GB + seq_len=384
grad_accumulation      = 2      # effective batch = 16
learning_rate          = 2e-5   # chuẩn fine-tuning BERT
warmup_ratio           = 0.1    # 10% steps đầu warmup
weight_clip            = 10.0   # neutral 154 → 10
dropout                = 0.2
early_stop_patience    = 3      # dựa trên dev Combined F1
max_grad_norm          = 1.0    # gradient clipping
seed                   = 42
```

### Ablation Baseline
- Chạy thêm `cls_only` (768 dim) để so sánh với `concat_4_layers` (3072 dim)
- Chứng minh kỹ thuật concat 4 layers có đóng góp thực sự

### Evaluation Metric (step4_eval.py — dùng chung toàn dự án)
- ACD F1: binary per-aspect (present=1/2/3 vs absent=0), macro average
- SPC F1: 3-class {pos/neg/neu} trên subset y_true!=0, macro average
- Combined F1: (ACD F1 + SPC F1) / 2
- Exclude ZERO_TRAIN_ASPECTS khỏi Macro-F1

### Proposed Method — LLM/RAG (Tuần 3)
- Prompt tiếng Việt + Chain-of-Thought
- Output: JSON `{"ASPECT#CATEGORY": "sentiment"}`
- Cache LLM responses để tiết kiệm API cost
- Rate limit: 1 req/giây
- Embedding: `keepitreal/vietnamese-sbert`
- Ablation: k=2,4,8 × GPT-4o-mini+Gemini × ICL+RAG = 12 experiments
- Gemini 1.5 Flash: miễn phí tại aistudio.google.com (15 req/phút)
- GPT-4o-mini: ~$0.70 cho toàn bộ experiments

---

## 7. CẤU TRÚC THƯ MỤC

```
absa-vlsp2018-hotel/
├── data/                            ← CSV gốc + preprocessed cache
├── code/
│   ├── week1/utils/constants.py     ← SINGLE SOURCE OF TRUTH
│   ├── week1/step1_eda.py           ← ✅ done
│   ├── week1/step2_dataloader.py    ← ✅ done
│   ├── week1/step3_preprocessing.py ← ✅ done
│   ├── week1/step4_eval.py          ← ✅ done (dùng chung toàn dự án)
│   ├── week2/model.py               ← 🔄 đang chạy (baseline)
│   ├── week2/train.py
│   ├── week2/predict.py
│   ├── week2/run_experiment.py
│   ├── week3/prompts.py             ← ✅ code sẵn (proposed method)
│   ├── week3/llm_client.py
│   ├── week3/icl_predictor.py
│   ├── week3/rag_retriever.py
│   ├── week3/rag_predictor.py
│   ├── week3/run_tier2.py           ← entry point ICL
│   ├── week3/run_tier3.py           ← entry point RAG
│   └── week3/compare_results.py     ← có __WEEK2_*__ placeholders
├── notebooks/
│   ├── week2_phobert_training.ipynb ← Colab/Kaggle T4 GPU
│   └── week3_llm_rag.ipynb          ← Mac local, gọi API
├── outputs/eda/                     ← biểu đồ + class_weights + encoder_config
├── outputs/models/                  ← checkpoint .pt
├── outputs/results/                 ← metrics JSON
├── CLAUDE.md                        ← context cho Claude Code
├── CODING_DIRECTIVE_WEEK1_v2.md     ← ✅ spec tuần 1
├── CODING_DIRECTIVE_WEEK2.md        ← ✅ spec tuần 2 (baseline)
└── CODING_DIRECTIVE_WEEK3.md        ← ✅ spec tuần 3 (proposed)
```

---

## 8. TRẠNG THÁI HIỆN TẠI

### ✅ Hoàn thành
- Proposal nộp (22/03/2026)
- Tuần 1: EDA, preprocessing, dataloader, eval script
- Tất cả Coding Directives (Week 1, 2, 3)
- CLAUDE.md cho Claude Code
- Tuần 3 code đã implement sẵn (chờ kết quả tuần 2 để điền placeholders)

### 🔄 Đang chạy
- Tuần 2: PhoBERT baseline training trên Kaggle/Colab
- Notebook: `week2_phobert_training.ipynb`
- Cần chạy đủ: main run (concat_4_layers) + ablation (cls_only) + learning curve

### ⏳ Sau khi tuần 2 xong
- Paste `outputs/results/week2_summary.md` vào chat mới
- Claude viết prompt để Claude Code điền các placeholders:
  - `__WEEK2_ACD_F1__`
  - `__WEEK2_SPC_F1__`
  - `__WEEK2_COMBINED_F1__`
  - `__WEEK2_ABLATION_COMBINED_F1__`
- Chạy tuần 3 trên Mac local (không cần GPU)

### ⏳ Chưa làm
- Tuần 4: Báo cáo cuối, slides, nộp trước 21/04/2026

---

## 9. VIỆC CẦN LÀM TIẾP THEO

**Ngay bây giờ (song song với Kaggle training):**
- [ ] Claude Code implement tuần 3 theo `CODING_DIRECTIVE_WEEK3.md`
- [ ] Lấy Gemini API key miễn phí tại aistudio.google.com

**Sau khi tuần 2 xong:**
- [ ] Download `week2_summary.md` từ Kaggle
- [ ] Mở chat mới, paste SESSION_HANDOFF.md + week2_summary.md
- [ ] Nhận prompt cập nhật placeholders → cho Claude Code chạy
- [ ] Chạy tuần 3: `python code/week3/run_tier2.py` + `run_tier3.py`

**Tuần 4 (15-21/04):**
- [ ] Tổng hợp bảng so sánh đầy đủ
- [ ] Viết báo cáo: mô tả baseline + proposed + so sánh + phân tích
- [ ] Giải thích learning curve + early stopping
- [ ] Làm slides
- [ ] Nộp trước 21/04/2026

---

## 10. QUY TẮC CODE BẮT BUỘC (cho Claude Code)

1. Import constants từ `code/week1/utils/constants.py` — không hardcode
2. Paths relative từ root project — không absolute path
3. `set_seed(42)` ở đầu mỗi script
4. Type hints + docstring cho mọi function/class
5. Tên hàm/class không đổi — tuần sau import từ tuần trước
6. Cache mọi thứ tốn thời gian compute
7. Không hardcode API key — dùng parameter hoặc env var

---

## 11. TÀI LIỆU THAM KHẢO CHÍNH

| # | Paper | Vai trò |
|---|-------|---------|
| [1] | Nguyen et al. (2018) VLSP Shared Task. J. Comp. Sci. Cybern. 34(4) | Dataset gốc |
| [2] | Nguyen & Nguyen (2020) PhoBERT. Findings EMNLP | Model backbone |
| [3] | **Huynh et al. (2022) Multi-task ACSA. IEEE MAPR** | **SOTA, kiến trúc baseline** |
| [4] | Nguyen et al. (2024) LLMs for Vietnamese. PACLIC | Zero-shot LLM benchmark |
| [5] | Brown et al. (2020) GPT-3. NeurIPS | Few-shot ICL |
| [6] | Wei et al. (2022) Chain-of-Thought. NeurIPS | CoT prompting |
| [7] | Lewis et al. (2020) RAG. NeurIPS | RAG framework |
| [8] | Gao et al. (2024) RAG Survey. arXiv:2312.10997 | RAG methodology |

---

## 12. CÂU HỎI THƯỜNG GẶP

**Tuần 2 là baseline hay proposed method?**
Baseline. Tuần 3 (LLM + RAG) mới là proposed method. Thầy yêu cầu: chạy baseline tuần 2 rồi so sánh với proposed method tuần 3.

**Tại sao phải có RAG mới lên mức Giỏi?**
Thầy cảnh báo: chỉ dùng BERT = tối đa Khá. Phải có ICL/RAG tiên tiến mới lên Giỏi.

**ICL và RAG có phải 2 hướng riêng không?**
Không. Chúng là 2 cách chọn examples trong cùng 1 paradigm LLM. ICL = random, RAG = semantic retrieval. So sánh để chứng minh retrieval có đóng góp.

**Tại sao MAX_SEQ_LEN=384 không phải 256?**
EDA: p99=243 words × 1.5 (word segment factor) = 364.5 → làm tròn bội số 64 = 384.

**Tại sao weight neutral=154 phải clip về 10?**
Weight quá lớn → gradient explosion → training mất ổn định.

**Tại sao ROOM_AMENITIES#PRICES bị exclude?**
0 mẫu train → model không học được → ACD F1=0 → kéo Macro-F1 xuống oan → exclude và giải thích trong báo cáo.

**Tại sao dùng underthesea không phải VnCoreNLP?**
VnCoreNLP cần Java 8+ và model ~200MB, không khả dụng. underthesea là fallback, kém ~1-2% F1 — ghi rõ trong báo cáo là 1 trong các lý do gap với SOTA.

**Tại sao concat 4 layers không phải cls_only?**
Phân tích repo ds4v (SOTA): concat layers[-4:] tại [CLS] → 3072 dim, tốt hơn cls_only ~1-2% F1. Chạy ablation để chứng minh trong báo cáo.

**Kết quả thấp hơn SOTA có phải lỗi không?**
Không. Gap ~5-11% có lý do cụ thể: underthesea (-1-2%), 0-sample aspect (-0.5-1%), neutral hiếm (-1-2%), dataset nhỏ (-1-2%). Giải thích rõ trong báo cáo.

**Môi trường chạy từng tuần?**
- Tuần 1: Mac local (không cần GPU)
- Tuần 2: Kaggle/Colab T4 GPU (bắt buộc)
- Tuần 3: Mac local (chỉ gọi API, không cần GPU)

---

*Handoff version: 2.0 | Cập nhật: thêm yêu cầu thầy, sửa baseline/proposed đúng*
*Chat mới: paste file này → tiếp tục ngay không cần giải thích lại*
