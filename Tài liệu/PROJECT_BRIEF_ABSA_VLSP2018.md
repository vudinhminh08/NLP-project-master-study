# PROJECT BRIEF — ABSA VLSP 2018 Hotel (NLP Course)

> Paste toàn bộ file này vào đầu session Claude mới để tiếp tục làm việc.

---

## 1. Thông tin cơ bản

- **Môn học:** Xử lý Ngôn ngữ Tự nhiên — Bài tập lớn
- **Trường:** Đại học Bách Khoa Hà Nội
- **Deadline nộp proposal:** 22/03/2026 | **Deadline cuối:** 21/04/2026
- **Nhóm:** 3 thành viên (TV1, TV2, TV3)

---

## 2. Bài toán

**Aspect-Based Sentiment Analysis (ABSA)** cho đánh giá khách sạn tiếng Việt.

Hai subtask thực hiện **đồng thời** (end-to-end):
- **ACD** — Aspect Category Detection: Phát hiện các cặp `Entity#Attribute` được đề cập trong review
- **SPC** — Sentiment Polarity Classification: Phân loại cảm xúc (positive / negative / neutral) cho mỗi aspect

**Ví dụ:**
```
Input:  "Phòng rộng rãi sạch sẽ nhưng nhân viên lễ tân thái độ hơi kém."
Output: {ROOMS#COMFORT: positive, ROOMS#CLEANLINESS: positive, SERVICE#GENERAL: negative}
```

---

## 3. Dataset

**Nguồn:** VLSP 2018 Shared Task — Sentiment Analysis, Hotel domain  
**Truy cập qua:** https://github.com/ds4v/absa-vlsp-2018/tree/main/datasets/vlsp2018_hotel  
**Cite:** Nguyen T.M.H. et al. (2018). VLSP Shared Task: Sentiment Analysis. *Journal of Computer Science and Cybernetics*, 34(4), 295–310.

| Split | Số reviews |
|---|---|
| Train | 3.000 |
| Dev | 2.000 |
| Test | 600 |

**Format CSV** (ds4v repo):
- Cột `Review`: text đánh giá
- 34 cột aspect: giá trị `0` (absent), `1` (positive), `2` (negative), `3` (neutral)

**34 Aspect Categories:**
```
FACILITIES: CLEANLINESS, COMFORT, DESIGN&FEATURES, GENERAL, MISCELLANEOUS, PRICES, QUALITY (7)
FOOD&DRINKS: MISCELLANEOUS, PRICES, QUALITY, STYLE&OPTIONS (4)
HOTEL: CLEANLINESS, COMFORT, DESIGN&FEATURES, GENERAL, MISCELLANEOUS, PRICES, QUALITY (7)
LOCATION: GENERAL (1) ← chỉ 1 attribute
ROOMS: CLEANLINESS, COMFORT, DESIGN&FEATURES, GENERAL, MISCELLANEOUS, PRICES, QUALITY (7)
ROOM_AMENITIES: CLEANLINESS, COMFORT, DESIGN&FEATURES, GENERAL, MISCELLANEOUS, PRICES, QUALITY (7)
SERVICE: GENERAL (1) ← chỉ 1 attribute
```

**Vấn đề mất cân bằng đã xác định:**
- SERVICE#GENERAL: ~66% reviews (dominant)
- ROOMS#COMFORT, ROOMS#CLEANLINESS: ~42-46%
- **Rất hiếm (<100 mẫu):** FACILITIES#MISCELLANEOUS (~56), ROOM_AMENITIES#PRICES (~62), ROOM_AMENITIES#MISCELLANEOUS (~69), ROOM_AMENITIES#CLEANLINESS (~86), ROOM_AMENITIES#DESIGN&FEATURES (~91), HOTEL#DESIGN&FEATURES (~99)
- **Xử lý:** Weighted loss cho PhoBERT; liệt kê aspect hiếm rõ trong prompt LLM

---

## 4. SOTA trên VLSP 2018 Hotel

| Phương pháp | ACD F1 | ACD+SPC F1 | Paper |
|---|---|---|---|
| SVM + features thủ công | 69% | 61% | Dang et al., VLSP 2018 |
| MLP + TF-IDF | ~65% | ~57% | Nguyen et al., VLSP 2018 |
| **Multi-task PhoBERT (SOTA)** | **82.55%** | **77.32%** | Huynh et al., IEEE 2022 |
| Prompt LLM zero-shot (GPT-4o) | — | ~65-70% | Nguyen et al., PACLIC 2024 |

---

## 5. Phương pháp đề xuất (3 tầng)

### Tầng 1 — Multi-task PhoBERT (Baseline)

**Model:** `vinai/phobert-base` (HuggingFace)  
**Kiến trúc:**
```
Input: [CLS] review_text [SEP]
→ PhoBERT encoder
→ [CLS] embedding (768 dim)
→ 34 Dense(4) output heads SONG SONG
→ Mỗi head: softmax {None, Positive, Negative, Neutral}
```

**Lý do multi-task thay vì 2 model riêng:**
- 1 lần forward pass (nhanh hơn)
- 34 heads chia sẻ encoder → học tốt hơn
- Không có error propagation từ ACD sang SPC
- Đây là kiến trúc đạt SOTA 82.55%

**Preprocessing bắt buộc:**
1. Chuẩn hóa Unicode (ờ != ờ)
2. Teencode: "sv" → "dịch vụ", "ntv" → "nhân viên"
3. Word segmentation: VnCoreNLP RDRSegmenter

**Hyperparameters:**
- LR: 2e-5, AdamW, batch 16, max 20 epochs
- Early stopping patience 3 dựa trên Macro-F1 dev set
- Loss: CrossEntropy với class weights (xử lý mất cân bằng)

**Yêu cầu báo cáo bắt buộc từ thầy:**
- Vẽ learning curve (train/val loss + Macro-F1 theo epoch)
- Giải thích TẠI SAO dừng ở epoch đó (overfitting bắt đầu từ đâu)

### Tầng 2 — Few-shot LLM + Chain-of-Thought tiếng Việt

**LLM chính:** GPT-4o-mini | **LLM so sánh:** Gemini 1.5 Flash  
**Ablation:** k = 2, 4, 8 examples (random selection)  
**Output format:** JSON `{"ASPECT#CATEGORY": "positive/negative/neutral"}`

**Prompt structure:**
```
[SYSTEM] Bạn là chuyên gia phân tích đánh giá khách sạn tiếng Việt.
[DANH SÁCH 34 ASPECTS HỢP LỆ]
[k EXAMPLES với CoT reasoning]
[TEST REVIEW]
```

**Cite:** Brown et al. (2020) Few-Shot Learners. Wei et al. (2022) Chain-of-Thought.

### Tầng 3 — Aspect-aware RAG Few-shot (Đề xuất chính)

**Embedding model:** `keepitreal/vietnamese-sbert` hoặc `paraphrase-multilingual-MiniLM-L12-v2`

**Pipeline:**
1. Embed toàn bộ 3.000 train reviews
2. Với mỗi test review → cosine similarity → lấy k reviews train gần nhất
3. Aspect-aware pool: đảm bảo retrieved examples đa dạng aspect
4. Cùng LLM + format như Tầng 2, chỉ khác cách chọn examples

**Điểm then chốt:** So sánh fair Tầng 2 vs Tầng 3 — cùng LLM, cùng k, cùng format. Mọi cải thiện F1 → do chất lượng retrieval.

**Cite:** Lewis et al. (2020) RAG. Gao et al. (2024) RAG Survey arXiv:2312.10997.

---

## 6. Kế hoạch thực hiện

| Tuần | Thời gian | Nội dung | Phân công |
|---|---|---|---|
| 1 | 23/03–30/03 | EDA, preprocessing, data loader, eval script | Cả nhóm |
| 2 | 31/03–07/04 | Train PhoBERT multi-task, learning curve, Tầng 1 | TV1+TV2 |
| 3 | 08/04–14/04 | Tầng 2 (ICL), Tầng 3 (RAG), ablation | TV1+TV3 |
| 4 | 15/04–21/04 | Tổng hợp, báo cáo, slides, nộp | Cả nhóm |

---

## 7. Môi trường & Tools

```
Python 3.10+
HuggingFace Transformers + PyTorch
vinai/phobert-base (model chính Tầng 1)
VnCoreNLP (RDRSegmenter — word segmentation)
keepitreal/vietnamese-sbert (embedding Tầng 3)
OpenAI API / Gemini API (Tầng 2 & 3)
FAISS (optional, tăng tốc retrieval Tầng 3)
Google Colab Pro hoặc Kaggle (GPU T4)
GitHub (quản lý code)
```

---

## 8. Tài liệu tham khảo chính

```
[1] Nguyen T.M.H. et al. (2018). VLSP Shared Task: Sentiment Analysis.
    Journal of Computer Science and Cybernetics, 34(4), 295–310.

[2] Nguyen D.Q. & Nguyen A.T. (2020). PhoBERT: Pre-trained language models
    for Vietnamese. Findings of EMNLP 2020.

[3] Huynh X.L. et al. (2022). Multi-task Solution for Aspect Category
    Sentiment Analysis on Vietnamese Datasets. IEEE MAPR 2022.
    → SOTA: ACD F1=82.55%, ACD+SPC F1=77.32%

[4] Nguyen C.V. et al. (2024). Prompt Engineering with LLMs for Vietnamese.
    PACLIC 2024. → LLM zero-shot trên VLSP Hotel

[5] Brown T. et al. (2020). Language Models are Few-Shot Learners. NeurIPS 2020.

[6] Wei J. et al. (2022). Chain-of-Thought Prompting. NeurIPS 2022.

[7] Lewis P. et al. (2020). RAG for Knowledge-Intensive NLP Tasks. NeurIPS 2020.

[8] Gao Y. et al. (2024). RAG for LLMs: A Survey. arXiv:2312.10997.

[9] Zhang W. et al. (2024). Sentiment Analysis in the Era of LLMs: A Reality Check.
    Findings of NAACL 2024.
```

---

## 9. Các quyết định đã chốt

- ✅ Dataset: VLSP 2018 Hotel (ds4v repo) — KHÔNG đăng ký VLSP gốc
- ✅ Không dùng SemEval-2016 English (dataset cũ của proposal v1)
- ✅ Không xây dựng nhãn 5-class — dùng 3-class gốc của VLSP
- ✅ Kiến trúc Tầng 1: multi-task 34 heads (không phải 2 model riêng)
- ✅ Xử lý mất cân bằng: weighted loss (KHÔNG gộp MISCELLANEOUS)
- ✅ Embedding Tầng 3: vietnamese-sbert (không phải all-MiniLM-L6-v2 tiếng Anh)
- ✅ Proposal đã nộp: Proposal_VLSP2018_Hotel_v1.docx (bản cập nhật hoàn chỉnh)

---

## 10. Việc cần làm tiếp theo (ở session mới)

- [ ] Bước 1: EDA thực tế trên data — đếm distribution thật của 34 aspects
- [ ] Bước 2: Viết data loader cho format CSV (cột Review + 34 cột aspect)
- [ ] Bước 3: Preprocessing pipeline (Unicode norm + teencode + VnCoreNLP)
- [ ] Bước 4: Xây dựng PhoBERT multi-task model + weighted loss
- [ ] Bước 5: Training loop + early stopping + learning curve logging
- [ ] Bước 6: Eval script (F1 từng aspect + Macro-F1 tổng)
- [ ] Bước 7: Tầng 2 — prompt tiếng Việt + ablation k=2,4,8
- [ ] Bước 8: Tầng 3 — RAG embedding + retrieval + so sánh vs Tầng 2
