# Plan Viết Báo Cáo Final — ABSA VLSP 2018 Hotel

> Đây là **plan viết tài liệu**, không phải là bản báo cáo. Đọc duyệt plan này trước, sau đó sẽ triển khai từng section thành `Group16_Report_Final.docx`.
>
> - Ngôn ngữ: **Tiếng Việt** (giữ nguyên thuật ngữ chuyên ngành tiếng Anh: ABSA, PhoBERT, TF-IDF, RAG, ...).
> - Format cuối cùng: **Word (.docx)**.
> - Độ dài target: **8–12 trang A4** (font 12, line 1.15, không kể phụ lục).
> - Phần evaluation explanation/recommendation: **placeholder + TODO** (sẽ điền sau, không viết content thật lần này).

---

## 1. Tổng Quan Plan

### 1.1 Mục tiêu báo cáo

- Trình bày rõ đề tài, phương pháp và kết quả cho thầy/cô chấm.
- Defend được các lựa chọn kỹ thuật (VnCoreNLP, encoder cls_only, RAG k=8, LLM explanation thay vì classifier).
- Cho thấy repo đã deliver đủ: 4 hướng + app demo + notebook executed + metric JSON.
- Đồng nhất số liệu với `outputs/results/final_four_direction_comparison.md` và các file `docs/*.md`.

### 1.2 Ràng buộc chính

| Ràng buộc | Cách xử lý |
|---|---|
| 8–12 trang | Mỗi section có budget trang cụ thể (xem §2). Dùng bảng thay cho đoạn dài. Không bullet list lan man. |
| Tiếng Việt | Các thuật ngữ ABSA / PhoBERT / VnCoreNLP / RAG / ICL / TF-IDF / LLM / ACD / SPC / F1 giữ nguyên tiếng Anh. Viết hoa nhãn `positive/negative/neutral/absent`. |
| Đồng nhất số liệu | Mọi bảng metric phải copy đúng từ `docs/*.md` hoặc `outputs/results/*.json`. Không tự tính lại. |
| Placeholder explanation eval | Mở một ô TODO rõ ràng để sau chèn vào, không làm gãy flow. |

### 1.3 Nguồn nội dung (đã có sẵn trong repo)

Plan bám vào các file đã viết trong repo để **không phải tạo nội dung mới**, chỉ biên tập và rút gọn:

| Cần viết | Lấy nội dung từ |
|----------|-----------------|
| Abstract | `PROJECT_HANDOFF.md` §2, §10 + `README.md` |
| Introduction | `docs/phobert_llm_explainability_plan.md` (Research Narrative) + README |
| Related Work | Viết mới ngắn, tham chiếu Huynh 2022 (đã có số SOTA) + PhoBERT paper + RAG paper |
| Data Analysis | `docs/eda_summary_report.md` + `outputs/eda/*.png` |
| Methodology (baseline + PhoBERT + PhoBERT+LLM Explanation) | `docs/svm_baseline_summary.md` + `docs/phobert_best_single_summary.md` + `docs/phobert_llm_explainability_plan.md` |
| Experiments & Results (có subsection LLM+RAG làm thí nghiệm bổ sung) | `outputs/results/final_four_direction_comparison.md` + `docs/phobert_vncorenlp_ablation.md` + `docs/llm_rag_comparison.md` + các JSON metric |
| Conclusion | `PROJECT_HANDOFF.md` §5.3 + §10 |

---

## 2. Cấu Trúc Báo Cáo — Chi Tiết Từng Section

Tổng budget ~10 trang. Dưới đây liệt kê: tên section, mục tiêu, content cụ thể, figure/table, page budget, nguồn tham chiếu.

### Section 1 — Abstract (≈ 0.5 trang)

**Mục tiêu:** cho thầy cô nắm bài chỉ trong 10 dòng.

**Nội dung (1 đoạn văn, 150–200 từ):**

1. Bài toán ABSA trên VLSP 2018 Hotel (34 aspects × 4 labels).
2. Ba thành phần chính nhóm xây dựng: **baseline SVM + TF-IDF** (0.3173 Combined F1), **PhoBERT predictor** (**0.5543**), và **PhoBERT + LLM Explanation** — hệ thống ứng dụng đóng gói trong Streamlit app.
3. Thí nghiệm bổ sung LLM + RAG k=8 với GPT-4o-mini (0.3532) cho thấy LLM chưa đủ tin cậy để thay PhoBERT làm predictor → củng cố lựa chọn kiến trúc của sản phẩm.
4. Kết luận: PhoBERT làm prediction, LLM làm explanation/evidence/recommended_action.
5. 1 câu về contribution (ablation VnCoreNLP +85.9%, tách vai predictor và explainer).

**Table/Figure:** không.

---

### Section 2 — Introduction (≈ 1 trang)

**Mục tiêu:** đặt vấn đề, lý do chọn đề tài, đóng góp.

**Cấu trúc (chia 3 subsection nhỏ, không đánh số):**

**2.1 Bối cảnh & động lực (≈ 0.3 trang)**
- Tại sao ABSA quan trọng với ngành du lịch / khách sạn.
- Review tiếng Việt có đặc thù: teencode, viết tắt, cần word segmentation.
- Tại sao VLSP 2018 Hotel là benchmark phù hợp.

**2.2 Mục tiêu & phạm vi (≈ 0.3 trang)**
- Task definition: ACD + SPC, 34 aspects, 4 labels.
- Metric: ACD F1, SPC F1, Combined F1 (macro).
- Phân biệt rõ: methodology gồm đúng **3 thành phần được commit vào hệ thống** (baseline, PhoBERT, PhoBERT + LLM Explanation). LLM + RAG là thí nghiệm bổ sung để so sánh, không phải method chính.

**2.3 Đóng góp của báo cáo (≈ 0.4 trang, dạng 4 gạch đầu dòng)**
1. Baseline SVM + TF-IDF có giải thích tham số — xác định "trần dưới" của bài toán.
2. PhoBERT single model với ablation VnCoreNLP (clean ablation +85.9%) — predictor chính.
3. Sản phẩm ứng dụng: PhoBERT + LLM Explanation, đóng gói Streamlit app — hệ thống deploy.
4. Thí nghiệm bổ sung LLM + RAG (ablation k ∈ {2,4,8,16}) củng cố lựa chọn PhoBERT làm predictor thay vì LLM.

**Table/Figure:** không (để dành cho các section sau).

---

### Section 3 — Related Work (≈ 1 trang)

**Mục tiêu:** ngắn gọn, tránh dài dòng vì đây là bài tập lớn chứ không phải paper hội nghị.

**Cấu trúc (3 đoạn ngắn):**

**3.1 ABSA tổng quan (≈ 0.3 trang)**
- Định nghĩa ABSA.
- 2 bài toán con: ACD + SPC.
- Các hướng phổ biến: feature-based (SVM, CRF), neural (LSTM, BiLSTM+attention), pretrained (BERT, PhoBERT).

**3.2 ABSA tiếng Việt & VLSP 2018 (≈ 0.3 trang)**
- Nhắc Huynh 2022 (SOTA hiện tại: ACD F1 0.8255, Combined F1 0.7732).
- Vai trò của VnCoreNLP trong NLP tiếng Việt.
- PhoBERT (Nguyen & Nguyen, 2020) — pretrained trên text đã word-segmented.

**3.3 LLM & RAG cho classification (≈ 0.4 trang)**
- ICL few-shot (Brown et al., 2020).
- RAG (Lewis et al., 2020) — retrieve rồi mới prompt.
- Hạn chế của LLM cho structured classification có nhiều label (hallucination, khó follow schema).

**Table/Figure:** không.

**Citations cần thêm (sẽ tra cứu lúc viết):** PhoBERT paper, VnCoreNLP paper, Huynh 2022, Brown 2020 GPT-3, Lewis 2020 RAG, BERT Devlin 2018.

---

### Section 4 — Data Analysis (≈ 1.5 trang)

**Mục tiêu:** người chấm hiểu được dataset và các đặc thù dẫn đến design choices sau.

**Cấu trúc:**

**4.1 Mô tả dataset (≈ 0.3 trang)**
- Nguồn: `ds4v/absa-vlsp-2018` (HuggingFace).
- Bảng split: Train 3,000 / Dev 2,000 / Test 600 / Total 5,600.
- Schema: 1 cột `Review` + 34 cột aspect, giá trị ∈ {0,1,2,3}.

**4.2 Phân phối nhãn & imbalance (≈ 0.5 trang)**
- **Table 1:** top-10 aspect xuất hiện nhiều nhất + 5 aspect hiếm nhất (lấy từ `docs/eda_summary_report.md`).
- **Figure 1:** `outputs/eda/train_label_breakdown.png` — cho thấy `absent` chiếm đa số, `neutral` rare.
- **Figure 2:** `outputs/eda/train_aspect_presence.png` — imbalance giữa các aspect.
- Nhận xét: dataset mất cân bằng 2 chiều (per-class + per-aspect) → cần `class_weight` / focal loss.

**4.3 Độ dài review (≈ 0.2 trang)**
- **Figure 3:** `outputs/eda/train_review_length.png`.
- p50, p95, p99 → chọn `MAX_SEQ_LEN=256` cho PhoBERT.

**4.4 Pipeline preprocessing (≈ 0.5 trang)**
- **Table 2:** 5 bước pipeline (Unicode NFC / whitespace / teencode / special chars / word segmentation). Lấy từ `docs/eda_summary_report.md`.
- Ví dụ before/after (1–2 mẫu).
- Cache file `data/*_preprocessed.csv`.

**Table/Figure tổng:** Table 1, Table 2; Figure 1, 2, 3.

---

### Section 5 — Methodology (≈ 2.5 trang)

**Mục tiêu:** mô tả 3 thành phần mà nhóm **thực sự xây dựng và commit** vào hệ thống cuối: (1) Baseline, (2) PhoBERT predictor, (3) PhoBERT + LLM Explanation (ứng dụng). LLM + RAG **không được xếp là một phương pháp chính** — nó được coi là thí nghiệm bổ sung và sẽ trình bày riêng ở §6.

**Lý do khung gồm đúng 3 phần:**

1. Methodology mô tả những gì được **xây dựng và giữ lại** trong hệ thống cuối, không phải mọi thí nghiệm đã thử. Baseline và PhoBERT được dùng để huấn luyện và đo lường; PhoBERT + LLM Explanation là sản phẩm demo. Cả 3 đều là thành phần chính thức của deliverable.
2. Ba bước tạo thành một narrative tuyến tính: **baseline chứng minh bài toán khó** → **PhoBERT cho predictor tin cậy** → **LLM biến prediction thành output có giá trị cho người dùng**. Mỗi bước dẫn đến bước sau.
3. LLM + RAG **không nằm trong sản phẩm**, không được deploy trong app, không dùng để tạo metric cuối. Nó chỉ là một thí nghiệm hỗ trợ quyết định "tại sao chọn PhoBERT làm predictor". Vì vậy nó thuộc về phần Experiments chứ không phải Methodology.

**Cấu trúc:**

**5.1 Tổng quan kiến trúc hệ thống (≈ 0.25 trang)**

- **Figure 5 (kiến trúc tổng thể — vẽ mới):** một sơ đồ duy nhất cho thấy mối quan hệ 3 thành phần.
  - Baseline (SVM + TF-IDF) — nhánh so sánh, chỉ để benchmark.
  - PhoBERT predictor — core prediction engine, output vector [34 aspect × 4 label].
  - LLM Explanation layer — nhận output của PhoBERT, tạo explanation / evidence / recommended_action cho người dùng.
- Nhấn mạnh: baseline và PhoBERT là 2 predictor, nhưng chỉ PhoBERT được đưa vào app. LLM không dự đoán nhãn — nó chỉ giải thích.

**5.2 Baseline — SVM + TF-IDF (≈ 0.5 trang)**

- **Vai trò:** mốc truyền thống để đo giá trị của pretrained representation. Không dùng trong app, nhưng là thành phần chính của báo cáo vì nó định nghĩa "trần dưới" của bài toán.
- **Lý do chọn (viết rõ):**
  - Bài toán ABSA trước kỷ nguyên pretrained thường dùng SVM + TF-IDF → đây là baseline hợp lý để so sánh.
  - Train nhanh, interpret được, không cần GPU → tiết kiệm chi phí thực nghiệm.
  - Nếu baseline đã cao gần bằng PhoBERT thì việc đầu tư PhoBERT không có ý nghĩa → baseline là phép thử tính **đáng đầu tư** của hướng pretrained.
- **Config:**
  - Model: LinearSVC (sklearn) × 34 aspects độc lập, mỗi aspect 4-class.
  - Features: TF-IDF unigram + bigram, `max_features=50000`, `C=1.0`, `class_weight='balanced'`.
  - Preprocessing: chỉ lowercase + remove special chars, **không** word segmentation (chủ ý để giữ baseline độc lập với pipeline phức tạp của PhoBERT).
- **Table 3:** config SVM (copy từ `docs/svm_baseline_summary.md`).

**5.3 PhoBERT Predictor (≈ 1.0 trang)**

- **Vai trò:** core prediction engine của hệ thống — đây là model được deploy trong app.
- **Lý do chọn (viết rõ):**
  - PhoBERT là pretrained language model tiếng Việt mạnh nhất công khai, được train trên corpus tiếng Việt đã word-segmented.
  - Fine-tune trực tiếp trên 34 aspect × 4 label → model học được chính xác label scheme của VLSP 2018, khác với LLM prompt-based chỉ thấy vài ví dụ.
  - Inference deterministic, không tốn API call → phù hợp làm engine cho app production.
  - Phù hợp với Kaggle T4 GPU (không cần cluster lớn), single model (không ensemble) → đủ để defend trong phạm vi bài tập lớn.
- **Kiến trúc:**
  - Backbone: `vinai/phobert-base-v2`.
  - `ABSAPhoBERT` multi-task: 34 classification heads song song, mỗi head output 4 logit (absent / positive / negative / neutral).
  - Encoder option: `cls_only` (đã chọn sau ablation với `concat_4_layers`; xem §6).
- **Loss & training:**
  - Weighted focal loss dựa trên `outputs/eda/class_weights.json` (tính từ train split để tránh leak).
  - `weight_clip=10.0` để tránh class hiếm chi phối loss.
  - Early stopping theo Combined F1 trên dev.
- **Preprocessing bắt buộc:** VnCoreNLP word segmentation (khớp với format pretraining). Phần ablation chứng minh thấy vai trò quyết định của bước này sẽ nằm ở §6.4.
- **Table 4:** hyperparameter (MAX_SEQ_LEN 256, BS 16, LR 1e-4, warmup 0.15, epochs 20, patience 7, seed 42) — copy từ `docs/phobert_best_single_summary.md`.
- **Figure 6 (architecture diagram — vẽ mới):** Input → VnCoreNLP → PhoBERT encoder → [CLS] vector 768-d → 34 heads × 4 logit.
- **Figure 7:** `outputs/eda/learning_curve_cls_only.png`.

**5.4 PhoBERT + LLM Explanation — hệ thống ứng dụng (≈ 0.75 trang)**

- **Vai trò:** đây là **sản phẩm demo cuối cùng** của đề tài, được đóng gói thành Streamlit app (`app/`). Không phải classifier mới, không thay đổi nhãn của PhoBERT.
- **Lý do chọn kiến trúc tách predictor và explainer (viết rõ):**
  - PhoBERT đã có metric có thể defend được (Combined F1 = 0.5543). Không nên để LLM thay đổi nhãn → giữ metric vững.
  - LLM rất mạnh ở phần sinh ngôn ngữ tự nhiên nhưng yếu khi phải tuân thủ schema 34 label nhất quán (xem §6.6). Vì vậy phân vai: **PhoBERT lo label**, **LLM lo diễn giải**.
  - Người dùng cuối (chủ khách sạn) cần không chỉ nhãn mà còn: (a) câu trong review dẫn đến nhãn đó, (b) lời giải thích dễ đọc, (c) gợi ý hành động. Đây là giá trị LLM mang lại mà PhoBERT một mình không cho được.
- **Pipeline:**
  - **Figure 8 (pipeline app — vẽ mới hoặc reuse `outputs/reports/app_demo_mockup.png`):** Review (text hoặc CSV) → PhoBERT predict → lọc chỉ giữ present predictions → LLM nhận (review + present predictions) → generate JSON {items, overall_summary, recommended_action} → Streamlit UI hiển thị.
- **Thiết kế prompt & output:**
  - JSON schema cố định: `items` (mỗi aspect có `aspect`, `sentiment`, `explanation`, `evidence`, `evidence_uncertain`), `overall_summary`, `recommended_action`.
  - Temperature thấp (0–0.2) để explanation deterministic, không "sáng tạo".
  - Evidence check: substring / normalize match vào review gốc; nếu không match thì flag `evidence_uncertain=true`.
- **Không kèm bảng metric tại đây** — metric đánh giá chất lượng explanation/recommendation sẽ nằm ở §6.7 (placeholder).

**Table/Figure tổng của §5:** Table 3, 4; Figure 5, 6, 7, 8.

---

### Section 6 — Experiments and Results (≈ 2.5 trang)

**Mục tiêu:** số liệu + phân tích, đây là core của báo cáo.

**Cấu trúc:**

**6.1 Setup chung (≈ 0.2 trang)**
- Train/Dev/Test splits.
- Metric: ACD F1, SPC F1, Combined F1 (macro across 34 aspects).
- Môi trường: Kaggle T4 GPU + local CPU cho SVM.
- Seed cố định (PhoBERT seed=42).

**6.2 Bảng kết quả tổng hợp (≈ 0.3 trang)**
- **Table 5 (bảng chính):** copy `outputs/results/final_four_direction_comparison.md`. Ghi chú rõ dòng LLM + RAG là **thí nghiệm bổ sung**, không phải method chính.

  | Method | Vai trò | ACD F1 | SPC F1 | Combined F1 |
  |---|---|---:|---:|---:|
  | SVM + TF-IDF | Baseline (method chính) | 0.4074 | 0.2272 | 0.3173 |
  | PhoBERT cls_only + VnCoreNLP | Predictor (method chính) | 0.6360 | 0.4727 | **0.5543** |
  | PhoBERT + LLM Explanation | Application (method chính) | — | — | — |
  | LLM + RAG k=8 (GPT-4o-mini) | *Thí nghiệm bổ sung* | 0.4034 | 0.3031 | 0.3532 |

**6.3 SVM baseline chi tiết (≈ 0.2 trang)**
- Dev vs Test metric (từ `outputs/results/svm_baseline_*_metrics.json`).
- Nhận xét: SPC F1 rất thấp (0.2272) vì TF-IDF không bắt được context phủ định / so sánh.

**6.4 PhoBERT: ablation VnCoreNLP (≈ 0.5 trang) — đây là phần nổi bật nhất**
- **Table 6 (ablation):** copy từ `docs/phobert_vncorenlp_ablation.md`.

  | Setting | Encoder | ACD F1 | SPC F1 | Combined F1 |
  |---|---|---:|---:|---:|
  | No VnCoreNLP | concat_4_layers | 0.3592 | 0.2371 | 0.2981 |
  | No VnCoreNLP | cls_only | 0.3349 | 0.2014 | 0.2681 |
  | VnCoreNLP | concat_4_layers | 0.5836 | 0.4658 | 0.5247 |
  | VnCoreNLP | cls_only | **0.6360** | **0.4727** | **0.5543** |

- Kết luận: **+0.2562 Combined F1 (+85.9% tương đối)**.
- Giải thích: PhoBERT pretrained trên text đã segmented → cần VnCoreNLP để khớp token boundary.

**6.5 PhoBERT: cls_only vs concat_4_layers (≈ 0.2 trang)**
- cls_only 0.5543 > concat_4_layers 0.5247 (+0.0296).
- Giải thích: ít param hơn ở head → giảm overfit trên 3k train samples.

**6.6 Thí nghiệm bổ sung — LLM + RAG như một predictor thay thế (≈ 0.5 trang)**

- **Đặt vấn đề:** trước khi chốt PhoBERT làm predictor cho app, nhóm đặt câu hỏi "nếu không train supervised model mà dùng LLM prompt-based + retrieval thì có đủ tốt không?". Mục của subsection này là trả lời câu hỏi đó bằng số liệu — đây là lý do LLM + RAG xuất hiện trong báo cáo.
- **Setup ngắn (≈ 5 dòng):** GPT-4o-mini + sentence-transformer retriever trên train set, prompt JSON schema cho 34 aspect labels, cache response.
- **Ablation theo k (Table 7, copy từ `docs/llm_rag_comparison.md`):**

  | Method | k | ACD F1 | SPC F1 | Combined F1 |
  |---|---:|---:|---:|---:|
  | ICL GPT-4o-mini | 8 | 0.3504 | 0.2567 | 0.3035 |
  | RAG GPT-4o-mini | 2 | 0.3793 | 0.2866 | 0.3330 |
  | RAG GPT-4o-mini | 4 | 0.3866 | 0.2883 | 0.3375 |
  | **RAG GPT-4o-mini** | **8** | **0.4034** | **0.3031** | **0.3532** |
  | RAG GPT-4o-mini | 16 | 0.3988 | 0.2879 | 0.3433 |

- **Quan sát:** RAG cải thiện so với ICL ngẫu nhiên (+0.0497). k=8 tối ưu; k=16 giảm vì context overload.
- **Kết luận (quan trọng — liên kết về methodology):** ngay cả với RAG k=8 tốt nhất, LLM vẫn thua PhoBERT 0.201 điểm Combined F1. **Điều này lý giải tại sao §5.3 chọn PhoBERT làm core predictor trong app thay vì LLM.** Thí nghiệm này không phải method chính — nó tồn tại để **ủng hộ quyết định kiến trúc**.

**6.7 Đánh giá PhoBERT + LLM Explanation (≈ 0.4 trang) — PLACEHOLDER**

> **[TODO — SẼ VIẾT SAU]**
>
> Phần này dành ô trống ~0.4 trang để sau này điền kết quả đánh giá chất lượng explanation/recommendation. Khi viết sẽ gồm:
>
> 1. Heuristic metrics (target đã set trong `docs/phobert_llm_explainability_plan.md`):
>    - JSON parse rate (target ≥ 95%)
>    - Aspect preservation rate (target = 100%)
>    - Sentiment preservation rate (target = 100%)
>    - Evidence substring match rate
>    - Evidence uncertain rate
>    - Average explanation length
> 2. Manual rubric 20–30 examples: Evidence correctness / Explanation quality / Hallucination / User usefulness (scale 0–2).
> 3. Vài ví dụ qualitative (good case + uncertain case).
>
> Không restructure section — chỉ cần điền bảng + 1–2 ví dụ vào ô trống này khi có kết quả.

**6.8 So sánh với SOTA (≈ 0.3 trang)**
- PhoBERT cls_only 0.5543 vs Huynh 2022 0.7732.
- Giải thích khoảng cách: không ensemble, không augmentation, PhoBERT base không phải large, tuning thực dụng trên Kaggle T4.
- Trong phạm vi bài tập lớn, khoảng cách này được accept.

**Table/Figure tổng của §6:** Table 5, 6, 7; không thêm figure mới.

---

### Section 7 — Conclusion (≈ 0.7 trang)

**Cấu trúc:**

**7.1 Tóm tắt kết quả chính (≈ 0.25 trang)**
- Liệt kê 3 số: 0.3173 / 0.5543 / 0.3532.
- Nhấn mạnh: PhoBERT vượt xa 2 hướng còn lại.
- Contribution: ablation VnCoreNLP clean +85.9%.

**7.2 Đóng góp ứng dụng (≈ 0.2 trang)**
- Streamlit app: PhoBERT prediction + LLM explanation + evidence + recommended_action.
- Tách biệt vai trò classifier và explainer → vừa defend được F1, vừa useful cho hotel owner.

**7.3 Hạn chế (≈ 0.1 trang)**
- Không ensemble, không augmentation, không tuning dài.
- Dataset nhỏ 3k train.
- Cache LLM còn phụ thuộc vào một provider (OpenAI).

**7.4 Future work (≈ 0.15 trang)**
- Đánh giá định lượng + định tính chất lượng explanation/recommendation (sẽ làm — hiện tại là placeholder trong §6.7).
- Thử PhoBERT large / XLM-R.
- Thử augmentation + ensemble.
- Mở rộng sang domain khác (restaurant, e-commerce).

---

## 3. Figure & Table Plan — Tổng Hợp

| ID | Loại | Nội dung | Nguồn |
|----|------|----------|-------|
| Table 1 | Table | Phân phối nhãn top-10 aspect | `docs/eda_summary_report.md` |
| Table 2 | Table | 5 bước pipeline preprocessing | `docs/eda_summary_report.md` |
| Table 3 | Table | Config SVM (baseline) | `docs/svm_baseline_summary.md` |
| Table 4 | Table | Hyperparameter PhoBERT | `docs/phobert_best_single_summary.md` |
| Table 5 | Table | Bảng kết quả tổng hợp (3 method chính + 1 thí nghiệm bổ sung) | `outputs/results/final_four_direction_comparison.md` |
| Table 6 | Table | Ablation VnCoreNLP × encoder | `docs/phobert_vncorenlp_ablation.md` |
| Table 7 | Table | Ablation k ∈ {2,4,8,16} của thí nghiệm LLM+RAG | `docs/llm_rag_comparison.md` |
| Figure 1 | PNG | Label breakdown (absent/pos/neg/neutral) | `outputs/eda/train_label_breakdown.png` |
| Figure 2 | PNG | Aspect presence imbalance | `outputs/eda/train_aspect_presence.png` |
| Figure 3 | PNG | Review length distribution | `outputs/eda/train_review_length.png` |
| Figure 4 | PNG | Learning curve PhoBERT cls_only | `outputs/eda/learning_curve_cls_only.png` |
| Figure 5 | Diagram (vẽ mới) | Kiến trúc tổng thể 3 thành phần: baseline / PhoBERT / PhoBERT+LLM Explanation | Vẽ trong docx bằng SmartArt hoặc export từ PowerPoint |
| Figure 6 | Diagram (vẽ mới) | Architecture PhoBERT multi-head (Input → VnCoreNLP → PhoBERT → [CLS] → 34 heads) | Vẽ trong docx |
| Figure 7 | PNG | Learning curve PhoBERT (duplicate Figure 4, có thể bỏ nếu trùng) | `outputs/eda/learning_curve_cls_only.png` |
| Figure 8 | Diagram (vẽ mới) | Pipeline app PhoBERT + LLM Explanation | Vẽ trong docx — có thể reuse `outputs/reports/app_demo_mockup.png` nếu hợp |

**Lưu ý:**
- Figure 5, 6, 8 cần vẽ mới. Có thể lấy từ `outputs/reports/absa_hotel_intro_slides_visual.pptx` nếu slide visual đã có diagram tương đương.
- Không còn "Table param LLM+RAG" riêng — thông tin này gộp vào phần narrative ngắn (~5 dòng) của §6.6.
- Nếu trùng figure (Fig 4 vs Fig 7), sẽ bỏ một khi viết thật để tránh dư.
- Số table giảm từ 8 xuống 7 vì bỏ table param riêng cho LLM+RAG (LLM+RAG không còn là method chính).

---

## 4. Style Guide

| Hạng mục | Quy ước |
|----------|---------|
| Ngôn ngữ | Tiếng Việt. Thuật ngữ chuyên ngành giữ tiếng Anh (ABSA, PhoBERT, RAG, ICL, TF-IDF, LLM, ACD, SPC). |
| Font | Times New Roman 12, line spacing 1.15, căn đều (justify). |
| Heading | Heading 1 (section) / Heading 2 (subsection). Đánh số `1.`, `1.1`, `1.1.1`. |
| Số thập phân | 4 chữ số sau dấu `.` cho F1 (ví dụ `0.5543`). Bold số cao nhất trong mỗi bảng. |
| Placeholder nhãn | `absent / positive / negative / neutral` giữ tiếng Anh, in nghiêng. |
| Citation | Kiểu `(Tác giả, năm)`, danh mục cuối báo cáo. |
| Code snippet | Tránh chèn code dài. Nếu cần, dùng monospace font, 1–5 dòng. |
| Bảng | Có tiêu đề `Bảng X.` phía trên. Figure: `Hình X.` phía dưới. |

---

## 5. Workflow Triển Khai

Chia thành 6 bước, mỗi bước là một task rõ ràng.

| Bước | Tên | Output | Est. thời gian |
|------|-----|--------|----------------|
| 1 | Tạo skeleton `Group16_Report_Final.docx` với heading đúng + page budget ghi chú ở mỗi section | File docx trống có cấu trúc | 15 phút |
| 2 | Viết Abstract + Introduction + Related Work (§1, §2, §3) | ~2.5 trang đầu | 45 phút |
| 3 | Viết Data Analysis (§4) + chèn Figure 1,2,3 + Table 1,2 | ~1.5 trang | 45 phút |
| 4 | Viết Methodology (§5) + chèn Table 3,4,5 + vẽ Figure 5,6,7 | ~2.5 trang | 60 phút |
| 5 | Viết Experiments & Results (§6) + chèn Table 6,7,8 + ô placeholder §6.7 | ~2.5 trang | 60 phút |
| 6 | Viết Conclusion (§7) + References + proofreading + chỉnh layout | ~1 trang + tidy | 45 phút |

Tổng: ~5 giờ thực làm.

**Điểm kiểm tra cuối (checklist trước khi nộp):**

- [ ] Đủ 7 section theo yêu cầu: Abstract, Introduction, Related Work, Data Analysis, Methodology, Experiments & Results, Conclusion.
- [ ] Methodology (§5) gồm **đúng 3 thành phần**: Baseline / PhoBERT / PhoBERT + LLM Explanation. Không có LLM+RAG ở đây.
- [ ] LLM+RAG xuất hiện trong §6.6 như **thí nghiệm bổ sung** để củng cố lựa chọn PhoBERT, không được gọi là "method" ở bất kỳ section nào.
- [ ] PhoBERT + LLM Explanation được phát biểu rõ là **hệ thống ứng dụng** được deploy trong app.
- [ ] Số liệu trong báo cáo trùng khớp với `outputs/results/final_four_direction_comparison.md`.
- [ ] Ablation VnCoreNLP +85.9% có giải thích rõ.
- [ ] Section 6.7 có ô placeholder + TODO rõ ràng cho explanation evaluation.
- [ ] Tổng số trang 8–12 (không kể phụ lục).
- [ ] References đầy đủ.
- [ ] Không còn từ "TODO" ngoài ô §6.7.

---

## 6. Sau Này Khi Làm Evaluation Explanation

Khi đã có kết quả đánh giá explanation/recommendation:

1. Chạy `code/llm_explainability/run_explainability.py` trên 20–30 samples.
2. Tính heuristic metrics + manual rubric (theo spec trong `docs/phobert_llm_explainability_plan.md`).
3. Mở file `Group16_Report_Final.docx`, tìm section **6.7**, xóa khung TODO, điền:
   - 1 bảng heuristic metric.
   - 1 bảng rubric (mean score 4 tiêu chí).
   - 1–2 ví dụ qualitative.
4. Cập nhật §7.4 Future Work: bỏ dòng "đánh giá explanation" nếu đã hoàn thành.
5. Không cần restructure section nào khác.

---

## 7. Editorial Notes (cập nhật sau v1)

Các quyết định biên tập đã được áp dụng từ v2 trở đi:

**Đã bỏ:**
- Mọi dấu hiệu phiên bản (`Phiên bản 1`, `v1`, ghi chú phiên bản, dòng kết thúc)
- §2.4 Cấu trúc báo cáo (boilerplate)
- §3.4 Vị trí của công trình (lặp lại intro)
- §4.5 Tính nhất quán split (minor technical)
- §6.5 So sánh encoder cls_only vs concat_4_layers — concat_4_layers kết quả kém hơn, không có giá trị thảo luận thêm
- §6.9 Chi phí tính toán (không học thuật)
- Toàn bộ đề cập đường dẫn file code/outputs trong nội dung văn bản

**Đổi cấu trúc §6:**
- §6.5 (cũ: So sánh encoder) → đã xóa
- §6.6 (cũ: Thí nghiệm bổ sung) → **§6.5 Thí nghiệm bổ sung — LLM + RAG**
- §6.7 (cũ: Đánh giá) → **§6.6 Đánh giá chất lượng Explanation và Recommendation**
- §6.8 (cũ: SOTA) → **§6.7 So sánh với SOTA**

**Tone & content:**
- §6.6/§6.5 (mới): bỏ "Đặt vấn đề", bỏ "Kết luận liên kết về methodology" — thay bằng 1 câu trung tính
- LLM+RAG chỉ được nhắc như thí nghiệm bổ sung để có số liệu so sánh; không viết rằng LLM "không đủ thay" supervised model
- §7.4 Future Work: chỉ 4 ý chung chung (đánh giá explanation, kiến trúc lớn hơn/ensemble, augmentation, mở rộng domain)
- Loại bỏ văn phong documentation/hướng dẫn

**File hiện tại:** `Group16_Report_v3.docx` (~22–24 trang)

---

## 8. Tóm Tắt Plan Một Dòng

> Báo cáo tiếng Việt 20–25 trang .docx, 7 section học thuật (Abstract / Introduction / Related Work / Data Analysis / Methodology / Experiments & Results / Conclusion). Methodology gồm 3 thành phần (Baseline SVM, PhoBERT predictor, PhoBERT + LLM Explanation). LLM+RAG là thí nghiệm bổ sung ở §6.5, chỉ đóng vai trò cung cấp số liệu so sánh. §6.6 để placeholder cho evaluation explanation.
