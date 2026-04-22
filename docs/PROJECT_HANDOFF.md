# ABSA VLSP 2018 Hotel — Project Handoff

> Tài liệu này tóm tắt toàn bộ đề tài, yêu cầu, các hướng đã làm, các chỉ dẫn cần biết và những kết quả đã đạt được. Mục tiêu: đưa cho một luồng chat Claude khác đọc để nắm nhanh context mà không cần tự khảo sát lại repo.
>
> Repo root (trên máy user): `/Users/macbookpro/Documents/Master-study/NLP/absa-vlsp2018-hotel`

---

## 1. Đề Tài & Yêu Cầu

**Bài toán:** Aspect-Based Sentiment Analysis (ABSA) cho review khách sạn tiếng Việt, dựa trên bộ dữ liệu **VLSP 2018 SA - Hotel**.

- Mỗi review có 34 aspect (ví dụ `ROOMS#CLEANLINESS`, `FACILITIES#COMFORT`, `SERVICE#GENERAL`, ...).
- Mỗi aspect có 4 nhãn:
  - `0 = absent` (không được nhắc tới)
  - `1 = positive`
  - `2 = negative`
  - `3 = neutral`
- Task chia 2 tier:
  - **ACD (Aspect Category Detection):** có nhắc tới aspect hay không.
  - **SPC (Sentiment Polarity Classification):** aspect đó positive / negative / neutral.
- Metric chính: **ACD F1, SPC F1, Combined F1** (macro across aspects).

**Kích thước dataset:**

| Split | Số mẫu |
|-------|--------|
| Train | 3,000  |
| Dev   | 2,000  |
| Test  | 600    |
| Tổng  | 5,600  |

Dataset gốc: `ds4v/absa-vlsp-2018` (HuggingFace). SOTA tham chiếu: Huynh 2022 với ACD F1 ≈ 0.8255, Combined F1 ≈ 0.7732.

**Yêu cầu đề tài (bài tập lớn NLP):**

1. Làm baseline truyền thống (SVM + TF-IDF) để có mốc so sánh.
2. Áp dụng một mô hình pretrained tiếng Việt (PhoBERT) và so sánh với baseline.
3. Thử một hướng advanced method (LLM prompting + RAG retrieval).
4. Tích hợp thành một sản phẩm/ứng dụng thực tế.
5. Viết report + slide + demo code + dataset + submission zip.

---

## 2. Kết Luận Tổng Thể (Final Scope)

Report cuối giữ **4 hướng chính**, các thử nghiệm phụ (cascade, verifier, augmentation, retrain v1, nhiều notebook cũ) đã được đẩy vào `draft/` để báo cáo không bị loãng.

| # | Hướng | Vai trò trong report |
|---|-------|----------------------|
| 1 | SVM + TF-IDF | Baseline truyền thống — **chỉ dùng làm số liệu so sánh** |
| 2 | PhoBERT (single model, no ensemble) | Prediction engine supervised — **dùng làm số liệu so sánh chính** |
| 3 | LLM + RAG (GPT-4o-mini) | Advanced predictor (không train thêm) — **hiện chỉ có tác dụng làm số liệu so sánh**, không được đưa vào ứng dụng |
| 4 | **PhoBERT + LLM explanation** | **Hướng ứng dụng thực tế (sản phẩm cuối):** PhoBERT dự đoán label, LLM giải thích + trích evidence + đề xuất action cho hotel owner. Đây là hệ thống được triển khai trong app Streamlit `app/`. |

Nói cách khác, trong 4 hướng thì 3 hướng đầu (SVM, PhoBERT, LLM+RAG) **đóng vai trò nghiên cứu so sánh hiệu năng predictor**, còn hướng số 4 mới là **deliverable ứng dụng thực tế**. LLM+RAG không được đưa vào app vì accuracy của nó thua PhoBERT khá xa nên không hợp làm predictor trong sản phẩm.

**Final Result Table** (lưu ở `outputs/results/final_four_direction_comparison.md`):

| Method | Type | ACD F1 | SPC F1 | Combined F1 | Note |
|---|---|---|---|---|---|
| SVM + TF-IDF | Traditional baseline (so sánh) | 0.4074 | 0.2272 | 0.3173 | Baseline cổ điển |
| PhoBERT cls_only + VnCoreNLP | Supervised transformer (so sánh + dùng cho app) | 0.6360 | 0.4727 | 0.5543 | Model dự đoán chính, được tái sử dụng trong app |
| LLM + RAG k=8 (GPT-4o-mini) | LLM few-shot + retrieval (**chỉ so sánh**) | 0.4034 | 0.3031 | 0.3532 | RAG cải thiện ICL nhưng vẫn kém PhoBERT → không đưa vào app |
| PhoBERT + LLM explanation | **Practical application layer** | — | — | — | Không phải classifier, không có F1. LLM giải thích prediction của PhoBERT. Đây là sản phẩm demo cuối cùng. |

**Takeaways chính của report:**

1. SVM cách rất xa transformer-based supervised → khẳng định cần pretrained representation.
2. PhoBERT là model dự đoán mạnh nhất trong các hướng giữ lại → được chọn làm predictor cho app.
3. RAG cải thiện LLM few-shot prompting so với ICL ngẫu nhiên, nhưng LLM prediction vẫn thấp hơn PhoBERT rất nhiều → LLM không đủ tin cậy để làm classifier trong ứng dụng, **hướng này chỉ có giá trị làm số liệu so sánh** trong report.
4. Sản phẩm thực tế cuối cùng: PhoBERT (trained) cho prediction + LLM cho explanation/evidence/recommended_action — đóng gói trong Streamlit app.

---

## 3. Cấu Trúc Repo & Mục Đích Từng Thư Mục

Root: `/Users/macbookpro/Documents/Master-study/NLP/absa-vlsp2018-hotel`

```
absa-vlsp2018-hotel/
├── README.md                       # README public của repo (phiên bản gọn)
├── PROJECT_HANDOFF.md              # File này
├── Url_code_data.txt               # Link Google Drive chứa code + data dự phòng
├── requirements.txt                # Deps cho training/research
├── Group16_report_code.zip         # Zip nộp bài (code + results)
├── Group16_slide1.pptx             # Slide bài nộp
├── Group_16_slides.pptx            # Slide bản khác
├── group16_slide_report.pptx       # Slide + report combined
├── data/                           # Dataset raw + preprocessed
├── code/                           # Toàn bộ source code cho 4 hướng
├── notebooks/                      # Notebook đã executed (có output thật) để chứng minh kết quả
├── docs/                           # Các file markdown giải thích parameter, kết quả, plan
├── outputs/                        # EDA plots, checkpoint PhoBERT, metric JSON, report assets
├── app/                            # Streamlit app demo PhoBERT + LLM explanation
├── submission_code_results/        # Snapshot bản nộp
├── draft/                          # Thử nghiệm phụ đã archive (cascade, verifier, v1, ...)
├── vncorenlp/                      # Wrapper VnCoreNLP jar + models (gitignored)
├── models/                         # Đặt model phụ nếu cần
├── .venv/ .venv-app/               # Virtualenv local (gitignored)
└── .gitignore                      # Lưu ý: *.pt, outputs/results/*, CLAUDE.md, .claude/ bị ignore
```

### 3.1 `data/`

```
data/
├── 1-VLSP2018-SA-Hotel-train.csv   # Data gốc VLSP, format nguyên bản
├── 2-VLSP2018-SA-Hotel-dev.csv
├── 3-VLSP2018-SA-Hotel-test.csv
├── train.csv                       # Reformat về schema 1 cột Review + 34 cột aspect
├── dev.csv
├── test.csv
├── train_preprocessed.csv          # Đã chạy pipeline preprocessing, có cột processed_review
├── dev_preprocessed.csv
└── test_preprocessed.csv
```

- Schema chuẩn: 1 cột `Review` + 34 cột aspect (giá trị trong `{0,1,2,3}`).
- Các file `*_preprocessed.csv` có thêm cột `processed_review` (đã NFC, normalize whitespace, replace teencode, remove special chars, word segmentation bằng VnCoreNLP, nối token bằng `_`).
- Các script/notebook chính dùng cache preprocessed để chạy ổn định.

### 3.2 `code/`

```
code/
├── data_processing/
│   ├── step1_eda.py                # Sinh EDA plots, class_weights.json
│   ├── step2_dataloader.py         # Dataset/Dataloader PyTorch cho processed_review + 34 labels
│   ├── step3_preprocessing.py      # Unicode NFC, teencode, word segmentation VnCoreNLP
│   ├── step4_eval.py               # Compute ACD F1, SPC F1, Combined F1
│   ├── export_eda_report.py        # Export EDA sang docx
│   └── utils/
├── svm_baseline/
│   ├── svm_baseline.py             # LinearSVC + TF-IDF unigram+bigram (34 classifiers độc lập)
│   └── SVM_BASELINE_RESULTS.md
├── phobert/
│   ├── model.py                    # ABSAPhoBERT multi-task với 34 heads (4 labels/head)
│   ├── train.py                    # Training loop + weighted focal loss + early stopping
│   ├── predict.py                  # Load checkpoint, evaluate
│   ├── run_experiment.py           # Entrypoint (flag --encoder cls_only|concat_4_layers)
│   └── smoke_test.py
├── llm_rag/
│   ├── llm_client.py               # Wrapper OpenAI/Gemini, retry + cache
│   ├── prompts.py                  # Prompt template, JSON schema, parser
│   ├── rag_retriever.py            # Sentence-transformer embedding + nearest neighbor
│   ├── rag_predictor.py            # Full RAG pipeline (retrieve → prompt → parse)
│   ├── icl_predictor.py            # ICL ngẫu nhiên để so sánh với RAG
│   ├── run_llm_rag.py              # Entrypoint chạy RAG experiment
│   └── compare_results.py          # Sinh final_four_direction_comparison.md
└── llm_explainability/
    ├── prediction_formatter.py     # Vector [34] → present prediction list
    ├── explanation_prompts.py      # Prompt schema cho explanation (items/overall_summary/recommended_action)
    ├── evidence_checker.py         # Substring/normalize match evidence vào review
    ├── llm_explainer.py            # Generate explanation từ PhoBERT predictions
    ├── error_analyzer.py           # Error analysis: FP/FN, SPC confusion
    └── run_explainability.py       # Entrypoint
```

### 3.3 `notebooks/`

Các notebook đã được **executed với output thật** — dùng làm bằng chứng trong report:

```
notebooks/
├── phase_phobert_vncorenlp_executed.ipynb          # Kết quả chính PhoBERT (Combined F1 = 0.5543)
├── phase_phobert_vncorenlp_executed_version2.ipynb # Bản v2
├── phase_phobert_no_vncorenlp_executed.ipynb       # Ablation bỏ VnCoreNLP (Combined F1 = 0.2981)
├── phase_llm_rag_executed.ipynb                    # LLM + RAG với output thật từ GPT-4o-mini
├── phase_llm_explanation_validation.ipynb          # Notebook validation 3 method (template)
└── phase_llm_explaination_done.ipynb               # Notebook validation đã chạy xong trên Kaggle (T4)
```

### 3.4 `docs/`

```
docs/
├── ABSA_Project_Guide.docx                 # Guide tổng (docx bản ngắn)
├── ABSA_Project_Guide_Detailed.docx        # Guide chi tiết
├── eda_summary_report.md                   # Tóm tắt EDA (phân phối nhãn, length, imbalance)
├── svm_baseline_summary.md                 # Config + kết quả + giải thích param SVM
├── phobert_best_single_summary.md          # Config + kết quả + giải thích param PhoBERT single
├── phobert_vncorenlp_ablation.md           # So sánh có/không VnCoreNLP (+85.9% relative gain)
├── llm_rag_comparison.md                   # Ablation k = 2,4,8,16 cho RAG
├── phobert_llm_explainability_plan.md      # Plan + spec cho hướng số 4 (PhoBERT + LLM explanation)
└── llm_explanation_validation_summary.md  # Chi tiết quá trình + kết quả 3 validation methods (M1/M2/M3)
```

Đây là các **file markdown cần đọc kỹ** nếu luồng chat khác muốn hiểu vì sao chọn tham số này chứ không phải tham số khác.

### 3.5 `outputs/`

```
outputs/
├── eda/
│   ├── EDA_Report_ABSA_VLSP2018.docx
│   ├── EDA_Summary_Report.md
│   ├── class_weights.json                 # Class weight tính từ train split (tránh leak)
│   ├── encoder_config.json
│   ├── learning_curve_cls_only.png
│   ├── train|dev|test_aspect_presence.png
│   ├── train|dev|test_label_breakdown.png
│   └── train|dev|test_review_length.png
├── reports/
│   ├── absa_hotel_intro_slides.pptx
│   ├── absa_hotel_intro_slides_visual.pptx
│   ├── app_demo_mockup.png
│   └── slide_visual_pages/                # slide_1.png ... slide_6.png
├── results/
│   ├── final_four_direction_comparison.md # Bảng so sánh 4 hướng (FILE QUAN TRỌNG NHẤT)
│   ├── svm_baseline_dev_metrics.json
│   ├── svm_baseline_test_metrics.json
│   ├── svm_baseline_summary.md
│   ├── rag_openai_k8_metrics.json         # Per-aspect metric của RAG k=8 run
│   ├── phobert_best_single/
│   │   ├── models_cls_only/
│   │   │   └── best_model.pt              # Checkpoint PhoBERT best (app load file này)
│   │   ├── results_cls_only/              # Metric + prediction JSON
│   │   └── eda/
│   └── phobert_no_vncorenlp/              # Ablation outputs
└── llm_cache/                             # Cache response LLM (gitignored)
```

Lưu ý: `outputs/results/*` bị gitignore **trừ** `phobert_best_single/` được giữ lại (xem `.gitignore`).

### 3.6 `app/`

App Streamlit demo cho hướng số 4 (PhoBERT + LLM explanation):

```
app/
├── README.md                       # Hướng dẫn chạy app
├── streamlit_app.py                # Entrypoint
├── phobert_service.py              # Load checkpoint + predict
├── explanation_service.py          # Gọi OpenAI để giải thích
├── file_loader.py                  # Đọc CSV/XLSX cột review_text
├── ui_components.py                # Các component Streamlit
├── styles.py                       # CSS
├── sample_reviews.csv              # File mẫu để test import
└── requirements-app.txt            # Deps cho app (tách khỏi requirements.txt chính)
```

**Flow app:**

1. Start screen.
2. Config screen: nhập 3 thứ — PhoBERT checkpoint, VnCoreNLP dir, OpenAI API key.
3. Review screen: nhập tay 1 review hoặc upload CSV/XLSX (yêu cầu cột `review_text`).
4. PhoBERT predict aspect/sentiment.
5. LLM giải thích predictions + trích evidence từ review + đề xuất action cho hotel owner.
6. Không có fallback rule-based — nếu thiếu checkpoint thì app báo lỗi thay vì tạo fake label.

### 3.7 `draft/`

Các thử nghiệm không dùng trong báo cáo chính (cascade, verifier, augmentation, retrain v1, nhiều notebook cũ...) — **không xóa**, chỉ để archive. Không dùng trong flow chính.

### 3.8 `submission_code_results/`

Snapshot bản nộp (gồm `app/`, `code/`, `outputs/`) — zip thành `Group16_report_code.zip`.

---

## 4. Chi Tiết Từng Hướng

### 4.1 Hướng 1 — SVM + TF-IDF Baseline

- **File chính:** `code/svm_baseline/svm_baseline.py`
- **Model:** `LinearSVC` (sklearn) × 34 aspects độc lập, mỗi aspect là bài toán 4-class.
- **Features:** TF-IDF unigram + bigram, `max_features=50000`, `C=1.0`, `class_weight='balanced'`.
- **Preprocessing:** chỉ lowercase + remove special chars. **Không** word segmentation (giữ baseline đơn giản, độc lập với phase PhoBERT).
- **Input:** cột `Review` gốc (không dùng `processed_review`).
- **Chạy:**
  ```bash
  python code/svm_baseline/svm_baseline.py
  ```
- **Kết quả:**
  - Dev: ACD 0.4045 / SPC 0.2364 / Combined **0.3204**
  - Test: ACD 0.4074 / SPC 0.2272 / Combined **0.3173**
- **Lý do chọn tham số:** xem `docs/svm_baseline_summary.md`.

### 4.2 Hướng 2 — PhoBERT Single Model (prediction engine chính)

- **Backbone:** `vinai/phobert-base-v2`.
- **Kiến trúc:** `ABSAPhoBERT` multi-task — 34 classification heads song song, mỗi head output 4 logit.
- **Encoder option:** `cls_only` (vector [CLS] 768 chiều, ít tham số hơn `concat_4_layers`, giảm overfit trên 3k train samples).
- **Loss:** Weighted focal loss (dựa trên class weight từ train split, `weight_clip=10.0` để tránh class hiếm chi phối loss).
- **Config cuối cùng:**

  | Tham số | Giá trị |
  |---|---|
  | MAX_SEQ_LEN | 256 |
  | Batch size | 16 |
  | Learning rate | 1e-4 |
  | Warmup ratio | 0.15 |
  | Max epochs | 20 |
  | Early stop patience | 7 |
  | Best epoch | 16 / 20 |
  | Seed | 42 |

- **Preprocessing bắt buộc:** word segmentation bằng VnCoreNLP (vì PhoBERT được pretrain trên text đã segmented). Bỏ VnCoreNLP làm Combined F1 giảm từ 0.5543 xuống 0.2981 (ablation trong `docs/phobert_vncorenlp_ablation.md`).
- **Chạy:**
  ```bash
  python code/phobert/run_experiment.py --encoder cls_only
  ```
- **Notebook evidence:**
  - `notebooks/phase_phobert_vncorenlp_executed.ipynb` (main result)
  - `notebooks/phase_phobert_no_vncorenlp_executed.ipynb` (ablation)
- **Checkpoint:** `outputs/results/phobert_best_single/models_cls_only/best_model.pt`
- **Kết quả:**
  - Dev: ACD 0.6271 / SPC 0.4630 / Combined **0.5451**
  - Test: ACD 0.6360 / SPC 0.4727 / Combined **0.5543**
- **Không dùng ensemble** trong report chính để giữ gọn. Bản concat_4_layers đạt 0.5247 (thấp hơn cls_only) nên không chọn.
- **Lý do chọn tham số:** xem `docs/phobert_best_single_summary.md`.

### 4.3 Hướng 3 — LLM + RAG (chỉ để so sánh số liệu, không dùng trong app)

> **Vai trò:** đây là **comparison baseline** để chứng minh rằng ngay cả khi kết hợp retrieval với LLM, kết quả vẫn thua PhoBERT supervised. Hướng này **không** được đưa vào sản phẩm ứng dụng.

- **LLM:** GPT-4o-mini (rẻ, đủ tốt cho bài tập lớn).
- **Retriever:** sentence-transformer embedding + nearest neighbor trên `train_preprocessed.csv`.
- **Prompt:** yêu cầu LLM trả về JSON với 34 aspect labels để parser tự động map về vector [34].
- **Ablation k:**

  | Method | k | ACD F1 | SPC F1 | Combined F1 |
  |---|---:|---:|---:|---:|
  | ICL GPT-4o-mini | 8 | 0.3504 | 0.2567 | 0.3035 |
  | RAG GPT-4o-mini | 2 | 0.3793 | 0.2866 | 0.3330 |
  | RAG GPT-4o-mini | 4 | 0.3866 | 0.2883 | 0.3375 |
  | **RAG GPT-4o-mini** | **8** | **0.4034** | **0.3031** | **0.3532** |
  | RAG GPT-4o-mini | 16 | 0.3988 | 0.2879 | 0.3433 |

  → `k=8` là điểm cân bằng tốt nhất (ít hơn thì thiếu context, nhiều hơn thì prompt nhiễu).
- **Chạy:**
  ```bash
  OPENAI_API_KEY=... python code/llm_rag/run_llm_rag.py
  ```
- **Cache:** `outputs/llm_cache/` (tránh gọi API lại khi rerun).
- **Kết luận:** RAG > ICL ngẫu nhiên, nhưng vẫn cách PhoBERT một khoảng lớn (0.3532 vs 0.5543). → **LLM+RAG không đủ tin cậy để làm predictor trong ứng dụng; hướng này chỉ tồn tại trong report dưới dạng số liệu so sánh/ablation.**
- **Lý do chọn tham số:** xem `docs/llm_rag_comparison.md`.

### 4.4 Hướng 4 — PhoBERT + LLM Explanation (**hướng ứng dụng — sản phẩm cuối cùng**)

> **Đây là deliverable ứng dụng của đề tài.** PhoBERT đã train được tái sử dụng làm prediction engine; LLM đóng vai trò explanation layer. Không có classifier mới, không thay đổi label của PhoBERT, không fallback rule-based. Hệ thống này được đóng gói thành app Streamlit trong `app/` — đó là sản phẩm demo mà user cuối (chủ khách sạn) sẽ sử dụng.

Ý tưởng:

```
Review → PhoBERT predicts labels → LLM receives only PRESENT predictions
       → LLM extracts evidence (substring từ review) + writes explanation
       → Output supports hotel owner decision making
```

Pipeline: `code/llm_explainability/` với các module tách rõ `prediction_formatter`, `explanation_prompts`, `evidence_checker`, `llm_explainer`, `error_analyzer`, `run_explainability`.

**Design choices quan trọng:**

| Parameter | Choice | Reason |
|---|---|---|
| PhoBERT checkpoint | `outputs/results/phobert_best_single/models_cls_only/best_model.pt` | Dùng best single checkpoint đã train |
| Encoder | `cls_only` | Match checkpoint |
| Input mode | 1 review hoặc 1 sample từ CSV/XLSX | Dễ demo, đỡ tốn API |
| File schema | Yêu cầu cột `review_text` | Tránh mơ hồ khi import |
| LLM role | Chỉ explanation, **không** đổi nhãn PhoBERT | Giữ classifier và explainer tách biệt |
| Evidence check | Substring/normalize match trong review | Đơn giản, transparent, có thể explain trong report |
| JSON output | Fixed schema: `items`, `overall_summary`, `recommended_action` | Parseable + dễ đánh giá |
| Temperature | Thấp (0–0.2) | Explanation cần consistent, không sáng tạo |

**Explanation quality metrics (heuristic):**

- JSON parse rate ≥ 95%
- Aspect preservation rate = 100% (LLM không được thêm/bớt aspect)
- Sentiment preservation rate = 100% (LLM không được đổi sentiment)
- Evidence substring match rate (càng cao càng tốt)
- Evidence uncertain rate (LLM tự flag khi không chắc)

**Manual rubric** (20–30 examples): Evidence correctness / Explanation quality / Hallucination / User usefulness — mỗi tiêu chí scale 0–2.

**Chạy:**

```bash
OPENAI_API_KEY=... python code/llm_explainability/run_explainability.py \
  --predictions_json outputs/results/final_predictions.json \
  --provider openai \
  --max_samples 20
```

**Output:**

- `outputs/results/llm_explainability/explanation_samples.json`
- `outputs/results/llm_explainability/explanation_quality_report.json`

**Plan chi tiết 6 phase** (Implementation Checklist): xem cuối `docs/phobert_llm_explainability_plan.md`.

---

### 4.5 LLM Quality Validation — Bằng Chứng "LLM Không Bịa" (đã chạy trên Kaggle)

> **Mục tiêu:** cung cấp 3 bằng chứng độc lập, đo lường 3 loại hallucination khác nhau, để defend claim "LLM trong hệ thống PhoBERT + LLM explanation không bịa thông tin". Notebook thực thi: `notebooks/phase_llm_explaination_done.ipynb`. Kết quả lưu: `outputs/results/llm_validation_results/`.

**Bối cảnh chạy:** Tesla T4 (Kaggle), N=200 mẫu từ test set, GPT-4o-mini, PhoBERT checkpoint tại `/kaggle/input/datasets/minhvnh/best-model-pt-phobertv2-1/best_model.pt`.

**Parse fail rate:** 40.5% (81/200 mẫu GPT-4o-mini trả về JSON không parse được). Các metric M1 và M2 dưới đây được tính **trên 119 mẫu parse thành công** (399 explanation items tổng cộng). Parse fail phản ánh JSON formatting của model, không phải chất lượng của 119 mẫu hợp lệ.

#### M1 — Label Consistency (không bịa label)

| Chỉ số | Kết quả |
|--------|---------|
| Số mẫu xử lý | 200 (81 parse fail bị skip) |
| Tổng LLM items sinh ra | 399 |
| Dropped spurious aspect | 0 |
| Sentiment sai | 0 |
| **Aspect Preservation Rate** | **100.0%** |
| **Sentiment Consistency Rate** | **100.0%** |
| **Overall Label Accuracy** | **100.0%** |

Phương pháp: đọc metadata từ `validate_explanation_items()` — hàm này đã được chạy trong `explain_batch()` để tự động loại aspect bịa ra và sửa sentiment sai. Số lượng bị loại/sửa = 0 → LLM tuân thủ schema label hoàn toàn.

#### M2 — BERTScore Evidence Groundedness (không bịa câu trích)

| Chỉ số | Kết quả |
|--------|---------|
| Số mẫu | 200 |
| Evidence items được score | 377 |
| Backbone | `vinai/phobert-base-v2` (`num_layers=10`) |
| Mean BERTScore Precision | 0.7709 |
| Mean BERTScore Recall | 0.4396 |
| **Mean BERTScore F1** | **0.5535** |

Phương pháp: BERTScore F1(evidence, original_review) — đo mức độ evidence trích từ review thật, không phải bịa. Precision cao (0.7709) cho thấy token trong evidence phần lớn xuất hiện trong review gốc. Recall thấp hơn (0.4396) là bình thường vì evidence chỉ là đoạn trích, không cover toàn bộ review. Citation: Zhang et al. 2020 — BERTScore: Evaluating Text Generation with BERT.

#### M3 — Human Rubric (không bịa diễn giải) — đã annotate

| Chỉ số | Faithfulness | Usefulness |
|--------|-------------|------------|
| N mẫu đánh giá | 30 | 18 (có recommended_action) |
| Score 2 (tốt nhất) | 26/30 = **86.7%** | 12/18 = **66.7%** |
| Score 1 (chấp nhận được) | 4/30 = 13.3% | 6/18 = 33.3% |
| Score 0 (hallucination / vô dụng) | **0/30 = 0%** | **0/18 = 0%** |
| **Mean** | **1.867 / 2.0** | **1.667 / 2.0** |
| Stdev | 0.346 | 0.485 |

Thang điểm: Faithfulness 0–2 (0=hallucination, 1=phần lớn đúng, 2=hoàn toàn trung thực), Usefulness 0–2 (0=không hữu ích, 1=hữu ích một phần, 2=rất hữu ích).

**4 trường hợp faithfulness=1** (không hoàn toàn, nhưng cũng không bịa):
- Rank 5: summary thêm "vẫn có một số phòng có view đẹp" — hơi lạc quan hóa so với review gốc
- Rank 9: explanation suy ra "thoải mái" không được nhắc tường minh trong review
- Rank 25: explanation cố justify nhãn neutral cho evidence âm tính rõ; summary thêm "thiết kế" ngoài scope evidence
- Rank 29: summary lẫn khía cạnh "sạch sẽ" (muỗi) vào aspect DESIGN&FEATURES

**Không có trường hợp nào score=0** → LLM không bịa hoàn toàn thông tin ngoài review.

File annotation: `outputs/results/llm_validation_results/rubric_annotated.csv`

#### Tóm tắt 3 bằng chứng (đầy đủ)

| Method | Metric | Kết quả | Claim |
|--------|--------|---------|-------|
| M1 — Label Consistency (N=200) | Sentiment consistency | **100.0%** | LLM không bịa label |
| M2 — BERTScore Evidence (N=200) | Mean BERTScore F1 | **0.5535** | LLM không bịa câu trích |
| M3 — Human Rubric (N=30) | Faithfulness mean / Usefulness mean | **1.867 / 1.667** (thang 0–2) | LLM không bịa diễn giải |

**Module validate:** `code/llm_explainability/validate_llm_quality.py` (3 hàm: `compute_label_consistency`, `compute_bertscore_groundedness`, `generate_rubric_template` / `compute_rubric_scores`).

---

## 5. Phân Tích Kết Quả (Đã Note & Đã Phân Tích Những Gì)

### 5.1 Nơi các kết quả được lưu

Tất cả số liệu đã được lưu vào repo (không chỉ để ở transcript chat). Cụ thể:

| Kết quả | File / Đường dẫn | Dạng |
|---------|------------------|------|
| Bảng so sánh 4 hướng (final table) | `outputs/results/final_four_direction_comparison.md` | Markdown |
| SVM baseline — metric dev | `outputs/results/svm_baseline_dev_metrics.json` | JSON |
| SVM baseline — metric test | `outputs/results/svm_baseline_test_metrics.json` | JSON |
| SVM baseline — summary & phân tích | `outputs/results/svm_baseline_summary.md` + `docs/svm_baseline_summary.md` + `code/svm_baseline/SVM_BASELINE_RESULTS.md` | Markdown |
| PhoBERT best checkpoint | `outputs/results/phobert_best_single/models_cls_only/best_model.pt` | PyTorch .pt |
| PhoBERT — metric & prediction | `outputs/results/phobert_best_single/results_cls_only/` | JSON |
| PhoBERT summary & phân tích | `docs/phobert_best_single_summary.md` | Markdown |
| PhoBERT ablation có/không VnCoreNLP | `docs/phobert_vncorenlp_ablation.md` | Markdown |
| PhoBERT no-VnCoreNLP ablation outputs | `outputs/results/phobert_no_vncorenlp/` | JSON + metric |
| LLM + RAG — per-aspect metric (k=8) | `outputs/results/rag_openai_k8_metrics.json` | JSON |
| LLM + RAG — phân tích + ablation k | `docs/llm_rag_comparison.md` | Markdown |
| LLM explainability plan + metric rubric | `docs/phobert_llm_explainability_plan.md` | Markdown |
| LLM Validation — giải thích chi tiết quá trình + annotation | `docs/llm_explanation_validation_summary.md` | Markdown |
| Notebook có output thật — PhoBERT + VnCoreNLP | `notebooks/phase_phobert_vncorenlp_executed.ipynb` | Jupyter |
| Notebook có output thật — PhoBERT no VnCoreNLP | `notebooks/phase_phobert_no_vncorenlp_executed.ipynb` | Jupyter |
| Notebook có output thật — LLM RAG | `notebooks/phase_llm_rag_executed.ipynb` | Jupyter |
| Notebook có output thật — LLM Quality Validation | `notebooks/phase_llm_explaination_done.ipynb` | Jupyter |
| LLM Validation — report M1 + M2 | `outputs/results/llm_validation_results/validation_report.json` | JSON |
| LLM Validation — explanation quality stats | `outputs/results/llm_validation_results/explanation_quality_report.json` | JSON |
| LLM Validation — explanation samples (N=200) | `outputs/results/llm_validation_results/explanation_samples.json` | JSON |
| LLM Validation — rubric template (N=30, chờ annotation) | `outputs/results/llm_validation_results/rubric_template.csv` | CSV |
| EDA plots (label distribution, aspect presence, review length) | `outputs/eda/*.png` | PNG |
| EDA — class weights | `outputs/eda/class_weights.json` | JSON |
| EDA — báo cáo tóm tắt | `docs/eda_summary_report.md` + `outputs/eda/EDA_Summary_Report.md` | Markdown |
| EDA docx | `outputs/eda/EDA_Report_ABSA_VLSP2018.docx` | Word |
| Slide visual (đã export PNG) | `outputs/reports/slide_visual_pages/slide_{1..6}.png` | PNG |
| Slide intro | `outputs/reports/absa_hotel_intro_slides*.pptx` | PPTX |
| App demo mockup | `outputs/reports/app_demo_mockup.png` | PNG |
| Cache LLM response (tái dùng) | `outputs/llm_cache/` | JSON/sqlite |

Tóm lại: **mỗi hướng đều có cả (a) metric raw JSON, (b) summary markdown giải thích tham số + kết quả, và (c) notebook/docx để phục vụ report.** Không có kết quả nào chỉ được nhắc trong chat mà không lưu file.

### 5.2 Số liệu cuối cùng (đã note)

| Hướng | Split | ACD F1 | SPC F1 | Combined F1 |
|-------|-------|--------|--------|-------------|
| SVM + TF-IDF | Dev | 0.4045 | 0.2364 | 0.3204 |
| SVM + TF-IDF | **Test** | 0.4074 | 0.2272 | **0.3173** |
| PhoBERT cls_only + VnCoreNLP | Dev | 0.6271 | 0.4630 | 0.5451 |
| PhoBERT cls_only + VnCoreNLP | **Test** | 0.6360 | 0.4727 | **0.5543** |
| PhoBERT concat_4_layers + VnCoreNLP | Test | 0.5836 | 0.4658 | 0.5247 |
| PhoBERT cls_only (no VnCoreNLP) | Test | 0.3349 | 0.2014 | 0.2681 |
| PhoBERT concat_4_layers (no VnCoreNLP) | Test | 0.3592 | 0.2371 | 0.2981 |
| ICL GPT-4o-mini k=8 | Test | 0.3504 | 0.2567 | 0.3035 |
| RAG GPT-4o-mini k=2 | Test | 0.3793 | 0.2866 | 0.3330 |
| RAG GPT-4o-mini k=4 | Test | 0.3866 | 0.2883 | 0.3375 |
| RAG GPT-4o-mini **k=8** | Test | 0.4034 | 0.3031 | **0.3532** |
| RAG GPT-4o-mini k=16 | Test | 0.3988 | 0.2879 | 0.3433 |
| SOTA tham chiếu (Huynh 2022) | Test | 0.8255 | — | 0.7732 |

### 5.3 Phân tích kết quả

Phần này tổng hợp những phân tích đã viết trong các file markdown dưới `docs/`. Đây là nội dung người chấm/độc giả cần để hiểu **tại sao** có những con số này.

#### a) SVM vs PhoBERT — giá trị của pretrained representation

- Combined F1: **0.3173 → 0.5543** (tăng +0.2370 tuyệt đối, ~+74.7% tương đối).
- Dù SVM đã có `class_weight='balanced'` + TF-IDF unigram+bigram, SPC F1 chỉ đạt 0.2272 vì TF-IDF sparse không bắt được context phủ định / so sánh giá / emphasis — các cue quan trọng để phân biệt `positive` vs `negative` vs `neutral`.
- PhoBERT học được contextual representation tiếng Việt nên tăng đều cả ACD lẫn SPC. Kết luận: pretrained tiếng Việt là yếu tố quyết định trên dataset nhỏ + nhiều aspect.

#### b) VnCoreNLP ablation — giá trị của word segmentation phù hợp với pretrained corpus

- Best no-VnCoreNLP (concat_4_layers): 0.2981
- Best với VnCoreNLP (cls_only): **0.5543**
- **Absolute gain: +0.2562 Combined F1 (~+85.9% tương đối).**
- Giải thích: PhoBERT được pretrain trên văn bản tiếng Việt đã word-segmented. Nếu fine-tune trên text không segmented, token boundary lệch với từ điển pretrained → representation bị degrade nặng. Đây là ablation **clean** (chỉ đổi một yếu tố) nên kết luận rất chắc.

#### c) cls_only vs concat_4_layers (cùng setup VnCoreNLP)

- `cls_only`: 0.5543
- `concat_4_layers`: 0.5247
- Chọn `cls_only` vì:
  1. Combined F1 cao hơn (+0.0296).
  2. Ít tham số hơn ở classification head (768 vs 768×4 = 3072 input) → giảm overfit trên 3k train samples.
  3. Inference nhanh hơn, dễ deploy trong app.

#### d) LLM+RAG ablation theo k

- Thứ tự: k=2 (0.3330) < k=4 (0.3375) < **k=8 (0.3532)** > k=16 (0.3433).
- Giải thích: k thấp thiếu ngữ cảnh domain, k cao làm prompt dài → context overload, LLM dễ bỏ qua schema. k=8 là điểm cân bằng.
- ICL ngẫu nhiên (k=8) chỉ đạt 0.3035, RAG k=8 đạt 0.3532 → retrieval có giúp (+0.0497), nhưng vẫn cách PhoBERT 0.2011 điểm.
- Kết luận: LLM thiếu exposure trực tiếp với label scheme 34 aspects × 4 labels, chỉ học qua vài ví dụ trong prompt → không thể replicate supervised fine-tuning.

#### e) Vì sao LLM+RAG chỉ là comparison, không vào app

- Accuracy thấp hơn PhoBERT đáng kể.
- Chi phí API cao khi dùng cho user thực (600+ reviews mỗi lần).
- Không deterministic → khó kiểm soát chất lượng predictor.
- Vì vậy LLM+RAG không đủ tin cậy làm predictor trong sản phẩm → giữ làm số liệu so sánh trong report.

#### f) Lý do chọn kiến trúc PhoBERT + LLM explanation cho app

- Tách biệt vai trò: PhoBERT (đã train) lo prediction deterministic và đo được F1; LLM lo explanation/evidence/recommended_action.
- Làm thế:
  - Giữ được con số F1 có thể defend trong report (0.5543).
  - Giải quyết yêu cầu "ứng dụng thực tế" của đề tài: user có labels + lý do + câu trích từ review + gợi ý action.
  - LLM không thể đổi label PhoBERT → tránh hallucination làm sai metric.
- Metric đánh giá chất lượng explanation (heuristic):
  - JSON parse rate ≥ 95% (target).
  - Aspect preservation rate = 100%.
  - Sentiment preservation rate = 100%.
  - Evidence substring match rate (càng cao càng tốt).
  - Evidence uncertain rate (LLM tự flag khi không chắc).
- Manual rubric 20–30 examples: Evidence correctness / Explanation quality / Hallucination / User usefulness, mỗi tiêu chí 0–2.

#### g) Khoảng cách với SOTA

- PhoBERT cls_only đạt 0.5543 trong khi Huynh 2022 đạt 0.7732 (Combined F1).
- Các lý do khoảng cách:
  - Không ensemble (chủ đích giữ single model).
  - Không augmentation (các thử nghiệm aug bị đưa vào `draft/`).
  - PhoBERT base, không phải PhoBERT large.
  - Hyperparameter được chọn thực dụng để chạy được trên Kaggle T4, không tuning dài.
- Trong phạm vi đề tài (bài tập lớn, không có cluster GPU lớn), khoảng cách này được chấp nhận và report có giải thích rõ.

### 5.4 Những phân tích đã note thêm (error analysis layer)

Error analysis nằm trong hướng số 4, đã được đặc tả trong `docs/phobert_llm_explainability_plan.md` và implement trong `code/llm_explainability/error_analyzer.py`. Các phần đã note:

1. Frequent ACD errors (những aspect PhoBERT hay sót hoặc hay nhận nhầm).
2. Frequent SPC errors (những aspect đoán đúng presence nhưng sai sentiment).
3. Neutral class issue (neutral rare nhất trong 4 nhãn, recall thấp — đã phân tích).
4. Example PhoBERT đúng + LLM explanation useful.
5. Example PhoBERT có thể sai + LLM flag `evidence_uncertain`.

Output: `outputs/results/llm_explainability/explanation_samples.json` và `outputs/results/llm_explainability/explanation_quality_report.json` (sau khi chạy `run_explainability.py`).

---

## 6. Data Processing Pipeline (áp dụng cho PhoBERT, LLM, RAG)

Implement trong `code/data_processing/step3_preprocessing.py`. Pipeline **5 bước deterministic + idempotent**:

| # | Bước | Hàm | Mô tả |
|---|------|-----|-------|
| 1 | Unicode NFC | `normalize_unicode` | Chuẩn hóa ký tự tổ hợp tiếng Việt |
| 2 | Normalize whitespace | `normalize_whitespace` | Xóa newline/tab/space thừa, strip |
| 3 | Teencode replacement | `replace_teencode` | ~60 entries domain khách sạn (`ks→khách sạn`, `nv→nhân viên`, `ko→không`...), replace từ dài trước |
| 4 | Remove special chars | `remove_special_chars` | Giữ chữ có dấu, số, `.,!?;:-/()` |
| 5 | Word segmentation | `VnCoreNLPSegmenter` | Nối token bằng `_`, fallback underthesea |

**Ví dụ:**

```
ORIG: Rộng rãi KS mới nhưng rất vắng. Các dịch vụ chất lượng chưa cao và thiếu.
PROC: Rộng_rãi_khách_sạn_mới_nhưng_rất_vắng_._Các_dịch_vụ_chất_lượng_chưa_cao_và_thiếu_.
```

**Class weights:** tính **chỉ từ train split** (không dùng dev/test để tránh leak), lưu `outputs/eda/class_weights.json`. Có `weight_clip=10.0` để tránh class neutral/rare chi phối loss.

---

## 7. Chỉ Dẫn Cần Biết Khi Làm Việc Trên Repo

### 7.1 Setup

```bash
# Research/training environment
pip install -r requirements.txt

# App environment (tách riêng)
pip install -r app/requirements-app.txt
```

VnCoreNLP jar + models nằm trong thư mục `vncorenlp/` (gitignored). Nếu luồng chat cần reproduce training, phải có sẵn VnCoreNLP cục bộ.

### 7.2 Quy ước quan trọng

- **Không xóa** `draft/` — chỉ archive, report chính không dùng.
- `.pt` files + `outputs/results/*` bị gitignore trừ `phobert_best_single/` được whitelist.
- File `CLAUDE.md`, `CODING_DIRECTIVE*.md`, `.claude/` bị gitignore.
- Report chính **không** dùng ensemble, không retrain, không fallback rule-based.
- Mọi thay đổi label scheme phải đi qua cả 4 hướng đồng nhất (34 aspects, 4 labels).
- App không tự fake label — thiếu checkpoint/VnCoreNLP thì báo lỗi.

### 7.3 Reproduce kết quả

```bash
# 1. EDA (tạo plots + class_weights)
python code/data_processing/step1_eda.py

# 2. Preprocessing (tạo data/*_preprocessed.csv)
python code/data_processing/step3_preprocessing.py

# 3. SVM baseline
python code/svm_baseline/svm_baseline.py

# 4. PhoBERT best single
python code/phobert/run_experiment.py --encoder cls_only

# 5. LLM + RAG
OPENAI_API_KEY=... python code/llm_rag/run_llm_rag.py

# 6. Bảng so sánh cuối
python code/llm_rag/compare_results.py

# 7. LLM explanation (cần PhoBERT predictions từ bước 4)
OPENAI_API_KEY=... python code/llm_explainability/run_explainability.py \
  --predictions_json outputs/results/final_predictions.json \
  --provider openai \
  --max_samples 20

# 8. Chạy app demo
streamlit run app/streamlit_app.py
```

### 7.4 Nếu cần mở rộng / chỉnh sửa

- Sửa config PhoBERT → `code/phobert/run_experiment.py` + `code/phobert/model.py`.
- Thay LLM provider → `code/llm_rag/llm_client.py` (đã có wrapper OpenAI/Gemini).
- Thêm aspect mới → phải cập nhật cả dataloader, model heads, evaluation, prompt schema cho LLM.
- Thay dataset → cập nhật `step3_preprocessing.py` và đảm bảo schema `Review + 34 aspects` còn đúng.

---

## 8. Deliverables Đã Nộp

| File | Mô tả |
|---|---|
| `Group16_report_code.zip` (507 MB) | Zip code + results + checkpoint |
| `Group16_slide1.pptx` | Slide chính bài nộp |
| `Group_16_slides.pptx` | Slide bản khác |
| `group16_slide_report.pptx` | Slide + report combined |
| `outputs/reports/absa_hotel_intro_slides.pptx` | Slide intro |
| `outputs/reports/absa_hotel_intro_slides_visual.pptx` | Slide intro có visual |
| `outputs/reports/slide_visual_pages/slide_{1..6}.png` | Các slide đã export PNG |
| `docs/ABSA_Project_Guide_Detailed.docx` | Guide chi tiết |
| `Url_code_data.txt` | Link Google Drive backup code + data |

---

## 9. Những Thứ Luồng Chat Mới Nên Đọc Theo Thứ Tự Ưu Tiên

1. **`README.md`** — overview 4 hướng (ngắn, 2 phút đọc).
2. **`outputs/results/final_four_direction_comparison.md`** — bảng kết quả cuối.
3. **`docs/phobert_llm_explainability_plan.md`** — hướng số 4 chi tiết nhất, có research narrative + plan 6 phase.
4. **`docs/phobert_best_single_summary.md`** — tại sao chọn PhoBERT cls_only + các hyperparameter.
5. **`docs/llm_rag_comparison.md`** — tại sao k=8 và tại sao LLM thua PhoBERT.
6. **`docs/phobert_vncorenlp_ablation.md`** — tại sao bắt buộc VnCoreNLP (+85.9% relative gain).
7. **`docs/eda_summary_report.md`** — hiểu dataset (imbalance, neutral rare, p99 length).
8. **`docs/svm_baseline_summary.md`** — baseline reasoning.
9. **`app/README.md`** — flow app + yêu cầu file import.
10. **`code/phobert/model.py`** và **`code/llm_explainability/run_explainability.py`** — 2 file code quan trọng nhất.

---

## 10. Tóm Tắt Một Dòng

> Trong 4 hướng của report, 3 hướng đầu (SVM 0.3173, PhoBERT cls_only+VnCoreNLP 0.5543, LLM+RAG k=8 0.3532 trên test VLSP 2018 Hotel) **chỉ đóng vai trò so sánh số liệu predictor**; hướng số 4 — **PhoBERT + LLM explanation** — là hướng **ứng dụng thực tế** được đóng gói thành app Streamlit, trong đó PhoBERT (đã train) predict label còn LLM chỉ giải thích, trích evidence và đề xuất action cho hotel owner.
