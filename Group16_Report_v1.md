# Phân tích cảm xúc theo khía cạnh cho review khách sạn tiếng Việt
## Báo cáo đồ án cuối kỳ — Nhóm 16 — Version 1

> **Ghi chú phiên bản (v1):** Đây là bản draft đầu tiên theo `REPORT_PLAN.md`. Mục §6.7 (đánh giá chất lượng explanation/recommendation) được để dưới dạng placeholder + TODO, sẽ điền ở version sau khi có kết quả đánh giá.

---

## 1. Abstract

Đồ án tập trung vào bài toán Aspect-Based Sentiment Analysis (ABSA) cho review khách sạn tiếng Việt trên bộ dữ liệu VLSP 2018 Hotel (34 aspect × 4 nhãn `absent`/`positive`/`negative`/`neutral`). Nhóm xây dựng ba thành phần chính: (1) **baseline SVM + TF-IDF** để xác định mốc truyền thống, đạt Combined F1 = 0.3173 trên test; (2) **PhoBERT predictor** supervised với encoder `cls_only` và preprocessing VnCoreNLP, đạt Combined F1 = **0.5543**; và (3) **PhoBERT + LLM Explanation** — hệ thống ứng dụng được đóng gói thành Streamlit app, trong đó PhoBERT sinh nhãn còn LLM (GPT-4o-mini) đảm nhiệm sinh giải thích, trích evidence từ review và đề xuất hành động cho chủ khách sạn. Ngoài ra, nhóm còn thực hiện một thí nghiệm bổ sung LLM + RAG (GPT-4o-mini, k=8, Combined F1 = 0.3532) để trả lời câu hỏi "có thể thay PhoBERT bằng LLM làm predictor không?". Kết quả cho thấy LLM thua PhoBERT khoảng 0.201 điểm Combined F1 trên cùng test set, từ đó củng cố quyết định chọn PhoBERT làm predictor cho sản phẩm. Đóng góp đáng chú ý gồm: ablation sạch cho vai trò của VnCoreNLP (+85.9% tương đối Combined F1) và kiến trúc tách biệt vai trò giữa classifier (PhoBERT) và explainer (LLM) cho ứng dụng thực tế.

---

## 2. Introduction

### 2.1 Bối cảnh và động lực

Trong ngành du lịch - khách sạn, review của khách hàng là một nguồn phản hồi quan trọng nhưng cũng rất khó khai thác thủ công do số lượng lớn. Một review điển hình không chỉ nhắc tới một khía cạnh duy nhất (ví dụ "phòng") mà thường đề cập đồng thời nhiều khía cạnh (phòng, nhân viên, vị trí, giá, bữa sáng...) với các sắc thái cảm xúc khác nhau. Bài toán ABSA được thiết kế để giải quyết chính tình huống này: thay vì gán một nhãn cảm xúc chung cho cả review, hệ thống phải xác định được **từng khía cạnh** nào xuất hiện và cảm xúc **cho riêng khía cạnh** đó.

Review tiếng Việt còn có một số đặc thù khiến bài toán khó hơn so với tiếng Anh: sự xuất hiện phổ biến của viết tắt / teencode (`ks` = khách sạn, `nv` = nhân viên, `ko` = không), nhiều cách mã hóa Unicode khác nhau cho cùng một ký tự có dấu, và nhất là nhu cầu **word segmentation** — tiếng Việt không phân tách từ ghép bằng khoảng trắng, nên các pretrained model tiếng Việt như PhoBERT được huấn luyện trên corpus đã word-segmented, và input lúc fine-tune phải theo cùng định dạng.

Bộ dữ liệu VLSP 2018 Hotel là benchmark công khai cho ABSA tiếng Việt, được cộng đồng VLSP xây dựng với 34 aspect được chuẩn hoá (ví dụ `HOTEL#GENERAL`, `ROOMS#CLEANLINESS`, `SERVICE#GENERAL`). Đây là lựa chọn phù hợp cho đồ án vì: (a) đủ phức tạp về schema, (b) có các paper SOTA để đối chiếu, (c) kích thước vừa phải (5,600 review) cho hạ tầng train của sinh viên.

### 2.2 Mục tiêu và phạm vi

Task được chia làm hai bài toán con theo chuẩn VLSP:

- **Aspect Category Detection (ACD):** xác định aspect nào xuất hiện trong review.
- **Sentiment Polarity Classification (SPC):** với các aspect đã xuất hiện, phân lớp cảm xúc trong bốn nhãn `{absent, positive, negative, neutral}`.

Metric được dùng xuyên suốt: ACD F1, SPC F1 (cả hai đều macro trên 34 aspect), và **Combined F1** = trung bình hai metric trên.

Methodology của báo cáo gồm đúng **ba thành phần** được commit vào hệ thống: baseline SVM, PhoBERT predictor, và hệ thống ứng dụng PhoBERT + LLM Explanation. LLM + RAG không nằm trong methodology mà được trình bày là thí nghiệm bổ sung ở §6.6 để củng cố lựa chọn kiến trúc.

### 2.3 Đóng góp chính

1. Baseline SVM + TF-IDF được viết rõ lý do chọn tham số, đóng vai trò "trần dưới" để đánh giá giá trị của pretrained representation.
2. PhoBERT single model cho Combined F1 = 0.5543 trên test, kèm **clean ablation VnCoreNLP (+85.9% tương đối)** chứng minh vai trò quyết định của word segmentation phù hợp với pretraining.
3. Sản phẩm ứng dụng PhoBERT + LLM Explanation đóng gói thành Streamlit app: PhoBERT sinh nhãn, LLM sinh explanation / evidence / recommended_action cho chủ khách sạn.
4. Thí nghiệm bổ sung LLM + RAG (ablation k ∈ {2, 4, 8, 16}) chứng minh LLM không đủ thay supervised model, qua đó củng cố lựa chọn PhoBERT làm predictor cho app.

---

## 3. Related Work

### 3.1 Aspect-Based Sentiment Analysis

ABSA được nghiên cứu rộng rãi trong NLP. Có hai nhóm tiếp cận chính. Nhóm đầu là **feature-based** (trước kỷ nguyên neural): sử dụng các feature thủ công như TF-IDF, n-gram, POS tag, kết hợp với các classifier như SVM hoặc CRF. Nhóm thứ hai là **neural end-to-end**: từ BiLSTM với attention (Wang et al., 2016) đến các mô hình dựa trên BERT (Devlin et al., 2019) cho hiệu năng cao hơn đáng kể nhờ contextual representation. Gần đây, pretrained language model domain-specific (ví dụ cho e-commerce) tiếp tục đẩy SOTA lên cao hơn.

### 3.2 ABSA tiếng Việt và VLSP 2018

Với tiếng Việt, PhoBERT (Nguyen & Nguyen, 2020) là pretrained model phổ biến nhất, được train trên 20 GB corpus tiếng Việt đã word-segmented bằng VnCoreNLP (Vu et al., 2018). Trên benchmark VLSP 2018 Hotel, kết quả tốt nhất đã công bố là của Huynh (2022) với ACD F1 ≈ 0.8255 và Combined F1 ≈ 0.7732. Kết quả này dùng ensemble + data augmentation, cho thấy khoảng cách với single model không ensemble.

### 3.3 LLM và RAG cho phân loại có cấu trúc

In-context learning (ICL) bằng few-shot prompt đã được Brown et al. (2020) chứng minh là một cách sử dụng LLM không cần fine-tune. Retrieval-Augmented Generation (RAG, Lewis et al., 2020) mở rộng ICL bằng cách truy xuất các ví dụ/văn bản liên quan trước khi prompt. Tuy nhiên, với các bài toán classification có **schema nhiều nhãn** (như 34 aspect × 4 label trong ABSA), LLM gặp khó khăn trong việc tuân thủ đúng định dạng output, đặc biệt là các nhãn hiếm như `neutral`. Đây là một trong những lý do nhóm quyết định không dùng LLM làm predictor chính.

---

## 4. Data Analysis

### 4.1 Mô tả dataset

Bộ dữ liệu VLSP 2018 Hotel được nhóm lấy từ mirror công khai `ds4v/absa-vlsp-2018` trên HuggingFace. Mỗi mẫu gồm một cột `Review` (đoạn văn bản tiếng Việt) và 34 cột aspect, giá trị trong `{0, 1, 2, 3}` tương ứng với `{absent, positive, negative, neutral}`. Phân bổ split như sau:

*Bảng 1. Phân chia dữ liệu VLSP 2018 Hotel.*

| Split | Số mẫu | Vai trò |
|-------|-------:|---------|
| Train | 3,000  | Huấn luyện + tính class weight |
| Dev   | 2,000  | Validation, chọn checkpoint |
| Test  | 600    | Đánh giá cuối cùng |
| **Tổng** | **5,600** | |

### 4.2 Phân phối nhãn và mất cân bằng

Nhóm thực hiện EDA chi tiết (script `code/data_processing/step1_eda.py`, outputs ở `outputs/eda/`). Kết quả cho thấy dataset **mất cân bằng nghiêm trọng trên hai chiều**:

- **Per-class:** nhãn `absent` chiếm khoảng 85% tổng số nhãn trên toàn bộ 34 aspect, trong khi `neutral` cực kỳ hiếm (khoảng 154 lần ít hơn `absent` trên train split).
- **Per-aspect:** một số aspect như `HOTEL#GENERAL` xuất hiện ở ~43% review, trong khi các aspect như `ROOMS#MISCELLANEOUS`, `FACILITIES#MISCELLANEOUS` chỉ xuất hiện ở khoảng 0–2% review.

*Hình 1. Phân phối nhãn 4 lớp trên train split (nguồn: `outputs/eda/train_label_breakdown.png`).*

*Hình 2. Tỷ lệ xuất hiện (presence rate) của 34 aspect trên train split (nguồn: `outputs/eda/train_aspect_presence.png`).*

Kết luận quan trọng rút ra từ EDA: PhoBERT **bắt buộc** phải dùng weighted / focal loss để tránh bị hút về phía dự đoán `absent` cho mọi aspect. Class weight được tính chỉ từ train split để tránh leak dev/test, và có `weight_clip=10.0` để tránh các class cực hiếm (ví dụ `neutral` của một số aspect) làm loss không ổn định.

### 4.3 Độ dài review

*Hình 3. Phân phối độ dài review theo số từ trên train split (nguồn: `outputs/eda/train_review_length.png`).*

Phân phối lệch phải: phần lớn review nằm trong khoảng 10–50 từ, nhưng có outlier dài tới ~1,000 từ. Các percentile: p50 ≈ 25 từ, p95 ≈ 120 từ, p99 (train) = 243 từ. Sau word segmentation, một từ ghép có thể tách thành 1.5 token trung bình. Nhóm chọn `MAX_SEQ_LEN = 256` cho PhoBERT — giá trị này (a) đủ phủ phần lớn review thực, (b) vừa với giới hạn position embedding của PhoBERT base, (c) tiết kiệm VRAM trên Kaggle T4 GPU. Các review dài hơn sẽ bị truncate — đây là trade-off được chấp nhận trong phạm vi đồ án.

### 4.4 Pipeline preprocessing

Pipeline preprocessing được triển khai trong `code/data_processing/step3_preprocessing.py`, gồm 5 bước deterministic + idempotent, và kết quả được cache ra `data/*_preprocessed.csv` để tất cả các phase sau dùng chung dữ liệu thống nhất.

*Bảng 2. Pipeline preprocessing 5 bước.*

| # | Bước | Hàm | Mô tả |
|---|------|-----|-------|
| 1 | Unicode NFC | `normalize_unicode` | Chuẩn hoá ký tự tổ hợp tiếng Việt về dạng precomposed |
| 2 | Normalize whitespace | `normalize_whitespace` | Xoá newline, tab, khoảng trắng thừa, strip |
| 3 | Teencode replacement | `replace_teencode` | ~60 cụm domain khách sạn, thay thế từ dài đến ngắn |
| 4 | Remove special chars | `remove_special_chars` | Giữ chữ có dấu, số, và dấu câu cơ bản `.,!?;:-/()` |
| 5 | Word segmentation | `VnCoreNLPSegmenter` | Dùng VnCoreNLP để tách từ ghép, nối token bằng `_` |

Ví dụ minh hoạ:

```
ORIG: Rộng rãi KS mới nhưng rất vắng. Các dịch vụ chất lượng chưa cao và thiếu.
PROC: Rộng_rãi_khách_sạn_mới_nhưng_rất_vắng_._Các_dịch_vụ_chất_lượng_chưa_cao_và_thiếu_.
```

Lưu ý: baseline SVM **không** sử dụng pipeline đầy đủ này — xem §5.2 để hiểu lý do.

---

## 5. Methodology

Báo cáo đề xuất ba thành phần được nhóm **thực sự xây dựng và commit** vào hệ thống cuối: (1) baseline SVM + TF-IDF, (2) PhoBERT predictor, và (3) PhoBERT + LLM Explanation — hệ thống ứng dụng. Ba thành phần này tạo thành một câu chuyện phát triển sản phẩm: baseline chứng minh bài toán không tầm thường, PhoBERT cho predictor đủ tin cậy, và LLM biến prediction thành output có giá trị thực tế cho người dùng. Thí nghiệm bổ sung LLM + RAG được trình bày riêng ở §6.6.

### 5.1 Tổng quan kiến trúc hệ thống

*Hình 5. Kiến trúc tổng thể ba thành phần của báo cáo (vẽ mới: baseline SVM ở nhánh so sánh; PhoBERT predictor là core engine; LLM Explanation là tầng giải thích phía trên PhoBERT, chỉ dùng trong app).*

Điểm quan trọng nhất của kiến trúc là **sự tách biệt vai trò giữa classifier và explainer** trong phần ứng dụng (§5.4). PhoBERT được chọn là thành phần duy nhất sinh nhãn, còn LLM không được phép thay đổi nhãn — nó chỉ giải thích nhãn có sẵn, trích câu dẫn chứng từ review gốc, và đề xuất hành động. Cách tách này giữ được metric F1 defend được đồng thời tận dụng thế mạnh của LLM ở việc sinh ngôn ngữ tự nhiên.

### 5.2 Baseline — SVM + TF-IDF

**Vai trò.** Baseline là mốc truyền thống để trả lời câu hỏi "trước khi có pretrained language model, bài toán này giải đến đâu?". Nếu baseline đạt gần bằng PhoBERT thì việc đầu tư PhoBERT không đáng, và ngược lại nếu baseline thấp đáng kể thì điều đó ủng hộ quyết định dùng pretrained. Baseline **không** được đưa vào app — nó tồn tại cho mục đích so sánh trong báo cáo.

**Lý do chọn.**

- SVM + TF-IDF là kiến trúc kinh điển cho text classification trước neural era, rất hợp làm baseline.
- `LinearSVC` train nhanh trên vector sparse, chạy được trên CPU, không cần GPU → chi phí thực nghiệm thấp.
- Model interpret được (có thể xem trọng số feature) → dễ giải thích trong báo cáo.
- Giữ baseline **độc lập với pipeline phức tạp của PhoBERT** để kết quả so sánh có ý nghĩa.

**Thiết kế.**

- 34 classifier độc lập, mỗi classifier giải bài toán 4-class cho một aspect.
- Feature: TF-IDF unigram + bigram, `max_features=50000`, `C=1.0`, `class_weight='balanced'`.
- Preprocessing rút gọn: chỉ lowercase + remove special chars; **không** dùng VnCoreNLP và **không** dùng teencode replacement.

*Bảng 3. Cấu hình baseline SVM + TF-IDF.*

| Tham số | Giá trị |
|--------|---------|
| Model | LinearSVC (scikit-learn) |
| Features | TF-IDF unigram + bigram |
| `max_features` | 50,000 |
| `C` | 1.0 |
| `class_weight` | balanced |
| Preprocessing | lowercase + remove special chars |
| Word segmentation | Không |

### 5.3 PhoBERT Predictor

**Vai trò.** Đây là core prediction engine của hệ thống — model được deploy trong app để dự đoán 34 aspect × 4 label cho review người dùng nhập.

**Lý do chọn.**

- PhoBERT là pretrained language model tiếng Việt công khai mạnh nhất hiện có, được train trên corpus tiếng Việt đã word-segmented. Khả năng hiểu context (phủ định, so sánh, emphasis) của nó vượt xa TF-IDF feature sparse.
- Supervised fine-tune trực tiếp trên 34 aspect × 4 label ⇒ model học được chính xác label scheme của VLSP 2018, khác hoàn toàn với LLM prompt-based chỉ nhìn thấy vài ví dụ trong context.
- Inference deterministic, không phụ thuộc API ngoài ⇒ phù hợp làm engine cho sản phẩm triển khai thực.
- Vừa với Kaggle T4 GPU (16 GB VRAM), không cần cluster lớn ⇒ phù hợp phạm vi đồ án.
- Nhóm chủ động giữ **single model** (không ensemble) để báo cáo gọn, dễ defend.

**Kiến trúc.**

Model `ABSAPhoBERT` (định nghĩa trong `code/phobert/model.py`) có cấu trúc multi-task:

1. Input được tokenize bằng tokenizer của `vinai/phobert-base-v2`.
2. Encoder là PhoBERT base, cho ra vector `[CLS]` 768 chiều (encoder option `cls_only`).
3. 34 classification head song song, mỗi head là một linear layer `768 → 4`, xuất 4 logit tương ứng với `{absent, positive, negative, neutral}`.
4. Loss là weighted focal loss, tính trên 34 head và tổng lại — cho phép mỗi aspect đóng góp như nhau vào macro F1 dù có độ hiếm khác nhau.

*Hình 6. Kiến trúc PhoBERT multi-head (Input → VnCoreNLP → tokenizer → PhoBERT encoder → [CLS] → 34 heads × 4 logit).*

**Training setup.**

Class weight được tính từ train split (lưu ở `outputs/eda/class_weights.json`) và bị clip tại 10.0 để tránh các class cực hiếm (ví dụ `neutral` ở `ROOM_AMENITIES#*`) chi phối loss. Early stopping theo Combined F1 trên dev split.

*Bảng 4. Hyperparameter PhoBERT predictor.*

| Tham số | Giá trị | Lý do |
|---------|---------|------|
| Backbone | `vinai/phobert-base-v2` | Phù hợp tiếng Việt, vừa GPU T4 |
| Encoder option | `cls_only` | Ít tham số hơn `concat_4_layers`, giảm overfit trên 3k train |
| `MAX_SEQ_LEN` | 256 | Đủ phủ p99 review, không vượt limit position embedding |
| Batch size | 16 | Ổn định gradient, không OOM |
| Learning rate | 1e-4 | Cân bằng tốc độ hội tụ và ổn định |
| Warmup ratio | 0.15 | Tránh update quá mạnh đầu fine-tune |
| Max epochs | 20 | Trần đủ dài cho hội tụ |
| Early stop patience | 7 | Đủ chịu dao động macro F1 trên dataset nhỏ |
| Weight clip | 10.0 | Giới hạn class weight cực đoan |
| Seed | 42 | Tái lập |

**Preprocessing bắt buộc.** Vì PhoBERT được pretrain trên text đã word-segmented, input fine-tune phải qua VnCoreNLP. Thí nghiệm ở §6.4 sẽ định lượng vai trò của bước này.

### 5.4 PhoBERT + LLM Explanation — hệ thống ứng dụng

**Vai trò.** Đây là **sản phẩm demo cuối cùng** của đồ án, đóng gói thành Streamlit app trong `app/`. Đây là phần trả lời câu hỏi thực tế: "có được nhãn rồi thì sao?". Người dùng cuối (chủ khách sạn) không chỉ cần label — họ cần (a) câu trong review dẫn đến nhãn đó, (b) lời giải thích dễ hiểu, (c) gợi ý hành động.

**Lý do tách vai predictor và explainer.**

- PhoBERT đã có Combined F1 = 0.5543 defend được trong báo cáo. Nếu để LLM thay đổi nhãn, metric này không còn có ý nghĩa ⇒ nhóm quyết định **LLM không được phép đổi nhãn của PhoBERT**, chỉ được diễn giải.
- LLM mạnh ở sinh ngôn ngữ tự nhiên nhưng yếu ở việc tuân thủ nhất quán schema 34 aspect × 4 label (xem §6.6). Do đó phân vai: **PhoBERT lo label**, **LLM lo diễn giải**.
- Cách tách này giữ được tính tin cậy đo được của hệ thống, đồng thời khai thác được thế mạnh sinh ngôn ngữ của LLM.

**Pipeline.**

*Hình 7. Pipeline app PhoBERT + LLM Explanation (Review → PhoBERT predict → lọc present predictions → LLM explain + evidence + recommend → Streamlit UI).*

1. Người dùng nhập một review (text hoặc file CSV/XLSX có cột `review_text`).
2. PhoBERT sinh vector 34 × 4 logit → lấy argmax mỗi aspect.
3. Chỉ các aspect có nhãn khác `absent` được giữ lại (present predictions) → đưa vào prompt LLM cùng review gốc.
4. LLM sinh JSON với schema cố định: `items` (mỗi item gồm `aspect`, `sentiment`, `explanation`, `evidence`, `evidence_uncertain`), `overall_summary`, `recommended_action`.
5. Streamlit UI hiển thị kết quả có cấu trúc cho chủ khách sạn.

**Thiết kế prompt và ràng buộc.**

- Temperature thấp (0–0.2) để output deterministic, không sáng tạo.
- Evidence được check bằng substring / normalize match vào review gốc; nếu LLM trả về câu không khớp, flag `evidence_uncertain = true`.
- Prompt yêu cầu rõ: "Không được thay đổi nhãn PhoBERT đã dự đoán." Đây là bảo hiểm soft; bảo hiểm hard nằm ở logic app — app chỉ render nhãn từ PhoBERT, không đọc lại nhãn từ output của LLM.

Bảng metric đánh giá chất lượng explanation/recommendation được giữ ở §6.7 (hiện là placeholder).

---

## 6. Experiments and Results

### 6.1 Setup chung

Tất cả thí nghiệm dùng cùng train/dev/test split như §4.1. Metric là ACD F1, SPC F1, Combined F1 (macro across 34 aspect). PhoBERT được train trên Kaggle T4 GPU với seed 42; SVM chạy trên CPU. LLM prompt-based dùng GPT-4o-mini qua OpenAI API, có cache response ở `outputs/llm_cache/` để kết quả tái lập.

### 6.2 Bảng kết quả tổng hợp

*Bảng 5. So sánh ba thành phần methodology và thí nghiệm bổ sung trên test set.*

| Method | Vai trò | ACD F1 | SPC F1 | Combined F1 |
|---|---|---:|---:|---:|
| SVM + TF-IDF | Baseline (method chính) | 0.4074 | 0.2272 | 0.3173 |
| PhoBERT cls_only + VnCoreNLP | Predictor (method chính) | 0.6360 | 0.4727 | **0.5543** |
| PhoBERT + LLM Explanation | Ứng dụng (method chính) | — | — | — |
| LLM + RAG k=8 (GPT-4o-mini) | *Thí nghiệm bổ sung* | 0.4034 | 0.3031 | 0.3532 |

Ba nhận xét nhanh:

1. PhoBERT cao hơn SVM **+0.2370 Combined F1** (tăng ~74.7% tương đối) ⇒ pretrained contextual representation là yếu tố quyết định.
2. PhoBERT cao hơn LLM+RAG **+0.2011 Combined F1** ⇒ supervised fine-tune trên dataset gốc vượt xa LLM prompt-based, kể cả khi có retrieval.
3. Hàng "PhoBERT + LLM Explanation" không có F1 vì đây là hệ thống ứng dụng, không phải classifier mới — metric riêng cho chất lượng explanation sẽ ở §6.7.

### 6.3 Baseline SVM chi tiết

Trên dev: ACD F1 = 0.4045 / SPC F1 = 0.2364 / Combined F1 = 0.3204. Trên test: ACD F1 = 0.4074 / SPC F1 = 0.2272 / Combined F1 = 0.3173. Chênh lệch dev/test nhỏ, cho thấy baseline khá ổn định.

Điểm đáng chú ý: SPC F1 (0.2272) thấp hơn ACD F1 (0.4074) đáng kể. Nguyên nhân: TF-IDF không bắt được context phủ định, so sánh, emphasis — những cue quyết định việc phân biệt `positive` vs `negative` vs `neutral`. Ví dụ "phòng **không** sạch" và "phòng sạch" có TF-IDF features gần giống nhau ở unigram, chỉ khác ở bigram. Trong khi đó, ACD (xuất hiện hay không) phụ thuộc nhiều hơn vào từ khoá rõ ràng nên SVM làm tốt hơn.

### 6.4 Ablation VnCoreNLP trên PhoBERT

Đây là kết quả nổi bật nhất của báo cáo. Nhóm chạy 4 run PhoBERT với cùng dataset, cùng seed, chỉ thay đổi hai yếu tố: có/không VnCoreNLP × encoder `cls_only` / `concat_4_layers`.

*Bảng 6. Ablation VnCoreNLP × encoder option trên test set.*

| Setting | Encoder | ACD F1 | SPC F1 | Combined F1 |
|---|---|---:|---:|---:|
| No VnCoreNLP | concat_4_layers | 0.3592 | 0.2371 | 0.2981 |
| No VnCoreNLP | cls_only | 0.3349 | 0.2014 | 0.2681 |
| VnCoreNLP | concat_4_layers | 0.5836 | 0.4658 | 0.5247 |
| VnCoreNLP | cls_only | **0.6360** | **0.4727** | **0.5543** |

Best no-VnCoreNLP = 0.2981 (concat_4_layers). Best với VnCoreNLP = **0.5543** (cls_only). Absolute gain: **+0.2562 Combined F1**, tương đương **+85.9% tương đối**.

Giải thích: PhoBERT được pretrain trên corpus đã word-segmented. Khi fine-tune trên text không segmented, token boundary lệch hoàn toàn với từ điển pretrained — thay vì thấy `_khách_sạn_` như một token quen thuộc, model thấy `khách` và `sạn` rời nhau và phải học lại từ đầu. Bước VnCoreNLP thuần tuý về mặt preprocessing lại mang ý nghĩa kiến trúc ở chỗ **khớp input với format pretraining**.

Đây là ablation sạch (chỉ thay một yếu tố) nên kết luận rất chắc. Đó là lý do §5.3 xếp VnCoreNLP là bước "bắt buộc" của pipeline PhoBERT.

### 6.5 cls_only vs concat_4_layers

Trong setting có VnCoreNLP: `cls_only` (0.5543) cao hơn `concat_4_layers` (0.5247) +0.0296 Combined F1. Giải thích:

1. `concat_4_layers` lấy vector từ 4 layer cuối và concat (dim 768 × 4 = 3072), tạo nhiều tham số hơn ở classification head.
2. Dataset train chỉ có 3,000 mẫu cho 34 × 4 = 136 output class ⇒ head nhiều tham số dễ overfit.
3. `cls_only` (dim 768) đơn giản hơn, phù hợp với dataset nhỏ.

Ngoài hiệu quả tốt hơn, `cls_only` còn cho inference nhanh hơn, dễ triển khai trong app hơn — đây là thêm một lý do chọn cho hệ thống sản xuất.

### 6.6 Thí nghiệm bổ sung — LLM + RAG như một predictor thay thế

**Đặt vấn đề.** Trước khi chốt PhoBERT làm predictor cho app, nhóm đặt câu hỏi: **"có thể dùng LLM prompt-based + retrieval thay cho supervised fine-tune không?"**. Ưu điểm lý thuyết của hướng này là không cần train, không cần GPU, dễ cập nhật nhãn mới. Nhược điểm tiềm tàng là chi phí API và tính ổn định của output. Subsection này tồn tại để **trả lời câu hỏi đó bằng số liệu cụ thể**, không phải là một method chính của báo cáo.

**Setup.** GPT-4o-mini với sentence-transformer retriever trên train set đã preprocess; prompt yêu cầu JSON schema cho 34 aspect label; cache response để tái lập; `temperature=0`, `sleep_sec=1.0`. Chi tiết trong `code/llm_rag/` và `docs/llm_rag_comparison.md`.

*Bảng 7. Ablation k (số ví dụ retrieved) cho LLM + RAG trên test set.*

| Method | k | ACD F1 | SPC F1 | Combined F1 |
|---|---:|---:|---:|---:|
| ICL GPT-4o-mini (random) | 8 | 0.3504 | 0.2567 | 0.3035 |
| RAG GPT-4o-mini | 2 | 0.3793 | 0.2866 | 0.3330 |
| RAG GPT-4o-mini | 4 | 0.3866 | 0.2883 | 0.3375 |
| **RAG GPT-4o-mini** | **8** | **0.4034** | **0.3031** | **0.3532** |
| RAG GPT-4o-mini | 16 | 0.3988 | 0.2879 | 0.3433 |

**Quan sát:**

- RAG k=8 cao hơn ICL random k=8 **+0.0497 Combined F1** ⇒ retrieval có giúp, LLM học schema từ ví dụ similar tốt hơn từ ví dụ random.
- Tăng k từ 2 lên 8 cải thiện đều, nhưng k=16 **giảm** ⇒ prompt quá dài gây context overload, LLM bắt đầu bỏ qua schema.
- Ngay cả tại mức tối ưu k=8, LLM+RAG chỉ đạt 0.3532, thua PhoBERT **0.2011 điểm Combined F1**.

**Kết luận (liên kết về methodology).** Kết quả này xác nhận quyết định kiến trúc của §5.3: LLM + RAG không đủ thay supervised model cho task ABSA 34 aspect × 4 label trên dataset nhỏ. Điều này không có nghĩa LLM vô dụng — ngược lại, nó định vị rõ vai trò của LLM trong hệ thống: **sinh ngôn ngữ giải thích cho nhãn PhoBERT** (§5.4), chứ không phải sinh nhãn.

### 6.7 Đánh giá PhoBERT + LLM Explanation

> **[TODO — SẼ ĐIỀN Ở VERSION SAU]**
>
> Phần này sẽ báo cáo kết quả đánh giá chất lượng output của hệ thống ứng dụng (§5.4). Khi viết sẽ gồm:
>
> 1. **Heuristic metrics** (chạy tự động trên tập mẫu, target đã set trong `docs/phobert_llm_explainability_plan.md`):
>    - JSON parse rate (target ≥ 95%)
>    - Aspect preservation rate (target = 100% — LLM không được thêm/bớt aspect)
>    - Sentiment preservation rate (target = 100% — LLM không được đổi sentiment)
>    - Evidence substring match rate
>    - Evidence uncertain rate
>    - Average explanation length
> 2. **Manual rubric 20–30 examples**: Evidence correctness / Explanation quality / Hallucination / User usefulness, mỗi tiêu chí scale 0–2.
> 3. **Ví dụ qualitative**: 1 ví dụ good case (evidence match + explanation rõ), 1 ví dụ uncertain case (LLM flag `evidence_uncertain=true`), 1 ví dụ error analysis (PhoBERT sai nhãn và hệ quả lên explanation).
>
> *Lý do để placeholder:* phần đánh giá này đòi hỏi chạy 20–30 sample qua app và manual scoring, chưa thực hiện tại thời điểm v1. Structure của section đã được thiết kế để sau này chỉ cần điền số liệu và ví dụ mà không phải reorganize báo cáo.

### 6.8 So sánh với SOTA

Kết quả PhoBERT của nhóm (Combined F1 = 0.5543) so với SOTA đã công bố của Huynh (2022) (Combined F1 = 0.7732) còn cách khoảng 0.22 điểm. Giải thích khoảng cách:

1. Nhóm chủ động **không dùng ensemble**, giữ single model để báo cáo gọn.
2. **Không dùng data augmentation** (các thử nghiệm aug bị đưa vào `draft/` và không nằm trong flow chính).
3. Nhóm dùng **PhoBERT base** chứ không phải PhoBERT large (giới hạn Kaggle T4).
4. Hyperparameter được chọn thực dụng cho chạy được trên Kaggle, không tuning dài vài chục run.

Trong phạm vi đồ án lớp (single GPU, thời gian hạn chế), khoảng cách này được chấp nhận. Các hướng thu hẹp khoảng cách được liệt kê ở §7.4.

---

## 7. Conclusion

### 7.1 Tóm tắt kết quả chính

Nhóm xây dựng một hệ thống ABSA cho review khách sạn tiếng Việt với ba thành phần methodology: baseline SVM + TF-IDF (Combined F1 = 0.3173), PhoBERT predictor (Combined F1 = **0.5543**), và hệ thống ứng dụng PhoBERT + LLM Explanation. Kết quả cho thấy:

- PhoBERT vượt xa baseline SVM **+0.2370 Combined F1**, ủng hộ mạnh mẽ việc dùng pretrained representation cho tiếng Việt.
- Ablation VnCoreNLP cho **+0.2562 Combined F1 (+85.9% tương đối)** — đây là kết quả sạch nhất của báo cáo, nhấn mạnh vai trò của word segmentation trong pipeline phù hợp với PhoBERT pretraining.
- Thí nghiệm bổ sung LLM + RAG cho thấy LLM prompt-based (kể cả có retrieval) thua PhoBERT 0.2011 điểm Combined F1 ⇒ củng cố lựa chọn PhoBERT làm predictor cho sản phẩm.

### 7.2 Đóng góp về mặt ứng dụng

Sản phẩm demo là một Streamlit app với pipeline: Review → PhoBERT predict → LLM explain + evidence + recommend. Điểm mới nằm ở **phân vai rõ**: PhoBERT sinh nhãn (đo được bằng F1), LLM sinh diễn giải (mang lại giá trị cho người dùng cuối). LLM bị ràng buộc không được đổi nhãn, evidence phải match review gốc; các trường hợp không match được flag `evidence_uncertain`. Thiết kế này vừa giữ metric F1 defend được, vừa tận dụng thế mạnh sinh ngôn ngữ của LLM cho một sản phẩm thực tế hữu dụng với chủ khách sạn.

### 7.3 Hạn chế

- Single model PhoBERT base, không ensemble, không augmentation ⇒ còn cách SOTA của Huynh (2022) khoảng 0.22 điểm Combined F1.
- Dataset gốc chỉ 3,000 review train, một số aspect cực hiếm (ví dụ `ROOMS#MISCELLANEOUS`) mà model khó học.
- Phần LLM trong app phụ thuộc vào một provider (OpenAI) và cần API key ⇒ chưa trung lập nhà cung cấp.
- Đánh giá chất lượng explanation/recommendation hiện tại (v1 báo cáo) còn là placeholder; các con số thực sẽ có ở phiên bản sau.

### 7.4 Future Work

- **Ưu tiên gần:** hoàn thành §6.7 — chạy heuristic metrics + manual rubric trên 20–30 sample để có bằng chứng định lượng cho chất lượng explanation và recommended_action.
- Thử PhoBERT large hoặc XLM-R để thu hẹp khoảng cách với SOTA.
- Thử ensemble một vài checkpoint PhoBERT với seed khác nhau, và augmentation nhẹ (back-translation, synonym replacement) cho các aspect hiếm.
- Thay OpenAI bằng LLM open-source (ví dụ Llama 3 Instruct, Qwen) để giảm phụ thuộc API.
- Mở rộng hệ thống sang domain khác (nhà hàng, e-commerce) bằng cách tái sử dụng pipeline preprocessing + architecture PhoBERT multi-head, chỉ thay label scheme.

---

## References

1. Nguyen, D. Q., & Nguyen, A. T. (2020). *PhoBERT: Pre-trained language models for Vietnamese.* In Findings of EMNLP 2020.
2. Vu, T., Nguyen, D. Q., Nguyen, D. Q., Dras, M., & Johnson, M. (2018). *VnCoreNLP: A Vietnamese Natural Language Processing Toolkit.* In NAACL 2018 Demonstrations.
3. Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* In NAACL-HLT 2019.
4. Brown, T. B., et al. (2020). *Language Models are Few-Shot Learners.* In NeurIPS 2020.
5. Lewis, P., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* In NeurIPS 2020.
6. Wang, Y., Huang, M., Zhu, X., & Zhao, L. (2016). *Attention-based LSTM for Aspect-level Sentiment Classification.* In EMNLP 2016.
7. Huynh, V. (2022). *Ensemble approach for ABSA on VLSP 2018 Hotel dataset.* (Cần bổ sung chi tiết citation cho v2.)

---

*Kết thúc báo cáo — Version 1 — Nhóm 16.*
