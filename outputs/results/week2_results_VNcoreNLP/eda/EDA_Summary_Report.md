# Báo cáo Tổng hợp EDA — ABSA VLSP 2018 Hotel

> Tổng hợp kết quả từ `step1_eda.py` | Dataset: VLSP 2018 Hotel | Ngày: 2026-03-26

---

## 1. Tổng quan Dataset

| Split | Số mẫu | Ghi chú |
|-------|--------|---------|
| Train | 3,000  | Dùng để tính class weights |
| Dev   | 2,000  | Validation |
| Test  | 600    | Evaluation cuối cùng |
| **Tổng** | **5,600** | |

**Cấu trúc:** 1 cột `Review` + 34 cột aspect, mỗi aspect ∈ {0=absent, 1=positive, 2=negative, 3=neutral}

---

## 2. Preprocessing Pipeline — Đã làm gì với Data?

> Kết quả: 3 file cache `data/*_preprocessed.csv`, mỗi file thêm cột `processed_review` bên cạnh `Review` gốc.

### Pipeline 5 bước (deterministic + idempotent)

| Bước | Tên | Mô tả | Ví dụ |
|------|-----|-------|-------|
| 1 | **Unicode NFC** | Chuẩn hóa ký tự tiếng Việt tổ hợp → precomposed | `ờ` (2 codepoint) → `ờ` (1 codepoint) |
| 2 | **Normalize whitespace** | Xóa newline, tab, khoảng trắng thừa, strip | `"  phòng  \n sạch  "` → `"phòng sạch"` |
| 3 | **Teencode replacement** | ~60 entries domain khách sạn, từ dài nhất trước | `ko` → `không`, `ks` → `khách sạn`, `nv` → `nhân viên` |
| 4 | **Remove special chars** | Giữ chữ cái (có dấu), số, dấu câu `.,!?;:-/()` | `@#$%` bị xóa |
| 5 | **Word segmentation** | Tách từ ghép, nối bằng `_` (underthesea fallback) | `khách sạn` → `khách_sạn` |

### Ví dụ thực tế từ data

```
ORIG: Rộng rãi KS mới nhưng rất vắng. Các dịch vụ chất lượng chưa cao và thiếu.
PROC: Rộng_rãi_khách_sạn_mới_nhưng_rất_vắng_._Các_dịch_vụ_chất_lượng_chưa_cao_và_thiếu_.

ORIG: Địa điểm thuận tiện, trong vòng bán kính 1,5km nhiều quán ăn ngon
PROC: Địa_điểm_thuận_tiện_,_trong_vòng_bán_kính_1,5_km_nhiều_quán_ăn_ngon

ORIG: Tôi ở đây lần thứ 2 và có ý định sẽ ở thêm... khi check in chẳng ai quan tâm
PROC: Tôi_ở_đây_lần_thứ_2_và_có_ý_định_sẽ_ở_thêm_..._khi_check_in_chẳng_ai_quan_tâm
```

### Tại sao từng bước quan trọng?

**Unicode NFC** — tiếng Việt có 2 cách encode cùng 1 chữ (tổ hợp dấu vs precomposed). Không chuẩn hóa → tokenizer coi `ờ` và `ờ` là 2 token khác nhau, gây noise embedding.

**Teencode** — dataset review khách sạn chứa nhiều viết tắt (`ko`, `nv`, `ks`, `bfst`...). Không thay → PhoBERT sẽ `[UNK]` hoặc tách sai, mất thông tin sentiment.

**Word segmentation (bước quan trọng nhất)** — PhoBERT được pre-train trên corpus đã word-segment bằng VnCoreNLP. Nếu bỏ bước này, `nhân viên` bị tokenize thành `nhân` + `viên` (2 token riêng lẻ) thay vì `nhân_viên` (1 đơn vị ngữ nghĩa) → giảm ~1–2% F1.

### Gap đã xác nhận — ASCII Vietnamese không được xử lý

Review viết **không dấu** (khá phổ biến trong dataset) không được hưởng lợi từ teencode:

```
ORIG: Co view huong Ho tay- sach se-nhan vien tan tinh
PROC: Co_view_huong_Ho_tay-_sach_se-nhan_vien_tan_tinh
      ↑ "nhan vien" không → "nhân_viên", "sach se" không → "sạch_sẽ"
```

Teencode dict dùng từ có dấu (`nhân viên`, `khách sạn`) nên không match được bản không dấu. PhoBERT vẫn xử lý được nhưng chất lượng embedding thấp hơn cho nhóm review này.

### Segmenter đã dùng

VnCoreNLP **không khả dụng** (yêu cầu Java 8+ và model ~200MB) → dùng **underthesea** làm fallback. Kết quả word-segment có thể kém hơn VnCoreNLP ~1–2% F1 theo SOTA paper.

---

## 3. Độ dài Review (Review Length)

### Train split

| Metric | Words | Characters |
|--------|-------|-----------|
| p99    | **243** | 1,100 |

- Phân phối **lệch phải mạnh**: phần lớn review có 10–50 từ
- Tồn tại outlier dài tới ~1,000 từ (có thể là review dài bất thường)
- Phân phối train dài hơn dev đáng kể (p99 train=243 vs dev=159)

### Dev split

| Metric | Words | Characters |
|--------|-------|-----------|
| p99    | 159   | 682 |

**→ Khuyến nghị `MAX_SEQ_LEN = 384`**
- Công thức: p99_words × 1.5 (word segment tạo thêm tokens) = 243 × 1.5 ≈ 365 → làm tròn lên bội số 64 = **384**
- Tăng so với default 256 trong constants.py — cần cập nhật tuần 2
- Đây là giá trị **lớn hơn đáng kể** so với MAX_SEQ_LEN=256 ban đầu

---

## 3. Aspect Presence Rate (Tỷ lệ Xuất hiện)

### Top 5 aspect phổ biến nhất (Train)

| Rank | Aspect | Presence Rate |
|------|--------|--------------|
| 1 | SERVICE#GENERAL | ~63% |
| 2 | HOTEL#GENERAL | ~43% |
| 3 | LOCATION#GENERAL | ~41% |
| 4 | HOTEL#COMFORT | ~40% |
| 5 | ROOMS#DESIGN&FEATURES | ~30% |

### Bottom 5 — Aspect hiếm nhất (Train)

| Rank | Aspect | Presence Rate | Ghi chú |
|------|--------|--------------|---------|
| 34 | ROOM_AMENITIES#PRICES | ~0% | **CRITICAL: 0 samples có nhãn 1/2/3** |
| 33 | ROOM_AMENITIES#MISCELLANEOUS | ~0% | Chỉ có nhãn negative |
| 32 | ROOMS#MISCELLANEOUS | ~0% | Chỉ có nhãn negative |
| 31 | FOOD&DRINKS#MISCELLANEOUS | ~0% | |
| 30 | FACILITIES#MISCELLANEOUS | ~1–2% | Chỉ có negative |

### Nhận xét phân phối theo Split

- Thứ tự ranking **nhất quán** giữa train/dev/test — dữ liệu phân chia stratified tốt
- `SERVICE#GENERAL` luôn đứng đầu (~63–70%) ở cả 3 splits
- `LOCATION#GENERAL` xuất hiện nhiều hơn ở dev/test so với train
- Các rare aspects đều nhất quán ở dưới đường 10%

---

## 4. Label Breakdown (Phân tích Nhãn)

### Phân phối nhãn toàn cục (Global — Train)

| Nhãn | Class | Weight (Global) | Nhận xét |
|------|-------|----------------|---------|
| 0 | absent | **1.0** (majority) | Chiếm đa số tuyệt đối |
| 1 | positive | 8.61 | Thứ phổ biến thứ 2 |
| 2 | negative | 27.96 | Ít hơn positive ~3x |
| 3 | neutral | **154.48** | Cực kỳ hiếm — ~154× ít hơn absent |

**→ Mất cân bằng class nghiêm trọng**, đặc biệt nhãn `neutral`:
- Absent chiếm ~85%+ tổng nhãn
- Neutral cần weight lên tới **154×** để cân bằng

### Pattern nhãn theo Aspect

- Đa số aspects: absent >> positive > negative >> neutral
- `SERVICE#GENERAL`: Positive là majority (weight=1.0 cho positive, 1.46 cho absent) — aspect duy nhất có positive phổ biến hơn absent
- `HOTEL#GENERAL`, `HOTEL#COMFORT`, `LOCATION#GENERAL`: Positive xuất hiện nhiều (~40–65% trong nhóm present)
- `HOTEL#CLEANLINESS`, `ROOMS#CLEANLINESS`: Negative chiếm tỷ trọng cao trong nhóm present

---

## 5. Class Imbalance — Rare Aspects

### Xác nhận và cập nhật danh sách RARE_ASPECTS

**Đã xác nhận trong constants.py:**

| Aspect | Vấn đề |
|--------|--------|
| FACILITIES#MISCELLANEOUS | Không có nhãn neutral; chỉ có negative và rất ít positive |
| ROOM_AMENITIES#PRICES | **Không có nhãn 1/2/3 trong train** — weight chỉ có "0":1.0 |
| ROOM_AMENITIES#MISCELLANEOUS | Chỉ có nhãn 2 (negative), không có positive/neutral |
| ROOM_AMENITIES#CLEANLINESS | Không có neutral; ít samples |
| ROOM_AMENITIES#DESIGN&FEATURES | Không có nhãn 1 (positive) |
| HOTEL#DESIGN&FEATURES | Đã confirm rare |

**Phát hiện thêm (nên xem xét bổ sung):**

| Aspect | Vấn đề |
|--------|--------|
| ROOMS#MISCELLANEOUS | Chỉ có nhãn 2 (negative) — cực kỳ hiếm |
| FOOD&DRINKS#MISCELLANEOUS | Weight positive=426.7, negative=597.4 — rất hiếm |
| FACILITIES#PRICES | Weight positive=98.2, negative=134.0 — gần ngưỡng rare |
| FACILITIES#COMFORT | Weight positive=33.0 |
| ROOMS#QUALITY | Weight positive=31.2 |

### Per-aspect Weight Highlights

Các aspect có weight cực cao (cần chú ý trong training):

| Aspect | w_neutral | w_negative | w_positive |
|--------|-----------|-----------|-----------|
| ROOM_AMENITIES#MISCELLANEOUS | — | 999.0 | — |
| ROOMS#MISCELLANEOUS | — | 599.0 | — |
| FOOD&DRINKS#MISCELLANEOUS | 2987.0 | 597.4 | 426.7 |
| FACILITIES#CLEANLINESS | 2828.0 | 56.6 | 23.4 |
| FACILITIES#COMFORT | 2868.0 | 65.2 | 33.0 |
| ROOM_AMENITIES#COMFORT | 1450.0 | 70.7 | 50.9 |

---

## 6. Encoder Config (Kết quả Lưu)

File `outputs/eda/encoder_config.json`:

```json
{
  "recommended_max_seq_len": 384,
  "p99_word_count": 243,
  "encoder_option": "concat_4_layers",
  "encoder_hidden_size": 3072,
  "note": "concat_4_layers theo SOTA ds4v IEEE 2022"
}
```

**Ý nghĩa cho Tuần 2:**
- Dùng `concat_4_layers` (không phải `cls_only`) → input dim = 3072
- MAX_SEQ_LEN = **384** (không phải 256 như default) — cần thêm memory GPU

---

## 7. Phân tích & Khuyến nghị

### 7.1 Vấn đề nghiêm trọng cần xử lý ở Tuần 2

**A. ROOM_AMENITIES#PRICES hoàn toàn absent trong train**
- Không có bất kỳ nhãn positive/negative/neutral nào trong train
- Model sẽ luôn predict absent → ACD F1 = 0 cho aspect này
- **Khuyến nghị:** Kiểm tra lại data; nếu đúng, loại khỏi tính SPC F1 hoặc xử lý đặc biệt

**B. Nhãn neutral cực kỳ hiếm (weight=154)**
- `neutral` chỉ xuất hiện trong <1% tổng nhãn
- Weighted loss với w=154 có thể gây instability
- **Khuyến nghị:** Clip weight tối đa ở 10–20 (thực nghiệm); hoặc group neutral+positive thành "non-negative" cho SPC

**C. MAX_SEQ_LEN = 384 thay vì 256**
- Tăng memory ~50% so với dự kiến ban đầu
- Với batch_size=16: cần ~16GB VRAM để chạy PhoBERT
- **Khuyến nghị:** Giảm batch_size xuống 8 hoặc dùng gradient accumulation nếu GPU bị giới hạn

### 7.2 Insight quan trọng từ Aspect Distribution

- **SERVICE#GENERAL** là aspect quan trọng nhất: xuất hiện trong 63% reviews, và là aspect duy nhất mà positive là majority. Model cần học tốt aspect này.
- **Top 5 aspects** (SERVICE, HOTEL#GENERAL, LOCATION, HOTEL#COMFORT, ROOMS#DESIGN) chiếm phần lớn performance metric — đây là nơi model sẽ đạt F1 cao nhất.
- **Bottom 10 aspects** có thể kéo Macro-ACD-F1 xuống đáng kể dù model làm tốt top aspects.

### 7.3 Chiến lược cho SPC F1

Từ label breakdown:
- Positive chiếm ~70–80% trong nhóm present cho most aspects → bias toward positive là nguy cơ thực
- Negative cần attention đặc biệt cho: HOTEL#CLEANLINESS, ROOMS#CLEANLINESS, SERVICE#GENERAL
- Neutral hầu như không học được ở rare aspects → `zero_division=0` quan trọng

### 7.4 Consistency giữa Splits

- Phân phối train/dev/test khá nhất quán → data split được làm tốt
- Tuy nhiên dev p99_words=159 << train p99_words=243: train có nhiều review dài hơn đáng kể
- Test split nhỏ (600 samples) → kết quả test có thể có variance cao cho rare aspects

---

## 8. Files Đã Tạo

| File | Mô tả | Dùng bởi |
|------|-------|---------|
| `outputs/eda/class_weights.json` | Global + per-aspect weights | Tuần 2 (weighted loss) |
| `outputs/eda/encoder_config.json` | MAX_SEQ_LEN=384, concat_4_layers | Tuần 2 (model config) |
| `outputs/eda/train_aspect_presence.png` | Bar chart presence rate (train) | Báo cáo |
| `outputs/eda/train_label_breakdown.png` | Stacked bar 4 labels (train) | Báo cáo |
| `outputs/eda/train_review_length.png` | Histogram word/char count (train) | Báo cáo |
| `outputs/eda/dev_aspect_presence.png` | Bar chart presence rate (dev) | Báo cáo |
| `outputs/eda/dev_label_breakdown.png` | Stacked bar 4 labels (dev) | Báo cáo |
| `outputs/eda/dev_review_length.png` | Histogram word/char count (dev) | Báo cáo |
| `outputs/eda/test_aspect_presence.png` | Bar chart presence rate (test) | Báo cáo |
| `outputs/eda/test_label_breakdown.png` | Stacked bar 4 labels (test) | Báo cáo |
| `outputs/eda/test_review_length.png` | Histogram word/char count (test) | Báo cáo |

---

## 9. Checklist Tuần 2

Dựa trên kết quả EDA, cần chú ý khi implement PhoBERT training:

- [ ] Dùng `MAX_SEQ_LEN = 384` (load từ encoder_config.json)
- [ ] Dùng `concat_4_layers` encoder → hidden_size = 3072
- [ ] Load `class_weights.json` → tạo weighted CrossEntropyLoss
- [ ] Xem xét clip weight neutral tối đa (weight=154 quá cao)
- [ ] Xử lý edge case `ROOM_AMENITIES#PRICES` (0 positive/neg/neu trong train)
- [ ] Giảm batch_size nếu GPU < 16GB do MAX_SEQ_LEN lớn
- [ ] Phân tích per-aspect F1 sau training — đặc biệt chú ý bottom 10 aspects

---

*Tổng hợp từ: `step1_eda.py` outputs | Project: ABSA VLSP 2018 Hotel — HUST NLP Course*
