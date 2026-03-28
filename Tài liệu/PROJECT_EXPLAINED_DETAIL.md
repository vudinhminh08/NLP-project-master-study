# Hiểu Dự án ABSA VLSP 2018 Hotel — Từ A đến Z
## Dành cho người mới học NLP

> Tài liệu này giải thích mọi thứ từ cơ bản nhất.
> Nếu bạn hiểu được tài liệu này, bạn có thể tự trình bày và bảo vệ bài tập lớn.

---

# CHƯƠNG 1 — BÀI TOÁN LÀ GÌ?

---

## 1.1 Câu chuyện thực tế

Bạn là quản lý một chuỗi khách sạn. Mỗi ngày có hàng trăm đánh giá mới trên
Booking.com, TripAdvisor, Google Maps. Bạn muốn biết:

> "Khách hàng đang khen gì? Chê gì? Cụ thể ở khía cạnh nào?"

Nếu làm thủ công: 1 nhân viên đọc 100 review/ngày, mất 3-4 tiếng.
Nếu dùng AI: xử lý 10,000 review trong vài giây.

Đây chính là bài toán bạn đang giải.

---

## 1.2 Sentiment Analysis đơn giản vs ABSA

**Sentiment Analysis đơn giản (cũ, đơn giản):**
```
Input:  "Phòng sạch nhưng nhân viên thái độ kém"
Output: NEGATIVE  ← chỉ biết tổng thể là tiêu cực, không biết gì thêm
```

**ABSA — Aspect-Based Sentiment Analysis (mới, chi tiết hơn):**
```
Input:  "Phòng sạch nhưng nhân viên thái độ kém"
Output:
  ROOMS#CLEANLINESS → POSITIVE  ← phòng: tốt
  SERVICE#GENERAL   → NEGATIVE  ← nhân viên: tệ
```

ABSA hữu ích hơn vì cho biết CHÍNH XÁC điều gì tốt, điều gì cần cải thiện.

---

## 1.3 Hai nhiệm vụ phải làm cùng lúc

**ACD — Aspect Category Detection:**
Câu hỏi: "Review này có đề cập đến khía cạnh X không?"
```
Review: "Phòng rộng rãi, view đẹp"
ACD phải phát hiện: ROOMS#COMFORT (đề cập), ROOMS#DESIGN&FEATURES (đề cập)
                    SERVICE#GENERAL (KHÔNG đề cập), HOTEL#PRICES (KHÔNG đề cập)
```

**SPC — Sentiment Polarity Classification:**
Câu hỏi: "Với những khía cạnh được đề cập, cảm xúc là gì?"
```
ROOMS#COMFORT       → POSITIVE (phòng rộng = tốt)
ROOMS#DESIGN&FEATURES → POSITIVE (view đẹp = tốt)
```

Tại sao làm cùng lúc thay vì làm riêng? Vì nếu ACD sai, SPC sẽ sai theo.
Model end-to-end làm cả 2 cùng lúc hiệu quả hơn.

---

## 1.4 34 khía cạnh là gì?

Dataset VLSP 2018 Hotel định nghĩa 34 khía cạnh theo cấu trúc ENTITY#ATTRIBUTE:

```
ENTITY (thực thể):     ATTRIBUTE (thuộc tính):
  FACILITIES             CLEANLINESS, COMFORT, DESIGN&FEATURES,
                         GENERAL, MISCELLANEOUS, PRICES, QUALITY
  FOOD&DRINKS            MISCELLANEOUS, PRICES, QUALITY, STYLE&OPTIONS
  HOTEL                  CLEANLINESS, COMFORT, DESIGN&FEATURES,
                         GENERAL, MISCELLANEOUS, PRICES, QUALITY
  LOCATION               GENERAL
  ROOMS                  CLEANLINESS, COMFORT, DESIGN&FEATURES,
                         GENERAL, MISCELLANEOUS, PRICES, QUALITY
  ROOM_AMENITIES         CLEANLINESS, COMFORT, DESIGN&FEATURES,
                         GENERAL, MISCELLANEOUS, PRICES, QUALITY
  SERVICE                GENERAL
```

Ví dụ cách đọc:
- ROOMS#CLEANLINESS = độ sạch sẽ của phòng
- SERVICE#GENERAL = chất lượng dịch vụ nói chung
- HOTEL#PRICES = giá cả của khách sạn
- LOCATION#GENERAL = vị trí địa lý

---

## 1.5 Format dữ liệu CSV

Mỗi dòng trong file CSV có dạng:

```
Review, FACILITIES#CLEANLINESS, FACILITIES#COMFORT, ..., SERVICE#GENERAL
"Phòng rất sạch, nhân viên nhiệt tình", 0, 0, ..., 1, ..., 1
```

Cột Review: text đánh giá
34 cột còn lại: 0, 1, 2, hoặc 3

Ý nghĩa của các con số:
- 0 = absent: review KHÔNG đề cập đến khía cạnh này
- 1 = positive: đề cập và khen
- 2 = negative: đề cập và chê
- 3 = neutral: đề cập nhưng không rõ tốt hay xấu

---

---

# CHƯƠNG 2 — TUẦN 1: ĐÃ LÀM GÌ VÀ TẠI SAO?

---

## 2.1 EDA là gì và tại sao phải làm?

EDA = Exploratory Data Analysis = Phân tích khám phá dữ liệu.

Hãy tưởng tượng bạn nhận được một hộp nguyên liệu nấu ăn mà chưa mở ra bao giờ.
Trước khi nấu, bạn cần:
- Kiểm tra có đủ nguyên liệu không?
- Nguyên liệu nào nhiều, nguyên liệu nào thiếu?
- Có nguyên liệu bị hỏng không?

EDA làm đúng điều đó với dữ liệu. Nếu bỏ qua EDA và train model ngay,
bạn có thể mất nhiều giờ training chỉ để phát hiện ra dữ liệu có vấn đề.

---

## 2.2 Phát hiện 1 — Review dài bao nhiêu?

**Số liệu thực tế:**
```
Train set:
  Phần lớn review: 10-50 từ (ngắn)
  p50 (trung vị): ~30 từ
  p99 (99% review ngắn hơn giá trị này): 243 từ
  Max: ~1,000 từ (vài review cực dài bất thường)

Dev set:
  p99: 159 từ (ngắn hơn train đáng kể)
```

**p99 là gì?**
Nếu sắp xếp 3000 review từ ngắn đến dài, p99 là độ dài của review đứng
thứ 2,970 (tức là 99% review ngắn hơn con số này). Ở đây p99=243 có nghĩa
là chỉ có 30 review (~1%) dài hơn 243 từ.

**Tại sao điều này quan trọng?**

PhoBERT có giới hạn độ dài input — gọi là MAX_SEQ_LEN. Nếu review dài hơn
giới hạn này, phần cuối bị CẮT BỎ hoàn toàn.

```
Review: "Phòng sạch, view đẹp, ... [rất nhiều chữ] ... nhưng nhân viên tệ quá"
                                                                ↑
         Nếu bị cắt ở đây → model không thấy phần "nhân viên tệ" → sai SERVICE#GENERAL
```

**Tính MAX_SEQ_LEN như thế nào?**
```
p99 = 243 từ (tiếng Việt chưa segment)
× 1.5 (sau word segmentation, "nhân viên" → "nhân_viên" = 1 token
        nhưng PhoBERT tokenizer tiếp tục tách thêm subword)
= 364.5
→ làm tròn lên bội số 64 gần nhất = 384
```

Tại sao bội số 64? Vì GPU xử lý hiệu quả nhất với kích thước là bội số của 64.

**Hệ quả thực tế:**
- Dùng MAX_SEQ_LEN=256 (default): ~10% reviews bị cắt mất phần cuối → mất thông tin
- Dùng MAX_SEQ_LEN=384 (từ EDA): ~1% reviews bị cắt → ít mất thông tin hơn
- Đánh đổi: MAX_SEQ_LEN=384 cần ~50% VRAM hơn so với 256
- Quyết định: dùng 384, giảm batch_size từ 16 xuống 8 để fit vào GPU

---

## 2.3 Phát hiện 2 — Aspect nào phổ biến, aspect nào hiếm?

**Concept Presence Rate:**
Presence rate của 1 aspect = % reviews có đề cập đến aspect đó (tức là giá trị != 0)

```
Ví dụ: SERVICE#GENERAL có presence rate 63%
→ Trong 3000 reviews train, có 1890 reviews đề cập dịch vụ
→ 1110 reviews không đề cập
```

**Top 5 phổ biến nhất:**

| Aspect | Presence Rate | Ý nghĩa thực tế |
|--------|--------------|----------------|
| SERVICE#GENERAL | 63% | Người Việt hay bình luận về dịch vụ nhất |
| HOTEL#GENERAL | 43% | Nhận xét tổng quan về khách sạn |
| LOCATION#GENERAL | 41% | Vị trí quan trọng với khách |
| HOTEL#COMFORT | 40% | Sự thoải mái |
| ROOMS#DESIGN&FEATURES | 30% | Thiết kế, tiện nghi phòng |

**Bottom 5 hiếm nhất:**

| Aspect | Presence Rate | Vấn đề |
|--------|--------------|--------|
| ROOM_AMENITIES#PRICES | ~0% | 0 mẫu có nhãn 1/2/3 |
| ROOM_AMENITIES#MISCELLANEOUS | ~0% | Chỉ có nhãn 2 (negative) |
| ROOMS#MISCELLANEOUS | ~0% | Chỉ có nhãn 2 (negative) |
| FOOD&DRINKS#MISCELLANEOUS | ~0% | Cực kỳ hiếm |
| FACILITIES#MISCELLANEOUS | ~1-2% | Chỉ có negative |

**Tại sao aspect hiếm là vấn đề?**

Hãy nghĩ đến học sinh học bài thi. Nếu 99% câu hỏi thi về Toán nhưng
chỉ 1% về Văn, học sinh sẽ chủ yếu ôn Toán và có thể bị điểm Văn rất thấp.

Model cũng vậy: nếu chỉ thấy 20 reviews có ROOM_AMENITIES#CLEANLINESS
trong tổng số 3000, nó không có đủ "ví dụ" để học cách nhận biết aspect đó.

**Phát hiện nghiêm trọng nhất — ROOM_AMENITIES#PRICES:**

```
Kiểm tra trong train set:
ROOM_AMENITIES#PRICES:
  - Số review có nhãn 1 (positive): 0
  - Số review có nhãn 2 (negative): 0
  - Số review có nhãn 3 (neutral):  0
  - Số review có nhãn 0 (absent):   3000 (TẤT CẢ)
```

Điều này có nghĩa: KHÔNG MỘT review nào trong 3000 review train đề cập đến
giá cả tiện nghi phòng. Model sẽ không bao giờ học được aspect này.

Lý do có thể: Người Việt ít bình luận về giá tiện nghi (đèn, khăn tắm...) riêng lẻ.
Họ thường nói giá phòng (HOTEL#PRICES) hoặc giá dịch vụ (SERVICE#GENERAL) tổng quát.

**Hệ quả và xử lý:**
- Model sẽ luôn predict ROOM_AMENITIES#PRICES = 0 (absent) vì chưa bao giờ thấy khác
- ACD F1 cho aspect này = 0 (predict tất cả absent = sai khi test set có vài mẫu)
- Quyết định: vẫn giữ trong model (để output format nhất quán), nhưng
  LOẠI ASPECT NÀY khỏi tính Macro-F1 và giải thích rõ trong báo cáo

---

## 2.4 Phát hiện 3 — Mất cân bằng nhãn (Class Imbalance)

Đây là vấn đề QUAN TRỌNG NHẤT trong dataset này.

**Số liệu thực tế:**

Hãy đếm TẤT CẢ ô nhãn trong train set:
```
Tổng số ô = 3000 reviews × 34 aspects = 102,000 ô

absent   (0): ~87,000 ô  → 85.3%
positive (1): ~12,000 ô  → 11.8%
negative (2):  ~2,500 ô  →  2.4%
neutral  (3):    ~500 ô  →  0.5%
```

**Vẽ ra trực quan:**
```
absent   ████████████████████████████████████████████████ 85.3%
positive ██████                                            11.8%
negative █                                                  2.4%
neutral  ▏                                                  0.5%
```

**Tại sao phân phối này xảy ra?**

Rất hợp lý khi nghĩ về thực tế:
- 1 review trung bình chỉ đề cập 2-4 trong 34 aspects
- 30+ aspects còn lại đều = 0 (absent) cho mỗi review
- Nên absent luôn chiếm đa số tuyệt đối

**Vấn đề với mô hình học máy:**

Khi model huấn luyện, nó muốn tối thiểu hóa loss (sai số). Cách dễ nhất:
```
Cứ predict tất cả = 0 (absent):
  Đúng 85,300 lần (các ô absent)
  Sai  16,700 lần (các ô positive/negative/neutral)
  Accuracy = 85.3%  ← trông rất cao!
```

Nhưng model này HOÀN TOÀN VÔ DỤNG vì không phát hiện được aspect nào.

**Tại sao dùng F1 thay Accuracy?**

F1 tính cả Precision và Recall:

```
Precision = Trong những lần model nói "có aspect", bao nhiêu % đúng?
Recall    = Trong những lần thực sự "có aspect", model phát hiện được bao nhiêu %?
F1        = Trung bình điều hòa của Precision và Recall

Ví dụ model luôn predict absent:
  Precision = 0% (không bao giờ predict có aspect)
  Recall    = 0% (không phát hiện được gì)
  F1        = 0%  ← phản ánh đúng sự vô dụng
```

**Weight là gì và tại sao cần?**

Weight = "trọng số" cho từng nhãn trong loss function (hàm tính sai số).
Ý tưởng: khi model đoán sai nhãn hiếm, phạt NẶNG hơn.

```
Công thức tính weight:
weight = số lượng majority class / số lượng class đó

absent   (0): 87000 / 87000 = 1.00  ← phạt bình thường
positive (1): 87000 / 12000 = 7.25  ← phạt 7x nặng hơn
negative (2): 87000 / 3000  = 29.0  ← phạt 29x
neutral  (3): 87000 / 500   = 174.0 ← phạt 174x
```

Global weight neutral = 154.48 (gần với tính toán trên).

**Vấn đề với weight quá cao:**

Gradient là "hướng cập nhật" tham số model trong mỗi bước training.
Weight 154 nghĩa là mỗi lần sai neutral, gradient tăng 154 lần.

```
Normal gradient: +0.01 (nhỏ, ổn định)
Gradient × 154: +1.54 (rất lớn)

Khi gradient quá lớn:
  - Model "nhảy" quá mạnh sang hướng khác
  - Mất ổn định trong training
  - Loss dao động lên xuống thay vì giảm đều
  - Kỹ thuật gọi là "gradient explosion"
```

**Giải pháp: Clip weight tại 10.0**

```
Trước clip:
  absent   = 1.0
  positive = 8.61
  negative = 27.96
  neutral  = 154.48  ← quá cao

Sau clip tại 10.0:
  absent   = 1.0
  positive = 8.61
  negative = 10.0  ← bị giới hạn
  neutral  = 10.0  ← bị giới hạn
```

Ý nghĩa: model vẫn chú ý đến negative và neutral nhiều hơn absent,
nhưng không đến mức mất kiểm soát. Đây là compromise (đánh đổi) thực tế.

**Per-aspect weight đặc biệt nghiêm trọng:**

| Aspect | Weight_negative | Lý do |
|--------|----------------|-------|
| ROOM_AMENITIES#MISCELLANEOUS | 999.0 | Chỉ có 3 mẫu negative trong 3000 reviews |
| ROOMS#MISCELLANEOUS | 599.0 | Chỉ có 5 mẫu negative |
| FOOD&DRINKS#MISCELLANEOUS | 597.4 | Chỉ có ~5 mẫu |

Weight 999 sau clip → 10. Giảm 100 lần. Nhưng đó là điều cần thiết
để training ổn định. Kết quả là model sẽ kém với các aspects này,
nhưng ít nhất không làm hỏng toàn bộ training.

---

## 2.5 Phát hiện 4 — Preprocessing Pipeline

**Tại sao phải preprocessing?**

Dữ liệu thô từ internet = nguyên liệu chưa sơ chế.
Bạn không thể nấu thịt mà chưa rửa, chưa cắt miếng, chưa ướp.
Tương tự, bạn không thể đưa text thô vào PhoBERT và mong kết quả tốt.

**Bước 1 — Unicode NFC: Vấn đề kỳ lạ của tiếng Việt**

```python
# Hai chuỗi này NHÌN GIỐNG HỆT NHAU nhưng máy tính thấy KHÁC NHAU:
s1 = "ờ"  # precomposed: 1 codepoint U+1EDD
s2 = "ờ"  # decomposed: 3 codepoint (o + ̛ + ̀)

# Kiểm tra:
len(s1) = 1
len(s2) = 3
s1 == s2 → FALSE  ← máy tính nói chúng KHÁC NHAU!
```

Hệ quả với tokenizer:
- "ờ" precomposed → token A
- "ờ" decomposed  → token B (khác A!)

Nếu trong train set dùng precomposed nhưng test set dùng decomposed,
model thấy 2 "từ" khác nhau cho cùng 1 chữ → embedding sai.

Unicode NFC normalization đưa tất cả về 1 dạng chuẩn → vấn đề biến mất.

**Bước 2 — Normalize whitespace: Đơn giản nhưng cần thiết**

```
"Phòng  rất  sạch\n\n" → "Phòng rất sạch"
```

Data từ web thường có newline (\n), tab (\t), khoảng trắng đôi.
Những ký tự này không mang thông tin ngữ nghĩa nhưng làm tốn tokens.

**Bước 3 — Teencode: Vấn đề ngôn ngữ mạng xã hội**

Người Việt khi viết review online thường dùng từ tắt:

```
"sv" → "dịch vụ"
"nv" / "ntv" → "nhân viên"
"ks" → "khách sạn"
"ko" / "k" / "kg" → "không"
"dc" / "đc" → "được"
"ok" / "oke" → "tốt"
"bfst" → "bữa sáng"
```

**Vấn đề nếu không xử lý:**
PhoBERT được train trên văn bản chuẩn (báo chí, sách, Wikipedia...).
Nó chưa bao giờ thấy "bfst" hay "ntv" → tokenize thành [UNK] (unknown token)
hoặc tách sai → mất hoàn toàn ngữ nghĩa của từ đó.

**Gap đã phát hiện (vấn đề chưa giải quyết):**
Reviews viết KHÔNG DẤU không được hỗ trợ:
```
"nhan vien tan tinh" → vẫn là "nhan_vien_tan_tinh" (không có dấu)
```
Vì teencode dict dùng từ có dấu ("nhân viên") nên không match được.
PhoBERT vẫn xử lý được nhưng embedding chất lượng thấp hơn.
Đây là limitation đã biết, cần ghi vào báo cáo.

**Bước 4 — Remove special chars: Dọn ký tự lạ**

```
"Phòng @tuyệt vời#! Giá $200" → "Phòng tuyệt vời ! Giá 200"
```

Giữ lại: chữ cái (kể cả tiếng Việt), số, dấu câu cơ bản (.,!?;:-)
Xóa: @#$%^&* và các ký hiệu không có nghĩa trong văn bản đánh giá.

**Bước 5 — Word Segmentation: Bước QUAN TRỌNG NHẤT**

Tiếng Việt là ngôn ngữ đặc biệt: từ gồm nhiều âm tiết nhưng viết rời.

```
"nhân viên" = 2 âm tiết, NHƯNG là 1 từ (1 đơn vị nghĩa)
"nhân"  nghĩa: con người, nhân vật
"viên"  nghĩa: viên thuốc, viên chức
"nhân viên" nghĩa: người làm việc ← hoàn toàn khác!
```

PhoBERT được train trên corpus ĐÃ word-segment (từ ghép nối bằng _):
```
"nhân_viên" → 1 token → embedding đúng nghĩa
"nhân" + "viên" → 2 tokens → embedding SAI nghĩa
```

Kết quả sau segmentation:
```
"Khách sạn sạch, nhân viên nhiệt tình"
→ "Khách_sạn sạch , nhân_viên nhiệt_tình"
```

**VnCoreNLP vs underthesea:**

VnCoreNLP là tool segmentation tốt nhất cho tiếng Việt,
được dùng trong SOTA paper (Huynh et al. 2022).
Nhưng nó cần Java 8+ và download model ~200MB → không khả dụng.

underthesea là tool Python thuần túy, không cần Java.
Chất lượng segmentation kém hơn VnCoreNLP ~1-2% F1.

→ Đây là 1 trong những lý do gap với SOTA. Phải ghi rõ trong báo cáo.

---

## 2.6 Tổng kết Tuần 1 — Những con số quan trọng

```
Dataset:
  Train: 3,000 reviews × 34 aspects = 102,000 ô nhãn
  Dev:   2,000 reviews
  Test:    600 reviews

Phân phối nhãn (train):
  absent (0):   85.3%  → model dễ bị bias predict absent
  positive (1): 11.8%  → phổ biến thứ 2
  negative (2):  2.4%  → ít hơn positive ~5 lần
  neutral (3):   0.5%  → cực hiếm, model khó học

Global class weights (trước clip):
  absent: 1.00, positive: 8.61, negative: 27.96, neutral: 154.48
Sau clip tại 10.0:
  absent: 1.00, positive: 8.61, negative: 10.00, neutral: 10.00

Độ dài review:
  p99 train = 243 words → MAX_SEQ_LEN = 384

Aspects đặc biệt:
  SERVICE#GENERAL: phổ biến nhất (63%), dễ học nhất
  ROOM_AMENITIES#PRICES: 0 mẫu trong train → model không học được
  RARE_ASPECTS (8 aspects): presence rate < 5%

Files quan trọng đã tạo:
  outputs/eda/class_weights.json  → tuần 2 dùng
  outputs/eda/encoder_config.json → tuần 2 dùng
  data/*_preprocessed.csv        → tuần 2 và 3 dùng
```

---

---

# CHƯƠNG 3 — TUẦN 2: TRAIN PHOBERT

---

## 3.1 PhoBERT là gì?

PhoBERT là một "mô hình ngôn ngữ lớn" cho tiếng Việt, được tạo bởi VinAI Research.

**Cách PhoBERT được tạo ra (pre-training):**
VinAI lấy 20GB văn bản tiếng Việt (báo chí, Wikipedia, sách...) và train một
mô hình học cách "hiểu" ngôn ngữ. Quá trình này mất hàng tuần trên hàng chục GPU.

Kết quả: PhoBERT "biết" tiếng Việt — nó có thể:
- Hiểu "nhân_viên" là người phục vụ
- Biết "sạch" và "bẩn" là trái nghĩa
- Nhận biết ngữ cảnh: "tệ" trong "dịch vụ tệ" khác với "tệ" trong "tệ thật"

**Fine-tuning là gì?**
PhoBERT gốc chỉ "hiểu" ngôn ngữ, chưa biết làm ABSA.
Fine-tuning = dạy thêm cho PhoBERT bài toán cụ thể của mình (ABSA với 3000 reviews).

Giống như thuê chuyên gia ngôn ngữ học tiếng Việt (PhoBERT),
rồi đào tạo thêm cho họ về lĩnh vực khách sạn.

---

## 3.2 Kiến trúc model chi tiết

```
BƯỚC 1: Tokenization
  "Phòng rộng, nhân_viên thân_thiện"
      ↓ PhoBERT tokenizer
  [CLS] Phòng rộng , nhân_viên thân_thiện [SEP] [PAD] [PAD] ...
     ↓
  input_ids = [0, 234, 567, 12, 891, 456, 2, 1, 1, ...]  (số hóa)
  attention_mask = [1, 1, 1, 1, 1, 1, 1, 0, 0, ...]  (1=từ thật, 0=padding)
  Shape: [batch_size, 384]

BƯỚC 2: PhoBERT Encoder
  input_ids + attention_mask
      ↓ 12 transformer layers
  hidden_states = 13 tensors, mỗi tensor shape [batch, 384, 768]
  (layer 0 = embedding, layer 1-12 = transformer layers)

BƯỚC 3: Lấy [CLS] representation
  Token [CLS] ở vị trí 0 encode thông tin TÒan bộ câu
  concat(hidden[-4][:,0,:],   ← layer thứ 9
         hidden[-3][:,0,:],   ← layer thứ 10
         hidden[-2][:,0,:],   ← layer thứ 11
         hidden[-1][:,0,:])   ← layer thứ 12
  → Shape [batch, 768×4] = [batch, 3072]

BƯỚC 4: 34 Classification Heads
  [batch, 3072] → Dropout(0.2) → 34 × Linear(3072, 4)
  Mỗi Linear layer cho ra [batch, 4] = xác suất của 4 nhãn
  → 34 tensors shape [batch, 4]

BƯỚC 5: Loss
  Với mỗi trong 34 aspects:
    CrossEntropyLoss(prediction, ground_truth, weight=per_aspect_weight)
  Cộng tất cả 34 losses lại / 34 = mean loss
```

**Tại sao dùng 4 layers cuối thay vì chỉ layer cuối?**

Mỗi layer của BERT/PhoBERT học thông tin khác nhau:
```
Layer 1-4:  Cú pháp (syntax) — "từ này là danh từ hay động từ?"
Layer 5-8:  Ngữ nghĩa cơ bản — "từ này có nghĩa gì?"
Layer 9-12: Ngữ nghĩa phức tạp — "câu này mang hàm ý gì?"
```

Khi concat 4 layers cuối, model có cả thông tin ngữ nghĩa cơ bản
lẫn phức tạp → phân loại aspect tốt hơn ~1-2% F1.
Đây là kỹ thuật từ bài báo BERT gốc (Devlin et al. 2019).

---

## 3.3 Các hyperparameters và lý do

**Learning Rate = 2e-5 (= 0.00002)**

Learning rate = "tốc độ học" = mỗi bước cập nhật tham số bao nhiêu?

Quá cao (0.01): model "nhảy" quá mạnh, không hội tụ, loss dao động loạn
Quá thấp (0.000001): model học quá chậm, cần hàng trăm epochs
2e-5: giá trị chuẩn cho fine-tuning BERT-based models, được validate qua nhiều paper

**Warmup Ratio = 0.1 (10% steps đầu)**

Vấn đề: Khi bắt đầu fine-tuning, PhoBERT đang ở trạng thái "đã học tốt tiếng Việt".
Nếu dùng full learning rate ngay, có thể làm hỏng kiến thức đã có.

Giải pháp warmup: bắt đầu với LR rất nhỏ (gần 0), tăng dần lên 2e-5 trong 10%
steps đầu, sau đó giảm dần về 0 theo linear schedule.

```
LR
2e-5 |        /\
     |       /  \_______________
     |      /
     |_____/
     0%  10%                 100%   steps
       warmup    decay
```

**Batch Size = 8 (sau khi giảm từ 16)**

Batch size = số reviews xử lý cùng lúc trước khi cập nhật tham số.

Batch=16 với MAX_SEQ_LEN=384 trên T4 16GB:
```
Mỗi tensor float32 = 4 bytes
1 input: 384 tokens × 768 dim × 12 layers = ~14MB
Batch 16: 14MB × 16 = 224MB chỉ cho 1 layer
Cộng gradient + optimizer state = ~15-16GB → OOM (Out of Memory)
```

Batch=8: dùng ~8GB → an toàn cho T4 16GB.
Để effective batch vẫn = 16, dùng gradient accumulation = 2
(accumulate gradient của 2 batches nhỏ trước khi update).

**Early Stopping Patience = 3**

Nếu Combined F1 trên dev set không cải thiện trong 3 epochs liên tiếp → dừng.

Tại sao cần early stopping?
```
Epoch  1: Train loss 0.40, Dev F1 0.55  ← đang học tốt
Epoch  5: Train loss 0.25, Dev F1 0.68  ← đang học tốt
Epoch  8: Train loss 0.15, Dev F1 0.71  ← best!
Epoch  9: Train loss 0.12, Dev F1 0.70  ← giảm nhẹ (patience 1)
Epoch 10: Train loss 0.10, Dev F1 0.70  ← không đổi (patience 2)
Epoch 11: Train loss 0.09, Dev F1 0.69  ← giảm (patience 3 → STOP)
```

Nếu tiếp tục train: train loss tiếp tục giảm nhưng dev F1 không tăng.
Hiện tượng này gọi là OVERFITTING — model bắt đầu "học thuộc" train set
thay vì học "pattern chung" có thể áp dụng cho data mới.

Early stopping lấy checkpoint tốt nhất (epoch 8 trong ví dụ trên).

**Gradient Clipping = 1.0**

Sau mỗi backward pass, clip norm của tất cả gradients về tối đa 1.0.
Tránh gradient explosion (cùng vấn đề với weight quá cao ở trên).

---

## 3.4 Đọc Learning Curve — Điều cần giải thích trong báo cáo

Learning curve là biểu đồ bắt buộc theo yêu cầu của thầy.
Sau khi training xong, bạn phải ĐỌC và GIẢI THÍCH được biểu đồ.

**Biểu đồ 1 — Loss theo epoch:**

```
Loss
0.45 |*
0.40 |  *         (train loss)
0.35 |    *
0.30 |      * * (dev loss bắt đầu plateau)
0.25 |        * *
0.20 |    * * * *   (dev loss bắt đầu tăng ← OVERFIT)
0.15 |  * * *
     ←────────────────→ epochs
     1  2  3  4  5  6  7  8  9  10
              ↑ Best epoch
```

Điều cần giải thích:
- "Train loss giảm đều → model đang học"
- "Dev loss giảm đến epoch X rồi tăng lại → model bắt đầu overfit từ epoch X+1"
- "Early stopping dừng tại epoch X+3 → lấy checkpoint epoch X"

**Biểu đồ 2 — F1 theo epoch:**

```
F1
0.75|              (SOTA: đường nằm ngang màu đỏ)
0.70|          * *
0.65|        *     * (bắt đầu plateau/giảm)
0.60|      *
0.55|    *
0.50|  *
0.45|*
     ←────────────→ epochs
     1  2  3  4  5  6  7
```

Điều cần giải thích:
- "F1 tăng đều trong các epoch đầu → model đang cải thiện"
- "F1 plateau/giảm từ epoch X → overfit bắt đầu"
- "Khoảng cách với SOTA (0.7732) là do [liệt kê nguyên nhân]"

---

## 3.5 Kết quả kỳ vọng và giải thích gap

**Kỳ vọng thực tế:**

| Metric | Kỳ vọng | SOTA | Gap |
|--------|---------|------|-----|
| ACD F1 | 0.70-0.76 | 0.8255 | 6-12% |
| SPC F1 | 0.62-0.68 | — | — |
| Combined F1 | 0.66-0.72 | 0.7732 | 5-11% |

Nếu kết quả của bạn trong khoảng này → BÌNH THƯỜNG, không phải lỗi.

**5 lý do giải thích gap với SOTA:**

**Lý do 1 — underthesea thay VnCoreNLP: ~1-2% gap**
SOTA paper dùng VnCoreNLP cho word segmentation.
Chúng ta dùng underthesea (fallback).
underthesea kém chính xác hơn trong phân tách từ ghép phức tạp.

**Lý do 2 — ROOM_AMENITIES#PRICES = 0 mẫu: ~0.5-1% gap**
Aspect này không có mẫu trong train → ACD F1 = 0 cho aspect này.
Khi tính Macro-F1 trung bình 33 aspects còn lại, 1 con số 0 kéo
trung bình xuống. Chúng ta exclude aspect này khỏi tính toán
nhưng vẫn ảnh hưởng một phần.

**Lý do 3 — Neutral cực hiếm (0.5%): ~1-2% gap SPC**
Weight neutral bị clip từ 154 xuống 10 → model vẫn không học tốt neutral.
SPC F1 cho nhãn neutral sẽ thấp ở mọi aspect, kéo SPC macro-F1 xuống.

**Lý do 4 — Dataset nhỏ: ~1-2% gap**
3000 reviews là dataset nhỏ cho bài toán 34-class.
SOTA paper có thể đã dùng thêm data augmentation hoặc train lâu hơn.

**Lý do 5 — Chỉ 1 run: ~0.5% gap**
Nếu train nhiều lần với seeds khác nhau rồi ensemble (kết hợp),
kết quả sẽ ổn định và tốt hơn. Chúng ta chỉ chạy 1 lần.

**Tổng ước tính gap có thể giải thích: 4-8% Combined F1**
Điều này phù hợp với gap thực tế kỳ vọng (~5-11%).

---

---

# CHƯƠNG 4 — TUẦN 3: LLM FEW-SHOT VÀ RAG

---

## 4.1 Ý tưởng cốt lõi

**Câu hỏi đặt ra:**
> "Chúng ta mất bao nhiêu thời gian để train PhoBERT?"
> "GPT/Gemini có thể làm được bài này mà không cần train gì không?"

Tuần 3 trả lời câu hỏi này bằng thực nghiệm.

**In-Context Learning (ICL) là gì?**

Thay vì train model, ta cho model xem vài ví dụ NGAY TRONG PROMPT:

```
[System] Bạn là chuyên gia phân tích review khách sạn tiếng Việt.
         Đây là 34 aspects hợp lệ: ...

[Example 1]
Review: "Phòng sạch, dịch vụ tuyệt vời"
Output: {"ROOMS#CLEANLINESS": "positive", "SERVICE#GENERAL": "positive"}

[Example 2]
Review: "Nhân viên thái độ kém, giá đắt"
Output: {"SERVICE#GENERAL": "negative", "HOTEL#PRICES": "negative"}

[Test]
Review: "Khách sạn đẹp nhưng hơi xa trung tâm"
Output: ???
```

GPT/Gemini đọc ví dụ và tự suy luận → không cần train.

**k là gì?**
k = số examples trong prompt. Ta thử k=2, k=4, k=8.
Nhiều examples hơn → context nhiều hơn → F1 thường cao hơn (đến mức nào đó).

**RAG là gì?**

RAG = Retrieval-Augmented Generation.

Vấn đề với ICL ngẫu nhiên: nếu review test là về HOTEL#PRICES nhưng
2 examples được chọn ngẫu nhiên lại về ROOMS#COMFORT và SERVICE#GENERAL,
model có ít context liên quan → F1 thấp hơn.

Giải pháp RAG: chọn examples THÔNG MINH hơn bằng cách tìm reviews
GIỐNG NHẤT với review cần phân tích.

```
Cách hoạt động:
1. Embed tất cả 3000 train reviews thành vector (dùng vietnamese-sbert)
2. Embed review test thành vector
3. Tìm k train reviews có vector gần nhất (cosine similarity)
4. Dùng k reviews đó làm examples trong prompt

Ví dụ:
  Review test: "Bữa sáng ngon, phòng ăn thoáng"
  → Tìm 4 train reviews tương tự về FOOD&DRINKS → examples liên quan hơn
  → F1 cho FOOD&DRINKS aspects cao hơn
```

**So sánh fair:**
Tầng 2 (ICL random) vs Tầng 3 (RAG):
- Cùng LLM (GPT-4o-mini)
- Cùng k (ví dụ k=4)
- Cùng prompt format
- Chỉ KHÁC cách chọn examples

Mọi cải thiện F1 từ Tầng 2 sang Tầng 3 = đóng góp của retrieval.

---

## 4.2 Chain-of-Thought (CoT) là gì?

Nếu chỉ hỏi "Output JSON cho review này?", GPT có thể đưa ra kết quả
mà không giải thích → khó debug khi sai.

Chain-of-Thought yêu cầu model suy nghĩ từng bước trước khi trả lời:

```
Review: "Phòng rộng nhưng điều hòa hơi ồn"

Phân tích:
- "Phòng rộng" → đề cập đến ROOMS#COMFORT → positive
- "điều hòa hơi ồn" → đề cập đến ROOM_AMENITIES#GENERAL → negative
- Không đề cập đến SERVICE#GENERAL, HOTEL#PRICES, v.v.

Output: {"ROOMS#COMFORT": "positive", "ROOM_AMENITIES#GENERAL": "negative"}
```

CoT giúp:
- Model ít bịa đặt aspects (ít hallucination)
- Dễ debug khi kết quả sai
- F1 thường cao hơn ~2-5%

---

## 4.3 Kỳ vọng kết quả

**Kỳ vọng Combined F1:**

| Phương pháp | Kỳ vọng | Lý do |
|-------------|---------|-------|
| GPT-4o-mini ICL k=2 | ~0.55-0.62 | Ít ví dụ, random |
| GPT-4o-mini ICL k=4 | ~0.58-0.65 | Thêm ví dụ |
| GPT-4o-mini ICL k=8 | ~0.60-0.68 | Nhiều ví dụ nhất |
| GPT-4o-mini RAG k=4 | ~0.62-0.70 | Examples liên quan hơn |
| Gemini 1.5 Flash | ~0.03-0.05 thấp hơn GPT | Model yếu hơn GPT-4o-mini |
| PhoBERT (Tầng 1) | ~0.66-0.72 | Có 3000 training examples |

**Tại sao LLM thường kém hơn PhoBERT?**

```
PhoBERT:
  Đã học từ 3000 examples trong domain khách sạn tiếng Việt
  Biết chính xác 34 aspects của bài toán
  Fine-tuned trực tiếp cho bài toán này

GPT/Gemini few-shot:
  Chỉ thấy 2-8 examples trong prompt
  Không được train chuyên biệt cho task này
  Phải suy luận từ ví dụ ít
  Có thể bịa đặt aspects không có trong 34 danh sách
```

Điều thú vị: GPT/Gemini không train gì cả mà vẫn đạt ~60-68% —
điều này cho thấy LLM lớn có khả năng generalization mạnh.

**Câu hỏi nghiên cứu cần trả lời trong báo cáo:**

1. RAG tốt hơn ICL bao nhiêu % với cùng k?
   → Kỳ vọng: +1-3%, vì examples liên quan hơn
   → Nếu không cải thiện: embedding model chưa đủ tốt? Aspect-aware selection chưa hiệu quả?

2. k tăng từ 2→4→8 thì F1 tăng thế nào?
   → Kỳ vọng: tăng từ k=2 đến k=4/8, nhưng có thể plateau
   → Lý do plateau: context window của LLM có giới hạn, quá nhiều ví dụ = ít attention cho từng cái

3. GPT-4o-mini vs Gemini 1.5 Flash — cái nào tốt hơn bao nhiêu?
   → Kỳ vọng: GPT tốt hơn ~3-5% vì lớn hơn, follow instruction tốt hơn

4. PhoBERT supervised vs LLM few-shot — khi nào dùng cái nào?
   → PhoBERT: tốt hơn khi có labeled data, cần accuracy cao, deploy production
   → LLM: tốt hơn khi không có labeled data, cần adapt nhanh, domain mới

---

---

# CHƯƠNG 5 — CÁC CON SỐ QUAN TRỌNG CẦN NHỚ

---

## 5.1 F1 Metric — Giải thích đầy đủ

**Tại sao không dùng Accuracy?**

```
Dataset: 102,000 ô nhãn, 85% là absent (0)

Model ngốc — luôn predict absent:
  Đúng: 87,000 ô (các ô absent)
  Sai:  15,000 ô (các ô positive/negative/neutral)
  Accuracy = 87,000 / 102,000 = 85.3%  ← trông cao!
  Nhưng model không phát hiện được bất kỳ aspect nào → vô dụng
```

**Precision, Recall, F1:**

```
Precision = TP / (TP + FP)
  TP = True Positive: model nói "có aspect", đúng
  FP = False Positive: model nói "có aspect", nhưng thực ra không có

Recall = TP / (TP + FN)
  FN = False Negative: model nói "không có aspect", nhưng thực ra có

F1 = 2 × Precision × Recall / (Precision + Recall)
```

Ví dụ cụ thể:
```
100 reviews có ROOMS#CLEANLINESS = positive (ground truth)
Model predict:
  70 reviews: đúng là positive → TP = 70
  10 reviews: predict positive nhưng thật ra absent → FP = 10
  30 reviews: predict absent nhưng thật ra positive → FN = 30

Precision = 70 / (70+10) = 87.5%
Recall    = 70 / (70+30) = 70.0%
F1        = 2 × 0.875 × 0.7 / (0.875+0.7) = 77.8%
```

**ACD F1 vs SPC F1:**

ACD F1: chỉ quan tâm có aspect không (binary)
```
Ground truth: [0, 1, 0, 2, 3, 0, ...]  (0=absent, 1/2/3=present)
→ chuyển về binary: [0, 1, 0, 1, 1, 0, ...]
Model predict:      [0, 1, 0, 0, 1, 0, ...]
→ tính F1 binary (aspect detected hay không)
```

SPC F1: chỉ tính trên các reviews thực sự có aspect
```
Chỉ lấy rows mà ground truth != 0:
  Ground truth: [1, 2, 3, 1, 2, ...]  (chỉ positive/negative/neutral)
  Model predict: [1, 2, 1, 1, 2, ...]
→ tính F1 3-class (positive vs negative vs neutral)
```

Combined F1 = (ACD F1 + SPC F1) / 2

---

## 5.2 Bảng tóm tắt toàn dự án

```
TUẦN 1 — DATA FOUNDATION (DONE)
─────────────────────────────────────────────────────────────────
Làm gì: EDA + Preprocessing + DataLoader + Eval script
Phát hiện quan trọng:
  MAX_SEQ_LEN: 384 (không phải 256)
  Weight neutral: 154 → clip tại 10.0
  ROOM_AMENITIES#PRICES: 0 mẫu train
  Segmenter: underthesea (không phải VnCoreNLP)
Output: class_weights.json, encoder_config.json, *_preprocessed.csv
Chạy ở đâu: Máy Mac (không cần GPU)

TUẦN 2 — PHOBERT TRAINING (Đang chạy)
─────────────────────────────────────────────────────────────────
Làm gì: Fine-tune PhoBERT, 34 classification heads song song
Kiến trúc: concat 4 layers cuối → 3072 dim → 34 × Linear(3072, 4)
Hyperparams: LR=2e-5, batch=8, grad_accum=2, warmup=10%, clip=10.0
Kỳ vọng: Combined F1 ~0.66-0.72
BẮT BUỘC: Learning curve (train/val loss + F1 theo epoch)
BẮT BUỘC: Ablation concat_4_layers vs cls_only
Chạy ở đâu: Google Colab / Kaggle (T4 GPU)

TUẦN 3 — LLM + RAG (Chờ)
─────────────────────────────────────────────────────────────────
Làm gì: GPT/Gemini few-shot (Tầng 2) + RAG retrieval (Tầng 3)
Ablation: k=2,4,8 × GPT+Gemini × ICL+RAG = 12 experiments
Kỳ vọng: Combined F1 ~0.58-0.70 (thấp hơn PhoBERT)
BẮT BUỘC: Biểu đồ ICL vs RAG theo k
BẮT BUỘC: Trả lời "RAG có tốt hơn ICL không và tại sao?"
Chạy ở đâu: Máy Mac (CPU đủ, chỉ gọi API)

TUẦN 4 — BÁO CÁO (Chờ)
─────────────────────────────────────────────────────────────────
Tổng hợp kết quả 3 tuần
Bảng so sánh 3 tầng + SOTA
Phân tích lỗi (error analysis)
Giải thích gap với SOTA
Deadline: 21/04/2026
```

---

## 5.3 Checklist tối thiểu phải có trong báo cáo

Đây là những thứ thầy sẽ hỏi/chấm:

- [ ] Mô tả bài toán ABSA (ACD + SPC là gì, 34 aspects)
- [ ] Phân tích EDA (distribution, class imbalance, độ dài review)
- [ ] Giải thích lý do mỗi bước preprocessing
- [ ] Kiến trúc PhoBERT multi-task (vẽ sơ đồ)
- [ ] Learning curve với giải thích overfitting xảy ra ở epoch nào
- [ ] Bảng ablation concat_4_layers vs cls_only
- [ ] Giải thích tại sao dùng F1 thay accuracy
- [ ] Giải thích class imbalance và cách xử lý (weighted loss + clip)
- [ ] Bảng so sánh 3 tầng + SOTA với số F1 thực tế
- [ ] Phân tích gap với SOTA (ít nhất 3 nguyên nhân cụ thể)
- [ ] Kết luận RAG có hiệu quả không so với ICL
- [ ] Per-aspect F1 — aspect nào model làm tốt/kém và tại sao

---

*Tài liệu này giải thích toàn bộ dự án từ góc độ người mới học NLP*
*Cập nhật sau khi có kết quả tuần 2 và 3*
