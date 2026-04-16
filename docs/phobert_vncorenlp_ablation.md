# PhoBERT: VnCoreNLP vs No VnCoreNLP

This comparison uses two executed notebooks with real cell outputs:

| Setting | Notebook | Encoder | ACD F1 | SPC F1 | Combined F1 |
|---|---|---|---:|---:|---:|
| No VnCoreNLP preprocessing | `notebooks/phase_phobert_no_vncorenlp_executed.ipynb` | concat_4_layers | 0.3592 | 0.2371 | 0.2981 |
| No VnCoreNLP preprocessing | `notebooks/phase_phobert_no_vncorenlp_executed.ipynb` | cls_only | 0.3349 | 0.2014 | 0.2681 |
| VnCoreNLP preprocessing | `notebooks/phase_phobert_vncorenlp_executed.ipynb` | concat_4_layers | 0.5836 | 0.4658 | 0.5247 |
| VnCoreNLP preprocessing | `notebooks/phase_phobert_vncorenlp_executed.ipynb` | cls_only | 0.6360 | 0.4727 | 0.5543 |

## Main Comparison

Best no-VnCoreNLP run:

```text
concat_4_layers Combined F1 = 0.2981
```

Best VnCoreNLP run:

```text
cls_only Combined F1 = 0.5543
```

Absolute gain:

```text
0.5543 - 0.2981 = +0.2562 Combined F1
```

Relative to the no-VnCoreNLP run:

```text
0.2562 / 0.2981 ≈ +85.9%
```

## Giải thích tham số so sánh

| Tham số | Cách giữ cố định | Lý do |
|---------|------------------|-------|
| Dataset split | Cùng train/dev/test của VLSP 2018 Hotel | Đảm bảo chênh lệch đến từ preprocessing, không phải đổi dữ liệu. |
| Metric | Cùng ACD F1, SPC F1, Combined F1 | So sánh trực tiếp với SVM, PhoBERT và LLM/RAG. |
| Model family | Đều dùng PhoBERT supervised | Không trộn thêm model khác để tránh nhiễu kết luận. |
| Encoder candidates | So sánh `concat_4_layers` và `cls_only` trong cả hai setting | Kiểm tra xem lợi ích của VnCoreNLP có ổn định với các cách lấy representation khác nhau không. |
| Word segmentation | Chỉ thay phần có/không có VnCoreNLP | Đây là biến chính của ablation. |

Ablation này được thiết kế theo nguyên tắc chỉ thay một yếu tố quan trọng:
chất lượng word segmentation. Vì vậy phần tăng điểm có thể giải thích rõ ràng
hơn so với các thí nghiệm thay nhiều thứ cùng lúc.

## Evidence From Notebook Outputs

The no-VnCoreNLP executed notebook shows a processed sample like:

```text
Rộng_rãi_khách_sạn_mới_nhưng_rất_vắng_._Các_dịch_vụ_chất_lượng_chưa_cao_và_thiếu
```

The VnCoreNLP executed notebook shows a processed sample like:

```text
Rộng_rãi khách_sạn mới nhưng rất vắng . Các dịch_vụ chất_lượng chưa cao và thiếu
```

This difference matters because PhoBERT expects Vietnamese word segmentation:
compound words such as `khách_sạn`, `dịch_vụ`, `chất_lượng` should be joined,
but the whole sentence should not be collapsed into one long underscore chain.

## Code và notebook liên quan

Hai notebook executed được dùng để đảm bảo so sánh có output thật:

- `notebooks/phase_phobert_no_vncorenlp_executed.ipynb`
- `notebooks/phase_phobert_vncorenlp_executed.ipynb`

Logic preprocessing nằm trong `code/data_processing/step3_preprocessing.py`.
Điểm khác biệt quan trọng là bước word segmentation:

- Bản không VnCoreNLP tạo chuỗi có xu hướng over-join bằng dấu `_`.
- Bản VnCoreNLP chỉ nối các từ ghép tiếng Việt, giữ khoảng trắng giữa các từ.

Sau preprocessing, hai notebook đều chạy cùng hướng PhoBERT supervised và
evaluate bằng cùng metric. Nhờ vậy, chênh lệch điểm có thể được giải thích chủ
yếu bởi chất lượng word segmentation, không phải do đổi bài toán hay đổi metric.

## Report Takeaway

VnCoreNLP-style word segmentation is one of the largest practical improvements
in the project. It raises PhoBERT from a weak baseline-level result to the main
supervised model used in the final report.

## Why VnCoreNLP Helps

Vietnamese does not use spaces to mark word boundaries in the same way English
does. A space in Vietnamese often separates syllables, not complete lexical
words. For example:

```text
khách sạn -> khách_sạn
dịch vụ -> dịch_vụ
chất lượng -> chất_lượng
nhân viên -> nhân_viên
```

PhoBERT was pretrained on Vietnamese text that had already been word-segmented
in this style. Therefore, fine-tuning PhoBERT on similarly segmented hotel
reviews makes the downstream input closer to the distribution PhoBERT saw during
pretraining.

### What VnCoreNLP Does In This Project

VnCoreNLP improves the input representation in three concrete ways:

1. It joins multi-syllable Vietnamese words into one lexical unit.

   Example:

   ```text
   khách sạn -> khách_sạn
   ```

   This helps PhoBERT understand that `khách_sạn` is one concept rather than two
   unrelated syllables.

2. It preserves sentence structure.

   The no-VnCoreNLP run produced text like:

   ```text
   Rộng_rãi_khách_sạn_mới_nhưng_rất_vắng_._Các_dịch_vụ...
   ```

   This over-joins the sentence into long underscore chains, which creates
   unnatural tokens and makes the review harder for PhoBERT to interpret.

   The VnCoreNLP run produced:

   ```text
   Rộng_rãi khách_sạn mới nhưng rất vắng . Các dịch_vụ...
   ```

   This keeps word-level units while preserving normal spacing between words.

3. It improves aspect and sentiment cues.

   ABSA depends heavily on local phrases such as:

   ```text
   phòng sạch
   nhân viên thân thiện
   vị trí thuận tiện
   giá hơi cao
   ```

   With better word segmentation, PhoBERT can represent these phrases more
   consistently. This helps both:

   - ACD: detecting whether an aspect is mentioned.
   - SPC: deciding whether the mentioned aspect is positive, negative, or neutral.

### Why The Score Improves So Much

The score improvement is large because the task is sensitive to small lexical
cues. Many aspect labels are triggered by short phrases:

| Aspect | Typical cue |
|---|---|
| `ROOMS#CLEANLINESS` | `phòng sạch`, `phòng bẩn` |
| `SERVICE#GENERAL` | `nhân_viên thân_thiện`, `lễ_tân khó_chịu` |
| `LOCATION#GENERAL` | `vị_trí thuận_tiện`, `gần trung_tâm` |
| `HOTEL#PRICES` | `giá hợp_lý`, `giá cao` |

If these phrases are segmented incorrectly, the model sees noisier token
sequences and has more difficulty learning stable aspect-sentiment patterns from
only 3,000 training reviews. VnCoreNLP reduces this noise, so the supervised
signal becomes much easier for PhoBERT to use.

The result is visible in both subtasks:

```text
ACD F1:      0.3592 -> 0.6360
SPC F1:      0.2371 -> 0.4727
Combined F1: 0.2981 -> 0.5543
```

### Detailed Result Analysis

ACD tăng `+0.2768`, lớn hơn mức tăng SPC `+0.2356`. Điều này cho thấy word
segmentation ảnh hưởng mạnh nhất đến bước phát hiện aspect. Khi các cụm như
`khách_sạn`, `dịch_vụ`, `phòng`, `nhân_viên` được tách đúng, model dễ nhận ra
review đang nói về entity nào hơn.

SPC cũng tăng mạnh vì sentiment cue được gắn với đúng cụm aspect. Ví dụ, trong
review có nhiều mệnh đề như "phòng sạch nhưng nhân viên khó chịu", model cần
hiểu `sạch` thuộc `ROOMS#CLEANLINESS` còn `khó_chịu` thuộc `SERVICE#GENERAL`.
Word segmentation tốt làm local context ổn định hơn, giúp classification head
cho từng aspect nhận tín hiệu rõ hơn.

Kết quả này cũng giải thích vì sao SVM và LLM/RAG không đạt cao: nếu đầu vào và
schema aspect đã khó, mô hình không được fine-tune trực tiếp sẽ dễ bỏ sót hoặc
gán nhầm aspect. PhoBERT chỉ phát huy mạnh khi preprocessing khớp với cách nó
được pretrain.

The ACD gain is especially large because detecting whether an aspect is present
depends directly on recognizing aspect words and phrases. SPC also improves
because sentiment phrases such as `sạch_sẽ`, `thân_thiện`, `khó_chịu`,
`hợp_lý`, and `tệ` become easier to associate with the correct aspect.

### How To Explain This In The Report

The key point is not simply "we used VnCoreNLP". The real contribution is:

```text
We aligned the downstream ABSA input format with the word-segmented format used
during PhoBERT pretraining. This reduced tokenization noise and made Vietnamese
multi-syllable aspect/sentiment expressions easier for the supervised model to
learn.
```

This is why VnCoreNLP should be presented as an important preprocessing
ablation, not just an implementation detail.
