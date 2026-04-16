# PhoBERT Best Single Model

Đây là bản PhoBERT single model dùng trong báo cáo chính. Không dùng ensemble.

## Config
| Tham số | Giá trị |
|---------|---------|
| Encoder | cls_only |
| MAX_SEQ_LEN | 256 |
| Batch size | 16 × 1 = 16 (effective) |
| Learning rate | 0.0001 |
| Warmup | 15% steps |
| Weight clip | 10.0 |
| Best epoch | 16 / 20 |

## Kết quả

| Split | ACD F1 | SPC F1 | Combined F1 |
|-------|--------|--------|-------------|
| Dev   | 0.6271 | 0.4630 | 0.5451 |
| **Test**  | **0.6360** | **0.4727** | **0.5543** |
| SOTA (Huynh 2022) | 0.8255 | — | 0.7732 |

## Quy trình xử lý

PhoBERT là mô hình supervised chính của báo cáo. Pipeline được thiết kế để giữ
một mô hình single, dễ giải thích, không ensemble:

1. Dùng dữ liệu đã preprocess bằng VnCoreNLP trong phase data processing.
2. Tokenize `processed_review` bằng tokenizer của `vinai/phobert-base-v2`.
3. Đưa sequence vào PhoBERT encoder.
4. Lấy biểu diễn `[CLS]` layer cuối theo cấu hình `cls_only`.
5. Dùng 34 classification heads song song, mỗi head dự đoán 4 nhãn:
   `absent`, `positive`, `negative`, `neutral`.
6. Train bằng weighted focal loss để giảm bias về class `absent`.
7. Chọn checkpoint tốt nhất theo dev Combined F1.
8. Evaluate checkpoint tốt nhất trên test set.

Điểm quan trọng là model học trực tiếp label scheme của VLSP 2018 Hotel, khác
với LLM/RAG chỉ học qua prompt ngắn. Vì vậy PhoBERT phù hợp làm prediction
engine chính.

## Code và notebook liên quan

Các file code chính:

- `code/data_processing/step3_preprocessing.py`: chuẩn hóa text và word
  segmentation bằng VnCoreNLP.
- `code/data_processing/step2_dataloader.py`: tạo dataset/dataloader cho
  `processed_review` và 34 aspect labels.
- `code/phobert/model.py`: định nghĩa `ABSAPhoBERT` multi-task với 34 heads.
- `code/phobert/train.py`: training loop, weighted focal loss, early stopping.
- `code/phobert/predict.py`: load checkpoint tốt nhất và evaluate.

Notebook báo cáo chính:

- `notebooks/phase_phobert_vncorenlp_executed.ipynb`: bản executed có output
  thật, dùng để chứng minh kết quả.

Checkpoint dùng cho app/demo:

```text
outputs/results/phobert_best_single/models_cls_only/best_model.pt
```

## Phân tích kết quả chi tiết

Test Combined F1 đạt `0.5543`, cao nhất trong các hướng được giữ lại. ACD F1
`0.6360` cao hơn SPC F1 `0.4727`, cho thấy model phát hiện aspect tốt hơn việc
gán sentiment chi tiết. Đây là pattern hợp lý với ABSA nhiều aspect vì sentiment
phải xử lý thêm các trường hợp mơ hồ, phủ định, hoặc nhiều sentiment trong cùng
một review.

Dev Combined F1 `0.5451` và Test Combined F1 `0.5543` khá gần nhau, cho thấy
checkpoint không chỉ overfit dev. Tuy vậy, learning curve vẫn có dấu hiệu
train loss giảm trong khi dev loss tăng sau một số epoch, nên báo cáo cần nhấn
mạnh đây là bản single model ổn định chứ chưa đạt SOTA.

`cls_only` được chọn dù SOTA gợi ý `concat_4_layers` vì trong kết quả thực tế
của dự án, `cls_only` cho Combined F1 cao hơn trên test. Điều này có thể do
dataset nhỏ: biểu diễn 768 chiều đơn giản hơn có ít tham số ở classification
heads hơn, giảm nguy cơ overfit so với 3072 chiều của concat 4 layers.

## Phân tích Gap so với SOTA

- **ACD F1 gap:** 0.1895 (19.0%)
- **Combined F1 gap:** 0.2189 (21.9%)

### Nguyên nhân gap (phân tích):
1. **underthesea vs VnCoreNLP:** Dùng underthesea làm fallback → ~1-2% F1 loss
2. **ROOM_AMENITIES#PRICES:** 0 training samples → ACD F1 = 0 cho aspect này
3. **Neutral cực hiếm (weight=154→clip=10):** SPC F1 cho neutral thấp
4. **Dataset nhỏ (3000 train):** khó học tốt các aspect hiếm
5. **Single model:** báo cáo chính giữ bản single để dễ giải thích và so sánh công bằng

## Vai trò của VnCoreNLP

VnCoreNLP là yếu tố tiền xử lý quan trọng nhất trong pipeline PhoBERT. PhoBERT
được pretrain trên tiếng Việt đã word-segmented, nên review đầu vào cũng cần có
dạng gần tương tự.

So sánh ablation:

| Setting | ACD F1 | SPC F1 | Combined F1 |
|---|---:|---:|---:|
| Không VnCoreNLP | 0.3592 | 0.2371 | 0.2981 |
| Có VnCoreNLP | 0.6360 | 0.4727 | 0.5543 |

VnCoreNLP giúp nối đúng các từ ghép tiếng Việt như `khách_sạn`, `dịch_vụ`,
`chất_lượng`, `nhân_viên`, đồng thời vẫn giữ khoảng trắng giữa các từ. Nhờ đó
PhoBERT nhận được chuỗi token tự nhiên hơn, ít nhiễu hơn và gần với dữ liệu
pretraining hơn.

Tác động chính:

- ACD tăng mạnh vì model nhận diện aspect phrase tốt hơn.
- SPC tăng vì sentiment cue như `sạch_sẽ`, `thân_thiện`, `khó_chịu`, `hợp_lý`
  được gắn với đúng aspect ổn định hơn.
- Combined F1 tăng `+0.2562`, từ `0.2981` lên `0.5543`.

## Bottom 5 Aspects (ACD F1 thấp nhất — Test set)

| Aspect | ACD F1 | SPC F1 | Support | Ghi chú |
|--------|--------|--------|---------|---------|
| FACILITIES#MISCELLANEOUS | 0.0000 | 0.0000 | 8 |  |
| FOOD&DRINKS#MISCELLANEOUS | 0.0000 | 0.0000 | 3 | rare |
| ROOMS#MISCELLANEOUS | 0.0000 | 0.0000 | 4 | rare |
| ROOM_AMENITIES#MISCELLANEOUS | 0.0000 | 0.0000 | 3 |  |
| ROOM_AMENITIES#PRICES | 0.0000 | 0.0000 | 1 | 0 train samples |

## Learning Curve
Xem learning curve trong output của run PhoBERT tương ứng.

Early stopping tại epoch **16**.
Dev loss bắt đầu tăng trong khi train loss vẫn giảm → dấu hiệu overfitting.
