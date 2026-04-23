# EDA Summary Report

Tài liệu này tóm tắt các điểm EDA dùng trong báo cáo v10 cho VLSP 2018 Hotel.

## Dataset

| Split | Số mẫu | Vai trò |
|---|---:|---|
| Train | 3,000 | Huấn luyện model và tính class weights |
| Dev | 2,000 | Chọn checkpoint, threshold tuning |
| Test | 600 | Đánh giá cuối cùng |

Mỗi review có 34 aspect columns. Mỗi aspect nhận một trong bốn nhãn:

```text
0 = absent
1 = positive
2 = negative
3 = neutral
```

## Mất cân bằng nhãn

Dữ liệu mất cân bằng mạnh ở cả hai chiều:

- Theo class: `absent` chiếm đa số tuyệt đối.
- Theo aspect: một vài aspect như `HOTEL#GENERAL`, `ROOMS#COMFORT`,
  `SERVICE#GENERAL` phổ biến hơn nhiều so với các aspect miscellaneous/prices.
- `ROOM_AMENITIES#PRICES` gần như không có supervision trong train, nên F1 thấp
  ở aspect này là hợp lý.

Hệ quả thiết kế:

- Cần `class_weight` và `weight_clip` để loss không bị class absent chi phối.
- Cần rare oversampling và rare loss multipliers để macro F1 không bỏ rơi aspect
  hiếm.
- Cần split ACD/SPC để sentiment classifier không bị lớp absent áp đảo.

## Độ dài review

Phần lớn review ngắn đến trung bình, nhưng vẫn có outlier dài. `max_seq_len=256`
được chọn vì:

- đủ phủ phần lớn review thực tế;
- phù hợp giới hạn VRAM trên Kaggle T4;
- giữ chi phí train/inference ổn định.

## Preprocessing

Pipeline chính:

1. Chuẩn hóa Unicode và whitespace.
2. Giữ lại dấu câu cơ bản vì có thể mang tín hiệu sentiment.
3. Word segmentation bằng VnCoreNLP.
4. Tokenize bằng PhoBERT tokenizer.
5. Tạo `attention_mask` để loại padding ở encoder và attention pooling.

Kết quả ablation xác nhận VnCoreNLP là bước quan trọng nhất trong preprocessing:
`cls_only` tăng từ Combined F1 `0.2681` lên `0.6218`.

## Ý nghĩa cho báo cáo

EDA giải thích vì sao report chọn kiến trúc PhoBERT v3 gồm VnCoreNLP,
Aspect-Aware Attention Pooling, Dual Head, weighted/focal loss và rare aspect
handling. Các quyết định này đều xuất phát từ đặc điểm dữ liệu, không phải chọn
tham số tùy ý.
