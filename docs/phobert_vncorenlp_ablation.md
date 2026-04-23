# PhoBERT VnCoreNLP Ablation

Mục tiêu của ablation này là kiểm tra tác động của word segmentation khi
fine-tune PhoBERT cho VLSP 2018 Hotel. Đây là một trong những kết quả dễ defend
nhất trong báo cáo: PhoBERT được pretrain trên tiếng Việt đã word-segmented, nên
input fine-tune cũng phải theo cùng quy ước.

## Kết quả Test

| Pipeline | Encoder | ACD F1 | SPC F1 | Combined F1 |
|---|---|---:|---:|---:|
| Không VnCoreNLP | `cls_only` | 0.3349 | 0.2014 | 0.2681 |
| Không VnCoreNLP | `concat_4_layers` | 0.3592 | 0.2371 | 0.2981 |
| Có VnCoreNLP | `concat_4_layers` | 0.6827 | 0.5374 | 0.6101 |
| Có VnCoreNLP | `cls_only` | 0.6849 | 0.5587 | 0.6218 |

So sánh chính:

```text
cls_only: 0.2681 -> 0.6218 = +0.3537 Combined F1
concat_4_layers: 0.2981 -> 0.6101 = +0.3120 Combined F1
```

## Vì sao VnCoreNLP quan trọng

- Tiếng Việt dùng khoảng trắng giữa âm tiết, không luôn tương ứng với ranh giới
  từ. Ví dụ `nhân viên phục vụ` nên trở thành `nhân_viên phục_vụ`.
- PhoBERT được pretrain trên corpus đã word-segmented. Nếu bỏ segmentation,
  tokenizer nhận input khác phân phối pretraining.
- ABSA phụ thuộc vào cụm từ domain như `phòng`, `khách_sạn`, `nhân_viên`,
  `bữa_sáng`, `vị_trí`, nên segmentation đúng giúp cả ACD và SPC.

## Kết luận

VnCoreNLP không phải bước phụ trợ nhỏ; nó là điều kiện gần như bắt buộc để
PhoBERT hoạt động đúng trên tiếng Việt. Kết quả này cũng giải thích vì sao report
v10 nhấn mạnh preprocessing trong kiến trúc PhoBERT Predictor.
