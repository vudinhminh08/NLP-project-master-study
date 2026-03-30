# Kết quả Tuần 2 — PhoBERT Multi-task

## Config thực tế
| Tham số | Giá trị |
|---------|---------|
| Encoder | concat_4_layers |
| MAX_SEQ_LEN | 256 |
| Batch size | 8 × 2 = 16 (effective) |
| Learning rate | 2e-05 |
| Warmup | 10% steps |
| Weight clip | 10.0 |
| Best epoch | 20 / 20 |

## Kết quả

| Split | ACD F1 | SPC F1 | Combined F1 |
|-------|--------|--------|-------------|
| Dev   | 0.3345 | 0.2297 | 0.2821 |
| **Test**  | **0.3592** | **0.2371** | **0.2981** |
| SOTA (Huynh 2022) | 0.8255 | — | 0.7732 |

## Phân tích Gap so với SOTA

- **ACD F1 gap:** 0.4663 (46.6%)
- **Combined F1 gap:** 0.4751 (47.5%)

### Nguyên nhân gap (phân tích):
1. **underthesea vs VnCoreNLP:** Dùng underthesea làm fallback → ~1-2% F1 loss
2. **ROOM_AMENITIES#PRICES:** 0 training samples → ACD F1 = 0 cho aspect này
3. **Neutral cực hiếm (weight=154→clip=10):** SPC F1 cho neutral thấp
4. **Dataset nhỏ (3000 train):** SOTA có thể dùng data augmentation
5. **Single run:** Chưa ensemble nhiều seeds

## Bottom 5 Aspects (ACD F1 thấp nhất — Test set)

| Aspect | ACD F1 | SPC F1 | Support | Ghi chú |
|--------|--------|--------|---------|---------|
| FACILITIES#MISCELLANEOUS | 0.0000 | 0.0000 | 8 |  |
| FACILITIES#PRICES | 0.0000 | 0.0000 | 13 |  |
| FOOD&DRINKS#MISCELLANEOUS | 0.0000 | 0.0000 | 3 | rare |
| ROOMS#MISCELLANEOUS | 0.0000 | 0.0000 | 4 | rare |
| ROOM_AMENITIES#MISCELLANEOUS | 0.0000 | 0.0000 | 3 |  |

## Learning Curve
Xem: `outputs/eda/learning_curve.png`

Early stopping tại epoch **20**.
Dev loss bắt đầu tăng trong khi train loss vẫn giảm → dấu hiệu overfitting.