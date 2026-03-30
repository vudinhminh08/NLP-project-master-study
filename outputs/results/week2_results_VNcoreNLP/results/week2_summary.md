# Kết quả Tuần 2 — PhoBERT Multi-task

## Config thực tế
| Tham số | Giá trị |
|---------|---------|
| Encoder | concat_4_layers |
| MAX_SEQ_LEN | 256 |
| Batch size | 16 × 1 = 16 (effective) |
| Learning rate | 0.0001 |
| Warmup | 15% steps |
| Weight clip | 10.0 |
| Best epoch | 7 / 14 |

## Kết quả

| Split | ACD F1 | SPC F1 | Combined F1 |
|-------|--------|--------|-------------|
| Dev   | 0.5624 | 0.4643 | 0.5134 |
| **Test**  | **0.5836** | **0.4658** | **0.5247** |
| SOTA (Huynh 2022) | 0.8255 | — | 0.7732 |

## Phân tích Gap so với SOTA

- **ACD F1 gap:** 0.2419 (24.2%)
- **Combined F1 gap:** 0.2485 (24.8%)

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
| FOOD&DRINKS#MISCELLANEOUS | 0.0000 | 0.0000 | 3 | rare |
| ROOMS#MISCELLANEOUS | 0.0000 | 0.0000 | 4 | rare |
| ROOM_AMENITIES#MISCELLANEOUS | 0.0000 | 0.0000 | 3 |  |
| ROOM_AMENITIES#PRICES | 0.0000 | 0.0000 | 1 | 0 train samples |

## Learning Curve
Xem: `outputs/eda/learning_curve.png`

Early stopping tại epoch **7**.
Dev loss bắt đầu tăng trong khi train loss vẫn giảm → dấu hiệu overfitting.