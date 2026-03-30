# Kết quả Tuần 2 — PhoBERT Multi-task

## Config thực tế
| Tham số | Giá trị |
|---------|---------|
| Encoder | cls_only |
| MAX_SEQ_LEN | 256 |
| Batch size | 8 × 2 = 16 (effective) |
| Learning rate | 2e-05 |
| Warmup | 10% steps |
| Weight clip | 10.0 |
| Best epoch | 14 / 17 |

## Kết quả

| Split | ACD F1 | SPC F1 | Combined F1 |
|-------|--------|--------|-------------|
| Dev   | 0.3155 | 0.2004 | 0.2580 |
| **Test**  | **0.3349** | **0.2014** | **0.2681** |
| SOTA (Huynh 2022) | 0.8255 | — | 0.7732 |

## Phân tích Gap so với SOTA

- **ACD F1 gap:** 0.4906 (49.1%)
- **Combined F1 gap:** 0.5051 (50.5%)

### Nguyên nhân gap (phân tích):
1. **underthesea vs VnCoreNLP:** Dùng underthesea làm fallback → ~1-2% F1 loss
2. **ROOM_AMENITIES#PRICES:** 0 training samples → ACD F1 = 0 cho aspect này
3. **Neutral cực hiếm (weight=154→clip=10):** SPC F1 cho neutral thấp
4. **Dataset nhỏ (3000 train):** SOTA có thể dùng data augmentation
5. **Single run:** Chưa ensemble nhiều seeds

## Bottom 5 Aspects (ACD F1 thấp nhất — Test set)

| Aspect | ACD F1 | SPC F1 | Support | Ghi chú |
|--------|--------|--------|---------|---------|
| FACILITIES#COMFORT | 0.0000 | 0.0000 | 26 |  |
| FACILITIES#MISCELLANEOUS | 0.0000 | 0.0000 | 8 |  |
| FACILITIES#PRICES | 0.0000 | 0.0000 | 13 |  |
| FOOD&DRINKS#MISCELLANEOUS | 0.0000 | 0.0000 | 3 | rare |
| FOOD&DRINKS#PRICES | 0.0000 | 0.0000 | 9 |  |

## Learning Curve
Xem: `outputs/eda/learning_curve.png`

Early stopping tại epoch **14**.
Dev loss bắt đầu tăng trong khi train loss vẫn giảm → dấu hiệu overfitting.