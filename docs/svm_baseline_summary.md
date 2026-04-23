# SVM Baseline Summary

Tài liệu này ghi lại baseline truyền thống dùng trong báo cáo v10. SVM không
được đưa vào app; nó chỉ đóng vai trò mốc so sánh để đo giá trị của PhoBERT.

## Kết quả

| Model | Test ACD F1 | Test SPC F1 | Test Combined F1 |
|---|---:|---:|---:|
| SVM + TF-IDF | 0.4074 | 0.2272 | 0.3173 |
| PhoBERT v3 `cls_only` | 0.6849 | 0.5587 | 0.6218 |

PhoBERT v3 cao hơn SVM `+0.3045` Combined F1 tuyệt đối. Điều này cho thấy
TF-IDF + SVM không đủ mạnh cho ABSA tiếng Việt vì feature sparse không nắm tốt
ngữ cảnh, phủ định và cảm xúc theo từng aspect.

## Cấu hình

| Thành phần | Giá trị |
|---|---|
| Feature | TF-IDF unigram + bigram |
| Classifier | `LinearSVC` |
| `max_features` | 50,000 |
| `C` | 1.0 |
| `class_weight` | `balanced` |
| Số classifier | 34, mỗi aspect một classifier 4-class |
| Input | `Review` gốc, không dùng `processed_review` |

## Nhận xét

- SVM phù hợp làm baseline vì đơn giản, deterministic và dễ giải thích.
- ACD F1 cao hơn SPC F1 vì phát hiện aspect thường dựa vào từ khóa trực tiếp.
- SPC khó hơn do cần hiểu phủ định, so sánh và cảm xúc trong ngữ cảnh.
- Baseline này giúp báo cáo chứng minh lợi ích của PhoBERT + VnCoreNLP + attention
  pooling thay vì chỉ báo cáo một model hiện đại đơn lẻ.
