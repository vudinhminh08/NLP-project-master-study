# CHANGELOG — ABSA VLSP 2018 Hotel

## [2026-04-07] Week 4 — Thay đổi định hướng

### Lý do thay đổi
- Week 3 LLM few-shot/RAG cho kết quả kém (Combined F1 = 0.3532 vs PhoBERT 0.5650)
- Feedback từ thầy: LLM week 3 "không có ý nghĩa" khi chỉ chạy rồi so sánh
- Thầy yêu cầu: kết hợp PhoBERT + LLM thành pipeline có giá trị thực tế

### Phân tích kỹ thuật
- LLM kém ở prediction (few-shot trên 34-class multi-label task quá khó)
- LLM mạnh ở generation (viết text tự nhiên, giải thích reasoning)
- PhoBERT mạnh ở classification nhưng là black-box
- Bottleneck lớn nhất: class imbalance (85% absent, 8 rare aspects)
- ROOM_AMENITIES#PRICES: 0 mẫu train → không thể học

### Định hướng mới (Week 4)
1. **LLM Data Augmentation:** Dùng LLM sinh training samples cho 8 rare aspects
   - Mục tiêu: tăng từ 0–90 mẫu lên 100–200 mẫu/aspect
   - Re-train PhoBERT trên augmented data → kỳ vọng +2–5% Combined F1
2. **LLM Explainability:** PhoBERT predict → LLM giải thích kết quả
   - Output: giải thích tiếng Việt cho từng aspect detected
   - Ứng dụng: hệ thống phân tích review tự động cho quản lý khách sạn

### Kết quả đạt được
- [ ] Augmented dataset tạo xong: data/train_augmented.csv
- [ ] PhoBERT re-train trên augmented data: outputs/results/week4_augmented/
- [ ] Explainability pipeline chạy được: code/week4/explainer.py
- [ ] Demo end-to-end: notebooks/week4_demo.ipynb

### So sánh kết quả (điền sau khi chạy xong)
| Model | ACD F1 | SPC F1 | Combined F1 |
|---|---|---|---|
| PhoBERT ensemble (week 2) | 0.6454 | 0.4834 | 0.5650 |
| PhoBERT + augmented data (week 4) | _____ | _____ | _____ |
| Delta | _____ | _____ | _____ |
