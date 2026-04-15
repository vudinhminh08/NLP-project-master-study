# Hotel Review Insight Assistant

App demo PhoBERT + LLM Explanation cho bài toán ABSA VLSP 2018 Hotel.

## Mục tiêu

- PhoBERT đã train sẵn dự đoán aspect và sentiment.
- LLM giải thích prediction, trích bằng chứng trong review và gợi ý hành động.
- LLM không thay đổi nhãn PhoBERT.
- Không train lại, không ensemble, không fallback rule-based.

## Chạy app

Từ thư mục root của project:

```bash
pip install -r app/requirements-app.txt
streamlit run app/streamlit_app.py
```

## Luồng demo

1. Bấm `Start`.
2. Điền đủ 3 cấu hình:
   - PhoBERT checkpoint
   - VnCoreNLP directory
   - OpenAI API key
3. Bấm `Lưu và tiếp tục`.
4. Chọn nhập 1 review hoặc import CSV/XLSX.
5. Bấm `Phân tích review`.

## Checkpoint PhoBERT

Đường dẫn mặc định:

```text
outputs/results/phobert_best_single/models_cls_only/best_model.pt
```

App dùng cấu hình:

```python
ABSAPhoBERT(
    model_name="vinai/phobert-base-v2",
    encoder_option="cls_only",
    num_aspects=34,
    num_labels=4,
)
```

Nếu thiếu checkpoint hoặc không load được VnCoreNLP, app sẽ báo lỗi và dừng phần phân tích.

## API key

Phần LLM explanation cần OpenAI API key. App yêu cầu nhập đủ API key ở màn config trước khi vào màn review.

## Định dạng file import

App chỉ chấp nhận `.csv`, `.xlsx`, `.xls` có cột:

```text
review_text
```

Ví dụ:

```csv
review_text
"Phòng sạch, nhân viên thân thiện nhưng bữa sáng hơi ít món."
"Khách sạn gần trung tâm, giá hợp lý, phòng hơi nhỏ."
```

Các cột khác sẽ bị bỏ qua. Review rỗng sẽ bị loại.

File mẫu có sẵn tại:

```text
app/sample_reviews.csv
```
