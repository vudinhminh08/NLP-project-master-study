# Luồng dự án — Mô tả chi tiết hiện tại

Mục tiêu: tài liệu hóa toàn bộ luồng xử lý sentiment cho TripAdvisor hotel reviews trong repo, giải thích từng bước, vai trò các file chính, trọng số (ưu tiên) giữa các phương pháp, lý do chọn phương pháp, và cách chạy hệ thống.

---

## 1. Tổng quan ngắn gọn
- Dữ liệu chính: `tripadvisor_hotel_reviews.csv` (cột `Review`, `Rating`).
- Ba hướng chính được triển khai trong repo:
  1. Classical / ML pipelines (TF-IDF, RandomForest, kNN, ...).
  2. Deep Learning (BERT / RoBERTa-based fine-tuning, nhiều script trong `train/`).
  3. LLM-based workflow (prompting + fine-tune LoRA + v1 inference) trong `llm-sentiment-analysis` và Gradio apps (`app/`).
- Ứng dụng trực quan: Gradio app ở `app/main.py` (aspect extraction qua Gemini) và `app/main_single_model.py` (single-review classifier + recommendation).

## 2. Luồng end-to-end (từng bước)
1. Data ingest
   - File nguồn: `tripadvisor_hotel_reviews.csv`.
   - Mục đích: đọc review và rating để tạo nhãn (offline) hoặc làm input cho LLM labeling.

2. Xử lý & xây dựng dataset cho LLM
   - Script: `llm-sentiment-analysis/build_gemini_dataset.py`.
   - Chế độ `offline`: suy label từ rating (1-2 negative, 3 neutral, 4-5 positive) → xuất JSONL sẵn sàng fine-tune (`train_sft.jsonl`).
   - Chế độ `gemini`: gọi Gemini để gán nhãn và filter các dòng không phù hợp.
   - Input: CSV; Output: JSONL (một record = messages: system, user, assistant)

3. Fine-tune LLM (tùy chọn)
   - Script: `llm-sentiment-analysis/finetune_llm.py`.
   - Dùng: `trl` + LoRA để huấn luyện adapter trên `train_sft.jsonl`.
   - Kết quả: adapter/weights (ví dụ thư mục `./corvustus-lora` hoặc `rating_lora_llama3_3_1000samples`).
   - Mục đích: tối ưu mô hình LLM để sản sinh output có cấu trúc (JSON list of aspects hoặc label). Giúp cải thiện độ chính xác trên domain hotel reviews.

4. Đánh giá / inference LLM
   - Script: `llm-sentiment-analysis/eval_classification.py`.
   - Kết nối: qua VLLM hoặc endpoint tương tự (`VLLM_BASE_URL`, `VLLM_API_KEY`) hoặc local LLM path (`VLLM_MODEL`).
   - Nhiệm vụ: gửi prompt (system + user) → nhận output text → parse (JSON hoặc regex) → quy về label final.
   - Output: predicted label/score, metrics: accuracy, F1, mean class accuracy.

5. Hệ thống Deep Learning (fine-tune transformer)
   - Location: `train/` (ví dụ `sem2_roberta_twit.py`, `seminar2_bertweet*.py`).
   - Mục đích: đào tạo classifier (3-class hoặc 5-class) sử dụng backbone transformer, thực hiện K-Fold CV để ước lượng hiệu năng.
   - Output: checkpoints, số liệu đo lường.

6. Ứng dụng người dùng (Gradio)
   - `app/main.py`: upload file (nhiều dòng), cho từng dòng gọi Gemini (system prompt `system_prompt_v2`) để extract aspects và polarity; vẽ biểu đồ, trả recommendation (via `recommendation_prompt` + Gemini).
   - `app/main_single_model.py`: dùng checkpoint BERT (`best_bert_multilingual_model.pth`) để phân lớp single-review (softmax 3 nhãn) và gọi Gemini để tạo recommendations tổng quan.
   - Mục đích: giao diện trực quan để phân tích sentiment & đề xuất hành động.

## 3. Mô tả chi tiết file & vai trò
- `README.md`: chỉ dẫn cài đặt và các lệnh chạy chính.
- `requirement.txt`: dependency cơ bản (`torch`, `transformers`, `gradio`, `pandas`, ...).

- `llm-sentiment-analysis/build_gemini_dataset.py`:
  - Chức năng: chuyển CSV → JSONL để dùng cho fine-tune; hỗ trợ offline (dựa trên rating) và gemini (gọi model để kiểm tra/label).
  - Tại sao: cho phép tạo dataset có cấu trúc chat messages phù hợp với SFT pipeline.

- `llm-sentiment-analysis/finetune_llm.py`:
  - Chức năng: ví dụ cấu hình SFT + LoRA (sử dụng `trl`, `peft`) để huấn luyện adapter.
  - Tại sao: giảm chi phí huấn luyện full model; LoRA cho phép fine-tune nhanh hơn và lưu adapter nhỏ.

- `llm-sentiment-analysis/eval_classification.py`:
  - Chức năng: wrapper đánh giá mô hình LLM thông qua API chat (vllm/OpenAI-like) và hàm `get_final_verdict` để parse output.
  - Tại sao: kiểm thử độ tin cậy khi dùng LLM cho phân loại sentiment.

- `app/main.py`:
  - Chức năng: pipeline aspect-based extraction bằng Gemini, tạo biểu đồ và recommendation.
  - Tại sao: sử dụng LLM trực tiếp cho task aspect extraction (khi muốn nhiều thông tin hơn nhãn tổng thể).

- `app/main_single_model.py`:
  - Chức năng: sử dụng một classifier BERT đã được fine-tune (checkpoint `best_bert_multilingual_model.pth`) để phân lớp 3 nhãn cho một review; dùng LLM cho phần recommendation.
  - Tại sao: nhanh, deterministic, ít tốn chi phí gọi LLM cho mỗi câu; kết hợp LLM để tạo recommendation tổng quan.

- `train/*.py`:
  - Chức năng: script truyền thống cho training/finetuning (Roberta/BERT), K-Fold CV.
  - Tại sao: baseline và tham chiếu cho kết quả LLM.

## 4. Trọng số đề xuất (ưu tiên giữa các phương pháp)
Gợi ý phân bổ trọng số (dùng để quyết định khi kết hợp outputs hoặc để ưu tiên development effort):
- LLM-based aspect extraction + recommendation: 0.50
  - Lý do: trả về thông tin chi tiết (aspects + polarity) và tạo recommendation dạng ngôn ngữ tự nhiên; rất có giá trị cho business insights.
  - Rủi ro: chi phí API, biến động output, cần parsing robust.

- Transformer fine-tuned classifier (BERT / RoBERTa): 0.30
  - Lý do: cho kết quả nhanh, nhất quán, có thể deploy offline; tốt cho throughput cao và khi chi phí LLM không khả thi.

- Classical ML (TF-IDF + RandomForest / kNN): 0.10
  - Lý do: lightweight, baseline nhanh, dễ hiểu; dùng khi cần giải pháp rất nhẹ.

- Fine-tune LLM (LoRA adapters): 0.10
  - Lý do: đầu tư để có LLM tùy chỉnh (giảm chi phí gọi API, tăng độ chính xác domain-specific). Tuy nhiên cần nhiều tài nguyên ban đầu.

Ghi chú: các trọng số trên là khuyến nghị để cân đối chi phí/độ chính xác/độ chi tiết; có thể thay đổi dựa trên mục tiêu dự án (ví dụ ưu tiên chi phí thấp → tăng trọng số BERT/classical; ưu tiên insight → tăng LLM).

## 5. Quy trình thực thi (runbook nhanh)
1. Cài dependency:

```bash
pip install -r requirement.txt
```

2. Chuẩn bị API keys / checkpoints:
- Tạo file `api.txt` chứa API key cho Google Gemini (nếu dùng `app/*.py` gọi `google.generativeai`).
- Hoặc set env `GEMINI_API_KEY` khi chạy `build_gemini_dataset.py --mode gemini`.
- Nếu dùng VLLM/local LLM: set `VLLM_BASE_URL`, `VLLM_API_KEY`, `VLLM_MODEL` tương ứng.
- Đảm bảo `best_bert_multilingual_model.pth` có ở thư mục gốc nếu chạy `app/main_single_model.py`.

3. Tạo dataset offline cho LLM (nếu muốn):

```bash
python llm-sentiment-analysis/build_gemini_dataset.py --mode offline --input_csv tripadvisor_hotel_reviews.csv --output_jsonl train_sft.jsonl
```

4. Fine-tune LLM (nếu có hạ tầng):

```bash
python llm-sentiment-analysis/finetune_llm.py
```

5. Chạy Gradio apps:
- Aspect-based (file upload):

```bash
cd app
python main.py
```

- Single-review classifier + recommendation:

```bash
cd app
python main_single_model.py
```

6. Chạy evaluation LLM:

```bash
python llm-sentiment-analysis/eval_classification.py
```

7. Chạy training transformer (ví dụ):

```bash
python train/sem2_roberta_twit.py
```

## 6. Những rủi ro hiện tại & việc cần kiểm tra
- Thiếu `api.txt` hoặc `GEMINI_API_KEY` → app gọi Gemini sẽ lỗi.
- Thiếu `best_bert_multilingual_model.pth` → `app/main_single_model.py` sẽ error khi load checkpoint.
- Gọi Gemini/VLLM tốn quota và thời gian; hãy test với số lượng nhỏ trước.
- Parsing output LLM cần robust (hiện code cố gắng parse JSON hoặc tìm label bằng regex, nhưng outputs không chuẩn có thể gây lỗi).

## 7. Khuyến nghị nhanh (next steps)
- Kiểm tra tồn tại file `api.txt` và `best_bert_multilingual_model.pth` ngay bây giờ.
- Thêm unit tests nhỏ cho `get_final_verdict` và `extract_json_list` (để robust parsing).
- Nếu muốn giảm chi phí, ưu tiên huấn luyện LoRA adapter và deploy local VLLM, giảm gọi Gemini trực tiếp.
- Thêm logging chi tiết và ví dụ input/output mẫu trong `llm-sentiment-analysis/README.md`.

---

Nếu bạn muốn, tôi sẽ bổ sung: 
- Sơ đồ luồng (Mermaid) trực quan; hoặc
- Kiểm tra tồn tại các file bắt buộc và báo lỗi nếu thiếu (script kiểm tra nhanh).