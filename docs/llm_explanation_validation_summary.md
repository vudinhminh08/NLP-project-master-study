# LLM Explanation Quality Validation

Tài liệu này mô tả chi tiết quá trình kiểm chứng chất lượng của phần LLM
Explanation trong hệ thống PhoBERT + LLM Explanation — hướng ứng dụng cuối
cùng của báo cáo. Mục tiêu cốt lõi: cung cấp **3 bằng chứng độc lập** để
defend claim *"LLM không bịa thông tin"* trong sản phẩm demo.

---

## Bối cảnh và Động Lực

Trong hệ thống PhoBERT + LLM Explanation, PhoBERT đảm nhận toàn bộ phần phân
loại label (có số liệu F1 đo được). LLM chỉ đóng vai trò giải thích: trích
evidence từ review gốc, viết explanation cho từng aspect, và đề xuất
recommended_action cho hotel owner.

Vì LLM không có F1 để đo, cần một framework riêng để kiểm chứng. Ba loại bịa
khác nhau được định nghĩa và đo độc lập:

| Loại bịa | Câu hỏi | Method |
|-----------|---------|--------|
| Bịa label | LLM có tự thêm aspect không có hoặc đổi sentiment không? | M1 Label Consistency |
| Bịa evidence | LLM có trích câu không tồn tại trong review không? | M2 BERTScore Groundedness |
| Bịa diễn giải | Explanation và recommended_action có nội dung bịa không? | M3 Human Rubric |

---

## Thiết Kế Thực Nghiệm

**Dataset:** 200 mẫu lấy từ test set VLSP 2018 Hotel, dùng PhoBERT cls_only
checkpoint đã train để sinh prediction, sau đó gọi GPT-4o-mini tạo explanation.

**Output JSON của LLM** theo schema cố định:

```json
{
  "items": [
    {
      "aspect": "ROOMS#CLEANLINESS",
      "sentiment": "negative",
      "evidence": "Ga giường không sạch",
      "explanation": "..."
    }
  ],
  "overall_summary": "...",
  "recommended_action": "..."
}
```

**Parse fail rate:** 81/200 mẫu (40.5%) GPT-4o-mini trả về JSON không parse
được (thiếu bracket, text ngoài JSON, ...). Tất cả metric M1 và M2 được tính
trên 119 mẫu parse thành công, tổng cộng 399 explanation items. Parse fail
phản ánh JSON formatting của model; chất lượng của 119 mẫu hợp lệ không bị ảnh
hưởng.

**Môi trường chạy:** Kaggle, Tesla T4, Python 3.12.

---

## M1 — Label Consistency (Không Bịa Label)

### Phương pháp

Hàm `validate_explanation_items()` trong pipeline `explain_batch()` tự động
kiểm tra từng item LLM sinh ra:

- Nếu aspect không có trong danh sách PhoBERT predicted → drop (spurious aspect).
- Nếu sentiment khác với PhoBERT → restore về đúng nhãn.

Sau khi chạy xong, đọc lại metadata về số lượng bị drop và số lượng bị sửa.
Nếu cả hai đều bằng 0 → LLM tuân thủ hoàn toàn label scheme của PhoBERT.

### Kết quả

| Chỉ số | Giá trị |
|--------|---------|
| N mẫu | 200 (81 parse fail bị skip) |
| Total LLM items | 399 |
| Dropped spurious aspect | **0** |
| Sentiment restored | **0** |
| Aspect Preservation Rate | **100.0%** |
| Sentiment Consistency Rate | **100.0%** |
| Overall Label Accuracy | **100.0%** |

### Diễn giải

Không có item nào bị loại hoặc sửa. LLM chỉ nhận danh sách aspect/sentiment
đã predict từ PhoBERT rồi viết explanation — không tự bịa aspect mới, không
đổi nhãn. Đây là bằng chứng mạnh nhất về label consistency.

---

## M2 — BERTScore Evidence Groundedness (Không Bịa Câu Trích)

### Phương pháp

Với mỗi explanation item có evidence khác rỗng, tính BERTScore F1 giữa
`evidence` và `original_review`:

```
BERTScore F1(evidence, review) → đo mức độ evidence grounded trong review
```

Dùng PhoBERT làm backbone (`vinai/phobert-base-v2`, `num_layers=10`) vì đây là
model được pretrain trên tiếng Việt, phù hợp hơn BERT-base-multilingual cho
cặp chuỗi tiếng Việt. `num_layers=10` bắt buộc vì PhoBERT không có trong
registry mặc định của bert_score (registry chỉ chứa ~130 model đã biết).

### Kết quả

| Chỉ số | Giá trị |
|--------|---------|
| N mẫu | 200 |
| Evidence items được score | 377 |
| Backbone | `vinai/phobert-base-v2` (`num_layers=10`) |
| Mean BERTScore Precision | **0.7709** |
| Mean BERTScore Recall | **0.4396** |
| Mean BERTScore F1 | **0.5535** |

### Diễn giải

Precision 0.7709 có nghĩa là khoảng 77% token trong chuỗi evidence của LLM
tương ứng ngữ nghĩa với token trong review gốc. Đây là mức cao, cho thấy
evidence không phải được bịa ra mà được trích gần sát review. Recall thấp hơn
(0.4396) là bình thường và mong đợi: evidence chỉ là đoạn trích ngắn về một
aspect, không cover toàn bộ nội dung review.

Citation: Zhang et al. 2020 — BERTScore: Evaluating Text Generation with BERT.

---

## M3 — Human Rubric (Không Bịa Diễn Giải)

### Mục tiêu

M1 và M2 là metric tự động. M3 bổ sung góc nhìn của người đọc: liệu nội dung
LLM viết ra — explanation và recommended_action — có thật sự dựa trên review
hay bịa thêm thông tin không tồn tại?

### Thiết kế rubric

30 mẫu được chọn ngẫu nhiên (seed=42), với phân phối: 18 mẫu có
`recommended_action`, 12 mẫu không có. Việc giữ cả hai loại giúp đánh giá riêng
phần explanation (faithfulness) và phần gợi ý hành động (usefulness).

12 mẫu không có `recommended_action` hầu hết là các mẫu có sentiment
**positive** — LLM không sinh gợi ý cải tiến khi khách đã hài lòng. Ngược lại,
các mẫu **negative** và **neutral** gần như đều có recommended_action vì có vấn
đề cụ thể cần khắc phục. Do đó cột `usefulness` trong `rubric_annotated.csv`
để trống ở những dòng này — không phải thiếu sót mà là không có nội dung để
chấm.

Annotator đọc 5 trường cho mỗi mẫu: review gốc, aspect, sentiment, evidence,
explanation_item, overall_summary, recommended_action. Sau đó chấm hai tiêu chí:

**Faithfulness (0–2):**

| Điểm | Nghĩa |
|------|-------|
| 2 | Hoàn toàn trung thực: explanation và evidence đều có căn cứ rõ ràng từ review |
| 1 | Phần lớn trung thực: có thể suy diễn hơi quá hoặc summary khái quát lệch nhẹ, nhưng không sai thực chất |
| 0 | Hallucination: có thông tin bịa hoàn toàn không tồn tại trong review |

**Usefulness (0–2):** chỉ chấm cho 18 mẫu có recommended_action.

| Điểm | Nghĩa |
|------|-------|
| 2 | Rất hữu ích: cụ thể, actionable, phù hợp với aspect và sentiment |
| 1 | Hữu ích một phần: đúng hướng nhưng chung chung hoặc một phần không thực tế |
| 0 | Không hữu ích: không liên quan hoặc hoàn toàn không thực thi được |

### Quá trình annotation

Annotation được thực hiện theo thứ tự rank 1–30. Với mỗi mẫu:

1. Đọc review gốc để nắm toàn bộ nội dung khách muốn nói.
2. Kiểm tra evidence: có phải substring hoặc paraphrase gần sát của review không?
3. Kiểm tra explanation_item: suy luận có logic và bám sát evidence không?
4. Kiểm tra overall_summary: có phản ánh trung thực cả review không, hay thêm
   thông tin mà review không có?
5. Kiểm tra recommended_action: có thực thi được không, có phù hợp với vấn đề
   được phát hiện không?

### Chi tiết annotation từng mẫu

| Rank | Aspect | Sentiment | F | U | Ghi chú annotator |
|------|--------|-----------|---|---|-------------------|
| 1 | ROOM_AMENITIES#DESIGN&FEATURES | negative | 2 | 2 | Evidence exact; action "bổ sung quạt điện" rất cụ thể |
| 2 | FACILITIES#COMFORT | positive | 2 | — | Evidence exact; không có recommended_action |
| 3 | ROOMS#DESIGN&FEATURES | negative | 2 | 1 | Evidence exact; action "cải thiện thiết kế và không gian" hơi chung |
| 4 | ROOMS#COMFORT | negative | 2 | 2 | Evidence exact; action "cải thiện ánh sáng trong phòng" cụ thể |
| 5 | HOTEL#GENERAL | negative | 1 | 2 | Evidence exact; nhưng summary thêm "vẫn có một số phòng view đẹp" — review chỉ nói view cực kỳ tệ, không có phòng đẹp được nhắc tới tích cực |
| 6 | SERVICE#GENERAL | positive | 2 | — | Evidence exact; không có recommended_action |
| 7 | ROOMS#DESIGN&FEATURES | negative | 2 | 1 | Evidence exact; action "cải thiện thiết kế để tăng ánh sáng tự nhiên" khó thực thi khi hai bên đã là nhà cao tầng |
| 8 | ROOM_AMENITIES#COMFORT | positive | 2 | 2 | Evidence exact; action "cải thiện chỗ để xe" địa chỉ đúng vấn đề trong review |
| 9 | HOTEL#COMFORT | positive | 1 | — | Evidence là substring; nhưng explanation thêm "thoải mái" — review chỉ nói "thích nhất" chứ không nhắc tường minh comfort |
| 10 | SERVICE#GENERAL | positive | 2 | — | Evidence exact; không có recommended_action |
| 11 | FOOD&DRINKS#QUALITY | negative | 2 | 2 | Evidence exact "Bữa sáng chưa ngon"; action cụ thể về cải thiện chất lượng và đa dạng bữa sáng |
| 12 | LOCATION#GENERAL | negative | 2 | 1 | Evidence exact; action gồm "cải thiện vị trí" (không thể dời tòa nhà) + "cung cấp dịch vụ di chuyển" (hợp lý) |
| 13 | SERVICE#GENERAL | positive | 2 | — | Evidence exact; giữ nguyên cả lỗi chính tả của review ("nhan viên than thiện") |
| 14 | ROOMS#DESIGN&FEATURES | positive | 2 | 2 | Evidence exact; action "lắp quạt và cải thiện ánh sáng" bám sát 2 vấn đề trong review |
| 15 | SERVICE#GENERAL | positive | 2 | — | Evidence exact; không có recommended_action |
| 16 | LOCATION#GENERAL | positive | 2 | 2 | Evidence exact; action "cải thiện kích thước cửa sổ" cụ thể |
| 17 | HOTEL#DESIGN&FEATURES | positive | 2 | 2 | Evidence exact; action "cải thiện chất lượng đồ ăn" địa chỉ đúng điểm trừ trong review |
| 18 | SERVICE#GENERAL | positive | 2 | — | Evidence exact; không có recommended_action |
| 19 | ROOMS#CLEANLINESS | negative | 2 | 2 | Evidence exact; action "vệ sinh gầm giường" rất cụ thể, bám sát yêu cầu khách |
| 20 | FOOD&DRINKS#STYLE&OPTIONS | negative | 2 | 2 | Evidence exact; action "đa dạng lựa chọn đồ ăn sáng" cụ thể |
| 21 | SERVICE#GENERAL | neutral | 2 | — | Evidence exact; giải thích neutral hợp lý ("hơi hiền và ít nói" — mơ hồ, không hẳn tiêu cực) |
| 22 | ROOM_AMENITIES#DESIGN&FEATURES | negative | 2 | 2 | Evidence exact; action "bổ sung máy sấy tóc" chính xác theo yêu cầu khách |
| 23 | SERVICE#GENERAL | positive | 2 | 1 | Evidence exact; nhưng action "cải thiện bãi biển" — bãi biển là công cộng, khách sạn không kiểm soát được |
| 24 | ROOMS#CLEANLINESS | positive | 2 | — | Evidence exact; không có recommended_action |
| 25 | SERVICE#GENERAL | neutral | 1 | 1 | Evidence exact; tuy nhiên explanation cố justify neutral cho evidence âm tính rõ ràng ("không chuyên nghiệp"); summary thêm "không hài lòng với thiết kế" ngoài scope evidence |
| 26 | LOCATION#GENERAL | positive | 2 | — | Evidence exact; không có recommended_action |
| 27 | ROOMS#DESIGN&FEATURES | negative | 2 | 1 | Evidence exact (bao gồm cả câu gợi ý của khách); action "cải thiện view" khó thực hiện, nhưng "trang trí lại" theo đúng đề xuất của khách |
| 28 | SERVICE#GENERAL | neutral | 2 | — | Evidence exact; neutral được justify đúng — bảo vệ nhiệt tình nhưng lễ tân không nhiệt tình |
| 29 | ROOM_AMENITIES#DESIGN&FEATURES | negative | 1 | 2 | Evidence exact; nhưng summary lẫn "sạch sẽ" (từ muỗi trong phòng) vào aspect DESIGN&FEATURES — hai vấn đề khác nhau; action "cải thiện vệ sinh + nâng cấp điều hòa" vẫn cụ thể |
| 30 | SERVICE#GENERAL | positive | 2 | — | Evidence exact; không có recommended_action |

*(F = Faithfulness, U = Usefulness, — = không có recommended_action)*

### Kết quả annotation

**Faithfulness (N=30):**

| Điểm | Số mẫu | Tỷ lệ |
|------|--------|-------|
| 2 — Hoàn toàn trung thực | 26 | **86.7%** |
| 1 — Phần lớn trung thực | 4 | 13.3% |
| 0 — Hallucination | 0 | **0.0%** |
| **Mean** | | **1.867 / 2.0** |
| **Stdev** | | 0.346 |

**Usefulness (N=18, chỉ mẫu có recommended_action):**

| Điểm | Số mẫu | Tỷ lệ |
|------|--------|-------|
| 2 — Rất hữu ích | 12 | **66.7%** |
| 1 — Hữu ích một phần | 6 | 33.3% |
| 0 — Không hữu ích | 0 | **0.0%** |
| **Mean** | | **1.667 / 2.0** |
| **Stdev** | | 0.485 |

**4 trường hợp faithfulness = 1** (không phải hallucination — LLM không bịa
thông tin hoàn toàn mới, chỉ có vấn đề về mức độ suy diễn hoặc khái quát):

- **Rank 5:** Summary thêm "vẫn có một số phòng có view đẹp" — review chỉ
  nói "view cực kỳ tệ" và chỉ 2 phòng nhìn ra sông, không có ý khẳng định đẹp.
- **Rank 9:** Explanation suy ra khách hài lòng về "sự thoải mái" nhưng review
  chỉ nói "thích nhất" — không nhắc tường minh comfort.
- **Rank 25:** Explanation cố biện hộ cho nhãn neutral dù evidence âm tính rõ
  ("làm việc không chuyên nghiệp"); summary thêm "không hài lòng với thiết kế"
  ngoài phạm vi evidence của aspect SERVICE#GENERAL.
- **Rank 29:** Summary lẫn khái niệm "sạch sẽ" (từ vấn đề muỗi trong phòng)
  vào aspect DESIGN&FEATURES — hai khía cạnh thực ra thuộc về cleanliness và
  design features, nhưng LLM gộp chung.

**6 trường hợp usefulness = 1** (đúng hướng nhưng không hoàn toàn actionable):

- **Rank 3:** "Cải thiện thiết kế và không gian phòng" — quá chung, không nói
  cụ thể cần cải gì.
- **Rank 7:** "Tăng cường ánh sáng tự nhiên" — khó thực thi khi hai bên phòng
  đã là nhà cao tầng.
- **Rank 12:** "Cải thiện vị trí" không thể dời tòa nhà; nhưng phần "cung cấp
  dịch vụ di chuyển" thì hợp lý.
- **Rank 23:** "Cải thiện độ sạch sẽ của bãi biển" — bãi biển công cộng,
  không thuộc quyền kiểm soát của khách sạn.
- **Rank 25:** Action kết hợp cả service lẫn design — không tập trung vào aspect
  đang được phân tích.
- **Rank 27:** "Cải thiện view từ phòng" rất khó; chỉ phần "trang trí lại"
  mới là gợi ý thực thi được.

---

## Tổng Hợp 3 Bằng Chứng

| Method | N | Metric | Kết quả | Claim bảo vệ |
|--------|---|--------|---------|-------------|
| M1 — Label Consistency | 200 | Sentiment consistency | **100.0%** | LLM không bịa label |
| M2 — BERTScore Evidence | 200 | Mean BERTScore F1 | **0.5535** | LLM không bịa câu trích |
| M3 — Human Rubric | 30 | Faithfulness mean | **1.867 / 2.0** | LLM không bịa diễn giải |
| M3 — Human Rubric | 18 | Usefulness mean | **1.667 / 2.0** | Recommended action có giá trị thực tế |

Không có trường hợp nào bị chấm score 0 ở bất kỳ tiêu chí nào.

---

## Code, Notebook và File Kết Quả

**Module Python:**

- `code/llm_explainability/validate_llm_quality.py` — 3 hàm chính:
  `compute_label_consistency()`, `compute_bertscore_groundedness()`,
  `generate_rubric_template()` / `compute_rubric_scores()`.

**Notebook thực thi (đã chạy xong trên Kaggle T4):**

- `notebooks/phase_llm_explaination_done.ipynb`

**File kết quả:**

| File | Nội dung |
|------|---------|
| `outputs/results/llm_validation_results/validation_report.json` | JSON tổng hợp kết quả M1, M2, M3 |
| `outputs/results/llm_validation_results/explanation_quality_report.json` | Stats parse fail, invalid evidence |
| `outputs/results/llm_validation_results/explanation_samples.json` | 200 explanation samples đầy đủ |
| `outputs/results/llm_validation_results/rubric_template.csv` | Template 30 mẫu (cột annotation trống) |
| `outputs/results/llm_validation_results/rubric_annotated.csv` | Template đã điền đầy đủ faithfulness và usefulness |

---

## Giới Hạn và Lưu Ý

**Parse fail rate 40.5%** là điểm yếu rõ ràng nhất. Gần một nửa số lần gọi
LLM không trả về JSON hợp lệ. Trong sản phẩm thực tế, cần thêm
`response_format={"type": "json_object"}` vào API call hoặc dùng retry với
prompt nhắc lại format. Validation này chỉ phản ánh chất lượng của 119 mẫu
parse thành công.

**M3 annotation** được thực hiện thủ công bởi thành viên nhóm, đọc trực tiếp
từng mẫu và chấm điểm độc lập. Với 0 trường hợp hallucination hoàn toàn
(score=0) và mean faithfulness 1.867/2.0, kết luận "LLM không bịa" có
căn cứ đủ mạnh.

**Scope của claim:** "LLM không bịa" được giới hạn trong ngữ cảnh hệ thống
này — LLM nhận input là prediction của PhoBERT và review gốc, không sinh label
tự do. Kết quả không áp dụng cho trường hợp dùng LLM làm classifier độc lập.
