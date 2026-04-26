# QA version2

Tài liệu này tổng hợp các câu hỏi thầy có thể hỏi trong lúc báo cáo, kèm đáp án
chi tiết nhưng vẫn bám sát đúng code, notebook và kết quả hiện tại của dự án.

## 1. Bài toán của nhóm là gì?

**Q:** Nhóm đang giải quyết bài toán gì?

**A:** Nhóm giải quyết bài toán **Aspect-Based Sentiment Analysis (ABSA)** cho
review khách sạn tiếng Việt trên bộ dữ liệu **VLSP 2018 Hotel**. Với mỗi
review, hệ thống cần:

- phát hiện những aspect nào được nhắc tới, gọi là **ACD**;
- với các aspect đó, xác định sentiment là `positive`, `negative` hay
  `neutral`, gọi là **SPC**.

Trong dataset này có **34 aspect**, nên đầu ra của model là một vector gồm 34
nhãn, mỗi nhãn nhận một trong bốn giá trị:

```text
0 = absent
1 = positive
2 = negative
3 = neutral
```

Khó khăn của bài toán là:

- một review có thể chứa nhiều aspect cùng lúc;
- mỗi aspect có thể có sentiment khác nhau trong cùng một câu;
- dữ liệu tiếng Việt có vấn đề word segmentation;
- dữ liệu mất cân bằng mạnh, đặc biệt class `absent` chiếm đa số.

## 2. Vì sao chọn VLSP 2018 Hotel?

**Q:** Tại sao lại chọn VLSP 2018 Hotel thay vì dataset khác?

**A:** Có ba lý do chính:

1. Đây là benchmark ABSA tiếng Việt được dùng phổ biến nhất trong bối cảnh học
   thuật ở Việt Nam.
2. Dataset có schema đủ phức tạp với 34 aspect, nên phù hợp để đánh giá mô hình
   nghiêm túc.
3. Đã có mốc tham chiếu SOTA để so sánh, nên có thể đặt kết quả của nhóm trong
   một bối cảnh rõ ràng.

Ngoài ra kích thước dataset vẫn đủ nhỏ để train được trên Kaggle T4 GPU, phù
hợp với giới hạn tài nguyên của bài tập lớn.

## 3. ACD, SPC và Combined F1 là gì?

**Q:** Ba metric ACD F1, SPC F1, Combined F1 khác nhau như thế nào?

**A:** 

- **ACD F1** đo khả năng phát hiện aspect có xuất hiện hay không.
- **SPC F1** đo khả năng phân loại sentiment đúng cho các aspect.
- **Combined F1** là trung bình cộng của ACD F1 và SPC F1.

Trong code và báo cáo, Combined F1 được dùng làm metric chính vì nó phản ánh
đồng thời cả hai kỹ năng cốt lõi của hệ thống:

```text
Combined F1 = (ACD F1 + SPC F1) / 2
```

Nếu model chỉ giỏi phát hiện aspect nhưng sentiment kém, hoặc ngược lại, thì
Combined F1 sẽ phản ánh điều đó.

## 4. Vì sao dùng macro F1 thay vì accuracy?

**Q:** Tại sao không dùng accuracy?

**A:** Accuracy không phù hợp vì dữ liệu bị lệch mạnh về class `absent`. Nếu
model dự đoán gần như tất cả aspect là `absent`, accuracy vẫn có thể cao nhưng
model thực tế vô dụng. Macro F1 phù hợp hơn vì:

- đánh giá công bằng hơn giữa các aspect;
- không để aspect phổ biến lấn át aspect hiếm;
- phản ánh rõ hơn chất lượng thực của ABSA.

## 5. Pipeline tổng thể của hệ thống là gì?

**Q:** Một review đi qua hệ thống như thế nào?

**A:** Luồng đầy đủ là:

```text
Review gốc
-> preprocessing
-> VnCoreNLP word segmentation
-> PhoBERT predictor
-> danh sách aspect + sentiment
-> LLM explanation
-> evidence + explanation + recommended action
```

Quan trọng là:

- **PhoBERT** chịu trách nhiệm sinh nhãn;
- **LLM** chỉ chịu trách nhiệm giải thích nhãn;
- LLM không được phép thay đổi aspect/sentiment của PhoBERT.

## 6. Vai trò của PhoBERT là gì?

**Q:** PhoBERT trong hệ thống dùng để làm gì?

**A:** PhoBERT là **classifier chính**. Nó nhận review đầu vào và trả ra các
aspect cùng sentiment tương ứng. Đây là phần chịu trách nhiệm cho toàn bộ số
liệu F1 trong phần predictor.

Ví dụ:

Input:

```text
Phòng sạch nhưng nhân viên phục vụ chậm.
```

PhoBERT có thể trả ra:

```json
[
  {"aspect": "ROOMS#CLEANLINESS", "sentiment": "positive"},
  {"aspect": "SERVICE#GENERAL", "sentiment": "negative"}
]
```

## 7. Vai trò của LLM explanation là gì?

**Q:** LLM explanation có tác dụng gì?

**A:** LLM không phải predictor. Nó lấy:

- review gốc;
- output của PhoBERT;

rồi sinh ra:

- `evidence`: câu hoặc cụm từ làm bằng chứng;
- `explanation`: diễn giải bằng ngôn ngữ tự nhiên;
- `recommended_action`: gợi ý hành động cho người quản lý khách sạn.

Ví dụ:

```json
{
  "aspect": "SERVICE#GENERAL",
  "sentiment": "negative",
  "evidence": "nhân viên phục vụ chậm",
  "explanation": "Khách không hài lòng với tốc độ phục vụ của nhân viên.",
  "recommended_action": "Rà soát quy trình phục vụ và đào tạo nhân viên để phản hồi nhanh hơn."
}
```


## 10. Vì sao VnCoreNLP lại quan trọng?

**Q:** Vì sao preprocessing bằng VnCoreNLP lại quan trọng đến vậy?

**A:** Vì PhoBERT được pretrain trên tiếng Việt đã word-segmented. Nếu input
không được segment theo cùng chuẩn, tokenizer sẽ tách subword kém ổn định và
representation của các cụm như `nhân viên`, `khách sạn`, `phục vụ` sẽ kém đi.

Đây không chỉ là trực giác mà đã được chứng minh bằng ablation:

| Pipeline | Encoder | Combined F1 |
|---|---|---:|
| Không VnCoreNLP | `cls_only` | 0.2681 |
| Có VnCoreNLP | `cls_only` | 0.6218 |

Gain là:

```text
0.6218 - 0.2681 = +0.3537
```

Đây là một trong những kết quả dễ defend nhất trong báo cáo.

## 11. TEENCODE_DICT được xây như thế nào?

**Q:** `TEENCODE_DICT` được tạo từ dataset hay tự làm thủ công?

**A:** Hiện tại `TEENCODE_DICT` trong
[`code/data_processing/step3_preprocessing.py`](/Users/macbookpro/Documents/Master-study/NLP/absa-vlsp2018-hotel/code/data_processing/step3_preprocessing.py:25)
là **dictionary hard-code thủ công**. Nó không được sinh tự động từ dataset.

Ý tưởng là chuẩn hóa các viết tắt và slang hay gặp trong domain khách sạn như:

- `nv -> nhân viên`
- `ks -> khách sạn`
- `bf -> bữa sáng`
- `ko -> không`
- `ckin -> check-in`

Nó được dùng trước bước VnCoreNLP để giảm nhiễu trong đầu vào.

## 12. Kiến trúc PhoBERT Predictor gồm những phần nào?

**Q:** PhoBERT predictor trong báo cáo gồm những khối nào?

**A:** Có ba khối chính:

1. **Kiến trúc tổng quan**: multi-task learning cho 34 aspect.
2. **Preprocessing**: normalize text, teencode replacement, VnCoreNLP, tokenizer.
3. **PhoBERT Encoder + Aspect-Aware Attention Pooling + Dual Head**.

Ba khối này được mô tả lại trong phần 5.3 của report v10.

## 13. Aspect-Aware Attention Pooling là gì?

**Q:** Attention pooling ở đây khác gì với dùng vector CLS?

**A:** Nếu chỉ dùng một vector CLS chung cho toàn bộ review, model phải ép toàn
bộ thông tin của nhiều aspect khác nhau vào một representation duy nhất. Điều đó
không lý tưởng khi một review có nhiều aspect với sentiment trái chiều.

Attention pooling theo aspect học:

- một query riêng cho mỗi aspect;
- tính attention giữa aspect đó và các token trong review;
- sinh một vector riêng `r_i` cho aspect `i`.

Nhờ đó:

- `ROOMS#CLEANLINESS` có thể nhìn vào `phòng`, `sạch`;
- `SERVICE#GENERAL` có thể nhìn vào `nhân_viên`, `phục_vụ`, `chậm`.

Đây cũng là lý do phần này dễ trực quan hóa bằng attention map.

## 14. Attention map có ý nghĩa gì?

**Q:** Attention map có chứng minh model hiểu thật không?

**A:** Không nên phát biểu quá mạnh rằng attention map “chứng minh” model hiểu
giống con người. Cách nói đúng hơn là:

- attention map cho thấy model **đang dùng token nào nhiều hơn** khi tạo
  representation cho từng aspect;
- đây là một hình thức **attention-based explanation**;
- nó phù hợp với kiến trúc của model và giúp ta giải thích tại sao mỗi aspect có
  prediction riêng.

Vì vậy attention map là công cụ giải thích hợp lý, nhưng không phải bằng chứng
tuyệt đối về causal reasoning.

## 15. Dual Head có tác dụng gì?

**Q:** Tại sao lại tách Presence Head và Sentiment Head?

**A:** Vì ABSA thực chất gồm hai bài toán:

- aspect có được nhắc tới không;
- nếu có thì sentiment là gì.

Presence Head giải quyết ACD. Sentiment Head giải quyết SPC. Cách tách này giúp:

- giảm ảnh hưởng áp đảo của lớp `absent`;
- làm bài toán rõ hơn về mặt mô hình;
- tăng tính giải thích vì ta thấy rõ bước phát hiện aspect và bước gán sentiment.

## 16. Split loss được tính như thế nào?

**Q:** Loss tổng của model là gì?

**A:** Loss tổng là:

```text
loss = lambda_presence * presence_loss + lambda_sentiment * sentiment_loss
```

Trong cấu hình hiện tại:

```text
lambda_presence = 1.0
lambda_sentiment = 1.0
```

`presence_loss` là BCE cho nhãn present/absent.  
`sentiment_loss` là focal-style cross-entropy cho sentiment, chỉ tính trên các
aspect present.

## 17. Vì sao dùng focal loss?

**Q:** Focal loss dùng để làm gì?

**A:** Focal loss giúp giảm trọng số của những mẫu dễ, tăng tập trung vào những
mẫu khó. Trong bài toán này, sentiment classification bị mất cân bằng mạnh, nên
focal loss giúp model học tốt hơn trên các trường hợp khó và các class hiếm.

## 18. Vì sao cần class weight và weight clip?

**Q:** Class weight và weight clip có vai trò gì?

**A:** Do dữ liệu bị lệch mạnh về `absent`, nếu không weighting thì loss sẽ bị
class majority chi phối. Class weight giúp cân bằng lại mức đóng góp của các
nhãn hiếm hơn. Tuy nhiên nếu nhãn quá hiếm thì weight có thể quá lớn, gây update
mất ổn định; vì thế nhóm dùng `weight_clip = 10.0`.

## 19. Rare oversampling là gì?

**Q:** Rare oversampling được làm thế nào?

**A:** Trong dataloader, nhóm dùng `WeightedRandomSampler` để tăng xác suất lấy
các review chứa aspect hiếm. Ý tưởng là macro F1 đánh giá đều trên aspect, nên
nếu không tăng tín hiệu cho aspect hiếm thì model sẽ học nghiêng về những aspect
phổ biến.

## 20. Vì sao dùng AdamW?

**Q:** Tại sao optimizer là AdamW chứ không phải Adam?

**A:** AdamW phù hợp hơn với fine-tuning Transformer vì tách weight decay khỏi
quá trình cập nhật adaptive gradient. Đây là lựa chọn chuẩn trong đa số bài toán
fine-tune BERT/PhoBERT.

## 21. Vì sao learning rate là 1e-4?

**Q:** Tại sao final v3 dùng `1e-4`?

**A:** Nhóm đã có kết quả thực nghiệm cho thấy `2e-5` thấp hơn rõ rệt trong
setting hiện tại. Với pipeline mới, các classification head cần học đủ nhanh,
nên `1e-4` là mức hợp lý hơn cho toàn bộ setup, đặc biệt khi kết hợp với:

- `head_lr_mult = 5.0`
- `layerwise_lr_decay = 0.92`

Điểm quan trọng là đây là quyết định dựa trên kết quả thật, không chỉ theo
kinh nghiệm chung.

## 22. head_lr_mult và layerwise_lr_decay có ý nghĩa gì?

**Q:** Hai tham số này để làm gì?

**A:** 

- `head_lr_mult = 5.0`: các head mới tạo ra cho task-specific learning được học
  nhanh hơn encoder pretrained.
- `layerwise_lr_decay = 0.92`: các layer thấp của PhoBERT cập nhật nhẹ hơn, giữ
  tri thức ngôn ngữ chung tốt hơn.

Đây là cách fine-tune Transformer ổn định hơn thay vì dùng cùng một learning
rate cho mọi tầng.

## 23. Vì sao best single là cls_only?

**Q:** Tại sao không chọn concat_4_layers nếu representation của nó giàu hơn?

**A:** Vì kết quả test thực tế cho thấy `cls_only` tốt hơn:

| Model | Test Combined F1 |
|---|---:|
| `concat_4_layers` | 0.6101 |
| `cls_only` | 0.6218 |

`concat_4_layers` dùng hidden size lớn hơn rất nhiều nên checkpoint nặng hơn,
head to hơn và dễ overfit hơn trên dataset chỉ có 3.000 train samples.

## 25. Threshold tuning là gì?

**Q:** Tại sao lại cần threshold tuning?

**A:** Presence Head sinh xác suất aspect có xuất hiện hay không. Nếu luôn dùng
ngưỡng `0.5` cho mọi aspect thì không chắc là tối ưu, vì:

- mỗi aspect có phân phối xác suất khác nhau;
- aspect hiếm và phổ biến có hành vi khác nhau;
- mục tiêu cuối cùng là tăng Combined F1 trên dev.

Vì vậy nhóm tune threshold trên dev theo grid:

```text
[0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
```

## 26. Có thực sự ra 34 threshold không?

**Q:** Threshold tuning có cho ra một list 34 giá trị không?

**A:** Có. Với mode `per_aspect_combined`, sau tuning sẽ có một threshold cho
mỗi aspect. Danh sách này đã được recompute từ checkpoint `cls_only` trên dev
set và lưu vào:

[presence_thresholds_cls_only_dev_tuned.json](/Users/macbookpro/Documents/Master-study/NLP/absa-vlsp2018-hotel/outputs/results/phobert_results_version3/presence_thresholds_cls_only_dev_tuned.json)

Mean threshold khoảng `0.462`, min `0.35`, max `0.60`.

## 29. Vì sao SPC thấp hơn ACD?

**Q:** Vì sao ACD F1 cao hơn SPC F1?

**A:** Vì phát hiện aspect thường dễ hơn phân loại sentiment. Aspect detection
thường dựa vào token khá trực tiếp như `phòng`, `nhân_viên`, `vị_trí`, trong khi
sentiment cần hiểu:

- phủ định;
- sắc thái mơ hồ;
- so sánh;
- nhiều aspect cùng xuất hiện trong một câu.

Vì vậy SPC luôn là phần khó hơn trong ABSA.

## 30. Vì sao chưa bằng SOTA?

**Q:** Tại sao model của nhóm chưa bằng SOTA?

**A:** Theo xác nhận lại từ nguồn DS4V/Dang et al. 2022, nguyên nhân chính không
phải là “chưa dùng PhoBERT-large”. Các nguyên nhân hợp lý hơn là:

- pipeline nghiên cứu của SOTA được tối ưu sâu hơn;
- preprocessing rộng hơn;
- tuning nhiều hơn;
- xử lý output theo hướng ACSA multi-task chuyên biệt hơn;
- rare aspect vẫn rất khó trong dataset hiện tại.

Điểm quan trọng là nhóm không nên trả lời theo kiểu “vì em không có model
large”, vì đó không đúng với phần đối chiếu hiện tại.

## 32. Phần LLM đang được đánh giá bằng gì?

**Q:** LLM explanation được đánh giá bằng những metric nào?

**A:** Có ba nhóm chính:

1. **Label Consistency**
2. **BERTScore Evidence Groundedness**
3. **Human Rubric**

Các metric này không đo cùng một thứ, mà bổ sung cho nhau.

## 33. Label Consistency 100% nghĩa là gì?

**Q:** 100% Label Consistency có nghĩa gì?

**A:** Nghĩa là LLM không làm thay đổi nhãn PhoBERT. Aspect và sentiment ở đầu
ra explanation hoàn toàn khớp với prediction đầu vào. Điều này rất quan trọng
vì nó cho thấy kiến trúc tách vai giữa classifier và explainer đang hoạt động
đúng.

## 34. BERTScore F1 = 0.5535 là cao hay thấp?

**Q:** `Mean BERTScore F1 = 0.5535` là cao hay thấp?

**A:** Không nên nói đơn giản là “cao” hay “thấp” khi đứng một mình. Cách bảo
vệ đúng là:

- đây là mức **chấp nhận được** cho task evidence extraction theo aspect;
- cần đọc cùng với **Precision = 0.7709**, **Label Consistency = 100%**, và
  **Human Faithfulness = 1.867/2**.

Precision cao cho thấy phần text do LLM sinh ra bám khá sát review gốc. F1 không
quá cao chủ yếu vì recall thấp, do evidence chỉ là một đoạn ngắn chứ không nhằm
bao phủ toàn bộ review.

## 35. Số liệu nào chứng minh 0.5535 là ổn?

**Q:** Nếu thầy hỏi “số liệu nào chứng minh 0.5535 là ổn?” thì trả lời sao?

**A:** Không có ngưỡng tuyệt đối cho BERTScore F1. Cần dùng bộ số liệu đi kèm:

- `Precision = 0.7709`
- `Label Consistency = 100%`
- `Human Faithfulness = 1.867 / 2`

Ba số này là nền để nói rằng explanation có groundedness tốt và ít hallucination,
ngay cả khi F1 tổng thể chỉ ở mức trung bình.

## 36. Recommended action có được đánh giá không?

**Q:** Phần `recommended_action` đã được đánh giá chưa?

**A:** Có, nhưng chủ yếu bằng **Human Rubric Usefulness**. Đây là chỉ số đánh
giá mức hữu ích và khả thi của gợi ý hành động. Hiện chưa có một metric tự động
riêng đủ mạnh cho phần action, nên nhóm dùng human evaluation là chính.

## 37. Human Faithfulness và Human Usefulness nghĩa là gì?

**Q:** Hai metric human rubric phản ánh điều gì?

**A:** 

- **Faithfulness**: explanation có bịa hay không, có bám review không.
- **Usefulness**: gợi ý hành động có cụ thể, liên quan và khả thi không.

Kết quả hiện tại:

- Faithfulness = `1.867 / 2`
- Usefulness = `1.667 / 2`

Điều này cho thấy explanation khá đáng tin, còn action nhìn chung hữu ích nhưng
vẫn còn room for improvement.

## 38. App thực tế chạy như thế nào?

**Q:** Trong app Streamlit, luồng chạy thực tế là gì?

**A:** 

1. User nhập review hoặc import CSV/XLSX.
2. App chạy preprocessing bằng VnCoreNLP.
3. PhoBERT v3 `cls_only` dự đoán aspect/sentiment.
4. Kết quả được format thành danh sách prediction.
5. LLM nhận review gốc + prediction để sinh explanation.
6. App hiển thị:
   - review đã preprocess;
   - bảng prediction;
   - explanation/evidence/action.



## 41. Nếu thầy hỏi điểm mạnh lớn nhất của nhóm là gì?

**Q:** Nếu chỉ chọn một điểm mạnh để defend thì là gì?

**A:** Điểm mạnh nhất là nhóm không chỉ có một con số F1, mà đã xây được một
pipeline có thể giải thích:

- preprocessing có ablation rõ;
- predictor có kiến trúc hợp lý cho ABSA tiếng Việt;
- attention map giúp giải thích token-level;
- LLM explanation biến output kỹ thuật thành thông tin hữu ích cho người dùng;
- phần explanation có validation bằng nhiều lớp metric.

Nói cách khác, nhóm không chỉ tối ưu điểm mà còn biến mô hình thành một hệ thống
có thể trình bày được với người dùng cuối.

## 42. Nếu thầy hỏi điểm yếu lớn nhất là gì?

**Q:** Điểm yếu lớn nhất hiện tại là gì?

**A:** Có ba điểm yếu chính:

1. Chưa bằng SOTA.
2. Một số aspect hiếm vẫn rất yếu do thiếu dữ liệu.
3. Threshold tuned chưa được persist tự động vào checkpoint, phải lưu riêng.

Điều quan trọng là nhóm hiểu rõ các điểm yếu này và có thể giải thích nguyên
nhân kỹ thuật của từng điểm.
