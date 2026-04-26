# Major Improvements Explained

Tài liệu này giải thích chi tiết 3 thay đổi lớn nhất của hệ thống hiện tại. Mục
tiêu là giúp trả lời phần thầy thường hỏi sâu nhất trong lúc báo cáo: **nhóm đã
cải tiến gì so với bản cũ, vì sao cải tiến đó hợp lý, và nó tác động thế nào đến
kết quả**.

Ba phần được tập trung là:

1. Aspect-Aware Attention Pooling
2. Dual Head / Split Loss
3. LLM Explanation và cách đánh giá

---

## 1. Aspect-Aware Attention Pooling

## 1.1. Bản cũ làm gì?

Ở bản cũ, cách đơn giản nhất là lấy một representation chung từ encoder
PhoBERT, thường là:

- vector CLS của layer cuối (`cls_only`), hoặc
- vector ghép từ 4 layer cuối (`concat_4_layers`)

Sau đó từ representation chung đó, model dự đoán toàn bộ 34 aspect.

Điểm mạnh của cách cũ:

- đơn giản;
- dễ cài đặt;
- chạy nhanh;
- ít tham số hơn nếu chỉ dùng `cls_only`.

Điểm yếu:

- cả review chỉ có **một vector chung**, trong khi một review có thể chứa nhiều
  aspect khác nhau;
- model không có cơ chế rõ ràng để nói rằng aspect này đang nhìn vào phần nào
  của câu;
- khó giải thích vì cùng một representation phải gánh đồng thời nhiều khía cạnh.

Ví dụ review:

```text
Phòng sạch nhưng nhân viên phục vụ chậm.
```

Trong câu này có ít nhất hai aspect:

- `ROOMS#CLEANLINESS` -> positive
- `SERVICE#GENERAL` -> negative

Nếu chỉ dùng một vector CLS chung, model phải nhồi cả hai tín hiệu trái chiều
vào cùng một representation. Điều này không tối ưu cho ABSA.

## 1.2. Bản mới thay đổi gì?

Bản mới dùng **Aspect-Aware Attention Pooling**.

Ý tưởng là:

- mỗi aspect có một **query vector riêng**;
- mỗi token sau PhoBERT có một vector ngữ cảnh riêng;
- model tính xem token nào quan trọng với aspect nào;
- từ đó tạo ra **một representation riêng cho từng aspect**.

Tức là:

- `ROOMS#CLEANLINESS` có vector riêng;
- `SERVICE#GENERAL` có vector riêng;
- `LOCATION#GENERAL` có vector riêng;
- ...

Thay vì 1 vector cho cả review, ta có 34 vector theo 34 aspect.

## 1.3. Cách tính cụ thể

Sau khi review đi qua PhoBERT, ta có các hidden states:

```text
H = [h1, h2, ..., hT]
```

với:

- `T` là số token;
- `h_j` là vector ngữ cảnh của token thứ `j`.

Mỗi aspect `i` có một query vector học được:

```text
q_i
```

Model chiếu từng token vào không gian attention:

```text
k_j = tanh(W_proj h_j + b_proj)
```

Sau đó tính attention score giữa aspect `i` và token `j`:

```text
s_(i,j) = q_i^T k_j
```

Rồi chuẩn hóa thành attention weight:

```text
alpha_(i,j) = exp(s_(i,j)) / sum_{t in valid} exp(s_(i,t))
```

Cuối cùng lấy weighted sum:

```text
r_i = sum_{j in valid} alpha_(i,j) h_j
```

`r_i` chính là representation riêng cho aspect `i`.

## 1.4. Trực giác dễ hiểu

Hãy xem review:

```text
Phòng sạch, vị trí thuận tiện nhưng nhân viên hơi thiếu nhiệt tình.
```

Với `ROOMS#CLEANLINESS`, attention có thể tập trung vào:

- `Phòng`
- `sạch`

Với `LOCATION#GENERAL`, attention có thể tập trung vào:

- `vị_trí`
- `thuận_tiện`

Với `SERVICE#GENERAL`, attention có thể tập trung vào:

- `nhân_viên`
- `thiếu`
- `nhiệt_tình`

Như vậy mỗi aspect tự “nhìn” vào phần câu liên quan nhất với nó.

## 1.5. Khác biệt cũ vs mới

| Thành phần | Bản cũ | Bản mới |
|---|---|---|
| Representation | Một vector chung cho cả review | Một vector riêng cho từng aspect |
| Khả năng phân biệt nhiều aspect | Hạn chế hơn | Tốt hơn |
| Giải thích token nào quan trọng | Khó | Dễ hơn qua attention map |
| Phù hợp với ABSA nhiều aspect | Chưa thật tối ưu | Phù hợp hơn về bản chất |

## 1.6. Vì sao cải tiến này hợp lý?

ABSA không giống sentiment analysis chung. Ở sentiment analysis thông thường,
một vector chung cho cả câu có thể đủ tốt. Nhưng ở ABSA:

- cùng một review có thể nói về nhiều aspect;
- mỗi aspect có thể mang sentiment khác nhau;
- thậm chí trái dấu trong cùng một câu.

Do đó, việc học representation riêng cho từng aspect là hợp logic hơn nhiều so
với dùng một vector duy nhất.

## 1.7. Nó giúp được gì trong thực tế?

Lợi ích chính:

- cải thiện khả năng tách nhiều aspect trong cùng review;
- hỗ trợ ACD và SPC tốt hơn;
- sinh attention map để giải thích model;
- dễ trình bày với thầy vì có thể chỉ ra model đang nhìn vào token nào.

Điểm rất quan trọng: attention map không nên được nói quá mạnh là “chứng minh
model hiểu giống con người”, nhưng nó là một **attention-based explanation**
hợp lý và bám sát kiến trúc thật của model.

---

## 2. Dual Head / Split Loss

## 2.1. Bản cũ làm gì?

Cách đơn giản của bài toán này là với mỗi aspect, dùng một classifier 4 lớp:

```text
absent / positive / negative / neutral
```

Tức là:

- aspect không xuất hiện -> `absent`
- aspect có xuất hiện và mang cảm xúc -> `positive/negative/neutral`

Điểm yếu lớn của cách này là:

- class `absent` chiếm đa số tuyệt đối;
- classifier bị kéo mạnh về hướng học “không xuất hiện”;
- sentiment branch bị ảnh hưởng vì phải học chung với `absent`.

Nói cách khác, model dễ học tốt phần “có/không có aspect”, nhưng chưa chắc học
tốt phần sentiment.

## 2.2. Bản mới thay đổi gì?

Bản mới tách bài toán thành hai nhánh:

1. **Presence Head**
   - dự đoán aspect có xuất hiện hay không
2. **Sentiment Head**
   - nếu aspect xuất hiện thì dự đoán sentiment

Đây là lý do gọi là **Dual Head**.

## 2.3. Vì sao gọi là Split Loss?

Vì loss không còn là một loss 4-class chung nữa, mà được tách thành hai phần:

```text
loss = lambda_presence * presence_loss + lambda_sentiment * sentiment_loss
```

Trong cấu hình hiện tại:

```text
lambda_presence = 1.0
lambda_sentiment = 1.0
```

### Presence loss

Presence loss là BCE cho bài toán:

```text
aspect absent hay present
```

### Sentiment loss

Sentiment loss chỉ tính trên những aspect **thực sự present**.

Nghĩa là:

- nếu aspect absent thì không ép model phân loại positive/negative/neutral;
- nếu aspect present thì mới tính sentiment loss.

Đây là điểm rất hợp lý về mặt bài toán.

## 2.4. Khác biệt cũ vs mới

| Thành phần | Bản cũ | Bản mới |
|---|---|---|
| Output | 1 classifier 4 lớp | 2 head tách riêng |
| Xử lý `absent` | Gộp chung với sentiment | Tách riêng ở Presence Head |
| Ảnh hưởng class imbalance | Nặng hơn | Giảm bớt |
| Diễn giải pipeline | Khó hơn | Rõ hơn: detect trước, sentiment sau |

## 2.5. Trực giác dễ hiểu

Review:

```text
Phòng sạch nhưng nhân viên phục vụ chậm.
```

### Với `ROOMS#CLEANLINESS`

- Presence Head: có nhắc tới aspect này không? -> Có
- Sentiment Head: nếu có thì sentiment là gì? -> Positive

### Với `SERVICE#GENERAL`

- Presence Head -> Có
- Sentiment Head -> Negative

### Với `FOOD&DRINKS#QUALITY`

- Presence Head -> Không
- Sentiment Head không cần đóng vai trò chính nữa

Như vậy pipeline tự nhiên hơn nhiều.

## 2.6. Tại sao cải tiến này tốt hơn?

Về mặt logic, ACD và SPC vốn là hai bài toán khác nhau:

- ACD: phát hiện aspect
- SPC: phân loại sentiment

Nếu bắt một classifier 4 lớp học tất cả cùng lúc, class `absent` rất dễ lấn át
phần sentiment vì nó chiếm đa số lớn. Split loss giải quyết đúng điểm nghẽn đó.

## 2.7. Liên hệ với class imbalance

Dataset VLSP 2018 Hotel có mất cân bằng mạnh:

- `absent` là class nhiều nhất;
- `neutral` rất hiếm;
- nhiều aspect bản thân đã rất hiếm.

Khi dùng split loss:

- Presence Head tập trung vào câu hỏi “có nhắc tới hay không”;
- Sentiment Head chỉ cần học trên các case có aspect thật;
- focal loss và class weight được dùng đúng nơi hơn.

Điều này làm training hợp lý hơn về mặt cấu trúc.

## 2.8. Giá trị khi báo cáo

Đây là điểm cải tiến rất tốt để trình bày vì:

- dễ giải thích về mặt bài toán;
- bám sát đúng ACD/SPC;
- thầy rất dễ hỏi và cũng rất dễ defend.

Câu trả lời ngắn gọn nhất là:

> “Bản cũ gộp absent và sentiment vào một classifier 4 lớp, nên class absent
> lấn át. Bản mới tách thành Presence Head và Sentiment Head, đúng hơn với bản
> chất của ABSA.”

---

## 3. LLM Explanation và cách đánh giá
## 3.2. Bản mới / thiết kế đúng hiện tại

Pipeline hiện tại tách rõ vai:

- **PhoBERT** = classifier
- **LLM** = explainer

Luồng:

```text
Review
-> PhoBERT dự đoán aspect + sentiment
-> LLM nhận review + prediction
-> sinh evidence + explanation + recommended_action
```

Điểm mạnh của thiết kế này:

- nhãn được giữ ổn định bởi model supervised;
- explanation được sinh tự nhiên bởi LLM;
- dễ kiểm soát lỗi hơn;
- dễ đánh giá hơn.

## 3.3. LLM sinh ra gì?

Từ output của PhoBERT, LLM sinh ra:

- `aspect`
- `sentiment`
- `evidence`
- `explanation`
- `recommended_action`

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

## 3.4. Điểm mới nằm ở đâu?

Điểm mới không chỉ là “thêm LLM”, mà là:

1. **tách vai rõ ràng giữa predictor và explainer**;
2. **đánh giá explanation bằng metric cụ thể**, thay vì chỉ demo cảm tính;
3. **kiểm tra hallucination theo nhiều tầng**.

Đây là điểm quan trọng nhất để defend.

## 3.5. Đánh giá LLM bằng gì?

Phần LLM hiện được đánh giá bằng ba nhóm:

### (a) Label Consistency

Kiểm tra:

- aspect có bị đổi không?
- sentiment có bị đổi không?

Kết quả hiện tại:

```text
Label Consistency = 100%
```

Ý nghĩa:

- LLM không làm sai nhãn PhoBERT;
- đây là bằng chứng mạnh rằng pipeline tách vai đang hoạt động đúng.

### (b) BERTScore Evidence Groundedness

Đo mức bám sát ngữ nghĩa giữa evidence do LLM sinh ra và review gốc.

Kết quả:

- Precision = `0.7709`
- Recall = `0.4396`
- F1 = `0.5535`

Điểm cần hiểu đúng:

- `0.5535` không tự nó là “rất cao”;
- nhưng **Precision 0.7709 khá cao**, cho thấy evidence bám sát review gốc;
- recall thấp hơn là hợp lý vì evidence chỉ là đoạn ngắn cho từng aspect, không
  nhằm cover toàn bộ review.

### (c) Human Rubric
Human Rubric thang diem 0,1,2
Có hai hướng chấm:

- **Faithfulness**
- **Usefulness**

Kết quả:

- Faithfulness = `1.867 / 2`
- Usefulness = `1.667 / 2`

## 3.6. Faithfulness và Usefulness khác nhau thế nào?

### Faithfulness

Đo xem explanation có:

- bịa ra thông tin không có trong review không;
- diễn giải có bám review không.

Điểm `1.867/2` nghĩa là phần lớn explanation được đánh giá là rất sát.

### Usefulness

Đo xem `recommended_action` có:

- liên quan tới vấn đề phát hiện được không;
- đủ cụ thể không;
- có khả thi không.

Điểm `1.667/2` nghĩa là phần lớn action hữu ích, nhưng đây vẫn là phần khó hơn
faithfulness.

## 3.7. Khác biệt cũ vs mới

| Khía cạnh | Cách hiểu cũ / dễ nhầm | Thiết kế hiện tại |
|---|---|---|
| Vai trò LLM | Có thể bị hiểu là predictor | Chỉ là explainer |
| Cách đánh giá | Chủ yếu demo | Có metric cụ thể |
| Kiểm soát hallucination | Mơ hồ | Có label consistency + BERTScore + human rubric |
| Vai trò trong hệ thống | Không rõ | Tạo explanation có thể trình bày cho người dùng cuối |

## 3.8. Vì sao đây là cải tiến đáng kể?

Vì dự án không chỉ dừng ở:

```text
predict label
```

mà đi tiếp tới:

```text
giải thích label đó cho người dùng cuối
```

Trong bài toán khách sạn, điều này rất quan trọng. Người quản lý không chỉ muốn
biết:

- review này là negative

mà còn muốn biết:

- negative ở aspect nào;
- bằng chứng nằm ở đâu;
- nên làm gì để cải thiện.

LLM explanation biến output kỹ thuật của PhoBERT thành thông tin có thể dùng
được trong thực tế.

## 3.9. Câu hỏi dễ bị hỏi nhất ở phần này

### “LLM có tự đổi nhãn không?”

Trả lời:

> Không. LLM explanation chỉ dùng để diễn giải output PhoBERT. Label
> consistency hiện tại là 100%.

### “0.5535 BERTScore F1 có thấp không?”

Trả lời:

> Không thể đánh giá chỉ bằng F1 đứng một mình. Phải đọc cùng Precision =
> 0.7709, Faithfulness = 1.867/2 và Label Consistency = 100%. Điều đó cho thấy
> evidence bám sát review gốc và explanation ít hallucination.

### “Recommended action có được đánh giá chưa?”

Trả lời:

> Có, bằng Human Usefulness = 1.667/2. Đây là đánh giá chủ yếu bằng người chấm,
> vì phần action khó có metric tự động mạnh như evidence.

---

## 4. Tóm tắt ngắn gọn 3 cải tiến lớn nhất

### Aspect-Aware Attention Pooling

- **Cũ:** một representation chung cho cả review
- **Mới:** một representation riêng cho từng aspect
- **Lợi ích:** phù hợp hơn với ABSA nhiều aspect, dễ giải thích bằng attention map

### Dual Head / Split Loss

- **Cũ:** một classifier 4 lớp gộp `absent` với sentiment
- **Mới:** tách Presence Head và Sentiment Head
- **Lợi ích:** đúng với bản chất ACD/SPC, giảm ảnh hưởng áp đảo của class `absent`

### LLM Explanation Evaluation

- **Cũ / dễ nhầm:** LLM có thể bị xem như predictor
- **Mới:** LLM chỉ là explainer, có đánh giá rõ bằng metric
- **Lợi ích:** biến hệ thống từ “chỉ có F1” thành một pipeline có thể giải thích
  và trình bày cho người dùng cuối

---

## 5. Một câu trả lời tổng hợp khi thầy hỏi “nhóm cải tiến gì?”

Bạn có thể trả lời ngắn gọn như sau:

> “Ba cải tiến lớn nhất của nhóm là: thứ nhất, thay representation chung bằng
> aspect-aware attention pooling để mỗi aspect có vector riêng; thứ hai, tách
> prediction thành Presence Head và Sentiment Head với split loss để xử lý đúng
> bản chất ACD/SPC và giảm ảnh hưởng của class absent; thứ ba, xây lớp LLM
> explanation đứng sau PhoBERT, đồng thời đánh giá explanation bằng label
> consistency, BERTScore và human rubric thay vì chỉ demo cảm tính.”
