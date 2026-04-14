# PhoBERT + LLM Explainability Plan

Updated: 2026-04-15

This is the Git-tracked context file for the final project direction. The local
Vietnamese notes under `Tài liệu/` may be ignored by Git, so future agents should
use this file when continuing from a fresh clone.

## Final Direction

Do not position the LLM as the main ABSA classifier.

The final direction is:

```text
PhoBERT / calibrated PhoBERT = prediction engine
LLM + RAG = explanation, evidence extraction, and error analysis layer
```

Rationale:

- VLSP 2018 Hotel ABSA has 34 aspects and 4 labels per aspect.
- The dataset is small and heavily imbalanced; `neutral` is especially rare.
- LLM few-shot/RAG must detect aspects, classify sentiment, and avoid
  hallucination from a short prompt, which performed poorly in experiments.
- Supervised PhoBERT is much more reliable for prediction because it is trained
  directly on the dataset label scheme.

Main retained results:

| Method | Combined F1 | Notes |
|---|---:|---|
| SVM + TF-IDF | 0.3173 | Traditional baseline |
| LLM + RAG k=8 | 0.3532 | Better than random ICL but far below PhoBERT |
| PhoBERT cls_only | 0.5543 | Strong supervised baseline |
| PhoBERT + LLM explanation | N/A | LLM explains PhoBERT predictions instead of replacing them |

Older ADD/cascade/verifier/ensemble experiments are archived under `draft/`.
They can be mentioned briefly as negative experiments, but they should not be
main methods in the final report.

Report claim:

```text
LLM does not replace supervised PhoBERT for prediction in this task. Its practical
value is explainability, evidence extraction, and error analysis.
```

## Research Narrative

The report should tell this story:

1. Build traditional baseline: SVM + TF-IDF.
2. Fine-tune PhoBERT for supervised ABSA prediction.
3. Try LLM ICL/RAG as a predictor.
4. Show that RAG improves over random ICL but remains far below PhoBERT.
5. Show that LLM prediction remains much weaker than supervised PhoBERT.
6. Conclude that PhoBERT should remain the prediction engine.
7. Reposition LLM as the explanation and error-analysis layer.

Recommended method name:

```text
PhoBERT-Centric ABSA with LLM-based Explainability
```

or:

```text
Explainable PhoBERT-RAG Framework for Vietnamese Hotel ABSA
```

## Scope Before App Work

Do not build the app/dashboard yet.

The next implementation phase should stop at:

- Formatting PhoBERT predictions for explanation.
- Designing LLM explanation prompts.
- Extracting evidence spans from reviews.
- Using RAG examples to stabilize explanations.
- Producing report-ready explanation samples.
- Producing error analysis artifacts.
- Evaluating explanation quality with heuristics/manual rubric.

Do not:

- Retrain PhoBERT.
- Continue spending time trying to make LLM verifier beat F1.
- Build Streamlit/Gradio UI in this phase.

## Proposed Artifacts

Code:

```text
code/llm_explainability/
├── __init__.py
├── prediction_formatter.py
├── explanation_prompts.py
├── llm_explainer.py
├── evidence_checker.py
├── error_analyzer.py
└── run_explainability.py
```

Outputs:

```text
outputs/results/llm_explainability/
├── explanation_samples.json
├── explanation_quality_report.json
├── error_analysis_report.json
├── aspect_error_summary.json
└── report_ready_examples.md
```

If time is short, collapse the implementation into:

```text
code/llm_explainability/llm_explainer.py
code/llm_explainability/run_explainability.py
```

## Explanation Input

The LLM receives only present PhoBERT predictions. It must not add new aspects.

Example input:

```json
{
  "review": "Phòng sạch, vị trí tốt nhưng nhân viên lễ tân hơi khó chịu.",
  "predictions": [
    {
      "aspect": "ROOMS#CLEANLINESS",
      "sentiment": "positive",
      "confidence": 0.82
    },
    {
      "aspect": "LOCATION#GENERAL",
      "sentiment": "positive",
      "confidence": 0.76
    },
    {
      "aspect": "SERVICE#GENERAL",
      "sentiment": "negative",
      "confidence": 0.64
    }
  ],
  "rag_examples": [
    {
      "review": "...",
      "present_labels": {
        "ROOMS#CLEANLINESS": "positive",
        "SERVICE#GENERAL": "negative"
      }
    }
  ]
}
```

## Explanation Output

Required JSON shape:

```json
{
  "items": [
    {
      "aspect": "ROOMS#CLEANLINESS",
      "sentiment": "positive",
      "evidence": "Phòng sạch",
      "explanation": "Khách hàng khen phòng sạch nên aspect vệ sinh phòng được gán positive.",
      "evidence_confidence": "high",
      "evidence_uncertain": false
    }
  ],
  "overall_summary": "Review khen phòng và vị trí nhưng phàn nàn về dịch vụ lễ tân.",
  "recommended_action": "Khách sạn nên kiểm tra thái độ phục vụ tại quầy lễ tân."
}
```

Validation rules:

- Aspect must be one of the PhoBERT present predictions.
- Sentiment must match PhoBERT's predicted sentiment.
- LLM must not add new aspects.
- Evidence should be a phrase from the original review.
- If evidence cannot be found, set `evidence_uncertain=true`.

## Prompt Design

System prompt:

```text
Bạn là trợ lý giải thích kết quả phân tích cảm xúc theo khía cạnh cho review khách sạn tiếng Việt.

Bạn KHÔNG phải là classifier. Aspect và sentiment đã được PhoBERT dự đoán.
Nhiệm vụ của bạn là:
1. Trích bằng chứng trong review cho từng prediction.
2. Giải thích ngắn gọn vì sao prediction hợp lý.
3. Không thêm aspect ngoài danh sách prediction.
4. Không sửa sentiment.
5. Nếu không tìm thấy bằng chứng rõ, đánh dấu evidence_uncertain=true.

Chỉ trả JSON hợp lệ.
```

User prompt should include:

- Original review.
- PhoBERT present predictions.
- RAG examples.
- Required JSON schema.

## Evidence Checking

Implement a simple evidence validator:

```python
def evidence_in_review(evidence, review):
    return normalize(evidence) in normalize(review)
```

Recommended normalization:

- Lowercase.
- Strip punctuation.
- Collapse whitespace.

If evidence fails validation:

```json
{
  "evidence": "",
  "evidence_uncertain": true,
  "explanation": "Prediction này được PhoBERT đưa ra nhưng LLM không tìm được bằng chứng rõ trong câu."
}
```

These uncertain cases are useful for error analysis.

## RAG for Explanation

RAG is not used to predict labels in the final direction. It is used to make LLM
explanations more consistent with annotated dataset examples.

Retrieval plan:

1. Use `ABSARetriever`.
2. Retrieve `k=4` or `k=6` similar train reviews.
3. Prefer examples that overlap with the current predicted aspects.
4. Format only present labels to keep prompts compact.

## Error Analysis Plan

Generate:

```json
{
  "top_acd_false_negative_aspects": [],
  "top_acd_false_positive_aspects": [],
  "top_spc_confused_aspects": [],
  "neutral_error_summary": {},
  "evidence_uncertain_cases": [],
  "representative_errors": []
}
```

Report-ready sections:

1. Frequent ACD errors.
2. Frequent SPC errors.
3. Neutral class issue.
4. Example where PhoBERT is correct and LLM explanation is useful.
5. Example where PhoBERT may be wrong and `evidence_uncertain` helps flag it.

## Explanation Quality Evaluation

Heuristic metrics:

- JSON parse rate.
- Aspect preservation rate.
- Sentiment preservation rate.
- Evidence substring match rate.
- Evidence uncertain rate.
- Average explanation length.

Targets:

```text
JSON parse rate >= 95%
Aspect preservation rate = 100%
Sentiment preservation rate = 100%
```

Manual rubric for 20-30 examples:

| Criterion | 0 | 1 | 2 |
|---|---|---|---|
| Evidence correctness | wrong | partial | clearly correct |
| Explanation quality | wrong | generic | clear and correct |
| Hallucination | hallucinated | slight inference | no hallucination |
| User usefulness | not useful | medium | useful |

## Implementation Checklist

Phase 1: Prediction formatting

- [ ] Load best PhoBERT checkpoint.
- [ ] Predict sample/test reviews.
- [ ] Convert vector `[34]` to present prediction list.
- [ ] Include confidence if probabilities are available.
- [ ] Save `phobert_predictions_sample.json`.

Phase 2: Explanation prompts

- [ ] Create prompt module.
- [ ] Create parser and validator.
- [ ] Smoke test 5-10 reviews.

Phase 3: Evidence validation

- [ ] Implement normalize function.
- [ ] Implement substring evidence check.
- [ ] Mark invalid evidence uncertain.

Phase 4: RAG examples

- [ ] Reuse `ABSARetriever`.
- [ ] Retrieve similar examples.
- [ ] Format examples with present labels only.
- [ ] Compare with/without RAG on a small sample.

Phase 5: Error analysis

- [ ] Count false positives/false negatives.
- [ ] Count SPC confusion cases.
- [ ] Extract representative examples.
- [ ] Generate `error_analysis_report.json`.
- [ ] Generate `report_ready_examples.md`.

Phase 6: Report integration

- [ ] Add final architecture diagram.
- [ ] Add "why LLM prediction failed" analysis.
- [ ] Add "why LLM explanation is useful" section.
- [ ] Add explanation examples.
- [ ] Add error analysis examples.
