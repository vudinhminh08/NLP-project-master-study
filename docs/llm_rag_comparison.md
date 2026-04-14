# LLM + RAG Comparison

Tài liệu này chỉ giữ phần cần thiết cho hướng LLM + RAG trong báo cáo chính.

## Main Result

| Method | k | ACD F1 | SPC F1 | Combined F1 | Note |
|---|---:|---:|---:|---:|---|
| ICL GPT-4o-mini | 8 | 0.3504 | 0.2567 | 0.3035 | Few-shot random examples |
| RAG GPT-4o-mini | 2 | 0.3793 | 0.2866 | 0.3330 | Retrieved examples |
| RAG GPT-4o-mini | 4 | 0.3866 | 0.2883 | 0.3375 | Retrieved examples |
| **RAG GPT-4o-mini** | **8** | **0.4034** | **0.3031** | **0.3532** | Best LLM predictor |
| RAG GPT-4o-mini | 16 | 0.3988 | 0.2879 | 0.3433 | Context overload / diminishing returns |

## Interpretation

RAG improves LLM prediction compared with random in-context examples:

```text
ICL k=8 Combined F1 = 0.3035
RAG k=8 Combined F1 = 0.3532
Delta = +0.0497
```

However, even the best LLM + RAG run remains far below supervised PhoBERT:

```text
PhoBERT cls_only + VnCoreNLP Combined F1 = 0.5543
LLM + RAG k=8 Combined F1 = 0.3532
Gap = -0.2011
```

## Report Claim

LLM + RAG is useful as an experimental direction because retrieval gives a clear
improvement over plain few-shot prompting. It should not be positioned as the
main classifier because supervised PhoBERT is much stronger for this dataset.

The final system therefore uses:

```text
PhoBERT = prediction
LLM + RAG = explanation and evidence extraction
```
