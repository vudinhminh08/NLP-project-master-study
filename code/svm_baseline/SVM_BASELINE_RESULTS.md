# SVM + TF-IDF Results

| Split | ACD F1 | SPC F1 | Combined F1 |
|---|---:|---:|---:|
| Dev | 0.4045 | 0.2364 | 0.3204 |
| Test | 0.4074 | 0.2272 | 0.3173 |

## Comparison

| Method | ACD F1 | SPC F1 | Combined F1 | Note |
|---|---:|---:|---:|---|
| SVM + TF-IDF | 0.4074 | 0.2272 | 0.3173 | Traditional baseline |
| PhoBERT cls_only + VnCoreNLP | 0.6360 | 0.4727 | 0.5543 | Main supervised model |
| LLM + RAG k=8 | 0.4034 | 0.3031 | 0.3532 | Best LLM predictor |

SVM remains useful as the classical baseline, but it is clearly below the
supervised PhoBERT model used as the prediction engine in the final system.
