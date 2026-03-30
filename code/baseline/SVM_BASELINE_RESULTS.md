# Bang Tong Hop Ket Qua SVM + TF-IDF

## 1) Ket qua SVM + TF-IDF (ours)

| Split | ACD F1 | SPC F1 | Combined F1 |
|------|------:|------:|------:|
| Dev  | 0.4045 | 0.2364 | 0.3204 |
| Test | 0.4074 | 0.2272 | 0.3173 |

## 2) So sanh nhanh voi PhoBERT tuan 2 (de tham chieu)

| Model | ACD F1 (Test) | SPC F1 (Test) | Combined F1 (Test) |
|------|------:|------:|------:|
| SVM + TF-IDF (ours) | 0.4074 | 0.2272 | 0.3173 |
| PhoBERT concat_4_layers (week2_results_version1) | 0.3592 | 0.2371 | 0.2981 |
| PhoBERT cls_only (ablation) | 0.3349 | 0.2014 | 0.2681 |

## 3) Nhan xet nhanh

- SVM hien tai dang cao hon PhoBERT v1 ve **ACD** va **Combined**.
- PhoBERT concat_4_layers nhinh hon SVM o **SPC**.
- Chenh lech Combined: `SVM - PhoBERT_concat4 = +0.0192`.

## 4) Nguon so lieu

- `outputs/results/svm_baseline_dev_metrics.json`
- `outputs/results/svm_baseline_test_metrics.json`
- `/Users/macbookpro/Documents/Master-study/NLP/Result week 2/week2_results_version1/results/week2_test_metrics.json`
- `/Users/macbookpro/Documents/Master-study/NLP/Result week 2/week2_results_version1/results_cls_only/week2_test_metrics.json`

