# SVM + TF-IDF Baseline

## Config
| Tham so | Gia tri |
|---------|---------|
| Model | LinearSVC (sklearn) |
| Features | TF-IDF, unigram + bigram |
| max_features | 50000 |
| C | 1.0 |
| class_weight | balanced |
| Preprocessing | lowercase + remove special chars |
| Word segmentation | Khong (co tinh giu don gian) |

## Ket qua

| Split | ACD F1 | SPC F1 | Combined F1 |
|-------|--------|--------|-------------|
| Dev | 0.4045 | 0.2364 | 0.3204 |
| **Test** | **0.4074** | **0.2272** | **0.3173** |

## So sanh

| Phuong phap | ACD F1 | Combined F1 | Ghi chu |
|-------------|--------|-------------|---------|
| SVM ours | 0.4074 | 0.3173 | File nay |
| PhoBERT cls_only + VnCoreNLP | 0.6360 | 0.5543 | Supervised baseline chinh |
| LLM + RAG k=8 | 0.4034 | 0.3532 | LLM prediction benchmark |
| SOTA (Huynh 2022) | 0.8255 | 0.7732 | Upper bound |

## Ghi chu
- SVM khong dung word segmentation -> feature extraction kem hon co the
- class_weight='balanced' la cach xu ly imbalance tuong duong weighted loss
- 34 classifiers doc lap -> khong chia se representation giua aspects (khac PhoBERT multi-task)
