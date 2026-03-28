# ABSA VLSP 2018 Hotel

Aspect-Based Sentiment Analysis trên tập đánh giá khách sạn tiếng Việt VLSP 2018.
Bài tập lớn môn Xử lý Ngôn ngữ Tự nhiên — HUST.

## Cài đặt

```bash
pip install -r requirements.txt
```

## Download data

```bash
git clone https://github.com/ds4v/absa-vlsp-2018.git
cp absa-vlsp-2018/datasets/vlsp2018_hotel/train.csv data/
cp absa-vlsp-2018/datasets/vlsp2018_hotel/dev.csv data/
cp absa-vlsp-2018/datasets/vlsp2018_hotel/test.csv data/
```

## Chạy

### Tuần 1 — Data Foundation

```bash
python code/week1/step1_eda.py           # EDA + class weights
python code/week1/step3_preprocessing.py  # Preprocessing + cache
python code/week1/step2_dataloader.py    # Test DataLoader
python code/week1/step4_eval.py          # Test eval script
```

## Kiến trúc 3 tầng

| Tầng | Phương pháp | Mục tiêu |
|------|-------------|----------|
| 1 | Multi-task PhoBERT (concat 4 layers) | Baseline mạnh, SOTA-level |
| 2 | Few-shot LLM + CoT | So sánh với supervised |
| 3 | Aspect-aware RAG | Cải thiện LLM bằng retrieval |

## Tham khảo

- SOTA: Huynh et al. (2022) IEEE MAPR — ACD F1=82.55%, Combined F1=77.32%
- Dataset: ds4v/absa-vlsp-2018 (CSV format)
