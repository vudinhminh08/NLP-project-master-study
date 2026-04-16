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

## Giai thich tham so va cach chon

| Tham so | Ly do chon |
|---------|------------|
| `LinearSVC` | SVM tuyen tinh phu hop voi vector TF-IDF sparse, train nhanh tren 34 classifiers va la baseline truyen thong de giai thich. |
| `unigram + bigram` | Unigram bat tu don nhu `phong`, `sach`; bigram bat cum tu co nghia hon nhu `phong sach`, `nhan vien`, `gan trung tam`. |
| `max_features=50000` | Gioi han so feature de tranh ma tran qua lon. 50k du lon de giu cac tu/cum tu pho bien trong 5,600 review nhung van train nhanh. |
| `C=1.0` | Gia tri mac dinh can bang giua margin va loi phan loai. Baseline can on dinh, khong toi uu qua muc de tranh bien thanh mot phase tuning rieng. |
| `class_weight=balanced` | Dataset lech nhan manh, nhat la `absent` chiem da so. Balanced weight giup SVM khong hoc cach doan tat ca la `absent`. |
| Khong word segmentation | Co tinh giu baseline doc lap va don gian. Neu baseline thap hon PhoBERT thi co the giai thich do thieu contextual embedding va thieu preprocessing tieng Viet sau. |

Cach chon tham so cua SVM mang tinh baseline, khong nham toi uu tuyet doi.
Muc tieu la tao moc so sanh hop ly: model co the train nhanh, de giai thich,
va du khac PhoBERT de cho thay gia tri cua pretrained representation.

## Ket qua

| Split | ACD F1 | SPC F1 | Combined F1 |
|-------|--------|--------|-------------|
| Dev | 0.4045 | 0.2364 | 0.3204 |
| **Test** | **0.4074** | **0.2272** | **0.3173** |

## Quy trinh xu ly

SVM duoc giu lam baseline truyen thong de tra loi cau hoi: neu khong dung
pretrained language model, chi dung feature sparse thi bai toan dat duoc den
muc nao. Pipeline co chu dich don gian:

1. Doc `data/train.csv`, `data/dev.csv`, `data/test.csv`.
2. Lay cot `Review` goc, khong dung `processed_review` de giu baseline doc lap
   voi phase PhoBERT.
3. Chuan hoa text muc nhe: lowercase, xoa ky tu dac biet, giam nhieu be mat.
4. Bien doi text sang TF-IDF unigram + bigram.
5. Train 34 bo phan loai doc lap, moi aspect la mot bai toan 4-class:
   `absent`, `positive`, `negative`, `neutral`.
6. Tinh ACD F1, SPC F1 va Combined F1 tren dev/test.

Thiet ke nay de baseline de giai thich: TF-IDF bat tu/cum tu, SVM tach class
bang hyperplane tuyen tinh. Neu ket qua thap hon PhoBERT thi co the quy ve viec
SVM khong co contextual embedding va khong chia se thong tin giua aspects.

## Code chinh

File chinh: `code/svm_baseline/svm_baseline.py`.

Thanh phan quan trong:

- `preprocess_for_tfidf`: tien xu ly text nhe cho TF-IDF.
- `TfidfVectorizer`: tao dac trung sparse unigram + bigram.
- `train_aspect_classifiers`: train tung classifier cho 34 aspects.
- `predict_all_aspects`: gom du doan cua 34 classifiers thanh ma tran `[N, 34]`.
- `evaluate_predictions`: tinh metric theo cung logic voi cac phase khac.

Baseline khong dung VnCoreNLP va khong dung LLM. Day la moc thap nhung can
thiet de chung minh cac mo hinh sau thuc su co gia tri.

## So sanh

| Phuong phap | ACD F1 | Combined F1 | Ghi chu |
|-------------|--------|-------------|---------|
| SVM ours | 0.4074 | 0.3173 | File nay |
| PhoBERT cls_only + VnCoreNLP | 0.6360 | 0.5543 | Supervised baseline chinh |
| LLM + RAG k=8 | 0.4034 | 0.3532 | LLM prediction benchmark |
| SOTA (Huynh 2022) | 0.8255 | 0.7732 | Upper bound |

## Phan tich ket qua chi tiet

SVM dat ACD F1 test `0.4074`, cao hon SPC F1 `0.2272`. Dieu nay hop ly vi ACD
chi can phat hien aspect co xuat hien hay khong, trong khi SPC phai phan biet
positive/negative/neutral cho aspect da xuat hien. Voi TF-IDF sparse, cac cum
tu nhu `phong sach`, `nhan vien`, `gan trung tam` co the giup detect aspect,
nhung sentiment lai phu thuoc ngu canh va gan ket aspect-sentiment nen kho hon.

Combined F1 `0.3173` thap hon LLM + RAG k=8 `0.3532` va thap xa PhoBERT
`0.5543`. Ket qua nay cho thay baseline truyen thong khong du manh cho ABSA
tieng Viet nhieu aspect. Tuy nhien, no van co gia tri bao cao:

- Lam moc so sanh toi thieu cho cac huong sau.
- Cho thay bai toan khong the giai quyet tot chi bang bag-of-words.
- Giai thich duoc vi sao can pretrained representation nhu PhoBERT.

## Ghi chu
- SVM khong dung word segmentation -> feature extraction kem hon co the
- class_weight='balanced' la cach xu ly imbalance tuong duong weighted loss
- 34 classifiers doc lap -> khong chia se representation giua aspects (khac PhoBERT multi-task)
