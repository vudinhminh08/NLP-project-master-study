"""
step2_dataloader.py — PyTorch Dataset + DataLoader cho ABSA VLSP 2018.

Tái sử dụng cho tuần 2 và 3.

Chạy từ root project:
    python code/data_processing/step2_dataloader.py
"""
import os
import sys
from typing import Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from utils.constants import (
    ASPECT_COLUMNS,
    PHOBERT_MODEL_NAME,
    MAX_SEQ_LEN,
    TRAIN_PATH,
    DEV_PATH,
    TEST_PATH,
    TRAIN_PREPROCESSED,
    DEV_PREPROCESSED,
    TEST_PREPROCESSED,
)
from utils.helpers import set_seed, log_versions


# ─── Dataset ─────────────────────────────────────────────────────────────────

class ABSADataset(torch.utils.data.Dataset):
    """
    Dataset cho ABSA VLSP 2018.

    Args:
        df:        DataFrame với cột text và 34 cột aspect
        tokenizer: PhoBERT AutoTokenizer
        max_len:   max sequence length
        text_col:  tên cột text ('processed_review' hoặc 'Review')

    __getitem__ trả về:
        input_ids:      LongTensor [max_len]
        attention_mask: LongTensor [max_len]
        labels:         LongTensor [34]  — values 0-3
        review_text:    str  — giữ lại để debug/error analysis
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        max_len: int = MAX_SEQ_LEN,
        text_col: str = "processed_review",
    ) -> None:
        # Fallback: nếu processed_review không tồn tại, dùng Review
        if text_col not in df.columns:
            print(f"[WARN] Cột '{text_col}' không tồn tại, dùng 'Review'")
            text_col = "Review"
        self.texts     = df[text_col].astype(str).tolist()
        self.labels    = df[ASPECT_COLUMNS].values.astype(int)  # [N, 34]
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
            "review_text":    self.texts[idx],
        }


# ─── DataLoader factory ──────────────────────────────────────────────────────

def create_dataloaders(
    train_path: str,
    dev_path: str,
    test_path: str,
    tokenizer,
    batch_size: int = 16,
    max_len: int = MAX_SEQ_LEN,
    num_workers: int = 2,
    use_preprocessed: bool = True,
) -> tuple:
    """
    Tạo DataLoader cho train/dev/test.

    Args:
        train_path:       path đến train CSV
        dev_path:         path đến dev CSV
        test_path:        path đến test CSV
        tokenizer:        PhoBERT tokenizer
        batch_size:       batch size
        max_len:          max sequence length
        num_workers:      số worker processes
        use_preprocessed: dùng cột 'processed_review' hay 'Review'

    Returns:
        tuple (train_loader, dev_loader, test_loader) — None nếu file không tồn tại
    """
    text_col = "processed_review" if use_preprocessed else "Review"
    loaders = []

    for path, shuffle, name in [
        (train_path, True,  "train"),
        (dev_path,   False, "dev"),
        (test_path,  False, "test"),
    ]:
        if not os.path.exists(path):
            print(f"[WARN] {path} không tồn tại — loader '{name}' = None")
            loaders.append(None)
            continue

        df = pd.read_csv(path)
        ds = ABSADataset(df, tokenizer, max_len=max_len, text_col=text_col)
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )
        print(f"[DataLoader] {name}: {len(ds)} samples, {len(loader)} batches")
        loaders.append(loader)

    return tuple(loaders)


# ─── Sanity check ────────────────────────────────────────────────────────────

def verify_dataloader(dataloader: DataLoader, split_name: str) -> None:
    """
    Sanity check: shape, label range, NaN, sample review.

    Args:
        dataloader: DataLoader to verify
        split_name: name for logging
    """
    batch = next(iter(dataloader))

    print(f"\n[Verify] {split_name}")
    print(f"  input_ids:      {batch['input_ids'].shape}")
    print(f"  attention_mask: {batch['attention_mask'].shape}")
    print(f"  labels:         {batch['labels'].shape}")

    labels = batch["labels"]
    assert labels.min() >= 0 and labels.max() <= 3, "Label ngoài range [0,3]!"
    assert not torch.isnan(batch["input_ids"].float()).any(), "NaN trong input_ids!"
    print(f"  Label range: [{labels.min()}, {labels.max()}] ✓")
    print(f"\n  Sample review:\n  {batch['review_text'][0][:200]}")
    aspects_present = [ASPECT_COLUMNS[i] for i, v in enumerate(labels[0]) if v > 0]
    print(f"  Aspects mentioned: {aspects_present}")


# ─── Main (test dataloader) ──────────────────────────────────────────────────

def main() -> None:
    set_seed(42)
    log_versions()

    # Kiểm tra constants
    from utils.constants import NUM_ASPECTS
    print(f"\n[Constants] NUM_ASPECTS = {NUM_ASPECTS}")
    assert NUM_ASPECTS == 34, f"Expected 34, got {NUM_ASPECTS}"
    print("[Constants] OK ✓")

    # Load tokenizer
    try:
        from transformers import AutoTokenizer
        print(f"\n[Tokenizer] Loading {PHOBERT_MODEL_NAME}...")
        tokenizer = AutoTokenizer.from_pretrained(PHOBERT_MODEL_NAME)
        print("[Tokenizer] Loaded ✓")
    except Exception as e:
        print(f"[ERROR] Không thể load tokenizer: {e}")
        print("  → Đảm bảo đã cài: pip install transformers")
        print("  → Cần internet để download PhoBERT lần đầu")
        return

    # Dùng preprocessed nếu có, fallback về raw
    train_p = TRAIN_PREPROCESSED if os.path.exists(TRAIN_PREPROCESSED) else TRAIN_PATH
    dev_p   = DEV_PREPROCESSED   if os.path.exists(DEV_PREPROCESSED)   else DEV_PATH
    test_p  = TEST_PREPROCESSED  if os.path.exists(TEST_PREPROCESSED)  else TEST_PATH

    use_preprocessed = os.path.exists(TRAIN_PREPROCESSED)
    if not use_preprocessed:
        print("\n[INFO] Preprocessed data chưa có, dùng raw data")
        print("       → Chạy step3_preprocessing.py trước để có processed_review")

    train_loader, dev_loader, test_loader = create_dataloaders(
        train_p, dev_p, test_p,
        tokenizer=tokenizer,
        batch_size=16,
        max_len=MAX_SEQ_LEN,
        num_workers=0,  # 0 để tránh lỗi trên Windows/macOS khi test
        use_preprocessed=use_preprocessed,
    )

    if train_loader is not None:
        verify_dataloader(train_loader, "train")
    if dev_loader is not None:
        verify_dataloader(dev_loader, "dev")

    print("\n✅ DataLoader verification hoàn tất.")


if __name__ == "__main__":
    main()
