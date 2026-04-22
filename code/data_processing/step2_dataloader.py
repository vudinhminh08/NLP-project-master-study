import os
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from utils.constants import (
    ASPECT_COLUMNS,
    RARE_ASPECTS,
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




class ABSADataset(torch.utils.data.Dataset):

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        max_len: int = MAX_SEQ_LEN,
        text_col: str = "processed_review",
    ) -> None:
        if text_col not in df.columns:
            print(f"[WARN] Cột '{text_col}' không tồn tại, dùng 'Review'")
            text_col = "Review"
        self.texts     = df[text_col].astype(str).tolist()
        self.labels    = df[ASPECT_COLUMNS].values.astype(int)
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


def _build_rare_aspect_sampler(
    labels: torch.Tensor,
    rare_indices: list,
    alpha: float = 2.0,
    power: float = 1.0,
) -> WeightedRandomSampler:
    rare_count = (labels[:, rare_indices] > 0).sum(dim=1).float()
    sample_weights = (1.0 + alpha * rare_count).pow(power)
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )




def create_dataloaders(
    train_path: str,
    dev_path: str,
    test_path: str,
    tokenizer,
    batch_size: int = 16,
    max_len: int = MAX_SEQ_LEN,
    num_workers: int = 2,
    use_preprocessed: bool = True,
    use_rare_oversampling: bool = False,
    rare_oversample_alpha: float = 2.0,
    rare_oversample_power: float = 1.0,
) -> tuple:
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

        sampler = None
        effective_shuffle = shuffle
        if name == "train" and use_rare_oversampling:
            rare_indices = [
                i for i, asp in enumerate(ASPECT_COLUMNS)
                if asp in RARE_ASPECTS
            ]
            labels_tensor = torch.tensor(ds.labels, dtype=torch.long)
            sampler = _build_rare_aspect_sampler(
                labels_tensor,
                rare_indices=rare_indices,
                alpha=rare_oversample_alpha,
                power=rare_oversample_power,
            )
            effective_shuffle = False

        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=effective_shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
        )
        if sampler is not None:
            print(
                f"[DataLoader] {name}: {len(ds)} samples, {len(loader)} batches, "
                f"rare_oversampling=ON (alpha={rare_oversample_alpha}, power={rare_oversample_power})"
            )
        else:
            print(f"[DataLoader] {name}: {len(ds)} samples, {len(loader)} batches")
        loaders.append(loader)

    return tuple(loaders)




def verify_dataloader(dataloader: DataLoader, split_name: str) -> None:
    batch = next(iter(dataloader))

    print(f"\n[Verify] {split_name}")
    print(f"  input_ids:      {batch['input_ids'].shape}")
    print(f"  attention_mask: {batch['attention_mask'].shape}")
    print(f"  labels:         {batch['labels'].shape}")

    labels = batch["labels"]
    assert labels.min() >= 0 and labels.max() <= 3, "Label ngoài range [0,3]!"
    assert not torch.isnan(batch["input_ids"].float()).any(), "NaN trong input_ids!"
    print(f"  Label range: [{labels.min()}, {labels.max()}] ")
    print(f"\n  Sample review:\n  {batch['review_text'][0][:200]}")
    aspects_present = [ASPECT_COLUMNS[i] for i, v in enumerate(labels[0]) if v > 0]
    print(f"  Aspects mentioned: {aspects_present}")




def main() -> None:
    set_seed(42)
    log_versions()


    from utils.constants import NUM_ASPECTS
    print(f"\n[Constants] NUM_ASPECTS = {NUM_ASPECTS}")
    assert NUM_ASPECTS == 34, f"Expected 34, got {NUM_ASPECTS}"
    print("[Constants] OK ")


    try:
        from transformers import AutoTokenizer
        print(f"\n[Tokenizer] Loading {PHOBERT_MODEL_NAME}...")
        tokenizer = AutoTokenizer.from_pretrained(PHOBERT_MODEL_NAME)
        print("[Tokenizer] Loaded ")
    except Exception as e:
        print(f"[ERROR] Không thể load tokenizer: {e}")
        print("  → Đảm bảo đã cài: pip install transformers")
        print("  → Cần internet để download PhoBERT lần đầu")
        return


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
        num_workers=0,
        use_preprocessed=use_preprocessed,
    )

    if train_loader is not None:
        verify_dataloader(train_loader, "train")
    if dev_loader is not None:
        verify_dataloader(dev_loader, "dev")

    print("\nDataLoader verification hoàn tất.")


if __name__ == "__main__":
    main()
