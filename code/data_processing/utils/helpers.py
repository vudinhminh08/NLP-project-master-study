"""
helpers.py — Utility functions dùng chung toàn bộ dự án.
"""
import os
import json
import random

import numpy as np
import torch
from tabulate import tabulate


def set_seed(seed: int = 42) -> None:
    """Set random seed cho reproducibility (random, numpy, torch, cuda)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Trả về cuda nếu có, ngược lại cpu. In thông tin device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[Device] GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print(f"[Device] CPU only")
    return device


def save_json(data: dict, path: str) -> None:
    """Lưu dict thành JSON, tạo thư mục nếu chưa có."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: str) -> dict:
    """Load JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def log_versions() -> None:
    """In Python + PyTorch + CUDA version để đảm bảo reproducibility."""
    import sys
    print(f"[Env] Python {sys.version}")
    print(f"[Env] PyTorch {torch.__version__}")
    print(f"[Env] CUDA available: {torch.cuda.is_available()}")


def format_metrics_table(
    per_aspect_metrics: dict,
    macro_acd_f1: float,
    macro_spc_f1: float,
    sort_by: str = "acd_f1",
    highlight_rare: list = None,
) -> str:
    """
    Tạo bảng ASCII đẹp từ metrics dict.

    Args:
        per_aspect_metrics: dict aspect_name → {acd_f1, spc_f1, support, ...}
        macro_acd_f1: float
        macro_spc_f1: float
        sort_by: cột sort ('acd_f1' hoặc 'spc_f1'), tăng dần để thấy aspect yếu
        highlight_rare: list tên aspects cần highlight (dùng * prefix)

    Returns:
        str bảng ASCII sẵn để print
    """
    highlight_rare = highlight_rare or []
    rows = []
    for aspect, m in sorted(
        per_aspect_metrics.items(), key=lambda x: x[1].get(sort_by, 0)
    ):
        marker = "* " if aspect in highlight_rare else "  "
        rows.append([
            marker + aspect,
            f"{m.get('acd_f1', 0):.4f}",
            f"{m.get('acd_precision', 0):.4f}",
            f"{m.get('acd_recall', 0):.4f}",
            f"{m.get('spc_f1', 0) or 0:.4f}",
            m.get("support", 0),
        ])
    # Footer
    rows.append(["─" * 35, "─" * 8, "─" * 8, "─" * 8, "─" * 8, "─" * 6])
    rows.append([
        "MACRO",
        f"{macro_acd_f1:.4f}",
        "", "",
        f"{macro_spc_f1:.4f}",
        "",
    ])
    headers = ["Aspect", "ACD-F1", "ACD-P", "ACD-R", "SPC-F1", "Support"]
    note = "* = rare aspect (<100 samples)"
    return tabulate(rows, headers=headers, tablefmt="simple") + f"\n{note}"
