import os
import json
import random

import numpy as np
import torch
from tabulate import tabulate


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[Device] GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print(f"[Device] CPU only")
    return device


def save_json(data: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def log_versions() -> None:
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
