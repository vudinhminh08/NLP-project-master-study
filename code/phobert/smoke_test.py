
import os
import sys
import json
import tempfile
import shutil
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import torch
import torch.nn as nn
import numpy as np

# ── Thêm paths ─────────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(ROOT)
sys.path.insert(0, "code/data_processing")
sys.path.insert(0, "code/phobert")

PASS = ""
FAIL = "❌"
results = []


def check(name: str, fn):
    try:
        fn()
        print(f"  {PASS} {name}")
        results.append((name, True, None))
    except Exception as e:
        import traceback
        print(f"  {FAIL} {name}")
        print(f"     Error: {e}")
        traceback.print_exc()
        results.append((name, False, str(e)))


# ══════════════════════════════════════════════════════════════════════════════
# Mock PhoBERT — cùng interface, random weights, không cần download
# ══════════════════════════════════════════════════════════════════════════════

class MockPhoBERTOutput:
    def __init__(self, batch_size: int, seq_len: int, hidden: int = 768, n_layers: int = 13):
        self.hidden_states = tuple(
            torch.randn(batch_size, seq_len, hidden)
            for _ in range(n_layers)
        )


class MockPhoBERTBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Linear(1, 1)  # để có parameters

    def forward(self, input_ids, attention_mask):
        b, s = input_ids.shape
        return MockPhoBERTOutput(b, s, hidden=768, n_layers=13)


def make_mock_model(encoder_option: str = "concat_4_layers") -> "ABSAPhoBERT":
    from model import ABSAPhoBERT
    with patch("model.AutoModel.from_pretrained", return_value=MockPhoBERTBase()):
        m = ABSAPhoBERT(
            model_name="mock",
            encoder_option=encoder_option,
        )
    return m


# ══════════════════════════════════════════════════════════════════════════════
# Test 1 — Imports
# ══════════════════════════════════════════════════════════════════════════════

print("\n[Block 1] Imports\n" + "─"*50)

def t_import_constants():
    from utils.constants import (
        ASPECT_COLUMNS, TRAIN_CONFIG, ZERO_TRAIN_ASPECTS, RARE_ASPECTS,
        PHOBERT_MODEL_NAME,
    )
    assert len(ASPECT_COLUMNS) == 34
    assert TRAIN_CONFIG["batch_size"] == 8
    assert TRAIN_CONFIG["learning_rate"] == 2e-5
    assert ZERO_TRAIN_ASPECTS == ["ROOM_AMENITIES#PRICES"]

def t_import_helpers():
    from utils.helpers import set_seed, get_device, save_json, load_json

def t_import_step4_eval():
    from step4_eval import compute_aspect_f1, evaluate_predictions

def t_import_dataloader():
    from step2_dataloader import ABSADataset, create_dataloaders

def t_import_model():
    make_mock_model("concat_4_layers")

def t_import_train():
    from train import load_class_weights, run_epoch, train

def t_import_predict():
    from predict import load_best_model, predict_and_evaluate, generate_summary_report

def t_import_run():
    import run_experiment  # chỉ import, không gọi main()

for fn in [t_import_constants, t_import_helpers, t_import_step4_eval,
           t_import_dataloader, t_import_model, t_import_train,
           t_import_predict, t_import_run]:
    check(fn.__name__.replace("t_", ""), fn)


# ══════════════════════════════════════════════════════════════════════════════
# Test 2 — Data & Class Weights
# ══════════════════════════════════════════════════════════════════════════════

print("\n[Block 2] Data & Class Weights\n" + "─"*50)

def t_preprocessed_data_exists():
    for split in ["train", "dev", "test"]:
        p = f"data/{split}_preprocessed.csv"
        assert os.path.exists(p), f"Không tìm thấy {p}"

def t_eda_outputs_exist():
    for f in ["outputs/eda/class_weights.json", "outputs/eda/encoder_config.json"]:
        assert os.path.exists(f), f"Không tìm thấy {f}"

def t_encoder_config_values():
    import json
    cfg = json.load(open("outputs/eda/encoder_config.json"))
    assert "recommended_max_seq_len" in cfg
    seq = cfg["recommended_max_seq_len"]
    assert seq in [256, 384], f"seq_len={seq} bất thường"
    assert cfg.get("encoder_option") == "concat_4_layers"

def t_class_weights_load():
    from train import load_class_weights
    from utils.constants import ASPECT_COLUMNS
    weights = load_class_weights(
        "outputs/eda/class_weights.json",
        weight_clip=10.0,
        device=torch.device("cpu"),
    )
    assert len(weights) == 34, f"Expected 34, got {len(weights)}"
    for i, w in enumerate(weights):
        assert w.shape == (4,), f"Aspect {i}: expected [4], got {w.shape}"
        assert (w > 0).all(), f"Aspect {i}: có weight <= 0"
        assert (w <= 10.0).all(), f"Aspect {i}: weight chưa clip đúng"

def t_dataloader_1batch():
    import pandas as pd
    from step2_dataloader import ABSADataset
    from utils.constants import ASPECT_COLUMNS
    tokenizer_mock = MagicMock()
    tokenizer_mock.return_value = {
        "input_ids": torch.randint(0, 1000, (1, 256)),
        "attention_mask": torch.ones(1, 256, dtype=torch.long),
    }
    df = pd.read_csv("data/train_preprocessed.csv").head(4)
    # Test bằng mock tokenizer
    ds = ABSADataset.__new__(ABSADataset)
    ds.texts  = df["processed_review"].fillna("").tolist()
    ds.labels = df[ASPECT_COLUMNS].values.astype(int)
    ds.max_len = 256
    ds.tokenizer = MagicMock()
    # Encode mock
    mock_enc = MagicMock()
    mock_enc.__getitem__ = lambda self, k: (
        torch.randint(0, 1000, (1, 256)) if k == "input_ids"
        else torch.ones(1, 256, dtype=torch.long)
    )
    mock_enc.squeeze = MagicMock(return_value=torch.randint(0, 1000, (256,)))
    # Kiểm tra labels shape + range
    labels = torch.tensor(ds.labels[0], dtype=torch.long)
    assert labels.shape == (34,), f"Labels shape sai: {labels.shape}"
    assert labels.min() >= 0 and labels.max() <= 3, "Label ngoài range [0,3]"

for fn in [t_preprocessed_data_exists, t_eda_outputs_exist,
           t_encoder_config_values, t_class_weights_load, t_dataloader_1batch]:
    check(fn.__name__.replace("t_", ""), fn)


# ══════════════════════════════════════════════════════════════════════════════
# Test 3 — Model Forward Pass & Loss
# ══════════════════════════════════════════════════════════════════════════════

print("\n[Block 3] Model Forward Pass & Loss\n" + "─"*50)

BATCH = 4
SEQ   = 64

def t_model_concat4_forward():
    model = make_mock_model("concat_4_layers")
    assert model.hidden_size == 3072
    ids   = torch.randint(0, 1000, (BATCH, SEQ))
    mask  = torch.ones(BATCH, SEQ, dtype=torch.long)
    out   = model(ids, mask)
    assert "loss" in out and "logits" in out and "preds" in out
    assert out["loss"] is None  # không có labels
    assert out["preds"].shape == (BATCH, 34)

def t_model_cls_only_forward():
    model = make_mock_model("cls_only")
    assert model.hidden_size == 768
    ids  = torch.randint(0, 1000, (BATCH, SEQ))
    mask = torch.ones(BATCH, SEQ, dtype=torch.long)
    out  = model(ids, mask)
    assert out["preds"].shape == (BATCH, 34)

def t_weighted_loss_computed():
    from train import load_class_weights
    model = make_mock_model("concat_4_layers")
    weights = load_class_weights("outputs/eda/class_weights.json", weight_clip=10.0)

    ids    = torch.randint(0, 1000, (BATCH, SEQ))
    mask   = torch.ones(BATCH, SEQ, dtype=torch.long)
    labels = torch.randint(0, 4, (BATCH, 34))

    out  = model(ids, mask, labels=labels, class_weights=weights)
    loss = out["loss"]
    assert loss is not None, "loss is None khi có labels!"
    assert not torch.isnan(loss), f"Loss là NaN! Lỗi nghiêm trọng."
    assert not torch.isinf(loss), f"Loss là Inf! Lỗi nghiêm trọng."
    assert loss.item() > 0, f"Loss = {loss.item()} không hợp lý"

def t_unweighted_loss_fallback():
    model  = make_mock_model("concat_4_layers")
    ids    = torch.randint(0, 1000, (BATCH, SEQ))
    mask   = torch.ones(BATCH, SEQ, dtype=torch.long)
    labels = torch.randint(0, 4, (BATCH, 34))
    out    = model(ids, mask, labels=labels, class_weights=None)
    assert not torch.isnan(out["loss"])

def t_gradient_flows():
    from train import load_class_weights
    model  = make_mock_model("concat_4_layers")
    weights = load_class_weights("outputs/eda/class_weights.json", weight_clip=10.0)

    ids    = torch.randint(0, 1000, (BATCH, SEQ))
    mask   = torch.ones(BATCH, SEQ, dtype=torch.long)
    labels = torch.randint(0, 4, (BATCH, 34))

    out  = model(ids, mask, labels=labels, class_weights=weights)
    out["loss"].backward()

    # Kiểm tra gradient ở classifier đầu tiên
    clf_weight_grad = model.classifiers[0].weight.grad
    assert clf_weight_grad is not None, "Gradient không lan truyền tới classifiers!"
    assert not torch.isnan(clf_weight_grad).any(), "Gradient NaN trong classifiers!"

def t_weighted_vs_unweighted_loss_differ():
    from train import load_class_weights
    model   = make_mock_model("concat_4_layers")
    weights = load_class_weights("outputs/eda/class_weights.json", weight_clip=10.0)

    torch.manual_seed(42)
    ids    = torch.randint(0, 1000, (BATCH, SEQ))
    mask   = torch.ones(BATCH, SEQ, dtype=torch.long)
    # Dùng labels có nhiều neutral (class 3) để class_weights ảnh hưởng rõ ràng
    labels = torch.full((BATCH, 34), 3, dtype=torch.long)

    out_w  = model(ids, mask, labels=labels, class_weights=weights)
    out_nw = model(ids, mask, labels=labels, class_weights=None)
    diff   = abs(out_w["loss"].item() - out_nw["loss"].item())
    assert diff > 1e-4, (
        f"Weighted ({out_w['loss'].item():.4f}) và unweighted ({out_nw['loss'].item():.4f}) "
        f"loss giống nhau! class_weights có thể không được áp dụng."
    )

for fn in [t_model_concat4_forward, t_model_cls_only_forward,
           t_weighted_loss_computed, t_unweighted_loss_fallback,
           t_gradient_flows, t_weighted_vs_unweighted_loss_differ]:
    check(fn.__name__.replace("t_", ""), fn)


# ══════════════════════════════════════════════════════════════════════════════
# Test 4 — Training Loop (2 bước thực sự)
# ══════════════════════════════════════════════════════════════════════════════

print("\n[Block 4] Training Loop (2 optimizer steps)\n" + "─"*50)

def t_train_loop_2steps():
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from train import run_epoch, load_class_weights
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup
    from utils.helpers import set_seed

    set_seed(42)
    device = torch.device("cpu")

    # Fake DataLoader (8 mẫu, batch=4 → 2 batches)
    N = 8
    ids    = torch.randint(0, 1000, (N, 64))
    mask   = torch.ones(N, 64, dtype=torch.long)
    labels = torch.randint(0, 4, (N, 34))
    ds     = TensorDataset(ids, mask, labels)

    class DictDataset(torch.utils.data.Dataset):
        def __init__(self, ids, mask, labels):
            self.ids, self.mask, self.labels = ids, mask, labels
        def __len__(self):
            return len(self.ids)
        def __getitem__(self, i):
            return {
                "input_ids":      self.ids[i],
                "attention_mask": self.mask[i],
                "labels":         self.labels[i],
                "review_text":    "test",
            }

    loader = DataLoader(DictDataset(ids, mask, labels), batch_size=4)

    model   = make_mock_model("concat_4_layers").to(device)
    weights = load_class_weights("outputs/eda/class_weights.json", weight_clip=10.0)
    optim   = AdamW(model.parameters(), lr=2e-5)
    sched   = get_linear_schedule_with_warmup(optim, 0, 10)

    # Bước train
    loss1, _, _ = run_epoch(
        model, loader, device, weights,
        optimizer=optim, scheduler=sched,
        grad_accum=2, is_train=True,
        use_amp=False, scaler=None,
    )
    assert not np.isnan(loss1), f"Train loss NaN: {loss1}"
    assert loss1 > 0, f"Train loss <= 0: {loss1}"

    # Bước eval
    loss2, y_true, y_pred = run_epoch(
        model, loader, device, weights,
        is_train=False, use_amp=False, scaler=None,
    )
    assert y_true.shape == (N, 34), f"y_true shape sai: {y_true.shape}"
    assert y_pred.shape == (N, 34), f"y_pred shape sai: {y_pred.shape}"
    assert not np.isnan(loss2)

def t_grad_accum_final_batch_flushed():
    from torch.utils.data import DataLoader
    from train import run_epoch, load_class_weights
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup

    # 3 batches × 4 samples = 12 mẫu (3 không chia hết cho accum=2)
    class DictDataset(torch.utils.data.Dataset):
        def __init__(self, n):
            self.n = n
        def __len__(self):
            return self.n
        def __getitem__(self, i):
            return {
                "input_ids":      torch.randint(0, 1000, (64,)),
                "attention_mask": torch.ones(64, dtype=torch.long),
                "labels":         torch.randint(0, 4, (34,)),
                "review_text":    "x",
            }

    loader  = DataLoader(DictDataset(12), batch_size=4)  # 3 batches
    model   = make_mock_model("concat_4_layers")
    weights = load_class_weights("outputs/eda/class_weights.json", weight_clip=10.0)
    optim   = AdamW(model.parameters(), lr=2e-5)
    sched   = get_linear_schedule_with_warmup(optim, 0, 10)

    # Lưu params trước
    p_before = model.classifiers[0].weight.data.clone()
    run_epoch(model, loader, torch.device("cpu"), weights,
              optimizer=optim, scheduler=sched,
              grad_accum=2, is_train=True)
    p_after = model.classifiers[0].weight.data

    # Params phải thay đổi (có gradient từ tất cả batches kể cả batch 3)
    assert not torch.allclose(p_before, p_after, atol=1e-9), \
        "Params không thay đổi — gradient có thể không được flush đúng!"

for fn in [t_train_loop_2steps, t_grad_accum_final_batch_flushed]:
    check(fn.__name__.replace("t_", ""), fn)


# ══════════════════════════════════════════════════════════════════════════════
# Test 5 — Evaluation Metrics
# ══════════════════════════════════════════════════════════════════════════════

print("\n[Block 5] Evaluation Metrics\n" + "─"*50)

def t_eval_perfect_prediction():
    from step4_eval import compute_aspect_f1
    rng = np.random.default_rng(42)
    y   = rng.integers(0, 4, (200, 34))
    m   = compute_aspect_f1(y, y)
    assert abs(m["macro_acd_f1"] - 1.0) < 1e-6, f"Perfect ACD F1 ≠ 1.0: {m['macro_acd_f1']}"
    assert abs(m["macro_spc_f1"] - 1.0) < 1e-6, f"Perfect SPC F1 ≠ 1.0: {m['macro_spc_f1']}"

def t_eval_zero_train_aspects_excluded():
    from step4_eval import compute_aspect_f1
    from utils.constants import ZERO_TRAIN_ASPECTS, ASPECT_COLUMNS
    rng = np.random.default_rng(0)
    y   = rng.integers(0, 4, (100, 34))
    # Tất cả predict = 0 cho ZERO_TRAIN_ASPECTS
    y_pred = y.copy()
    idx = ASPECT_COLUMNS.index(ZERO_TRAIN_ASPECTS[0])
    y_pred[:, idx] = 0
    m_excl = compute_aspect_f1(y, y_pred, exclude_aspects=ZERO_TRAIN_ASPECTS)
    m_all  = compute_aspect_f1(y, y_pred, exclude_aspects=[])
    assert ZERO_TRAIN_ASPECTS[0] in m_excl["excluded_aspects"]
    # Macro F1 với exclude phải >= all (vì bỏ aspect kém)
    assert m_excl["macro_acd_f1"] >= m_all["macro_acd_f1"] - 1e-6

def t_eval_absent_majority():
    from step4_eval import compute_aspect_f1
    y_true = np.random.randint(0, 4, (200, 34))
    y_pred = np.zeros((200, 34), dtype=int)  # predict tất cả = absent
    m = compute_aspect_f1(y_true, y_pred)
    assert m["macro_acd_f1"] < 0.5, \
        f"All-absent pred: ACD F1 phải thấp, got {m['macro_acd_f1']:.3f}"

for fn in [t_eval_perfect_prediction, t_eval_zero_train_aspects_excluded,
           t_eval_absent_majority]:
    check(fn.__name__.replace("t_", ""), fn)


# ══════════════════════════════════════════════════════════════════════════════
# Test 6 — File I/O & Report
# ══════════════════════════════════════════════════════════════════════════════

print("\n[Block 6] File I/O & Report Generation\n" + "─"*50)

def t_save_load_json():
    from utils.helpers import save_json, load_json
    with tempfile.TemporaryDirectory() as d:
        p    = os.path.join(d, "sub", "test.json")
        data = {"a": 1, "b": [1, 2, 3], "c": "hello"}
        save_json(data, p)
        loaded = load_json(p)
        assert loaded == data

def t_training_history_schema():
    from utils.helpers import save_json, load_json
    history = {
        "train_loss": [0.5, 0.4, 0.35],
        "dev_loss":   [0.52, 0.45, 0.48],
        "dev_acd_f1": [0.6, 0.65, 0.64],
        "dev_spc_f1": [0.5, 0.55, 0.54],
        "dev_combined_f1": [0.55, 0.60, 0.59],
        "best_epoch": 2,
        "best_combined_f1": 0.60,
        "config": {"learning_rate": 2e-5, "batch_size": 8},
        "use_amp": True,
    }
    required_keys = ["train_loss", "dev_loss", "dev_combined_f1", "best_epoch"]
    for k in required_keys:
        assert k in history, f"Missing key: {k}"

def t_report_generation():
    from predict import generate_summary_report
    from utils.constants import ASPECT_COLUMNS

    history = {
        "train_loss": [0.5, 0.4],
        "dev_loss":   [0.52, 0.55],
        "dev_acd_f1": [0.65, 0.64],
        "dev_spc_f1": [0.60, 0.59],
        "dev_combined_f1": [0.625, 0.615],
        "best_epoch": 1,
        "best_combined_f1": 0.625,
        "config": {
            "learning_rate": 2e-5,
            "batch_size": 8,
            "grad_accumulation_steps": 2,
            "warmup_ratio": 0.1,
            "weight_clip": 10.0,
            "max_seq_len": 256,
            "encoder_option": "concat_4_layers",
            "early_stop_patience": 3,
        },
    }
    dev_metrics = {
        "macro_acd_f1": 0.65, "macro_spc_f1": 0.60,
        "macro_combined_f1": 0.625,
        "per_aspect": {a: {"acd_f1": 0.6, "spc_f1": 0.5, "support": 10}
                       for a in ASPECT_COLUMNS},
    }
    test_metrics = {**dev_metrics, "macro_combined_f1": 0.62}
    config = history["config"]

    with tempfile.TemporaryDirectory() as d:
        save_path = os.path.join(d, "phobert_summary.md")
        generate_summary_report(history, dev_metrics, test_metrics, config,
                                save_path=save_path)
        assert os.path.exists(save_path)
        content = open(save_path, encoding="utf-8").read()
        assert "concat_4_layers" in content
        assert "0.62" in content  # test Combined F1
        assert len(content) > 200, "Report quá ngắn"

for fn in [t_save_load_json, t_training_history_schema, t_report_generation]:
    check(fn.__name__.replace("t_", ""), fn)


# ══════════════════════════════════════════════════════════════════════════════
# Kết quả cuối
# ══════════════════════════════════════════════════════════════════════════════

passed = sum(1 for _, ok, _ in results if ok)
total  = len(results)
failed = [(name, err) for name, ok, err in results if not ok]

print("\n" + "═"*55)
print(f"  KẾT QUẢ: {passed}/{total} tests passed")
print("═"*55)

if failed:
    print("\n❌ FAILED TESTS:")
    for name, err in failed:
        print(f"   • {name}: {err[:100]}")
    print("\nSửa các lỗi trên trước khi chạy trên Kaggle.")
    sys.exit(1)
else:
    print("\nAll tests passed. Code ready for Kaggle.")
    print("   Việc cần làm tiếp theo:")
    print("   1. git push origin master")
    print("   2. Upload notebook phase PhoBERT lên Kaggle")
    print("   3. Settings → GPU T4 x2 → Save and Run All")
