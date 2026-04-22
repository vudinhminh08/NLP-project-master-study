
import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data_processing"))
sys.path.insert(0, os.path.dirname(__file__))

from utils.constants import TRAIN_CONFIG, PHOBERT_MODEL_NAME
from utils.helpers import set_seed, get_device, load_json
from step2_dataloader import create_dataloaders
from transformers import AutoTokenizer

from model import ABSAPhoBERT
from train import train, load_class_weights
from predict import load_best_model, predict_and_evaluate, generate_summary_report
from ensemble import ensemble_predict_and_evaluate


def build_run_tag(encoder_option: str, lr: float, data_suffix: str) -> str:
    """Tạo tag định danh cho run, dùng làm suffix cho output dirs.

    Convention (để không ghi đè kết quả cũ):
      - encoder=concat_4_layers, lr=1e-4, data=""  → ""          (kết quả baseline cũ)
      - encoder=cls_only,        lr=1e-4, data=""  → "_cls_only"
      - encoder=concat_4_layers, lr=2e-5, data=""  → "_lr2e5"
      - encoder=cls_only,        lr=2e-5, data=""  → "_cls_only_lr2e5"
      - bất kỳ + data="_sota"                      → thêm "_sota" vào cuối
    """
    parts = []
    if encoder_option != "concat_4_layers":
        parts.append(encoder_option)          # e.g. "cls_only"
    default_lr = TRAIN_CONFIG["learning_rate"]
    if abs(lr - default_lr) > 1e-10:
        # Biểu diễn ngắn gọn: 2e-05 → "lr2e5", 5e-05 → "lr5e5"
        lr_str = f"{lr:.0e}".replace("-0", "").replace("-", "").replace("e", "e").replace(".", "")
        parts.append(f"lr{lr_str}")
    if data_suffix:
        parts.append(data_suffix.lstrip("_"))
    return ("_" + "_".join(parts)) if parts else ""


def main(
    encoder_option: str = None,
    use_amp: bool = True,
    lr: float = None,
    data_suffix: str = "",
) -> dict:

    set_seed(TRAIN_CONFIG["seed"])
    device = get_device()
    use_amp = use_amp and (device.type == "cuda")

    # ── Encoder option ─────────────────────────────────────────────
    enc_cfg = load_json("outputs/eda/encoder_config.json")
    encoder_option = encoder_option or enc_cfg.get("encoder_option", "concat_4_layers")

    # ── Config: override lr nếu được truyền vào ───────────────────
    effective_lr = lr if lr is not None else TRAIN_CONFIG["learning_rate"]
    config = {
        **TRAIN_CONFIG,
        "encoder_option":  encoder_option,
        "learning_rate":   effective_lr,
        "ensemble_top_k":  3,
    }
    max_seq_len = config["max_seq_len"]

    # ── Output dirs (không ghi đè kết quả cũ) ────────────────────
    run_tag     = build_run_tag(encoder_option, effective_lr, data_suffix)
    save_dir    = f"outputs/models{run_tag}"
    results_dir = f"outputs/results{run_tag}"

    print(f"\n{'='*60}")
    print(f"PHASE PHOBERT — Multi-task ABSA")
    print(f"  encoder:     {encoder_option}")
    print(f"  lr:          {effective_lr:.2e}")
    print(f"  data_suffix: '{data_suffix}' ('' = baseline preprocessed)")
    print(f"  seq_len:     {max_seq_len}")
    print(f"  batch:       {config['batch_size']} × {config['grad_accumulation_steps']}"
          f" = {config['batch_size'] * config['grad_accumulation_steps']} effective")
    print(f"  weight_clip: {config['weight_clip']}")
    print(f"  amp:         {'ON' if use_amp else 'OFF'}")
    print(f"  save_dir:    {save_dir}")
    print(f"  results_dir: {results_dir}")
    print(f"{'='*60}")

    # ── Tokenizer ─────────────────────────────────────────────────
    print(f"\n[Tokenizer] Loading {PHOBERT_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(PHOBERT_MODEL_NAME)
    print("[Tokenizer] Loaded ✓")

    # ── DataLoaders ───────────────────────────────────────────────
    print("\n[Data] Creating DataLoaders...")
    train_loader, dev_loader, test_loader = create_dataloaders(
        train_path=f"data/train_preprocessed{data_suffix}.csv",
        dev_path  =f"data/dev_preprocessed{data_suffix}.csv",
        test_path =f"data/test_preprocessed{data_suffix}.csv",
        tokenizer=tokenizer,
        batch_size=config["batch_size"],
        max_len=max_seq_len,
        num_workers=2,
        use_preprocessed=True,
    )
    if train_loader is None or dev_loader is None:
        raise FileNotFoundError(
            f"Không tìm thấy data/train_preprocessed{data_suffix}.csv. "
            f"Chạy step3_preprocessing.py (--sota nếu dùng sota data) trước!"
        )

    # ── Class weights ─────────────────────────────────────────────
    class_weights = load_class_weights(
        "outputs/eda/class_weights.json",
        weight_clip=config["weight_clip"],
        device=torch.device("cpu"),
    )

    # ── Model ─────────────────────────────────────────────────────
    print(f"\n[Model] Building ABSAPhoBERT ({encoder_option})...")
    model = ABSAPhoBERT(
        model_name=PHOBERT_MODEL_NAME,
        dropout=config["dropout"],
        encoder_option=encoder_option,
    ).to(device)

    # ── Train ─────────────────────────────────────────────────────
    history = train(
        model, train_loader, dev_loader, class_weights, device,
        config,
        save_dir=save_dir,
        results_dir=results_dir,
        use_amp=use_amp,
    )

    # ── Evaluate best single model ────────────────────────────────
    model = load_best_model(f"{save_dir}/best_model.pt", model, device)

    dev_metrics, _, _ = predict_and_evaluate(
        model, dev_loader, class_weights, device,
        split_name="dev",
        save_path=f"{results_dir}/phobert_dev_metrics.json",
    )
    test_metrics, _, _ = predict_and_evaluate(
        model, test_loader, class_weights, device,
        split_name="test",
        save_path=f"{results_dir}/phobert_test_metrics.json",
    )

    # ── Ensemble top-k checkpoints ────────────────────────────────
    top_k_path = os.path.join(save_dir, "top_k_checkpoints.json")
    ensemble_test_metrics = None
    if os.path.exists(top_k_path):
        top_k_info = load_json(top_k_path)
        ckpt_paths = [item["path"] for item in top_k_info if os.path.exists(item["path"])]
        if len(ckpt_paths) > 1:
            print(f"\n[Ensemble] Running top-{len(ckpt_paths)} checkpoint ensemble...")
            ensemble_test_metrics, _ = ensemble_predict_and_evaluate(
                ckpt_paths=ckpt_paths,
                model_template=model,
                dataloader=test_loader,
                class_weights=class_weights,
                device=device,
                save_path=f"{results_dir}/phobert_ensemble_test_metrics.json",
            )
        else:
            print("\n[Ensemble] Chỉ có 1 checkpoint khả dụng, bỏ qua ensemble.")

    # ── Summary report ────────────────────────────────────────────
    generate_summary_report(
        history, dev_metrics, test_metrics, config,
        save_path=f"{results_dir}/phobert_summary.md",
        ensemble_metrics=ensemble_test_metrics,
    )

    # ── Print kết quả ─────────────────────────────────────────────
    gap = 0.7732 - test_metrics["macro_combined_f1"]
    print(f"\n{'='*60}")
    print(f"PHASE PHOBERT HOÀN TẤT — {run_tag or 'concat_4_layers_baseline'}")
    print(f"  Dev  Combined F1       : {dev_metrics['macro_combined_f1']:.4f}")
    print(f"  Test ACD F1            : {test_metrics['macro_acd_f1']:.4f}")
    print(f"  Test SPC F1            : {test_metrics['macro_spc_f1']:.4f}")
    print(f"  Test Combined F1       : {test_metrics['macro_combined_f1']:.4f}")
    if ensemble_test_metrics:
        gap_ens = 0.7732 - ensemble_test_metrics["macro_combined_f1"]
        print(f"  Ensemble Combined F1   : {ensemble_test_metrics['macro_combined_f1']:.4f}  (gap={gap_ens:.4f})")
    print(f"  SOTA Combined F1       : 0.7732")
    print(f"  Gap (single)           : {gap:.4f} ({gap * 100:.1f}%)")
    print(f"  Xem report             : {results_dir}/phobert_summary.md")
    print(f"{'='*60}")

    return test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train ABSAPhoBERT cho VLSP 2018 Hotel"
    )
    parser.add_argument(
        "--encoder",
        default=None,
        choices=["concat_4_layers", "cls_only"],
        help="Encoder option. None = đọc từ encoder_config.json",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate override. None = dùng TRAIN_CONFIG (1e-4). SOTA dùng 2e-5.",
    )
    parser.add_argument(
        "--data-suffix",
        default="",
        help="Suffix của preprocessed data files. '' = baseline, '_sota' = sota preprocessing.",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Tắt Mixed Precision training",
    )
    args = parser.parse_args()
    main(
        encoder_option=args.encoder,
        use_amp=not args.no_amp,
        lr=args.lr,
        data_suffix=args.data_suffix,
    )
