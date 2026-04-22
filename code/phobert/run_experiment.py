
import os
import sys
import argparse
import torch
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data_processing"))
sys.path.insert(0, os.path.dirname(__file__))

from utils.constants import TRAIN_CONFIG, PHOBERT_MODEL_NAME
from utils.helpers import set_seed, get_device, load_json
from step2_dataloader import create_dataloaders
from transformers import AutoTokenizer

from model import ABSAPhoBERT
from train import train, load_class_weights
from predict import load_best_model, predict_and_evaluate, generate_summary_report
from ensemble import ensemble_from_top_k_file


def _format_lr_tag(lr: float) -> str:
    if abs(lr - 2e-5) < 1e-12:
        return "lr2e5"
    tag = f"lr{lr:.0e}"
    return tag.replace("e-0", "e").replace("e+0", "e").replace("e+", "e")


def build_run_tag(encoder_option: str, lr: float) -> str:
    parts = []
    if encoder_option != "concat_4_layers":
        parts.append(encoder_option)
    if abs(lr - 1e-4) >= 1e-12:
        parts.append(_format_lr_tag(lr))
    return "" if not parts else "_" + "_".join(parts)


def ensure_output_available(save_dir: str, results_dir: str, overwrite: bool = False) -> None:
    if overwrite:
        return

    existing = []
    for path in [
        os.path.join(save_dir, "best_model.pt"),
        os.path.join(save_dir, "top_k_checkpoints.json"),
        os.path.join(results_dir, "training_history.json"),
        os.path.join(results_dir, "phobert_test_metrics.json"),
    ]:
        if os.path.exists(path):
            existing.append(path)

    if existing:
        raise FileExistsError(
            "Output đã tồn tại, dừng để tránh ghi đè. "
            "Dùng run tag khác hoặc truyền overwrite=True nếu muốn chạy lại:\n"
            + "\n".join(f"  - {p}" for p in existing)
        )


def main(
    encoder_option: str = None,
    use_amp: bool = True,
    lr: Optional[float] = None,
    max_epochs: Optional[int] = None,
    early_stop_patience: Optional[int] = None,
    overwrite: bool = False,
) -> dict:

    set_seed(TRAIN_CONFIG["seed"])
    device = get_device()


    use_amp = use_amp and (device.type == "cuda")



    enc_cfg = load_json("outputs/eda/encoder_config.json")
    encoder_option = encoder_option or enc_cfg.get("encoder_option", "concat_4_layers")

    config = {
        **TRAIN_CONFIG,
        "encoder_option": encoder_option,
    }
    if lr is not None:
        config["learning_rate"] = lr
    if max_epochs is not None:
        config["max_epochs"] = max_epochs
    if early_stop_patience is not None:
        config["early_stop_patience"] = early_stop_patience

    max_seq_len = config["max_seq_len"]
    run_tag     = build_run_tag(encoder_option, config["learning_rate"])
    save_dir    = f"outputs/models{run_tag}"
    results_dir = f"outputs/results{run_tag}"
    ensure_output_available(save_dir, results_dir, overwrite=overwrite)

    print(f"\n{'='*60}")
    print(f"PHASE PHOBERT — Multi-task ABSA")
    print(f"  encoder:    {encoder_option}")
    print(f"  seq_len:    {max_seq_len}")
    print(f"  batch:      {config['batch_size']} × {config['grad_accumulation_steps']}"
          f" = {config['batch_size'] * config['grad_accumulation_steps']} effective")
    print(f"  lr:         {config['learning_rate']}")
    print(f"  weight_clip:{config['weight_clip']}")
    print(f"  amp:        {'ON' if use_amp else 'OFF'}")
    print(f"{'='*60}")


    print(f"\n[Tokenizer] Loading {PHOBERT_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(PHOBERT_MODEL_NAME)
    print("[Tokenizer] Loaded ")


    print("\n[Data] Creating DataLoaders...")
    train_loader, dev_loader, test_loader = create_dataloaders(
        train_path="data/train_preprocessed.csv",
        dev_path  ="data/dev_preprocessed.csv",
        test_path ="data/test_preprocessed.csv",
        tokenizer=tokenizer,
        batch_size=config["batch_size"],
        max_len=max_seq_len,
        num_workers=2,
        use_preprocessed=True,
    )

    if train_loader is None or dev_loader is None:
        raise FileNotFoundError(
            "Không tìm thấy preprocessed data. "
            "Chạy code/data_processing/step3_preprocessing.py trước!"
        )



    class_weights = load_class_weights(
        "outputs/eda/class_weights.json",
        weight_clip=config["weight_clip"],
        device=torch.device("cpu"),
    )


    print(f"\n[Model] Building ABSAPhoBERT ({encoder_option})...")
    model = ABSAPhoBERT(
        model_name=PHOBERT_MODEL_NAME,
        dropout=config["dropout"],
        encoder_option=encoder_option,
    ).to(device)


    history = train(
        model, train_loader, dev_loader, class_weights, device,
        config,
        save_dir=save_dir,
        results_dir=results_dir,
        use_amp=use_amp,
    )


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

    ensemble_metrics = None
    top_k_path = os.path.join(save_dir, "top_k_checkpoints.json")
    if os.path.exists(top_k_path):
        ensemble_metrics, _, _ = ensemble_from_top_k_file(
            model,
            test_loader,
            top_k_path,
            device,
            split_name="test",
            save_path=f"{results_dir}/phobert_test_ensemble_metrics.json",
        )

    generate_summary_report(
        history, dev_metrics, test_metrics, config,
        ensemble_metrics=ensemble_metrics,
        save_path=f"{results_dir}/phobert_summary.md",
    )

    primary_metrics = test_metrics
    primary_label = "Single"
    if (
        ensemble_metrics is not None
        and ensemble_metrics["macro_combined_f1"] > test_metrics["macro_combined_f1"]
    ):
        primary_metrics = ensemble_metrics
        primary_label = "Ensemble Top-3"

    gap = 0.7732 - primary_metrics["macro_combined_f1"]
    print(f"\n{'='*60}")
    print(f"PHASE PHOBERT HOÀN TẤT — encoder={encoder_option}")
    print(f"  Dev  Combined F1 : {dev_metrics['macro_combined_f1']:.4f}")
    print(f"  Test ACD F1      : {test_metrics['macro_acd_f1']:.4f}")
    print(f"  Test SPC F1      : {test_metrics['macro_spc_f1']:.4f}")
    print(f"  Test Combined F1 : {test_metrics['macro_combined_f1']:.4f}")
    if ensemble_metrics is not None:
        print(f"  Ensemble Combined: {ensemble_metrics['macro_combined_f1']:.4f}")
    print(f"  Primary result    : {primary_label}")
    print(f"  SOTA Combined F1 : 0.7732")
    print(f"  Gap              : {gap:.4f} ({gap * 100:.1f}%)")
    print(f"  Xem report       : {results_dir}/phobert_summary.md")
    print(f"{'='*60}")

    test_metrics["ensemble_metrics"] = ensemble_metrics
    test_metrics["primary_result"] = primary_label
    test_metrics["primary_combined_f1"] = primary_metrics["macro_combined_f1"]
    return test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train ABSAPhoBERT cho VLSP 2018 Hotel"
    )
    parser.add_argument(
        "--encoder",
        default=None,
        choices=["concat_4_layers", "cls_only"],
        help="Encoder option. None = đọc từ encoder_config.json (concat_4_layers)",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Tắt Mixed Precision training (dùng nếu gặp lỗi AMP)",
    )
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--max-epochs", type=int, default=None, help="Override max epochs")
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=None,
        help="Override early stopping patience",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Cho phép ghi đè output dir hiện có",
    )
    args = parser.parse_args()
    main(
        encoder_option=args.encoder,
        use_amp=not args.no_amp,
        lr=args.lr,
        max_epochs=args.max_epochs,
        early_stop_patience=args.early_stop_patience,
        overwrite=args.overwrite,
    )
