"""
run_experiment.py — Entry point duy nhất cho tuần 2.

Chạy từ root project:
    python code/week2/run_experiment.py                           # concat_4_layers (SOTA)
    python code/week2/run_experiment.py --encoder cls_only        # ablation

Output:
    outputs/models/best_model.pt                 (hoặc outputs/models_cls_only/)
    outputs/results/training_history.json
    outputs/results/week2_dev_metrics.json
    outputs/results/week2_test_metrics.json
    outputs/results/week2_summary.md
"""

import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "week1"))
sys.path.insert(0, os.path.dirname(__file__))

from utils.constants import TRAIN_CONFIG, PHOBERT_MODEL_NAME
from utils.helpers import set_seed, get_device, load_json
from step2_dataloader import create_dataloaders
from transformers import AutoTokenizer

from model import ABSAPhoBERT
from train import train, load_class_weights
from predict import load_best_model, predict_and_evaluate, generate_summary_report


def main(encoder_option: str = None, use_amp: bool = True) -> dict:
    """
    Full pipeline: load data → build model → train → evaluate → report.

    Args:
        encoder_option: "concat_4_layers" (SOTA) hoặc "cls_only" (ablation).
                        None = đọc từ encoder_config.json (mặc định concat_4_layers).
        use_amp:        True để dùng Mixed Precision training (tự động tắt nếu CPU).

    Returns:
        test_metrics dict với các keys: macro_acd_f1, macro_spc_f1, macro_combined_f1, ...
    """
    # === Setup ===
    set_seed(TRAIN_CONFIG["seed"])
    device = get_device()

    # Tắt AMP tự động nếu không có CUDA
    use_amp = use_amp and (device.type == "cuda")

    # Load encoder config từ EDA — chỉ lấy encoder_option, KHÔNG override max_seq_len
    # max_seq_len là hyperparameter training → nguồn chính xác là TRAIN_CONFIG (constants.py)
    enc_cfg = load_json("outputs/eda/encoder_config.json")
    encoder_option = encoder_option or enc_cfg.get("encoder_option", "concat_4_layers")

    config = {
        **TRAIN_CONFIG,
        "encoder_option": encoder_option,
    }
    max_seq_len = config["max_seq_len"]  # đọc từ TRAIN_CONFIG (384)

    print(f"\n{'='*60}")
    print(f"TUẦN 2 — PhoBERT Multi-task ABSA")
    print(f"  encoder:    {encoder_option}")
    print(f"  seq_len:    {max_seq_len}")
    print(f"  batch:      {config['batch_size']} × {config['grad_accumulation_steps']}"
          f" = {config['batch_size'] * config['grad_accumulation_steps']} effective")
    print(f"  lr:         {config['learning_rate']}")
    print(f"  weight_clip:{config['weight_clip']}")
    print(f"  amp:        {'ON' if use_amp else 'OFF'}")
    print(f"{'='*60}")

    # === Tokenizer ===
    print(f"\n[Tokenizer] Loading {PHOBERT_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(PHOBERT_MODEL_NAME)
    print("[Tokenizer] Loaded ✓")

    # === DataLoaders — dùng preprocessed cache từ tuần 1 ===
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
            "Chạy code/week1/step3_preprocessing.py trước!"
        )

    # === Class Weights — load từ EDA, clip neutral=154 → 10.0 ===
    # Quan trọng: class weights phải load trên CPU trước, sau đó chuyển lên device trong run_epoch
    class_weights = load_class_weights(
        "outputs/eda/class_weights.json",
        weight_clip=config["weight_clip"],
        device=torch.device("cpu"),
    )

    # === Model ===
    print(f"\n[Model] Building ABSAPhoBERT ({encoder_option})...")
    model = ABSAPhoBERT(
        model_name=PHOBERT_MODEL_NAME,
        dropout=config["dropout"],
        encoder_option=encoder_option,
    ).to(device)

    # Thư mục riêng cho ablation (để so sánh sau)
    suffix      = "" if encoder_option == "concat_4_layers" else f"_{encoder_option}"
    save_dir    = f"outputs/models{suffix}"
    results_dir = f"outputs/results{suffix}"

    # === Train ===
    history = train(
        model, train_loader, dev_loader, class_weights, device,
        config,
        save_dir=save_dir,
        results_dir=results_dir,
        use_amp=use_amp,
    )

    # === Evaluate best checkpoint ===
    model = load_best_model(f"{save_dir}/best_model.pt", model, device)

    dev_metrics, _, _ = predict_and_evaluate(
        model, dev_loader, class_weights, device,
        split_name="dev",
        save_path=f"{results_dir}/week2_dev_metrics.json",
    )
    test_metrics, _, _ = predict_and_evaluate(
        model, test_loader, class_weights, device,
        split_name="test",
        save_path=f"{results_dir}/week2_test_metrics.json",
    )

    # === Summary Report ===
    generate_summary_report(
        history, dev_metrics, test_metrics, config,
        save_path=f"{results_dir}/week2_summary.md",
    )

    # === Final Summary ===
    gap = 0.7732 - test_metrics["macro_combined_f1"]
    print(f"\n{'='*60}")
    print(f"TUẦN 2 HOÀN TẤT — encoder={encoder_option}")
    print(f"  Dev  Combined F1 : {dev_metrics['macro_combined_f1']:.4f}")
    print(f"  Test ACD F1      : {test_metrics['macro_acd_f1']:.4f}")
    print(f"  Test SPC F1      : {test_metrics['macro_spc_f1']:.4f}")
    print(f"  Test Combined F1 : {test_metrics['macro_combined_f1']:.4f}")
    print(f"  SOTA Combined F1 : 0.7732")
    print(f"  Gap              : {gap:.4f} ({gap * 100:.1f}%)")
    print(f"  Xem report       : {results_dir}/week2_summary.md")
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
        help="Encoder option. None = đọc từ encoder_config.json (concat_4_layers)",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Tắt Mixed Precision training (dùng nếu gặp lỗi AMP)",
    )
    args = parser.parse_args()
    main(encoder_option=args.encoder, use_amp=not args.no_amp)
