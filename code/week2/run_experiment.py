"""
run_experiment.py — Entry point duy nhất cho tuần 2.

Chạy từ root project:
    python code/week2/run_experiment.py                          # concat_4_layers (SOTA)
    python code/week2/run_experiment.py --encoder cls_only       # ablation
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "week1"))
sys.path.insert(0, os.path.dirname(__file__))

from utils.constants import TRAIN_CONFIG, PHOBERT_MODEL_NAME
from utils.helpers import set_seed, get_device, load_json
from step2_dataloader import create_dataloaders
from transformers import AutoTokenizer

from model import ABSAPhoBERT
from train import train, load_class_weights
from predict import load_best_model, predict_and_evaluate, generate_summary_report


def main(encoder_option: str = None) -> dict:
    """
    Full pipeline: load data → build model → train → evaluate → report.

    Args:
        encoder_option: override encoder config.
                        None = đọc từ encoder_config.json (concat_4_layers).
                        "cls_only" = ablation study (768 dim).

    Returns:
        test_metrics dict
    """
    set_seed(TRAIN_CONFIG["seed"])
    device = get_device()

    # Load encoder config từ EDA — không hardcode MAX_SEQ_LEN
    enc_cfg = load_json("outputs/eda/encoder_config.json")
    max_seq_len    = enc_cfg.get("recommended_max_seq_len", 384)
    encoder_option = encoder_option or enc_cfg.get("encoder_option", "concat_4_layers")

    config = {
        **TRAIN_CONFIG,
        "max_seq_len":    max_seq_len,
        "encoder_option": encoder_option,
    }
    print(f"\n[Config] encoder={encoder_option}, seq_len={max_seq_len}")
    print(f"[Config] batch={config['batch_size']} × accum={config['grad_accumulation_steps']}"
          f" = {config['batch_size'] * config['grad_accumulation_steps']} effective")
    print(f"[Config] lr={config['learning_rate']}, weight_clip={config['weight_clip']}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(PHOBERT_MODEL_NAME)

    # DataLoaders — dùng preprocessed cache từ tuần 1
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

    # Class weights — load từ EDA, clip neutral=154 → 10.0
    class_weights = load_class_weights(
        "outputs/eda/class_weights.json",
        weight_clip=config["weight_clip"],
        device=torch.device("cpu"),  # ← load CPU, run_epoch sẽ move lên GPU khi cần
    )

    # Model
    model = ABSAPhoBERT(
        model_name=PHOBERT_MODEL_NAME,
        dropout=config["dropout"],
        encoder_option=encoder_option,
    ).to(device)

    # Suffix cho ablation (lưu riêng để so sánh)
    suffix      = "" if encoder_option == "concat_4_layers" else f"_{encoder_option}"
    save_dir    = f"outputs/models{suffix}"
    results_dir = f"outputs/results{suffix}"

    # Train
    history = train(
        model, train_loader, dev_loader, class_weights, device,
        config, save_dir=save_dir, results_dir=results_dir,
    )

    # Evaluate với best checkpoint
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

    generate_summary_report(
        history, dev_metrics, test_metrics, config,
        save_path=f"{results_dir}/week2_summary.md",
    )

    # Final summary
    gap = 0.7732 - test_metrics["macro_combined_f1"]
    print(f"\n{'='*60}")
    print(f"TUẦN 2 HOÀN TẤT — encoder={encoder_option}")
    print(f"  Dev  Combined F1 : {dev_metrics['macro_combined_f1']:.4f}")
    print(f"  Test Combined F1 : {test_metrics['macro_combined_f1']:.4f}")
    print(f"  SOTA Combined F1 : 0.7732")
    print(f"  Gap              : {gap:.4f} ({gap * 100:.1f}%)")
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
    args = parser.parse_args()
    main(encoder_option=args.encoder)
