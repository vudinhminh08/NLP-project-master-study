import argparse
import csv
import html
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data_processing"))
sys.path.insert(0, os.path.dirname(__file__))

from model import ABSAPhoBERT
from utils.constants import ASPECT_COLUMNS, IDX_TO_LABEL, PHOBERT_MODEL_NAME, RARE_ASPECTS


DEFAULT_SAMPLE_INDICES = [0, 1, 2, 7, 21]
SPECIAL_TOKENS = {"<s>", "</s>", "<pad>"}


def slugify(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def build_model(checkpoint_path: Path, encoder_option: str, device: torch.device) -> ABSAPhoBERT:
    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt.get("config", {})
    rare_aspect_ids = [i for i, asp in enumerate(ASPECT_COLUMNS) if asp in RARE_ASPECTS]

    model = ABSAPhoBERT(
        model_name=PHOBERT_MODEL_NAME,
        dropout=float(config.get("dropout", 0.2)),
        encoder_option=encoder_option,
        focal_gamma=float(config.get("focal_gamma", 2.0)),
        attn_dim=int(config.get("attn_dim", 128)),
        use_split_loss=bool(config.get("use_split_loss", True)),
        lambda_presence=float(config.get("lambda_presence", 1.0)),
        lambda_sentiment=float(config.get("lambda_sentiment", 1.0)),
        presence_threshold=float(config.get("presence_threshold", 0.5)),
        rare_aspect_ids=rare_aspect_ids,
        rare_presence_pos_mult=float(config.get("rare_presence_pos_mult", 1.0)),
        rare_sentiment_mult=float(config.get("rare_sentiment_mult", 1.0)),
        use_gradient_checkpointing=False,
        mc_dropout_passes=int(config.get("mc_dropout_passes", 1)),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model


def compute_attention(model: ABSAPhoBERT, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> np.ndarray:
    outputs = model.phobert(input_ids=input_ids, attention_mask=attention_mask)
    hidden_states = outputs.hidden_states
    if model.encoder_option == "concat_4_layers":
        hidden = torch.cat([hidden_states[i] for i in [-4, -3, -2, -1]], dim=-1)
    else:
        hidden = hidden_states[-1]

    projected = torch.tanh(model.token_proj(hidden))
    scores = torch.matmul(projected, model.aspect_queries.t())
    mask = (attention_mask == 0).unsqueeze(-1)
    scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
    alphas = torch.softmax(scores, dim=1)
    return alphas[0].detach().cpu().numpy()


def predict_one(model: ABSAPhoBERT, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple:
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
    presence_logits = torch.stack(out["presence_logits"], dim=1)[0]
    presence_probs = torch.sigmoid(presence_logits).detach().cpu().numpy()
    thresholds = model.aspect_presence_thresholds.detach().cpu().numpy()
    sentiment = torch.stack(
        [logit[:, 1:].argmax(dim=-1) + 1 for logit in out["logits"]],
        dim=1,
    )[0].detach().cpu().numpy()
    preds = out["preds"][0].detach().cpu().numpy()
    return presence_probs, thresholds, sentiment, preds


def visible_token_indices(input_ids: torch.Tensor, attention_mask: torch.Tensor, tokenizer) -> tuple:
    ids = input_ids[0].detach().cpu().tolist()
    mask = attention_mask[0].detach().cpu().tolist()
    tokens = tokenizer.convert_ids_to_tokens(ids)
    keep = [i for i, (tok, m) in enumerate(zip(tokens, mask)) if m == 1 and tok not in SPECIAL_TOKENS]
    labels = [tokens[i].replace("@@ ", "@@").replace("▁", "") for i in keep]
    return keep, labels


def choose_aspects(labels: np.ndarray, preds: np.ndarray, presence_probs: np.ndarray, top_k: int) -> list:
    true_present = set(np.where(labels > 0)[0].tolist())
    pred_present = set(np.where(preds > 0)[0].tolist())
    selected = list(true_present | pred_present)
    ranked = np.argsort(-presence_probs).tolist()
    for idx in ranked:
        if idx not in selected:
            selected.append(idx)
        if len(selected) >= top_k:
            break
    return sorted(selected, key=lambda i: (-presence_probs[i], ASPECT_COLUMNS[i]))[:top_k]


def attention_matrix_for_aspects(alphas: np.ndarray, token_indices: list, aspect_indices: list) -> np.ndarray:
    matrix = alphas[token_indices, :][:, aspect_indices].T
    row_sums = matrix.sum(axis=1, keepdims=True)
    return matrix / np.clip(row_sums, 1e-12, None)


def top_tokens_for_aspect(matrix_row: np.ndarray, tokens: list, k: int = 5) -> str:
    top_ids = np.argsort(-matrix_row)[:k]
    return ", ".join(f"{tokens[i]}:{matrix_row[i]:.3f}" for i in top_ids)


def save_matrix_png(
    matrix: np.ndarray,
    tokens: list,
    aspect_names: list,
    title: str,
    save_path: Path,
) -> None:
    width = max(12, min(26, 0.45 * len(tokens)))
    height = max(4.5, 0.45 * len(aspect_names) + 1.8)
    fig, ax = plt.subplots(figsize=(width, height))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(np.arange(len(tokens)))
    ax.set_xticklabels(tokens, rotation=50, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(aspect_names)))
    ax.set_yticklabels(aspect_names, fontsize=8)
    ax.set_title(title, fontsize=12)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Normalized attention weight", rotation=90)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def token_span(token: str, weight: float) -> str:
    alpha = min(0.92, 0.12 + 0.88 * float(weight))
    style = (
        "display:inline-block;margin:2px;padding:3px 5px;border-radius:4px;"
        f"background:rgba(220,38,38,{alpha:.3f});"
    )
    return f'<span style="{style}" title="attention={weight:.4f}">{html.escape(token)}</span>'


def save_token_html(
    sample_id: int,
    raw_review: str,
    processed_review: str,
    matrix: np.ndarray,
    tokens: list,
    aspect_indices: list,
    predictions: list,
    save_path: Path,
) -> None:
    sections = []
    for row_idx, aspect_idx in enumerate(aspect_indices):
        aspect = ASPECT_COLUMNS[aspect_idx]
        pred = predictions[aspect_idx]
        presence_prob = float(pred["presence_prob"])
        spans = " ".join(token_span(tok, matrix[row_idx, i]) for i, tok in enumerate(tokens))
        sections.append(
            f"<h3>{html.escape(aspect)} "
            f"<small>presence={presence_prob:.3f}, pred={html.escape(pred['pred_label'])}, "
            f"true={html.escape(pred['true_label'])}</small></h3>\n<p>{spans}</p>"
        )

    body = "\n".join(sections)
    save_path.write_text(
        f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Attention map sample {sample_id}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; line-height: 1.5; margin: 24px; }}
    code, pre {{ white-space: pre-wrap; }}
    small {{ color: #555; font-weight: 400; }}
  </style>
</head>
<body>
  <h1>Attention map - sample {sample_id}</h1>
  <h2>Review gốc</h2>
  <p>{html.escape(raw_review)}</p>
  <h2>Processed review</h2>
  <p><code>{html.escape(processed_review)}</code></p>
  {body}
</body>
</html>
""",
        encoding="utf-8",
    )


def write_predictions_csv(
    save_path: Path,
    aspect_indices: list,
    labels: np.ndarray,
    preds: np.ndarray,
    presence_probs: np.ndarray,
    thresholds: np.ndarray,
    matrix: np.ndarray,
    tokens: list,
) -> list:
    rows = []
    for row_idx, aspect_idx in enumerate(aspect_indices):
        rows.append(
            {
                "aspect": ASPECT_COLUMNS[aspect_idx],
                "presence_prob": f"{presence_probs[aspect_idx]:.6f}",
                "threshold": f"{thresholds[aspect_idx]:.6f}",
                "pred_label": IDX_TO_LABEL[int(preds[aspect_idx])],
                "true_label": IDX_TO_LABEL[int(labels[aspect_idx])],
                "top_attention_tokens": top_tokens_for_aspect(matrix[row_idx], tokens),
            }
        )

    with save_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def render_sample(
    model: ABSAPhoBERT,
    tokenizer,
    row: pd.Series,
    sample_id: int,
    output_dir: Path,
    device: torch.device,
    max_len: int,
    top_k_aspects: int,
) -> dict:
    text = str(row.get("processed_review") or row.get("Review"))
    raw_review = str(row.get("Review", text))
    encoding = tokenizer(
        text,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    labels = row[ASPECT_COLUMNS].to_numpy(dtype=np.int64)
    alphas = compute_attention(model, input_ids, attention_mask)
    presence_probs, thresholds, _, preds = predict_one(model, input_ids, attention_mask)
    token_indices, tokens = visible_token_indices(input_ids, attention_mask, tokenizer)
    aspect_indices = choose_aspects(labels, preds, presence_probs, top_k_aspects)
    matrix = attention_matrix_for_aspects(alphas, token_indices, aspect_indices)

    prefix = f"sample_{sample_id:03d}"
    png_path = output_dir / f"{prefix}_aspect_token_matrix.png"
    html_path = output_dir / f"{prefix}_token_heatmap.html"
    csv_path = output_dir / f"{prefix}_predictions.csv"
    aspect_names = [ASPECT_COLUMNS[i] for i in aspect_indices]

    save_matrix_png(
        matrix,
        tokens,
        aspect_names,
        f"Aspect-token attention map - test sample {sample_id}",
        png_path,
    )
    rows = write_predictions_csv(
        csv_path,
        aspect_indices,
        labels,
        preds,
        presence_probs,
        thresholds,
        matrix,
        tokens,
    )
    save_token_html(
        sample_id,
        raw_review,
        text,
        matrix,
        tokens,
        aspect_indices,
        {i: rows[j] for j, i in enumerate(aspect_indices)},
        html_path,
    )
    return {
        "sample_id": sample_id,
        "raw_review": raw_review,
        "png": png_path,
        "html": html_path,
        "csv": csv_path,
        "rows": rows,
    }


def save_summary(output_dir: Path, rendered: list, checkpoint_path: Path) -> None:
    lines = [
        "# PhoBERT cls_only Attention Maps",
        "",
        f"Checkpoint: `{checkpoint_path}`",
        "",
        "Các ảnh dưới đây là attention pooling theo aspect, không phải self-attention thô của PhoBERT.",
        "Mỗi hàng là một aspect, mỗi cột là token trong review; màu càng đậm nghĩa là aspect đó dùng token nhiều hơn khi pooling representation.",
        "",
    ]
    for item in rendered:
        lines.extend(
            [
                f"## Test sample {item['sample_id']}",
                "",
                item["raw_review"],
                "",
                f"![Attention map]({item['png'].name})",
                "",
                f"- HTML token heatmap: `{item['html'].name}`",
                f"- Prediction detail: `{item['csv'].name}`",
                "",
            ]
        )
    (output_dir / "attention_maps_summary.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export aspect-aware attention maps for PhoBERT ABSA.")
    parser.add_argument(
        "--checkpoint",
        default="outputs/results/phobert_results_version3/models_cls_only/best_model.pt",
        help="Path tới best_model.pt.",
    )
    parser.add_argument("--encoder", default="cls_only", choices=["cls_only", "concat_4_layers"])
    parser.add_argument("--data", default="data/test_preprocessed.csv")
    parser.add_argument(
        "--output-dir",
        default="outputs/results/phobert_results_version3/attention_maps_cls_only",
    )
    parser.add_argument("--sample-indices", nargs="*", type=int, default=DEFAULT_SAMPLE_INDICES)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--top-k-aspects", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    data_path = Path(args.data)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {checkpoint_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Không tìm thấy data: {data_path}")

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(PHOBERT_MODEL_NAME)
    model = build_model(checkpoint_path, args.encoder, device)
    df = pd.read_csv(data_path)

    rendered = []
    for sample_id in args.sample_indices:
        if sample_id < 0 or sample_id >= len(df):
            print(f"[Skip] sample index ngoài range: {sample_id}")
            continue
        print(f"[Render] sample {sample_id}")
        rendered.append(
            render_sample(
                model,
                tokenizer,
                df.iloc[sample_id],
                sample_id,
                output_dir,
                device,
                args.max_len,
                args.top_k_aspects,
            )
        )

    save_summary(output_dir, rendered, checkpoint_path)
    print(f"[Done] Saved {len(rendered)} attention map groups to {output_dir}")


if __name__ == "__main__":
    main()
