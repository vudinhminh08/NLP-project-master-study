"""
model.py — ABSAPhoBERT Multi-task model.

Kiến trúc theo ds4v (Huynh et al. IEEE MAPR 2022):
    PhoBERT → concat last 4 hidden layers tại [CLS]
    → [batch, 3072] → Dropout(0.2)
    → 34 × Linear(3072, 4) song song

BUG FIX so với version trước:
    - Version cũ: tự ý bỏ class_weights → model bias hoàn toàn về class absent
    - Version này: dùng F.cross_entropy(weight=...) đúng spec EDA → xử lý imbalance
    - Lý do dùng class weights: 85% nhãn là absent, neutral global weight=154
      → không có weighted loss thì model không học được minority classes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from typing import Optional


class ABSAPhoBERT(nn.Module):
    """
    Multi-task PhoBERT cho ABSA VLSP 2018 Hotel.

    34 classification heads chạy song song trên cùng 1 [CLS] representation.
    Mỗi head predict 4 classes: absent(0) / positive(1) / negative(2) / neutral(3).

    Args:
        model_name:     HuggingFace model id, mặc định "vinai/phobert-base"
        num_aspects:    số lượng aspect heads, mặc định 34
        num_labels:     số lượng classes mỗi head, mặc định 4
        dropout:        dropout rate trước classifiers, mặc định 0.2
        encoder_option: "concat_4_layers" (SOTA, 3072 dim) hoặc "cls_only" (768 dim)
    """

    def __init__(
        self,
        model_name: str = "vinai/phobert-base",
        num_aspects: int = 34,
        num_labels: int = 4,
        dropout: float = 0.2,
        encoder_option: str = "concat_4_layers",
    ) -> None:
        super().__init__()
        self.encoder_option = encoder_option
        self.num_aspects    = num_aspects
        self.num_labels     = num_labels

        # PhoBERT — cần output_hidden_states=True để lấy 4 layers cuối
        self.phobert = AutoModel.from_pretrained(
            model_name,
            output_hidden_states=True,
        )

        # Hidden size: 768*4=3072 cho concat_4_layers, 768 cho cls_only
        self.hidden_size = 768 * 4 if encoder_option == "concat_4_layers" else 768
        self.dropout     = nn.Dropout(dropout)

        # 34 classification heads — ModuleList để PyTorch track params đúng
        self.classifiers = nn.ModuleList([
            nn.Linear(self.hidden_size, num_labels)
            for _ in range(num_aspects)
        ])

        # Fallback criterion (không có class weights) — dùng khi class_weights=None
        self.criterion = nn.CrossEntropyLoss()

    def get_cls_representation(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Trả về [CLS] representation tại vị trí token 0.

        concat_4_layers: concat hidden[-4:] tại [CLS] → [batch, 3072]
        cls_only:        hidden[-1] tại [CLS]          → [batch, 768]

        Args:
            input_ids:      [batch, seq_len]
            attention_mask: [batch, seq_len]

        Returns:
            cls_repr: [batch, hidden_size]
        """
        outputs       = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.hidden_states  # tuple 13 tensors × [batch, seq, 768]

        if self.encoder_option == "concat_4_layers":
            # Đúng theo ds4v: concat 4 layers cuối tại [CLS] token (index 0)
            cls_repr = torch.cat(
                [hidden_states[i][:, 0, :] for i in [-4, -3, -2, -1]],
                dim=-1,
            )  # [batch, 3072]
        else:
            # cls_only: ablation study (768 dim)
            cls_repr = hidden_states[-1][:, 0, :]  # [batch, 768]

        return cls_repr

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        class_weights: Optional[list] = None,
    ) -> dict:
        """
        Forward pass.

        FIX: class_weights ĐƯỢC SỬ DỤNG, không bị bỏ qua.
        Lý do: mất cân bằng nghiêm trọng (85% absent, neutral weight=154 → clip=10).
        Nếu không dùng weighted loss, model sẽ bias về absent và F1 sẽ rất thấp.

        Args:
            input_ids:      [batch, seq_len]
            attention_mask: [batch, seq_len]
            labels:         [batch, 34] values 0-3, None khi inference
            class_weights:  list of 34 tensors [4] — per-aspect weights, None = unweighted

        Returns:
            dict:
                loss:   scalar tensor (None khi inference)
                logits: list of 34 tensors [batch, 4]
                preds:  [batch, 34] predictions (argmax per head)
        """
        # === Encoder ===
        cls_repr = self.get_cls_representation(input_ids, attention_mask)
        cls_repr = self.dropout(cls_repr)

        # === 34 heads song song ===
        logits = [clf(cls_repr) for clf in self.classifiers]  # 34 × [batch, 4]

        # === Loss (chỉ tính khi có labels) ===
        loss = None
        if labels is not None:
            losses = []
            for i, logit in enumerate(logits):
                if class_weights is not None:
                    # Per-aspect weighted loss — xử lý class imbalance
                    # F.cross_entropy tính inline, không tạo object mới → efficient
                    loss_i = F.cross_entropy(
                        logit, labels[:, i],
                        weight=class_weights[i],  # tensor [4] trên cùng device
                    )
                else:
                    # Fallback: unweighted (chỉ dùng cho inference debug)
                    loss_i = F.cross_entropy(logit, labels[:, i])
                losses.append(loss_i)

            # torch.stack → [34] → mean → scalar gradient flow đúng
            loss = torch.stack(losses).mean()

        # === Predictions ===
        preds = torch.stack(
            [logit.argmax(dim=-1) for logit in logits], dim=1
        )  # [batch, 34]

        return {"loss": loss, "logits": logits, "preds": preds}
