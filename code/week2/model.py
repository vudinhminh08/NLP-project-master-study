"""
model.py — ABSAPhoBERT Multi-task model.

Viết lại theo đúng kiến trúc ds4v (Huynh et al. IEEE MAPR 2022):
    PhoBERT → concat last 4 hidden layers tại [CLS]
    → [batch, 3072] → Dropout(0.2)
    → 34 × Linear(3072, 4) song song
    → CrossEntropyLoss (tương đương binary_crossentropy + softmax của ds4v)

Khác với version cũ:
    - Bỏ class_weights hoàn toàn (ds4v không dùng weighted loss)
    - Loss tính đúng cách: torch.stack(losses).mean()
    - Không dùng requires_grad=True trick
"""

import torch
import torch.nn as nn
from transformers import AutoModel
from typing import Optional


class ABSAPhoBERT(nn.Module):
    """
    Multi-task PhoBERT cho ABSA VLSP 2018 Hotel.

    34 classification heads chạy song song trên cùng 1 [CLS] representation.
    Mỗi head predict 4 classes: absent / positive / negative / neutral.

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

        # PhoBERT — cần output_hidden_states để lấy 4 layers cuối
        self.phobert = AutoModel.from_pretrained(
            model_name,
            output_hidden_states=True,
        )

        # Hidden size: 768*4=3072 cho concat_4_layers, 768 cho cls_only
        self.hidden_size = 768 * 4 if encoder_option == "concat_4_layers" else 768
        self.dropout     = nn.Dropout(dropout)

        # 34 classification heads — ModuleList để track params đúng cách
        self.classifiers = nn.ModuleList([
            nn.Linear(self.hidden_size, num_labels)
            for _ in range(num_aspects)
        ])

        # Loss — không dùng class weights, đúng như ds4v
        self.criterion = nn.CrossEntropyLoss()

    def get_cls_representation(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Trả về [CLS] representation.

        concat_4_layers: concat hidden[-4:] tại token 0 → [batch, 3072]
        cls_only:        hidden[-1] tại token 0          → [batch, 768]
        """
        outputs       = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.hidden_states  # tuple 13 layers × [batch, seq, 768]

        if self.encoder_option == "concat_4_layers":
            # Đúng theo ds4v: concat 4 layers cuối tại [CLS] token
            cls_repr = torch.cat(
                [hidden_states[i][:, 0, :] for i in [-4, -3, -2, -1]],
                dim=-1,
            )  # [batch, 3072]
        else:
            cls_repr = hidden_states[-1][:, 0, :]  # [batch, 768]

        return cls_repr

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        class_weights: Optional[list] = None,  # giữ signature cũ, nhưng không dùng
    ) -> dict:
        """
        Forward pass.

        Args:
            input_ids:      [batch, seq_len]
            attention_mask: [batch, seq_len]
            labels:         [batch, 34] values 0-3, None khi inference
            class_weights:  KHÔNG DÙNG — giữ lại để không break API cũ

        Returns:
            dict: loss, logits, preds
        """
        # Encoder
        cls_repr = self.get_cls_representation(input_ids, attention_mask)
        cls_repr = self.dropout(cls_repr)

        # 34 heads song song
        logits = [clf(cls_repr) for clf in self.classifiers]  # 34 × [batch, 4]

        # Loss — đúng theo ds4v (không dùng class weights)
        loss = None
        if labels is not None:
            losses = []
            for i, logit in enumerate(logits):
                losses.append(self.criterion(logit, labels[:, i]))
            # torch.stack → [34] → mean → scalar, gradient flow đúng
            loss = torch.stack(losses).mean()

        # Predictions: argmax mỗi head
        preds = torch.stack(
            [logit.argmax(dim=-1) for logit in logits], dim=1
        )  # [batch, 34]

        return {"loss": loss, "logits": logits, "preds": preds}