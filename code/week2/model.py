"""
model.py — ABSAPhoBERT Multi-task model.

Architecture (theo SOTA ds4v IEEE 2022):
    PhoBERT → concat last 4 hidden layers tại [CLS]
    → [batch, 3072] → Dropout(0.2)
    → 34 × Linear(3072, 4) song song
    → 34 × softmax(4 classes)

Ablation option:
    encoder_option="cls_only" → [batch, 768] → 34 × Linear(768, 4)

Chạy từ root project:
    python -c "from code.week2.model import ABSAPhoBERT"
"""

import torch
import torch.nn as nn
from transformers import AutoModel
from typing import Optional, List


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
        self.num_aspects = num_aspects

        # PhoBERT — cần output_hidden_states để lấy 4 layers cuối
        self.phobert = AutoModel.from_pretrained(
            model_name,
            output_hidden_states=True,
        )

        self.hidden_size = 768 * 4 if encoder_option == "concat_4_layers" else 768
        self.dropout = nn.Dropout(dropout)

        # 34 classification heads — ModuleList để track params đúng cách
        self.classifiers = nn.ModuleList([
            nn.Linear(self.hidden_size, num_labels)
            for _ in range(num_aspects)
        ])

    def get_cls_representation(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Trả về [CLS] representation.

        concat_4_layers: concat hidden[-4:] tại token 0 → shape [batch, 3072]
        cls_only:        hidden[-1] tại token 0              → shape [batch, 768]

        Lý do concat 4 layers: mỗi layer học đặc trưng khác nhau (syntax, semantics...),
        concat cho phép model tận dụng thông tin đa tầng → +1-2% F1 theo ds4v SOTA.
        """
        outputs = self.phobert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden_states = outputs.hidden_states  # tuple: 13 layers × [batch, seq, 768]

        if self.encoder_option == "concat_4_layers":
            cls_repr = torch.cat(
                [hidden_states[i][:, 0, :] for i in range(-4, 0)],
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
        class_weights: Optional[List[torch.Tensor]] = None,
    ) -> dict:
        """
        Forward pass.

        Args:
            input_ids:      [batch, seq_len] — token ids từ PhoBERT tokenizer
            attention_mask: [batch, seq_len] — 1 cho real tokens, 0 cho padding
            labels:         [batch, 34] values 0-3, None khi inference
            class_weights:  list of 34 tensors [4] — per-aspect class weights

        Returns:
            dict:
                loss:   scalar tensor (None nếu labels=None)
                logits: list of 34 tensors [batch, 4]
                preds:  [batch, 34] — argmax predictions
        """
        cls_repr = self.get_cls_representation(input_ids, attention_mask)
        cls_repr = self.dropout(cls_repr)

        logits = [clf(cls_repr) for clf in self.classifiers]  # 34 × [batch, 4]

        loss = None
        if labels is not None:
            loss = torch.tensor(0.0, device=input_ids.device, requires_grad=True)
            for i, logit in enumerate(logits):
                w = class_weights[i].to(input_ids.device) if class_weights else None
                criterion = nn.CrossEntropyLoss(weight=w)
                loss = loss + criterion(logit, labels[:, i])
            loss = loss / self.num_aspects

        preds = torch.stack(
            [logit.argmax(dim=-1) for logit in logits], dim=1
        )  # [batch, 34]

        return {"loss": loss, "logits": logits, "preds": preds}
