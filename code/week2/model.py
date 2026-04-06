"""
model.py — ABSAPhoBERT Multi-task model.

Kiến trúc theo ds4v (Huynh et al. IEEE MAPR 2022):
    PhoBERT → concat last 4 hidden layers tại [CLS]
    → [batch, 3072] → Dropout(0.2)
    → 34 × Linear(3072, 4) song song

Version 2.5 (Flat BCE Loss):
    - Loss: flat binary_crossentropy trên concatenated softmax outputs [batch, 136]
      thay vì 34 cross_entropy riêng biệt
    - Lý do: ds4v v1 dùng kiến trúc này → Combined F1 0.7732, trong khi v2 (separate heads)
      chỉ đạt ~0.55 — cùng dataset, cùng architecture, chỉ khác loss formulation
    - Cơ chế: gradient từ 1 loss node chảy qua tất cả 34 heads → model học được
      correlation giữa các aspects (nếu ROOMS#CLEANLINESS positive thì ROOMS#GENERAL
      có xu hướng positive). 34 CE riêng không học được điều này.
    - Class weights vẫn giữ nguyên: build weight vector [136] từ per-aspect weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from transformers import AutoModel
from typing import Optional


class ABSAPhoBERT(nn.Module):
    """
    Multi-task PhoBERT cho ABSA VLSP 2018 Hotel.

    34 classification heads chạy song song trên cùng 1 [CLS] representation.
    Mỗi head predict 4 classes: absent(0) / positive(1) / negative(2) / neutral(3).

    Args:
        model_name:     HuggingFace model id, mặc định "vinai/phobert-base-v2"
        num_aspects:    số lượng aspect heads, mặc định 34
        num_labels:     số lượng classes mỗi head, mặc định 4
        dropout:        dropout rate trước classifiers, mặc định 0.2
        encoder_option: "concat_4_layers" (SOTA, 3072 dim) hoặc "cls_only" (768 dim)
    """

    def __init__(
        self,
        model_name: str = "vinai/phobert-base-v2",
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

        # Multi-sample dropout (Inoue 2019) — chỉ khi training
        # Dropout N lần, average logits → regularization mạnh hơn cho dataset nhỏ
        if self.training:
            N_DROPOUT = 5
            all_logits = []
            for _ in range(N_DROPOUT):
                dropped = self.dropout(cls_repr)
                all_logits.append([clf(dropped) for clf in self.classifiers])
            logits = [
                torch.stack([all_logits[n][i] for n in range(N_DROPOUT)]).mean(0)
                for i in range(len(self.classifiers))
            ]
        else:
            cls_repr = self.dropout(cls_repr)
            logits = [clf(cls_repr) for clf in self.classifiers]  # 34 × [batch, 4]

        # === Loss: Flat BCE (ds4v v1 architecture) ===
        # Thay vì 34 cross_entropy riêng biệt, concat tất cả softmax outputs
        # thành vector 136-dim rồi dùng binary_crossentropy.
        # Gradient chảy qua 1 loss node → model học correlation giữa aspects.
        loss = None
        if labels is not None:
            # [batch, 136] — concat softmax probs của 34 heads
            probs = torch.cat(
                [F.softmax(logit, dim=-1) for logit in logits], dim=-1
            ).clamp(1e-7, 1 - 1e-7)

            # [batch, 34, 4] → [batch, 136] — one-hot flat labels
            flat_labels = F.one_hot(labels, num_classes=self.num_labels).float().view(
                labels.size(0), -1
            )

            # F.binary_cross_entropy không tương thích với AMP autocast (float16).
            # Disable autocast cục bộ tại đây → BCE luôn chạy ở float32.
            # Encoder vẫn chạy float16 bình thường, chỉ loss step này là float32.
            with autocast(enabled=False):
                if class_weights is not None:
                    weight_vec = torch.cat(class_weights, dim=0)  # [136]
                    loss = F.binary_cross_entropy(
                        probs.float(), flat_labels.float(),
                        weight=weight_vec.float().unsqueeze(0).expand_as(probs),
                    )
                else:
                    loss = F.binary_cross_entropy(probs.float(), flat_labels.float())

        # === Predictions ===
        preds = torch.stack(
            [logit.argmax(dim=-1) for logit in logits], dim=1
        )  # [batch, 34]

        return {"loss": loss, "logits": logits, "preds": preds}
