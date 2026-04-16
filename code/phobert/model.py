
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from typing import Optional


class ABSAPhoBERT(nn.Module):

    def __init__(
        self,
        model_name: str = "vinai/phobert-base-v2",
        num_aspects: int = 34,
        num_labels: int = 4,
        dropout: float = 0.2,
        encoder_option: str = "concat_4_layers",
        focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.encoder_option = encoder_option
        self.num_aspects    = num_aspects
        self.focal_gamma    = focal_gamma
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
        token_type_ids: Optional[torch.Tensor] = None,  # ignored: PhoBERT is RoBERTa-based
    ) -> dict:
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

        # === Loss: 34 × weighted Focal Loss (v2.6) ===
        # Focal loss = -w_c × (1 - p_true)^γ × log(p_true)
        # γ=2: down-weight easy samples (absent đã đúng),
        #       focus gradient vào hard/rare aspects
        #
        # Implementation đúng:
        #   1. log_softmax → lấy log(p_true) per sample qua gather
        #   2. p_true = exp(log_p_true) — xác suất raw của đúng class
        #   3. focal_factor = (1 - p_true)^γ
        #   4. sample_w = class_weight[true_class] per sample
        #   5. loss = mean(sample_w × focal_factor × (-log_p_true))
        # Cách này đảm bảo focal_loss ≤ CE và semantics weight đúng như CE
        loss = None
        if labels is not None:
            losses = []
            for i, logit in enumerate(logits):
                lbl = labels[:, i]                                    # [batch]
                log_probs = F.log_softmax(logit, dim=-1)              # [batch, 4]

                # log(p_true) và p_true
                log_p_true = log_probs.gather(1, lbl.unsqueeze(1)).squeeze(1)  # [batch]
                p_true     = log_p_true.exp()                         # [batch]

                # Focal factor: (1 - p_true)^γ
                focal_factor = (1.0 - p_true.detach()) ** self.focal_gamma  # [batch]

                # CE per-sample (không weight, không smoothing) = -log_p_true
                ce_per_sample = -log_p_true                           # [batch]

                # Apply class weight per sample: w[true_class]
                # Normalize bằng sum(weights) để khớp với F.cross_entropy(weight=w)
                # PyTorch CE: mean = sum(w[c] * loss) / sum(w[c]), không chia batch_size
                if class_weights is not None:
                    sample_w = class_weights[i][lbl]                  # [batch]
                    weighted = sample_w * focal_factor * ce_per_sample
                    loss_i   = weighted.sum() / sample_w.sum().clamp(min=1e-8)
                else:
                    loss_i = (focal_factor * ce_per_sample).mean()

                losses.append(loss_i)
            loss = torch.stack(losses).mean()

        # === Predictions ===
        preds = torch.stack(
            [logit.argmax(dim=-1) for logit in logits], dim=1
        )  # [batch, 34]

        return {"loss": loss, "logits": logits, "preds": preds}
