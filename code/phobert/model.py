
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
        acd_loss_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.encoder_option = encoder_option
        self.num_aspects = num_aspects
        self.focal_gamma = focal_gamma
        self.num_labels = num_labels
        # Kept for backward compatibility with run_experiment config.
        self.acd_loss_weight = acd_loss_weight

        self.phobert = AutoModel.from_pretrained(
            model_name,
            output_hidden_states=True,
        )

        self.hidden_size = 768 * 4 if encoder_option == "concat_4_layers" else 768
        self.dropout = nn.Dropout(dropout)

        # Stable baseline head: 4-class classifier per aspect
        self.classifiers = nn.ModuleList([
            nn.Linear(self.hidden_size, num_labels)
            for _ in range(num_aspects)
        ])

    def get_cls_representation(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.hidden_states

        if self.encoder_option == "concat_4_layers":
            cls_repr = torch.cat(
                [hidden_states[i][:, 0, :] for i in [-4, -3, -2, -1]],
                dim=-1,
            )
        else:
            cls_repr = hidden_states[-1][:, 0, :]

        return cls_repr

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        class_weights: Optional[list] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> dict:
        cls_repr = self.get_cls_representation(input_ids, attention_mask)

        # MC dropout style averaging helps stabilize per-aspect logits.
        if self.training:
            n_dropout = 5
            all_logits = []
            for _ in range(n_dropout):
                dropped = self.dropout(cls_repr)
                all_logits.append([clf(dropped) for clf in self.classifiers])
            logits = [
                torch.stack([all_logits[n][i] for n in range(n_dropout)]).mean(0)
                for i in range(len(self.classifiers))
            ]
        else:
            dropped = self.dropout(cls_repr)
            logits = [clf(dropped) for clf in self.classifiers]

        loss = None
        if labels is not None:
            losses = []
            for i in range(self.num_aspects):
                logit = logits[i]
                lbl = labels[:, i]

                log_probs = F.log_softmax(logit, dim=-1)
                log_p_true = log_probs.gather(1, lbl.unsqueeze(1)).squeeze(1)
                p_true = log_p_true.exp()
                focal_factor = (1.0 - p_true.detach()) ** self.focal_gamma
                ce_per_sample = -log_p_true

                if class_weights is not None:
                    sample_w = class_weights[i][lbl]
                    weighted = sample_w * focal_factor * ce_per_sample
                    loss_i = weighted.sum() / sample_w.sum().clamp(min=1e-8)
                else:
                    loss_i = (focal_factor * ce_per_sample).mean()

                losses.append(loss_i)

            loss = torch.stack(losses).mean()

        preds = torch.stack([logit.argmax(dim=-1) for logit in logits], dim=1)
        return {
            "loss": loss,
            "logits": logits,
            "preds": preds,
        }
