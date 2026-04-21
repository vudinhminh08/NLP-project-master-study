
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
        self.num_aspects    = num_aspects
        self.focal_gamma    = focal_gamma
        self.num_labels     = num_labels
        self.acd_loss_weight = acd_loss_weight


        self.phobert = AutoModel.from_pretrained(
            model_name,
            output_hidden_states=True,
        )


        self.hidden_size = 768 * 4 if encoder_option == "concat_4_layers" else 768
        self.dropout     = nn.Dropout(dropout)

        # Two-head setup per aspect:
        # 1) ACD head predicts aspect present/absent
        # 2) SPC head predicts sentiment among {positive, negative, neutral}
        self.acd_heads = nn.ModuleList([
            nn.Linear(self.hidden_size, 1)
            for _ in range(num_aspects)
        ])
        self.spc_heads = nn.ModuleList([
            nn.Linear(self.hidden_size, 3)
            for _ in range(num_aspects)
        ])

    def get_cls_representation(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs       = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
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

        cls_repr = self.dropout(cls_repr)

        acd_logits = [head(cls_repr).squeeze(-1) for head in self.acd_heads]
        spc_logits = [head(cls_repr) for head in self.spc_heads]

        loss = None
        if labels is not None:
            acd_losses = []
            spc_losses = []

            for i in range(self.num_aspects):
                lbl = labels[:, i]  # 0=absent, 1=pos, 2=neg, 3=neu
                present_target = (lbl > 0).float()

                # --- ACD binary loss ---
                bce_per_sample = F.binary_cross_entropy_with_logits(
                    acd_logits[i],
                    present_target,
                    reduction="none",
                )

                if class_weights is not None:
                    w_abs = class_weights[i][0]
                    w_pos = class_weights[i][1:].mean()
                    acd_sample_w = torch.where(present_target > 0.5, w_pos, w_abs)
                    acd_loss_i = (bce_per_sample * acd_sample_w).sum() / acd_sample_w.sum().clamp(min=1e-8)
                else:
                    acd_loss_i = bce_per_sample.mean()
                acd_losses.append(acd_loss_i)

                # --- SPC sentiment loss, only where aspect is present ---
                present_mask = lbl > 0
                if present_mask.any():
                    spc_target = lbl[present_mask] - 1  # map {1,2,3} -> {0,1,2}
                    spc_logit_present = spc_logits[i][present_mask]

                    ce_per_sample = F.cross_entropy(
                        spc_logit_present,
                        spc_target,
                        reduction="none",
                    )
                    probs = F.softmax(spc_logit_present, dim=-1)
                    p_true = probs.gather(1, spc_target.unsqueeze(1)).squeeze(1)
                    focal_factor = (1.0 - p_true.detach()) ** self.focal_gamma
                    ce_focal = ce_per_sample * focal_factor

                    if class_weights is not None:
                        # Use original per-label weights for sentiment classes {1,2,3}
                        spc_sample_w = class_weights[i][lbl[present_mask]]
                        spc_loss_i = (ce_focal * spc_sample_w).sum() / spc_sample_w.sum().clamp(min=1e-8)
                    else:
                        spc_loss_i = ce_focal.mean()
                    spc_losses.append(spc_loss_i)

            acd_loss = torch.stack(acd_losses).mean() if acd_losses else torch.tensor(0.0, device=cls_repr.device)
            spc_loss = torch.stack(spc_losses).mean() if spc_losses else torch.tensor(0.0, device=cls_repr.device)
            loss = self.acd_loss_weight * acd_loss + (1.0 - self.acd_loss_weight) * spc_loss

        # Reconstruct original 4-class prediction format for evaluation pipeline.
        acd_pred = torch.stack([(torch.sigmoid(lg) > 0.5).long() for lg in acd_logits], dim=1)
        spc_pred = torch.stack([torch.argmax(lg, dim=-1) + 1 for lg in spc_logits], dim=1)
        preds = torch.where(acd_pred > 0, spc_pred, torch.zeros_like(spc_pred))

        return {
            "loss": loss,
            "logits": {"acd": acd_logits, "spc": spc_logits},
            "preds": preds,
        }
        return {"loss": loss, "logits": logits, "preds": preds}
