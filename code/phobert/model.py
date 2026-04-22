
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from typing import Optional

try:
    from utils.constants import ASPECT_COLUMNS, ENTITY_GROUPS
except Exception:
    ASPECT_COLUMNS = []
    ENTITY_GROUPS = {}


class ABSAPhoBERT(nn.Module):

    def __init__(
        self,
        model_name: str = "vinai/phobert-base-v2",
        num_aspects: int = 34,
        num_labels: int = 4,
        dropout: float = 0.2,
        encoder_option: str = "concat_4_layers",
        focal_gamma: float = 2.0,
        acd_loss_weight: float = 0.4,
        acd_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.encoder_option = encoder_option
        self.num_aspects    = num_aspects
        self.focal_gamma    = focal_gamma
        self.num_labels     = num_labels
        self.acd_loss_weight = acd_loss_weight
        self.acd_threshold = acd_threshold


        self.phobert = AutoModel.from_pretrained(
            model_name,
            output_hidden_states=True,
        )


        self.hidden_size = 768 * 4 if encoder_option == "concat_4_layers" else 768
        self.dropout = nn.Dropout(dropout)
        self.aspect_columns = ASPECT_COLUMNS if len(ASPECT_COLUMNS) == num_aspects else [f"A{i}" for i in range(num_aspects)]

        self.entity_names = list(ENTITY_GROUPS.keys())
        self.entity_to_aspect_idx = []
        if self.entity_names:
            for entity in self.entity_names:
                idxs = []
                for aspect in ENTITY_GROUPS[entity]:
                    if aspect in self.aspect_columns:
                        idxs.append(self.aspect_columns.index(aspect))
                if idxs:
                    self.entity_to_aspect_idx.append(idxs)
                else:
                    self.entity_to_aspect_idx.append([])
        self.num_entities = len(self.entity_names)

        self.shared_layer = nn.Linear(self.hidden_size, self.hidden_size)
        self.entity_classifiers = nn.ModuleList([
            nn.Linear(self.hidden_size, 1)
            for _ in range(self.num_entities)
        ])
        self.entity_proj = nn.Linear(max(self.num_entities, 1), self.hidden_size)
        fused_size = self.hidden_size * 2

        self.acd_classifiers = nn.ModuleList([
            nn.Linear(fused_size, 1)
            for _ in range(num_aspects)
        ])
        self.spc_classifiers = nn.ModuleList([
            nn.Linear(fused_size, 3)
            for _ in range(num_aspects)
        ])

        # Backward-compatible alias for internal smoke tests and older utilities.
        self.classifiers = self.spc_classifiers

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
    ) -> dict:

        cls_repr = self.get_cls_representation(input_ids, attention_mask)



        if self.training:
            N_DROPOUT = 5
            for _ in range(N_DROPOUT):
                dropped = self.dropout(cls_repr)
                shared = F.gelu(self.shared_layer(dropped))
                shared = self.dropout(shared)

                if self.num_entities > 0:
                    entity_logits = [clf(shared).squeeze(-1) for clf in self.entity_classifiers]
                    entity_prob = torch.sigmoid(torch.stack(entity_logits, dim=1))
                else:
                    entity_logits = []
                    entity_prob = torch.zeros(shared.size(0), 1, device=shared.device, dtype=shared.dtype)

                entity_context = torch.tanh(self.entity_proj(entity_prob))
                fused = torch.cat([shared, entity_context], dim=-1)
                fused = self.dropout(fused)

                acd_logits = [clf(fused).squeeze(-1) for clf in self.acd_classifiers]
                spc_logits = [clf(fused) for clf in self.spc_classifiers]

                # Weakly supervise entity branch from aspect presence to stabilize shared layer.
                entity_loss = None
                if labels is not None and self.num_entities > 0:
                    acd_targets = (labels > 0).float()
                    entity_losses = []
                    for i, idxs in enumerate(self.entity_to_aspect_idx):
                        if not idxs:
                            continue
                        target = acd_targets[:, idxs].max(dim=1).values
                        loss_i = F.binary_cross_entropy_with_logits(entity_logits[i], target)
                        entity_losses.append(loss_i)
                    if entity_losses:
                        entity_loss = torch.stack(entity_losses).mean()

                loss_acd = None
                loss_spc = None
                loss = None

                if labels is not None:
                    acd_targets = (labels > 0).float()
                    spc_targets = (labels - 1).clamp(min=0, max=2)

                    acd_losses = []
                    spc_losses = []

                    for i in range(self.num_aspects):
                        acd_logit = acd_logits[i]
                        acd_target = acd_targets[:, i]
                        bce = F.binary_cross_entropy_with_logits(acd_logit, acd_target, reduction="none")

                        if class_weights is not None:
                            neg_w = class_weights[i][0]
                            pos_w = class_weights[i][1:].mean()
                            sample_w = torch.where(acd_target > 0.5, pos_w, neg_w)
                            acd_loss_i = (bce * sample_w).sum() / sample_w.sum().clamp(min=1e-8)
                        else:
                            acd_loss_i = bce.mean()
                        acd_losses.append(acd_loss_i)

                        pos_mask = acd_target > 0.5
                        if pos_mask.any():
                            logits_i = spc_logits[i][pos_mask]
                            target_i = spc_targets[pos_mask, i].long()
                            ce = F.cross_entropy(logits_i, target_i, reduction="none")

                            if class_weights is not None:
                                spc_w = class_weights[i][1:]
                                sample_w = spc_w[target_i]
                                focal_factor = (1.0 - torch.softmax(logits_i, dim=-1).gather(1, target_i.unsqueeze(1)).squeeze(1).detach()) ** self.focal_gamma
                                weighted = ce * sample_w * focal_factor
                                spc_loss_i = weighted.sum() / sample_w.sum().clamp(min=1e-8)
                            else:
                                spc_loss_i = ce.mean()
                            spc_losses.append(spc_loss_i)

                    loss_acd = torch.stack(acd_losses).mean() if acd_losses else torch.tensor(0.0, device=labels.device)
                    loss_spc = torch.stack(spc_losses).mean() if spc_losses else torch.tensor(0.0, device=labels.device)
                    loss = self.acd_loss_weight * loss_acd + (1.0 - self.acd_loss_weight) * loss_spc
                    if entity_loss is not None:
                        loss = loss + 0.1 * entity_loss

                acd_prob = torch.sigmoid(torch.stack(acd_logits, dim=1))
                acd_pred = acd_prob >= self.acd_threshold
                spc_pred = torch.stack([logit.argmax(dim=-1) + 1 for logit in spc_logits], dim=1)
                preds = torch.where(acd_pred, spc_pred, torch.zeros_like(spc_pred))

                return {
                    "loss": loss,
                    "logits": spc_logits,
                    "acd_logits": acd_logits,
                    "preds": preds,
                    "loss_acd": loss_acd,
                    "loss_spc": loss_spc,
                }
