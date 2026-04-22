
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
        attn_dim: int = 128,
        use_split_loss: bool = True,
        lambda_presence: float = 1.0,
        lambda_sentiment: float = 1.0,
        presence_threshold: float = 0.5,
        rare_aspect_ids: Optional[list] = None,
        rare_presence_pos_mult: float = 1.0,
        rare_sentiment_mult: float = 1.0,
    ) -> None:
        super().__init__()
        self.encoder_option = encoder_option
        self.num_aspects    = num_aspects
        self.focal_gamma    = focal_gamma
        self.num_labels     = num_labels
        self.attn_dim       = attn_dim
        self.use_split_loss = use_split_loss
        self.lambda_presence = lambda_presence
        self.lambda_sentiment = lambda_sentiment
        self.presence_threshold = presence_threshold
        self.rare_aspect_ids = set(rare_aspect_ids or [])
        self.rare_presence_pos_mult = float(rare_presence_pos_mult)
        self.rare_sentiment_mult = float(rare_sentiment_mult)
        self.register_buffer(
            "aspect_presence_thresholds",
            torch.full((self.num_aspects,), float(presence_threshold), dtype=torch.float32),
        )


        self.phobert = AutoModel.from_pretrained(
            model_name,
            output_hidden_states=True,
        )


        self.hidden_size = 768 * 4 if encoder_option == "concat_4_layers" else 768
        self.dropout     = nn.Dropout(dropout)

        # Aspect-aware attention pooling parameters
        self.token_proj = nn.Linear(self.hidden_size, self.attn_dim)
        self.aspect_queries = nn.Parameter(torch.randn(self.num_aspects, self.attn_dim))
        nn.init.xavier_uniform_(self.token_proj.weight)
        nn.init.normal_(self.aspect_queries, mean=0.0, std=0.02)


        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.LayerNorm(self.hidden_size),
                nn.Linear(self.hidden_size, num_labels),
            )
            for _ in range(num_aspects)
        ])
        self.presence_heads = nn.ModuleList([
            nn.Linear(self.hidden_size, 1) for _ in range(num_aspects)
        ])

    def get_aspect_representations(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.hidden_states

        if self.encoder_option == "concat_4_layers":
            H = torch.cat([hidden_states[i] for i in [-4, -3, -2, -1]], dim=-1)
        else:
            H = hidden_states[-1]

        proj = torch.tanh(self.token_proj(H))  # (B, T, D)
        scores = torch.matmul(proj, self.aspect_queries.t())  # (B, T, A)

        mask = (attention_mask == 0).unsqueeze(-1)  # (B, T, 1)
        neg_inf = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(mask, neg_inf)

        alphas = torch.softmax(scores, dim=1)  # (B, T, A)
        pooled = torch.einsum("bta,bth->bah", alphas, H)  # (B, A, H)

        return pooled

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        class_weights: Optional[list] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> dict:

        aspect_repr = self.get_aspect_representations(input_ids, attention_mask)



        if self.training:
            N_DROPOUT = 5
            all_logits = []
            all_presence = []
            for _ in range(N_DROPOUT):
                dropped = self.dropout(aspect_repr)
                all_logits.append([
                    clf(dropped[:, i, :]) for i, clf in enumerate(self.classifiers)
                ])
                all_presence.append([
                    head(dropped[:, i, :]).squeeze(-1)
                    for i, head in enumerate(self.presence_heads)
                ])
            logits = [
                torch.stack([all_logits[n][i] for n in range(N_DROPOUT)]).mean(0)
                for i in range(len(self.classifiers))
            ]
            presence_logits = [
                torch.stack([all_presence[n][i] for n in range(N_DROPOUT)]).mean(0)
                for i in range(len(self.presence_heads))
            ]
        else:
            aspect_repr = self.dropout(aspect_repr)
            logits = [
                clf(aspect_repr[:, i, :]) for i, clf in enumerate(self.classifiers)
            ]
            presence_logits = [
                head(aspect_repr[:, i, :]).squeeze(-1)
                for i, head in enumerate(self.presence_heads)
            ]













        loss = None
        if labels is not None:
            losses = []
            for i, logit in enumerate(logits):
                lbl = labels[:, i]
                if self.use_split_loss:
                    exist_target = (lbl > 0).float()
                    presence_bce = F.binary_cross_entropy_with_logits(
                        presence_logits[i],
                        exist_target,
                        reduction="none",
                    )

                    if class_weights is not None:
                        presence_w = torch.where(
                            exist_target > 0,
                            class_weights[i][1],
                            class_weights[i][0],
                        )
                        if i in self.rare_aspect_ids and self.rare_presence_pos_mult != 1.0:
                            presence_w = torch.where(
                                exist_target > 0,
                                presence_w * self.rare_presence_pos_mult,
                                presence_w,
                            )
                        presence_loss = (presence_bce * presence_w).sum() / presence_w.sum().clamp(min=1e-8)
                    else:
                        presence_loss = presence_bce.mean()

                    present_mask = lbl > 0
                    if present_mask.any():
                        sent_logits = logit[present_mask, 1:]
                        sent_lbl = lbl[present_mask] - 1
                        sent_log_probs = F.log_softmax(sent_logits, dim=-1)
                        log_p_true = sent_log_probs.gather(1, sent_lbl.unsqueeze(1)).squeeze(1)
                        p_true = log_p_true.exp()
                        focal_factor = (1.0 - p_true.detach()) ** self.focal_gamma
                        ce_per_sample = -log_p_true

                        if class_weights is not None:
                            sample_w = class_weights[i][lbl[present_mask]]
                            weighted = sample_w * focal_factor * ce_per_sample
                            sentiment_loss = weighted.sum() / sample_w.sum().clamp(min=1e-8)
                        else:
                            sentiment_loss = (focal_factor * ce_per_sample).mean()

                        if i in self.rare_aspect_ids and self.rare_sentiment_mult != 1.0:
                            sentiment_loss = sentiment_loss * self.rare_sentiment_mult
                    else:
                        sentiment_loss = torch.zeros((), device=logit.device)

                    loss_i = (
                        self.lambda_presence * presence_loss
                        + self.lambda_sentiment * sentiment_loss
                    )
                else:
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


        if self.use_split_loss:
            preds = []
            for i, logit in enumerate(logits):
                exist_prob = torch.sigmoid(presence_logits[i])
                thr_i = self.aspect_presence_thresholds[i].to(exist_prob.dtype)
                present_pred = exist_prob >= thr_i
                sent_pred = logit[:, 1:].argmax(dim=-1) + 1
                pred_i = torch.where(
                    present_pred,
                    sent_pred,
                    torch.zeros_like(sent_pred),
                )
                preds.append(pred_i)
            preds = torch.stack(preds, dim=1)
        else:
            preds = torch.stack(
                [logit.argmax(dim=-1) for logit in logits], dim=1
            )

        return {
            "loss": loss,
            "logits": logits,
            "presence_logits": presence_logits,
            "preds": preds,
        }
