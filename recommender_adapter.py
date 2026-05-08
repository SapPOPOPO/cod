import torch
import torch.nn as nn
import torch.nn.functional as F
from models import SASRecModel as _BaseSASRec


class RecommenderWrapper(nn.Module):
    """Adds a uniform forward() and next_item_loss() around the base SASRec."""

    def __init__(self, args):
        super().__init__()
        self.base = _BaseSASRec(args)
        self.item_embeddings = self.base.item_embeddings   # expose for augmenter sharing

    def transformer_encoder(self, input_ids):
        """Passthrough so evaluate.py keeps working."""
        return self.base.transformer_encoder(input_ids)

    def _encode_seq(self, input_ids):
        h = self.base.transformer_encoder(input_ids)
        own_mask = (input_ids > 0)
        return h, own_mask

    def forward(self, input_ids):
        """Last-position pooled feature [B, D] for InfoNCE."""
        h = self.base.transformer_encoder(input_ids)
        return h[:, -1, :]

    def next_item_loss(self, batch):
        """CoSeRec-style BCE — identical to Trainer.cross_entropy in your original code."""
        input_ids  = batch["input_ids"]
        target_pos = batch["target_pos"]
        target_neg = batch["target_neg"]

        seq_out = self.base.transformer_encoder(input_ids)        # [B, L, D]
        pos_emb = self.item_embeddings(target_pos)
        neg_emb = self.item_embeddings(target_neg)

        D = seq_out.size(-1)
        pos = pos_emb.view(-1, D)
        neg = neg_emb.view(-1, D)
        seq = seq_out.view(-1, D)

        pos_logits = (pos * seq).sum(-1)
        neg_logits = (neg * seq).sum(-1)
        istarget = (target_pos > 0).view(-1).float()

        loss = torch.sum(
            -torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget
            -torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
        ) / istarget.sum().clamp(min=1.0)
        return loss