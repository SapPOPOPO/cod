"""
Adapter around a SASRec-style recommender so the v2 training step can call it
uniformly. Wrap your existing model OR modify it to expose:

    forward(input_ids) -> [B, D]   (last-token / pooled representation)
    next_item_loss(batch) -> scalar tensor

You probably already have these inside trainers.py; just expose them on the
model class and import in main_v2.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import Encoder, LayerNorm


class SASRecModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.D = args.hidden_size
        self.item_embeddings = nn.Embedding(args.item_size, self.D, padding_idx=0)
        self.position_embeddings = nn.Embedding(args.max_seq_length, self.D)
        self.LayerNorm = LayerNorm(self.D, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.item_encoder = Encoder(args)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def encode(self, input_ids):
        B, L = input_ids.shape
        own_mask = (input_ids > 0)
        pos_ids = torch.arange(L, device=input_ids.device).unsqueeze(0)
        emb = self.item_embeddings(input_ids) + self.position_embeddings(pos_ids)
        emb = self.LayerNorm(self.dropout(emb))

        # causal mask for SASRec
        causal = torch.tril(torch.ones((L, L), device=input_ids.device)).bool()
        attn_mask = causal.unsqueeze(0).unsqueeze(0) & own_mask.unsqueeze(1).unsqueeze(2)
        ext_mask = (1.0 - attn_mask.float()) * -10000.0
        layers = self.item_encoder(emb, ext_mask, output_all_encoded_layers=True)
        return layers[-1], own_mask

    def forward(self, input_ids):
        """Pooled sequence representation (last valid token)."""
        h, own_mask = self.encode(input_ids)
        # last-valid-position pooling
        lengths = own_mask.sum(dim=1).clamp(min=1) - 1     # [B]
        idx = lengths.view(-1, 1, 1).expand(-1, 1, self.D)
        return h.gather(1, idx).squeeze(1)                  # [B, D]

    def next_item_loss(self, batch):
        """
        Standard SASRec BPR/CE loss. `batch` should provide:
            input_ids   [B, L]
            target_pos  [B, L]
            target_neg  [B, L]
        Adapt to your existing loss; this is a reference implementation.
        """
        input_ids = batch["input_ids"]
        target_pos = batch["target_pos"]
        target_neg = batch["target_neg"]

        h, own_mask = self.encode(input_ids)                # [B, L, D]
        pos_emb = self.item_embeddings(target_pos)
        neg_emb = self.item_embeddings(target_neg)

        pos_logits = (h * pos_emb).sum(-1)
        neg_logits = (h * neg_emb).sum(-1)

        valid = (target_pos > 0).float()
        loss = -torch.log(torch.sigmoid(pos_logits - neg_logits) + 1e-24)
        return (loss * valid).sum() / valid.sum().clamp(min=1.0)