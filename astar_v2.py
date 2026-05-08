"""
astar_v2.py — Discrete-policy augmenter with targeted SWAP_OWN.

Per position j, two-stage decision:
    Stage 1 (WHAT):  op_j      ~ softmax(op_logits[j])         over K=5 ops
    Stage 2 (WHERE): target_j  ~ softmax(swap_scores[j])       over other positions
                     (used only when op_j == SWAP_OWN)

Operations:
    0: KEEP        x'_j = x_j
    1: MASK        x'_j = [MASK]
    2: SUB_SIM     x'_j = uniform sample from top-k similar items of x_j
    3: SUB_RAND    x'_j = uniform random item from vocab
    4: SWAP_OWN    x'_j = x_{target_j}     (target chosen by pointer)

Output: tensor of REAL item IDs (or MASK) — fully on-manifold.

Key changes vs original ASTAR:
    - Item embeddings SHARED with B (stop-grad). No Frankenstein mixing.
    - λ removed; "don't augment" handled by budget loss.
    - Two views = two stochastic samples of same policy.
    - Constant Gumbel-Softmax τ.
    - Pointer-based targeting only for SWAP_OWN.
"""

from typing import Tuple, Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import Encoder, LayerNorm

OP_KEEP, OP_MASK, OP_SUB_SIM, OP_SUB_RAND, OP_SWAP_OWN = 0, 1, 2, 3, 4
NUM_OPS = 5
OP_NAMES = ["KEEP", "MASK", "SUB_SIM", "SUB_RAND", "SWAP_OWN"]


class DiscretePolicyAugmenter(nn.Module):
    def __init__(
        self,
        args,
        recommender_item_embeddings: nn.Embedding,
        mask_token_id: int,
        n_sim_candidates: int = 10,
        n_rand_candidates: int = 20,
        gumbel_tau: float = 1.0,
        target_tau: float = 1.0,
    ):
        super().__init__()
        self.args = args
        self.D = args.hidden_size
        self.L = args.max_seq_length
        self.num_items = args.item_size

        self._shared_item_table = recommender_item_embeddings
        self.mask_token_id = mask_token_id
        self.n_sim_candidates = n_sim_candidates
        self.n_rand_candidates = n_rand_candidates
        self.gumbel_tau = gumbel_tau
        self.target_tau = target_tau

        self.position_embeddings = nn.Embedding(args.max_seq_length, self.D)
        self.LayerNorm = LayerNorm(self.D, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.item_encoder = Encoder(args)

        self.op_head = nn.Linear(self.D, NUM_OPS)
        self.swap_query_proj = nn.Linear(self.D, self.D)
        self.swap_key_proj = nn.Linear(self.D, self.D)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _embed_with_shared_table(self, ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(ids, self._shared_item_table.weight.detach(), padding_idx=0)

    def encode(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L = input_ids.shape
        own_mask = (input_ids > 0)

        item_emb = self._embed_with_shared_table(input_ids)
        pos_ids = torch.arange(L, device=input_ids.device).unsqueeze(0)
        pos_emb = self.position_embeddings(pos_ids)
        h = self.LayerNorm(self.dropout(item_emb + pos_emb))

        ext_mask = own_mask.unsqueeze(1).unsqueeze(2).float()
        ext_mask = (1.0 - ext_mask) * -10000.0
        layers = self.item_encoder(h, ext_mask, output_all_encoded_layers=True)
        return layers[-1], own_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        sim_topk_idx: Optional[torch.Tensor] = None,
        sample: bool = True,
    ) -> Dict[str, torch.Tensor]:
        B, L = input_ids.shape
        h_own, own_mask = self.encode(input_ids)

        # ── Stage 1: op distribution ────────────────────────────────────────
        op_logits = self.op_head(h_own)
        op_probs = F.softmax(op_logits, dim=-1)
        if sample:
            op_onehot = F.gumbel_softmax(op_logits, tau=self.gumbel_tau, hard=True, dim=-1)
        else:
            op_onehot = F.one_hot(op_logits.argmax(-1), NUM_OPS).float()

        # ── Stage 2: pointer over positions for SWAP_OWN ────────────────────
        Q = self.swap_query_proj(h_own)
        K_ = self.swap_key_proj(h_own)
        swap_scores = torch.bmm(Q, K_.transpose(1, 2)) / (self.D ** 0.5)
        eye = torch.eye(L, device=input_ids.device, dtype=torch.bool).unsqueeze(0)
        valid_target = own_mask.unsqueeze(1) & ~eye
        swap_scores = swap_scores.masked_fill(~valid_target, -1e9)
        target_probs = F.softmax(swap_scores / self.target_tau, dim=-1)
        if sample:
            target_onehot = F.gumbel_softmax(
                swap_scores / self.target_tau, tau=1.0, hard=True, dim=-1
            )
        else:
            target_onehot = F.one_hot(swap_scores.argmax(-1), L).float()

        # ── Build candidate ids per op ──────────────────────────────────────
        ids_keep = input_ids
        ids_mask = torch.full_like(input_ids, self.mask_token_id)

        if sim_topk_idx is not None and self.n_sim_candidates > 0:
            top_k = sim_topk_idx.size(1)
            k_pick = min(self.n_sim_candidates, top_k)
            cand = sim_topk_idx[input_ids.clamp(min=0)][..., :k_pick]   # [B, L, k_pick]
            pick_idx = torch.randint(0, k_pick, (B, L, 1), device=input_ids.device)
            ids_sim = cand.gather(-1, pick_idx).squeeze(-1)
        else:
            ids_sim = input_ids

        ids_rand = torch.randint(1, self.num_items, (B, L), device=input_ids.device)

        target_idx = target_onehot.argmax(-1)
        ids_swap = input_ids.gather(1, target_idx)

        all_ids = torch.stack([ids_keep, ids_mask, ids_sim, ids_rand, ids_swap], dim=-1)
        chosen_op = op_onehot.argmax(-1)
        aug_ids = all_ids.gather(-1, chosen_op.unsqueeze(-1)).squeeze(-1)
        aug_ids = aug_ids * own_mask.long()

        edit_mask = (chosen_op != OP_KEEP) & own_mask

        return {
            "aug_ids":       aug_ids,
            "op_probs":      op_probs,
            "op_onehot":     op_onehot,
            "op_logits":     op_logits,
            "target_probs":  target_probs,
            "target_onehot": target_onehot,
            "chosen_op":     chosen_op,
            "swap_target":   target_idx,
            "edit_mask":     edit_mask,
            "own_mask":      own_mask,
            "h_own":         h_own,
        }