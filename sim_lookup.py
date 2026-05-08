"""Sparse rank-based similarity lookup keyed by precomputed top-k neighbors."""

import torch


"""Sparse rank-based similarity lookup keyed by precomputed top-k neighbors."""

import torch


def build_sim_topk(item_similarity_model, num_items: int, top_k: int = 20) -> torch.Tensor:
    """Returns LongTensor [num_items, top_k] of top-k similar item ids per item.

    Robust to:
      - string-keyed similarity dicts (ItemCFBasedSimilarity uses str ids)
      - items missing from the similarity dict (falls back to self-id padding)
      - tuples or scalars returned by most_similar()
    """
    out = torch.zeros(num_items, top_k, dtype=torch.long)

    for item_id in range(1, num_items):
        ids = []
        # Try string key first (CoSeRec-style sim dict), then int key as fallback
        for key in (str(item_id), item_id):
            try:
                sims = item_similarity_model.most_similar(key, top_k=top_k)
                if sims:
                    break
            except (KeyError, AttributeError):
                sims = []
        else:
            sims = []

        for s in list(sims)[:top_k]:
            if isinstance(s, (list, tuple)):
                s = s[0]
            try:
                ids.append(int(s))
            except (TypeError, ValueError):
                continue

        if len(ids) < top_k:
            ids += [item_id] * (top_k - len(ids))
        out[item_id] = torch.tensor(ids[:top_k], dtype=torch.long)

    return out


def make_sim_lookup(sim_topk_idx: torch.Tensor, top_k: int = 20):
    """Returns callable (a, b) -> sim ∈ [0, 1] (rank-based, with floor)."""
    V, K = sim_topk_idx.shape
    ranks = torch.arange(K, device=sim_topk_idx.device).float()
    rank_scores = 1.0 - 0.9 * (ranks / max(K - 1, 1))

    def lookup(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        neighbors = sim_topk_idx[a]                                 # [B, L, K]
        match = (neighbors == b.unsqueeze(-1)).float()              # [B, L, K]
        scores = (match * rank_scores.to(a.device)).sum(dim=-1)     # [B, L]
        scores = torch.where(a == b, torch.ones_like(scores), scores)
        return scores.clamp(min=0.05)
    return lookup