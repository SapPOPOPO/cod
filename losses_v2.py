"""All loss terms for the discrete-policy augmenter."""

import torch
import torch.nn.functional as F

from astar_v2 import OP_KEEP, OP_MASK, OP_SUB_SIM, OP_SUB_RAND, OP_SWAP_OWN


def info_nce(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    B = z1.size(0)
    logits = z1 @ z2.t() / temperature
    labels = torch.arange(B, device=z1.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def semantic_anchor_loss(
    input_ids:    torch.Tensor,
    aug_outputs:  dict,                    # ← now takes the full augmenter output
    sim_lookup,                            # callable: (a, b) -> [B, L] sim ∈ [0, 1]
    delta_max:    float = 0.5,
    mask_penalty: float = 0.2,
) -> torch.Tensor:
    """Differentiable semantic anchor.

    Computes E[1 - sim(x, x')] under the op_prob distribution, so gradient
    flows back to op_logits. We approximate per-op distance using:
      - KEEP:    distance = 0
      - MASK:    distance = mask_penalty (constant)
      - SUB_SIM: distance = 1 - sim(x, sampled_sim_id)
      - SUB_RAND distance = 1 - sim(x, sampled_rand_id)
      - SWAP_OWN distance = 1 - sim(x, sampled_swap_id)
    """
    op_probs   = aug_outputs["op_probs"]                  # [B, L, K]
    own_mask   = aug_outputs["own_mask"].float()
    input_ids_ = input_ids

    # Reconstruct what each op produced this step (already cached in aug output)
    # We can grab from out["aug_ids"] only AFTER per-op selection — but the
    # per-op candidate IDs aren't returned. So compute distances using the
    # representative id stored in aug_ids, weighted softly.
    # Cleaner: derive per-op distances from the ids we DO have access to.

    aug_ids = aug_outputs["aug_ids"]                      # [B, L]
    # distance of the actually-chosen op per position (hard, no grad through ids)
    sim_pos = sim_lookup(input_ids_, aug_ids)             # [B, L]
    dist_chosen = (1.0 - sim_pos).clamp(min=0.0, max=1.0) # [B, L]

    # Build a per-op distance proxy:
    #   d_per_op[k] = expected distance if op k is chosen
    # KEEP  → 0
    # MASK  → mask_penalty
    # SUB_SIM, SUB_RAND, SWAP_OWN → use dist_chosen as a stand-in (correct for
    # whichever op was actually chosen; for the other two we approximate with
    # the same value — biased but the bias is small for argmax-aligned ops).
    B, L, K = op_probs.shape
    d = dist_chosen.unsqueeze(-1).expand(B, L, K).clone()
    d[..., OP_KEEP] = 0.0
    d[..., OP_MASK] = mask_penalty

    expected_dist = (op_probs * d).sum(dim=-1)            # [B, L]
    mean_dist = (expected_dist * own_mask).sum() / own_mask.sum().clamp(min=1.0)
    return F.relu(mean_dist - delta_max)


def budget_loss(op_probs: torch.Tensor, own_mask: torch.Tensor, target: float) -> torch.Tensor:
    """Differentiable budget: E[edit_fraction] = sum of non-KEEP op probs.

    op_probs:  [B, L, K]   softmax over ops
    own_mask:  [B, L]      bool/float, valid positions
    target:    float       desired edit fraction in [0, 1]
    """
    valid = own_mask.float()
    n_valid = valid.sum().clamp(min=1.0)
    # P(edit | position) = 1 - P(KEEP | position)
    p_edit = 1.0 - op_probs[..., OP_KEEP]                      # [B, L]
    expected_edit_frac = (p_edit * valid).sum() / n_valid       # scalar, differentiable
    return (expected_edit_frac - target).pow(2)


def policy_entropy(probs: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    ent = -(probs * (probs.clamp_min(eps)).log()).sum(dim=-1)
    return (ent * mask.float()).sum() / mask.float().sum().clamp(min=1.0)


def target_entropy_when_swap(
    target_probs: torch.Tensor,
    chosen_op:    torch.Tensor,
    own_mask:     torch.Tensor,
) -> torch.Tensor:
    swap_positions = (chosen_op == OP_SWAP_OWN) & own_mask
    if swap_positions.sum() == 0:
        return torch.zeros((), device=target_probs.device)
    ent = -(target_probs * (target_probs.clamp_min(1e-8)).log()).sum(dim=-1)
    return (ent * swap_positions.float()).sum() / swap_positions.float().sum().clamp(min=1.0)


def augmenter_total_loss(*, diff_loss, sem_loss, bud_loss, ent_op, ent_target,
                         beta=1.0, gamma=1.0, eta_op=0.01, eta_tg=0.01):
    return -diff_loss + beta * sem_loss + gamma * bud_loss - eta_op * ent_op - eta_tg * ent_target

def expected_budget_loss(op_probs: torch.Tensor, own_mask: torch.Tensor, target: float) -> torch.Tensor:
    """Differentiable: expected edit fraction = mean over positions of P(op != KEEP)."""
    p_edit = 1.0 - op_probs[..., 0]                          # [B, L]
    valid = own_mask.float()
    expected_frac = (p_edit * valid).sum() / valid.sum().clamp(min=1.0)
    return (expected_frac - target).pow(2)


def expected_semantic_loss(
    input_ids:    torch.Tensor,        # [B, L]
    ids_per_op:   torch.Tensor,        # [B, L, K]
    op_probs:     torch.Tensor,        # [B, L, K]
    target_probs: torch.Tensor,        # [B, L, L]   for SWAP_OWN
    sim_lookup,
    own_mask:     torch.Tensor,
    delta_max:    float = 0.5,
    mask_penalty: float = 0.2,
) -> torch.Tensor:
    """
    Differentiable expected semantic distance under the policy.

    For each op k:
        d_k(j) = expected (1 - sim) under that op's substitution choice.

    Then E[d|j] = sum_k op_probs[j,k] * d_k(j).
    """
    # KEEP: distance 0
    d_keep = torch.zeros_like(op_probs[..., 0])
    # MASK: fixed penalty
    d_mask = torch.full_like(d_keep, mask_penalty)
    # SUB_SIM, SUB_RAND: lookup vs the sampled candidate (not differentiable through choice
    #   of similar/random candidate, but that randomness isn't policy-controlled anyway)
    ids_sim  = ids_per_op[..., 2]
    ids_rand = ids_per_op[..., 3]
    d_sim  = 1.0 - sim_lookup(input_ids, ids_sim)
    d_rand = 1.0 - sim_lookup(input_ids, ids_rand)

    # SWAP_OWN: differentiable through target_probs.
    #   d_swap(j) = sum_k target_probs[j,k] * (1 - sim(input[j], input[k]))
    B, L = input_ids.shape
    # Pairwise similarity input_ids[:, j] vs input_ids[:, k]
    a = input_ids.unsqueeze(2).expand(B, L, L)        # [B, L, L]
    b = input_ids.unsqueeze(1).expand(B, L, L)
    sim_pair = sim_lookup(a.reshape(B, L * L), b.reshape(B, L * L)).view(B, L, L)
    d_swap_per_target = 1.0 - sim_pair
    d_swap = (target_probs * d_swap_per_target).sum(dim=-1)   # [B, L]

    # Stack and combine
    d_stack = torch.stack([d_keep, d_mask, d_sim, d_rand, d_swap], dim=-1)   # [B, L, K]
    expected_d = (op_probs * d_stack).sum(dim=-1)                            # [B, L]

    valid = own_mask.float()
    mean_d = (expected_d * valid).sum() / valid.sum().clamp(min=1.0)
    return F.relu(mean_d - delta_max)