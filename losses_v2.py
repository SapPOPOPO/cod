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
    aug_ids:      torch.Tensor,
    chosen_op:    torch.Tensor,
    sim_lookup,                     # callable: (a, b) -> [B, L] sim ∈ [0, 1]
    own_mask:     torch.Tensor,
    delta_max:    float = 0.5,
    mask_penalty: float = 0.2,
) -> torch.Tensor:
    valid = own_mask.float()
    sim_pos = sim_lookup(input_ids, aug_ids)
    dist = 1.0 - sim_pos
    dist = torch.where(chosen_op == OP_KEEP, torch.zeros_like(dist), dist)
    dist = torch.where(chosen_op == OP_MASK, torch.full_like(dist, mask_penalty), dist)
    mean_dist = (dist * valid).sum() / valid.sum().clamp(min=1.0)
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