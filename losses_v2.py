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


def budget_loss(edit_mask: torch.Tensor, own_mask: torch.Tensor, target: float) -> torch.Tensor:
    valid = own_mask.float()
    n_valid = valid.sum().clamp(min=1.0)
    edit_frac = (edit_mask.float() * valid).sum() / n_valid
    return (edit_frac - target).pow(2)


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