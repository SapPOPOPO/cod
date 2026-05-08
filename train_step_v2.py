"""Per-step training glue: A-update, B-update, EMA-update."""

import torch

from losses_v2 import (
    info_nce, semantic_anchor_loss, budget_loss,
    policy_entropy, target_entropy_when_swap, augmenter_total_loss,
)


def train_one_step(*, batch, augmenter, recommender, ema_recommender,
                   sim_lookup, sim_topk_idx,
                   aug_optimizer, rec_optimizer,
                   stats, cfg):
    input_ids = batch["input_ids"]

    # ── Phase 1: update A ────────────────────────────────────────────────
    aug_optimizer.zero_grad()
    out_v1 = augmenter(input_ids, sim_topk_idx=sim_topk_idx)
    out_v2 = augmenter(input_ids, sim_topk_idx=sim_topk_idx)

    # Use STE embeddings so gradient flows from diff_loss back to A
    z1 = ema_recommender.module.forward_from_embeddings(out_v1["aug_emb"], out_v1["aug_ids"])
    z2 = ema_recommender.module.forward_from_embeddings(out_v2["aug_emb"], out_v2["aug_ids"])
    diff_loss = info_nce(z1, z2, temperature=cfg.contrastive_temp)

    sem1 = semantic_anchor_loss(input_ids, out_v1["aug_ids"], out_v1["chosen_op"],
                                sim_lookup, out_v1["own_mask"], delta_max=cfg.delta_max)
    sem2 = semantic_anchor_loss(input_ids, out_v2["aug_ids"], out_v2["chosen_op"],
                                sim_lookup, out_v2["own_mask"], delta_max=cfg.delta_max)
    sem_loss = 0.5 * (sem1 + sem2)

    target_edit = cfg.edit_target_schedule(cfg.current_epoch)
    bud_loss = 0.5 * (
        budget_loss(out_v1["op_probs"], out_v1["own_mask"], target_edit)
        + budget_loss(out_v2["op_probs"], out_v2["own_mask"], target_edit)
    )

    ent_op = 0.5 * (
        policy_entropy(out_v1["op_probs"], out_v1["own_mask"])
        + policy_entropy(out_v2["op_probs"], out_v2["own_mask"])
    )
    ent_tg = 0.5 * (
        target_entropy_when_swap(out_v1["target_probs"], out_v1["chosen_op"], out_v1["own_mask"])
        + target_entropy_when_swap(out_v2["target_probs"], out_v2["chosen_op"], out_v2["own_mask"])
    )

    L_A = augmenter_total_loss(
        diff_loss=diff_loss, sem_loss=sem_loss, bud_loss=bud_loss,
        ent_op=ent_op, ent_target=ent_tg,
        beta=cfg.beta, gamma=cfg.gamma, eta_op=cfg.eta_op, eta_tg=cfg.eta_tg,
    )
    L_A.backward()
    torch.nn.utils.clip_grad_norm_(augmenter.parameters(), cfg.grad_clip)
    aug_optimizer.step()

    # ── Phase 2: update B ────────────────────────────────────────────────
    rec_optimizer.zero_grad()
    with torch.no_grad():
        out_v1_d = augmenter(input_ids, sim_topk_idx=sim_topk_idx)
        out_v2_d = augmenter(input_ids, sim_topk_idx=sim_topk_idx)

    z1_b = recommender(out_v1_d["aug_ids"])
    z2_b = recommender(out_v2_d["aug_ids"])
    L_contrast = info_nce(z1_b, z2_b, temperature=cfg.contrastive_temp)
    L_main = recommender.next_item_loss(batch)
    L_B = L_main + cfg.alpha * L_contrast
    L_B.backward()
    torch.nn.utils.clip_grad_norm_(recommender.parameters(), cfg.grad_clip)
    rec_optimizer.step()

    # ── Phase 3: EMA ─────────────────────────────────────────────────────
    ema_recommender.update(recommender)

    stats.update(out_v1, sim_lookup=sim_lookup, input_ids=input_ids, out_view2=out_v2)

    return {
        "L_A": L_A.item(), "L_B": L_B.item(),
        "diff": diff_loss.item(), "sem": sem_loss.item(), "bud": bud_loss.item(),
        "ent_op": ent_op.item(),
        "ent_tg": ent_tg.item() if ent_tg.numel() > 0 else 0.0,
        "L_main": L_main.item(), "L_contrast": L_contrast.item(),
        "edit_target": target_edit,
    }