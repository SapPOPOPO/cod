"""Validation/test eval for the v2 recommender. Computes HR@k and NDCG@k."""

import numpy as np
import torch
from tqdm import tqdm


def get_metric(pred_list, topk=10):
    NDCG = 0.0
    HIT  = 0.0
    for rank in pred_list:
        if rank < topk:
            NDCG += 1.0 / np.log2(rank + 2)
            HIT  += 1.0
    return HIT / len(pred_list), NDCG / len(pred_list)


@torch.no_grad()
def evaluate(model, dataloader, args, mode="valid"):
    """
    `model.forward(input_ids) -> [B, D]` (last-token pooled).
    `model.item_embeddings.weight` is the full item table [V, D].
    Each batch yields (user_id, input_ids, target_pos, target_neg, answer).
    """
    model.eval()
    device = next(model.parameters()).device
    pred_list = []

    for batch in tqdm(dataloader, desc=f"Eval [{mode}] EP:{getattr(args, 'current_epoch', 0)}"):
        if isinstance(batch, (list, tuple)):
            user_id, input_ids, target_pos, target_neg, answer = batch[:5]
        else:
            input_ids, answer = batch["input_ids"], batch["answer"]

        input_ids = input_ids.to(device)
        answer    = answer.to(device)                 # [B] or [B, 1]
        if answer.dim() == 2:
            answer = answer.squeeze(-1)

        seq_repr = model(input_ids)                   # [B, D]
        all_item_emb = model.item_embeddings.weight   # [V, D]
        scores = seq_repr @ all_item_emb.t()          # [B, V]

        # mask out items already in the input
        scores.scatter_(1, input_ids, -1e9)

        # rank of the ground-truth answer
        gt_score = scores.gather(1, answer.view(-1, 1))         # [B, 1]
        rank = (scores > gt_score).sum(dim=1)                   # 0 = top
        pred_list.extend(rank.cpu().tolist())

    HR_5,  NDCG_5  = get_metric(pred_list, topk=5)
    HR_10, NDCG_10 = get_metric(pred_list, topk=10)
    HR_20, NDCG_20 = get_metric(pred_list, topk=20)

    post_fix = {
        "HR@5":  f"{HR_5:.4f}",  "NDCG@5":  f"{NDCG_5:.4f}",
        "HR@10": f"{HR_10:.4f}", "NDCG@10": f"{NDCG_10:.4f}",
        "HR@20": f"{HR_20:.4f}", "NDCG@20": f"{NDCG_20:.4f}",
    }
    scores = [HR_5, NDCG_5, HR_10, NDCG_10, HR_20, NDCG_20]
    return scores, post_fix