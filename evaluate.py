"""Evaluation function distilled from trainers.py iteration() else-branch.

Drop into main_v2.py (replace the placeholder evaluate()) and import:
    from utils import recall_at_k, ndcg_k, get_metric
"""

import numpy as np
import torch
from tqdm import tqdm
from utils import recall_at_k, ndcg_k, get_metric


@torch.no_grad()
def evaluate(model, dataloader, args, epoch=0, full_sort=True, mode="valid"):
    """
    Returns:
        scores: list of metrics
            full_sort=True  → [HIT@5, NDCG@5, HIT@10, NDCG@10, HIT@20, NDCG@20]
            full_sort=False → [HIT@1, NDCG@1, HIT@5, NDCG@5, HIT@10, NDCG@10, MRR]
        post_fix: dict of formatted metrics
    """
    device = next(model.parameters()).device
    model.eval()
    pred_list = None

    rec_data_iter = tqdm(
        enumerate(dataloader),
        desc=f"Eval [{mode}] EP:{epoch}",
        total=len(dataloader),
        bar_format="{l_bar}{r_bar}",
    )

    if full_sort:
        answer_list = None
        for i, batch in rec_data_iter:
            batch = tuple(t.to(device) for t in batch)
            user_ids, input_ids, target_pos, target_neg, answers = batch

            # Encode sequence and take last position
            recommend_output = model.transformer_encoder(input_ids)
            recommend_output = recommend_output[:, -1, :]                  # [B, D]

            # Score against full item table
            test_item_emb = model.item_embeddings.weight                   # [V, D]
            rating_pred = torch.matmul(recommend_output, test_item_emb.t())
            rating_pred = rating_pred.cpu().data.numpy().copy()

            # Mask out items the user has already interacted with in train set
            batch_user_index = user_ids.cpu().numpy()
            rating_pred[args.train_matrix[batch_user_index].toarray() > 0] = 0

            # Top-20 prediction ids per user
            ind = np.argpartition(rating_pred, -20)[:, -20:]
            arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
            arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
            batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

            if i == 0:
                pred_list = batch_pred_list
                answer_list = answers.cpu().data.numpy()
            else:
                pred_list = np.append(pred_list, batch_pred_list, axis=0)
                answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)

        recall, ndcg = [], []
        for k in [5, 10, 15, 20]:
            recall.append(recall_at_k(answer_list, pred_list, k))
            ndcg.append(ndcg_k(answer_list, pred_list, k))

        post_fix = {
            "Epoch":   epoch,
            "HIT@5":   f"{recall[0]:.4f}", "NDCG@5":  f"{ndcg[0]:.4f}",
            "HIT@10":  f"{recall[1]:.4f}", "NDCG@10": f"{ndcg[1]:.4f}",
            "HIT@20":  f"{recall[3]:.4f}", "NDCG@20": f"{ndcg[3]:.4f}",
        }
        scores = [recall[0], ndcg[0], recall[1], ndcg[1], recall[3], ndcg[3]]
        return scores, post_fix

    else:
        # Sampled negatives evaluation
        for i, batch in rec_data_iter:
            batch = tuple(t.to(device) for t in batch)
            user_ids, input_ids, target_pos, target_neg, answers, sample_negs = batch

            recommend_output = model.finetune(input_ids)
            test_neg_items = torch.cat((answers, sample_negs), -1)
            recommend_output = recommend_output[:, -1, :]

            test_item_emb = model.item_embeddings(test_neg_items)
            test_logits = torch.bmm(test_item_emb, recommend_output.unsqueeze(-1)).squeeze(-1)
            test_logits = test_logits.cpu().detach().numpy().copy()

            if i == 0:
                pred_list = test_logits
            else:
                pred_list = np.append(pred_list, test_logits, axis=0)

        ranks = (-pred_list).argsort().argsort()[:, 0]
        HIT_1,  NDCG_1,  MRR = get_metric(ranks, 1)
        HIT_5,  NDCG_5,  _   = get_metric(ranks, 5)
        HIT_10, NDCG_10, _   = get_metric(ranks, 10)
        post_fix = {
            "Epoch":  epoch,
            "HIT@1":  f"{HIT_1:.4f}",  "NDCG@1":  f"{NDCG_1:.4f}",
            "HIT@5":  f"{HIT_5:.4f}",  "NDCG@5":  f"{NDCG_5:.4f}",
            "HIT@10": f"{HIT_10:.4f}", "NDCG@10": f"{NDCG_10:.4f}",
            "MRR":    f"{MRR:.4f}",
        }
        scores = [HIT_1, NDCG_1, HIT_5, NDCG_5, HIT_10, NDCG_10, MRR]
        return scores, post_fix