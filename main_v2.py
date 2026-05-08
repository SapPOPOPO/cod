"""Entry point. Edit imports for SASRec / dataset / sim model to match your repo."""

import os, argparse, torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
# === REPLACE with your actual ports ===
from datasets import SASRecDataset as SeqRecDataset           # your dataset class
from recommender_adapter import RecommenderWrapper as Recommender
from generate_similarity import ItemCFBasedSimilarity as OfflineItemSimilarity
# =======================================

from astar_v2 import DiscretePolicyAugmenter
from ema      import EMAModel
from sim_lookup import build_sim_topk, make_sim_lookup
from train_step_v2 import train_one_step
from viz_v2 import (
    AugmenterStatsAccumulator, log_summary_stats,
    plot_op_heatmap_by_position, plot_swap_distance_by_position,
    plot_swap_distance_histogram, plot_edit_fraction_curve,
    plot_op_entropy_curve, plot_semantic_distance_hist,
    plot_view_diversity, plot_table_drift,
)
from evaluate import evaluate

def adapt_batch(batch_tuple, device):
    """SASRecDataset returns (user_id, input_ids, target_pos, target_neg, answer)."""
    batch_tuple = tuple(t.to(device) for t in batch_tuple)
    user_id, input_ids, target_pos, target_neg, answer = batch_tuple
    return {
        "user_id":    user_id,
        "input_ids":  input_ids,
        "target_pos": target_pos,
        "target_neg": target_neg,
        "answer":     answer,
    }

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_name", type=str, required=True)
    p.add_argument("--data_dir",  type=str, default="data/")
    p.add_argument("--output_dir", type=str, default="output/")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--max_seq_length", type=int, default=50)
    p.add_argument("--hidden_size", type=int, default=64)
    p.add_argument("--num_hidden_layers", type=int, default=2)
    p.add_argument("--num_attention_heads", type=int, default=2)
    p.add_argument("--hidden_dropout_prob", type=float, default=0.5)
    p.add_argument("--attention_probs_dropout_prob", type=float, default=0.5)
    p.add_argument("--initializer_range", type=float, default=0.02)
    p.add_argument("--hidden_act", type=str, default="gelu")
    p.add_argument("--no_cuda", action="store_true")

    p.add_argument("--n_sim_candidates", type=int, default=10)
    p.add_argument("--n_rand_candidates", type=int, default=20)
    p.add_argument("--gumbel_tau", type=float, default=1.0)
    p.add_argument("--target_tau", type=float, default=1.0)
    p.add_argument("--sim_top_k", type=int, default=20)

    p.add_argument("--alpha",  type=float, default=0.2)
    p.add_argument("--beta",   type=float, default=1.0)
    p.add_argument("--gamma",  type=float, default=1.0)
    p.add_argument("--eta_op", type=float, default=0.01)
    p.add_argument("--eta_tg", type=float, default=0.01)
    p.add_argument("--delta_max", type=float, default=0.5)
    p.add_argument("--contrastive_temp", type=float, default=0.5)

    p.add_argument("--rec_lr", type=float, default=1e-3)
    p.add_argument("--aug_lr", type=float, default=2e-4)
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--grad_clip", type=float, default=5.0)

    p.add_argument("--edit_target_start", type=float, default=0.1)
    p.add_argument("--edit_target_end",   type=float, default=0.3)
    p.add_argument("--edit_target_warmup_epochs", type=int, default=10)

    p.add_argument("--viz_every", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def edit_target_schedule(epoch, args):
    if epoch >= args.edit_target_warmup_epochs:
        return args.edit_target_end
    frac = epoch / max(args.edit_target_warmup_epochs, 1)
    return args.edit_target_start + frac * (args.edit_target_end - args.edit_target_start)

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "viz"), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.cuda_condition = torch.cuda.is_available() and not getattr(args, "no_cuda", False)


    from utils import get_user_seqs
    args.data_file = os.path.join(args.data_dir, f"{args.data_name}.txt")
    user_seq, _, _, args.train_matrix = get_user_seqs(args.data_file)

    # ── Data ──────────────────────────────────────────────────────────────
    from utils import get_user_seqs

    args.data_file = os.path.join(args.data_dir, f"{args.data_name}.txt")
    user_seq, max_item, valid_rating_matrix, test_rating_matrix = get_user_seqs(args.data_file)

    args.item_size = max_item + 2          # +2: pad(0) + buffer; matches original main.py
    args.mask_id = 0                       # original convention
    args.train_matrix = valid_rating_matrix  # used by evaluate.py to mask seen items

    train_set = SeqRecDataset(args, user_seq, data_type="train")
    valid_set = SeqRecDataset(args, user_seq, data_type="valid")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False)

    # ── Sim model ─────────────────────────────────────────────────────────
    sim_model = OfflineItemSimilarity(
        data_file=os.path.join(args.data_dir, f"{args.data_name}.txt"),
        similarity_path=os.path.join(args.data_dir, f"{args.data_name}_sim.pkl"),
        model_type="ItemCF_IUF",
    )

    sim_topk_idx = build_sim_topk(sim_model, args.item_size, top_k=args.sim_top_k).to(device)
    sim_lookup = make_sim_lookup(sim_topk_idx, top_k=args.sim_top_k)

    # ── Reserve MASK token ────────────────────────────────────────────────
    mask_token_id = args.item_size
    args.item_size += 1

    # ── Models ────────────────────────────────────────────────────────────
    recommender = Recommender(args).to(device)
    augmenter = DiscretePolicyAugmenter(
        args,
        recommender_item_embeddings=recommender.item_embeddings,
        mask_token_id=mask_token_id,
        n_sim_candidates=args.n_sim_candidates,
        n_rand_candidates=args.n_rand_candidates,
        gumbel_tau=args.gumbel_tau,
        target_tau=args.target_tau,
    ).to(device)
    ema_recommender = EMAModel(recommender, decay=args.ema_decay)

    rec_opt = torch.optim.Adam(recommender.parameters(), lr=args.rec_lr, betas=(0.9, 0.98))
    aug_opt = torch.optim.Adam(augmenter.parameters(),    lr=args.aug_lr, betas=(0.9, 0.98))

    stats = AugmenterStatsAccumulator(max_seq_len=args.max_seq_length)

    cfg = type("cfg", (), {})()
    cfg.alpha, cfg.beta, cfg.gamma = args.alpha, args.beta, args.gamma
    cfg.eta_op, cfg.eta_tg = args.eta_op, args.eta_tg
    cfg.delta_max = args.delta_max
    cfg.contrastive_temp = args.contrastive_temp
    cfg.grad_clip = args.grad_clip
    cfg.edit_target_schedule = lambda e, a=args: edit_target_schedule(e, a)

    best_ndcg, best_epoch = -1.0, -1
    table_drift_history = []

    n_aug = sum(p.numel() for p in augmenter.parameters())
    n_rec = sum(p.numel() for p in recommender.parameters())
    shared = sum(p.numel() for p in augmenter.parameters() if any(p is q for q in recommender.parameters()))
    print(f"augmenter params: {n_aug:,}")
    print(f"recommender params: {n_rec:,}")
    print(f"params shared (must be 0): {shared}")

    for epoch in range(args.epochs):
        cfg.current_epoch = epoch
        recommender.train(); augmenter.train()
        for batch in train_loader:
            # SASRecDataset returns tuple: (user_id, input_ids, target_pos, target_neg, answer)
            if isinstance(batch, (list, tuple)):
                user_id, input_ids, target_pos, target_neg, answer = batch[:5]
                batch = {
                    "user_id":    user_id.to(device)    if torch.is_tensor(user_id)    else user_id,
                    "input_ids":  input_ids.to(device),
                    "target_pos": target_pos.to(device),
                    "target_neg": target_neg.to(device),
                    "answer":     answer.to(device)     if torch.is_tensor(answer)     else answer,
                }
            else:
                batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

            train_one_step(
                batch=batch, augmenter=augmenter,
                recommender=recommender, ema_recommender=ema_recommender,
                sim_lookup=sim_lookup, sim_topk_idx=sim_topk_idx,
                aug_optimizer=aug_opt, rec_optimizer=rec_opt,
                stats=stats, cfg=cfg,
            )

        scores, post_fix = evaluate(recommender, valid_loader, args, mode="valid")
        ndcg = scores[3]   # NDCG@10
        hr   = scores[2]   # HR@10
        print(post_fix)

        if ndcg > best_ndcg:
            best_ndcg, best_epoch = ndcg, epoch
            torch.save({
                "recommender": recommender.state_dict(),
                "augmenter":   augmenter.state_dict(),
                "ema_recommender": ema_recommender.module.state_dict(),
                "epoch": epoch,
            }, os.path.join(args.output_dir, "best.pt"))

        with torch.no_grad():
            drift = (recommender.item_embeddings.weight
                     - ema_recommender.module.item_embeddings.weight).norm().item()
        table_drift_history.append(drift)

        print(f"epoch {epoch:3d}  ndcg={ndcg:.4f}  hr={hr:.4f}  drift={drift:.3f}")
        print(log_summary_stats(stats))

        if epoch % args.viz_every == 0:
            d = os.path.join(args.output_dir, "viz", f"e{epoch:03d}")
            os.makedirs(d, exist_ok=True)
            plot_op_heatmap_by_position(stats).savefig(f"{d}/op_heatmap.png");           
            plot_swap_distance_by_position(stats).savefig(f"{d}/swap_dist_heatmap.png")
            plot_swap_distance_histogram(stats).savefig(f"{d}/swap_dist_hist.png")
            plot_edit_fraction_curve(stats, target=cfg.edit_target_schedule(epoch))\
                .savefig(f"{d}/edit_frac.png")
            plot_op_entropy_curve(stats).savefig(f"{d}/entropy.png")
            plot_semantic_distance_hist(stats).savefig(f"{d}/sem_dist.png")
            plot_view_diversity(stats).savefig(f"{d}/view_div.png")
            plot_table_drift(table_drift_history).savefig(f"{d}/table_drift.png")
            plt.close("all")
        stats.reset()

    print(f"best valid NDCG={best_ndcg:.4f} @ epoch {best_epoch}")


if __name__ == "__main__":
    main()