"""Rich visualization for the discrete-policy augmenter."""

from typing import Dict, List, Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from astar_v2 import OP_NAMES, OP_KEEP, OP_MASK, OP_SUB_SIM, OP_SUB_RAND, OP_SWAP_OWN, NUM_OPS


class AugmenterStatsAccumulator:
    def __init__(self, max_seq_len: int):
        self.L = max_seq_len
        self.reset()

    def reset(self):
        self.op_count_by_pos = np.zeros((self.L, NUM_OPS), dtype=np.float64)
        self.valid_count_by_pos = np.zeros(self.L, dtype=np.float64)
        self.swap_dist_by_pos = np.zeros((self.L, 2 * self.L + 1), dtype=np.float64)
        self.semantic_dist_by_op = {k: [] for k in range(NUM_OPS)}
        self.edit_fraction_per_step = []
        self.op_entropy_per_step = []
        self.target_entropy_per_step = []
        self.view_disagreement_per_step = []

    @torch.no_grad()
    def update(self, out, sim_lookup=None, input_ids=None, out_view2=None):
        op = out["chosen_op"].cpu().numpy()
        own = out["own_mask"].cpu().numpy()
        tgt = out["swap_target"].cpu().numpy()
        edit = out["edit_mask"].cpu().numpy()

        for k in range(NUM_OPS):
            mask = (op == k) & own
            self.op_count_by_pos[:, k] += mask.sum(axis=0)
        self.valid_count_by_pos += own.sum(axis=0)

        swap_mask = (op == OP_SWAP_OWN) & own
        if swap_mask.sum() > 0:
            B, L = op.shape
            j_idx = np.broadcast_to(np.arange(L)[None, :], (B, L))
            distances = tgt - j_idx
            for j in range(L):
                ds = distances[swap_mask & (j_idx == j)]
                if ds.size > 0:
                    bins = ds + L
                    np.add.at(self.swap_dist_by_pos[j], bins, 1)

        if sim_lookup is not None and input_ids is not None:
            with torch.no_grad():
                sim_pos = sim_lookup(input_ids, out["aug_ids"]).cpu().numpy()
            for k in range(NUM_OPS):
                mask = (op == k) & own
                if mask.sum() > 0:
                    self.semantic_dist_by_op[k].extend((1.0 - sim_pos[mask]).tolist())

        own_sum = own.sum()
        if own_sum > 0:
            self.edit_fraction_per_step.append(float(edit.sum()) / float(own_sum))

        op_probs = out["op_probs"].cpu().numpy()
        ent = -np.sum(op_probs * np.log(op_probs + 1e-8), axis=-1)
        self.op_entropy_per_step.append(float((ent * own).sum() / max(own_sum, 1)))

        if swap_mask.sum() > 0:
            tp = out["target_probs"].cpu().numpy()
            tent = -np.sum(tp * np.log(tp + 1e-8), axis=-1)
            self.target_entropy_per_step.append(
                float((tent * swap_mask).sum() / swap_mask.sum())
            )

        if out_view2 is not None:
            op2 = out_view2["chosen_op"].cpu().numpy()
            disagree = (op != op2) & own
            self.view_disagreement_per_step.append(float(disagree.sum()) / max(own_sum, 1))


def plot_op_heatmap_by_position(stats):
    counts = stats.op_count_by_pos
    valid = stats.valid_count_by_pos[:, None].clip(min=1)
    probs = counts / valid
    fig, ax = plt.subplots(figsize=(6, 8))
    im = ax.imshow(probs, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(NUM_OPS)); ax.set_xticklabels(OP_NAMES, rotation=30)
    ax.set_ylabel("Sequence position (0=oldest)")
    ax.set_title("Operation probability by position")
    fig.colorbar(im, ax=ax, label="P(op | position)")
    return fig


def plot_swap_distance_histogram(stats):
    total = stats.swap_dist_by_pos.sum(axis=0)
    L = stats.L
    xs = np.arange(-L, L + 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(xs, total, width=1.0)
    ax.set_xlabel("target - source position")
    ax.set_ylabel("# swap events")
    ax.set_title("SWAP_OWN: signed distance distribution")
    ax.axvline(0, color="red", linestyle="--", alpha=0.5)
    return fig


def plot_swap_distance_by_position(stats):
    grid = stats.swap_dist_by_pos
    row_sums = grid.sum(axis=1, keepdims=True).clip(min=1)
    grid_norm = grid / row_sums
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(grid_norm, aspect="auto", cmap="magma", origin="lower",
                   extent=[-stats.L, stats.L, 0, stats.L])
    ax.set_xlabel("target - source")
    ax.set_ylabel("source position")
    ax.set_title("SWAP_OWN target-distance distribution per source position")
    ax.axvline(0, color="cyan", linestyle="--", alpha=0.5)
    fig.colorbar(im, ax=ax, label="P(distance | source)")
    return fig


def plot_edit_fraction_curve(stats, target=None):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(stats.edit_fraction_per_step, label="actual edit fraction")
    if target is not None:
        ax.axhline(target, color="red", linestyle="--", label=f"target={target:.2f}")
    ax.set_xlabel("step"); ax.set_ylabel("edit fraction")
    ax.set_title("Realized edit fraction during epoch"); ax.legend()
    return fig


def plot_op_entropy_curve(stats):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(stats.op_entropy_per_step, label="op entropy")
    if stats.target_entropy_per_step:
        ax.plot(stats.target_entropy_per_step, label="swap-target entropy", alpha=0.7)
    ax.set_xlabel("step"); ax.set_ylabel("nats")
    ax.set_title("Policy entropy (exploration health)"); ax.legend()
    return fig


def plot_semantic_distance_hist(stats):
    fig, axes = plt.subplots(1, NUM_OPS, figsize=(4 * NUM_OPS, 3), sharey=True)
    for k, ax in enumerate(axes):
        data = stats.semantic_dist_by_op[k]
        if data:
            ax.hist(data, bins=30, range=(0, 1))
        ax.set_title(f"{OP_NAMES[k]}\n(n={len(data)})")
        ax.set_xlabel("1 - sim(x, x')")
    axes[0].set_ylabel("count")
    fig.suptitle("Semantic-distance distribution per operation")
    fig.tight_layout()
    return fig


def plot_view_diversity(stats):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(stats.view_disagreement_per_step)
    ax.set_xlabel("step"); ax.set_ylabel("frac of positions where view1.op != view2.op")
    ax.set_title("View diversity (collapse warning if → 0)")
    ax.set_ylim(0, 1)
    return fig


def plot_pointer_attention_example(out, example_idx=0):
    tp = out["target_probs"][example_idx].cpu().numpy()
    own = out["own_mask"][example_idx].cpu().numpy()
    seq_end = int(own.sum())
    tp = tp[:seq_end, :seq_end]
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(tp, cmap="viridis", aspect="auto")
    ax.set_xlabel("target position"); ax.set_ylabel("source position")
    ax.set_title(f"Pointer attention (sample {example_idx}, length {seq_end})")
    fig.colorbar(im, ax=ax)
    return fig


def plot_table_drift(history):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(history, marker="o")
    ax.set_xlabel("epoch"); ax.set_ylabel("||B - EMA_B||_F")
    ax.set_title("Recommender table drift vs EMA")
    return fig


def log_summary_stats(stats) -> str:
    counts = stats.op_count_by_pos.sum(axis=0)
    valid = stats.valid_count_by_pos.sum().clip(min=1)
    op_freq = counts / valid
    parts = [f"{OP_NAMES[k]}={op_freq[k]:.3f}" for k in range(NUM_OPS)]
    avg_edit = np.mean(stats.edit_fraction_per_step) if stats.edit_fraction_per_step else 0
    avg_op_ent = np.mean(stats.op_entropy_per_step) if stats.op_entropy_per_step else 0
    avg_tgt_ent = np.mean(stats.target_entropy_per_step) if stats.target_entropy_per_step else 0
    avg_disag = np.mean(stats.view_disagreement_per_step) if stats.view_disagreement_per_step else 0
    return ("[augmenter] " + " ".join(parts)
            + f"  edit_frac={avg_edit:.3f}"
            + f"  H(op)={avg_op_ent:.3f}"
            + f"  H(tgt|swap)={avg_tgt_ent:.3f}"
            + f"  view_disagree={avg_disag:.3f}")