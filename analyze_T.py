"""
analyze_T.py — Post-training analysis of ASTAR's T matrix.

This version is robust to dynamic pool sizes (e.g., when N_sim/N_hist changes),
and will not crash when own-sequence block length differs from L.

Graphs:
    1. Operation distribution over training epochs (time graph)
    2. Per-position operation distribution
    3. Average λ per position
    4. λ and operation distribution vs sequence length
    5. T matrix heatmap

Usage:
    analyzer = TMatrixAnalyzer(args, N_rand=args.N_rand, N_sim=args.N_sim)
    analyzer.record(T, lam, own_mask, epoch)
    analyzer.plot(save_dir='plots/', dataset_name='dataset')
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from collections import defaultdict


class TMatrixAnalyzer:
    """
    Records and analyzes T matrix patterns during/after training.

    Pool structure (intended):
        Position 0:              mask token
        Position 1..N_rand:      random substitution candidates
        Position ...N_sim:       similar-item substitution candidates
        Position ...N_hist:      user-history candidates
        Position ...end:         own sequence positions

    NOTE:
        In practice, pool length can vary by config/runtime.
        This implementation infers own block dynamically from T.shape:
            own_start_dyn = P - L
    """

    def __init__(self, args, N_rand=20, N_sim=0):
        self.L      = args.max_seq_length
        self.N_rand = int(N_rand)
        self.N_sim  = int(N_sim)
        self.N_hist = int(getattr(args, "N_hist", 0))

        # Static boundaries (used as hints; clamped at runtime)
        self.mask_idx   = 0
        self.rand_start = 1
        self.rand_end   = self.rand_start + self.N_rand
        self.sim_start  = self.rand_end
        self.sim_end    = self.sim_start + self.N_sim
        self.hist_start = self.sim_end
        self.hist_end   = self.hist_start + self.N_hist
        self.own_start  = self.hist_end  # intended start (may differ from runtime)

        # Storage
        self.epoch_records   = defaultdict(lambda: defaultdict(list))
        self.lam_records     = defaultdict(list)
        self.length_records  = defaultdict(list)
        self.T_records       = defaultdict(list)
        self.recorded_epochs = []

    @torch.no_grad()
    def record(self, T, lam, own_mask, epoch):
        """
        Record one batch.

        Args:
            T:        [B, P, L]
            lam:      [B, L, 1]
            own_mask: [B, L]
            epoch:    int
        """
        B, P, L = T.shape
        T        = T.detach().cpu().float()
        lam      = lam.detach().cpu().float().squeeze(-1)   # [B, L]
        own_mask = own_mask.detach().cpu().float()          # [B, L]

        # Runtime-own block start inferred from actual pool size
        own_start_dyn = max(P - L, 0)

        # Clamp static boundaries to valid range
        rand_start = min(max(self.rand_start, 0), P)
        rand_end   = min(max(self.rand_end,   0), P)
        sim_start  = min(max(self.sim_start,  0), P)
        sim_end    = min(max(self.sim_end,    0), P)

        # Region weights per output position
        w_mask = T[:, 0, :] if P > 0 else torch.zeros(B, L)

        w_rsub = (
            T[:, rand_start:rand_end, :].sum(dim=1)
            if rand_end > rand_start else torch.zeros_like(w_mask)
        )

        w_ssub = (
            T[:, sim_start:sim_end, :].sum(dim=1)
            if sim_end > sim_start else torch.zeros_like(w_mask)
        )

        # Own-seq region
        w_own = T[:, own_start_dyn:, :]  # expected [B, L, L], but keep robust
        own_len = w_own.shape[1]

        if own_len > 0:
            d = min(own_len, L)
            diag_idx = torch.arange(d)

            w_identity = torch.zeros(B, L)
            w_identity[:, :d] = w_own[:, diag_idx, diag_idx]

            w_shuffle = w_own.sum(dim=1)
            w_shuffle[:, :d] = w_shuffle[:, :d] - w_identity[:, :d]
        else:
            w_identity = torch.zeros_like(w_mask)
            w_shuffle  = torch.zeros_like(w_mask)

        # Masked per-position averages
        denom = own_mask.sum(0).clamp(min=1.0)  # [L]
        for name, w in [
            ("mask",     w_mask),
            ("r_sub",    w_rsub),
            ("s_sub",    w_ssub),
            ("identity", w_identity),
            ("shuffle",  w_shuffle),
        ]:
            masked = (w * own_mask).sum(0) / denom
            self.epoch_records[epoch][name].append(masked.numpy())

        lam_masked = (lam * own_mask).sum(0) / denom
        self.lam_records[epoch].append(lam_masked.numpy())

        seq_lengths = own_mask.sum(dim=1).long()
        self.length_records[epoch].append(seq_lengths.numpy())

        if len(self.T_records[epoch]) < 50:
            self.T_records[epoch].append(T.mean(0).numpy())  # [P, L]

        if epoch not in self.recorded_epochs:
            self.recorded_epochs.append(epoch)

    def _aggregate_epoch(self, epoch):
        ops = {}
        for name, batches in self.epoch_records[epoch].items():
            ops[name] = np.stack(batches, axis=0).mean(axis=0)  # [L]
        lam = np.stack(self.lam_records[epoch], axis=0).mean(axis=0)  # [L]
        return ops, lam

    def plot(self, save_dir="plots/", dataset_name="dataset"):
        os.makedirs(save_dir, exist_ok=True)
        epochs = sorted(self.recorded_epochs)
        if len(epochs) == 0:
            print("[TMatrixAnalyzer] No data recorded.")
            return

        colors = {
            "mask":     "#e74c3c",
            "r_sub":    "#3498db",
            "s_sub":    "#2ecc71",
            "identity": "#95a5a6",
            "shuffle":  "#f39c12",
        }
        labels = {
            "mask":     "Mask",
            "r_sub":    "Random Sub",
            "s_sub":    "Semantic Sub",
            "identity": "Identity",
            "shuffle":  "Shuffle",
        }

        self._plot_time_graph(epochs, colors, labels, save_dir, dataset_name)
        self._plot_position_ops(epochs, colors, labels, save_dir, dataset_name)
        self._plot_lambda_position(epochs, save_dir, dataset_name)
        self._plot_ops_vs_length(epochs, colors, labels, save_dir, dataset_name)
        self._plot_T_heatmap(epochs, save_dir, dataset_name)

        print(f"[TMatrixAnalyzer] Plots saved to {save_dir}")

    def _plot_time_graph(self, epochs, colors, labels, save_dir, dataset_name):
        fig, ax = plt.subplots(figsize=(12, 5))
        op_names = ["mask", "r_sub", "s_sub", "identity", "shuffle"]
        op_means = {name: [] for name in op_names}

        for epoch in epochs:
            ops, _ = self._aggregate_epoch(epoch)
            for name in op_names:
                op_means[name].append(float(np.mean(ops[name])))

        totals = np.array([
            sum(op_means[name][i] for name in op_names)
            for i in range(len(epochs))
        ])
        totals = np.clip(totals, 1e-8, None)

        bottom = np.zeros(len(epochs))
        for name in op_names:
            vals = np.array(op_means[name]) / totals
            ax.fill_between(epochs, bottom, bottom + vals,
                            alpha=0.8, color=colors[name], label=labels[name])
            bottom += vals

        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Operation Proportion", fontsize=12)
        ax.set_title(f"{dataset_name} — Operation Distribution Over Training", fontsize=13)
        ax.legend(loc="upper right", fontsize=10)
        ax.set_xlim(epochs[0], epochs[-1])
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{dataset_name}_ops_time.png"), dpi=150)
        plt.close()

    def _plot_position_ops(self, epochs, colors, labels, save_dir, dataset_name):
        fig, ax = plt.subplots(figsize=(12, 5))
        last_epoch = epochs[-1]
        ops, _ = self._aggregate_epoch(last_epoch)
        op_names = ["mask", "r_sub", "s_sub", "identity", "shuffle"]
        positions = np.arange(self.L)

        total = sum(ops[name] for name in op_names)
        total = np.clip(total, 1e-8, None)

        bottom = np.zeros(self.L)
        for name in op_names:
            vals = ops[name] / total
            ax.bar(positions, vals, bottom=bottom,
                   color=colors[name], label=labels[name],
                   width=1.0, alpha=0.85)
            bottom += vals

        ax.set_xlabel("Sequence Position (0=oldest, L-1=newest)", fontsize=12)
        ax.set_ylabel("Operation Proportion", fontsize=12)
        ax.set_title(f"{dataset_name} — Per-Position Operation Distribution (Epoch {last_epoch})", fontsize=13)
        ax.legend(loc="upper left", fontsize=10)
        ax.set_xlim(-0.5, self.L - 0.5)
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{dataset_name}_ops_position.png"), dpi=150)
        plt.close()

    def _plot_lambda_position(self, epochs, save_dir, dataset_name):
        fig, ax = plt.subplots(figsize=(12, 5))
        positions = np.arange(self.L)
        n_epochs = min(5, len(epochs))
        sampled = [epochs[int(i * (len(epochs)-1) / max(n_epochs-1, 1))] for i in range(n_epochs)]
        cmap = plt.cm.viridis

        for idx, epoch in enumerate(sampled):
            _, lam = self._aggregate_epoch(epoch)
            color = cmap(idx / max(n_epochs - 1, 1))
            ax.plot(positions, lam, color=color, linewidth=2,
                    label=f"Epoch {epoch}", alpha=0.85)

        ax.set_xlabel("Sequence Position (0=oldest, L-1=newest)", fontsize=12)
        ax.set_ylabel("Mean λ", fontsize=12)
        ax.set_title(f"{dataset_name} — Average λ Per Position", fontsize=13)
        ax.legend(fontsize=10)
        ax.set_xlim(0, self.L - 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{dataset_name}_lambda_position.png"), dpi=150)
        plt.close()

    def _plot_ops_vs_length(self, epochs, colors, labels, save_dir, dataset_name):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        last_epoch = epochs[-1]
        lengths = np.concatenate(self.length_records[last_epoch]) if self.length_records[last_epoch] else np.array([])

        buckets = [(1, 5), (6, 10), (11, 20), (21, 35), (36, 50)]
        bucket_labels = ["1-5", "6-10", "11-20", "21-35", "36-50"]
        op_names = ["mask", "r_sub", "s_sub", "identity", "shuffle"]

        bucket_ops = {name: [] for name in op_names}
        bucket_lam = []
        bucket_counts = []

        ops, lam = self._aggregate_epoch(last_epoch)

        for lo, hi in buckets:
            if lengths.size == 0:
                count = 0
            else:
                mask = (lengths >= lo) & (lengths <= hi)
                count = int(mask.sum())

            bucket_counts.append(count)

            if count == 0:
                for name in op_names:
                    bucket_ops[name].append(0.0)
                bucket_lam.append(0.0)
                continue

            mean_len = int(np.mean(lengths[(lengths >= lo) & (lengths <= hi)]))
            mean_len = max(1, min(mean_len, self.L))
            start_pos = self.L - mean_len

            for name in op_names:
                bucket_ops[name].append(float(np.mean(ops[name][start_pos:])))
            bucket_lam.append(float(np.mean(lam[start_pos:])))

        ax = axes[0]
        x = np.arange(len(buckets))
        totals = np.array([
            sum(bucket_ops[name][i] for name in op_names)
            for i in range(len(buckets))
        ])
        totals = np.clip(totals, 1e-8, None)

        bottom = np.zeros(len(buckets))
        for name in op_names:
            vals = np.array(bucket_ops[name]) / totals
            ax.bar(x, vals, bottom=bottom, color=colors[name], label=labels[name], alpha=0.85)
            bottom += vals

        ax.set_xticks(x)
        ax.set_xticklabels([f"{l}\n(n={c})" for l, c in zip(bucket_labels, bucket_counts)], fontsize=9)
        ax.set_ylabel("Operation Proportion", fontsize=12)
        ax.set_title(f"{dataset_name} — Operations vs Sequence Length", fontsize=12)
        ax.legend(fontsize=9, loc="upper right")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)

        ax2 = axes[1]
        bars = ax2.bar(x, bucket_lam, color="#8e44ad", alpha=0.8, width=0.6)
        for bar, val in zip(bars, bucket_lam):
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.01,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=9)

        ax2.set_xticks(x)
        ax2.set_xticklabels(bucket_labels, fontsize=10)
        ax2.set_ylabel("Mean λ", fontsize=12)
        ax2.set_title(f"{dataset_name} — Mean λ vs Sequence Length", fontsize=12)
        ax2.set_ylim(0, 1)
        ax2.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{dataset_name}_ops_vs_length.png"), dpi=150)
        plt.close()

    def _plot_T_heatmap(self, epochs, save_dir, dataset_name):
        last_epoch = epochs[-1]
        if not self.T_records[last_epoch]:
            return

        T_avg = np.stack(self.T_records[last_epoch], axis=0).mean(0)  # [P, L]
        P, L = T_avg.shape

        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(T_avg, aspect="auto", cmap="YlOrRd", interpolation="nearest")
        plt.colorbar(im, ax=ax, label="Mean Attention Weight")

        # Dynamic boundaries from runtime dimensions
        own_start_dyn = max(P - L, 0)
        rand_end = min(1 + self.N_rand, P)
        sim_end = min(rand_end + self.N_sim, P)
        hist_end = min(sim_end + self.N_hist, P)

        # separator lines
        if P > 0:
            ax.axhline(y=0.5, color="blue", linewidth=1.2, linestyle="--", label="Mask token")
        if rand_end > 1:
            ax.axhline(y=rand_end - 0.5, color="green", linewidth=1.2, linestyle="--", label="Random pool end")
        if sim_end > rand_end:
            ax.axhline(y=sim_end - 0.5, color="orange", linewidth=1.2, linestyle="--", label="Sim pool end")
        if hist_end > sim_end:
            ax.axhline(y=hist_end - 0.5, color="purple", linewidth=1.2, linestyle="--", label="Hist pool end")
        if own_start_dyn > 0:
            ax.axhline(y=own_start_dyn - 0.5, color="black", linewidth=1.2, linestyle="--", label="Own seq start")

        # y tick labels by regions
        ytick_pos = []
        ytick_labels = []

        if P > 0:
            ytick_pos.append(0)
            ytick_labels.append("Mask")

        if rand_end > 1:
            ytick_pos.append((1 + rand_end - 1) // 2)
            ytick_labels.append("Random\nSub")

        if sim_end > rand_end:
            ytick_pos.append((rand_end + sim_end - 1) // 2)
            ytick_labels.append("Semantic\nSub")

        if hist_end > sim_end:
            ytick_pos.append((sim_end + hist_end - 1) // 2)
            ytick_labels.append("History")

        if own_start_dyn < P:
            ytick_pos.append((own_start_dyn + P - 1) // 2)
            ytick_labels.append("Own Seq")

        ax.set_yticks(ytick_pos)
        ax.set_yticklabels(ytick_labels, fontsize=10)
        ax.set_xlabel("Output Position (0=oldest, L-1=newest)", fontsize=12)
        ax.set_ylabel("Pool Position (Source)", fontsize=12)
        ax.set_title(f"{dataset_name} — Average T Matrix Heatmap (Epoch {last_epoch})", fontsize=13)
        if len(ax.get_legend_handles_labels()[0]) > 0:
            ax.legend(loc="upper right", fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{dataset_name}_T_heatmap.png"), dpi=150)
        plt.close()

    def print_summary(self, dataset_name="dataset"):
        if not self.recorded_epochs:
            return

        last_epoch = sorted(self.recorded_epochs)[-1]
        ops, lam = self._aggregate_epoch(last_epoch)
        op_names = ["mask", "r_sub", "s_sub", "identity", "shuffle"]

        total = sum(float(np.mean(ops[name])) for name in op_names)
        total = max(total, 1e-8)

        print(f"\n[TMatrixAnalyzer] {dataset_name} — Epoch {last_epoch} Summary")
        print(f"  {'Operation':<15} {'Mean Weight':>12}  {'Proportion':>12}")
        print(f"  {'-'*42}")
        for name in op_names:
            mean_w = float(np.mean(ops[name]))
            print(f"  {name:<15} {mean_w:>12.4f}  {mean_w/total:>11.1%}")

        print(f"\n  Mean λ (all positions): {float(np.mean(lam)):.4f}")
        print(f"  Mean λ (recent half):   {float(np.mean(lam[self.L//2:])):.4f}")
        print(f"  Mean λ (early half):    {float(np.mean(lam[:self.L//2])):.4f}")