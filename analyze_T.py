import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
from collections import defaultdict


class TMatrixAnalyzer:
    """
    Records and analyzes T matrix patterns during/after training.
    Supports dual T-head mode for two_head architecture.
    """

    def __init__(self, args, N_rand=20, N_sim=0):
        self.L      = args.max_seq_length
        self.N_rand = int(N_rand)
        self.N_sim  = int(N_sim)
        self.N_hist = int(getattr(args, "N_hist", 0))
        aug_opt = getattr(args, "augmentation_option", None)
        if aug_opt is None:
            aug_opt = "no_mask" if getattr(args, "ablation", "none") == "no_mask" else "mask"
        self.use_mask_token = (aug_opt == "mask")

        # Static boundaries
        self.mask_idx   = 0 if self.use_mask_token else None
        self.rand_start = 1 if self.use_mask_token else 0
        self.rand_end   = self.rand_start + self.N_rand
        self.sim_start  = self.rand_end
        self.sim_end    = self.sim_start + self.N_sim
        self.hist_start = self.sim_end
        self.hist_end   = self.hist_start + self.N_hist
        self.own_start  = self.hist_end

        # Storage for T1 (primary)
        self.epoch_records   = defaultdict(lambda: defaultdict(list))
        self.lam_records     = defaultdict(list)
        self.length_records  = defaultdict(list)
        self.T_records       = defaultdict(list)

        # Storage for T2 (dual-head)
        self.epoch_records_T2 = defaultdict(lambda: defaultdict(list))
        self.T_records_T2     = defaultdict(list)

        # Divergence tracking
        self.T_divergence     = defaultdict(list)  # ||T1 - T2|| per epoch

        self.recorded_epochs  = []
        self.dual_head_mode   = True

    @torch.no_grad()
    def record(self, T, lam, own_mask, epoch, T2=None):
        """
        Record one batch.

        Args:
            T:        [B, P, L] - primary T matrix
            lam:      [B, L, 1]
            own_mask: [B, L]
            epoch:    int
            T2:       [B, P, L] - secondary T matrix (dual-head mode), optional
        """
        B, P, L = T.shape
        T        = T.detach().cpu().float()
        lam      = lam.detach().cpu().float().squeeze(-1)
        own_mask = own_mask.detach().cpu().float()

        # Record T1
        self._record_T(T, own_mask, epoch, self.epoch_records, self.T_records, P, L, B)

        # Record lambda
        denom = own_mask.sum(0).clamp(min=1.0)
        lam_masked = (lam * own_mask).sum(0) / denom
        self.lam_records[epoch].append(lam_masked.numpy())

        # Record sequence lengths
        seq_lengths = own_mask.sum(dim=1).long()
        self.length_records[epoch].append(seq_lengths.numpy())

        # Record T2 if provided (dual-head mode)
        if T2 is not None:
            self.dual_head_mode = True
            T2 = T2.detach().cpu().float()
            self._record_T(T2, own_mask, epoch, self.epoch_records_T2, self.T_records_T2, P, L, B)

            # Compute divergence between T1 and T2
            divergence = (T - T2).abs().mean().item()
            self.T_divergence[epoch].append(divergence)

        if epoch not in self.recorded_epochs:
            self.recorded_epochs.append(epoch)

    def _record_T(self, T, own_mask, epoch, epoch_records, T_records, P, L, B):
        """Helper to record ops for a single T matrix."""
        own_start_dyn = max(P - L, 0)

        rand_start = min(max(self.rand_start, 0), P)
        rand_end   = min(max(self.rand_end,   0), P)
        sim_start  = min(max(self.sim_start,  0), P)
        sim_end    = min(max(self.sim_end,    0), P)

        w_mask = T[:, 0, :] if (self.use_mask_token and P > 0) else torch.zeros(B, L)

        w_rsub = (
            T[:, rand_start:rand_end, :].sum(dim=1)
            if rand_end > rand_start else torch.zeros_like(w_mask)
        )

        w_ssub = (
            T[:, sim_start:sim_end, :].sum(dim=1)
            if sim_end > sim_start else torch.zeros_like(w_mask)
        )

        w_own = T[:, own_start_dyn:, :]
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

        denom = own_mask.sum(0).clamp(min=1.0)
        for name, w in [
            ("mask",     w_mask),
            ("r_sub",    w_rsub),
            ("s_sub",    w_ssub),
            ("identity", w_identity),
            ("shuffle",  w_shuffle),
        ]:
            masked = (w * own_mask).sum(0) / denom
            epoch_records[epoch][name].append(masked.numpy())

        if len(T_records[epoch]) < 50:
            T_records[epoch].append(T.mean(0).numpy())

    def _aggregate_epoch(self, epoch, epoch_records=None):
        if epoch_records is None:
            epoch_records = self.epoch_records
        ops = {}
        for name, batches in epoch_records[epoch].items():
            ops[name] = np.stack(batches, axis=0).mean(axis=0)
        lam = np.stack(self.lam_records[epoch], axis=0).mean(axis=0)
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

        print(f"[TMatrixAnalyzer] Generating plots (dual_head_mode={self.dual_head_mode})...")

        # Determine suffix for single vs dual head mode
        if self.dual_head_mode:
            # === DUAL HEAD: Generate separate plots for T1 and T2 ===
            print("  → Generating T1 plots...")
            self._plot_time_graph(epochs, colors, labels, save_dir, dataset_name, 
                                  suffix="_T1", epoch_records=self.epoch_records, head_name="T1")
            self._plot_position_ops(epochs, colors, labels, save_dir, dataset_name, 
                                    suffix="_T1", epoch_records=self.epoch_records, head_name="T1")
            self._plot_T_heatmap(epochs, save_dir, dataset_name, 
                                 suffix="_T1", T_records=self.T_records, head_name="T1")
            
            print("  → Generating T2 plots...")
            self._plot_time_graph(epochs, colors, labels, save_dir, dataset_name, 
                                  suffix="_T2", epoch_records=self.epoch_records_T2, head_name="T2")
            self._plot_position_ops(epochs, colors, labels, save_dir, dataset_name, 
                                    suffix="_T2", epoch_records=self.epoch_records_T2, head_name="T2")
            self._plot_T_heatmap(epochs, save_dir, dataset_name, 
                                 suffix="_T2", T_records=self.T_records_T2, head_name="T2")
            
            print("  → Generating comparison plots...")
            self._plot_T_comparison(epochs, save_dir, dataset_name)
            self._plot_divergence(epochs, save_dir, dataset_name)
        else:
            # === SINGLE HEAD: No suffix needed ===
            print("  → Generating single-head plots...")
            self._plot_time_graph(epochs, colors, labels, save_dir, dataset_name, 
                                  suffix="", epoch_records=self.epoch_records, head_name=None)
            self._plot_position_ops(epochs, colors, labels, save_dir, dataset_name, 
                                    suffix="", epoch_records=self.epoch_records, head_name=None)
            self._plot_T_heatmap(epochs, save_dir, dataset_name, 
                                 suffix="", T_records=self.T_records, head_name=None)

        # Common plots (same for both modes)
        print("  → Generating common plots (λ, ops vs length)...")
        self._plot_lambda_position(epochs, save_dir, dataset_name)
        self._plot_ops_vs_length(epochs, colors, labels, save_dir, dataset_name)

        print(f"[TMatrixAnalyzer] All plots saved to {save_dir}")

    def _plot_time_graph(self, epochs, colors, labels, save_dir, dataset_name, 
                         suffix="", epoch_records=None, head_name=None):
        if epoch_records is None:
            epoch_records = self.epoch_records

        fig, ax = plt.subplots(figsize=(12, 5))
        op_names = ["mask", "r_sub", "s_sub", "identity", "shuffle"]
        op_means = {name: [] for name in op_names}

        for epoch in epochs:
            ops, _ = self._aggregate_epoch(epoch, epoch_records)
            for name in op_names:
                op_means[name].append(float(np.mean(ops.get(name, np.zeros(self.L)))))

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
        
        title_suffix = f" ({head_name})" if head_name else ""
        ax.set_title(f"{dataset_name} — Operation Distribution Over Training{title_suffix}", fontsize=13)
        ax.legend(loc="upper right", fontsize=10)
        ax.set_xlim(epochs[0], epochs[-1])
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        filename = f"{dataset_name}_ops_time{suffix}.png"
        plt.savefig(os.path.join(save_dir, filename), dpi=150)
        plt.close()
        print(f"    Saved: {filename}")

    def _plot_position_ops(self, epochs, colors, labels, save_dir, dataset_name, 
                           suffix="", epoch_records=None, head_name=None):
        if epoch_records is None:
            epoch_records = self.epoch_records

        fig, ax = plt.subplots(figsize=(12, 5))
        last_epoch = epochs[-1]
        ops, _ = self._aggregate_epoch(last_epoch, epoch_records)
        op_names = ["mask", "r_sub", "s_sub", "identity", "shuffle"]
        positions = np.arange(self.L)

        total = sum(ops.get(name, np.zeros(self.L)) for name in op_names)
        total = np.clip(total, 1e-8, None)

        bottom = np.zeros(self.L)
        for name in op_names:
            vals = ops.get(name, np.zeros(self.L)) / total
            ax.bar(positions, vals, bottom=bottom,
                   color=colors[name], label=labels[name],
                   width=1.0, alpha=0.85)
            bottom += vals

        ax.set_xlabel("Sequence Position (0=oldest, L-1=newest)", fontsize=12)
        ax.set_ylabel("Operation Proportion", fontsize=12)
        
        title_suffix = f" ({head_name})" if head_name else ""
        ax.set_title(f"{dataset_name} — Per-Position Operation Distribution (Epoch {last_epoch}){title_suffix}", fontsize=13)
        ax.legend(loc="upper left", fontsize=10)
        ax.set_xlim(-0.5, self.L - 0.5)
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        filename = f"{dataset_name}_ops_position{suffix}.png"
        plt.savefig(os.path.join(save_dir, filename), dpi=150)
        plt.close()
        print(f"    Saved: {filename}")

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
        filename = f"{dataset_name}_lambda_position.png"
        plt.savefig(os.path.join(save_dir, filename), dpi=150)
        plt.close()
        print(f"    Saved: {filename}")

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
        filename = f"{dataset_name}_ops_vs_length.png"
        plt.savefig(os.path.join(save_dir, filename), dpi=150)
        plt.close()
        print(f"    Saved: {filename}")

    def _normalize_T_per_column(self, T_avg):
        """
        Normalize T matrix so each column sums to 1 (softmax per output position).
        
        T_avg: [P, L] - pool positions × output positions
        Returns: [P, L] with each column summing to 1
        """
        col_sums = T_avg.sum(axis=0, keepdims=True)  # [1, L]
        col_sums = np.clip(col_sums, 1e-8, None)
        T_normalized = T_avg / col_sums
        return T_normalized

    def _plot_T_heatmap(self, epochs, save_dir, dataset_name, suffix="", T_records=None, head_name=None):
        if T_records is None:
            T_records = self.T_records

        last_epoch = epochs[-1]
        if not T_records[last_epoch]:
            print(f"    Skipped heatmap{suffix}: no T records for epoch {last_epoch}")
            return

        T_avg = np.stack(T_records[last_epoch], axis=0).mean(0)  # [P, L]
        P, L = T_avg.shape

        # Normalize per column (each output position sums to 1)
        T_normalized = self._normalize_T_per_column(T_avg)

        fig, ax = plt.subplots(figsize=(14, 8))
        
        # === BETTER COLOR SCALING ===
        # Find appropriate vmax based on data (use 95th percentile to avoid outliers)
        vmax_percentile = np.percentile(T_normalized, 98)
        vmax = max(vmax_percentile, 0.1)  # at least 0.1
        
        # Use power normalization for better contrast on small values
        gamma = 0.5  # enhance small-to-medium values
        norm = mcolors.PowerNorm(gamma=gamma, vmin=0, vmax=vmax)
        
        im = ax.imshow(T_normalized, aspect="auto", cmap="hot", interpolation="nearest", norm=norm)
        cbar = plt.colorbar(im, ax=ax, label="Attention Weight (column-normalized)", shrink=0.8)
        cbar.ax.set_ylabel("Attention Weight\n(γ=0.5 power scale)", fontsize=10)

        # Add region separators
        own_start_dyn = max(P - L, 0)
        rand_start = 1 if self.use_mask_token else 0
        rand_end = min(rand_start + self.N_rand, P)
        sim_end = min(rand_end + self.N_sim, P)
        hist_end = min(sim_end + self.N_hist, P)

        line_style = dict(linewidth=2, linestyle="-", alpha=0.9)
        if self.use_mask_token and P > 0:
            ax.axhline(y=0.5, color="cyan", label="Mask token", **line_style)
        if rand_end > rand_start:
            ax.axhline(y=rand_end - 0.5, color="lime", label="Random pool end", **line_style)
        if sim_end > rand_end:
            ax.axhline(y=sim_end - 0.5, color="orange", label="Sim pool end", **line_style)
        if hist_end > sim_end:
            ax.axhline(y=hist_end - 0.5, color="magenta", label="Hist pool end", **line_style)
        if own_start_dyn > 0:
            ax.axhline(y=own_start_dyn - 0.5, color="white", label="Own seq start", **line_style)

        # Y-axis labels
        ytick_pos = []
        ytick_labels_list = []

        if self.use_mask_token and P > 0:
            ytick_pos.append(0)
            ytick_labels_list.append("Mask")

        if rand_end > rand_start:
            ytick_pos.append((rand_start + rand_end - 1) // 2)
            ytick_labels_list.append("Random\nSub")

        if sim_end > rand_end:
            ytick_pos.append((rand_end + sim_end - 1) // 2)
            ytick_labels_list.append("Semantic\nSub")

        if hist_end > sim_end:
            ytick_pos.append((sim_end + hist_end - 1) // 2)
            ytick_labels_list.append("History")

        if own_start_dyn < P:
            ytick_pos.append((own_start_dyn + P - 1) // 2)
            ytick_labels_list.append("Own Seq")

        ax.set_yticks(ytick_pos)
        ax.set_yticklabels(ytick_labels_list, fontsize=11, fontweight='bold')
        ax.set_xlabel("Output Position (0=oldest, L-1=newest)", fontsize=12)
        ax.set_ylabel("Pool Position (Source)", fontsize=12)
        
        title_suffix = f" ({head_name})" if head_name else ""
        ax.set_title(f"{dataset_name} — T Matrix Heatmap (Epoch {last_epoch}){title_suffix}\n"
                     f"Column-normalized (each output position sums to 1)", fontsize=13)
        
        if len(ax.get_legend_handles_labels()[0]) > 0:
            ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

        plt.tight_layout()
        filename = f"{dataset_name}_T_heatmap{suffix}.png"
        plt.savefig(os.path.join(save_dir, filename), dpi=150)
        plt.close()
        print(f"    Saved: {filename}")

    def _plot_T_comparison(self, epochs, save_dir, dataset_name):
        """Side-by-side T1 vs T2 heatmap comparison (column-normalized)."""
        last_epoch = epochs[-1]
        if not self.T_records[last_epoch] or not self.T_records_T2[last_epoch]:
            print("    Skipped T comparison: missing T records")
            return

        T1_avg = np.stack(self.T_records[last_epoch], axis=0).mean(0)
        T2_avg = np.stack(self.T_records_T2[last_epoch], axis=0).mean(0)
        
        # Normalize per column
        T1_norm = self._normalize_T_per_column(T1_avg)
        T2_norm = self._normalize_T_per_column(T2_avg)
        T_diff = np.abs(T1_norm - T2_norm)

        # Better color scaling
        vmax = max(np.percentile(T1_norm, 98), np.percentile(T2_norm, 98), 0.1)
        gamma = 0.5
        norm = mcolors.PowerNorm(gamma=gamma, vmin=0, vmax=vmax)
        
        diff_vmax = max(np.percentile(T_diff, 98), 0.05)
        diff_norm = mcolors.PowerNorm(gamma=0.5, vmin=0, vmax=diff_vmax)

        fig, axes = plt.subplots(1, 3, figsize=(20, 7))

        # T1
        im1 = axes[0].imshow(T1_norm, aspect="auto", cmap="hot", interpolation="nearest", norm=norm)
        axes[0].set_title("T1 (Head 1)\nColumn Normalized", fontsize=12, fontweight='bold')
        axes[0].set_xlabel("Output Position")
        axes[0].set_ylabel("Pool Position")
        plt.colorbar(im1, ax=axes[0], fraction=0.046, label="Attention")

        # T2
        im2 = axes[1].imshow(T2_norm, aspect="auto", cmap="hot", interpolation="nearest", norm=norm)
        axes[1].set_title("T2 (Head 2)\nColumn Normalized", fontsize=12, fontweight='bold')
        axes[1].set_xlabel("Output Position")
        axes[1].set_ylabel("Pool Position")
        plt.colorbar(im2, ax=axes[1], fraction=0.046, label="Attention")

        # Difference
        im3 = axes[2].imshow(T_diff, aspect="auto", cmap="coolwarm", interpolation="nearest", norm=diff_norm)
        axes[2].set_title("|T1 - T2| Difference\n(Where heads disagree)", fontsize=12, fontweight='bold')
        axes[2].set_xlabel("Output Position")
        axes[2].set_ylabel("Pool Position")
        plt.colorbar(im3, ax=axes[2], fraction=0.046, label="Abs Difference")

        # Add mean difference annotation
        mean_diff = T_diff.mean()
        max_diff = T_diff.max()
        axes[2].text(0.02, 0.98, f"Mean: {mean_diff:.4f}\nMax: {max_diff:.4f}", 
                     transform=axes[2].transAxes, fontsize=10, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        fig.suptitle(f"{dataset_name} — T1 vs T2 Comparison (Epoch {last_epoch})", fontsize=14, fontweight='bold')
        plt.tight_layout()
        filename = f"{dataset_name}_T_comparison.png"
        plt.savefig(os.path.join(save_dir, filename), dpi=150)
        plt.close()
        print(f"    Saved: {filename}")

    def _plot_divergence(self, epochs, save_dir, dataset_name):
        """Plot ||T1 - T2|| divergence over training."""
        if not self.T_divergence:
            print("    Skipped divergence plot: no data")
            return

        fig, ax = plt.subplots(figsize=(10, 5))

        div_means = []
        for epoch in epochs:
            if self.T_divergence[epoch]:
                div_means.append(np.mean(self.T_divergence[epoch]))
            else:
                div_means.append(0.0)

        ax.plot(epochs, div_means, linewidth=2.5, color="#e74c3c", marker='o', markersize=4)
        ax.fill_between(epochs, 0, div_means, alpha=0.3, color="#e74c3c")

        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Mean |T1 - T2|", fontsize=12)
        ax.set_title(f"{dataset_name} — T1 vs T2 Divergence Over Training", fontsize=13)
        ax.grid(alpha=0.3)
        ax.set_xlim(epochs[0], epochs[-1])
        ax.set_ylim(0, None)

        # Add annotation for final divergence
        if div_means:
            final_div = div_means[-1]
            ax.annotate(f"Final: {final_div:.4f}",
                        xy=(epochs[-1], final_div),
                        xytext=(epochs[-1] - len(epochs)*0.2, final_div * 1.2 + 0.01),
                        fontsize=11, fontweight='bold',
                        arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
            
            # Warning if divergence is too low
            if final_div < 0.01:
                ax.text(0.5, 0.9, "⚠️ WARNING: T1 ≈ T2 (L_push will be ~0)", 
                       transform=ax.transAxes, fontsize=12, color='red', fontweight='bold',
                       ha='center', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

        plt.tight_layout()
        filename = f"{dataset_name}_T_divergence.png"
        plt.savefig(os.path.join(save_dir, filename), dpi=150)
        plt.close()
        print(f"    Saved: {filename}")

    def print_summary(self, dataset_name="dataset"):
        if not self.recorded_epochs:
            print("[TMatrixAnalyzer] No data to summarize.")
            return

        last_epoch = sorted(self.recorded_epochs)[-1]
        ops, lam = self._aggregate_epoch(last_epoch)
        op_names = ["mask", "r_sub", "s_sub", "identity", "shuffle"]

        total = sum(float(np.mean(ops[name])) for name in op_names)
        total = max(total, 1e-8)

        print(f"\n{'='*60}")
        print(f"[TMatrixAnalyzer] {dataset_name} — Epoch {last_epoch} Summary")
        print(f"{'='*60}")
        
        if self.dual_head_mode:
            print(f"\n--- T1 (Head 1) ---")
        
        print(f"  {'Operation':<15} {'Mean Weight':>12}  {'Proportion':>12}")
        print(f"  {'-'*42}")
        for name in op_names:
            mean_w = float(np.mean(ops[name]))
            print(f"  {name:<15} {mean_w:>12.4f}  {mean_w/total:>11.1%}")

        print(f"\n  λ Statistics:")
        print(f"    Mean λ (all positions): {float(np.mean(lam)):.4f}")
        print(f"    Mean λ (recent half):   {float(np.mean(lam[self.L//2:])):.4f}")
        print(f"    Mean λ (early half):    {float(np.mean(lam[:self.L//2])):.4f}")

        if self.dual_head_mode:
            ops2, _ = self._aggregate_epoch(last_epoch, self.epoch_records_T2)
            total2 = sum(float(np.mean(ops2.get(name, np.zeros(self.L)))) for name in op_names)
            total2 = max(total2, 1e-8)

            print(f"\n--- T2 (Head 2) ---")
            print(f"  {'Operation':<15} {'Mean Weight':>12}  {'Proportion':>12}")
            print(f"  {'-'*42}")
            for name in op_names:
                mean_w = float(np.mean(ops2.get(name, np.zeros(self.L))))
                print(f"  {name:<15} {mean_w:>12.4f}  {mean_w/total2:>11.1%}")

            # Divergence summary
            if self.T_divergence[last_epoch]:
                final_div = np.mean(self.T_divergence[last_epoch])
                print(f"\n--- T1 vs T2 Divergence ---")
                print(f"  Mean |T1 - T2|: {final_div:.4f}")
                if final_div < 0.01:
                    print(f"  ⚠️  WARNING: T1 and T2 are nearly identical!")
                    print(f"       This means L_push will be ~0.")
                    print(f"       Consider: separate pools, orthogonal init, or diversity loss.")
                elif final_div < 0.05:
                    print(f"  ⚠️  CAUTION: T1 and T2 are quite similar.")
                else:
                    print(f"  ✓ T1 and T2 are sufficiently different.")
        
        print(f"{'='*60}\n")