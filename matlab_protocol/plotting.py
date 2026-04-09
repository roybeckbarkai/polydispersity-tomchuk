from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def plot_tenor_violin_closest(results_table, instrument, output_path=None):
    data_tab = results_table.copy()
    dq = (
        4.0
        * np.pi
        / instrument.lambda_
        * instrument.det_side
        / instrument.SD_dist
        / (2 * round(instrument.DETpix / 2) + 1)
    )
    data_tab = data_tab[data_tab["Noise"] != 0].copy()
    noislist = list(dict.fromkeys(data_tab["Noise"].tolist()))
    num_n = len(noislist)

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    ax.grid(True)
    colors = plt.cm.tab10(np.linspace(0, 1, max(num_n, 1)))
    jitter_width = 0.2
    robust_pct = 95

    all_means = np.full(num_n, np.nan)
    all_upper = np.full(num_n, np.nan)
    all_lower = np.full(num_n, np.nan)
    tick_labels = []
    rng = np.random.default_rng(12345)

    for i, curr_noise in enumerate(noislist, start=1):
        sub = data_tab[data_tab["Noise"] == curr_noise]
        valid_idx = sub["Primary_V"].notna().to_numpy()
        total_attempts = len(sub)
        num_valid = int(valid_idx.sum())
        p_valid = 100.0 * num_valid / max(total_attempts, 1)
        true_v = sub.loc[valid_idx, "True_V"].to_numpy(dtype=float)
        primary_v = sub.loc[valid_idx, "Primary_V"].to_numpy(dtype=float)
        deltas = np.sqrt(np.maximum(0.0, primary_v)) - np.sqrt(true_v)
        if len(deltas) == 0:
            ax.text(i, 0.0, "0%", ha="center", color="r", fontweight="bold")
            tick_labels.append(f"{curr_noise:0.1e}")
            continue

        s_data = np.sort(deltas)
        N = len(s_data)
        p_off = (100 - robust_pct) / 2 / 100
        idx_low = max(0, int(round(p_off * N)) - 1)
        idx_high = min(N, int(round((1 - p_off) * N)))
        subset = s_data[idx_low:idx_high]
        avg = float(np.mean(subset))
        r_std = float(np.std(subset))
        all_means[i - 1] = avg
        all_upper[i - 1] = avg + r_std
        all_lower[i - 1] = avg - r_std

        counts, edges = np.histogram(deltas, bins=30)
        centers = edges[:-1] + np.diff(edges) / 2.0
        if counts.max() > 0:
            widths = counts / counts.max() * 0.4
            ax.fill(
                np.r_[i - widths, (i + widths)[::-1]],
                np.r_[centers, centers[::-1]],
                color=colors[i - 1],
                alpha=0.25,
                edgecolor=colors[i - 1],
            )

        x_jitter = i + (rng.random(len(deltas)) - 0.5) * jitter_width
        ax.plot(x_jitter, deltas, ".", color=(0.5, 0.5, 0.5), markersize=4)
        ax.plot([i - 0.2, i + 0.2], [avg, avg], color="r", linewidth=2)
        ax.plot([i, i], [avg - r_std, avg + r_std], color=(0.2, 0.2, 0.2), linewidth=1.5)
        ax.text(i, 0.18, f"{round(p_valid)}%\n valid", ha="center", va="bottom", fontsize=9, fontweight="bold")

        if curr_noise >= 0:
            tick_labels.append(f"{curr_noise:0.1e}")
        else:
            ph_density = -curr_noise / (dq**2)
            base, exp = f"{ph_density:0.1e}".split("e")
            tick_labels.append(rf"${base}\times 10^{{{int(exp)}}}$")

    if num_n > 1:
        x = np.arange(1, num_n + 1)
        ax.plot(x, all_means, "r-", linewidth=1.2, label="Mean discrepancy")
        ax.plot(x, all_upper, "k--", linewidth=1, label=f"Robust ({robust_pct}%) ± 1 STD")
        ax.plot(x, all_lower, "k--", linewidth=1)

    ax.axhline(0.0, color="k", alpha=0.4)
    ax.set_xticks(np.arange(1, num_n + 1))
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel(r"Photon density (photon/$\mathrm{nm}^{-2}$)")
    ax.set_ylabel(r"$\Delta(V^{1/2})$ discrepancy")
    ax.set_title("Convergence and Validity of Closest Solutions")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper right")
    if output_path is not None:
        fig.savefig(output_path, dpi=200)
    return fig, ax
