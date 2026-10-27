"""Figures for the mutational scan.

The headline figure is a diverging ddG heatmap centred on zero. The original
notebook used a sequential ``Blues_r`` map, which hides the sign of the effect;
ddG is signed and the sign is the whole point, so a diverging map with a
symmetric, percentile-clipped range is used instead. Clipping matters because a
single buried clash can reach +60 REU and would otherwise flatten the entire
colour scale.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

from .logging_utils import get_logger

DPI = 300


def _save(fig, outdir: Path, stem: str, also_pdf: bool = True) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{stem}.png", dpi=DPI, bbox_inches="tight")
    if also_pdf:
        fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    get_logger().info("wrote %s", outdir / f"{stem}.png")


def _figsize(table: pd.DataFrame) -> tuple[float, float]:
    width = max(6.0, 0.45 * table.shape[1] + 3.0)
    height = max(4.0, 0.32 * table.shape[0] + 2.0)
    return width, min(height, 40.0)


def _heatmap(
    table: pd.DataFrame,
    title: str,
    cbar_label: str,
    *,
    cmap: str,
    norm=None,
    annotate: bool = False,
    fmt: str = "{:.1f}",
):
    fig, ax = plt.subplots(figsize=_figsize(table))
    data = table.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(data)
    cmap_obj = matplotlib.colormaps[cmap].with_extremes(bad="0.88")
    image = ax.imshow(masked, aspect="auto", cmap=cmap_obj, norm=norm, interpolation="nearest")

    ax.set_xticks(range(table.shape[1]), table.columns, fontsize=9)
    ax.set_yticks(range(table.shape[0]), table.index, fontsize=8)
    ax.set_xlabel("substitution")
    ax.set_ylabel("position (wild-type residue)")
    ax.set_title(title)
    ax.set_xticks(np.arange(-0.5, table.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, table.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.6)
    ax.tick_params(which="minor", length=0)

    if annotate and table.size <= 900:
        mid = np.nanmean(np.abs(data)) if np.isfinite(data).any() else 0.0
        for i in range(table.shape[0]):
            for j in range(table.shape[1]):
                value = data[i, j]
                if not np.isfinite(value):
                    continue
                ax.text(
                    j,
                    i,
                    fmt.format(value),
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white" if abs(value) > 1.8 * mid else "black",
                )

    bar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    bar.set_label(cbar_label)
    return fig


def _symmetric_norm(data: np.ndarray, clip_percentile: float = 98.0) -> TwoSlopeNorm:
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    limit = float(np.percentile(np.abs(finite), clip_percentile))
    limit = max(limit, 1e-3)
    return TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)


def plot_ddg_heatmap(mats: dict[str, pd.DataFrame], outdir: Path, *, annotate: bool = False) -> None:
    """Mean ddG, with a symmetric diverging scale clipped at the 98th percentile."""
    table = mats["ddg_mean"]
    norm = _symmetric_norm(table.to_numpy(dtype=float))
    fig = _heatmap(
        table,
        "Mean interface ddG across the conformational ensemble\n"
        "blue = stabilizing (ddG < 0), red = destabilizing",
        "ddG (REU)",
        cmap="RdBu_r",
        norm=norm,
        annotate=annotate,
    )
    _save(fig, outdir, "ddg_mean_heatmap")


def plot_sd_heatmap(mats: dict[str, pd.DataFrame], outdir: Path) -> None:
    """Conformational spread of each substitution."""
    table = mats["ddg_sd"]
    fig = _heatmap(
        table,
        "Standard deviation of ddG across the ensemble\n"
        "high values mark substitutions whose effect depends on the conformer",
        "SD of ddG (REU)",
        cmap="viridis",
    )
    _save(fig, outdir, "ddg_sd_heatmap")


def plot_significant_heatmap(
    mats: dict[str, pd.DataFrame], outdir: Path, *, alpha: float = 0.05
) -> None:
    """Mean ddG with substitutions that fail the FDR threshold greyed out."""
    mean = mats["ddg_mean"].copy()
    q = mats["qvalue"]
    mean = mean.where(q < alpha)
    if not np.isfinite(mean.to_numpy(dtype=float)).any():
        get_logger().warning(
            "no substitution reached FDR < %.3g; skipping the significance heatmap", alpha
        )
        return
    norm = _symmetric_norm(mean.to_numpy(dtype=float))
    fig = _heatmap(
        mean,
        f"Mean ddG, showing only substitutions with FDR q < {alpha:g}\n"
        "grey = not distinguishable from zero across the ensemble",
        "ddG (REU)",
        cmap="RdBu_r",
        norm=norm,
        annotate=True,
    )
    _save(fig, outdir, "ddg_significant_heatmap")


def plot_top_mutations(stats: pd.DataFrame, outdir: Path, *, top_n: int = 25) -> None:
    """The lowest-ddG substitutions with 95% confidence intervals.

    Ranked by mean ddG, so on a scan where nothing improves binding this is
    simply the least damaging set -- the title says so rather than implying
    every bar is an improvement.
    """
    real = stats[~stats.is_wt_self].sort_values("ddg_mean").head(top_n)
    if real.empty:
        return
    labels = [
        f"{row.wt_aa}{row.resnum}{row.mut_aa} ({row.chain})" for row in real.itertuples()
    ]
    y = np.arange(len(real))[::-1]
    lower = (real.ddg_mean - real.ci95_low).fillna(0.0)
    upper = (real.ci95_high - real.ddg_mean).fillna(0.0)

    fig, ax = plt.subplots(figsize=(8.0, max(3.0, 0.34 * len(real) + 1.5)))
    colors = ["#2c7fb8" if sig else "#bdbdbd" for sig in real.significant]
    ax.barh(y, real.ddg_mean, color=colors, height=0.72)
    ax.errorbar(
        real.ddg_mean,
        y,
        xerr=[lower.clip(lower=0), upper.clip(lower=0)],
        fmt="none",
        ecolor="0.25",
        elinewidth=1.0,
        capsize=2.5,
    )
    ax.axvline(0.0, color="black", linewidth=0.9)
    ax.set_yticks(y, labels, fontsize=8)
    ax.set_xlabel("mean ddG (REU), error bars = 95% CI across conformers")
    ax.set_title(
        f"{len(real)} most favourable substitutions, ranked by mean ddG\n"
        "ddG < 0 predicts tighter binding; "
        "solid = significant after Benjamini-Hochberg FDR correction"
    )
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    _save(fig, outdir, "top_mutations")


def plot_position_profile(raw: pd.DataFrame, outdir: Path) -> None:
    """Per-position distribution of ddG over all substitutions and conformers."""
    from .stats import usable_rows

    usable = usable_rows(raw)
    usable = usable[usable.mut_name3 != usable.wt_name3]
    if usable.empty:
        return
    order = (
        usable[["position", "chain", "resnum", "icode", "wt_aa"]]
        .drop_duplicates()
        .sort_values(["chain", "resnum", "icode"])
    )
    groups = [usable[usable.position == p].ddg.to_numpy() for p in order.position]
    labels = [f"{p}\n({w})" for p, w in zip(order.position, order.wt_aa)]

    fig, ax = plt.subplots(figsize=(max(7.0, 0.46 * len(groups) + 2.0), 5.0))
    parts = ax.violinplot(groups, showextrema=False, widths=0.85)
    for body in parts["bodies"]:
        body.set_facecolor("#9ecae1")
        body.set_alpha(0.65)
    ax.boxplot(groups, widths=0.22, showfliers=False, medianprops={"color": "crimson"})
    ax.axhline(0.0, color="black", linewidth=0.9, linestyle="--")
    ax.set_xticks(range(1, len(groups) + 1), labels, fontsize=7)
    ax.set_ylabel("ddG (REU)")
    ax.set_xlabel("scanned position (wild-type residue)")
    ax.set_title(
        "Tolerance to substitution per position\n"
        "every substitution in every conformer; low median = position tolerates change"
    )
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    _save(fig, outdir, "position_profile")


def plot_convergence(conv: pd.DataFrame, outdir: Path) -> None:
    """Drift of the mean ddG matrix as conformers accumulate."""
    if conv.empty or len(conv) < 3:
        return
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.0, 4.0))
    left.plot(conv.n_conformers, conv.mean_abs_drift, "o-", color="#2c7fb8", label="mean")
    left.plot(conv.n_conformers, conv.max_abs_drift, "s--", color="#d95f0e", label="max")
    left.set_xlabel("conformers included")
    left.set_ylabel("|running mean - final mean| (REU)")
    left.set_title("Convergence of the ddG matrix")
    left.legend(frameon=False)
    left.grid(alpha=0.3)

    right.plot(conv.n_conformers, conv.mean_sem, "o-", color="#31a354")
    right.set_xlabel("conformers included")
    right.set_ylabel("mean SEM of ddG (REU)")
    right.set_title("Precision vs ensemble size")
    right.grid(alpha=0.3)
    fig.suptitle(
        "Flat curves mean the ensemble is large enough; "
        "still-falling curves mean more conformers are needed",
        fontsize=9,
    )
    _save(fig, outdir, "convergence")


def plot_position_hotspots(summary: pd.DataFrame, outdir: Path) -> None:
    """Wild-type interface contribution against substitution tolerance."""
    if "wt_interface_contribution" not in summary.columns:
        return
    data = summary[np.isfinite(summary.wt_interface_contribution)]
    if data.empty:
        return
    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    sizes = 30 + 14 * data.n_significant_stabilizing
    points = ax.scatter(
        data.wt_interface_contribution,
        data.mean_ddg_all_substitutions,
        c=data.best_ddg_mean,
        s=sizes,
        cmap="RdBu_r",
        norm=_symmetric_norm(data.best_ddg_mean.to_numpy(dtype=float)),
        edgecolor="0.3",
        linewidth=0.5,
    )
    for row in data.itertuples():
        ax.annotate(
            f"{row.wt_aa}{row.resnum}",
            (row.wt_interface_contribution, row.mean_ddg_all_substitutions),
            fontsize=7,
            xytext=(3, 3),
            textcoords="offset points",
        )
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("wild-type contribution to the interface energy (REU)")
    ax.set_ylabel("mean ddG over all substitutions (REU)")
    ax.set_title(
        "Hotspots and tolerance\n"
        "left = residue already binds strongly; high = position resists substitution\n"
        "marker size = number of significant stabilizing substitutions"
    )
    bar = fig.colorbar(points, ax=ax, fraction=0.04)
    bar.set_label("best ddG at the position (REU)")
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    _save(fig, outdir, "position_hotspots")


def plot_term_decomposition(terms: pd.DataFrame, stats: pd.DataFrame, outdir: Path, *, top_n: int = 20) -> None:
    """Per-score-term breakdown of the most stabilizing substitutions."""
    if terms is None or terms.empty:
        return
    columns = [c for c in terms.columns if c.startswith("ddg_")]
    if not columns:
        return
    best = stats[~stats.is_wt_self].sort_values("ddg_mean").head(top_n)
    keys = set(zip(best.position, best.mut_name3))
    chosen = terms[[(p, m) in keys for p, m in zip(terms.position, terms.mut_name3)]]
    if chosen.empty:
        return
    grouped = chosen.groupby(["position", "mut_name3"])[columns].mean()
    grouped = grouped.loc[:, grouped.abs().max() > 0.05]
    if grouped.empty:
        return
    lookup = {(row.position, row.mut_name3): row.ddg_mean for row in best.itertuples()}
    grouped = grouped.reindex(sorted(grouped.index, key=lambda k: lookup.get(k, 0.0)))
    labels = [f"{p} {m}" for p, m in grouped.index]
    grouped.index = labels
    grouped.columns = [c.removeprefix("ddg_") for c in grouped.columns]

    norm = _symmetric_norm(grouped.to_numpy(dtype=float))
    fig = _heatmap(
        grouped,
        "Score-term decomposition of the most stabilizing substitutions\n"
        "which energy term each ddG comes from",
        "ddG contribution (REU)",
        cmap="RdBu_r",
        norm=norm,
        annotate=True,
        fmt="{:.2f}",
    )
    fig.axes[0].set_ylabel("position and substitution")
    fig.axes[0].set_xlabel("score term")
    _save(fig, outdir, "term_decomposition")


def make_all(
    stats: pd.DataFrame,
    summary: pd.DataFrame,
    conv: pd.DataFrame,
    mats: dict[str, pd.DataFrame],
    raw: pd.DataFrame,
    terms: pd.DataFrame | None,
    outdir: Path,
    *,
    alpha: float = 0.05,
    annotate: bool = False,
    top_n: int = 25,
) -> None:
    """Produce every figure into ``outdir``."""
    plot_ddg_heatmap(mats, outdir, annotate=annotate)
    plot_sd_heatmap(mats, outdir)
    plot_significant_heatmap(mats, outdir, alpha=alpha)
    plot_top_mutations(stats, outdir, top_n=top_n)
    plot_position_profile(raw, outdir)
    plot_convergence(conv, outdir)
    plot_position_hotspots(summary, outdir)
    plot_term_decomposition(terms, stats, outdir)
