"""Ensemble statistics for the mutational scan.

The scan produces one ddG per (conformer, position, substitution). Averaging
those into a single matrix throws away most of the information, so this module
reports, for every substitution:

Location
    ``ddg_mean`` -- arithmetic mean over conformers, the headline number.
    ``ddg_median`` / ``ddg_iqr`` -- robust alternatives. A large gap between
    mean and median means one or two conformers dominate, usually through a
    clash that the shell repack could not resolve.
    ``ddg_boltzmann`` -- ``-kT ln <exp(-ddG/kT)>``, the ensemble free-energy
    average. Unlike the arithmetic mean it is dominated by the conformers in
    which the substitution works, and by Jensen's inequality it is always
    <= ``ddg_mean``. Use it when the conformers are meant to represent
    thermally accessible states rather than repeated measurements.

Spread and uncertainty
    ``ddg_sd`` (sample SD, ddof=1), ``ddg_sem`` = SD/sqrt(n), and a
    t-distribution ``ci95_low``/``ci95_high``. The t interval is used rather
    than a normal one because an ensemble of 10-25 conformers is small.

Significance
    ``t_stat``, ``p_value`` from a paired one-sample t-test of ddG against 0 --
    valid because each mutant is differenced against its own conformer's
    wild-type control -- and ``q_value``, the Benjamini-Hochberg FDR across
    every substitution tested. With 20 substitutions at 23 positions there are
    460 tests, so the raw p-values cannot be read directly.

Effect size and consistency
    ``cohens_dz`` = mean/SD, the paired effect size: how large the shift is
    relative to conformational noise.
    ``frac_stabilizing`` -- the fraction of conformers that agree the ddG is
    negative. A substitution with a good mean but ``frac_stabilizing`` near
    0.5 is not reproducible across the ensemble.

Wild-type self-substitutions (WT->WT) are kept in the table as a reference
row and flagged with ``is_wt_self``, but excluded from the FDR correction,
where a near-zero-variance test is meaningless. Packing alone reproduces the
control exactly, so their ddG is 0; with ``--minimize`` two independent
minimisations of the same problem differ by a tiny amount, which makes these
rows a direct read-out of the protocol's numerical noise floor (typically
<= 0.001 REU).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import special as spspecial
from scipy import stats as sps

from tmol.pose import PoseStack
from tmol.score import ScoreFunction

from .logging_utils import get_logger
from .structure import Residue

DEFAULT_KT = 0.593
"""kT at 298 K in kcal/mol; Rosetta energy units are close to kcal/mol."""


def normalize_raw(raw: pd.DataFrame) -> pd.DataFrame:
    """Make a raw table safe to filter, whether in memory or round-tripped.

    Writing the table to CSV turns the empty ``skipped`` string into a missing
    value, so a later ``skipped == ""`` test would match nothing. Normalising
    once here keeps the in-memory and reloaded paths identical.
    """
    raw = raw.copy()
    if "skipped" not in raw.columns:
        raw["skipped"] = ""
    text_columns = (
        "skipped", "icode", "wt_aa", "wt_name3", "mut_aa", "mut_name3", "position",
        "chain", "achieved_name3",
    )
    for column in text_columns:
        if column in raw.columns:
            # via object dtype: an all-missing column reloads as float64, and
            # filling a float column with a string is an error in pandas 4.
            raw[column] = raw[column].astype(object).where(raw[column].notna(), "")
            raw[column] = raw[column].astype(str)
    for column in ("ddg", "e_wt", "e_mut", "ratio", "percentage", "conformer",
                   "total_wt", "total_mut", "ddg_total", "ddg_ref", "strain"):
        if column in raw.columns:
            raw[column] = pd.to_numeric(raw[column], errors="coerce")
    return raw


def usable_rows(raw: pd.DataFrame) -> pd.DataFrame:
    """Rows that were honoured by the packer and produced a finite ddG."""
    raw = normalize_raw(raw)
    return raw[(raw.skipped == "") & np.isfinite(raw.ddg)]


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR-adjusted p-values, NaN-safe."""
    q = np.full_like(p_values, np.nan, dtype=float)
    finite = np.isfinite(p_values)
    if not finite.any():
        return q
    p = p_values[finite]
    order = np.argsort(p)
    ranked = p[order]
    n = ranked.size
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(adjusted, 0.0, 1.0)
    q[finite] = out
    return q


def _boltzmann_mean(values: np.ndarray, kt: float) -> float:
    """``-kT ln <exp(-ddG/kT)>`` computed with log-sum-exp for stability."""
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(-kt * (spspecial.logsumexp(-values / kt) - np.log(values.size)))


def aggregate(
    raw: pd.DataFrame,
    *,
    alpha: float = 0.05,
    kt: float = DEFAULT_KT,
    stabilizing_cutoff: float = -0.5,
    strain_cutoff: float = 1.0,
) -> pd.DataFrame:
    """Collapse the per-conformer table into per-substitution statistics.

    Args:
        raw: Output of :func:`tmol_mutscan.scan.run_scan`.
        alpha: FDR threshold for the ``significant`` column.
        kt: kT used for the Boltzmann average, in REU.
        stabilizing_cutoff: ddG below which a significant substitution is
            called stabilizing, in REU.
        strain_cutoff: ``strain_mean`` above which an apparently stabilizing
            substitution is flagged as paying hidden intramolecular cost.
    """
    log = get_logger()
    usable = usable_rows(raw)
    dropped = len(raw) - len(usable)
    if dropped:
        log.warning("%d of %d rows dropped (skipped or non-finite)", dropped, len(raw))
    if usable.empty:
        raise SystemExit("no usable rows: every mutation was skipped or non-finite")

    keys = ["position", "chain", "resnum", "icode", "wt_aa", "wt_name3", "mut_aa", "mut_name3"]
    records = []
    for key, group in usable.groupby(keys, sort=False, dropna=False):
        ddg = group.ddg.to_numpy(dtype=float)
        n = ddg.size
        sd = float(np.std(ddg, ddof=1)) if n > 1 else float("nan")
        mean = float(np.mean(ddg))
        sem = sd / np.sqrt(n) if n > 1 and np.isfinite(sd) else float("nan")

        if n > 1 and np.isfinite(sd) and sd > 1e-12:
            t_stat, p_value = sps.ttest_1samp(ddg, 0.0)
            half = sps.t.ppf(1 - 0.05 / 2, n - 1) * sem
            ci_low, ci_high = mean - half, mean + half
            dz = mean / sd
        else:
            t_stat = p_value = ci_low = ci_high = dz = float("nan")

        q1, q3 = (np.percentile(ddg, [25, 75]) if n > 1 else (np.nan, np.nan))
        record = dict(zip(keys, key))
        record.update(
            n_conformers=n,
            ddg_mean=mean,
            ddg_sd=sd,
            ddg_sem=sem,
            ci95_low=float(ci_low),
            ci95_high=float(ci_high),
            ddg_median=float(np.median(ddg)),
            ddg_iqr=float(q3 - q1) if n > 1 else float("nan"),
            ddg_min=float(np.min(ddg)),
            ddg_max=float(np.max(ddg)),
            ddg_boltzmann=_boltzmann_mean(ddg, kt),
            cohens_dz=float(dz),
            frac_stabilizing=float(np.mean(ddg < 0.0)),
            t_stat=float(t_stat),
            p_value=float(p_value),
            e_wt_mean=float(np.mean(group.e_wt)),
            e_wt_sd=float(np.std(group.e_wt, ddof=1)) if n > 1 else float("nan"),
            e_mut_mean=float(np.mean(group.e_mut)),
            e_mut_sd=float(np.std(group.e_mut, ddof=1)) if n > 1 else float("nan"),
            ratio_mean=float(np.nanmean(group.ratio)),
            percentage_mean=float(np.nanmean(group.percentage)),
            # v0.2.0: whole-pose change and its non-interface component.
            ddg_total_mean=float(np.nanmean(group.ddg_total))
            if "ddg_total" in group
            else float("nan"),
            ddg_total_sd=float(np.nanstd(group.ddg_total, ddof=1))
            if "ddg_total" in group and n > 1
            else float("nan"),
            strain_mean=float(np.nanmean(group.strain))
            if "strain" in group
            else float("nan"),
            strain_sd=float(np.nanstd(group.strain, ddof=1))
            if "strain" in group and n > 1
            else float("nan"),
            ddg_ref_mean=float(np.nanmean(group.ddg_ref))
            if "ddg_ref" in group
            else float("nan"),
            is_wt_self=bool(key[keys.index("mut_name3")] == key[keys.index("wt_name3")]),
        )
        records.append(record)

    stats = pd.DataFrame.from_records(records)

    # FDR over the real substitutions only; WT->WT is zero by construction.
    p_for_fdr = stats.p_value.to_numpy(dtype=float).copy()
    p_for_fdr[stats.is_wt_self.to_numpy()] = np.nan
    stats["q_value"] = benjamini_hochberg(p_for_fdr)
    stats["significant"] = (stats.q_value < alpha).fillna(False)
    stats["effect"] = np.select(
        [
            stats.significant & (stats.ddg_mean <= stabilizing_cutoff),
            stats.significant & (stats.ddg_mean >= -stabilizing_cutoff),
        ],
        ["stabilizing", "destabilizing"],
        default="neutral",
    )
    stats.loc[stats.is_wt_self, "effect"] = "wild-type"

    # A candidate that improves the interface while straining the molecule is
    # buying contacts with internal cost that the interface sum cannot see.
    stats["strain_flag"] = (
        (stats.ddg_mean < 0) & (stats.strain_mean > strain_cutoff)
    ).fillna(False)

    stats = stats.sort_values(["chain", "resnum", "icode", "ddg_mean"]).reset_index(drop=True)
    stats.insert(0, "rank", stats.ddg_mean.rank(method="min").astype("Int64"))

    n_flagged = int((stats.strain_flag & ~stats.is_wt_self).sum())
    if n_flagged:
        log.warning(
            "%d apparently stabilizing substitution(s) carry intramolecular strain "
            "above %.2f REU; see the strain_mean column and "
            "plots/interface_vs_total.png",
            n_flagged,
            strain_cutoff,
        )
    n_sig = int(stats.significant.sum())
    log.info(
        "statistics: %d substitutions, %d significant at FDR %.3g "
        "(%d stabilizing, %d destabilizing)",
        len(stats),
        n_sig,
        alpha,
        int((stats.effect == "stabilizing").sum()),
        int((stats.effect == "destabilizing").sum()),
    )
    return stats


def position_summary(stats: pd.DataFrame, contributions: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per scanned position: sensitivity and the best substitution found."""
    real = stats[~stats.is_wt_self]
    rows = []
    for key, group in real.groupby(["position", "chain", "resnum", "icode", "wt_aa"], sort=False):
        if not np.isfinite(group.ddg_mean).any():
            continue
        best = group.loc[group.ddg_mean.idxmin()]
        stabilizing = group[group.effect == "stabilizing"]
        rows.append(
            {
                "position": key[0],
                "chain": key[1],
                "resnum": key[2],
                "icode": key[3],
                "wt_aa": key[4],
                "n_substitutions": len(group),
                "mean_ddg_all_substitutions": float(group.ddg_mean.mean()),
                "median_ddg_all_substitutions": float(group.ddg_mean.median()),
                "sd_across_substitutions": float(group.ddg_mean.std(ddof=1)),
                "best_substitution": best.mut_aa,
                "best_ddg_mean": float(best.ddg_mean),
                "best_ddg_sem": float(best.ddg_sem),
                "best_q_value": float(best.q_value),
                "n_significant_stabilizing": len(stabilizing),
                "n_significant_destabilizing": int((group.effect == "destabilizing").sum()),
                "stabilizing_substitutions": "".join(
                    stabilizing.sort_values("ddg_mean").mut_aa.tolist()
                ),
            }
        )
    summary = pd.DataFrame(rows)
    if contributions is not None and not contributions.empty:
        summary = summary.merge(contributions, on="position", how="left")
    return summary.sort_values(["chain", "resnum", "icode"]).reset_index(drop=True)


def convergence(raw: pd.DataFrame) -> pd.DataFrame:
    """How the mean ddG matrix settles as conformers accumulate.

    For each prefix of ``k`` conformers the running mean of every substitution
    is compared with the mean over the full ensemble. ``mean_abs_drift`` is the
    average absolute difference in REU; when it has flattened near zero the
    ensemble is large enough, and when it is still falling at ``k = n`` more
    conformers are needed.
    """
    usable = usable_rows(raw)
    if usable.empty:
        return pd.DataFrame()
    wide = usable.pivot_table(
        index="conformer", columns=["position", "mut_name3"], values="ddg"
    ).sort_index()
    final = wide.mean(axis=0)
    rows = []
    for k in range(1, len(wide) + 1):
        running = wide.iloc[:k].mean(axis=0)
        drift = (running - final).abs()
        sem = (
            wide.iloc[:k].std(axis=0, ddof=1) / np.sqrt(k) if k > 1 else pd.Series(np.nan, index=final.index)
        )
        rows.append(
            {
                "n_conformers": k,
                "mean_abs_drift": float(drift.mean()),
                "median_abs_drift": float(drift.median()),
                "max_abs_drift": float(drift.max()),
                "mean_sem": float(sem.mean()) if k > 1 else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def wt_interface_contributions(
    ensemble: PoseStack,
    sfxn: ScoreFunction,
    positions: list[Residue],
    partner_a: torch.Tensor,
    partner_b: torch.Tensor,
) -> pd.DataFrame:
    """Each scanned residue's own share of the wild-type interface energy.

    Summed over both block-pair orientations against every block of the
    opposite partner, averaged across the ensemble. This is a hotspot map that
    needs no mutations: a residue with a large negative contribution is already
    doing the binding, independently of what substituting it would do.
    """
    log = get_logger()
    scorer = sfxn.render_block_pair_scoring_module(ensemble, interaction_only=True)
    with torch.no_grad():
        matrix = scorer(ensemble.coords, sum_terms=True, apply_weights=True)
    matrix = matrix.detach()
    if matrix.dim() == 4:  # a leading score-type axis survived
        matrix = matrix.sum(dim=0)

    a_row = partner_a[0].detach()
    b_row = partner_b[0].detach()
    rows = []
    for residue in positions:
        block = residue.block
        opposite = b_row if bool(a_row[block]) else a_row
        cols = torch.nonzero(opposite).flatten()
        if cols.numel() == 0:
            continue
        per_conformer = (
            matrix[:, block, :][:, cols].sum(dim=1) + matrix[:, :, block][:, cols].sum(dim=1)
        )
        values = per_conformer.cpu().numpy()
        rows.append(
            {
                "position": residue.label,
                "wt_interface_contribution": float(np.mean(values)),
                "wt_interface_contribution_sd": float(np.std(values, ddof=1))
                if values.size > 1
                else float("nan"),
            }
        )
    log.info("computed wild-type interface contributions for %d positions", len(rows))
    return pd.DataFrame(rows)


def matrices(stats: pd.DataFrame, mutation_order: list[str]) -> dict[str, pd.DataFrame]:
    """Position x substitution matrices, rows ordered by chain and number."""
    order = [c for c in mutation_order if c in set(stats.mut_aa)]
    index = (
        stats[["position", "chain", "resnum", "icode", "wt_aa"]]
        .drop_duplicates()
        .sort_values(["chain", "resnum", "icode"])
    )
    labels = [f"{p} ({w})" for p, w in zip(index.position, index.wt_aa)]
    out = {}
    for name, column in [
        ("ddg_mean", "ddg_mean"),
        ("ddg_sd", "ddg_sd"),
        ("ddg_sem", "ddg_sem"),
        ("ddg_median", "ddg_median"),
        ("ddg_boltzmann", "ddg_boltzmann"),
        ("qvalue", "q_value"),
        ("frac_stabilizing", "frac_stabilizing"),
        ("cohens_dz", "cohens_dz"),
    ]:
        table = stats.pivot_table(index="position", columns="mut_aa", values=column, dropna=False)
        table = table.reindex(index=index.position, columns=order)
        table.index = labels
        table.index.name = "position (wt)"
        out[name] = table
    return out


def write_stats(
    stats: pd.DataFrame,
    summary: pd.DataFrame,
    conv: pd.DataFrame,
    mats: dict[str, pd.DataFrame],
    outdir: Path,
    *,
    top_n: int = 25,
) -> None:
    """Write every statistics table and matrix."""
    log = get_logger()
    stats_dir = outdir / "stats"
    matrix_dir = outdir / "matrices"
    stats_dir.mkdir(parents=True, exist_ok=True)
    matrix_dir.mkdir(parents=True, exist_ok=True)

    stats.to_csv(stats_dir / "ddg_statistics.csv", index=False)
    summary.to_csv(stats_dir / "position_summary.csv", index=False)
    if not conv.empty:
        conv.to_csv(stats_dir / "convergence.csv", index=False)

    best = (
        stats[~stats.is_wt_self]
        .sort_values("ddg_mean")
        .head(top_n)[
            [
                "position",
                "wt_aa",
                "mut_aa",
                "ddg_mean",
                "ddg_sd",
                "ddg_sem",
                "ci95_low",
                "ci95_high",
                "ddg_boltzmann",
                "frac_stabilizing",
                "cohens_dz",
                "q_value",
                "effect",
            ]
        ]
    )
    best.to_csv(stats_dir / "top_mutations.csv", index=False)

    for name, table in mats.items():
        table.to_csv(matrix_dir / f"{name}.csv")

    log.info("wrote statistics tables to %s and matrices to %s", stats_dir, matrix_dir)
