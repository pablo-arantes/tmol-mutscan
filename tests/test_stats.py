"""Statistics on synthetic ddG data, with no structure or scoring involved."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tmol_mutscan.scan import parse_mutations
from tmol_mutscan.stats import (
    aggregate,
    benjamini_hochberg,
    convergence,
    matrices,
    normalize_raw,
    position_summary,
    usable_rows,
)


# --------------------------------------------------------------------------- #
# --mutations parsing
# --------------------------------------------------------------------------- #
def test_parse_mutations_all():
    assert len(parse_mutations("all")) == 20
    assert parse_mutations("all")[0] == "ALA"
    assert parse_mutations("canonical") == parse_mutations("all")


def test_parse_mutations_forms():
    assert parse_mutations("AVL") == ["ALA", "VAL", "LEU"]
    assert parse_mutations("A,V,L") == ["ALA", "VAL", "LEU"]
    assert parse_mutations("ALA,TRP") == ["ALA", "TRP"]
    assert parse_mutations("ala,trp") == ["ALA", "TRP"]
    assert parse_mutations("A,A,V") == ["ALA", "VAL"]  # de-duplicated
    # a 2- or 3-character token is only a residue name if it really names one,
    # so "AL" is Ala+Leu while "ALA" is alanine
    assert parse_mutations("AL") == ["ALA", "LEU"]
    assert parse_mutations("AVL") == ["ALA", "VAL", "LEU"]


def test_parse_mutations_rejects_junk():
    for bad in ("B", "ZZZ", "A,ZZZ", "XYZ", "A,B"):
        with pytest.raises(SystemExit):
            parse_mutations(bad)


# --------------------------------------------------------------------------- #
# Benjamini-Hochberg
# --------------------------------------------------------------------------- #
def test_bh_is_monotone_and_bounded():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.74])
    q = benjamini_hochberg(p)
    assert np.all(q >= p - 1e-12)
    assert np.all(np.diff(q) >= -1e-12)
    assert np.all((q >= 0) & (q <= 1))


def test_bh_handles_nan():
    q = benjamini_hochberg(np.array([0.01, np.nan, 0.5]))
    assert np.isnan(q[1])
    assert np.isfinite(q[0]) and np.isfinite(q[2])


def test_bh_all_nan():
    assert np.all(np.isnan(benjamini_hochberg(np.array([np.nan, np.nan]))))


def test_bh_single_value_is_unchanged():
    assert benjamini_hochberg(np.array([0.03]))[0] == pytest.approx(0.03)


# --------------------------------------------------------------------------- #
# raw-table normalisation
# --------------------------------------------------------------------------- #
def _raw(n_conformers=5, ddg_by_mut=None, seed=0):
    """Synthetic raw table: one position, a few substitutions."""
    rng = np.random.default_rng(seed)
    ddg_by_mut = ddg_by_mut or {"ALA": 2.0, "TRP": -1.5, "ASP": 0.0}
    rows = []
    for conformer in range(1, n_conformers + 1):
        e_wt = -40.0 + rng.normal(0, 0.5)
        for name3, shift in ddg_by_mut.items():
            ddg = shift + rng.normal(0, 0.2)
            rows.append(
                {
                    "conformer": conformer,
                    "position": "B:23",
                    "chain": "B",
                    "resnum": 23,
                    "icode": "",
                    "wt_aa": "W",
                    "wt_name3": "TRP",
                    "mut_aa": {"ALA": "A", "TRP": "W", "ASP": "D"}[name3],
                    "mut_name3": name3,
                    "e_wt": e_wt,
                    "e_mut": e_wt + ddg,
                    "ddg": ddg,
                    "ratio": (e_wt + ddg) / e_wt,
                    "percentage": ddg / e_wt,
                    "achieved_name3": name3,
                    "skipped": "",
                }
            )
    return pd.DataFrame(rows)


def test_normalize_raw_recovers_csv_round_trip(tmp_path):
    raw = _raw()
    path = tmp_path / "raw.csv"
    raw.to_csv(path, index=False)
    # an empty 'skipped' string comes back as NaN; normalising must fix it
    reloaded = pd.read_csv(path)
    assert reloaded.skipped.isna().all()
    assert len(usable_rows(normalize_raw(reloaded))) == len(raw)


def test_usable_rows_filters_skipped_and_nonfinite():
    raw = _raw()
    raw.loc[0, "skipped"] = "PRO not offered"
    raw.loc[1, "ddg"] = np.inf
    raw.loc[2, "ddg"] = np.nan
    assert len(usable_rows(raw)) == len(raw) - 3


# --------------------------------------------------------------------------- #
# aggregate
# --------------------------------------------------------------------------- #
def test_aggregate_columns_and_values():
    stats = aggregate(_raw(n_conformers=8))
    assert len(stats) == 3
    for column in (
        "ddg_mean", "ddg_sd", "ddg_sem", "ci95_low", "ci95_high", "ddg_median",
        "ddg_iqr", "ddg_boltzmann", "cohens_dz", "frac_stabilizing", "t_stat",
        "p_value", "q_value", "significant", "effect", "n_conformers", "rank",
    ):
        assert column in stats.columns

    ala = stats[stats.mut_name3 == "ALA"].iloc[0]
    assert ala.ddg_mean == pytest.approx(2.0, abs=0.3)
    assert ala.n_conformers == 8
    assert ala.effect == "destabilizing"
    assert ala.frac_stabilizing == 0.0

    trp = stats[stats.mut_name3 == "TRP"].iloc[0]
    assert trp.is_wt_self  # TRP at a TRP position
    assert trp.effect == "wild-type"
    assert np.isnan(trp.q_value)  # excluded from the FDR correction


def test_aggregate_ci_brackets_the_mean():
    stats = aggregate(_raw(n_conformers=10))
    real = stats[~stats.is_wt_self]
    assert np.all(real.ci95_low <= real.ddg_mean)
    assert np.all(real.ddg_mean <= real.ci95_high)
    assert np.all(real.ddg_sem <= real.ddg_sd)


def test_boltzmann_average_never_exceeds_the_mean():
    # Jensen's inequality: -kT ln<exp(-x/kT)> <= <x>
    stats = aggregate(_raw(n_conformers=12, ddg_by_mut={"ALA": 3.0, "ASP": -2.0}))
    assert np.all(stats.ddg_boltzmann <= stats.ddg_mean + 1e-9)


def test_aggregate_single_conformer_is_graceful():
    stats = aggregate(_raw(n_conformers=1))
    assert np.all(stats.n_conformers == 1)
    assert np.all(np.isnan(stats.ddg_sd))
    assert not stats.significant.any()


def test_aggregate_raises_when_everything_is_skipped():
    raw = _raw()
    raw["skipped"] = "nope"
    with pytest.raises(SystemExit):
        aggregate(raw)


def test_alpha_controls_significance():
    raw = _raw(n_conformers=10)
    assert aggregate(raw, alpha=0.5).significant.sum() >= aggregate(
        raw, alpha=1e-12
    ).significant.sum()


# --------------------------------------------------------------------------- #
# derived tables
# --------------------------------------------------------------------------- #
def test_position_summary():
    summary = position_summary(aggregate(_raw(n_conformers=6)))
    assert len(summary) == 1
    row = summary.iloc[0]
    assert row.position == "B:23"
    assert row.wt_aa == "W"
    assert row.best_substitution == "D"  # ASP, the only non-WT negative shift
    assert row.n_substitutions == 2  # WT self-substitution excluded


def test_matrices_shape_and_order():
    stats = aggregate(_raw(n_conformers=6))
    mats = matrices(stats, list("ACDEFGHIKLMNPQRSTVWY"))
    assert set(mats) >= {"ddg_mean", "ddg_sd", "qvalue", "frac_stabilizing"}
    table = mats["ddg_mean"]
    assert list(table.index) == ["B:23 (W)"]
    assert list(table.columns) == ["A", "D", "W"]  # alphabetical, present only
    assert table.loc["B:23 (W)", "A"] == pytest.approx(2.0, abs=0.3)


def test_convergence_drift_vanishes_at_full_ensemble():
    conv = convergence(_raw(n_conformers=7))
    assert list(conv.n_conformers) == list(range(1, 8))
    assert conv.iloc[-1].mean_abs_drift == pytest.approx(0.0, abs=1e-12)
    assert np.isnan(conv.iloc[0].mean_sem)  # no SEM from one conformer
    assert conv.iloc[-1].mean_sem > 0


def test_convergence_empty_input():
    assert convergence(pd.DataFrame(columns=["conformer", "position", "mut_name3", "ddg", "skipped"])).empty
