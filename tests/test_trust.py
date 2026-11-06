"""The v0.3.0 burial classification and trust rule."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from tmol.io import extended_pose_stack_from_sequences

from tmol_mutscan.interface import (
    BURIAL_THRESHOLD,
    MAX_ASA,
    TRUST_DDG_CUTOFF,
    assign_trust,
    classify_interface,
    expected_accuracy,
)
from tmol_mutscan.stats import aggregate


# --------------------------------------------------------------------------- #
# the rule itself
# --------------------------------------------------------------------------- #
def test_strain_flag_always_wins():
    # Flagged predictions were anti-predictive in benchmarking, so nothing
    # else should be able to promote them.
    assert assign_trust("core", -9.0, True) == "none"
    assign_ = assign_trust("core", -9.0, False)
    assert assign_ == "high"


def test_high_trust_needs_core_and_effect_size():
    assert assign_trust("core", -3.0, False) == "high"
    assert assign_trust("core", 3.0, False) == "high"      # sign-agnostic
    assert assign_trust("core", -1.0, False) == "low"      # too small
    assert assign_trust("rim", -9.0, False) == "low"       # wrong location
    assert assign_trust("support", -9.0, False) == "low"
    assert assign_trust("?", -9.0, False) == "low"         # unknown burial


def test_trust_cutoff_is_respected():
    assert assign_trust("core", -1.6, False, cutoff=1.5) == "high"
    assert assign_trust("core", -1.6, False, cutoff=2.0) == "low"


def test_non_finite_ddg_is_not_trusted():
    assert assign_trust("core", float("nan"), False) == "low"


def test_expected_accuracy_decreases_with_size():
    bands = [expected_accuracy(n) for n in (200, 400, 600, 900, 5000)]
    values = [float(b.split("~")[1]) for b in bands]
    assert values == sorted(values, reverse=True)
    assert values[0] > values[-1]


def test_defaults_match_the_benchmarked_rule():
    assert TRUST_DDG_CUTOFF == 2.0
    assert BURIAL_THRESHOLD == 25.0
    assert len(MAX_ASA) == 20          # one reference per canonical residue


# --------------------------------------------------------------------------- #
# classification on a real structure
# --------------------------------------------------------------------------- #
def test_classify_interface_labels_every_block():
    pose = extended_pose_stack_from_sequences(["AAKLWG:DEFTY"], torch.device("cpu"))
    chain_a = torch.zeros_like(pose.block_type_ind64, dtype=torch.bool)
    chain_a[0, :6] = True
    chain_b = torch.zeros_like(pose.block_type_ind64, dtype=torch.bool)
    chain_b[0, 6:11] = True

    labels = classify_interface(pose, chain_a, chain_b)
    assert len(labels) == 11
    allowed = {"core", "support", "rim", "interior", "surface", "?"}
    assert set(labels.values()) <= allowed
    # An extended, non-contacting pair of chains has no buried interface.
    assert "core" not in set(labels.values())


def test_classify_interface_never_raises_on_bad_input():
    """A failed SASA must degrade to '?' rather than abort a scan."""
    pose = extended_pose_stack_from_sequences(["AAKLWG"], torch.device("cpu"))
    empty = torch.zeros_like(pose.block_type_ind64, dtype=torch.bool)
    labels = classify_interface(pose, empty, empty)
    assert set(labels.values()) == {"?"}


# --------------------------------------------------------------------------- #
# integration with aggregate()
# --------------------------------------------------------------------------- #
def _raw(ddg, n=6, position="B:23", strain=0.0):
    rows = []
    for conformer in range(1, n + 1):
        rows.append({
            "conformer": conformer, "position": position, "chain": "B",
            "resnum": 23, "icode": "", "wt_aa": "W", "wt_name3": "TRP",
            "mut_aa": "A", "mut_name3": "ALA",
            "e_wt": -40.0, "e_mut": -40.0 + ddg, "ddg": ddg + 0.01 * conformer,
            "total_wt": -300.0, "total_mut": -300.0 + ddg + strain,
            "ddg_total": ddg + strain, "ddg_ref": 0.0, "strain": strain,
            "ratio": 1.0, "percentage": 0.0, "achieved_name3": "ALA", "skipped": "",
        })
    return pd.DataFrame(rows)


def test_aggregate_adds_location_and_trust():
    stats = aggregate(_raw(-3.0), locations={"B:23": "core"})
    assert "location" in stats.columns and "trust" in stats.columns
    row = stats.iloc[0]
    assert row.location == "core"
    assert row.trust == "high"


def test_aggregate_without_locations_cannot_reach_high():
    stats = aggregate(_raw(-3.0))
    assert stats.iloc[0].location == "?"
    assert stats.iloc[0].trust == "low"


def test_aggregate_marks_strained_predictions_untrusted():
    stats = aggregate(_raw(-3.0, strain=5.0), locations={"B:23": "core"})
    row = stats.iloc[0]
    assert row.strain_flag
    assert row.trust == "none"


def test_trust_cutoff_flows_through_aggregate():
    raw = _raw(-1.6)
    assert aggregate(raw, locations={"B:23": "core"}, trust_cutoff=1.5).iloc[0].trust == "high"
    assert aggregate(raw, locations={"B:23": "core"}, trust_cutoff=3.0).iloc[0].trust == "low"
