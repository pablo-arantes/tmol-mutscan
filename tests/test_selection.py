"""Selection parsing and residue bookkeeping, on a synthetic two-chain pose."""

from __future__ import annotations

import pytest
import torch

from tmol.io import extended_pose_stack_from_sequences

from tmol_mutscan.structure import (
    describe_structure,
    interface_blocks,
    parse_selection,
    residue_table,
    resolve_device,
)


@pytest.fixture(scope="module")
def pose():
    """Chain A = AAKLWG (residues 1-6), chain B = DEFTY (residues 7-11)."""
    return extended_pose_stack_from_sequences(["AAKLWG:DEFTY"], torch.device("cpu"))


def test_residue_table(pose):
    residues = residue_table(pose)
    assert len(residues) == 11
    assert [r.chain for r in residues] == ["A"] * 6 + ["B"] * 5
    assert "".join(r.one_letter for r in residues) == "AAKLWGDEFTY"
    assert residues[0].label == "A:1"
    assert all(r.is_protein and not r.is_ligand for r in residues)


def test_chain_selection(pose):
    assert int(parse_selection(pose, "A").sum()) == 6
    assert int(parse_selection(pose, "B").sum()) == 5
    assert int(parse_selection(pose, "AB").sum()) == 11
    assert int(parse_selection(pose, "A,B").sum()) == 11
    assert int(parse_selection(pose, "*").sum()) == 11
    assert int(parse_selection(pose, "chain:A").sum()) == 6


def test_residue_range_selection(pose):
    assert int(parse_selection(pose, "A:2-4").sum()) == 3
    assert int(parse_selection(pose, "A:1,3,5").sum()) == 3
    assert int(parse_selection(pose, "A:1+B:11").sum()) == 2
    assert int(parse_selection(pose, "B:8-10").sum()) == 3


def test_keyword_selections(pose):
    assert int(parse_selection(pose, "protein").sum()) == 11
    with pytest.raises(SystemExit):
        parse_selection(pose, "ligand")  # no non-polymer residue present


def test_selection_errors(pose):
    with pytest.raises(SystemExit):
        parse_selection(pose, "Z")  # unknown chain
    with pytest.raises(SystemExit):
        parse_selection(pose, "A:99")  # no such residue
    with pytest.raises(SystemExit):
        parse_selection(pose, "A:4-2")  # inverted range
    with pytest.raises(SystemExit):
        parse_selection(pose, "resname:XYZ")
    with pytest.raises(SystemExit):
        parse_selection(pose, "interface")  # needs a partner


def test_selection_is_broadcast_over_poses(pose):
    mask = parse_selection(pose, "A")
    assert mask.shape == pose.block_type_ind64.shape
    assert mask.dtype == torch.bool


def test_interface_selection_is_a_subset(pose):
    chain_a = parse_selection(pose, "A")
    blocks = interface_blocks(pose, chain_a, cutoff=8.0)
    chain_b_blocks = set(torch.nonzero(parse_selection(pose, "B")[0]).flatten().tolist())
    # every interface block lies outside the queried partner
    assert set(blocks) <= chain_b_blocks


def test_describe_structure_mentions_both_chains(pose):
    text = describe_structure(pose)
    assert "chain 'A'" in text and "chain 'B'" in text
    assert "11 residues" in text


def test_resolve_device_rejects_mps():
    assert resolve_device("cpu").type == "cpu"
    with pytest.raises(SystemExit):
        resolve_device("mps")
