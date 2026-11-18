"""The cross-release TMol compatibility layer.

These run against whichever TMol is installed, so they assert the contract the
rest of the package relies on rather than a specific release's layout.
"""

from __future__ import annotations

import inspect

import pytest

from tmol_mutscan import _compat


REQUIRED = [
    # core types
    "ParameterDatabase", "PoseStack", "PoseStackBuilder",
    # structure I/O
    "atom_array_from_file", "atom_array_from_pose_stack", "file_from_pose_stack",
    "pose_stack_from_atom_array", "pose_stack_from_file",
    "extended_pose_stack_from_sequences",
    # pose helpers
    "get_ligand_mask", "get_polymer_type_mask",
    # scoring
    "ScoreFunction", "beta2016_score_function",
    # packing
    "PackerPalette", "PackerTask", "pack_rotamers",
    "FixedAAChiSampler", "IncludeCurrentSampler",
    "create_dunbrack_sampler_from_database",
    # minimization / relax
    "run_cart_min", "fast_relax", "relax_poses",
    "CartesianMoveMap", "FoldForest",
    # batched ops
    "build_coord_mask_for_mask_and_interacting_atoms", "calculate_block_pair_ddg",
    "compute_block_adjacency", "compute_block_centroids_and_furthest_dist",
    "get_chain_mask",
]


@pytest.mark.parametrize("name", REQUIRED)
def test_every_required_symbol_resolves(name):
    assert hasattr(_compat, name), f"{name} did not resolve on this TMol install"
    assert getattr(_compat, name) is not None


def test_resolve_raises_a_useful_error_for_a_missing_symbol():
    with pytest.raises(ImportError) as excinfo:
        _compat._resolve("definitely_not_a_tmol_symbol", "tmol.pose")
    message = str(excinfo.value)
    assert "definitely_not_a_tmol_symbol" in message
    assert _compat.TMOL_VERSION in message      # tells the user what they have


def test_resolve_falls_back_to_a_legacy_module():
    # get_polymer_type_mask lives on tmol.pose, not the top level, in 1.x.
    assert _compat._resolve("get_polymer_type_mask", "tmol.pose") is not None


# --------------------------------------------------------------------------- #
# residue naming: owned here, since TMol 1.0.0 made its version private
# --------------------------------------------------------------------------- #
def test_residue_name_round_trip_covers_all_twenty():
    assert len(_compat.THREE_TO_ONE) == 20
    assert len(_compat.ONE_TO_THREE) == 20
    for name3, one in _compat.THREE_TO_ONE.items():
        assert _compat.three2one(name3) == one
        assert _compat.one2three(one) == name3


def test_residue_naming_is_case_and_space_tolerant():
    assert _compat.three2one(" ala ") == "A"
    assert _compat.one2three("a") == "ALA"


def test_residue_naming_returns_none_for_noncanonical():
    for bad in ("XYZ", "", "B", "MSE", "Z"):
        assert _compat.three2one(bad) is None or len(bad) == 1
    assert _compat.one2three("B") is None
    assert _compat.three2one("MSE") is None


# --------------------------------------------------------------------------- #
# relax wrapper
# --------------------------------------------------------------------------- #
def test_relax_poses_matches_the_installed_fast_relax_signature():
    """The wrapper must pass arguments the installed fast_relax accepts."""
    parameters = inspect.signature(_compat.fast_relax).parameters
    legacy = "fold_forest" in parameters
    # Whichever shape it is, these are the knobs the wrapper forwards.
    assert "num_repeats" in parameters
    assert "verbose" in parameters
    if not legacy:
        # TMol 1.x: the defaults must cover what 0.2.x passed explicitly.
        assert "minimizer" in parameters
        assert parameters["minimizer"].default is None
        assert "packer_palette" in parameters
        assert parameters["packer_palette"].default is None


def test_tmol_layout_reports_a_version():
    text = _compat.tmol_layout()
    assert "TMol" in text and _compat.TMOL_VERSION in text


# --------------------------------------------------------------------------- #
# macOS Metal guard
# --------------------------------------------------------------------------- #
def test_metal_guard_respects_an_explicit_setting(monkeypatch):
    monkeypatch.setenv("TMOL_MPS_OFFLOAD", "1")
    assert _compat.guard_macos_metal_offload() is None


def test_metal_guard_is_a_noop_off_darwin(monkeypatch):
    monkeypatch.delenv("TMOL_MPS_OFFLOAD", raising=False)
    monkeypatch.setattr(_compat.sys, "platform", "linux")
    assert _compat.guard_macos_metal_offload() is None
