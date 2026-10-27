"""Argument parsing, without running any science."""

from __future__ import annotations

import pytest

from tmol_mutscan.cli import build_parser


def test_run_requires_partners_and_scan():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "-i", "x.pdb"])


def test_run_defaults():
    args = build_parser().parse_args(
        ["run", "-i", "x.pdb", "--partner-a", "A", "--partner-b", "B", "--scan", "B"]
    )
    assert args.n_conformers == 20
    assert args.minimize is True       # minimisation is on by default
    assert args.mutations == "all"
    assert args.device == "auto"
    assert args.shell_gap == 5.0
    assert args.alpha == 0.05
    assert args.perturb == 0.0
    assert args.decompose is False


def test_no_minimize_flag():
    args = build_parser().parse_args(
        ["run", "-i", "x.pdb", "--partner-a", "A", "--partner-b", "B",
         "--scan", "B", "--no-minimize"]
    )
    assert args.minimize is False


def test_prepare_ligands_tristate():
    base = ["run", "-i", "x.cif", "--partner-a", "A", "--partner-b", "B", "--scan", "B"]
    parser = build_parser()
    assert parser.parse_args(base).prepare_ligands is None          # auto
    assert parser.parse_args([*base, "--prepare-ligands"]).prepare_ligands is True
    assert parser.parse_args([*base, "--no-prepare-ligands"]).prepare_ligands is False


def test_subcommands_exist():
    parser = build_parser()
    for command, extra in [
        ("info", ["-i", "x.pdb"]),
        ("ensemble", ["-i", "x.pdb", "--partner-a", "A", "--partner-b", "B"]),
        ("scan", ["-i", "x.pdb", "--ensemble", "d", "--partner-a", "A",
                  "--partner-b", "B", "--scan", "B"]),
        ("stats", ["--raw", "r.csv"]),
    ]:
        args = parser.parse_args([command, *extra])
        assert args.command == command
        assert callable(args.func)


def test_scan_requires_ensemble_directory():
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["scan", "-i", "x.pdb", "--partner-a", "A", "--partner-b", "B", "--scan", "B"]
        )
