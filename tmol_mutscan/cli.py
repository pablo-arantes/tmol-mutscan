"""Command-line interface for tmol-mutscan."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import pandas as pd
import torch

from . import __version__
from .logging_utils import get_logger, setup_logging

EPILOG = """\
selection syntax (--partner-a, --partner-b, --scan)
  '*'                every residue
  'A'                chain A          'AB' or 'A,B'  chains A and B
  'A:17-29'          chain A residues 17 to 29 (PDB numbering)
  'A:5,9,14-18'      an explicit list and/or ranges
  'resname:BEN'      every residue named BEN (ligands, ions, cofactors)
  'ligand'           every non-polymer residue
  'protein'          every amino-acid residue
  'interface'        (--scan only) residues near the opposite partner
  terms join with '+', e.g. --scan 'A:10-20+A:45'

examples
  # peptide-protein: scan the whole peptide chain B against protein chain A
  tmol-mutscan run -i complex.pdb --partner-a A --partner-b B --scan B -o out

  # protein-protein: scan only the interface residues of chain A
  tmol-mutscan run -i dimer.pdb --partner-a A --partner-b B \\
      --scan interface -n 20 --device cuda -o out

  # protein-ligand: keep bonds by using mmCIF, put the ligand in partner B
  tmol-mutscan run -i complex.cif --partner-a protein --partner-b ligand \\
      --scan interface --minimize -o out

  # inspect chains and numbering before choosing selections
  tmol-mutscan info -i complex.pdb
"""


# --------------------------------------------------------------------------- #
# argument groups
# --------------------------------------------------------------------------- #
def _add_common(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("runtime")
    group.add_argument(
        "--device",
        default="auto",
        help="auto (CUDA when available), cpu, cuda, or cuda:N. TMol has no MPS backend.",
    )
    group.add_argument("--seed", type=int, default=0, help="base RNG seed (default: 0)")
    group.add_argument(
        "--threads",
        type=int,
        default=0,
        help="torch CPU threads; 0 leaves the torch default",
    )
    group.add_argument("-v", "--verbose", action="count", default=0, help="debug logging")


def _add_io(parser: argparse.ArgumentParser, *, need_input: bool = True) -> None:
    group = parser.add_argument_group("input/output")
    group.add_argument(
        "-i",
        "--input",
        required=need_input,
        metavar="FILE",
        help="input complex: .pdb, .cif/.mmcif, .sdf or .mol2. Use mmCIF for "
        "ligands, since PDB files carry no bond records.",
    )
    group.add_argument(
        "-o", "--outdir", default="mutscan_out", metavar="DIR", help="output directory"
    )
    group.add_argument(
        "--prepare-ligands",
        dest="prepare_ligands",
        action="store_true",
        default=None,
        help="force ligand chemistry preparation (default: on for cif/sdf/mol2)",
    )
    group.add_argument(
        "--no-prepare-ligands",
        dest="prepare_ligands",
        action="store_false",
        help="skip ligand preparation even for cif/sdf/mol2",
    )
    group.add_argument("--model", type=int, default=1, help="model number for multi-model files")


def _add_partners(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("binding partners")
    group.add_argument(
        "--partner-a",
        required=True,
        metavar="SEL",
        help="first binding partner, e.g. 'A' or 'protein'",
    )
    group.add_argument(
        "--partner-b",
        required=True,
        metavar="SEL",
        help="second binding partner, e.g. 'B' or 'ligand'",
    )


def _add_ensemble(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("ensemble generation")
    group.add_argument(
        "-n",
        "--n-conformers",
        type=int,
        default=20,
        help="ensemble members relaxed in one batch (default: 20)",
    )
    group.add_argument(
        "--relax-repeats",
        type=int,
        default=2,
        help="FastRelax pack/minimise ramps per conformer (default: 2)",
    )
    group.add_argument(
        "--perturb",
        type=float,
        default=0.0,
        help="Gaussian coordinate noise sigma in A before relaxing. 0 (default) "
        "relies on packer stochasticity and gives a tight, fully relaxed "
        "ensemble; 0.2 widens it but needs more --relax-repeats.",
    )
    group.add_argument(
        "--relax-batch-size",
        type=int,
        default=0,
        help="conformers per FastRelax call; 0 relaxes all at once",
    )


def _add_scan(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("mutational scan")
    group.add_argument(
        "--scan",
        required=True,
        metavar="SEL",
        help="positions to mutate, e.g. 'B', 'A:17-29' or 'interface'",
    )
    group.add_argument(
        "--mutations",
        default="all",
        help="substitutions to try: 'all' for the 20 canonical, or a one-letter "
        "string ('AVLIF'), or 3-letter codes ('ALA,TRP') (default: all)",
    )
    group.add_argument(
        "--shell-gap",
        type=float,
        default=5.0,
        help="repack neighbours whose enclosing sphere is within this gap of the "
        "mutated residue, in A (default: 5.0)",
    )
    group.add_argument(
        "--minimize",
        dest="minimize",
        action="store_true",
        default=True,
        help="Cartesian-minimise the mutated residue and its interacting atoms "
        "after packing (DEFAULT). Costs roughly 3x the scan time and is worth "
        "it: without it, a large substitution into a tight pocket cannot "
        "relieve its strain and reports a clash artefact of tens or hundreds "
        "of REU instead of a usable ddG.",
    )
    group.add_argument(
        "--no-minimize",
        dest="minimize",
        action="store_false",
        help="pack only, without minimisation. Faster, and reproduces the "
        "original PyRosetta protocol, but expect large positive ddG outliers "
        "wherever a substitution clashes.",
    )
    group.add_argument(
        "--interface-cutoff",
        type=float,
        default=8.0,
        help="heavy-atom distance defining 'interface' in --scan, in A (default: 8.0)",
    )
    group.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="mutant poses scored per chunk; 0 picks a size from the system size",
    )
    group.add_argument(
        "--decompose",
        action="store_true",
        help="also record the per-score-term breakdown of every ddG",
    )


def _add_stats(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("statistics and figures")
    group.add_argument(
        "--alpha", type=float, default=0.05, help="FDR threshold (default: 0.05)"
    )
    group.add_argument(
        "--kt",
        type=float,
        default=0.593,
        help="kT for the Boltzmann-averaged ddG, in REU (default: 0.593, i.e. 298 K)",
    )
    group.add_argument(
        "--stabilizing-cutoff",
        type=float,
        default=-0.5,
        help="ddG below which a significant substitution is called stabilizing "
        "(default: -0.5 REU)",
    )
    group.add_argument(
        "--top-n", type=int, default=25, help="substitutions in the top-hits outputs"
    )
    group.add_argument(
        "--annotate", action="store_true", help="print ddG values inside heatmap cells"
    )
    group.add_argument("--no-plots", action="store_true", help="skip figure generation")


# --------------------------------------------------------------------------- #
# manifest
# --------------------------------------------------------------------------- #
def write_manifest(args: argparse.Namespace, outdir: Path, extra: dict) -> None:
    """Record everything needed to reproduce or report the run.

    The TMol interface-analysis guidance asks for the exact computational
    experiment to be reported: score function, input, partner masks, packing
    shell, allowed identities, samplers, device and settings. This file is that
    record.
    """
    import numpy
    import scipy

    import tmol

    manifest = {
        "tmol_mutscan_version": __version__,
        "command": " ".join(sys.argv),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        # 'func' is the argparse subcommand callback; its repr carries a memory
        # address, which is noise in a record meant to be reproducible.
        "arguments": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
            if key != "func"
        },
        "score_function": "beta2016",
        "rotamer_samplers": ["Dunbrack", "FixedAAChi", "IncludeCurrent"],
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "tmol": tmol.__version__,
            "tmol_has_cuda": bool(tmol.has_cuda()),
            "torch": torch.__version__,
            "torch_cuda_available": bool(torch.cuda.is_available()),
            "cuda_device": torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None,
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
            "pandas": pd.__version__,
            "torch_num_threads": torch.get_num_threads(),
        },
        **extra,
    }
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    get_logger().info("wrote %s", outdir / "run_manifest.json")


# --------------------------------------------------------------------------- #
# shared setup
# --------------------------------------------------------------------------- #
def _prepare(args: argparse.Namespace, log_name: str):
    """Set up logging, threads and the device; load the input structure."""
    from .structure import load_structure, resolve_device

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    log = setup_logging(args.verbose, outdir / log_name)
    log.info("tmol-mutscan %s", __version__)

    if args.threads > 0:
        torch.set_num_threads(args.threads)
        log.info("torch CPU threads: %d", args.threads)

    device = resolve_device(args.device)
    structure = load_structure(
        args.input, device, prepare_ligands=args.prepare_ligands, model=args.model
    )
    return log, outdir, device, structure


def _resolve_partners(structure, args, log):
    """Build the two partner masks and sanity-check them."""
    from .structure import parse_selection

    partner_a = parse_selection(structure.pose, args.partner_a, what="--partner-a")
    partner_b = parse_selection(structure.pose, args.partner_b, what="--partner-b")

    overlap = int((partner_a & partner_b).sum())
    if overlap:
        raise SystemExit(
            f"--partner-a and --partner-b share {overlap} residues; the interface "
            "energy between overlapping sets is not meaningful. Make them disjoint."
        )
    log.info(
        "partner A: %d residues (%s) | partner B: %d residues (%s)",
        int(partner_a[0].sum()),
        args.partner_a,
        int(partner_b[0].sum()),
        args.partner_b,
    )
    return partner_a, partner_b


def _resolve_positions(structure, args, partner_a, partner_b, log):
    """Resolve --scan into a list of mutable protein residues."""
    from .structure import parse_selection, residue_table

    # 'interface' means "near the opposite partner", which differs for each
    # partner, so it is resolved twice and intersected with its own partner.
    if "interface" in args.scan.lower():
        scan_a = parse_selection(
            structure.pose,
            args.scan,
            interface_against=partner_b,
            interface_cutoff=args.interface_cutoff,
            what="--scan",
        )
        scan_b = parse_selection(
            structure.pose,
            args.scan,
            interface_against=partner_a,
            interface_cutoff=args.interface_cutoff,
            what="--scan",
        )
        scan = (scan_a & partner_a) | (scan_b & partner_b)
    else:
        scan = parse_selection(structure.pose, args.scan, what="--scan")

    blocks = set(torch.nonzero(scan[0]).flatten().tolist())
    table = {res.block: res for res in residue_table(structure.pose)}

    positions, rejected = [], []
    for block in sorted(blocks):
        residue = table[block]
        if not residue.is_protein:
            rejected.append(f"{residue.label} ({residue.name3}, not an amino acid)")
        elif not bool(partner_a[0, block] or partner_b[0, block]):
            rejected.append(f"{residue.label} (in neither partner)")
        else:
            positions.append(residue)

    for item in rejected:
        log.warning("not scanning %s", item)
    if not positions:
        raise SystemExit(
            f"--scan {args.scan!r} resolved to no mutable amino-acid residue inside "
            "either partner"
        )
    log.info(
        "scanning %d positions: %s",
        len(positions),
        ", ".join(f"{p.one_letter}{p.resnum}{p.icode}({p.chain})" for p in positions[:40])
        + (" ..." if len(positions) > 40 else ""),
    )
    return positions


def _expand_masks(mask: torch.Tensor, n: int) -> torch.Tensor:
    """Broadcast a one-row partner mask to ``n`` poses."""
    return mask[0:1].expand(n, -1).contiguous()


# --------------------------------------------------------------------------- #
# subcommands
# --------------------------------------------------------------------------- #
def cmd_info(args: argparse.Namespace) -> int:
    """Print chains, numbering and composition, then exit."""
    from .structure import describe_structure, load_structure, resolve_device

    setup_logging(args.verbose)
    device = resolve_device(args.device)
    structure = load_structure(
        args.input, device, prepare_ligands=args.prepare_ligands, model=args.model
    )
    print(f"\n{structure.path.name}")
    print(describe_structure(structure.pose))
    print(
        "\nUse the chain labels and numbering above in --partner-a / --partner-b / --scan."
    )
    if structure.prepared_ligands:
        print("Ligand chemistry was prepared; score with this structure's database.")
    return 0


def cmd_ensemble(args: argparse.Namespace) -> int:
    """Generate and write the relaxed conformational ensemble."""
    from .ensemble import generate_ensemble, write_ensemble

    log, outdir, device, structure = _prepare(args, "ensemble.log")
    partner_a, partner_b = _resolve_partners(structure, args, log)
    sfxn = structure.score_function()

    ensemble = generate_ensemble(
        structure.pose,
        sfxn,
        n_conformers=args.n_conformers,
        seed=args.seed,
        perturb=args.perturb,
        relax_repeats=args.relax_repeats,
        batch_size=args.relax_batch_size,
        verbose=args.verbose > 1,
    )
    table = write_ensemble(
        ensemble,
        outdir / "ensemble",
        sfxn,
        _expand_masks(partner_a, ensemble.n_poses),
        _expand_masks(partner_b, ensemble.n_poses),
        structure_format="cif" if structure.prepared_ligands else "pdb",
    )
    write_manifest(args, outdir, {"stage": "ensemble", "n_conformers": len(table)})
    return 0


def cmd_scan(args: argparse.Namespace) -> int:
    """Scan an existing ensemble and write statistics and figures."""
    from .ensemble import load_ensemble

    log, outdir, device, structure = _prepare(args, "scan.log")
    ensemble = load_ensemble(
        Path(args.ensemble), device, structure.param_db, context=structure.context
    )
    partner_a, partner_b = _resolve_partners(structure, args, log)
    return _scan_and_report(args, log, outdir, structure, ensemble, partner_a, partner_b)


def cmd_run(args: argparse.Namespace) -> int:
    """Full pipeline: ensemble, scan, statistics, figures."""
    from .ensemble import generate_ensemble, write_ensemble

    log, outdir, device, structure = _prepare(args, "mutscan.log")
    partner_a, partner_b = _resolve_partners(structure, args, log)
    sfxn = structure.score_function()

    ensemble_dir = outdir / "ensemble"
    if args.reuse_ensemble and any(ensemble_dir.glob("conformer_*")):
        from .ensemble import load_ensemble

        log.info("--reuse-ensemble: loading the existing ensemble in %s", ensemble_dir)
        ensemble = load_ensemble(
            ensemble_dir, device, structure.param_db, context=structure.context
        )
    else:
        ensemble = generate_ensemble(
            structure.pose,
            sfxn,
            n_conformers=args.n_conformers,
            seed=args.seed,
            perturb=args.perturb,
            relax_repeats=args.relax_repeats,
            batch_size=args.relax_batch_size,
            verbose=args.verbose > 1,
        )
        write_ensemble(
            ensemble,
            ensemble_dir,
            sfxn,
            _expand_masks(partner_a, ensemble.n_poses),
            _expand_masks(partner_b, ensemble.n_poses),
            structure_format="cif" if structure.prepared_ligands else "pdb",
        )
    return _scan_and_report(args, log, outdir, structure, ensemble, partner_a, partner_b)


def _scan_and_report(args, log, outdir, structure, ensemble, partner_a, partner_b) -> int:
    """Shared tail of ``scan`` and ``run``."""
    from . import plots
    from .scan import parse_mutations, run_scan, write_raw
    from .stats import (
        aggregate,
        convergence,
        matrices,
        position_summary,
        write_stats,
        wt_interface_contributions,
    )

    positions = _resolve_positions(structure, args, partner_a, partner_b, log)
    mutations = parse_mutations(args.mutations)
    log.info("substitutions (%d): %s", len(mutations), " ".join(mutations))

    sfxn = structure.score_function()
    n = ensemble.n_poses

    # The partner masks are resolved against --input, but the scan runs on the
    # ensemble. They agree for an ensemble built from that same input; a
    # mismatch means the --ensemble directory belongs to a different structure.
    if ensemble.max_n_blocks != partner_a.shape[1]:
        raise SystemExit(
            f"the ensemble has {ensemble.max_n_blocks} blocks per pose but "
            f"{Path(args.input).name} has {partner_a.shape[1]}. The --ensemble "
            "directory was built from a different structure, so the partner "
            "selections cannot be applied to it."
        )
    if any(
        int((ensemble.block_type_ind64[i] >= 0).sum())
        != int((structure.pose.block_type_ind64[0] >= 0).sum())
        for i in range(n)
    ):
        log.warning(
            "some conformers have a different residue count from --input; "
            "check that the ensemble matches the structure"
        )

    mask_a = _expand_masks(partner_a, n)
    mask_b = _expand_masks(partner_b, n)

    raw, terms = run_scan(
        ensemble,
        positions,
        mutations,
        sfxn,
        structure.param_db,
        mask_a,
        mask_b,
        shell_gap=args.shell_gap,
        minimize=args.minimize,
        seed=args.seed,
        batch_size=args.batch_size,
        decompose=args.decompose,
    )
    write_raw(raw, terms, outdir / "raw")

    stats = aggregate(
        raw, alpha=args.alpha, kt=args.kt, stabilizing_cutoff=args.stabilizing_cutoff
    )
    contributions = wt_interface_contributions(ensemble, sfxn, positions, mask_a, mask_b)
    summary = position_summary(stats, contributions)
    conv = convergence(raw)
    order = [c for c in "ACDEFGHIKLMNPQRSTVWY"]
    mats = matrices(stats, order)
    write_stats(stats, summary, conv, mats, outdir, top_n=args.top_n)

    if not args.no_plots:
        plots.make_all(
            stats,
            summary,
            conv,
            mats,
            raw,
            terms,
            outdir / "plots",
            alpha=args.alpha,
            annotate=args.annotate,
            top_n=args.top_n,
        )

    write_manifest(
        args,
        outdir,
        {
            "stage": "scan",
            "n_conformers": int(n),
            "n_positions": len(positions),
            "n_substitutions": len(mutations),
            "n_poses_built": int(n * len(positions) * (len(mutations) + 1)),
        },
    )
    _print_highlights(stats, summary, conv, outdir)
    return 0


def _print_highlights(stats, summary, conv, outdir: Path) -> None:
    """Short human-readable wrap-up on stdout."""
    real = stats[~stats.is_wt_self]
    best = real.sort_values("ddg_mean").head(10)
    print("\n" + "=" * 78)
    print("10 most favourable substitutions, ranked by mean ddG")
    print("ddG < 0 predicts tighter binding; ddG > 0 predicts weaker binding")
    print("=" * 78)
    print(f"{'mutation':<16}{'ddG':>8}{'SD':>8}{'95% CI':>18}{'q':>10}  effect")
    for row in best.itertuples():
        ci = f"[{row.ci95_low:6.2f},{row.ci95_high:6.2f}]"
        q = "n/a" if pd.isna(row.q_value) else f"{row.q_value:.2e}"
        print(
            f"{row.wt_aa}{row.resnum}{row.mut_aa} ({row.chain})".ljust(16)
            + f"{row.ddg_mean:8.2f}{row.ddg_sd:8.2f}{ci:>18}{q:>10}  {row.effect}"
        )
    if not conv.empty:
        final = conv.iloc[-1]
        # drift at n conformers is 0 by definition, so quote n-1.
        informative = conv.iloc[-2] if len(conv) > 1 else final
        print(
            f"\nensemble: {int(final.n_conformers)} conformers, "
            f"mean SEM {final.mean_sem:.3f} REU. Dropping to "
            f"{int(informative.n_conformers)} conformers would move the mean ddG by "
            f"{informative.mean_abs_drift:.3f} REU on average "
            f"({informative.max_abs_drift:.3f} REU at worst)."
        )
    print(f"\nall results in {outdir}")
    print(f"  tables  {outdir / 'stats'}")
    print(f"  figures {outdir / 'plots'}")


def cmd_stats(args: argparse.Namespace) -> int:
    """Recompute statistics and figures from an existing raw table."""
    from . import plots
    from .stats import aggregate, convergence, matrices, position_summary, write_stats

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    log = setup_logging(args.verbose, outdir / "stats.log")
    from .stats import normalize_raw

    raw = normalize_raw(pd.read_csv(args.raw))
    log.info("loaded %d rows from %s", len(raw), args.raw)

    stats = aggregate(
        raw, alpha=args.alpha, kt=args.kt, stabilizing_cutoff=args.stabilizing_cutoff
    )
    summary = position_summary(stats)
    log.info(
        "wild-type interface contributions need the ensemble, so "
        "position_hotspots.png is only produced by 'run' and 'scan'"
    )
    conv = convergence(raw)
    mats = matrices(stats, list("ACDEFGHIKLMNPQRSTVWY"))
    write_stats(stats, summary, conv, mats, outdir, top_n=args.top_n)

    write_manifest(
        args,
        outdir,
        {
            "stage": "stats",
            "source_table": str(args.raw),
            "n_rows": int(len(raw)),
            "n_conformers": int(raw.conformer.nunique()),
            "note": "re-aggregated from an existing raw table; no scoring was repeated",
        },
    )

    terms_path = Path(args.raw).with_name("per_conformer_terms.csv")
    terms = pd.read_csv(terms_path) if terms_path.is_file() else None
    if not args.no_plots:
        plots.make_all(
            stats,
            summary,
            conv,
            mats,
            raw,
            terms,
            outdir / "plots",
            alpha=args.alpha,
            annotate=args.annotate,
            top_n=args.top_n,
        )
    _print_highlights(stats, summary, conv, outdir)
    return 0


# --------------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tmol-mutscan",
        description=(
            "Ensemble-based interface mutational scanning with TMol. Builds a "
            "relaxed conformational ensemble, performs saturation mutagenesis at "
            "chosen positions, and reports interface ddG with full ensemble "
            "statistics. Works for peptide-protein, protein-protein and "
            "protein-ligand complexes."
        ),
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"tmol-mutscan {__version__}")
    subparsers = parser.add_subparsers(dest="command", metavar="SUBCOMMAND")

    info = subparsers.add_parser(
        "info",
        help="list chains, numbering and composition of a structure",
        description="Inspect a structure before choosing selections.",
    )
    _add_io(info)
    _add_common(info)
    info.set_defaults(func=cmd_info)

    ensemble = subparsers.add_parser(
        "ensemble",
        help="build the relaxed conformational ensemble only",
        description="Relax N copies of the input in one batched FastRelax call.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_io(ensemble)
    _add_partners(ensemble)
    _add_ensemble(ensemble)
    _add_common(ensemble)
    ensemble.set_defaults(func=cmd_ensemble)

    scan = subparsers.add_parser(
        "scan",
        help="scan an existing ensemble",
        description="Run the mutational scan over an ensemble built earlier.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_io(scan)
    scan.add_argument(
        "--ensemble",
        required=True,
        metavar="DIR",
        help="directory of conformer_* structures from the 'ensemble' subcommand",
    )
    _add_partners(scan)
    _add_scan(scan)
    _add_stats(scan)
    _add_common(scan)
    scan.set_defaults(func=cmd_scan)

    run = subparsers.add_parser(
        "run",
        help="full pipeline: ensemble, scan, statistics, figures",
        description="The usual entry point: everything in one command.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_io(run)
    _add_partners(run)
    _add_ensemble(run)
    _add_scan(run)
    _add_stats(run)
    run.add_argument(
        "--reuse-ensemble",
        action="store_true",
        help="reuse conformer_* files already in <outdir>/ensemble instead of relaxing again",
    )
    _add_common(run)
    run.set_defaults(func=cmd_run)

    stats = subparsers.add_parser(
        "stats",
        help="recompute statistics and figures from a raw table",
        description=(
            "Re-aggregate an existing raw/per_conformer_ddg.csv, for example to "
            "change --alpha or --kt without repeating the scan."
        ),
    )
    stats.add_argument(
        "--raw", required=True, metavar="CSV", help="raw/per_conformer_ddg.csv from a scan"
    )
    stats.add_argument("-o", "--outdir", default="mutscan_out", metavar="DIR")
    _add_stats(stats)
    stats.add_argument("-v", "--verbose", action="count", default=0)
    stats.set_defaults(func=cmd_stats)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 1
    # TMol's CPU kernels and OpenMP can collide with a second OpenMP runtime
    # loaded by conda-installed numerics; make the failure non-fatal.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    try:
        return args.func(args)
    except KeyboardInterrupt:
        get_logger().error("interrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
