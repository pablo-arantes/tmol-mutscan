"""Conformational ensemble generation with batched TMol FastRelax.

The PyRosetta original built its ensemble with a BackrubMover driven by a
GenericMonteCarloMover, then ran FastRelax on each accepted structure, one
structure at a time. TMol has no backrub mover, and does not need one: its
packer anneal is stochastic, so relaxing N copies of the same input in a
single batched ``fast_relax`` call lands them in N different nearby minima.
That is one GPU call instead of N CPU trajectories.

Two knobs control how wide the ensemble is:

``--n-conformers``
    How many copies to relax together.
``--perturb SIGMA``
    Gaussian noise (A, per coordinate) added to every real atom before
    relaxing. Zero relies on packer stochasticity alone and gives a tight
    ensemble (~0.4 A RMSD) of fully relaxed structures. Raising it widens the
    ensemble (~1.5 A RMSD at 0.2 A) but strains bond geometry, so it needs
    more ``--relax-repeats`` to re-converge. Always check the score spread in
    ``ensemble/ensemble_scores.csv`` before trusting a perturbed ensemble.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import torch

from tmol.database import ParameterDatabase
from tmol.io import (
    atom_array_from_file,
    file_from_pose_stack,
    pose_stack_from_atom_array,
    pose_stack_from_file,
)
from tmol.kinematics import CartesianMoveMap, FoldForest
from tmol.ops import calculate_block_pair_ddg
from tmol.pack import PackerPalette
from tmol.pose import PoseStack, PoseStackBuilder
from tmol.relax import fast_relax
from tmol.score import ScoreFunction

from .logging_utils import get_logger


def _perturb(pose_stack: PoseStack, sigma: float, seed: int, keep_first: bool) -> PoseStack:
    """Add Gaussian coordinate noise to every real atom of each pose."""
    out = pose_stack.clone_sharing_topology()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(out.coords.shape, generator=generator).to(out.device) * sigma
    noise[~out.real_atoms] = 0.0
    if keep_first:
        noise[0] = 0.0
    with torch.no_grad():
        out.coords.add_(noise)
    return out


def generate_ensemble(
    pose: PoseStack,
    sfxn: ScoreFunction,
    *,
    n_conformers: int,
    seed: int = 0,
    perturb: float = 0.0,
    relax_repeats: int = 2,
    batch_size: int = 0,
    keep_input_conformer: bool = True,
    verbose: bool = False,
) -> PoseStack:
    """Relax ``n_conformers`` copies of ``pose`` into one PoseStack.

    Args:
        pose: Single-pose input stack.
        sfxn: Score function built from the pose's parameter database.
        n_conformers: Number of ensemble members.
        seed: Base RNG seed; chunk ``k`` uses ``seed + k``.
        perturb: Gaussian coordinate noise sigma in A, applied before relax.
        relax_repeats: ``num_repeats`` for FastRelax (full pack/minimise ramps).
        batch_size: Conformers relaxed per batch; 0 relaxes all at once.
        keep_input_conformer: Leave the first copy unperturbed, so conformer 1
            is the plain relaxed input.
        verbose: Pass TMol's own stage timings through.

    Returns:
        A PoseStack with ``n_conformers`` relaxed poses.
    """
    log = get_logger()
    if pose.n_poses != 1:
        raise ValueError("generate_ensemble expects a single-pose PoseStack")
    if n_conformers < 1:
        raise ValueError("--n-conformers must be at least 1")

    chunk = n_conformers if batch_size in (0, None) else min(batch_size, n_conformers)
    relaxed_chunks: list[PoseStack] = []
    done = 0
    started = time.perf_counter()

    while done < n_conformers:
        size = min(chunk, n_conformers - done)
        batch = PoseStackBuilder.from_poses([pose] * size, pose.device)
        if perturb > 0.0:
            batch = _perturb(
                batch, perturb, seed + done, keep_first=keep_input_conformer and done == 0
            )
        log.info(
            "FastRelax batch %d-%d of %d (num_repeats=%d%s)",
            done + 1,
            done + size,
            n_conformers,
            relax_repeats,
            f", perturb sigma={perturb} A" if perturb > 0 else "",
        )
        t0 = time.perf_counter()
        torch.manual_seed(seed + done)
        relaxed = fast_relax(
            batch,
            sfxn,
            PackerPalette(),
            CartesianMoveMap(),
            FoldForest.reasonable_fold_forest(batch),
            num_repeats=relax_repeats,
            verbose=verbose,
        )
        elapsed = time.perf_counter() - t0
        log.info("  %d conformers in %.1fs (%.1fs/conformer)", size, elapsed, elapsed / size)
        relaxed_chunks.extend(relaxed.split(i) for i in range(size))
        done += size

    ensemble = PoseStackBuilder.from_poses(relaxed_chunks, pose.device)
    log.info(
        "ensemble of %d conformers built in %.1fs",
        ensemble.n_poses,
        time.perf_counter() - started,
    )
    return ensemble


def write_ensemble(
    ensemble: PoseStack,
    outdir: Path,
    sfxn: ScoreFunction,
    partner_a: torch.Tensor,
    partner_b: torch.Tensor,
    *,
    structure_format: str = "pdb",
) -> pd.DataFrame:
    """Write one PDB per conformer and a table of per-conformer energies.

    The table is the ensemble's quality control: a wide spread in
    ``total_score`` means the members are not equivalently relaxed, and the
    spread in ``interface_energy`` is the conformational component of the
    error bars the scan will report.
    """
    log = get_logger()
    outdir.mkdir(parents=True, exist_ok=True)

    scorer = sfxn.render_whole_pose_scoring_module(ensemble)
    with torch.no_grad():
        totals = scorer(ensemble.coords).detach().cpu().numpy()
        interface = (
            calculate_block_pair_ddg(
                ensemble, partner_a, partner_b, sfxn=sfxn, minimize=False, pack=False
            )
            .detach()
            .cpu()
            .numpy()
        )

    rows = []
    width = max(3, len(str(ensemble.n_poses)))
    for i in range(ensemble.n_poses):
        name = f"conformer_{i + 1:0{width}d}.{structure_format}"
        file_from_pose_stack(ensemble.split(i), str(outdir / name))
        rows.append(
            {
                "conformer": i + 1,
                "pdb_file": name,
                "total_score": float(totals[i]),
                "interface_energy": float(interface[i]),
            }
        )

    table = pd.DataFrame(rows)
    table.to_csv(outdir / "ensemble_scores.csv", index=False)
    log.info(
        "wrote %d conformer %s files to %s",
        ensemble.n_poses,
        structure_format.upper(),
        outdir,
    )
    log.info(
        "  total score     %.2f +/- %.2f REU (min %.2f, max %.2f)",
        table.total_score.mean(),
        table.total_score.std(ddof=1) if len(table) > 1 else 0.0,
        table.total_score.min(),
        table.total_score.max(),
    )
    log.info(
        "  interface energy %.2f +/- %.2f REU",
        table.interface_energy.mean(),
        table.interface_energy.std(ddof=1) if len(table) > 1 else 0.0,
    )
    return table


def load_ensemble(
    directory: Path,
    device: torch.device,
    param_db: ParameterDatabase,
    *,
    context: object | None = None,
) -> PoseStack:
    """Load every ``conformer_*`` structure in a directory into one PoseStack.

    When the ensemble was written from a ligand-prepared structure, pass the
    original :class:`PoseBuildContext` as ``context``: the conformers are then
    rebuilt against the same prepared chemistry and extended parameter
    database, which both skips re-preparation and keeps every pose in the stack
    mutually compatible.
    """
    log = get_logger()
    directory = Path(directory)
    files = sorted(
        [*directory.glob("conformer_*.pdb"), *directory.glob("conformer_*.cif")]
    )
    if not files:
        raise SystemExit(f"no conformer_* structures found in {directory}")
    log.info("loading %d conformers from %s", len(files), directory)
    if context is None:
        poses = [pose_stack_from_file(str(f), device, param_db=param_db) for f in files]
    else:
        poses = [
            pose_stack_from_atom_array(
                atom_array_from_file(str(f)), device, param_db=param_db, context=context
            )
            for f in files
        ]
    return PoseStackBuilder.from_poses(poses, device)
