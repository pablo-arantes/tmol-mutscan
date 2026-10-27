"""Batched saturation mutagenesis of an interface.

For every (conformer, position, substitution) the script builds one pose, packs
the mutated residue and its neighbour shell, and reads the cross-partner
interaction energy. The quantity reported is

    ddG = E_int(mutant) - E_int(wild-type control)

where ``E_int`` is the sum of weighted block-pair interactions between partner
A and partner B, and the wild-type control is an *independently packed* pose of
the same conformer with the same shell and the same protocol. Pairing each
mutant with its own conformer's control is what makes the ensemble average
meaningful: it cancels the conformer-to-conformer offset in absolute energy.

``E_int`` is the same quantity the PyRosetta original computed as
``bound - unbound``: separating two partners by a 100 A rigid-body translation
without repacking removes exactly the cross-partner terms, which is what
``calculate_block_pair_ddg`` sums directly. It needs no separated pose.

Work is submitted to TMol in chunks of poses. Within a chunk there is one
``pack_rotamers`` call and one scoring call, no matter how many mutations the
chunk holds, which is where the GPU speedup comes from.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tmol.chemical import one2three, three2one
from tmol.database import ParameterDatabase
from tmol.optimization import run_cart_min
from tmol.ops import (
    build_coord_mask_for_mask_and_interacting_atoms,
    calculate_block_pair_ddg,
    compute_block_adjacency,
    compute_block_centroids_and_furthest_dist,
)
from tmol.pack import PackerPalette, PackerTask, pack_rotamers
from tmol.pack.rotamer import FixedAAChiSampler, IncludeCurrentSampler
from tmol.pack.rotamer.dunbrack import create_dunbrack_sampler_from_database
from tmol.pose import PoseStack, PoseStackBuilder
from tmol.score import ScoreFunction

from .logging_utils import get_logger
from .structure import Residue

CANONICAL_AA = "ACDEFGHIKLMNPQRSTVWY"
"""The 20 canonical amino acids, one-letter, alphabetical."""

WT_CONTROL = "__WT__"
"""Sentinel in place of a substitution, marking the wild-type control pose."""


@dataclass(frozen=True)
class Request:
    """One pose to build: a conformer, a position, and a requested identity."""

    conformer: int
    residue: Residue
    name3: str  # requested 3-letter identity, or WT_CONTROL

    @property
    def is_control(self) -> bool:
        return self.name3 == WT_CONTROL

    @property
    def wanted(self) -> str:
        """The identity the packer must be restricted to."""
        return self.residue.name3 if self.is_control else self.name3


def parse_mutations(spec: str) -> list[str]:
    """Turn ``--mutations`` into an ordered list of 3-letter codes.

    Accepts a one-letter string (``"AVLI"``), a comma-separated one-letter list
    (``"A,V,L"``), comma-separated 3-letter codes (``"ALA,VAL"``), or the
    keywords ``all`` / ``canonical`` for all 20.
    """
    spec = spec.strip()
    if spec.lower() in {"all", "canonical", "20"}:
        return [one2three(aa) for aa in CANONICAL_AA]

    pieces = [p.strip() for p in spec.split(",") if p.strip()]
    if len(pieces) == 1 and len(pieces[0]) > 1:
        # "AVL" is three one-letter codes, "ALA" is one 3-letter code: a
        # 3-character token is only a residue name if it really names one.
        token = pieces[0].upper()
        if not (len(token) == 3 and three2one(token) is not None):
            pieces = list(pieces[0])

    out: list[str] = []
    for piece in pieces:
        if len(piece) == 1:
            name3 = one2three(piece.upper())
            if name3 is None:
                raise SystemExit(f"--mutations: {piece!r} is not an amino-acid letter")
        elif len(piece) == 3:
            name3 = piece.upper()
            if three2one(name3) is None:
                raise SystemExit(f"--mutations: {name3!r} is not a canonical residue name")
        else:
            raise SystemExit(f"--mutations: cannot parse {piece!r}")
        if name3 not in out:
            out.append(name3)
    return out


def build_requests(
    positions: list[Residue], mutations: list[str], n_conformers: int
) -> list[Request]:
    """Enumerate every pose the scan needs, control poses first per site."""
    requests: list[Request] = []
    for conformer in range(n_conformers):
        for residue in positions:
            requests.append(Request(conformer, residue, WT_CONTROL))
            for name3 in mutations:
                requests.append(Request(conformer, residue, name3))
    return requests


def auto_batch_size(pose: PoseStack, device: torch.device, budget_bytes: int = 2 << 30) -> int:
    """Poses per chunk that keep the block-pair score tensor near ``budget_bytes``.

    The block-pair scorer materialises ``[n_terms, n_poses, n_blocks, n_blocks]``
    floats, which is the largest allocation in the scan and grows with the
    square of the system size.
    """
    n_blocks = max(1, pose.max_n_blocks)
    per_pose = 25 * n_blocks * n_blocks * 4
    size = max(1, budget_bytes // per_pose)
    return int(min(size, 256 if device.type == "cuda" else 64))


def _allowed_name3s_by_block(task: PackerTask, blocks: set[int]) -> dict[int, set[str]]:
    """Identities the palette offers at each of ``blocks``, before restriction.

    Every pose in a chunk is the same complex, and this is read before any
    pose-specific restriction is applied, so the answer depends only on the
    block index. Resolving it once per block instead of once per request turns
    thousands of small device-to-host transfers into two.
    """
    index = torch.tensor(sorted(blocks), dtype=torch.long)
    considered = task.per_block_considered_block_types[0, index].cpu().numpy()
    allowed = task.per_block_is_block_type_allowed[0, index].cpu().numpy()
    types = task.pbt.active_block_types
    out: dict[int, set[str]] = {}
    for row, block in enumerate(sorted(blocks)):
        out[block] = {
            types[int(t)].name3
            for t, ok in zip(considered[row], allowed[row])
            if t >= 0 and ok
        }
    return out


def _run_chunk(
    ensemble: PoseStack,
    chunk: list[Request],
    sfxn: ScoreFunction,
    param_db: ParameterDatabase,
    partner_a: torch.Tensor,
    partner_b: torch.Tensor,
    *,
    shell_gap: float,
    minimize: bool,
    seed: int,
    decompose: bool,
    memory_efficient: bool,
) -> tuple[np.ndarray, np.ndarray | None, list[str], list[str]]:
    """Build, pack, optionally minimise and score one chunk of requests.

    Returns the per-request interaction energies, the optional per-term
    breakdown, the identity each pose actually ended up with, and a parallel
    list of skip reasons (empty string when the request was honoured).
    """
    log = get_logger()
    device = ensemble.device
    n = len(chunk)

    batch = PoseStackBuilder.from_poses(
        [ensemble.split(request.conformer) for request in chunk], device
    )

    # Target = the residue being substituted in each pose.
    target = torch.zeros_like(batch.block_type_ind64, dtype=torch.bool)
    for i, request in enumerate(chunk):
        target[i, request.residue.block] = True

    # Shell = target plus every block whose enclosing sphere is within
    # shell_gap of it. Poses of the same conformer/position share coordinates
    # before packing, so a mutant and its control always get the same shell.
    centroids, radii = compute_block_centroids_and_furthest_dist(batch)
    adjacency = compute_block_adjacency(centroids, radii, shell_gap)
    shell = target | (target.unsqueeze(2) & adjacency).any(dim=1)

    task = PackerTask(batch, PackerPalette())

    # Snapshot what each target position can become before any restriction.
    offered = _allowed_name3s_by_block(task, {r.residue.block for r in chunk})
    skip_reason = [""] * n
    for i, request in enumerate(chunk):
        available = offered[request.residue.block]
        if request.wanted not in available:
            skip_reason[i] = (
                f"{request.wanted} not offered at {request.residue.label} "
                f"({request.residue.name3}); the palette allows "
                f"{','.join(sorted(available)) or 'nothing'}"
            )

    task.disable_packing_by_block_mask(~shell)
    task.restrict_to_repacking(~target)

    # One restrict_absent_name3s call per distinct identity rather than per
    # request: the masks are disjoint and the operation only removes choices,
    # so the composition is exact and the call count stays at ~21 per chunk.
    masks_by_name3: dict[str, torch.Tensor] = {}
    for i, request in enumerate(chunk):
        name3 = request.residue.name3 if skip_reason[i] else request.wanted
        mask = masks_by_name3.get(name3)
        if mask is None:
            mask = torch.zeros_like(target)
            masks_by_name3[name3] = mask
        mask[i, request.residue.block] = True
    for name3, mask in masks_by_name3.items():
        task.restrict_absent_name3s({name3}, mask)

    for sampler in (
        create_dunbrack_sampler_from_database(param_db, device),
        FixedAAChiSampler(),
        IncludeCurrentSampler(),
    ):
        task.add_conformer_sampler(sampler)

    t0 = time.perf_counter()
    packed = pack_rotamers(batch, sfxn, task, seed=seed)
    t_pack = time.perf_counter() - t0

    if minimize:
        t0 = time.perf_counter()
        coord_mask = build_coord_mask_for_mask_and_interacting_atoms(packed, target)
        packed = run_cart_min(packed, sfxn, coord_mask)
        log.debug("  chunk minimisation %.2fs", time.perf_counter() - t0)

    t0 = time.perf_counter()
    with torch.no_grad():
        per_term = calculate_block_pair_ddg(
            packed,
            partner_a[:n],
            partner_b[:n],
            sfxn=sfxn,
            sum_terms=False,
            minimize=False,
            pack=False,
            memory_efficient=memory_efficient,
        )
    energies = per_term.sum(dim=0).detach().cpu().numpy()
    t_score = time.perf_counter() - t0
    log.debug("  chunk pack %.2fs, score %.2fs", t_pack, t_score)

    achieved = [packed.block_type(i, chunk[i].residue.block).name3 for i in range(n)]
    terms = per_term.detach().cpu().numpy() if decompose else None
    return energies, terms, achieved, skip_reason


def run_scan(
    ensemble: PoseStack,
    positions: list[Residue],
    mutations: list[str],
    sfxn: ScoreFunction,
    param_db: ParameterDatabase,
    partner_a_mask: torch.Tensor,
    partner_b_mask: torch.Tensor,
    *,
    shell_gap: float = 5.0,
    minimize: bool = False,
    seed: int = 0,
    batch_size: int = 0,
    decompose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Run the full ensemble x position x substitution scan.

    Returns the tidy per-conformer table and, when ``decompose`` is set, a
    second table of per-score-term ddG contributions.
    """
    log = get_logger()
    device = ensemble.device
    n_conformers = ensemble.n_poses

    requests = build_requests(positions, mutations, n_conformers)
    chunk_size = batch_size or auto_batch_size(ensemble, device)
    memory_efficient = ensemble.max_n_blocks > 400
    n_chunks = (len(requests) + chunk_size - 1) // chunk_size

    log.info(
        "scan: %d conformers x %d positions x (%d substitutions + 1 WT control) "
        "= %d poses",
        n_conformers,
        len(positions),
        len(mutations),
        len(requests),
    )
    log.info("  %d chunks of up to %d poses each", n_chunks, chunk_size)
    if minimize:
        log.info("  Cartesian minimisation of the mutated shell is ON")

    # Partner masks are per-pose; build them once at the chunk width.
    a_row = partner_a_mask[0:1]
    b_row = partner_b_mask[0:1]
    partner_a = a_row.expand(chunk_size, -1).contiguous()
    partner_b = b_row.expand(chunk_size, -1).contiguous()

    term_names = [st.name for st in sfxn.all_score_types()]
    energies = np.full(len(requests), np.nan)
    achieved: list[str] = [""] * len(requests)
    skipped: list[str] = [""] * len(requests)
    term_rows: list[np.ndarray] = []

    started = time.perf_counter()
    for index in range(n_chunks):
        lo = index * chunk_size
        hi = min(lo + chunk_size, len(requests))
        chunk = requests[lo:hi]
        chunk_energies, chunk_terms, chunk_achieved, chunk_skip = _run_chunk(
            ensemble,
            chunk,
            sfxn,
            param_db,
            partner_a,
            partner_b,
            shell_gap=shell_gap,
            minimize=minimize,
            seed=seed + index,
            decompose=decompose,
            memory_efficient=memory_efficient,
        )
        energies[lo:hi] = chunk_energies
        achieved[lo:hi] = chunk_achieved
        skipped[lo:hi] = chunk_skip
        if chunk_terms is not None:
            term_rows.append(chunk_terms.T)  # [n_poses, n_terms]

        done = hi
        elapsed = time.perf_counter() - started
        rate = done / elapsed
        log.info(
            "  chunk %d/%d done (%d/%d poses, %.1f poses/s, eta %.1f min)",
            index + 1,
            n_chunks,
            done,
            len(requests),
            rate,
            (len(requests) - done) / rate / 60 if rate > 0 else float("nan"),
        )

    total = time.perf_counter() - started
    log.info(
        "scan finished: %d poses in %.1f min (%.2f s/pose)",
        len(requests),
        total / 60,
        total / max(1, len(requests)),
    )

    for reason in sorted({r for r in skipped if r}):
        log.warning("skipped: %s", reason)

    # Interaction energy of each conformer/position wild-type control.
    control = {
        (r.conformer, r.residue.block): energies[i]
        for i, r in enumerate(requests)
        if r.is_control
    }

    rows = []
    for i, request in enumerate(requests):
        if request.is_control:
            continue
        residue = request.residue
        e_wt = control[(request.conformer, residue.block)]
        e_mut = energies[i]
        mutant_letter = three2one(request.name3) or "X"
        ratio = e_mut / e_wt if e_wt not in (0.0,) and np.isfinite(e_wt) else np.nan
        rows.append(
            {
                "conformer": request.conformer + 1,
                "position": residue.label,
                "chain": residue.chain,
                "resnum": residue.resnum,
                "icode": residue.icode,
                "wt_aa": residue.one_letter,
                "wt_name3": residue.name3,
                "mut_aa": mutant_letter,
                "mut_name3": request.name3,
                "e_wt": e_wt,
                "e_mut": e_mut,
                "ddg": e_mut - e_wt,
                "ratio": ratio,
                "percentage": (e_mut - e_wt) / e_wt
                if e_wt not in (0.0,) and np.isfinite(e_wt)
                else np.nan,
                "achieved_name3": achieved[i],
                "skipped": skipped[i],
            }
        )

    raw = pd.DataFrame(rows)
    _report_identity_mismatches(raw, log)

    terms = None
    if term_rows:
        stacked = np.concatenate(term_rows, axis=0)
        term_frame = pd.DataFrame(stacked, columns=term_names)
        term_frame.insert(0, "is_control", [r.is_control for r in requests])
        term_frame.insert(0, "mut_name3", [r.name3 for r in requests])
        term_frame.insert(0, "position", [r.residue.label for r in requests])
        term_frame.insert(0, "conformer", [r.conformer + 1 for r in requests])
        terms = _term_ddgs(term_frame, term_names)

    return raw, terms


def _report_identity_mismatches(raw: pd.DataFrame, log) -> None:
    """Warn when the packer did not produce the requested identity."""
    if raw.empty:
        return
    honoured = raw.skipped == ""
    wrong = raw[honoured & (raw.achieved_name3 != raw.mut_name3)]
    if not wrong.empty:
        log.warning(
            "%d of %d mutant poses did not reach the requested identity; "
            "their rows are dropped",
            len(wrong),
            int(honoured.sum()),
        )
    wt_check = raw[(raw.mut_name3 == raw.wt_name3) & honoured]
    if not wt_check.empty:
        worst = wt_check.ddg.abs().max()
        log.info(
            "wild-type self-consistency (WT->WT ddG): max |ddG| = %.3f REU over %d rows",
            worst,
            len(wt_check),
        )


def _term_ddgs(term_frame: pd.DataFrame, term_names: list[str]) -> pd.DataFrame:
    """Convert per-term interaction energies into per-term ddGs."""
    controls = term_frame[term_frame.is_control].set_index(["conformer", "position"])
    mutants = term_frame[~term_frame.is_control]
    base = controls[term_names]
    keys = list(zip(mutants.conformer, mutants.position))
    aligned = base.loc[keys].to_numpy()
    out = mutants[["conformer", "position", "mut_name3"]].reset_index(drop=True)
    deltas = pd.DataFrame(
        mutants[term_names].to_numpy() - aligned, columns=[f"ddg_{t}" for t in term_names]
    )
    return pd.concat([out, deltas], axis=1)


def write_raw(raw: pd.DataFrame, terms: pd.DataFrame | None, outdir: Path) -> None:
    """Persist the per-conformer tables."""
    log = get_logger()
    outdir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(outdir / "per_conformer_ddg.csv", index=False)
    log.info("wrote %s (%d rows)", outdir / "per_conformer_ddg.csv", len(raw))
    if terms is not None:
        terms.to_csv(outdir / "per_conformer_terms.csv", index=False)
        log.info("wrote %s", outdir / "per_conformer_terms.csv")
