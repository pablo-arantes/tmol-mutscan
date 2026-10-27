"""Structure loading, residue bookkeeping and selection parsing.

Selections are written as ``+``-joined terms, each one of:

===========================  ====================================================
``*``                        every real residue
``A``                        every residue of chain A (``AB`` means chains A and B)
``chain:A``                  the same, written explicitly
``A:12-30``                  residues 12 to 30 of chain A, PDB (author) numbering
``A:5,9,14-18``              an explicit list and/or ranges
``A:52A``                    residue 52 insertion code A; a bare ``52`` matches
                             residue 52 whatever its insertion code
``resname:BEN``              every residue whose 3-letter name is BEN
``ligand``                   every non-polymer residue (ligand, ion, cofactor)
``protein``                  every amino-acid residue
``nucleic``                  every nucleic-acid residue
``interface``                residues close to the opposite partner (scan only)
===========================  ====================================================

For example ``--scan 'A:17-29'`` scans a 13-residue peptide, and
``--partner-b 'ligand'`` puts every non-polymer residue in partner B.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from tmol.chemical import three2one
from tmol.database import ParameterDatabase
from tmol.io import atom_array_from_file, pose_stack_from_atom_array, pose_stack_from_file
from tmol.pose import PoseStack, get_ligand_mask, get_polymer_type_mask
from tmol.score import beta2016_score_function

from .logging_utils import get_logger

LIGAND_FORMATS = {".cif", ".mmcif", ".sdf", ".mol2"}
_RANGE_RE = re.compile(r"^(-?\d+)\s*-\s*(-?\d+)$")
_SINGLE_RE = re.compile(r"^(-?\d+)([A-Za-z]?)$")


# --------------------------------------------------------------------------- #
# device
# --------------------------------------------------------------------------- #
def resolve_device(spec: str) -> torch.device:
    """Turn a ``--device`` string into a torch device, checking availability.

    TMol's compiled kernels target CPU and CUDA only; Apple MPS is not a
    supported backend, so ``auto`` never selects it.
    """
    log = get_logger()
    spec = spec.lower()
    if spec == "auto":
        spec = "cuda" if torch.cuda.is_available() else "cpu"
        log.info("device auto-selected: %s", spec)
    if spec.startswith("cuda"):
        if not torch.cuda.is_available():
            raise SystemExit(
                "CUDA was requested but torch.cuda.is_available() is False. Install a "
                "CUDA-enabled PyTorch plus the matching TMol operators "
                "(pip install 'tmol[cu12]' or 'tmol[cu13]'), or use --device cpu."
            )
        import tmol

        if not tmol.has_cuda():
            raise SystemExit(
                "PyTorch sees a GPU but TMol was built without CUDA operators. "
                "Install pip install 'tmol[cu12]' (or [cu13]) matching your CUDA major."
            )
        return torch.device(spec)
    if spec == "mps":
        raise SystemExit("TMol has no MPS backend. Use --device cpu on Apple Silicon.")
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
@dataclass
class LoadedStructure:
    """A single-pose :class:`PoseStack` plus the database it must be scored with.

    ``context`` is TMol's :class:`PoseBuildContext` from ligand preparation.
    Passing it back to :func:`pose_stack_from_atom_array` when reloading
    conformers reuses the prepared chemistry instead of re-deriving it, which
    also guarantees every pose shares one parameter database.
    """

    pose: PoseStack
    param_db: ParameterDatabase
    path: Path
    prepared_ligands: bool
    context: object | None = None

    def score_function(self):
        """Score function built from this structure's own parameter database."""
        return beta2016_score_function(self.pose.device, param_db=self.param_db)


def load_structure(
    path: str | Path,
    device: torch.device,
    *,
    prepare_ligands: bool | None = None,
    model: int = 1,
) -> LoadedStructure:
    """Read a PDB/mmCIF/SDF/MOL2 file into a one-pose PoseStack.

    Ligand preparation (chemistry typing, bond orders, charges) extends the
    parameter database, so the returned database -- not the default one -- must
    be used to build the score function. It is skipped for plain ``.pdb``
    input unless asked for, because PDB files carry no bond records.

    Args:
        path: Structure file.
        device: Torch device for the pose tensors.
        prepare_ligands: Force ligand preparation on or off. ``None`` enables
            it for formats that carry bonds (mmCIF, SDF, MOL2).
        model: Model number for multi-model files.
    """
    log = get_logger()
    path = Path(path)
    if not path.is_file():
        raise SystemExit(f"input structure not found: {path}")

    base_db = ParameterDatabase.get_default()
    if prepare_ligands is None:
        prepare_ligands = path.suffix.lower() in LIGAND_FORMATS

    if not prepare_ligands:
        log.info("loading %s (no ligand preparation)", path.name)
        pose = pose_stack_from_file(str(path), device, model=model, param_db=base_db)
        return LoadedStructure(pose, base_db, path, False, None)

    log.info("loading %s with ligand preparation (this is the slow part of loading)", path.name)
    array = atom_array_from_file(str(path), model=model)
    pose, context = pose_stack_from_atom_array(
        array, device, prepare_ligands=True, param_db=base_db, return_context=True
    )
    return LoadedStructure(pose, context.parameter_database, path, True, context)


# --------------------------------------------------------------------------- #
# residue bookkeeping
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Residue:
    """One real block of pose 0, with its PDB identity."""

    block: int
    chain: str
    resnum: int
    icode: str
    name3: str
    is_protein: bool
    is_ligand: bool

    @property
    def one_letter(self) -> str:
        """One-letter code, or ``X`` for anything non-canonical."""
        return three2one(self.name3) or "X"

    @property
    def label(self) -> str:
        """Compact ``A:23`` / ``A:52A`` identifier used in output tables."""
        return f"{self.chain}:{self.resnum}{self.icode}"


def residue_table(pose: PoseStack, pose_index: int = 0) -> list[Residue]:
    """Describe every real block of one pose, in block order."""
    info = pose.pdb_info
    real = pose.block_type_ind64[pose_index].detach().cpu().numpy() >= 0
    protein = get_polymer_type_mask(pose, "amino_acid")[pose_index].detach().cpu().numpy()
    ligand = get_ligand_mask(pose)[pose_index].detach().cpu().numpy()

    out: list[Residue] = []
    for block in np.flatnonzero(real):
        icode = str(info.residue_insertion_codes[pose_index, block] or "").strip()
        out.append(
            Residue(
                block=int(block),
                chain=str(info.chain_labels[pose_index, block]),
                resnum=int(info.residue_labels[pose_index, block]),
                icode=icode,
                name3=pose.block_type(pose_index, int(block)).name3,
                is_protein=bool(protein[block]),
                is_ligand=bool(ligand[block]),
            )
        )
    return out


def describe_structure(pose: PoseStack) -> str:
    """Multi-line human summary: chains, residue ranges, composition."""
    residues = residue_table(pose)
    lines = [
        f"{len(residues)} residues, {int(pose.real_atoms[0].sum())} atoms, "
        f"{pose.max_n_blocks} blocks allocated"
    ]
    by_chain: dict[str, list[Residue]] = {}
    for res in residues:
        by_chain.setdefault(res.chain, []).append(res)
    for chain, group in sorted(by_chain.items()):
        kinds = []
        n_prot = sum(r.is_protein for r in group)
        n_lig = sum(r.is_ligand for r in group)
        n_other = len(group) - n_prot - n_lig
        if n_prot:
            kinds.append(f"{n_prot} amino acid")
        if n_lig:
            kinds.append(f"{n_lig} ligand/ion")
        if n_other:
            kinds.append(f"{n_other} other")
        nums = [r.resnum for r in group]
        seq = "".join(r.one_letter for r in group if r.is_protein)
        lines.append(
            f"  chain {chain!r}: {len(group)} residues "
            f"({', '.join(kinds)}), numbering {min(nums)}-{max(nums)}"
        )
        if n_lig:
            lig = sorted({r.name3 for r in group if r.is_ligand})
            lines.append(f"      ligand/ion residue names: {', '.join(lig)}")
        if seq and n_prot <= 60:
            lines.append(f"      sequence: {seq}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# selections
# --------------------------------------------------------------------------- #
def _residue_numbers(spec: str) -> list[tuple[int, str]] | None:
    """Parse ``5,9,14-18`` into (number, insertion-code) pairs; ``*`` -> None."""
    spec = spec.strip()
    if spec in {"*", ""}:
        return None
    wanted: list[tuple[int, str]] = []
    for piece in spec.split(","):
        piece = piece.strip()
        if not piece:
            continue
        span = _RANGE_RE.match(piece)
        if span:
            lo, hi = int(span.group(1)), int(span.group(2))
            if lo > hi:
                raise SystemExit(f"inverted residue range {piece!r}")
            wanted.extend((n, "") for n in range(lo, hi + 1))
            continue
        single = _SINGLE_RE.match(piece)
        if not single:
            raise SystemExit(f"cannot parse residue specifier {piece!r}")
        wanted.append((int(single.group(1)), single.group(2)))
    return wanted


def parse_selection(
    pose: PoseStack,
    expression: str,
    *,
    interface_against: torch.Tensor | None = None,
    interface_cutoff: float = 8.0,
    what: str = "selection",
) -> torch.Tensor:
    """Build a ``[n_poses, max_n_blocks]`` boolean mask from a selection string.

    See the module docstring for the syntax. The mask is broadcast to every
    pose in the stack, which is what an ensemble of the same complex needs.

    Args:
        pose: Pose stack the selection is resolved against (via pose 0).
        expression: ``+``-joined selection terms.
        interface_against: Partner mask required by the ``interface`` term.
        interface_cutoff: Heavy-atom distance cutoff for ``interface``, in A.
        what: Name used in error messages.
    """
    residues = residue_table(pose)
    by_block = {res.block: res for res in residues}
    chosen: set[int] = set()

    for raw_term in expression.split("+"):
        term = raw_term.strip()
        if not term:
            continue
        lowered = term.lower()

        if term == "*" or lowered == "all":
            chosen.update(by_block)
        elif lowered == "ligand":
            chosen.update(b for b, r in by_block.items() if r.is_ligand)
        elif lowered == "protein":
            chosen.update(b for b, r in by_block.items() if r.is_protein)
        elif lowered == "nucleic":
            na = get_polymer_type_mask(pose, "nucleic_acid")[0].detach().cpu().numpy()
            chosen.update(b for b in by_block if na[b])
        elif lowered == "interface":
            if interface_against is None:
                raise SystemExit(f"'interface' is not usable in {what}")
            chosen.update(
                interface_blocks(pose, interface_against, cutoff=interface_cutoff)
            )
        elif lowered.startswith("resname:"):
            names = {n.strip().upper() for n in term.split(":", 1)[1].split(",") if n.strip()}
            hit = {b for b, r in by_block.items() if r.name3.upper() in names}
            if not hit:
                raise SystemExit(
                    f"{what}: no residue named {sorted(names)} in the structure"
                )
            chosen.update(hit)
        else:
            if lowered.startswith("chain:"):
                term = term.split(":", 1)[1]
            chain_part, _, residue_part = term.partition(":")
            chain_part = chain_part.strip()
            known = {r.chain for r in residues}
            if chain_part in {"*", ""}:
                chains = set(known)
            elif "," in chain_part:
                chains = {c.strip() for c in chain_part.split(",") if c.strip()}
            elif chain_part in known:
                # a multi-character label such as mmCIF 'AA' is one chain
                chains = {chain_part}
            else:
                chains = set(chain_part)
            unknown = chains - known
            if unknown:
                raise SystemExit(
                    f"{what}: chain(s) {sorted(unknown)} not in the structure "
                    f"(available: {sorted(known)})"
                )
            wanted = _residue_numbers(residue_part)
            for block, res in by_block.items():
                if res.chain not in chains:
                    continue
                if wanted is None or (res.resnum, res.icode) in wanted:
                    chosen.add(block)
                elif any(n == res.resnum and code == "" for n, code in wanted):
                    # a bare number also matches that number's insertion codes
                    chosen.add(block)

    if not chosen:
        raise SystemExit(f"{what} {expression!r} selected no residues")

    mask = torch.zeros_like(pose.block_type_ind64, dtype=torch.bool)
    mask[:, torch.tensor(sorted(chosen), dtype=torch.long, device=pose.device)] = True
    mask &= pose.block_type_ind64 >= 0
    return mask


def interface_blocks(
    pose: PoseStack, other: torch.Tensor, *, cutoff: float = 8.0, pose_index: int = 0
) -> list[int]:
    """Blocks with any heavy atom within ``cutoff`` A of the ``other`` mask."""
    coords = pose.coords[pose_index].detach()
    real = pose.real_atoms[pose_index].detach()
    block_for_atom = _block_for_atom(pose, pose_index)

    other_blocks = torch.nonzero(other[pose_index]).flatten()
    is_other = torch.zeros_like(real)
    for block in other_blocks.tolist():
        is_other |= (block_for_atom == block) & real
    if not bool(is_other.any()):
        raise SystemExit("interface selection: the opposite partner has no atoms")

    other_xyz = coords[is_other]
    near_blocks: list[int] = []
    for block in torch.nonzero(pose.block_type_ind64[pose_index] >= 0).flatten().tolist():
        if other[pose_index, block]:
            continue
        sel = (block_for_atom == block) & real
        if not bool(sel.any()):
            continue
        dist = torch.cdist(coords[sel], other_xyz)
        if float(dist.min()) <= cutoff:
            near_blocks.append(int(block))
    return near_blocks


def _block_for_atom(pose: PoseStack, pose_index: int = 0) -> torch.Tensor:
    """Map each atom slot of one pose to its block index (-1 for padding)."""
    offsets = pose.block_coord_offset64[pose_index]
    n_atoms = pose.n_ats_per_block[pose_index]
    out = torch.full((pose.max_n_pose_atoms,), -1, dtype=torch.long, device=pose.device)
    for block in range(pose.max_n_blocks):
        count = int(n_atoms[block])
        if count <= 0:
            continue
        start = int(offsets[block])
        out[start : start + count] = block
    return out
