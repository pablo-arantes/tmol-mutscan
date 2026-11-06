"""Interface burial classification, and the benchmark-derived trust rule.

Added in v0.3.0 because the SKEMPI 2.0 benchmark showed that this method's
accuracy depends strongly on *where* a residue sits in the interface, and on
whether the v0.2.0 strain diagnostic fired. Reporting a ddG without that
context invites it to be read as uniformly meaningful, which it is not.

Classification follows Levy (2010), which is also what SKEMPI annotates, using
relative solvent accessibility (rASA) computed with Biotite's Shrake-Rupley
implementation and the Tien et al. (2013) theoretical maximum ASA values:

=============  =========================================================
``core``       exposed in the free partner (rASA >= 25%) and buried in the
               complex (rASA < 25%) -- the residue the interface is made of
``support``    buried in the free partner already (rASA < 25%) and loses
               further area on binding
``rim``        at the interface but still solvent-exposed in the complex
``interior``   buried in the free partner and unaffected by binding
``surface``    exposed and unaffected by binding
=============  =========================================================

This implementation was checked against SKEMPI's own labels for the 496
benchmark mutations: 87.7% exact agreement, 88.7% on core vs non-core, and it
reproduces the trust rule's discriminating power to within noise (r = 0.585
with these labels against 0.594 with SKEMPI's).
"""

from __future__ import annotations

import warnings

import numpy as np
import torch

from tmol.io import atom_array_from_pose_stack
from tmol.pose import PoseStack

from .logging_utils import get_logger

# Tien et al. (2013), theoretical maximum ASA per residue, square angstroms.
MAX_ASA = {
    "ALA": 129.0, "ARG": 274.0, "ASN": 195.0, "ASP": 193.0, "CYS": 167.0,
    "GLN": 225.0, "GLU": 223.0, "GLY": 104.0, "HIS": 224.0, "ILE": 197.0,
    "LEU": 201.0, "LYS": 236.0, "MET": 224.0, "PHE": 240.0, "PRO": 159.0,
    "SER": 155.0, "THR": 172.0, "TRP": 285.0, "TYR": 263.0, "VAL": 174.0,
}

BURIAL_THRESHOLD = 25.0
"""rASA percentage below which a residue counts as buried (Levy 2010)."""

UNKNOWN = "?"
"""Label used when a residue's burial could not be determined."""


def _residue_rasa(array, point_number: int) -> dict[tuple[str, int, str], float]:
    """Relative ASA per residue, keyed by (chain, residue number, insertion code)."""
    import biotite.structure as struc

    atom_sasa = struc.sasa(array, point_number=point_number)
    starts = struc.get_residue_starts(array, add_exclusive_stop=True)
    out: dict[tuple[str, int, str], float] = {}
    for lo, hi in zip(starts[:-1], starts[1:]):
        name3 = str(array.res_name[lo])
        reference = MAX_ASA.get(name3)
        if reference is None:
            continue
        key = (
            str(array.chain_id[lo]),
            int(array.res_id[lo]),
            str(array.ins_code[lo]).strip(),
        )
        out[key] = 100.0 * float(np.nansum(atom_sasa[lo:hi])) / reference
    return out


def classify_interface(
    pose: PoseStack,
    partner_a: torch.Tensor,
    partner_b: torch.Tensor,
    *,
    pose_index: int = 0,
    point_number: int = 200,
) -> dict[int, str]:
    """Label every protein block of one pose by its role in the interface.

    Returns a mapping from block index to one of ``core``, ``support``,
    ``rim``, ``interior``, ``surface`` or ``?``. Failure is non-fatal: if the
    SASA calculation cannot be done, every block is labelled ``?`` and the
    caller falls back to a rule that does not need burial.
    """
    log = get_logger()
    from .structure import residue_table

    residues = residue_table(pose, pose_index)
    labels: dict[int, str] = {res.block: UNKNOWN for res in residues}

    try:
        import biotite.structure as struc

        single = pose.split(pose_index) if pose.n_poses > 1 else pose
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            array = atom_array_from_pose_stack(single)
        if hasattr(array, "stack_depth"):          # AtomArrayStack
            array = array[0]
        array = array[struc.filter_amino_acids(array)]
        if not array.array_length():
            raise ValueError("no amino-acid atoms to measure")

        a_blocks = set(torch.nonzero(partner_a[pose_index]).flatten().tolist())
        b_blocks = set(torch.nonzero(partner_b[pose_index]).flatten().tolist())
        by_key = {(r.chain, r.resnum, r.icode): r.block for r in residues}
        chains_a = {r.chain for r in residues if r.block in a_blocks}
        chains_b = {r.chain for r in residues if r.block in b_blocks}

        in_a = np.isin(array.chain_id, list(chains_a))
        in_b = np.isin(array.chain_id, list(chains_b))
        complex_array = array[in_a | in_b]
        if not complex_array.array_length():
            raise ValueError("partner chains matched no atoms")

        complex_rasa = _residue_rasa(complex_array, point_number)
        free_rasa: dict[tuple[str, int, str], float] = {}
        for mask in (in_a, in_b):
            part = array[mask]
            if part.array_length():
                free_rasa.update(_residue_rasa(part, point_number))

        for key, free in free_rasa.items():
            block = by_key.get(key)
            if block is None:
                continue
            bound = complex_rasa.get(key, free)
            buried_on_binding = (free - bound) > 1e-6
            if free < BURIAL_THRESHOLD:
                labels[block] = "support" if buried_on_binding else "interior"
            elif not buried_on_binding:
                labels[block] = "surface"
            elif bound >= BURIAL_THRESHOLD:
                labels[block] = "rim"
            else:
                labels[block] = "core"

        counts: dict[str, int] = {}
        for value in labels.values():
            counts[value] = counts.get(value, 0) + 1
        log.info(
            "interface burial: %s",
            ", ".join(f"{n} {k}" for k, n in sorted(counts.items(), key=lambda kv: -kv[1])),
        )
    except Exception as exc:  # noqa: BLE001 - never fail a scan over a label
        log.warning(
            "could not classify interface burial (%s: %s); the trust rule will "
            "fall back to effect size and the strain flag alone",
            type(exc).__name__,
            exc,
        )
    return labels


# --------------------------------------------------------------------------- #
# the benchmark-derived trust rule
# --------------------------------------------------------------------------- #
TRUST_DDG_CUTOFF = 2.0
"""Minimum |ddG| in REU for a prediction to be trusted. Cutoffs between 1.5
and 3.0 performed equivalently on the benchmark (r 0.58-0.60)."""

#: Expected Pearson r against measured ddG, by complex size, from the SKEMPI
#: 2.0 benchmark (42 complexes, 496 single-point interface mutations).
ACCURACY_BY_SIZE = (
    (300, "r ~ 0.68"),
    (450, "r ~ 0.50"),
    (700, "r ~ 0.49"),
    (10**9, "r ~ 0.32"),
)


def expected_accuracy(n_residues: int) -> str:
    """Benchmark-measured accuracy band for a complex of this size."""
    for limit, text in ACCURACY_BY_SIZE:
        if n_residues < limit:
            return text
    return ACCURACY_BY_SIZE[-1][1]


def assign_trust(location: str, ddg_mean: float, strain_flagged: bool,
                 *, cutoff: float = TRUST_DDG_CUTOFF) -> str:
    """Trust level for one prediction, per the benchmark-validated rule.

    ``none``
        The strain diagnostic fired. On the benchmark these predictions were
        *anti*-predictive: Pearson r = -0.09 and the sign was wrong 84% of the
        time (n = 54). Worse than no prediction.
    ``high``
        Core position, effect at least ``cutoff`` REU, not strain-flagged.
        r = 0.59, Spearman rho = 0.45, sign correct 95% of the time (n = 140,
        28% of all predictions).
    ``low``
        Everything else: r = 0.22, sign correct 70% of the time (n = 356).
        Directionally weak; do not rank on it.
    """
    if strain_flagged:
        return "none"
    if location == "core" and np.isfinite(ddg_mean) and abs(ddg_mean) >= cutoff:
        return "high"
    return "low"
