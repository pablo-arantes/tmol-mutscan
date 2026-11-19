"""One import surface that works across TMol releases.

TMol 1.0.0 (November 2026) consolidated its public API onto the top-level
``tmol`` namespace and dropped the submodule re-exports that 0.2.x used, so
``from tmol.pose import PoseStack`` and about twenty similar imports stopped
resolving. ``one2three``/``three2one`` also became private.

Rather than pin to one release, every TMol name this package needs is resolved
here: the top-level namespace first, then the 0.2.x submodule it used to live
in. That keeps a single codebase working on both, which matters because the
accuracy benchmark in the README was measured on 0.2.0 and should stay
reproducible against it.

The canonical one-letter/three-letter mapping is defined here rather than
imported, since TMol 1.0.0 made its version private and a 20-entry table for
the standard amino acids is not going to change.
"""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys

import tmol

TMOL_VERSION = getattr(tmol, "__version__", "unknown")

BENCHMARKED_TMOL = "0.2.0"
"""The TMol release the SKEMPI benchmark and the trust cutoff were measured on.

TMol 1.0.0 shifts individual ddG values relative to 0.2.0 -- on a 15-substitution
check at 8 conformers, mean |difference| 1.19 REU and up to 6.7 REU, with 7 of 15
differences exceeding twice the combined standard error, almost all in the
more-destabilizing direction. The ranking largely survives (Pearson r = 0.90,
Spearman rho = 0.87), but absolute magnitudes do not, so a cutoff calibrated on
0.2.0 does not transfer unchanged.
"""


def tmol_matches_benchmark() -> bool:
    """True when the installed TMol is the release the trust rule was tuned on."""
    return str(TMOL_VERSION).split(".")[:2] == BENCHMARKED_TMOL.split(".")[:2]


def _resolve(name: str, *legacy_modules: str):
    """Find a TMol symbol on the top-level namespace or an older submodule."""
    if hasattr(tmol, name):
        return getattr(tmol, name)
    for module_name in legacy_modules:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        if hasattr(module, name):
            return getattr(module, name)
    raise ImportError(
        f"could not find {name!r} in this TMol installation "
        f"(version {TMOL_VERSION}). Tried the top-level 'tmol' namespace and "
        f"{list(legacy_modules)}. tmol-mutscan supports TMol 0.2.x and 1.x; "
        "a newer release may have moved it again."
    )


# --- databases and core types ---------------------------------------------
ParameterDatabase = _resolve("ParameterDatabase", "tmol.database")
PoseStack = _resolve("PoseStack", "tmol.pose")
PoseStackBuilder = _resolve("PoseStackBuilder", "tmol.pose")

# --- structure I/O ---------------------------------------------------------
atom_array_from_file = _resolve("atom_array_from_file", "tmol.io")
atom_array_from_pose_stack = _resolve("atom_array_from_pose_stack", "tmol.io")
file_from_pose_stack = _resolve("file_from_pose_stack", "tmol.io")
pose_stack_from_atom_array = _resolve("pose_stack_from_atom_array", "tmol.io")
pose_stack_from_file = _resolve("pose_stack_from_file", "tmol.io")
extended_pose_stack_from_sequences = _resolve(
    "extended_pose_stack_from_sequences", "tmol.io"
)

# --- pose helpers ---------------------------------------------------------
get_ligand_mask = _resolve("get_ligand_mask", "tmol.pose")
# Still only on tmol.pose in 1.0.0, not promoted to the top level.
get_polymer_type_mask = _resolve("get_polymer_type_mask", "tmol.pose")

# --- scoring --------------------------------------------------------------
ScoreFunction = _resolve("ScoreFunction", "tmol.score")
beta2016_score_function = _resolve("beta2016_score_function", "tmol.score")

# --- packing --------------------------------------------------------------
PackerPalette = _resolve("PackerPalette", "tmol.pack")
PackerTask = _resolve("PackerTask", "tmol.pack")
pack_rotamers = _resolve("pack_rotamers", "tmol.pack")
# The samplers moved from tmol.pack.rotamer (0.2.x) up to tmol.pack (1.0.0).
FixedAAChiSampler = _resolve("FixedAAChiSampler", "tmol.pack", "tmol.pack.rotamer")
IncludeCurrentSampler = _resolve(
    "IncludeCurrentSampler", "tmol.pack", "tmol.pack.rotamer"
)
create_dunbrack_sampler_from_database = _resolve(
    "create_dunbrack_sampler_from_database",
    "tmol.pack",
    "tmol.pack.rotamer.dunbrack",
)

# --- minimization and relax -----------------------------------------------
run_cart_min = _resolve("run_cart_min", "tmol.optimization")
fast_relax = _resolve("fast_relax", "tmol.relax")
CartesianMoveMap = _resolve("CartesianMoveMap", "tmol.kinematics")
FoldForest = _resolve("FoldForest", "tmol.kinematics")

# --- batched operations (tmol.ops is unchanged across both releases) ------
_ops = importlib.import_module("tmol.ops")
build_coord_mask_for_mask_and_interacting_atoms = (
    _ops.build_coord_mask_for_mask_and_interacting_atoms
)
calculate_block_pair_ddg = _ops.calculate_block_pair_ddg
compute_block_adjacency = _ops.compute_block_adjacency
compute_block_centroids_and_furthest_dist = _ops.compute_block_centroids_and_furthest_dist
get_chain_mask = _ops.get_chain_mask


# --- residue naming -------------------------------------------------------
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}
ONE_TO_THREE = {one: three for three, one in THREE_TO_ONE.items()}


def three2one(name3: str) -> str | None:
    """One-letter code for a canonical residue name, else None."""
    return THREE_TO_ONE.get(str(name3).strip().upper())


def one2three(one: str) -> str | None:
    """Three-letter name for a canonical one-letter code, else None."""
    return ONE_TO_THREE.get(str(one).strip().upper())


def guard_macos_metal_offload() -> str | None:
    """Disable TMol's Metal offload when the Metal compiler is missing.

    TMol 1.0.0 added transparent Metal (MPS) offload of the LJ/LK and
    electrostatic kernels, so even a CPU score function reaches for the GPU on
    Apple Silicon. That is a genuine speedup, but compiling the shader library
    needs the ``metal`` tool, which ships with full Xcode and *not* with the
    Command Line Tools alone. Without it, a scan dies partway through packing
    with ``RuntimeError: tmol MPS: shader library compilation failed``.

    Checking for the compiler up front and setting ``TMOL_MPS_OFFLOAD=0`` turns
    that into a one-line notice instead. An explicit setting is always
    respected.

    Returns a message to log, or None when nothing was changed.
    """
    if sys.platform != "darwin" or "TMOL_MPS_OFFLOAD" in os.environ:
        return None
    if not shutil.which("xcrun"):
        return None
    try:
        found = subprocess.run(
            ["xcrun", "-f", "metal"], capture_output=True, text=True, timeout=20
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if found.returncode == 0:
        return None                       # compiler present; let TMol use Metal
    os.environ["TMOL_MPS_OFFLOAD"] = "0"
    return (
        "no Metal compiler found (Xcode Command Line Tools alone do not ship "
        "one), so TMOL_MPS_OFFLOAD=0 was set to keep TMol on the CPU. Install "
        "full Xcode to let TMol 1.x offload to the Apple GPU, or set "
        "TMOL_MPS_OFFLOAD=1 to try anyway."
    )


def relax_poses(pose_stack, sfxn, *, num_repeats: int = 2, verbose: bool = False):
    """Run FastRelax with Cartesian minimisation of every atom, on either release.

    The signature changed in TMol 1.0.0: ``move_map`` and ``fold_forest`` were
    replaced by a single ``minimizer``, and ``packer_pallete`` was renamed (and
    made optional) as ``packer_palette``. The 1.0.0 defaults -- a
    ``CartesianRelaxMinimizer`` with every atom free and a plain
    ``PackerPalette()`` -- are exactly what 0.2.x needed those four arguments to
    express, so on 1.x nothing has to be passed at all.

    Dispatch is on the actual signature rather than the version string, so a
    future release that keeps either shape still works.
    """
    import inspect

    parameters = inspect.signature(fast_relax).parameters
    if "fold_forest" in parameters:          # TMol 0.2.x
        return fast_relax(
            pose_stack,
            sfxn,
            PackerPalette(),
            CartesianMoveMap(),
            FoldForest.reasonable_fold_forest(pose_stack),
            num_repeats=num_repeats,
            verbose=verbose,
        )
    return fast_relax(                        # TMol 1.x
        pose_stack, sfxn, num_repeats=num_repeats, verbose=verbose
    )


#: Message from the macOS Metal guard, which runs once when this module is
#: imported so library users are protected too, not only the CLI. The CLI logs
#: it; a library caller can read it.
METAL_GUARD_NOTE: str | None = None


def _run_metal_guard_once() -> None:
    global METAL_GUARD_NOTE
    METAL_GUARD_NOTE = guard_macos_metal_offload()


# Must happen before any scoring module is rendered, and importing this module
# is the earliest point every entry path shares.
_run_metal_guard_once()


def tmol_layout() -> str:
    """Human-readable note on the detected TMol release.

    Both 0.2.x and 1.x expose many names at the top level (0.2.x through a
    lazy ``__getattr__``), so the version string is reported rather than a
    guess at the layout.
    """
    return f"TMol {TMOL_VERSION}"
