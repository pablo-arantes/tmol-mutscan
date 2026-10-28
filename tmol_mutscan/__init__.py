"""Ensemble-based mutational scanning of molecular interfaces with TMol.

Replaces a PyRosetta FastRelax + PackRotamers interface scan with the
GPU-capable TMol stack: one batched ``fast_relax`` builds the conformational
ensemble, one batched ``pack_rotamers`` builds every mutant, and
``calculate_block_pair_ddg`` reads the cross-partner interaction energy.

Supports peptide-protein, protein-protein and protein-ligand complexes.
"""

__version__ = "0.2.0"

__all__ = ["__version__"]
