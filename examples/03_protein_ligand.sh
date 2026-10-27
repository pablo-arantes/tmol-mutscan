#!/usr/bin/env bash
# Protein-ligand: bovine trypsin bound to benzamidine (PDB 1BTY).
#
# Note the mmCIF input. A PDB file carries no bond records, so ligand
# chemistry cannot be typed from one; with mmCIF the ligand is perceived,
# protonated at pH 7.4 and typed automatically, and the extended parameter
# database is used for scoring.
#
# 1BTY keeps the protein and both heteroatoms (BEN and a calcium ion) in
# chain A, so the partners are selected by role rather than by chain:
# 'resname:BEN' is the ligand alone, which leaves the calcium with the
# protein where it belongs.
set -euo pipefail

OUT=${OUT:-results_1bty}
N=${N:-12}

mkdir -p data
if [ ! -f data/1bty.cif ]; then
    echo "fetching 1BTY ..."
    curl -sL https://files.rcsb.org/download/1BTY.cif -o data/1bty.cif
fi

tmol-mutscan info -i data/1bty.cif

tmol-mutscan run \
    --input data/1bty.cif \
    --partner-a 'protein' \
    --partner-b 'resname:BEN' \
    --scan interface \
    --interface-cutoff 5.0 \
    --n-conformers "$N" \
    --decompose \
    --seed 42 \
    --outdir "$OUT"

echo
echo "results in $OUT/"
echo "Ligand preparation is the slow part of loading; the ensemble is written"
echo "as mmCIF so that the prepared bonds survive a reload by 'tmol-mutscan scan'."
