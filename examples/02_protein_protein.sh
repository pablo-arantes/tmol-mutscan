#!/usr/bin/env bash
# Protein-protein: human growth hormone bound to its receptor (PDB 1A22).
#
# Only the interface residues of chain A are scanned, which is the usual
# approach for a large interface: scanning every residue of a 190-residue
# chain is mostly wasted work, since a residue far from the partner has no
# interface energy to change.
set -euo pipefail

OUT=${OUT:-results_1a22}
N=${N:-20}

mkdir -p data
if [ ! -f data/1a22.cif ]; then
    echo "fetching 1A22 ..."
    curl -sL https://files.rcsb.org/download/1A22.cif -o data/1a22.cif
fi

tmol-mutscan info -i data/1a22.cif

# '--scan interface' selects residues within --interface-cutoff of the
# opposite partner, resolved separately for each partner.
tmol-mutscan run \
    --input data/1a22.cif \
    --partner-a A \
    --partner-b B \
    --scan interface \
    --interface-cutoff 6.0 \
    --n-conformers "$N" \
    --seed 42 \
    --outdir "$OUT"

echo
echo "results in $OUT/"
echo "  hotspot map: $OUT/plots/position_hotspots.png"
echo "  which residues already carry the interface: column"
echo "  wt_interface_contribution in $OUT/stats/position_summary.csv"
