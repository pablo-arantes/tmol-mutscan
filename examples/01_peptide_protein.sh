#!/usr/bin/env bash
# Peptide-protein: the p53 transactivation peptide bound to MDM2 (PDB 1YCR).
#
# Chain A is MDM2 (85 residues), chain B is the p53 peptide (13 residues).
# Every peptide position is scanned against every canonical amino acid.
#
# Expected: substitutions at W23 -- and to a lesser degree F19 and L26 -- come
# out as the most destabilizing, since those are the experimental hotspots.
set -euo pipefail

OUT=${OUT:-results_1ycr}
N=${N:-20}

mkdir -p data
if [ ! -f data/1ycr.pdb ]; then
    echo "fetching 1YCR ..."
    curl -sL https://files.rcsb.org/download/1YCR.pdb -o data/1ycr.pdb
fi

# Look at the chains and numbering first.
tmol-mutscan info -i data/1ycr.pdb

tmol-mutscan run \
    --input data/1ycr.pdb \
    --partner-a A \
    --partner-b B \
    --scan B \
    --n-conformers "$N" \
    --mutations all \
    --decompose \
    --seed 42 \
    --outdir "$OUT"

echo
echo "results in $OUT/"
echo "  headline figure : $OUT/plots/ddg_mean_heatmap.png"
echo "  ranked hits     : $OUT/stats/top_mutations.csv"
echo
echo "On a CPU, try a reduced run first:"
echo "  N=4 OUT=quick_1ycr bash $0   # plus --mutations 'A,W,F,D' --scan 'B:19,23,26'"
