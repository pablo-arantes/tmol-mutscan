# Examples

Three runnable examples, one per system type. Each fetches its own structure
from the PDB, so they need a network connection on first run.

| Script | System | PDB | What it scans |
|---|---|---|---|
| `01_peptide_protein.sh` | peptide–protein | 1YCR | the whole p53 peptide against MDM2 |
| `02_protein_protein.sh` | protein–protein | 1A22 | hGH interface residues against its receptor |
| `03_protein_ligand.sh` | protein–ligand | 1BTY | trypsin residues around benzamidine |

All three default to `--device auto` (CUDA when available). On CPU, start with
the peptide example and the reduced settings it suggests — the full runs are
sized for a GPU.

## 1YCR is a good first test

The p53/MDM2 complex has three well-characterised hotspot residues on the
peptide — **F19, W23 and L26** — which bury into a hydrophobic cleft on MDM2,
with W23 the single most important. A correct run ranks substitutions at W23
among the most destabilizing in the whole scan, so it is an easy check that
the partner selections and the energy are set up the way you intended.

```bash
bash 01_peptide_protein.sh
sort -t, -k4 -g results_1ycr/stats/ddg_statistics.csv | head
```
