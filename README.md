# tmol-mutscan

**Ensemble-based interface mutational scanning (ΔΔG) on the GPU, built on [TMol](https://github.com/uw-ipd/tmol).**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pablo-arantes/tmol-mutscan/blob/main/colab/tmol_mutscan_colab.ipynb)

**No GPU? [Run it on a free Colab GPU](https://colab.research.google.com/github/pablo-arantes/tmol-mutscan/blob/main/colab/tmol_mutscan_colab.ipynb)** — no installation, no cluster.

`tmol-mutscan` relaxes a complex into a conformational ensemble, performs
saturation mutagenesis at the positions you choose, and reports the change in
binding energy (ΔΔG) for every substitution — with proper ensemble statistics,
not just an average.

It works on **peptide–protein**, **protein–protein** and **protein–ligand**
complexes, on CPU or CUDA.

This is a port of a PyRosetta notebook workflow (FastRelax ensemble →
`PackRotamersMover` scan → bound/unbound ΔΔG) to TMol, which is the Institute
for Protein Design's PyTorch reimplementation of the Rosetta energy function.
The port replaces per-structure Python loops with batched tensor operations:
the entire ensemble is relaxed in **one** `fast_relax` call, and each chunk of
mutants is packed and scored in **one** `pack_rotamers` call plus **one**
scoring call.

---

## Contents

- [Run it on Google Colab](#run-it-on-google-colab)
- [How it works](#how-it-works)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Inputs](#inputs)
- [Outputs](#outputs)
- [Understanding the statistics](#understanding-the-statistics)
- [What changed from the PyRosetta version](#what-changed-from-the-pyrosetta-version)
- [Performance and tuning](#performance-and-tuning)
- [Reproducibility](#reproducibility)
- [Limitations and caveats](#limitations-and-caveats)
- [Citation](#citation)

---

## Run it on Google Colab

If you do not have a local GPU or cluster access, the notebook in
[`colab/tmol_mutscan_colab.ipynb`](https://colab.research.google.com/github/pablo-arantes/tmol-mutscan/blob/main/colab/tmol_mutscan_colab.ipynb) runs the whole pipeline on
Colab's free GPU.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pablo-arantes/tmol-mutscan/blob/main/colab/tmol_mutscan_colab.ipynb)

It installs everything, fetches or accepts your structure, exposes every option
as a form field (no code editing), streams the scan log with a live ETA, shows
the figures and tables inline, and zips the results for download or saves them
to your Google Drive.

Points worth knowing before you start:

- **Turn the GPU on first**: Runtime → Change runtime type → T4 GPU. The
  notebook checks and tells you if it is missing; it still runs on CPU, just
  much more slowly.
- **Colab assigns whatever GPU is free**, so speed varies between sessions. The
  notebook prints the pose count before you commit and measures its own
  throughput as it goes, rather than quoting a number that may not match your
  session.
- **Free sessions disconnect** after roughly 90 minutes idle or 12 hours total,
  and a GPU is not guaranteed. For long scans, use the notebook's *save to
  Google Drive* option, or build the ensemble once and scan subsets separately.
- **Use mmCIF for ligands.** The notebook warns you if it finds `HETATM`
  records in a `.pdb` input, since PDB files carry no bond orders.

---

## How it works

```
input complex (.pdb / .cif / .sdf / .mol2)
        │
        ▼
  1. ENSEMBLE      one batched TMol FastRelax of N copies
        │          → N relaxed conformers, each in its own local minimum
        ▼
  2. SCAN          for every (conformer, position, substitution):
        │            • design the position to the new residue
        │            • repack its neighbour shell
        │            • Cartesian-minimise the shell
        │            • read the cross-partner interaction energy
        │          all of it batched, one packing call per chunk
        ▼
  3. ΔΔG           ΔΔG = E_int(mutant) − E_int(wild-type control)
        │          paired per conformer, against an independently
        │          packed wild-type control at the same position
        ▼
  4. STATISTICS    mean, SD, SEM, 95% CI, median/IQR, Boltzmann average,
        │          effect size, paired t-test, Benjamini–Hochberg FDR,
        │          sign consistency, convergence
        ▼
  5. FIGURES       heatmaps, ranked hits with error bars, per-position
                   tolerance, hotspot map, score-term decomposition
```

### The energy being computed

`E_int` is the sum of weighted **block-pair interaction energies between
partner A and partner B** — every pairwise term that spans the interface.

This is the quantity the PyRosetta original computed as `bound − unbound`:
translating two partners 100 Å apart without repacking leaves the intra-partner
pairs and all one-body terms untouched, so the difference is just the
cross-partner pairs. TMol's `calculate_block_pair_ddg` sums those directly,
so no separated pose has to be built or scored at all.

(Strictly, the two agree to the extent that the energy is pairwise
decomposable. A few terms are mildly context-dependent — `lk_ball` places
explicit water positions using neighbouring atoms — so a physical separation
can shift an intra-partner value by a small amount that the pairwise
decomposition attributes differently. The block-pair sum is the cleaner
definition of the two, since it does not depend on a separation distance being
large enough.)

ΔΔG is then the difference between a mutant and the **wild-type control for the
same conformer** — an independently packed pose with the same neighbour shell
and the same protocol. Pairing within a conformer is what makes the ensemble
average meaningful: it cancels the conformer-to-conformer offset in absolute
interface energy, which is much larger than most mutational effects.

Energies are in Rosetta Energy Units (REU) from the **beta2016** score
function, which is what TMol implements. The PyRosetta original used
`get_fa_scorefxn()` (ref2015); beta2016 is its successor and close relative,
but the two are **not** the same function, so absolute energies from the two
workflows are not directly comparable. Rankings should broadly agree.

REU are close to kcal/mol but are not calibrated free energies; treat them as
a ranking signal, not an absolute prediction.

---

## Installation

TMol requires **Python ≥ 3.11**. If your base environment is older, make a new
one — this is the single most common installation problem.

### CPU (Linux, macOS including Apple Silicon)

```bash
conda create -n mutscan python=3.12 -y
conda activate mutscan

git clone https://github.com/pablo-arantes/tmol-mutscan.git
cd tmol-mutscan
pip install .
```

Or with a plain virtual environment:

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install .
```

### GPU (CUDA)

TMol ships its GPU kernels as a separate wheel that must match your CUDA major
version. Install a CUDA-enabled PyTorch **first**, then the matching extra:

```bash
conda create -n mutscan python=3.12 -y
conda activate mutscan

# 1. CUDA-enabled PyTorch. Get the exact command for your CUDA version from
#    https://pytorch.org/get-started/locally/ -- for example:
pip install torch --index-url https://download.pytorch.org/whl/cu128

# 2. tmol-mutscan with the matching TMol GPU operators
git clone https://github.com/pablo-arantes/tmol-mutscan.git
cd tmol-mutscan
pip install ".[cu12]"     # or ".[cu13]" for CUDA 13
```

Verify that both PyTorch and TMol see the GPU:

```bash
python -c "import torch, tmol; print(torch.cuda.is_available(), tmol.has_cuda())"
# must print: True True
```

`pip` does not detect GPUs or pick a CUDA build for you. If `tmol.has_cuda()`
is `False` while `torch.cuda.is_available()` is `True`, you installed the CPU
TMol wheel — reinstall with the `[cu12]` or `[cu13]` extra. `tmol-mutscan`
checks both and refuses `--device cuda` with a clear message rather than
silently running on the CPU.

### Notes

- **Apple Silicon**: CPU only. TMol has no MPS backend, so `--device mps` is
  rejected and `--device auto` selects `cpu`. The script runs correctly but
  without the GPU speedup.
- **`OMP: Error #15` (duplicate OpenMP runtime)**: common in conda
  environments that mix PyTorch with conda-installed numerics. `tmol-mutscan`
  sets `KMP_DUPLICATE_LIB_OK=TRUE` for you; a clean environment is still the
  better fix.
- Installing TMol pulls in RDKit and Open Babel for ligand chemistry, so the
  first install downloads a few hundred MB.

---

## Quick start

Inspect the structure first, to see the chain labels and residue numbering the
selections must use:

```bash
tmol-mutscan info -i complex.pdb
```

```
complex.pdb
98 residues, 1670 atoms, 98 blocks allocated
  chain 'A': 85 residues (85 amino acid), numbering 25-109
  chain 'B': 13 residues (13 amino acid), numbering 17-29
      sequence: ETFSDLWKLLPEN
```

Then run the full pipeline — scan every residue of peptide chain B against
protein chain A, over a 20-member ensemble:

```bash
tmol-mutscan run \
    -i complex.pdb \
    --partner-a A \
    --partner-b B \
    --scan B \
    -n 20 \
    --device cuda \
    -o results
```

That's the whole workflow. `results/` then holds the ensemble, the raw
per-conformer table, the statistics, the ΔΔG matrices and the figures.

### The three system types

```bash
# peptide–protein: scan the whole peptide
tmol-mutscan run -i pep_prot.pdb --partner-a A --partner-b B --scan B -o out

# protein–protein: scan only chain A's interface residues
tmol-mutscan run -i dimer.pdb --partner-a A --partner-b B --scan interface -o out

# protein–ligand: use mmCIF so bond orders survive, and select by role
tmol-mutscan run -i complex.cif --partner-a protein --partner-b 'resname:BEN' \
    --scan interface -o out
```

For protein–ligand work **use mmCIF, SDF or MOL2, not PDB**: PDB files carry no
bond records, so ligand chemistry cannot be typed from them. With a bond-bearing
format the ligand is prepared automatically (perception, protonation at
pH 7.4, charges) and the extended parameter database is used for scoring.

### Staged use

`run` does everything, but the stages are separately available — useful when
relaxation is expensive and you want to try several scans against one ensemble:

```bash
# 1. build the ensemble once
tmol-mutscan ensemble -i complex.pdb --partner-a A --partner-b B -n 20 -o out

# 2. scan it (repeatable with different --scan / --mutations)
tmol-mutscan scan -i complex.pdb --ensemble out/ensemble \
    --partner-a A --partner-b B --scan B -o out

# 3. re-aggregate with different thresholds, no rescoring
tmol-mutscan stats --raw out/raw/per_conformer_ddg.csv --alpha 0.01 -o out
```

`run --reuse-ensemble` does the same in one command: it picks up an existing
`<outdir>/ensemble` instead of relaxing again.

Staged results differ very slightly from a single `run`, in the third decimal
place of a ΔΔG: a PDB file stores coordinates to 0.001 Å, so reloading an
ensemble is not bit-identical to keeping it in memory. For protein–ligand runs
the ensemble is written as mmCIF instead, so that the prepared bond orders
survive the round trip.

---

## Inputs

### Required

| Flag | Meaning |
|---|---|
| `-i, --input` | The complex: `.pdb`, `.cif`/`.mmcif`, `.sdf` or `.mol2`. Must contain **both** binding partners in one file. |
| `--partner-a` | Selection for the first binding partner. |
| `--partner-b` | Selection for the second binding partner. |
| `--scan` | Selection for the positions to mutate (`run` and `scan` only). |

The two partners must be disjoint; the script stops if they overlap, because an
interaction energy between overlapping sets is not meaningful.

### Selection syntax

Used by `--partner-a`, `--partner-b` and `--scan`. Terms join with `+`.

| Term | Selects |
|---|---|
| `*` | every residue |
| `A` | every residue of chain A |
| `AB` or `A,B` | chains A and B |
| `A:17-29` | residues 17–29 of chain A, **PDB (author) numbering** |
| `A:5,9,14-18` | an explicit list and/or ranges |
| `A:52A` | residue 52 with insertion code A |
| `resname:BEN` | every residue named `BEN` (ligands, ions, cofactors) |
| `ligand` | every non-polymer residue (ligand, ion, cofactor) |
| `protein` | every amino-acid residue |
| `nucleic` | every nucleic-acid residue |
| `interface` | (`--scan` only) residues within `--interface-cutoff` of the opposite partner |

Examples: `--scan 'A:10-20+A:45'`, `--partner-b 'ligand'`,
`--partner-a 'A,B' --partner-b 'C'`.

`--scan interface` is resolved per partner: residues of A near B, plus residues
of B near A. Non-protein residues inside a `--scan` selection are skipped with
a warning, since only amino acids can be substituted.

### Main options

| Flag | Default | Meaning |
|---|---|---|
| `-n, --n-conformers` | `20` | Ensemble members. The error bars come from this. |
| `--relax-repeats` | `2` | FastRelax pack/minimise ramps per conformer. |
| `--perturb` | `0.0` | Gaussian coordinate noise (Å) before relaxing; widens the ensemble. |
| `--mutations` | `all` | `all` for the 20 canonical, or `AVLIF`, or `ALA,TRP`. |
| `--minimize` / `--no-minimize` | **on** | Cartesian-minimise the mutated shell after packing. |
| `--shell-gap` | `5.0` | Repack neighbours within this gap (Å) of the mutated residue. |
| `--interface-cutoff` | `8.0` | Heavy-atom distance defining `interface`. |
| `--decompose` | off | Also record the per-score-term breakdown of every ΔΔG. |
| `--device` | `auto` | `auto`, `cpu`, `cuda`, `cuda:N`. |
| `--batch-size` | auto | Mutant poses per chunk; auto-sized from the system. |
| `--seed` | `0` | Base RNG seed. |
| `--alpha` | `0.05` | FDR threshold for significance. |
| `--kt` | `0.593` | kT for the Boltzmann-averaged ΔΔG (REU; 0.593 ≈ 298 K). |
| `--threads` | torch default | CPU threads. |

`tmol-mutscan run --help` lists everything.

### Why `--minimize` is on by default

Packing alone cannot relieve the strain of a large substitution into a tight
pocket, and the resulting ΔΔG is a clash artefact rather than a measurement.
On the p53/MDM2 test case, packing only:

| Substitution | pack only | pack + minimise |
|---|---:|---:|
| F19W | **+155.2** | **+2.7** |
| L26W | **+77.8** | **+2.7** |
| L26F | +7.5 | **−1.1** |
| W23A | +13.7 | +13.6 |

The real hotspot (W23A) is unchanged, while the two meaningless numbers become
physical. Minimisation costs roughly 3× the scan time. `--no-minimize`
reproduces the original PyRosetta protocol if you need to compare against it.

---

## Outputs

```
results/
├── run_manifest.json              every parameter, version and device — the reproducibility record
├── mutscan.log                    full run log at DEBUG level
│
├── ensemble/
│   ├── conformer_001.pdb …        the relaxed ensemble (.cif when ligands were prepared)
│   └── ensemble_scores.csv        per-conformer total score and interface energy
│
├── raw/
│   ├── per_conformer_ddg.csv      ONE ROW PER (conformer, position, substitution) — the primary data
│   └── per_conformer_terms.csv    per-score-term ΔΔG (only with --decompose)
│
├── stats/
│   ├── ddg_statistics.csv         ONE ROW PER (position, substitution) with all statistics
│   ├── position_summary.csv       one row per position: sensitivity, best substitution, hotspot score
│   ├── top_mutations.csv          the --top-n most favourable substitutions
│   └── convergence.csv            how the mean ΔΔG settles as conformers accumulate
│
├── matrices/                      position × substitution matrices, ready to plot elsewhere
│   ├── ddg_mean.csv               ← the matrix the original notebook produced
│   ├── ddg_sd.csv                 ← the standard deviation that was missing
│   ├── ddg_sem.csv  ddg_median.csv  ddg_boltzmann.csv
│   └── qvalue.csv   frac_stabilizing.csv  cohens_dz.csv
│
└── plots/                         PNG (300 dpi) and PDF
    ├── ddg_mean_heatmap.png       headline figure: diverging, centred on zero
    ├── ddg_sd_heatmap.png         where the ensemble disagrees
    ├── ddg_significant_heatmap.png  mean ΔΔG, non-significant cells greyed out
    ├── top_mutations.png          ranked hits with 95% CI error bars
    ├── position_profile.png       per-position tolerance (violin + box)
    ├── position_hotspots.png      wild-type interface contribution vs tolerance
    └── term_decomposition.png     which energy term drives each hit (--decompose)
```

### `raw/per_conformer_ddg.csv`

The primary data; everything else is derived from it. One row per
(conformer, position, substitution).

| Column | Meaning |
|---|---|
| `conformer` | Ensemble member, 1-based. |
| `position` | `chain:resnum`, e.g. `B:23`. |
| `chain`, `resnum`, `icode` | Parsed components of the position. |
| `wt_aa`, `wt_name3` | Wild-type residue, 1- and 3-letter. |
| `mut_aa`, `mut_name3` | Substituted residue, 1- and 3-letter. |
| `e_wt` | Interface energy of this conformer's wild-type control (REU). |
| `e_mut` | Interface energy of the mutant (REU). |
| `ddg` | `e_mut − e_wt`. **Negative = predicted tighter binding.** |
| `ratio` | `e_mut / e_wt` — the original notebook's ratio, for comparison. |
| `percentage` | `(e_mut − e_wt) / e_wt` — the original notebook's percentage. |
| `achieved_name3` | What the packer actually built (a cross-check on `mut_name3`). |
| `skipped` | Empty when the request was honoured, otherwise why it was not. |

`ratio` and `percentage` are carried over for continuity with the original
notebook, but **use `ddg`**: both legacy columns divide by a negative number
that can approach zero, which makes them unstable and awkward to interpret.

### `stats/ddg_statistics.csv`

One row per (position, substitution), with the columns described in the next
section. Wild-type → wild-type rows are kept as a reference and flagged with
`is_wt_self`; see the note on the noise floor below.

---

## Understanding the statistics

The original workflow produced a mean ΔΔG matrix. Averaging an ensemble into
one number discards the information that tells you whether to believe it, so
every substitution gets the following.

### Location — what is the effect?

| Column | Meaning |
|---|---|
| `ddg_mean` | Arithmetic mean over conformers. **The headline number.** |
| `ddg_median`, `ddg_iqr` | Robust location and spread. A large mean/median gap means one or two conformers dominate — usually an unresolved clash. |
| `ddg_boltzmann` | `−kT·ln⟨exp(−ΔΔG/kT)⟩`, the ensemble free-energy average. Dominated by the conformers in which the substitution *works*, and always ≤ `ddg_mean`. Use it when the conformers represent thermally accessible states rather than repeated measurements. |
| `ddg_min`, `ddg_max` | Range across the ensemble. |

### Uncertainty — how precise is it?

| Column | Meaning |
|---|---|
| `ddg_sd` | Sample standard deviation (ddof = 1) — the conformational spread. |
| `ddg_sem` | `SD/√n`, the precision of the mean. |
| `ci95_low`, `ci95_high` | 95% confidence interval from the **t** distribution (not normal — an ensemble of 10–25 is small). |
| `n_conformers` | Conformers that contributed. |

### Significance — is it distinguishable from zero?

| Column | Meaning |
|---|---|
| `t_stat`, `p_value` | Paired one-sample t-test of ΔΔG against 0. Valid because each mutant is differenced against its own conformer's control. |
| `q_value` | **Benjamini–Hochberg FDR** across every substitution. A 23-position × 20-substitution scan is 460 tests; raw p-values cannot be read directly. |
| `significant` | `q_value < --alpha`. |
| `effect` | `stabilizing` / `neutral` / `destabilizing` / `wild-type`, combining significance with `--stabilizing-cutoff`. |

### Reliability — would you bet on it?

| Column | Meaning |
|---|---|
| `cohens_dz` | Paired effect size, `mean/SD`: the shift relative to conformational noise. |
| `frac_stabilizing` | Fraction of conformers that agree ΔΔG < 0. **A good mean with `frac_stabilizing ≈ 0.5` is not reproducible** — the ensemble does not agree on the sign. |

### Reading it in practice

A substitution worth testing experimentally has **all** of:

- `ddg_mean` clearly negative (say ≤ −0.5 REU),
- `ci95_high` also below zero — the interval excludes no effect,
- `q_value < 0.05`,
- `frac_stabilizing` close to 1.0 — the whole ensemble agrees,
- `|cohens_dz|` above ~1 — the effect exceeds conformational noise.

Sorting `ddg_statistics.csv` by `ddg_mean` alone will put noise at the top.
`stats/top_mutations.csv` and `plots/top_mutations.png` carry the confidence
intervals alongside the ranking for this reason.

> **What the error bars do and do not cover.** `ddg_sd` is the spread across
> *the ensemble you generated*, and nothing else. With the default
> `--perturb 0` the ensemble is tight (~0.4 Å RMSD), so the SD is small and
> `cohens_dz` can reach the hundreds — that means the conformers agree with
> each other, **not** that the ΔΔG is accurate to three decimal places. The
> error bars do not include the error of the energy function, the backbone
> held fixed during mutation, or the input model. For a spread that reflects
> more of the real conformational uncertainty, raise `--perturb` (and
> `--relax-repeats` with it) and expect the intervals to widen honestly.
>
> The wild-type → wild-type rows are a useful cross-check here: they are the
> same calculation performed twice independently, so their ΔΔG is the
> numerical noise floor of the protocol. The run log reports the largest one
> as `wild-type self-consistency`; it is typically ≤ 0.001 REU.

### Position-level output

`stats/position_summary.csv` reports, per position:

- `mean_ddg_all_substitutions` — **substitution tolerance**. High means the
  position resists change; a hotspot.
- `best_substitution`, `best_ddg_mean`, `best_ddg_sem`, `best_q_value` — the
  most favourable substitution found.
- `n_significant_stabilizing` / `n_significant_destabilizing`, and
  `stabilizing_substitutions` as a one-letter string.
- `wt_interface_contribution` — the wild-type residue's **own share of the
  interface energy**, summed over both block-pair orientations against the
  opposite partner and averaged over the ensemble. This is a hotspot map that
  needs no mutations at all: a large negative value means the residue is
  already doing the binding.

`plots/position_hotspots.png` plots those last two against each other, which
separates the four interesting cases: residues that bind strongly and resist
change (true hotspots, leave alone), residues that bind weakly but resist
change (structurally constrained), residues that bind strongly and tolerate
change, and residues that contribute little and tolerate change (the best
places to try an improvement).

### Was the ensemble big enough?

`stats/convergence.csv` and `plots/convergence.png` answer this. For each
prefix of *k* conformers, the running mean of every substitution is compared
with the mean over the full ensemble:

- `mean_abs_drift` — average |running mean − final mean|, in REU.
- `max_abs_drift` — worst case over all substitutions.
- `mean_sem` — average SEM at that ensemble size.

A curve that has flattened near zero means the ensemble is large enough. A
curve still falling at *k = n* means **increase `-n`**. The last row has zero
drift by construction, so the run summary quotes the second-to-last point.

---

## What changed from the PyRosetta version

| Step | PyRosetta original | tmol-mutscan |
|---|---|---|
| Structure I/O | `pose_from_pdb` | `pose_stack_from_file`; `prepare_ligands` for ligand chemistry |
| Score function | `get_fa_scorefxn()` (ref2015) | `beta2016_score_function` |
| Ensemble | `BackrubMover` + `GenericMonteCarloMover`, then `FastRelax` per structure, in a Python loop | one **batched** `fast_relax` over N copies; stochastic packing puts each copy in its own minimum. Optional `--perturb` widens it. |
| Mutation | `PackRotamersMover` + `RestrictAbsentCanonicalAASRLT`, one pose at a time | **batched** `pack_rotamers`: one packing call per chunk, per-pose identity masks |
| Neighbour shell | `NeighborhoodResidueSelector` | enclosing-sphere adjacency, `--shell-gap` |
| Interface energy | `bound − unbound` via a 100 Å `RigidBodyTransMover` | `calculate_block_pair_ddg` sums the cross-partner terms directly — same quantity, no separated pose |
| Relief of strain | none (pack only) | Cartesian minimisation of the mutated shell, on by default |
| Wild-type reference | `wt_energy` from a repack at the same position | the same, as an explicit per-conformer control pose, paired with each mutant |
| Statistics | mean and SD of the matrix | mean, SD, SEM, t-based 95% CI, median/IQR, Boltzmann average, effect size, paired t-test, BH FDR, sign consistency, convergence |
| Interface | Jupyter notebook with edit-in-place constants | `argparse` CLI with subcommands |

### Migrating from the notebook

The notebook's edit-in-place constants map onto flags like this:

| Notebook variable | Flag |
|---|---|
| `fileinput`, `workDir` | `--input`, `--outdir` |
| `peptide_chain = "A"` | the partner the peptide is in, e.g. `--partner-b A` |
| `target_chain = "B"` | the other partner, e.g. `--partner-a B` |
| `conformation_number = 20` | `--n-conformers 20` |
| `mutation_sites = [1..23]` | `--scan 'A:1-23'` |
| `possible_mutations = [...]` | `--mutations` (default `all` is the same 20) |
| `FastRelax(standard_repeats=5)` | `--relax-repeats` (default 2) |
| `partners = target + '_' + peptide` | implied by `--partner-a` / `--partner-b` |
| `UNBOUND_CUTOFF = -250` | not needed — no separated pose is scored, so there is no unbound energy to filter on |
| `RATIO_CUTOFF = 1.0` | that filter kept only substitutions predicted to *improve* binding. The equivalent is `ddg_mean < 0`; `plots/ddg_significant_heatmap.png` additionally requires the improvement to be statistically resolved, which the ratio filter could not test |

The notebook's `ratio` and `percentage` columns are still produced in
`raw/per_conformer_ddg.csv` and as `ratio_mean` / `percentage_mean` in
`stats/ddg_statistics.csv`, so an existing downstream analysis can be pointed
at them — but `ddg` is the better quantity, for the reasons given above.

Because the notebook wrote one CSV per mutation and concatenated them with
`cat`, its output had no header and column order mattered. The tables here are
proper CSVs with headers.

### Note on the ensemble method

TMol has no backrub mover, so the ensemble is generated differently and the
two ensembles are **not** expected to be identical. Backrub+MC samples local
backbone rotations explicitly; here the diversity comes from the stochastic
packer inside FastRelax, optionally seeded with coordinate noise. On the test
case, plain batched relaxation gives ~0.4 Å RMSD between conformers with every
member fully relaxed, and `--perturb 0.2` gives ~1.5 Å at the cost of needing
more `--relax-repeats` to re-converge.

Always check `ensemble/ensemble_scores.csv`: a wide `total_score` spread means
the members are not equivalently relaxed, and the ΔΔG error bars will inherit
that inconsistency. Raise `--relax-repeats` if so.

---

## Performance and tuning

The speedup over per-structure PyRosetta loops comes from batching, so it is
largest when there is a lot to batch — many conformers, many positions, many
substitutions — and on a GPU.

```
poses built = n_conformers × n_positions × (n_substitutions + 1)
```

A 20-conformer, 23-position, 20-substitution scan is 20 × 23 × 21 = **9,660
poses**. Each chunk of those is one packing call and one scoring call.

Tuning:

- **`--device cuda`** is where the benefit is. The CPU path is correct and
  useful for small jobs, but TMol's reason to exist is the GPU.
- **`--batch-size`** defaults to a size derived from the system, keeping the
  block-pair score tensor (`n_terms × n_poses × n_blocks²`) near 2 GB. Raise it
  on a large GPU; lower it if you hit out-of-memory.
- **`TMOL_PACK_MAX_POSES_PER_CHUNK=10`** additionally caps TMol's internal
  packing chunk, for diagnosing memory pressure.
- **`--no-minimize`** cuts scan time ~3×, at the cost described above.
- **`--mutations`** — scanning 5 chosen substitutions instead of 20 is 4× less
  work. `--scan interface` instead of a whole chain likewise.
- **`--threads`** sets the CPU thread count on the CPU path.
- Build the ensemble once with `ensemble`, then reuse it across several `scan`
  invocations rather than re-relaxing.

Progress, throughput in poses/s and an ETA are logged per chunk.

---

## Reproducibility

Every run writes `run_manifest.json` with the full command line, every
resolved argument, the score function and samplers used, and the versions of
TMol, PyTorch, NumPy, SciPy and pandas, plus the device and thread count. This
is the record TMol's own interface-analysis guidance asks for when reporting
a computational experiment.

`--seed` controls the ensemble and packing seeds. Note that FastRelax and the
packer are stochastic: the same seed on the same device and versions
reproduces a run, but results are **not** guaranteed to be bit-identical
across devices (CPU vs CUDA) or TMol versions. Report the ensemble size and
the spread, not a single pack.

---

## Limitations and caveats

- **REU are not kcal/mol.** beta2016 energies rank substitutions; they are not
  calibrated binding free energies. Validate hits experimentally.
- **No backbone flexibility during mutation.** The scan repacks and minimises
  side chains and nearby atoms; it does not rebuild the backbone around a
  substitution. Large substitutions in tight pockets are therefore
  pessimistic, even with minimisation.
- **No explicit solvent or entropy** beyond what the implicit terms in
  beta2016 provide.
- **Proline and glycine** change backbone preferences in ways a side-chain
  packer cannot model. Treat their ΔΔG with particular suspicion.
- **Disulfide-bonded cysteines** and metal-coordinating residues cannot be
  substituted; those requests are skipped with a warning naming the position.
- **The ensemble is not a Boltzmann ensemble.** It is a set of local minima
  near the input structure, so `ddg_boltzmann` is a weighted average over
  those minima, not a rigorous free energy.
- **Error bars measure ensemble spread, not accuracy.** See the note in
  [Understanding the statistics](#reading-it-in-practice). A tight ensemble
  produces small confidence intervals that say nothing about how well
  beta2016 predicts a real binding change.
- **The input matters more than anything here.** A scan of a poorly resolved
  or badly modelled interface produces confident-looking numbers about
  nothing. Check `ensemble_scores.csv` and the input structure first.
- Insertion codes are supported, but alternate locations are not handled
  specially — the first is taken.

---

## Citation

If this script contributes to published work, cite TMol, which does the
science:

> Andrew Leaver-Fay, Jeff Flatten, Alex Ford, Joseph Kleinhenz, Henry Solberg,
> David Baker, Andrew M. Watkins, Brian Kuhlman, Frank DiMaio. *tmol: a
> GPU-accelerated, PyTorch implementation of Rosetta's relax protocol*
> (manuscript in preparation). https://github.com/uw-ipd/tmol

and note the protocol: ensemble-based interface ΔΔG scanning with
`tmol-mutscan` (https://github.com/pablo-arantes/tmol-mutscan), beta2016 score
function, N conformers, with or without shell minimisation, as recorded in
`run_manifest.json`.

## License

MIT — see [LICENSE](LICENSE). TMol is Apache-2.0 and is not redistributed here.
