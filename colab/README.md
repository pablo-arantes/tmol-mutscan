# Colab notebooks

Two notebooks, both runnable on Colab's free GPU.

| Notebook | Purpose | |
|---|---|---|
| `tmol_mutscan_colab.ipynb` | **Run a scan** on your own complex | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pablo-arantes/tmol-mutscan/blob/main/colab/tmol_mutscan_colab.ipynb) |
| `tmol_mutscan_skempi_benchmark.ipynb` | **Benchmark the method** against measured ΔΔG from SKEMPI 2.0 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pablo-arantes/tmol-mutscan/blob/main/colab/tmol_mutscan_skempi_benchmark.ipynb) |

---

## 1. Running a scan — `tmol_mutscan_colab.ipynb`

Runs the full `tmol-mutscan` pipeline on Google
Colab's free GPU, for people without a local GPU or cluster.

Open it with the badge above, or paste this into Colab's *File → Open notebook
→ GitHub* box:

```
pablo-arantes/tmol-mutscan
```

## What the notebook does

| Cell | Purpose |
|---|---|
| 1 | Report the GPU and Python version, and warn in advance if PyTorch will need upgrading — all without importing PyTorch, so the warning is accurate |
| 2 | Install `tmol[cu12]` and `tmol-mutscan`, and say whether a session restart is needed |
| 3 | Gate: confirm **both** `torch.cuda.is_available()` and `tmol.has_cuda()`, then score a test pose |
| 4 | Fetch a structure by PDB ID, upload one, or read one from Google Drive |
| 5 | List chains and residue numbering |
| 6 | Selection-syntax reference |
| 7 | Every option as a form field, plus the pose count for the job you configured |
| 8 | Run the scan, streaming the log with a live ETA |
| 9 | Show all figures inline, each with a caption explaining how to read it |
| 10 | Show the hit table, per-position summary and ensemble quality check |
| 11 | Zip for download, and optionally copy to Google Drive |

The final markdown cell covers sizing a run, surviving a disconnect, a
troubleshooting table, and the method's limitations.

## Why `tmol.has_cuda()` is checked separately

TMol ships its GPU kernels in a separate wheel (`tmol-ops-cu12`). If only the
CPU wheel is installed, `torch.cuda.is_available()` is still `True` while TMol
quietly runs on the CPU — a scan that should take minutes then takes hours. The
notebook checks both and says exactly what to do if they disagree.

---

## 2. Benchmarking — `tmol_mutscan_skempi_benchmark.ipynb`

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pablo-arantes/tmol-mutscan/blob/main/colab/tmol_mutscan_skempi_benchmark.ipynb)

Answers the question everything else in this repository can only argue about:
**does the predicted ΔΔG correlate with measured binding ΔΔG?**

It pulls experimental values from [SKEMPI 2.0](https://life.bsc.es/pid/skempi2/),
runs `tmol-mutscan` on the same mutations, and reports Pearson r, Spearman ρ,
the fitted REU → kcal/mol slope, sign agreement, and an AUC for picking out
badly destabilising mutations — broken down by subset.

Needs **v0.2.0 or newer**: it uses `run_scan(explicit_pairs=...)` to scan only
the measured substitutions instead of the full cross product, and the `strain`
diagnostic.

### The decision-relevant outputs

- **Stabilizing subset (ΔΔG_exp < 0)** — the case peptide optimisation needs and
  the one the method is theoretically worst at. If ρ is near zero here while the
  pooled ρ looks fine, the method should be used to *exclude* candidates, never
  to *rank* them.
- **Within-complex ρ** — what you actually rely on. A good pooled r with a
  median within-complex ρ near zero means the method separates systems, not
  mutations.
- **Does the strain filter help?** Cell 10 tests whether excluding
  `strain_flag` hits, or using `ddg_total`, or blending the two, improves the
  correlation. If none beats the baseline, `strain` is informational only.

### What the parsing gets right (and is easy to get wrong)

| Pitfall | Handling |
|---|---|
| `Mutation(s)_cleaned` is **renumbered** | Uses `Mutation(s)_PDB`; the same mutation is `LI45G` vs `LI38G` |
| SKEMPI stores one row per **measurement**, not per mutation | Deduplicated to the median per (complex, mutation); 7,085 rows → 3,299 unique pairs |
| Four protease complexes hold ~23% of interface mutations | `max_mutations_per_complex` caps each one, sampled across its ΔΔG range |
| Second copies in the asymmetric unit | Structure filtered to the partner chains only |
| **Glycans break TMol loading** | Sugar residues removed from the file; this took load success from 22/28 to 26/28 on a probe |
| Modified residues in-chain (CGU, IAS) | Cannot be fixed; those complexes are skipped and reported |
| Wrong chain or numbering silently scanning the wrong residue | Every mutation's wild-type identity is verified against the structure before use |

### Practicalities

- **Resumable.** Predictions are appended to `predictions.csv` per complex and
  finished complexes are skipped on re-run, so a Colab disconnect costs nothing.
  `save_to_drive` keeps the checkpoint off the VM.
- **Run with `n_complexes = 2` first** to calibrate per-complex cost on whatever
  GPU you were assigned.
- `n_conformers = 3` is the default for coverage, which makes the correlation a
  **lower bound**; cell 10 estimates how much that costs.
- SKEMPI is protein–protein, so the numbers are an **upper bound** for
  peptide–protein and say nothing about protein–ligand.

---

## Editing the notebooks

Both are generated, but they are normal `.ipynb` files and can be edited
directly. Keep outputs stripped before committing so diffs stay readable:

```bash
jupyter nbconvert --clear-output --inplace colab/*.ipynb
```
