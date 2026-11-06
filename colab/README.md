# Colab notebooks

Two notebooks, both runnable on Colab's free GPU. They install
**v0.3.0** by default.

| Notebook | Purpose | |
|---|---|---|
| `tmol_mutscan_colab.ipynb` | **Run a scan** on your own complex | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pablo-arantes/tmol-mutscan/blob/main/colab/tmol_mutscan_colab.ipynb) |
| `tmol_mutscan_skempi_benchmark.ipynb` | **Reproduce or extend the accuracy benchmark** against SKEMPI 2.0 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pablo-arantes/tmol-mutscan/blob/main/colab/tmol_mutscan_skempi_benchmark.ipynb) |

Open either with the badge, or paste this into Colab's *File → Open notebook →
GitHub* box:

```
pablo-arantes/tmol-mutscan
```

---

## Read this before acting on any output

The benchmark has been run: 496 single-point interface mutations across 42
protein–protein complexes with measured binding ΔΔG, on a Colab T4. The results
are in the [main README](../README.md#how-accurate-is-it-benchmark-results).
The short version, because it changes how you read the notebook's output:

- Overall Pearson **r = 0.53**, but only **0.25** for effects under 3 kcal/mol,
  which is where most design decisions live.
- Accuracy falls with system size: **r ≈ 0.68** below 300 residues, **≈ 0.32**
  above 700. Each run prints the band for your complex.
- **No usable signal for stabilizing mutations** (Spearman ρ = 0.011 over 103
  of them). The tool cannot rank improvements, only shortlist them.
- Signal is concentrated at **core** interface positions (r = 0.57) and is weak
  at rim (0.30) and support (0.37).

Hence the `trust` column, which v0.3.0 attaches to every prediction:

| `trust` | criterion | measured performance |
|---|---|---|
| `none` | strain-flagged | r = −0.09, **sign wrong 84% of the time** — discard |
| `high` | core position, \|ΔΔG\| ≥ 2 REU, unflagged | r = 0.59, **sign right 95%** |
| `low` | everything else | r = 0.22, sign 70% — do not rank on it |

**Read `trust` before `ddg_mean`.** Both notebooks surface it.

---

## 1. Running a scan — `tmol_mutscan_colab.ipynb`

Runs the full `tmol-mutscan` pipeline on Colab's free GPU, for people without a
local GPU or cluster. Every option is a form field; no code editing.

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
| 9 | Show all figures inline, each with a caption on how to read it |
| 10 | Hit table (**trust first**), per-position summary with burial class, ensemble quality check |
| 11 | Zip for download, and optionally copy to Google Drive |

### What v0.3.0 changed in the output

- A **`trust`** and **`location`** column on every row of the hit table, and
  `n_high_trust` per position.
- `ddg_mean_heatmap.png` now **blanks** strain-flagged cells instead of drawing
  them, because those predictions were anti-predictive in benchmarking. They
  remain in the CSVs.
- A new **`ddg_high_trust_heatmap.png`** showing only the validated regime. If
  one figure should drive a decision, it is that one. An empty figure means
  nothing in your scan met the bar — which is informative, not a failure.
- Each run prints the **expected-accuracy band** for your complex's size.
- Two caveats the notebook now states, to apply by hand: stabilizing mutations
  cannot be ranked, and bulkier-and-hydrophobic substitutions were optimistic
  by ~0.8 kcal/mol (n = 28, p = 0.014) even at high trust.

### Practicalities

- **Turn the GPU on first:** Runtime → Change runtime type → T4 GPU.
- `--relax-repeats 1` is enough. Raising it to 4 moved the benchmark r by 0.005
  at four times the cost: the error is in the energy function, not the
  sampling. Spend compute on more positions instead.
- **Use mmCIF for ligands.** The notebook warns if it finds `HETATM` records in
  a `.pdb` input, since PDB files carry no bond orders.
- Free sessions disconnect after ~90 min idle and 12 h total, and a GPU is not
  guaranteed. Use the *save to Google Drive* option in cell 11 for long scans.

---

## 2. The accuracy benchmark — `tmol_mutscan_skempi_benchmark.ipynb`

This is the notebook that produced the numbers above. Re-run it to extend the
benchmark, to test a change to the method, or to check the result for yourself.

It pulls experimental ΔΔG from [SKEMPI 2.0](https://life.bsc.es/pid/skempi2/),
runs `tmol-mutscan` on the same mutations, and reports Pearson r, Spearman ρ,
the fitted REU → kcal/mol slope, sign agreement and an AUC — broken down by the
subsets that drive a decision.

Needs **v0.2.0 or newer** for `run_scan(explicit_pairs=...)` and the `strain`
diagnostic; v0.3.0 is the default.

### The decision-relevant outputs

- **Stabilizing subset** — the case peptide optimisation needs and the one the
  method is worst at. This is what established ρ = 0.011.
- **Within-complex ρ** — what you actually rely on. Measured median **+0.47**
  over 42 complexes with 39/42 positive, so ranking *within* one system works
  better than comparing across systems. But it ranges from +0.90 to −0.13, and
  you cannot tell in advance which you have.
- **Does the strain filter help?** Cell 10 tests it. It does: excluding flagged
  predictions lifted r from 0.530 to 0.563, and flagged predictions on their
  own are anti-predictive. That result is what `trust = none` encodes.

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
  finished complexes are skipped on re-run, so a Colab disconnect costs
  nothing. `save_to_drive` keeps the checkpoint off the VM.
- **Run with `n_complexes = 2` first** to calibrate per-complex cost on the GPU
  you were assigned. For reference, the 42-complex run took **6.4 h on a T4**
  at `relax_repeats = 4`; at `relax_repeats = 1` it would be far less for the
  same accuracy.
- `max_residues` matters more than it looks. The first run used 450 and thereby
  excluded **every antibody and TCR complex**, which flattered the result
  (r = 0.65 against 0.53 once they were included). Use 1200.
- `n_conformers = 3` is the default and is enough: ensemble noise is 0.5% of
  prediction variance.
- SKEMPI is protein–protein, so the numbers are an **upper bound** for
  peptide–protein and say nothing about protein–ligand.

### Worth doing if you extend it

The notebook records `strain` but not the v0.3.0 `trust`/`location` columns, so
validating the trust rule currently needs the burial classification to be
recomputed afterwards. Adding `trust` and `location` to the recorded columns in
cell 7 would let a re-run validate the rule directly.

---

## Editing the notebooks

Both are generated, but they are normal `.ipynb` files and can be edited
directly. Keep outputs stripped before committing so diffs stay readable:

```bash
jupyter nbconvert --clear-output --inplace colab/*.ipynb
```
