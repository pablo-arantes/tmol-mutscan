# Colab notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pablo-arantes/tmol-mutscan/blob/main/colab/tmol_mutscan_colab.ipynb)

`tmol_mutscan_colab.ipynb` runs the full `tmol-mutscan` pipeline on Google
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

## Editing the notebook

The notebook is generated, but it is a normal `.ipynb` and can be edited
directly. Keep outputs stripped before committing so diffs stay readable:

```bash
jupyter nbconvert --clear-output --inplace colab/tmol_mutscan_colab.ipynb
```
