# anubis-ml

Machine learning on real ATLAS proANUBIS calibration data, using ROOT's `RDataFrame`.

**The task:** predict whether a muon came from a Z boson decay, from its kinematics.
It's a deliberately instructive problem — see [Why this task](#why-this-task).

## Data

[ATLAS proANUBIS Calibration Data Set](https://opendata.cern.ch/record/atlas-93943) — real (not simulated)
13.6 TeV pp collision data from 2024-2025, 49.1M events across 423 ROOT files (6.0 GiB total),
DOI `10.7483/OPENDATA.ATLAS.2J92.7ASX`.

Each file is a ROOT TTree named `analysis`. Per-event it stores variable-length arrays of
muons (`muon_pt/eta/phi/charge`, plus truth flags `muon_isFromZ`, `muon_isFromJPsi`), jets
(`jet_pt/eta/phi/M/EMRatio`), and muon segments (`mseg_x/y/z` + directions — the proANUBIS
trajectory info), alongside event-level trigger flags and timing. Units are ATLAS default:
**MeV, mm, ns**.

This repo works against a 9-file local subset (~240 MB) in `data/`. Only `.gitkeep` is
committed; download your own files from the portal.

## Why this task

Every muon in this dataset already passed a muon trigger, so they're *all* high-pt — you
cannot separate Z muons from the rest by pt alone. The original plan was to engineer the
**dimuon invariant mass** (Z → two opposite-charge muons at ~91 GeV) as the killer feature.

**That plan failed, and the failure is the most useful thing in this repo.** The data is
skimmed to keep only muons pointing at proANUBIS (one side of the detector), so ~99.9% of
events contain a *single* muon — the Z's partner was thrown away before the file was written.
No partner ⇒ no dimuon mass. Full story in [WRITEUP.md](WRITEUP.md).

So `train.py` does the honest comparison instead:

- **Model A ("full")** — kinematics + trigger bits + object counts → **AUC ~0.89**, but mostly
  from reading the `diMuTrigger` bit (a proxy for "the full event had two muons").
- **Model B ("kinematics only")** — the muon's own motion, no shortcuts → **AUC ~0.70**. This is
  the real "can you tell a Z muon from how it moves?" answer. Modest, *because* the skim removed
  the partner that would have made it easy.

The takeaway — *understand how your data was selected before you trust a feature* — transfers
directly to robotics perception/estimation.

(Note: the J/ψ trigger stream isn't in this subset — only 3 J/ψ muons across all 9 files — so
"Z vs J/ψ" isn't viable here either.)

## Setup

ROOT's Python bindings (PyROOT/RDataFrame) are Linux/macOS only — there is **no** conda or pip
build for native Windows. On Windows, run everything inside **WSL2**:

```bash
# in WSL (Ubuntu):
# 1. install miniforge (one time)
curl -fsSL -o ~/miniforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash ~/miniforge.sh -b -p ~/miniforge3

# 2. create the env
~/miniforge3/bin/conda env create -f /mnt/c/Users/Raahim/Downloads/anubis-ml/environment.yml

# 3. sanity check
~/miniforge3/envs/anubis-ml/bin/python -c "import ROOT; print(ROOT.__version__)"
```

On Linux/macOS natively it's just `conda env create -f environment.yml && conda activate anubis-ml`.

## Pipeline

Run from the project root (in WSL, prefix with the env's python as above):

```bash
python src/explore.py data/<file>.root   # 1. dump branches / types / summary stats
python src/features.py                    # 2. all data/*.root -> data/muons.parquet (per-muon table)
python src/train.py                       # 3. full vs kinematics-only models, print AUC comparison
python src/diagnose.py                    # (optional) univariate AUC + ablations — how we found the skim
```

- `src/explore.py` — inspects a file: every branch, its type, scalar summary stats, quick histograms.
- `src/features.py` — explodes per-event muon arrays into a flat per-muon table of honest
  features (muon kinematics + event context that survives the skim); writes `data/muons.parquet`.
- `src/train.py` — trains `HistGradientBoostingClassifier` on the full feature set and on
  kinematics only, reports ROC-AUC / precision / recall for each, and the gap between them.
- `src/diagnose.py` — the univariate-AUC and ablation script that exposed why the original
  dimuon-mass idea was dead (see [WRITEUP.md](WRITEUP.md)).
