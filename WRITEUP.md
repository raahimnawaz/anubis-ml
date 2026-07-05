# WRITEUP — Z-muon tagging on ATLAS proANUBIS open data

A short, honest record of what we built, what went wrong, and what we learned. The
mistake in the middle is the most useful part, so it's kept in full.

## Goal

Predict, from an ATLAS proANUBIS ntuple, whether a reconstructed muon came from a
Z-boson decay (`muon_isFromZ`). The label is real (physicist-assigned in the full
ATLAS reconstruction), so this is honest supervised learning, not a fabricated task.

## Data

[ATLAS proANUBIS Calibration Data Set](https://opendata.cern.ch/record/atlas-93943),
a 9-file / ~240 MB local subset of the 6 GiB release. ROOT TTree `analysis`, one row
per collision event, with variable-length arrays of muons, jets, and muon segments.
789,019 muons total; 14.2% flagged `isFromZ`.

## The hypothesis that was wrong (and why that's the point)

**Plan:** every muon here is high-pt (they all passed a muon trigger), so pt alone
can't separate Z muons from the rest. The textbook fix: a Z decays to two
opposite-charge muons whose combined *invariant mass* is ~91 GeV. Engineer that
dimuon mass as a feature and watch the model improve.

**What happened:** the engineered mass features scored ROC-AUC of *exactly* 0.5000 —
i.e. no signal at all. Not "weak," but dead-constant.

**Diagnosis:** a feature that is exactly 0.5000 is constant. Checking the data
explained it in one line:

| muons per event | count   |
|-----------------|---------|
| 1               | 788,391 |
| 2               | 628     |

Almost every event has **one** muon, so there is no partner to build a dimuon mass
with. Why? **Skimming.** proANUBIS sits on one side of the ATLAS cavern, and this
ntuple keeps only objects pointing toward it. A Z's two muons fly back-to-back, so
usually only one points at proANUBIS — the partner is discarded before the file is
written. The second muon wasn't lost by our code; it was never in the file.

**Lesson:** understand how your data was *selected* before you design features on it.
No modeling cleverness recovers information the skim already removed.

## The honest model

With the dimuon-mass path closed, we use only what survives the skim: the muon's own
kinematics, plus event context (trigger bits, object counts, pileup). We train the
same gradient-boosted-tree model twice to separate real skill from shortcuts:

| model                        | features                              | ROC-AUC |
|------------------------------|---------------------------------------|---------|
| full                         | kinematics + triggers + object counts | ~0.89   |
| kinematics only              | pt, eta, phi, charge, pileup          | ~0.70   |

Univariate check: `diMuTrigger` alone scores ~0.80. So most of the "full" model's
skill is just reading the dimuon-trigger bit — which fires when the *full* event had
two muons, i.e. it's a proxy for exactly the dimuon information the skim removed at
the object level. From the muon's motion alone, Z-tagging is only modestly better
than chance (~0.70), and pt does most of that work.

That ~0.70 is the real, defensible result. It's modest *because* of the skim, and
saying so plainly is the correct scientific outcome.

## Files

- `src/explore.py`  — inspect a ROOT file's branches / types / summary stats.
- `src/features.py` — explode per-event muon arrays into a per-muon table -> parquet.
- `src/train.py`    — full vs kinematics-only comparison.
- `src/diagnose.py` — the univariate-AUC / ablation script that exposed the null result.

## If continuing

The unused, genuinely proANUBIS-specific data is the **muon segments** (`mseg_x/y/z`
+ direction cosines) — the reconstructed trajectory hits in the prototype detector.
A more detector-focused task (e.g. does a muon have an associated proANUBIS segment,
or regress the segment direction from muon kinematics) would use the part of this
dataset that is actually novel, rather than re-deriving standard ATLAS muon physics.
