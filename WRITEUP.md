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

That ~0.70 is the real, defensible result. It's modest *because* of the skim, and
saying so plainly is the correct scientific outcome.

---

# TASK 2 — proANUBIS segment detection (detector efficiency)

Task 1 re-derived standard ATLAS muon physics. Task 2 uses the part of the dataset that
is genuinely about *this detector*: the reconstructed **muon segments** (`mseg_*`), the
track hits proANUBIS actually recorded.

## Goal

Given a muon, predict whether proANUBIS reconstructed a segment for it (`has_segment`).
This is a detector **acceptance / efficiency** problem — the physics analogue of "does my
sensor detect an object, as a function of where it is in the field of view." We restrict to
single-muon events (~99.9% of the data) so the event's segment count unambiguously belongs
to that one muon.

The natural feature is geometric: **Delta-R**, the angular distance from the muon's
direction to proANUBIS's fixed direction (eta=0.956, phi=1.5). Unlike Task 1's invariant
mass, the object we need (the segment) *is* in the file, so this feature actually works.

## Two models, on purpose

| model                         | data                         | ROC-AUC | what it really measures |
|-------------------------------|------------------------------|---------|-------------------------|
| **A — acceptance**            | full sample (788k muons)     | **0.995** | mostly geometry: signal muons point at the detector, control muons point away |
| **B — efficiency**            | only muons with dR < 0.3 (413k) | **0.981** on a 98.4% base rate | the genuine detector response: of muons that *should* be seen, which aren't |

Model A's 0.995 looks spectacular but is largely a **re-derivation of the event selection** —
the dataset deliberately mixes proANUBIS-pointing muons with an eta-flipped control sample, and
Delta-R just tells them apart. `docs/figures/acceptance_map.png` shows this literally: a bright
P≈1 blob at the proANUBIS direction and a dark P≈0 blob at the mirror-image eta (the control),
with nothing in between.

Model B is the more honest question — restricted to muons that already point at the detector,
where ~98.4% leave a segment. Its AUC stays high (0.981) because the residual inefficiency isn't
random: it sits right at the **acceptance edge**, so fine Delta-R still separates the missed
muons, while pt barely matters. The physics finding is that proANUBIS's response is almost
entirely **geometric** — a sharp field-of-view with a thin turn-on edge — not kinematic.

**Lesson (a mirror of Task 1's):** a high headline AUC can mean the model learned your
*selection*, not your *physics*. The honest move is to ablate down to the regime where the
trivial signal is held constant, and report what's left.

## Files

- `src/explore.py`           — inspect a ROOT file's branches / types / summary stats.
- `src/geometry.py`          — pure-NumPy proANUBIS Delta-R helper (unit-tested, no ROOT).
- `src/features.py`          — Task 1 per-muon table -> `muons.parquet`.
- `src/train.py`             — Task 1 full vs kinematics-only comparison.
- `src/diagnose.py`          — Task 1 ablation script that exposed the skim.
- `src/features_segments.py` — Task 2 per-muon table -> `segments.parquet`.
- `src/train_segments.py`    — Task 2 acceptance vs efficiency comparison.
- `src/make_figures.py`      — Task 2 acceptance map + turn-on curve.
- `tests/`                   — geometry unit tests (run in CI, no ROOT/data needed).

## If continuing

A true estimation task would **regress the segment direction** (`mseg_?Dir`) from the muon
kinematics, or model efficiency vs. pileup — closer to sensor-fusion/state-estimation work.
Getting the full 423-file release would help only a segment-hungry deep model, not these
tree-based baselines (see the download discussion in the project notes).
