"""
Generate a TINY synthetic ANUBIS-shaped .root file so the ROOT pipeline can be smoke-
tested (in CI, or offline) without the real 6 GB dataset.

This is NOT physics -- it just reproduces the branch *structure* of the real `analysis`
TTree, with enough variety that both tasks' feature/train scripts run and see both label
classes. The numbers are fabricated; do not draw conclusions from them.

Writes data/sample.ANALYSIS.root using uproot (no ROOT needed to *write*; the point is
that ROOT/RDataFrame then *reads* it in the same way it reads real files).

Usage:
    python tests/make_sample_root.py
"""
from pathlib import Path

import numpy as np
import awkward as ak
import uproot

# Fixed seed (Math.random-free); deterministic sample for reproducible CI.
rng = np.random.default_rng(12345)
N = 2000
PANUBIS_ETA, PANUBIS_PHI = 0.956, 1.5

# Half the events point AT proANUBIS (-> should have a segment), half are the eta-flipped
# control (-> no segment). Every event has exactly one muon, like the real skimmed data.
points_at = rng.random(N) < 0.5
eta = np.where(points_at,
               rng.normal(PANUBIS_ETA, 0.08, N),
               rng.normal(-PANUBIS_ETA, 0.08, N))
phi = rng.normal(PANUBIS_PHI, 0.08, N)
pt = rng.normal(35000.0, 8000.0, N).clip(5000.0)          # MeV
charge = rng.choice([-1.0, 1.0], N)

# ~15% flagged from-Z; give those the dimuon trigger + a pt bump so task 1 has signal.
is_z = rng.random(N) < 0.15
pt = pt + is_z * 6000.0
di_mu = is_z | (rng.random(N) < 0.05)
single_mu = np.ones(N, dtype=bool)
jet_trig = rng.random(N) < 0.3
pileup = rng.normal(45.0, 6.0, N).astype(np.float32)

# jagged branches: one list per event
muon_pt = ak.Array([[float(v)] for v in pt])
muon_eta = ak.Array([[float(v)] for v in eta])
muon_phi = ak.Array([[float(v)] for v in phi])
muon_charge = ak.Array([[float(v)] for v in charge])
muon_isFromZ = ak.Array([[bool(v)] for v in is_z])
# segments for muons that point at the detector, but ~97% efficient (not 100%) so the
# acceptance region contains a few negatives -> the efficiency model has both classes.
gets_segment = points_at & (rng.random(N) < 0.97)
n_seg = np.where(gets_segment, rng.integers(1, 5, N), 0)
mseg_z = ak.Array([list(rng.normal(0, 100, k).astype(float)) for k in n_seg])
# a few jets sprinkled in
n_jet = rng.integers(0, 3, N)
jet_pt = ak.Array([list(rng.normal(30000, 5000, k).astype(float)) for k in n_jet])

out = Path("data/sample.ANALYSIS.root")
out.parent.mkdir(exist_ok=True)
with uproot.recreate(out) as f:
    f["analysis"] = {
        "muon_pt": muon_pt,
        "muon_eta": muon_eta,
        "muon_phi": muon_phi,
        "muon_charge": muon_charge,
        "muon_isFromZ": muon_isFromZ,
        "mseg_z": mseg_z,
        "jet_pt": jet_pt,
        "diMuTrigger": di_mu,
        "singleMuTrigger": single_mu,
        "jetTrigger": jet_trig,
        "averageInteractionsPerCrossing": pileup,
    }

print(f"wrote {out}: {N} events, "
      f"{points_at.mean()*100:.0f}% point at proANUBIS, {is_z.mean()*100:.0f}% from-Z")
