"""
Step 2: turn raw ANUBIS ntuples into a flat per-muon feature table for ML.

Each row = one muon. Label = muon_isFromZ (did this muon come from a Z boson?).

IMPORTANT data-understanding note (learned the hard way -- see WRITEUP.md):
This dataset is *skimmed* to keep only objects whose trajectory points toward the
proANUBIS detector, which sits on ONE side of the ATLAS cavern. A Z boson decays to
two back-to-back muons, so typically only ONE of them points at proANUBIS -- the
partner is dropped before the file is written. Result: ~99.9% of events contain
exactly one muon. So the classic "reconstruct the Z from the dimuon invariant mass"
feature is IMPOSSIBLE here (there's no second muon to pair with). We don't attempt it.

What we CAN use, honestly:
  - the muon's own kinematics (pt, eta, phi, charge)
  - event context that survives skimming: trigger bits, pileup, and the COUNTS of
    other reconstructed objects (segments, jets) in the event.

Usage:
    python src/features.py                # process all data/*.root -> data/muons.parquet
    python src/features.py data/one.root  # process a single file
"""
import sys
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import ROOT

EVENT_COLS = [
    "diMuTrigger", "singleMuTrigger", "jetTrigger",
    "averageInteractionsPerCrossing",  # pileup
]
# vector branches we only need the LENGTH of (per-event object counts)
COUNT_COLS = ["jet_pt", "mseg_z"]
MUON_COLS = ["muon_pt", "muon_eta", "muon_phi", "muon_charge", "muon_isFromZ"]


def process_file(path):
    df = ROOT.RDataFrame("analysis", path)
    data = df.AsNumpy(EVENT_COLS + COUNT_COLS + MUON_COLS)
    n_events = len(data["muon_pt"])

    rows = []
    for ev in range(n_events):
        pt = np.asarray(data["muon_pt"][ev], dtype=float)
        eta = np.asarray(data["muon_eta"][ev], dtype=float)
        phi = np.asarray(data["muon_phi"][ev], dtype=float)
        charge = np.asarray(data["muon_charge"][ev], dtype=float)
        is_z = np.asarray(data["muon_isFromZ"][ev], dtype=bool)
        n_mu = pt.size
        if n_mu == 0:
            continue

        n_jets = len(data["jet_pt"][ev])
        n_segs = len(data["mseg_z"][ev])

        for i in range(n_mu):
            rows.append({
                "isFromZ": bool(is_z[i]),
                # --- muon kinematics (the honest "from the muon itself" features) ---
                "pt_gev": pt[i] / 1000.0,
                "eta": eta[i],
                "abs_eta": abs(eta[i]),
                "phi": phi[i],
                "charge": charge[i],
                # --- event context that survives the skim ---
                "n_muons": n_mu,
                "n_jets": n_jets,
                "n_segments": n_segs,
                "diMuTrigger": bool(data["diMuTrigger"][ev]),
                "singleMuTrigger": bool(data["singleMuTrigger"][ev]),
                "jetTrigger": bool(data["jetTrigger"][ev]),
                "pileup": float(data["averageInteractionsPerCrossing"][ev]),
            })
    return rows


def main():
    args = sys.argv[1:]
    files = args if args else sorted(glob.glob("data/*.ANALYSIS.root"))
    if not files:
        print("no input files found under data/")
        sys.exit(1)

    all_rows = []
    for f in files:
        rows = process_file(f)
        all_rows.extend(rows)
        print(f"{Path(f).name}: {len(rows)} muons")

    df = pd.DataFrame(all_rows)
    out = "data/muons.parquet"
    df.to_parquet(out, index=False)
    print(f"\nwrote {out}: {len(df)} muons, "
          f"{df['isFromZ'].sum()} from Z ({100*df['isFromZ'].mean():.1f}%)")


if __name__ == "__main__":
    main()
