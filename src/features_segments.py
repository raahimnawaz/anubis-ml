"""
Feature builder for TASK 2: proANUBIS segment detection.

Question: given a muon, does the proANUBIS detector reconstruct a track segment for
it? This is a detector *acceptance / efficiency* problem -- the physics analogue of
"does my sensor detect an object, as a function of where it is in the field of view."

Unlike the Z-tagging task (task 1), the object we care about -- the muon segment --
IS present in the file, so the physics-motivated geometric feature (Delta-R to the
proANUBIS direction) actually carries signal here.

We restrict to single-muon events (~99.9% of the data). In those, the event's segment
count unambiguously belongs to that one muon, so the label is clean.

Output: data/segments.parquet, one row per muon.
    label  = has_segment  (event had >= 1 reconstructed muon segment)
Usage:
    python src/features_segments.py                # all data/*.root
    python src/features_segments.py data/one.root
"""
import sys
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import ROOT

from geometry import delta_r_to_proanubis, PANUBIS_ETA, PANUBIS_PHI, wrap_dphi

EVENT_COLS = ["averageInteractionsPerCrossing", "singleMuTrigger"]
COUNT_COLS = ["jet_pt", "mseg_z"]
MUON_COLS = ["muon_pt", "muon_eta", "muon_phi", "muon_charge"]


def process_file(path):
    df = ROOT.RDataFrame("analysis", path)
    data = df.AsNumpy(EVENT_COLS + COUNT_COLS + MUON_COLS)
    rows = []
    for ev in range(len(data["muon_pt"])):
        eta = np.asarray(data["muon_eta"][ev], dtype=float)
        if eta.size != 1:            # single-muon events only -> unambiguous label
            continue
        pt = float(data["muon_pt"][ev][0]) / 1000.0
        e = float(eta[0])
        phi = float(data["muon_phi"][ev][0])
        n_seg = len(data["mseg_z"][ev])
        rows.append({
            "has_segment": n_seg > 0,
            "n_segments": n_seg,
            # muon kinematics
            "pt_gev": pt,
            "eta": e,
            "phi": phi,
            "charge": float(data["muon_charge"][ev][0]),
            # engineered geometry: how well the muon points at proANUBIS
            "dEta_proanub": e - PANUBIS_ETA,
            "dPhi_proanub": float(wrap_dphi(phi - PANUBIS_PHI)),
            "dR_proanub": float(delta_r_to_proanubis(e, phi)),
            # event context
            "pileup": float(data["averageInteractionsPerCrossing"][ev]),
            "n_jets": len(data["jet_pt"][ev]),
        })
    return rows


def main():
    files = sys.argv[1:] or sorted(glob.glob("data/*.ANALYSIS.root"))
    if not files:
        print("no input files found under data/")
        sys.exit(1)

    all_rows = []
    for f in files:
        rows = process_file(f)
        all_rows.extend(rows)
        print(f"{Path(f).name}: {len(rows)} single-muon events")

    df = pd.DataFrame(all_rows)
    out = "data/segments.parquet"
    df.to_parquet(out, index=False)
    print(f"\nwrote {out}: {len(df)} muons, "
          f"{df['has_segment'].mean()*100:.1f}% with a proANUBIS segment")


if __name__ == "__main__":
    main()
