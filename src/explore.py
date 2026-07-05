"""
Step 1: figure out what's actually in an ANUBIS ntuple before writing any ML code.

Usage:
    python src/explore.py data/<some_file>.root [tree_name]

Prints every branch/column, its type, entry count, and a few summary stats.
Also drops a couple of quick histograms in data/plots/ so you can eyeball
that the file reads back sane values (pt in MeV, eta unitless, etc).
"""
import sys
from pathlib import Path

import ROOT


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    file_path = sys.argv[1]
    tree_name = sys.argv[2] if len(sys.argv) > 2 else "analysis"

    df = ROOT.RDataFrame(tree_name, file_path)

    columns = sorted(str(c) for c in df.GetColumnNames())
    n_entries = df.Count().GetValue()

    print(f"tree: {tree_name!r}  entries: {n_entries}")
    print(f"columns ({len(columns)}):")
    for c in columns:
        col_type = df.GetColumnType(c)
        print(f"  {c:30s} {col_type}")

    # Quick numeric summary for anything that looks like a scalar float/int column.
    # Vector<float> columns (e.g. per-event muon arrays) need Define()+flattening,
    # which we'll add to features.py once we know which columns are actually vectors.
    scalar_types = {"Float_t", "Double_t", "Int_t", "UInt_t", "Long64_t"}
    print("\nsummary stats (scalar columns only):")
    for c in columns:
        if df.GetColumnType(c) in scalar_types:
            stats = df.Describe() if hasattr(df, "Describe") else None
            mean = df.Mean(c)
            stddev = df.StdDev(c)
            print(f"  {c:30s} mean={mean.GetValue():.4g}  stddev={stddev.GetValue():.4g}")

    plots_dir = Path(file_path).parent / "plots"
    plots_dir.mkdir(exist_ok=True)

    ROOT.gROOT.SetBatch(True)
    for c in columns:
        if df.GetColumnType(c) in scalar_types and ("pt" in c.lower() or "eta" in c.lower()):
            h = df.Histo1D((c, c, 100, 0, 0), c)  # (0,0) -> auto-range
            canvas = ROOT.TCanvas()
            h.Draw()
            out = plots_dir / f"{c}.png"
            canvas.SaveAs(str(out))
            print(f"wrote {out}")


if __name__ == "__main__":
    main()
