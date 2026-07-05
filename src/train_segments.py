"""
TASK 2 training: predict whether a muon produces a proANUBIS segment, and separate
the trivial part (geometry / selection) from the real physics (detector efficiency).

Two models, deliberately:

  MODEL A ("acceptance")  -> full sample, all features. Scores ~0.99 AUC, but most of
                             that is just Delta-R telling signal muons (point at the
                             detector) from control muons (point away). Impressive,
                             largely a re-derivation of the event selection.

  MODEL B ("efficiency")  -> ONLY muons already pointing at proANUBIS (dR < 0.3). Here
                             ~98% have a segment, so the question becomes the genuine
                             detector-efficiency one: of the muons that geometrically
                             should be seen, which ones aren't -- and why? Geometry
                             (edge effects) still helps; kinematics barely do.

The A-vs-B gap is the lesson: a high headline AUC can hide that the model learned your
selection, not your physics. (Same discipline as task 1's trigger ablation.)

Usage:
    python src/train_segments.py        # reads data/segments.parquet
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report

ALL_FEATURES = ["pt_gev", "eta", "phi", "charge",
                "dEta_proanub", "dPhi_proanub", "dR_proanub", "pileup", "n_jets"]
# inside the acceptance region dR is nearly constant, so the "efficiency" model leans
# on kinematics + fine edge geometry.
EFF_FEATURES = ["pt_gev", "eta", "phi", "charge", "dR_proanub", "pileup", "n_jets"]


def fit_report(name, df, feats, y):
    X = df[feats].astype(float).values
    tr, te = train_test_split(np.arange(len(df)), test_size=0.25,
                              random_state=42, stratify=y)
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1,
                                         max_depth=6, class_weight="balanced",
                                         random_state=42)
    clf.fit(X[tr], y[tr])
    proba = clf.predict_proba(X[te])[:, 1]
    auc = roc_auc_score(y[te], proba)
    print(f"\n{'='*60}\nMODEL: {name}   ({len(feats)} features, {len(df):,} muons)\n{'='*60}")
    print(f"ROC-AUC: {auc:.4f}   baseline P(segment)={y.mean():.3f}")
    print(classification_report(y[te], (proba >= 0.5).astype(int),
                                target_names=["no-seg", "has-seg"], digits=3))
    return auc


def main():
    path = "data/segments.parquet"
    if not Path(path).exists():
        print(f"{path} not found -- run: python src/features_segments.py")
        return
    df = pd.read_parquet(path)
    print(f"loaded {len(df):,} single-muon events, "
          f"{df['has_segment'].mean()*100:.1f}% with a segment")

    auc_a = fit_report("acceptance (full sample)", df,
                       ALL_FEATURES, df["has_segment"].astype(int).values)

    region = df[df["dR_proanub"] < 0.3].reset_index(drop=True)
    if region["has_segment"].nunique() < 2:
        # e.g. a tiny/synthetic sample where every in-acceptance muon has a segment:
        # AUC is undefined with one class. Report instead of emitting NaN.
        print("\nefficiency model skipped: only one class present in the acceptance "
              f"region (P(segment)={region['has_segment'].mean():.3f} over {len(region)} muons)")
        auc_b = None

    else:
        auc_b = fit_report("efficiency (muons pointing at proANUBIS, dR<0.3)", region,
                           EFF_FEATURES, region["has_segment"].astype(int).values)

    print(f"\n{'#'*60}")
    print(f"RESULT:  acceptance AUC = {auc_a:.4f}  (mostly geometry/selection)")
    if auc_b is not None:
        print(f"         efficiency AUC = {auc_b:.4f}  (the real detector-response signal,")
        print(f"                         on a {region['has_segment'].mean()*100:.1f}%-positive base rate)")
    print(f"{'#'*60}")


if __name__ == "__main__":
    main()
