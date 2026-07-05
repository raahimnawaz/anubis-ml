"""
Step 3: train a classifier to predict whether a muon came from a Z boson,
and -- the actually interesting part -- measure how much of that ability is
"real physics from the muon" vs "just reading the dimuon-trigger bit".

We train the same gradient-boosted-tree model on two feature sets:

  MODEL A ("full")        -> everything, including trigger bits & object counts.
                             Scores high (~0.89 AUC), but see MODEL B before
                             getting excited.
  MODEL B ("kinematics")  -> ONLY the muon's own motion (pt, eta, phi, charge,
                             pileup). No trigger, no multiplicity. This is the
                             honest "can you tell from the muon itself?" number.

The gap between them is how much the classifier leans on the trigger shortcut.
Why does the trigger help so much? Because diMuTrigger fires when the *full* event
had two muons -- i.e. a Z candidate -- even though the skim saved only one of them.
So the trigger bit is a proxy for exactly the dimuon information the skim removed.
(See WRITEUP.md for the whole story.)

Usage:
    python src/train.py        # reads data/muons.parquet
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

FULL_FEATURES = [
    "pt_gev", "eta", "abs_eta", "phi", "charge",
    "n_muons", "n_jets", "n_segments",
    "diMuTrigger", "singleMuTrigger", "jetTrigger", "pileup",
]
# strictly the muon's own motion -- no event-level shortcuts
KINEMATIC_FEATURES = ["pt_gev", "eta", "abs_eta", "phi", "charge", "pileup"]


def train_and_eval(name, df, feats, idx_train, idx_test, y):
    X = df[feats].astype(float).values
    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.1, max_depth=6,
        class_weight="balanced", random_state=42,
    )
    clf.fit(X[idx_train], y[idx_train])

    proba = clf.predict_proba(X[idx_test])[:, 1]
    pred = (proba >= 0.5).astype(int)
    auc = roc_auc_score(y[idx_test], proba)

    print(f"\n{'='*60}\nMODEL: {name}   ({len(feats)} features)\n{'='*60}")
    print(f"ROC-AUC: {auc:.4f}")
    print(classification_report(y[idx_test], pred,
                                target_names=["not-Z", "from-Z"], digits=3))
    print("confusion matrix [rows=true, cols=pred]:")
    print(confusion_matrix(y[idx_test], pred))
    return auc


def main():
    path = "data/muons.parquet"
    if not Path(path).exists():
        print(f"{path} not found -- run: python src/features.py")
        return
    df = pd.read_parquet(path)
    print(f"loaded {len(df):,} muons, {df['isFromZ'].mean()*100:.1f}% from Z")

    y = df["isFromZ"].astype(int).values
    idx_train, idx_test = train_test_split(
        np.arange(len(df)), test_size=0.25, random_state=42, stratify=y,
    )

    auc_full = train_and_eval("full (with trigger + counts)", df, FULL_FEATURES,
                              idx_train, idx_test, y)
    auc_kin = train_and_eval("kinematics only (muon motion)", df, KINEMATIC_FEATURES,
                             idx_train, idx_test, y)

    print(f"\n{'#'*60}")
    print(f"RESULT:  full AUC = {auc_full:.4f}   kinematics-only AUC = {auc_kin:.4f}")
    print(f"         -> {auc_full-auc_kin:.4f} of the skill comes from the trigger/"
          f"count shortcut,")
    print(f"            not from the muon's own kinematics.")
    print(f"{'#'*60}")


if __name__ == "__main__":
    main()
