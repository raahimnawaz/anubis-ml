"""
Diagnostic: WHY did the engineered mass features not help?

Hypothesis: the "raw" set already leaks the answer through diMuTrigger / n_muons
(Z->mumu events fire the dimuon trigger and have >=2 muons), so the invariant-mass
feature is redundant. We test this three ways:

  1. Univariate AUC of every feature on its own -- which single columns carry signal?
  2. A model on kinematics ONLY (drop trigger + multiplicity) -- the honest "can you
     tell from the muon itself?" baseline.
  3. That same kinematics-only model + the mass features -- NOW does physics help?
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

df = pd.read_parquet("data/muons.parquet")
y = df["isFromZ"].astype(int).values

ALL = ["pt_gev", "eta", "phi", "charge", "n_muons", "diMuTrigger",
       "singleMuTrigger", "pileup", "best_dimuon_mass_gev", "mass_dist_to_z",
       "has_os_partner", "best_partner_pt_gev"]

# 1. univariate AUC (does the raw column, on its own, separate Z from not-Z?)
print("univariate ROC-AUC (feature alone):")
for c in ALL:
    v = df[c].astype(float).values
    auc = roc_auc_score(y, v)
    auc = max(auc, 1 - auc)  # direction-agnostic
    print(f"  {c:24s} {auc:.4f}")

tr, te = train_test_split(np.arange(len(df)), test_size=0.25,
                          random_state=42, stratify=y)


def run(name, feats):
    X = df[feats].astype(float).values
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1,
                                         max_depth=6, class_weight="balanced",
                                         random_state=42)
    clf.fit(X[tr], y[tr])
    auc = roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1])
    print(f"  {name:38s} AUC = {auc:.4f}")
    return auc


# 2 & 3: strip the "is this a dimuon event" shortcut, then see if mass rescues it.
KINEMATIC = ["pt_gev", "eta", "phi", "charge", "pileup"]
MASS = ["best_dimuon_mass_gev", "mass_dist_to_z", "has_os_partner", "best_partner_pt_gev"]

print("\nmodels:")
run("kinematics only (no trigger/multiplicity)", KINEMATIC)
run("kinematics + physics mass features", KINEMATIC + MASS)
run("mass features ALONE", MASS)
