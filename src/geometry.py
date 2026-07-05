"""
proANUBIS geometry helpers.

Pure Python / NumPy -- no ROOT dependency -- so this module is importable and unit-
testable without the physics stack. The ML feature code and the tests both use it.

The proANUBIS prototype sits at a fixed direction in ATLAS (eta, phi) coordinates.
A muon is described by its own (eta, phi). How closely a muon points at proANUBIS is
captured by Delta-R, the standard collider angular distance:

    Delta-R = sqrt( (eta_muon - eta_proANUBIS)^2 + wrap(phi_muon - phi_proANUBIS)^2 )

where wrap() folds the phi difference into (-pi, pi] because phi is periodic.
Values from the proANUBIS validation notebook (validateATLASData.ipynb).
"""
import numpy as np

# Fixed proANUBIS acceptance direction, ATLAS (eta, phi) coordinates.
PANUBIS_ETA = 0.956
PANUBIS_PHI = 1.5


def wrap_dphi(dphi):
    """Fold a phi difference into (-pi, pi]. Works on scalars or arrays."""
    return np.arctan2(np.sin(dphi), np.cos(dphi))


def delta_r_to_proanubis(eta, phi):
    """Angular distance Delta-R from a muon (eta, phi) to the proANUBIS direction.

    Accepts scalars or NumPy arrays; returns the same shape.
    Small Delta-R  -> muon points at the detector (expect a reconstructed segment).
    Large Delta-R  -> muon points elsewhere / the eta-flipped control region.
    """
    deta = np.asarray(eta, dtype=float) - PANUBIS_ETA
    dphi = wrap_dphi(np.asarray(phi, dtype=float) - PANUBIS_PHI)
    return np.hypot(deta, dphi)
