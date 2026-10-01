"""Kendall tau-b on one window.

Same definition as the technical note
*Cancellation-aware order coherence* (Padilla-Villanueva, 2026;
https://doi.org/10.5281/zenodo.23073069): tau-b is the cosine of the
pairwise sign vectors. A constant stream has an undefined coefficient;
it is stored as 0 and marked in the definedness mask.

This is tau-b. The Numba path of systemictau 4.6.1 divides by n(n-1)/2
and is tau-a. The two agree when the window has no ties.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=16)
def _pair_index(n_bins: int):
    return np.triu_indices(int(n_bins), 1)


def sign_vectors(Z: np.ndarray) -> np.ndarray:
    """Sign of every within-stream comparison. Z has shape (n_modules, n_bins)."""
    Z = np.asarray(Z, dtype=float)
    k, l = _pair_index(Z.shape[1])
    return np.sign(Z[:, l] - Z[:, k])


def tau_b_matrix(Z: np.ndarray):
    """Return (tau, defined) for one window. Z has shape (n_modules, n_bins)."""
    S = sign_vectors(Z)
    G = S @ S.T
    nrm = np.sqrt(np.clip(np.diag(G), 0.0, None))
    den = np.outer(nrm, nrm)
    defined = den > 0
    T = np.zeros_like(G, dtype=float)
    np.divide(G, den, out=T, where=defined)
    np.fill_diagonal(T, 1.0)
    return T, defined


def all_pairs(n_modules: int) -> np.ndarray:
    """Upper-triangle pairs as an array of shape (n_pairs, 2)."""
    i, j = np.triu_indices(int(n_modules), 1)
    return np.column_stack((i, j))
