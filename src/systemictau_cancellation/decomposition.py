"""Exact split of a vector of pairwise changes.

    D = |mean(delta)| + C
    kappa = C / D

C is the absolute change cancelled by opposite signs. The identity is the
one proved and checked in Padilla-Villanueva (2026),
https://doi.org/10.5281/zenodo.23073069.
"""
from __future__ import annotations

import numpy as np


def decompose(delta: np.ndarray) -> dict:
    """Split one vector of edge changes into signed mean, D, C and kappa."""
    d = np.asarray(delta, dtype=float).ravel()
    m = int(d.size)
    if m == 0:
        raise ValueError("empty edge set")
    s_plus = float(d[d > 0].sum())
    s_minus = float((-d[d < 0]).sum())
    D = float(np.abs(d).mean())
    dbar = float(d.mean())
    C = D - abs(dbar)
    if dbar > 0:
        minority = s_minus
    elif dbar < 0:
        minority = s_plus
    else:
        minority = s_plus
    C_min = 2.0 * minority / m
    kappa = C / D if D > 0 else float("nan")
    return {
        "D": D,
        "dbar": dbar,
        "abs_dbar": abs(dbar),
        "C": C,
        "C_minority": C_min,
        "kappa": kappa,
        "S_plus": s_plus,
        "S_minus": s_minus,
        "argmax": int(np.argmax(np.abs(d))),
    }


def stat_D(delta: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(delta, dtype=float))))


def stat_abs_mean(delta: np.ndarray) -> float:
    return float(abs(np.mean(np.asarray(delta, dtype=float))))


def conformal_scores(cal: np.ndarray, new: np.ndarray, stat) -> tuple[np.ndarray, float]:
    """Full-conformal scores. Calibration scores depend on the new window.

    For the monitored window the baseline is the mean of the calibration
    windows. Same construction as the technical note.
    """
    allw = np.vstack([np.asarray(cal, dtype=float), np.asarray(new, dtype=float)[None, :]])
    n1 = allw.shape[0]
    total = allw.sum(axis=0)
    base = (total[None, :] - allw) / (n1 - 1)
    scores = np.array([stat(allw[k] - base[k]) for k in range(n1)], dtype=float)
    return scores[:-1], float(scores[-1])


def conformal_pvalue(cal_scores: np.ndarray, new_score: float) -> float:
    cal_scores = np.asarray(cal_scores, dtype=float)
    return float((1.0 + np.sum(cal_scores >= new_score)) / (len(cal_scores) + 1.0))
