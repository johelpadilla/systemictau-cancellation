"""Cancellation channel for a multivariate series.

Input layout matches systemictau: X has shape (n_time, n_modules).
The signed mean on each window is tau-b. D, C and kappa are measured
against the mean of a calibration block of non-overlapping windows.

Modes on a monitoring window:

- coherent: the signed mean leaves the calibration range
- cancellation: the absolute change leaves that range and the signed mean does not
- mixed: both scores leave the range
- quiet: neither score leaves the range

The gate g(tau_s), the accumulated clock T_n and Protocol.frozen_v1 are
left untouched. This module does not import them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .decomposition import conformal_pvalue, conformal_scores, decompose, stat_D, stat_abs_mean
from .tau import all_pairs, tau_b_matrix

NOTE_DOI = "10.5281/zenodo.23073069"
NOTE_RECORD = "https://doi.org/10.5281/zenodo.23073069"


def minimum_calibration_windows(alpha: float) -> int:
    """Smallest n_cal whose conformal p-value can fall at or under alpha."""
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")
    return int(math.ceil(1.0 / float(alpha) - 1.0))


@dataclass
class ChannelResult:
    """One row per window. Calibration rows leave the scores empty."""

    window_end: np.ndarray
    tau_s: np.ndarray
    dbar: np.ndarray
    D: np.ndarray
    C: np.ndarray
    kappa: np.ndarray
    p_D: np.ndarray
    p_signed: np.ndarray
    mode: np.ndarray
    undefined_fraction: np.ndarray
    baseline: np.ndarray
    pairs: np.ndarray
    alpha: float
    n_cal: int
    window: int
    stride: int
    windows_overlap: bool
    note_doi: str = NOTE_DOI

    def monitoring_mask(self) -> np.ndarray:
        return self.mode != "calibration"

    def summary(self) -> dict:
        mon = self.monitoring_mask()
        modes, counts = np.unique(self.mode[mon], return_counts=True)
        return {
            "n_monitoring": int(mon.sum()),
            "mean_abs_dbar": float(np.mean(np.abs(self.dbar[mon]))) if mon.any() else float("nan"),
            "mean_D": float(np.mean(self.D[mon])) if mon.any() else float("nan"),
            "mean_kappa": float(np.nanmean(self.kappa[mon])) if mon.any() else float("nan"),
            "modes": {str(k): int(v) for k, v in zip(modes, counts)},
        }


def format_report(result: ChannelResult) -> str:
    """Spanish one-screen summary of a channel result."""
    s = result.summary()
    modes = ", ".join(f"{k}: {v}" for k, v in sorted(s["modes"].items()))
    overlap = ""
    if result.windows_overlap:
        overlap = (
            "\nLas ventanas se solapan. El valor p describe el puntaje; "
            "la nota calibra ventanas no solapadas."
        )
    return (
        f"Canal de cancelación (nota {result.note_doi})\n"
        f"ventana={result.window}  paso={result.stride}  "
        f"calibración={result.n_cal}  alpha={result.alpha}\n"
        f"ventanas de monitoreo: {s['n_monitoring']}\n"
        f"media |Δ̄| = {s['mean_abs_dbar']:.4f}\n"
        f"media D    = {s['mean_D']:.4f}\n"
        f"media κ    = {s['mean_kappa']:.4f}\n"
        f"modos: {modes}"
        f"{overlap}"
    )


def _as_series(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must have shape (n_time, n_modules)")
    if X.shape[1] < 2:
        raise ValueError("at least two modules are required")
    if not np.isfinite(X).all():
        raise ValueError("X contains a non-finite value")
    return X


def _pairs_of(n_modules: int, pairs) -> np.ndarray:
    if pairs is None:
        return all_pairs(n_modules)
    out = np.asarray(pairs, dtype=int)
    if out.ndim != 2 or out.shape[1] != 2 or out.shape[0] == 0:
        raise ValueError("pairs must have shape (n_pairs, 2)")
    if out.min() < 0 or out.max() >= n_modules:
        raise ValueError("a pair index is outside the module range")
    if np.any(out[:, 0] == out[:, 1]):
        raise ValueError("a pair repeats the same module")
    return out


def _window_ends(n_time: int, window: int, stride: int) -> np.ndarray:
    if window < 2:
        raise ValueError("window must be at least 2")
    if stride < 1:
        raise ValueError("stride must be at least 1")
    if n_time < window:
        raise ValueError("the series is shorter than one window")
    return np.arange(window - 1, n_time, stride, dtype=int)


def _pair_rows(X: np.ndarray, window: int, stride: int, pairs: np.ndarray):
    ends = _window_ends(X.shape[0], window, stride)
    n = len(ends)
    m = pairs.shape[0]
    values = np.empty((n, m), dtype=float)
    undefined = np.empty(n, dtype=float)
    left = pairs[:, 0]
    right = pairs[:, 1]
    for k, end in enumerate(ends):
        block = X[end - window + 1:end + 1].T
        tau, defined = tau_b_matrix(block)
        values[k] = tau[left, right]
        undefined[k] = 1.0 - float(defined[left, right].mean())
    return ends, values, undefined


def signed_tau_series(X: np.ndarray, window: int = 13, stride: int = 1, pairs=None) -> np.ndarray:
    """Signed mean of pairwise tau-b, indexed like systemictau.compute_taus.

    The value sits at the last time index of each window. Earlier positions,
    and positions skipped by the stride, are nan.
    """
    X = _as_series(X)
    chosen = _pairs_of(X.shape[1], pairs)
    ends, values, _ = _pair_rows(X, int(window), int(stride), chosen)
    taus = np.full(X.shape[0], np.nan)
    taus[ends] = values.mean(axis=1)
    return taus


def _mode(p_D: float, p_signed: float, alpha: float) -> str:
    hit_D = p_D <= alpha
    hit_signed = p_signed <= alpha
    if hit_signed and hit_D:
        return "mixed"
    if hit_signed:
        return "coherent"
    if hit_D:
        return "cancellation"
    return "quiet"


def cancellation_channel(
    X: np.ndarray,
    window: int = 13,
    stride: int | None = None,
    n_cal: int = 19,
    alpha: float = 0.05,
    pairs=None,
) -> ChannelResult:
    """Baseline the first n_cal windows and label every later window.

    The default stride equals the window, so the blocks do not overlap.
    That is the layout in which the note calibrates the conformal p-value,
    and the calibration applies when those blocks are exchangeable.
    On a continuous orbit the windows stay serially dependent even when they
    do not overlap, so the p-value is descriptive. A smaller stride is
    accepted; ChannelResult.windows_overlap is then true and the report says so.

    A mode of ``cancellation`` means D left the calibration range and the
    signed mean did not. On a short cycle this can be noise flipping ranks.
    """
    X = _as_series(X)
    window = int(window)
    if stride is None:
        stride = window
    stride = int(stride)
    n_cal = int(n_cal)
    alpha = float(alpha)
    need = minimum_calibration_windows(alpha)
    if n_cal < need:
        raise ValueError(
            f"n_cal={n_cal} cannot reach alpha={alpha}. "
            f"Use at least {need} calibration windows."
        )
    chosen = _pairs_of(X.shape[1], pairs)
    ends, values, undefined = _pair_rows(X, window, stride, chosen)
    if len(ends) <= n_cal:
        raise ValueError(
            f"the series yields {len(ends)} windows and n_cal={n_cal}. "
            "One monitoring window is required."
        )

    n = len(ends)
    dbar = np.full(n, np.nan)
    D = np.full(n, np.nan)
    C = np.full(n, np.nan)
    kappa = np.full(n, np.nan)
    p_D = np.full(n, np.nan)
    p_signed = np.full(n, np.nan)
    mode = np.full(n, "calibration", dtype="<U12")
    cal = values[:n_cal]
    baseline = cal.mean(axis=0)

    for k in range(n_cal, n):
        part = decompose(values[k] - baseline)
        scores_D, new_D = conformal_scores(cal, values[k], stat_D)
        scores_s, new_s = conformal_scores(cal, values[k], stat_abs_mean)
        dbar[k] = part["dbar"]
        D[k] = part["D"]
        C[k] = part["C"]
        kappa[k] = part["kappa"]
        p_D[k] = conformal_pvalue(scores_D, new_D)
        p_signed[k] = conformal_pvalue(scores_s, new_s)
        mode[k] = _mode(p_D[k], p_signed[k], alpha)

    return ChannelResult(
        window_end=ends,
        tau_s=values.mean(axis=1),
        dbar=dbar,
        D=D,
        C=C,
        kappa=kappa,
        p_D=p_D,
        p_signed=p_signed,
        mode=mode,
        undefined_fraction=undefined,
        baseline=baseline,
        pairs=chosen,
        alpha=alpha,
        n_cal=n_cal,
        window=window,
        stride=stride,
        windows_overlap=stride < window,
    )
