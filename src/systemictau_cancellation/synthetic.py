"""Small series with a known order pattern.

These are illustrations for the channel. They are not the Monte Carlo of
the technical note (seed 20260930, 1000 windows).
"""
from __future__ import annotations

import numpy as np


def _independent(n_windows: int, window: int, n_modules: int, rng) -> np.ndarray:
    return rng.normal(size=(n_windows, window, n_modules))


def _ordered(window: int, n_modules: int, reversed_modules: tuple[int, ...]) -> np.ndarray:
    ticks = np.arange(window, dtype=float)
    block = np.tile(ticks[:, None], (1, n_modules))
    for j in reversed_modules:
        block[:, j] = ticks[::-1]
    return block


def stack_windows(blocks: np.ndarray) -> np.ndarray:
    """(n_windows, window, n_modules) -> (n_time, n_modules)."""
    n_windows, window, n_modules = blocks.shape
    return np.ascontiguousarray(blocks.reshape(n_windows * window, n_modules))


def quiet_series(n_cal: int, n_mon: int, window: int, n_modules: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    blocks = _independent(n_cal + n_mon, window, n_modules, rng)
    return stack_windows(blocks)


def coherent_series(n_cal: int, n_mon: int, window: int, n_modules: int, seed: int) -> np.ndarray:
    """Calibration is independent noise. Monitoring windows share one order."""
    rng = np.random.default_rng(seed)
    cal = _independent(n_cal, window, n_modules, rng)
    mono = _ordered(window, n_modules, ())
    mon = np.broadcast_to(mono, (n_mon, window, n_modules)).copy()
    return stack_windows(np.concatenate([cal, mon], axis=0))


def cancellation_series(n_cal: int, n_mon: int, window: int, seed: int) -> np.ndarray:
    """Four modules. Monitoring taus are +1,+1,+1,-1,-1,-1.

    Modules 0, 1 and 3 rise together. Module 2 is the reverse order.
    The signed mean of the six pairs is 0. Every pair is at ±1, so the
    absolute change is large when the calibration baseline is near 0.
    """
    rng = np.random.default_rng(seed)
    n_modules = 4
    cal = _independent(n_cal, window, n_modules, rng)
    mono = _ordered(window, n_modules, (2,))
    mon = np.broadcast_to(mono, (n_mon, window, n_modules)).copy()
    return stack_windows(np.concatenate([cal, mon], axis=0))
