import sys
from pathlib import Path

import numpy as np
import pytest

from systemictau_cancellation import (
    cancellation_channel,
    decompose,
    signed_tau_series,
    tau_b_matrix,
)
from systemictau_cancellation.synthetic import (
    cancellation_series,
    coherent_series,
    quiet_series,
)


def test_identity_on_random_changes():
    rng = np.random.default_rng(1)
    for _ in range(2000):
        d = rng.normal(size=int(rng.integers(2, 40)))
        r = decompose(d)
        assert r["D"] == pytest.approx(r["abs_dbar"] + r["C"], abs=1e-12)
        assert r["C"] == pytest.approx(r["C_minority"], abs=1e-12)
        if r["D"] > 0:
            assert r["kappa"] == pytest.approx(1.0 - r["abs_dbar"] / r["D"], abs=1e-12)


def test_kappa_extremes():
    assert decompose(np.array([0.1, 0.3, 0.0, 0.2]))["kappa"] == pytest.approx(0.0)
    assert decompose(np.array([0.1, -0.1, 0.4, -0.4]))["kappa"] == pytest.approx(1.0)


def test_channel_identity_and_index():
    X = quiet_series(n_cal=19, n_mon=6, window=12, n_modules=4, seed=2)
    out = cancellation_channel(X, window=12, n_cal=19)
    mon = out.monitoring_mask()
    assert np.allclose(out.D[mon], np.abs(out.dbar[mon]) + out.C[mon])
    assert out.mode[:19].tolist() == ["calibration"] * 19
    assert out.window_end[0] == 11
    assert out.windows_overlap is False
    series = signed_tau_series(X, window=12, stride=12)
    assert series[out.window_end] == pytest.approx(out.tau_s)


def test_coherent_series_moves_the_signed_mean():
    X = coherent_series(n_cal=19, n_mon=4, window=12, n_modules=4, seed=3)
    out = cancellation_channel(X, window=12, n_cal=19)
    mon = out.monitoring_mask()
    assert set(out.mode[mon]).issubset({"coherent", "mixed"})
    assert float(np.mean(out.dbar[mon])) > 0.7
    assert float(np.nanmean(out.kappa[mon])) < 0.15


def test_balanced_reversal_is_cancellation():
    X = cancellation_series(n_cal=19, n_mon=4, window=12, seed=4)
    out = cancellation_channel(X, window=12, n_cal=19)
    mon = out.monitoring_mask()
    assert set(out.mode[mon]) == {"cancellation"}
    assert abs(float(np.mean(out.dbar[mon]))) < 0.15
    assert float(np.nanmean(out.kappa[mon])) > 0.85
    assert float(np.mean(out.D[mon])) > 0.7


def test_quiet_series_rarely_alarms():
    X = quiet_series(n_cal=19, n_mon=40, window=10, n_modules=3, seed=5)
    out = cancellation_channel(X, window=10, n_cal=19)
    mon = out.mode[out.monitoring_mask()]
    alarmed = np.isin(mon, ["coherent", "cancellation", "mixed"]).mean()
    assert alarmed < 0.25


def test_short_calibration_is_refused():
    X = quiet_series(n_cal=5, n_mon=2, window=8, n_modules=3, seed=6)
    with pytest.raises(ValueError, match="n_cal"):
        cancellation_channel(X, window=8, n_cal=5, alpha=0.05)


def test_tau_b_matches_scipy_when_present():
    scipy_stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(7)
    Z = rng.poisson(0.8, size=(4, 30)).astype(float)
    T, defined = tau_b_matrix(Z)
    for i, j in [(0, 1), (0, 2), (1, 3)]:
        if defined[i, j]:
            ref = scipy_stats.kendalltau(Z[i], Z[j], variant="b").statistic
            assert T[i, j] == pytest.approx(ref, abs=1e-12)


def test_matches_published_note_code():
    note = Path("/Users/johelpadilla/grok-safe/cancellation-aware-tau/code")
    if not (note / "cancellation_tau" / "decomposition.py").is_file():
        pytest.skip("local note tree is absent")
    sys.path.insert(0, str(note))
    from cancellation_tau import decompose as note_decompose
    from cancellation_tau import tau_b_matrix as note_tau
    from cancellation_tau.calibration import conformal_pvalue as note_p
    from cancellation_tau.calibration import conformal_scores as note_scores
    from cancellation_tau.calibration import stat_D as note_stat_D
    from systemictau_cancellation.decomposition import conformal_pvalue, conformal_scores, stat_D

    rng = np.random.default_rng(20260930)
    Z = rng.poisson(0.8, size=(5, 30)).astype(float)
    got, got_defined = tau_b_matrix(Z)
    ref, ref_defined = note_tau(Z)
    assert np.allclose(got, ref)
    assert np.array_equal(got_defined, ref_defined)
    delta = rng.normal(size=15)
    a, b = decompose(delta), note_decompose(delta)
    for key in ("D", "dbar", "C", "kappa", "C_minority"):
        assert a[key] == pytest.approx(b[key], abs=1e-12)
    cal = rng.normal(size=(19, 6))
    new = rng.normal(size=6)
    got_scores, got_new = conformal_scores(cal, new, stat_D)
    ref_scores, ref_new = note_scores(cal, new, note_stat_D)
    assert got_scores == pytest.approx(ref_scores, abs=1e-12)
    assert got_new == pytest.approx(ref_new, abs=1e-12)
    assert conformal_pvalue(got_scores, got_new) == pytest.approx(note_p(ref_scores, ref_new))
