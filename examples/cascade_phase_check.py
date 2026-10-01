"""Is the accumulation cancellation alarm a phase of a long orbit?

Seeds 0, 3 and 5 at r=3.5675 kept a high signed mean and still alarmed.
This script asks whether that alarm is the same orbit seen through a
13-step window at different phases.

The generator is the one in cascade_channel.py. The published run uses
noise 0.003. The noise-free twin uses the same seed, so the same initial
condition, and then no kick.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cascade_channel import logistic_coupled, seed_for
from systemictau_cancellation import cancellation_channel
from systemictau_cancellation.tau import all_pairs, tau_b_matrix

OUT = Path(__file__).resolve().parent / "cascade_phase_check.png"
R = 3.5675
SEEDS = (0, 3, 5, 1)
LAGS = (8, 16, 32, 48, 64)


def return_rms(X: np.ndarray, lag: int) -> float:
    d = X[lag:] - X[:-lag]
    return float(np.sqrt(np.mean(d * d)))


def best_lag(X: np.ndarray, max_lag: int = 128, tol: float = 1e-8) -> tuple[int, float]:
    """Shortest return whose RMS is numerical noise. Else the lag of minimum RMS."""
    scores = [return_rms(X, lag) for lag in range(1, max_lag + 1)]
    for lag, score in enumerate(scores, start=1):
        if score < tol:
            return lag, score
    lag = int(np.argmin(scores)) + 1
    return lag, scores[lag - 1]


def cycle_gap(X: np.ndarray, period: int) -> float:
    """Smallest separation among the distinct samples of one clean cycle."""
    gaps = []
    for comp in range(X.shape[1]):
        pts = np.sort(X[:period, comp])
        gaps.append(float(np.min(np.diff(pts))))
    return float(np.min(gaps))


def pair_table(X: np.ndarray, window: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    pairs = all_pairs(X.shape[1])
    ends = np.arange(window - 1, X.shape[0], stride)
    values = np.empty((len(ends), pairs.shape[0]))
    for k, end in enumerate(ends):
        tau, _ = tau_b_matrix(X[end - window + 1:end + 1].T)
        values[k] = tau[pairs[:, 0], pairs[:, 1]]
    return ends, values


def phase_r2(values: np.ndarray, phase: np.ndarray) -> float:
    """Fraction of the pair-vector variance accounted for by the phase label."""
    grand = values.mean(axis=0)
    sst = float(np.sum((values - grand) ** 2))
    if sst == 0:
        return 1.0
    ssb = 0.0
    for label in np.unique(phase):
        block = values[phase == label]
        ssb += block.shape[0] * float(np.sum((block.mean(axis=0) - grand) ** 2))
    return ssb / sst


def mean_abs_dev(values: np.ndarray, phase: np.ndarray | None) -> float:
    """Mean over windows of D = mean_e |tau_e - baseline_e|."""
    if phase is None:
        base = values.mean(axis=0)
        return float(np.mean(np.abs(values - base)))
    total = 0.0
    for label in np.unique(phase):
        block = values[phase == label]
        base = block.mean(axis=0)
        total += float(np.sum(np.abs(block - base)))
    return total / values.size


def alarm_fraction(X: np.ndarray, window: int) -> tuple[float, float, float]:
    out = cancellation_channel(X, window=window, stride=window, n_cal=19, alpha=0.05)
    mon = out.monitoring_mask()
    mode = out.mode[mon]
    hit = np.isin(mode, ["cancellation", "mixed"])
    return (
        float(np.mean(hit)),
        float(np.mean(out.tau_s[mon])),
        float(np.mean(out.D[mon])),
    )


def main():
    print(f"r={R}  retorno RMS. El ruido publicado es 0.003.")
    print(f"{'k':>3} {'ruido':>7} {'mejor':>6} {'rms':>10} " + " ".join(f"p{p:>2}" for p in LAGS))
    rows = []
    for k in SEEDS:
        seed = seed_for(R, k)
        for noise in (0.0, 0.003):
            X = logistic_coupled(n_steps=2200, r=R, seed=seed, noise=noise, burn_in=400)
            lag, rms = best_lag(X)
            named = [return_rms(X, p) for p in LAGS]
            print(
                f"{k:3d} {noise:7.3f} {lag:6d} {rms:10.3e} "
                + " ".join(f"{v:8.3e}" for v in named)
            )
            rows.append((k, noise, X, lag))

    print("\nVentanas no solapadas. D_global compara cada ventana con la media de todas.")
    print("D_fase compara cada ventana con la media de su misma fase. La fase sale del retorno sin ruido.")
    print(f"{'k':>3} {'ruido':>7} {'P':>4} {'R2':>6} {'D_global':>9} {'D_fase':>9} {'D_fase/D':>9}")
    summary = []
    for k, noise, X, _lag in rows:
        clean = logistic_coupled(n_steps=2200, r=R, seed=seed_for(R, k), noise=0.0, burn_in=400)
        period, _ = best_lag(clean)
        ends, values = pair_table(X, window=13, stride=13)
        starts = ends - 12
        phase = starts % period
        r2 = phase_r2(values, phase)
        d_global = mean_abs_dev(values, None)
        d_phase = mean_abs_dev(values, phase)
        ratio = d_phase / d_global if d_global > 0 else float("nan")
        gap = cycle_gap(clean, period) if noise == 0.0 else float("nan")
        print(
            f"{k:3d} {noise:7.3f} {period:4d} {r2:6.3f} {d_global:9.4f} {d_phase:9.4f} "
            f"{ratio:9.3f}  gap={gap:.4f}"
        )
        summary.append({
            "k": k, "noise": noise, "period": period, "r2": r2,
            "d_global": d_global, "d_phase": d_phase,
        })

    print("\nAlarma de cancelación o mixta. La serie con ruido usa 8000 pasos para las ventanas largas.")
    print(f"{'k':>3} {'ruido':>7} " + " ".join(f"{'w'+str(w):>14}" for w in (13, 16, 32, 78)))
    alarm = []
    for k in SEEDS:
        seed = seed_for(R, k)
        for noise, n_steps in ((0.0, 2200), (0.003, 8000)):
            X = logistic_coupled(n_steps=n_steps, r=R, seed=seed, noise=noise, burn_in=400)
            parts = []
            for window in (13, 16, 32, 78):
                if (X.shape[0] - window) // window + 1 <= 19:
                    parts.append((float("nan"), float("nan"), float("nan")))
                    continue
                parts.append(alarm_fraction(X, window))
            alarm.append((k, noise, parts))
            text = " ".join(
                "  nan" if not np.isfinite(frac) else f"{frac:5.2f} τ{tau:+.2f}"
                for frac, tau, _D in parts
            )
            print(f"{k:3d} {noise:7.3f} {text}")

    print("\nSemilla 5, ruido 0.003. D de cada ventana contra la media de las 19 primeras.")
    X5 = logistic_coupled(n_steps=2200, r=R, seed=seed_for(R, 5), noise=0.003, burn_in=400)
    ends, values = pair_table(X5, 13, 13)
    base = values[:19].mean(axis=0)
    d_each = np.mean(np.abs(values - base), axis=1)
    print(f"  D media calibración {d_each[:19].mean():.3f}  monitoreo {d_each[19:].mean():.3f}")
    print(f"  D media ventanas 19-68 {d_each[19:69].mean():.3f}  ventanas 69-fin {d_each[69:].mean():.3f}")

    # Figure: return curve for seed 5, and D_global vs D_phase for the published noise.
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.2))
    ax = axes[0]
    seed5 = seed_for(R, 5)
    for noise, color, label in (
        (0.0, "#C47B00", "semilla 5, sin ruido"),
        (0.003, "#0072B2", "semilla 5, ruido 0.003"),
    ):
        X = logistic_coupled(n_steps=2200, r=R, seed=seed5, noise=noise, burn_in=400)
        lags = np.arange(1, 65)
        rms = [return_rms(X, int(lag)) for lag in lags]
        ax.plot(lags, rms, color=color, lw=1.4, label=label)
    ax.axvline(16, color="#6b7280", lw=0.8, ls="--")
    ax.set_xlabel("rezago")
    ax.set_ylabel("RMS del retorno")
    ax.set_title("Semilla 5: sin ruido el retorno cierra en 16")
    ax.legend(frameon=False, fontsize=8)
    ax.set_yscale("log")

    ax = axes[1]
    published = [row for row in summary if row["noise"] == 0.003]
    x = np.arange(len(published))
    width = 0.36
    ax.bar(x - width / 2, [row["d_global"] for row in published], width, color="#b45309", label="contra la media global")
    ax.bar(x + width / 2, [row["d_phase"] for row in published], width, color="#1f4e79", label="contra su misma fase")
    labels = []
    for row in published:
        labels.append(f"k={row['k']}\nP={row['period']}\nR²={row['r2']:.2f}")
    ax.set_xticks(x, labels)
    ax.set_ylabel("cambio absoluto medio de los pares")
    ax.set_title("Con el ruido publicado, la fase no baja D")
    ax.legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT, dpi=160)
    print(f"\nfigura: {OUT}")


if __name__ == "__main__":
    main()
