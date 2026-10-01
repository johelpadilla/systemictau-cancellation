"""Cancellation channel on the published coupled-logistic cascade.

Generator body copied from Fundacion_RECD/scripts/cascade_r_sweep.py
(logistic_coupled). Same constants as the excess3 accumulation note:
four maps, epsilon 0.05, noise 0.003, burn-in 400, 2200 steps,
seed = 10000 + int(r * 10000) + k * 17, k = 0..5.

r = 3.5675 is the accumulation station of that note.
r = 3.85 is the developed-chaos station of that note.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from systemictau_cancellation import cancellation_channel, decompose
from systemictau_cancellation.tau import all_pairs, tau_b_matrix

OUT_PNG = Path(__file__).resolve().parent / "cascade_channel.png"
OUT_CSV = Path(__file__).resolve().parent / "cascade_channel.csv"

STATIONS = (
    ("acumulación", 3.5675),
    ("caos", 3.85),
)
WINDOW = 13
N_CAL = 19
N_STEPS = 2200
BURN_IN = 400


def logistic_coupled(n_steps, n_comp=4, r=3.8, coupling=0.05, noise=0.003, seed=0, burn_in=400):
    """Diffusive weakly coupled logistic maps. Same recurrence as the note."""
    rng = np.random.default_rng(seed)
    X = np.zeros((n_steps + burn_in, n_comp))
    X[0] = rng.uniform(0.2, 0.8, n_comp)
    eps = coupling
    for t in range(1, n_steps + burn_in):
        f = r * X[t - 1] * (1.0 - X[t - 1])
        mean_others = (f.sum() - f) / max(n_comp - 1, 1)
        X[t] = (1.0 - eps) * f + eps * mean_others
        if noise > 0:
            X[t] += rng.normal(0.0, noise, n_comp)
        X[t] = np.clip(X[t], 0.0, 1.0)
    return X[burn_in:]


def seed_for(r: float, k: int) -> int:
    return 10_000 + int(float(r) * 10_000) + int(k) * 17


def level_split(X: np.ndarray, window: int, stride: int) -> dict:
    """Kappa of the pairwise taus themselves, baseline 0.

    This is the stationary reading of the same identity. The channel's
    kappa is the kappa of the change away from the calibration block.
    """
    pairs = all_pairs(X.shape[1])
    ends = np.arange(window - 1, X.shape[0], stride)
    tau_s = []
    D = []
    kappa = []
    for end in ends:
        block = X[end - window + 1:end + 1].T
        tau, _ = tau_b_matrix(block)
        part = decompose(tau[pairs[:, 0], pairs[:, 1]])
        tau_s.append(part["dbar"])
        D.append(part["D"])
        kappa.append(part["kappa"])
    return {
        "level_tau_s": float(np.mean(tau_s)),
        "level_abs_pair": float(np.mean(D)),
        "level_kappa": float(np.nanmean(kappa)),
    }


def one(r: float, k: int) -> dict:
    seed = seed_for(r, k)
    X = logistic_coupled(
        n_steps=N_STEPS, n_comp=4, r=r, coupling=0.05, noise=0.003,
        seed=seed, burn_in=BURN_IN,
    )
    out = cancellation_channel(X, window=WINDOW, stride=WINDOW, n_cal=N_CAL, alpha=0.05)
    mon = out.monitoring_mask()
    modes, counts = np.unique(out.mode[mon], return_counts=True)
    mix = {str(a): int(b) for a, b in zip(modes, counts)}
    n_mon = int(mon.sum())
    row = {
        "r": r,
        "k": k,
        "seed": seed,
        "n_monitoring": n_mon,
        "mean_tau_s": float(np.mean(out.tau_s[mon])),
        "mean_abs_tau_s": float(np.mean(np.abs(out.tau_s[mon]))),
        "mean_abs_dbar": float(np.mean(np.abs(out.dbar[mon]))),
        "mean_D": float(np.mean(out.D[mon])),
        "mean_kappa_change": float(np.nanmean(out.kappa[mon])),
        "n_quiet": mix.get("quiet", 0),
        "n_coherent": mix.get("coherent", 0),
        "n_cancellation": mix.get("cancellation", 0),
        "n_mixed": mix.get("mixed", 0),
    }
    row.update(level_split(X, WINDOW, WINDOW))
    tau = out.tau_s[mon]
    mode = out.mode[mon]
    row["frac_tau_ge_050"] = float(np.mean(tau >= 0.50))
    row["frac_tau_below_041"] = float(np.mean(np.abs(tau) < 0.41))
    row["frac_cancel_or_mixed"] = float(np.mean(np.isin(mode, ["cancellation", "mixed"])))
    gap = np.max(np.abs(out.D[mon] - (np.abs(out.dbar[mon]) + out.C[mon])))
    if gap > 1e-9:
        raise SystemExit(f"identity gap {gap} at r={r} k={k}")
    return row


def main():
    rows = []
    for _name, r in STATIONS:
        for k in range(6):
            row = one(r, k)
            rows.append(row)
            print(
                f"r={r:.4f} k={k} seed={row['seed']}  "
                f"tau_s={row['mean_tau_s']:+.3f}  |par|={row['level_abs_pair']:.3f}  "
                f"κ_nivel={row['level_kappa']:.3f}  "
                f"D_cambio={row['mean_D']:.3f}  κ_cambio={row['mean_kappa_change']:.3f}  "
                f"quiet {row['n_quiet']}/{row['n_monitoring']}  "
                f"coh {row['n_coherent']}  can {row['n_cancellation']}  mix {row['n_mixed']}"
            )

    fields = list(rows[0].keys())
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.3))
    palette = {"acumulación": "#C47B00", "caos": "#0072B2"}
    ax = axes[0]
    for name, r in STATIONS:
        group = [row for row in rows if row["r"] == r]
        ax.scatter(
            [row["mean_tau_s"] for row in group],
            [row["level_kappa"] for row in group],
            s=56, color=palette[name], label=f"{name}  r={r}", zorder=3,
        )
        for row in group:
            ax.annotate(str(row["k"]), (row["mean_tau_s"], row["level_kappa"]),
                        textcoords="offset points", xytext=(4, 4), fontsize=8, color=palette[name])
    ax.set_xlabel(r"media de $\tau_s$")
    ax.set_ylabel(r"$\kappa$ de nivel")
    ax.set_xlim(-0.05, 0.85)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(r"Cada semilla: media y $\kappa$ de nivel")

    ax = axes[1]
    for name, r in STATIONS:
        group = [row for row in rows if row["r"] == r]
        ax.scatter(
            [row["frac_tau_ge_050"] for row in group],
            [row["frac_cancel_or_mixed"] for row in group],
            s=56, color=palette[name], zorder=3,
        )
        for row in group:
            ax.annotate(str(row["k"]), (row["frac_tau_ge_050"], row["frac_cancel_or_mixed"]),
                        textcoords="offset points", xytext=(4, 4), fontsize=8, color=palette[name])
    ax.set_xlabel(r"fracción de ventanas con $\tau_s \geq 0.50$")
    ax.set_ylabel("fracción cancelación o mixta")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(r"Banda $\tau_s \geq 0.50$ y alarma de pares")
    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=160)
    print(f"\nfigura: {OUT_PNG}")
    print(f"tabla:  {OUT_CSV}")


if __name__ == "__main__":
    main()
