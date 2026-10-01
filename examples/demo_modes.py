"""Run the three illustrations and write a figure next to this file."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from systemictau_cancellation import cancellation_channel, format_report
from systemictau_cancellation.synthetic import cancellation_series, coherent_series, quiet_series

OUT = Path(__file__).resolve().parent / "modes_demo.png"


def _run(name, X, window):
    out = cancellation_channel(X, window=window, n_cal=19, alpha=0.05)
    print(f"\n== {name} ==")
    print(format_report(out))
    mon = out.monitoring_mask()
    identity_gap = np.max(np.abs(out.D[mon] - (np.abs(out.dbar[mon]) + out.C[mon])))
    if identity_gap > 1e-9:
        raise SystemExit(f"{name}: identity gap {identity_gap}")
    return out


MODE_COLOR = {
    "coherent": "#1d4ed8",
    "cancellation": "#c2410c",
    "mixed": "#7c3aed",
    "quiet": "#6b7280",
}
MODE_LABEL = {
    "coherent": "coherente",
    "cancellation": "cancelación",
    "mixed": "mixto",
    "quiet": "quieto",
}


def _panel(ax, out, title):
    x = np.arange(len(out.mode))
    ax.axvspan(-0.5, out.n_cal - 0.5, color="#e5e7eb", zorder=0)
    ax.plot(x, out.tau_s, color="#1f4e79", lw=1.4, label=r"$\tau_s$")
    mon = out.monitoring_mask()
    ax.plot(x[mon], out.D[mon], color="#b45309", lw=1.4, label=r"$D$")
    for i, mode in enumerate(out.mode):
        if mode == "calibration":
            continue
        ax.scatter([i], [out.tau_s[i]], s=22, color=MODE_COLOR[mode], zorder=3)
    ax.set_title(title)
    ax.set_ylabel("valor")
    ax.set_xlim(-0.5, len(out.mode) - 0.5)


def _legend(fig):
    handles = [
        plt.Line2D([], [], color="#1f4e79", lw=1.4, label=r"$\tau_s$"),
        plt.Line2D([], [], color="#b45309", lw=1.4, label=r"$D$"),
    ]
    for mode in ("quiet", "coherent", "cancellation", "mixed"):
        handles.append(
            plt.Line2D(
                [], [], color=MODE_COLOR[mode], marker="o", lw=0,
                label=MODE_LABEL[mode],
            )
        )
    fig.legend(handles=handles, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.02))


def main():
    window = 12
    quiet = _run("quieto", quiet_series(19, 12, window, 4, seed=11), window)
    coherent = _run("coherente", coherent_series(19, 8, window, 4, seed=12), window)
    cancelled = _run("cancelación", cancellation_series(19, 8, window, seed=13), window)

    fig, axes = plt.subplots(3, 1, figsize=(8.2, 7.6), sharex=False)
    _panel(axes[0], quiet, "Ruido independiente")
    _panel(axes[1], coherent, "Orden compartido")
    _panel(axes[2], cancelled, "Tres pares suben y tres bajan")
    axes[2].set_xlabel("ventana (la franja gris es la calibración)")
    _legend(fig)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT, dpi=160)
    print(f"\nfigura: {OUT}")


if __name__ == "__main__":
    main()
