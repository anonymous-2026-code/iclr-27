#!/usr/bin/env python3
"""Render Figures 1 and 2 from submitted or freshly aggregated cells."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CELLS = None
OBJECTIVE = None
OUT = ROOT / "outputs/figures"
TAU = 0.005


def outcome_figure() -> None:
    frame = pd.read_csv(CELLS)
    assert len(frame) == 180
    colors = {"BI": "#2b8cbe", "OOG": "#31a354", "NI": "#636363", "POG": "#de2d26"}
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    for category in ("NI", "OOG", "BI", "POG"):
        part = frame[frame["corrected_auprc_category"] == category]
        if part.empty:
            continue
        ax.scatter(
            100 * part["corrected_auprc_oracle_delta"],
            100 * part["corrected_auprc_practical_delta"],
            s=25,
            alpha=0.78,
            color=colors[category],
            label=f"{category} (n={len(part)})",
            edgecolors="none",
        )
    ax.axvline(100 * TAU, color="black", linestyle="--", linewidth=1)
    ax.axhline(100 * TAU, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Oracle − Frozen AUPRC (pp)")
    ax.set_ylabel("Practical − Frozen AUPRC (pp)")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_2_1.pdf", bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)


def objective_figure() -> None:
    frame = pd.read_csv(OBJECTIVE)
    assert len(frame) == 180
    datasets = sorted(frame["dataset"].unique())
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    for index, dataset in enumerate(datasets):
        part = frame[frame["dataset"] == dataset]
        ax.scatter(
            part["oracle_normal_loss_reduction"],
            100 * part["oracle_delta_auprc"],
            marker="o",
            s=24,
            alpha=0.75,
            color=cmap(index % 10),
            label="SWAT" if dataset == "SWAT2015" else dataset,
            edgecolors="none",
        )
    ax.axhline(100 * TAU, color="black", linestyle="--", linewidth=1)
    ax.axvline(0, color="black", linestyle=":", linewidth=1)
    ax.set_xlabel("Oracle pre-post reconstruction-loss decrease")
    ax.set_ylabel("Oracle − Frozen AUPRC (pp)")
    ax.legend(frameon=False, fontsize=7, ncol=3)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_3_1.pdf", bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=Path, required=True)
    parser.add_argument("--objective", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    CELLS, OBJECTIVE, OUT = args.cells, args.objective, args.output
    OUT.mkdir(parents=True, exist_ok=False)
    outcome_figure()
    objective_figure()
