"""Figure C.1: selected numerical and drawing functions from the submitted renderer."""
from __future__ import annotations
from io import BytesIO
from pathlib import Path
from typing import List, Sequence
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
PERCENTILE_CURVE_PALETTE = {"Normal": "#1f77b4", "Anomaly": "#d62728"}
HARD_NORMAL_LABELS = ["90-100%", "80-90%", "70-80%"]
BOUNDARY_ANOMALY_LABELS = ["30-40%", "20-30%", "10-20%", "0-10%"]

def _empty_bucket_rows(num_bins: int) -> List[dict]:
    return [
        {
            "bin_id": b,
            "percentile_bin": f"{b * 100 // num_bins}-{(b + 1) * 100 // num_bins}%",
        }
        for b in range(num_bins)
    ]


def _bucket_stats_single_seed(
    frozen_scores: np.ndarray,
    delta_scores: np.ndarray,
    num_bins: int,
) -> pd.DataFrame:
    assert num_bins > 0
    if frozen_scores.size != delta_scores.size:
        raise ValueError("frozen_scores and delta_scores must have the same length.")

    out_rows = []
    if frozen_scores.size == 0:
        for row in _empty_bucket_rows(num_bins):
            row.update(
                {
                    "count": 0,
                    "median_delta": np.nan,
                    "frozen_q_lo": np.nan,
                    "frozen_q_hi": np.nan,
                }
            )
            out_rows.append(row)
        return pd.DataFrame(out_rows)

    pct = pd.Series(frozen_scores).rank(method="average", pct=True).to_numpy(dtype=np.float64)
    bin_id = np.clip((pct * num_bins).astype(int), 0, num_bins - 1)
    q_grid = np.linspace(0.0, 1.0, num_bins + 1)
    frozen_q = np.quantile(frozen_scores, q_grid)

    for b in range(num_bins):
        m = bin_id == b
        curr = delta_scores[m]
        cnt = int(curr.size)
        if cnt > 0:
            row = {
                "bin_id": b,
                "percentile_bin": f"{b * 100 // num_bins}-{(b + 1) * 100 // num_bins}%",
                "count": cnt,
                "median_delta": float(np.median(curr)),
                "frozen_q_lo": float(frozen_q[b]),
                "frozen_q_hi": float(frozen_q[b + 1]),
            }
        else:
            row = {
                "bin_id": b,
                "percentile_bin": f"{b * 100 // num_bins}-{(b + 1) * 100 // num_bins}%",
                "count": 0,
                "median_delta": np.nan,
                "frozen_q_lo": float(frozen_q[b]),
                "frozen_q_hi": float(frozen_q[b + 1]),
            }
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def _compute_seed_and_aggregate_bucket_stats(class_df: pd.DataFrame, num_bins: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "seed" not in class_df.columns:
        raise ValueError("class_df must contain a 'seed' column.")

    seed_stats = []
    valid_seed_df = class_df.loc[class_df["seed"].notna()].copy()
    if valid_seed_df.empty:
        out = pd.DataFrame(_empty_bucket_rows(num_bins))
        out["seed_count"] = 0
        out["sample_count_median"] = np.nan
        out["sample_count_q25"] = np.nan
        out["sample_count_q75"] = np.nan
        out["median_delta_median"] = np.nan
        out["median_delta_q25"] = np.nan
        out["median_delta_q75"] = np.nan
        out["frozen_q_lo_median"] = np.nan
        out["frozen_q_hi_median"] = np.nan
        empty_seed_stats = pd.DataFrame(
            columns=["bin_id", "percentile_bin", "count", "median_delta", "frozen_q_lo", "frozen_q_hi", "seed"]
        )
        return empty_seed_stats, out

    valid_seed_df["seed"] = valid_seed_df["seed"].astype(int)
    for seed, seed_df in valid_seed_df.groupby("seed", sort=True):
        stats = _bucket_stats_single_seed(
            frozen_scores=seed_df["score_frozen"].to_numpy(dtype=float),
            delta_scores=seed_df["delta_score"].to_numpy(dtype=float),
            num_bins=num_bins,
        ).copy()
        stats["seed"] = int(seed)
        seed_stats.append(stats)

    seed_stats_df = pd.concat(seed_stats, axis=0, ignore_index=True)
    agg_stats_df = (
        seed_stats_df.groupby(["bin_id", "percentile_bin"], as_index=False)
        .agg(
            seed_count=("median_delta", "count"),
            sample_count_median=("count", "median"),
            sample_count_q25=("count", lambda x: float(np.nanquantile(x.to_numpy(dtype=float), 0.25))),
            sample_count_q75=("count", lambda x: float(np.nanquantile(x.to_numpy(dtype=float), 0.75))),
            median_delta_median=("median_delta", "median"),
            median_delta_q25=("median_delta", lambda x: float(np.nanquantile(x.to_numpy(dtype=float), 0.25))),
            median_delta_q75=("median_delta", lambda x: float(np.nanquantile(x.to_numpy(dtype=float), 0.75))),
            frozen_q_lo_median=("frozen_q_lo", "median"),
            frozen_q_hi_median=("frozen_q_hi", "median"),
        )
        .sort_values("bin_id")
        .reset_index(drop=True)
    )
    return seed_stats_df, agg_stats_df


def _prepare_plot_arrays(stats_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    ordered = stats_df.sort_values("bin_id", ascending=False).reset_index(drop=True)
    x = np.arange(len(ordered), dtype=int)
    y = ordered["median_delta_median"].to_numpy(dtype=float)
    y_q25 = ordered["median_delta_q25"].to_numpy(dtype=float)
    y_q75 = ordered["median_delta_q75"].to_numpy(dtype=float)
    labels = ordered["percentile_bin"].astype(str).tolist()
    return x, y, y_q25, y_q75, labels


def _draw_percentile_bar_panel(
    ax: plt.Axes,
    normal_stats_df: pd.DataFrame,
    anomaly_stats_df: pd.DataFrame,
    y_limits: tuple[float, float] | None = None,
    show_ylabel: bool = True,
    show_legend: bool = True,
) -> None:
    normal_x, normal_y, _, _, labels = _prepare_plot_arrays(normal_stats_df)
    _, anomaly_y, _, _, anomaly_labels = _prepare_plot_arrays(anomaly_stats_df)
    if anomaly_labels != labels:
        raise ValueError("Normal and anomaly percentile bins must be aligned.")

    label_to_x = {label: int(idx) for idx, label in enumerate(labels)}
    width = 0.38
    ax.bar(
        normal_x - width / 2,
        normal_y,
        width=width,
        color=PERCENTILE_CURVE_PALETTE["Normal"],
        label="Normal median",
        zorder=3,
    )
    ax.bar(
        normal_x + width / 2,
        anomaly_y,
        width=width,
        color=PERCENTILE_CURVE_PALETTE["Anomaly"],
        label="Anomaly median",
        zorder=3,
    )

    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.1, alpha=0.7)
    ax.set_xticks(normal_x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_xlabel("Within-class frozen-score percentile (high to low)")
    if show_ylabel:
        ax.set_ylabel(r"Oracle score shift ($s_{\mathrm{oracle}}-s_{\mathrm{frozen}}$)")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    if y_limits is not None:
        ax.set_ylim(*y_limits)

    hard_normal_x = [label_to_x[label] for label in HARD_NORMAL_LABELS if label in label_to_x]
    boundary_anomaly_x = [label_to_x[label] for label in BOUNDARY_ANOMALY_LABELS if label in label_to_x]
    if hard_normal_x:
        ax.axvspan(min(hard_normal_x) - 0.5, max(hard_normal_x) + 0.5, color="#d9d9d9", alpha=0.14, zorder=0)
        ax.text(
            float(np.mean(hard_normal_x)),
            0.97,
            "Hardest normals",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=9,
            color="#555555",
        )
    if boundary_anomaly_x:
        ax.axvspan(min(boundary_anomaly_x) - 0.5, max(boundary_anomaly_x) + 0.5, color="#d9d9d9", alpha=0.14, zorder=0)
        ax.text(
            float(np.mean(boundary_anomaly_x)),
            0.97,
            "Boundary anomalies",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=9,
            color="#555555",
        )

    if show_legend:
        ax.legend(
            loc="lower left",
            frameon=True,
            facecolor="white",
            edgecolor="#cccccc",
            framealpha=0.9,
        )


def save_side_by_side_percentile_curves(
    panel_specs: Sequence[dict[str, object]],
    output_path: Path,
    y_limits: tuple[float, float] | None = None,
) -> None:
    if not panel_specs:
        raise ValueError("panel_specs must not be empty.")
    fig, axes = plt.subplots(1, len(panel_specs), figsize=(15.6, 5.2), sharey=True, constrained_layout=True)
    if len(panel_specs) == 1:
        axes = [axes]
    else:
        axes = list(axes)

    legend_handles = None
    legend_labels = None
    for idx, (ax, spec) in enumerate(zip(axes, panel_specs)):
        _draw_percentile_bar_panel(
            ax=ax,
            normal_stats_df=spec["normal_stats_df"],
            anomaly_stats_df=spec["anomaly_stats_df"],
            y_limits=y_limits,
            show_ylabel=(idx == 0),
            show_legend=bool(spec.get("show_legend", False)),
        )
        panel_title = str(spec.get("panel_title", "")).strip()
        if panel_title:
            ax.set_title(panel_title, fontsize=11, pad=12)
        if idx > 0:
            ax.tick_params(axis="y", labelleft=False)

    # Preserve the existing PDF layout before restoring shared-axis labels.
    fig.savefig(BytesIO(), format="pdf")
    fig.set_layout_engine("none")
    for ax in axes[1:]:
        # Use the empty inner margin so labels do not cross the left panel's spine.
        ax.tick_params(axis="y", labelleft=True, pad=-8.5)
        for label in ax.get_yticklabels():
            label.set_horizontalalignment("left")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    if output_path.suffix.lower() == ".png":
        fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)
