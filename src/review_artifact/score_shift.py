"""Submitted score-distribution and exact pair-order diagnostics."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from section5_repro.data import Entity
from section5_repro.paths import Workspace, frozen_result_dir, tta_result_dir
from section5_repro.protocol import load_protocol
from .cli import entity_records

ALPHAS = (0.10, 0.15, 0.20)
BETAS = (0.20, 0.30, 0.40)
PRIMARY = (0.15, 0.30)

class Fenwick:
    def __init__(self, size: int) -> None:
        self.tree = np.zeros(size + 1, dtype=np.int64)

    def add(self, index: int) -> None:
        index += 1
        while index < len(self.tree):
            self.tree[index] += 1
            index += index & -index

    def prefix(self, end: int) -> int:
        total = 0
        while end > 0:
            total += int(self.tree[end])
            end -= end & -end
        return total


def count_both_correct(
    normal_frozen: np.ndarray,
    normal_oracle: np.ndarray,
    anomaly_frozen: np.ndarray,
    anomaly_oracle: np.ndarray,
) -> int:
    """Count pairs with anomaly > normal under both score vectors, exactly."""
    coordinates = np.unique(anomaly_frozen)
    anomaly_order = np.argsort(-anomaly_oracle, kind="stable")
    normal_order = np.argsort(-normal_oracle, kind="stable")
    fenwick = Fenwick(len(coordinates))
    inserted = 0
    cursor = 0
    both = 0
    for normal_index in normal_order:
        threshold_oracle = normal_oracle[normal_index]
        while (
            cursor < len(anomaly_order)
            and anomaly_oracle[anomaly_order[cursor]] > threshold_oracle
        ):
            anomaly_index = anomaly_order[cursor]
            position = int(np.searchsorted(coordinates, anomaly_frozen[anomaly_index]))
            fenwick.add(position)
            inserted += 1
            cursor += 1
        not_greater = fenwick.prefix(
            int(np.searchsorted(coordinates, normal_frozen[normal_index], side="right"))
        )
        both += inserted - not_greater
    return both


def pair_transitions(
    normal_frozen: np.ndarray,
    normal_oracle: np.ndarray,
    anomaly_frozen: np.ndarray,
    anomaly_oracle: np.ndarray,
) -> dict[str, int | float]:
    n_pairs = int(len(normal_frozen) * len(anomaly_frozen))
    frozen_correct = int(
        (len(anomaly_frozen) - np.searchsorted(
            np.sort(anomaly_frozen), normal_frozen, side="right"
        )).sum()
    )
    oracle_correct = int(
        (len(anomaly_oracle) - np.searchsorted(
            np.sort(anomaly_oracle), normal_oracle, side="right"
        )).sum()
    )
    unchanged_correct = count_both_correct(
        normal_frozen, normal_oracle, anomaly_frozen, anomaly_oracle
    )
    correct_to_wrong = frozen_correct - unchanged_correct
    wrong_to_correct = oracle_correct - unchanged_correct
    unchanged_wrong = n_pairs - frozen_correct - oracle_correct + unchanged_correct
    if min(correct_to_wrong, wrong_to_correct, unchanged_correct, unchanged_wrong) < 0:
        raise AssertionError("Invalid exact pair-transition count")
    return {
        "wrong_to_correct": wrong_to_correct,
        "correct_to_wrong": correct_to_wrong,
        "unchanged_correct": unchanged_correct,
        "unchanged_wrong": unchanged_wrong,
        "n_pairs": n_pairs,
        "net_pair_gain": (wrong_to_correct - correct_to_wrong) / n_pairs,
    }


def pair_features(frozen_score, oracle_score, labels):
    """Frozen-defined tails; ties are included at quantile cutoffs."""
    frozen_score = np.asarray(frozen_score, dtype=np.float64).reshape(-1)
    oracle_score = np.asarray(oracle_score, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    if len(frozen_score) != len(oracle_score) or len(labels) != len(frozen_score):
        raise ValueError("Frozen/Oracle/label length mismatch")
    if not np.isfinite(labels).all() or set(np.unique(labels)) != {0, 1}:
        raise ValueError("Score-shift analysis requires both binary label classes")
    valid = bool(np.isfinite(frozen_score).all() and np.isfinite(oracle_score).all())
    invalid_reason = "" if valid else (
        f"nonfinite-score:frozen={int((~np.isfinite(frozen_score)).sum())};"
        f"oracle={int((~np.isfinite(oracle_score)).sum())}"
    )
    rows = []
    for alpha in ALPHAS:
        for beta in BETAS:
            if not valid:
                values = {key: np.nan for key in FEATURE_COLUMNS}
            else:
                normal = labels == 0
                anomaly = labels != 0
                hard_threshold = float(np.quantile(frozen_score[normal], 1.0 - alpha))
                boundary_threshold = float(np.quantile(frozen_score[anomaly], beta))
                hard = normal & (frozen_score >= hard_threshold)
                boundary = anomaly & (frozen_score <= boundary_threshold)
                drop = frozen_score - oracle_score
                hard_drop = float(np.median(drop[hard]))
                boundary_drop = float(np.median(drop[boundary]))
                values = {
                    "hard_normal_drop": hard_drop, "boundary_anomaly_drop": boundary_drop,
                    "selective_compression": hard_drop - boundary_drop,
                    **pair_transitions(frozen_score[hard], oracle_score[hard], frozen_score[boundary], oracle_score[boundary]),
                    "hard_normal_points": int(hard.sum()), "boundary_anomaly_points": int(boundary.sum()),
                }
            rows.append({"alpha": alpha, "beta": beta, **values, "valid": valid,
                         "invalid_reason": invalid_reason,
                         "pair_evaluation": "exact_cartesian" if valid else "not-evaluated"})
    return rows


FEATURE_COLUMNS = (
    "hard_normal_drop", "boundary_anomaly_drop", "selective_compression",
    "wrong_to_correct", "correct_to_wrong", "unchanged_correct", "unchanged_wrong",
    "n_pairs", "net_pair_gain", "hard_normal_points", "boundary_anomaly_points",
)


def read_core_arrays(directory, entry, seed):
    protocol = load_protocol()
    meta = json.loads((directory / "complete.json").read_text())
    if (meta["entity"] != entry or meta["seed"] != seed or meta["setting"] != directory.name
            or meta["protocol"] != json.loads(json.dumps(protocol.as_dict()))):
        raise ValueError(f"Core identity/protocol mismatch: {directory}")
    score = np.load(directory / "test_energy.npy", allow_pickle=False).reshape(-1)
    label = np.load(directory / "true.npy", allow_pickle=False).reshape(-1)
    expected = entry["test_rows"] // protocol.sequence_length * protocol.sequence_length
    if score.dtype != np.float32 or label.dtype != np.int8:
        raise ValueError(f"Unexpected score/label dtype: {directory}")
    if len(score) != expected or len(label) != expected:
        raise ValueError(f"Unexpected point support: {directory}")
    return score, label


def physical_features(workspace_root):
    protocol = load_protocol()
    workspace = Workspace(Path(workspace_root).resolve())
    rows = []
    for entry in entity_records():
        entity = Entity(entry["dataset"], entry["subset"], Path("unused"))
        for seed in protocol.seeds:
            frozen, labels = read_core_arrays(frozen_result_dir(workspace, entity, seed, protocol), entry, seed)
            for config in protocol.tta_configs:
                oracle, oracle_labels = read_core_arrays(tta_result_dir(workspace, entity, seed, protocol, config, True), entry, seed)
                if not np.array_equal(labels, oracle_labels):
                    raise ValueError(f"Unmatched Frozen/Oracle labels: {entity.display_name}/{seed}")
                identity = {"dataset": entity.dataset, "subset": entity.subset, "seed": seed, "optimizer": config.key}
                rows.extend({**identity, **row} for row in pair_features(frozen, oracle, labels))
    return pd.DataFrame(rows)


def aggregate_features(physical):
    keys = ["dataset", "optimizer", "seed", "alpha", "beta"]
    expected_entities = pd.DataFrame(entity_records()).groupby("dataset").size().to_dict()
    if len(physical) != 2320 * 9 or physical.duplicated([*keys, "subset"]).any():
        raise ValueError("Expected 20,880 unique physical score-shift rows")
    rows = []
    for key, group in physical.groupby(keys, sort=True):
        identity = dict(zip(keys, key))
        expected = int(expected_entities[identity["dataset"]])
        observed, n_valid = group["subset"].nunique(dropna=False), int(group["valid"].sum())
        if observed != expected or not n_valid:
            raise ValueError(f"Incomplete score-shift support: {identity}")
        row = {**identity}
        for column in FEATURE_COLUMNS:
            row[column] = float(np.median(group.loc[group["valid"], column].to_numpy(dtype=np.float64)))
        reasons = sorted(set(group.loc[~group["valid"], "invalid_reason"].astype(str)))
        row.update({"n_entities_aggregated": n_valid, "n_entities_observed": observed,
                    "n_entities_expected": expected, "n_entities_valid": n_valid, "valid": True,
                    "validity_status": "complete" if n_valid == expected else "complete-with-undefined",
                    "invalid_reason": ";".join(reasons), "aggregation_level": "dataset-median-over-subsets",
                    "pair_evaluation": "exact_cartesian-per-entity"})
        rows.append(row)
    return pd.DataFrame(rows)


def attach_detection(features, cells):
    keys = ["dataset", "optimizer", "seed"]
    protocol = load_protocol()
    expected = {(d, c.key, s) for d in protocol.datasets for c in protocol.tta_configs for s in protocol.seeds}
    expected_features = {(*key, a, b) for key in expected for a in ALPHAS for b in BETAS}
    if len(cells) != 180 or set(cells[keys].itertuples(index=False, name=None)) != expected:
        raise ValueError("Detection input must contain the 180 submitted protocol cells")
    if len(features) != 1620 or set(features[[*keys, "alpha", "beta"]].itertuples(index=False, name=None)) != expected_features:
        raise ValueError("Feature input must cover all nine tail settings and 180 cells")
    if not features.valid.all():
        raise ValueError("Logical score-shift cells contain undefined diagnostics")
    features = features.drop(columns=[c for c in ("oracle_delta_auprc", "oracle_clear_gain", "category") if c in features])
    delta = cells[[*keys, "corrected_auprc_oracle_delta", "corrected_auprc_category"]]
    result = features.merge(delta, on=keys, validate="many_to_one").rename(columns={
        "corrected_auprc_oracle_delta": "oracle_delta_auprc", "corrected_auprc_category": "category"})
    result["oracle_clear_gain"] = result.category.isin(["BI", "OOG"])
    if not np.isfinite(result[["oracle_delta_auprc", "selective_compression", "net_pair_gain"]].to_numpy()).all():
        raise ValueError("Undefined values cannot enter score-shift correlations")
    return result


def associations(result):
    rows = []
    def add(alpha, beta, analysis, feature, group, x, y):
        rows.append({"alpha": alpha, "beta": beta, "analysis": analysis, "feature": feature,
                     "group": group, "n": len(x), "rho": float(spearmanr(x, y).statistic)})
    for alpha in ALPHAS:
        for beta in BETAS:
            x = result.loc[(result.alpha == alpha) & (result.beta == beta)].copy()
            for feature in ("selective_compression", "net_pair_gain"):
                add(alpha, beta, "pooled", feature, "all", x[feature], x.oracle_delta_auprc)
            if (alpha, beta) != PRIMARY:
                continue
            med = x.groupby("dataset", sort=True)[["selective_compression", "net_pair_gain", "oracle_delta_auprc"]].median()
            for feature in ("selective_compression", "net_pair_gain"):
                add(alpha, beta, "dataset-median", feature, "all", med[feature], med.oracle_delta_auprc)
            for feature in ("selective_compression", "net_pair_gain"):
                a, b = [], []
                for _, group in x.groupby("dataset"):
                    a.extend(rankdata(group[feature], method="average"))
                    b.extend(rankdata(group.oracle_delta_auprc, method="average"))
                add(alpha, beta, "pooled-within-dataset-rank", feature, "all", a, b)
            for dataset in sorted(x.dataset.unique()):
                keep = x.loc[x.dataset != dataset]
                for feature in ("selective_compression", "net_pair_gain"):
                    add(alpha, beta, "leave-one-dataset-out", feature, dataset, keep[feature], keep.oracle_delta_auprc)
            complete = x.loc[x.n_entities_valid == x.n_entities_expected]
            for feature in ("selective_compression", "net_pair_gain"):
                add(alpha, beta, "complete-only", feature, "all", complete[feature], complete.oracle_delta_auprc)
    return pd.DataFrame(rows)

