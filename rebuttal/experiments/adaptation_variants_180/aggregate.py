#!/usr/bin/env python3
"""Aggregate the extension without dropping invalid entities or partial runs."""
import json
from pathlib import Path

import run as runner
import numpy as np
import pandas as pd

COMPARISONS = {
    "M2N2": ("m2n2_frozen", "m2n2_practical_p95", "m2n2_oracle_point"),
    "FPM-only": ("fpm_frozen", "fpm_style_practical", "fpm_oracle"),
    "FPM+SANA normal-window Oracle": ("sana_frozen", "practical_fpm_sana", "oracle_normal_window_sana"),
    "FPM+SANA clean-FPM Oracle": ("sana_frozen", "practical_fpm_sana", "oracle_clean_fpm_sana"),
}


def collect(plan, optimizers):
    physical, pairs = {}, []
    for base_id, base in plan["bases"].items():
        for optimizer in optimizers:
            for analysis, arms in COMPARISONS.items():
                records = []
                for arm in arms:
                    key = f"fpm_oracle:{optimizer}" if arm == "fpm_oracle" else arm
                    reused = arm.endswith("_frozen") or arm == "fpm_oracle" or optimizer == runner.EXISTING_OPT
                    if reused:
                        ref = base["references"][key]
                        record = {**ref, "dataset": base["dataset"], "subset": base["subset"],
                                  "seed": base["seed"], "optimizer": "shared" if arm.endswith("_frozen") else optimizer,
                                  "arm": arm, "status": "complete" if ref["valid"] else "existing_invalid",
                                  "reused": True}
                        identity = ref["score_path"]
                    else:
                        directory = runner.output_dir(base, optimizer, arm)
                        success, failed = directory / "extension_metrics.json", directory / "failure.json"
                        if success.is_file() and failed.is_file():
                            raise ValueError(f"Conflicting terminal records: {directory}")
                        if success.is_file():
                            record = json.loads(success.read_text())
                            if record["status"] != "complete" or not record["valid"]:
                                raise ValueError(f"Non-complete success record: {directory}")
                        elif failed.is_file():
                            record = json.loads(failed.read_text())
                            record.update(auprc=None, output_path=str(directory))
                        else:
                            raise FileNotFoundError(f"No terminal result for {directory}")
                        record["reused"] = False
                        identity = str(directory / "test_energy.npy")
                    physical[identity] = record
                    records.append(record)
                valid = all(r["valid"] for r in records)
                f, p, o = [r["auprc"] if r["valid"] else np.nan for r in records]
                pairs.append({"analysis": analysis, "dataset": base["dataset"], "subset": base["subset"],
                              "seed": base["seed"], "optimizer": optimizer, "base_id": base_id,
                              "valid": valid, "frozen": f, "practical": p, "oracle": o,
                              "practical_delta": p - f if valid else np.nan,
                              "oracle_delta": o - f if valid else np.nan,
                              "invalid_arms": ",".join(arm for arm, r in zip(arms, records) if not r["valid"])})
    return pd.DataFrame(physical.values()), pd.DataFrame(pairs)


def build_cells(pairs):
    rows = []
    for key, group in pairs.groupby(["analysis", "dataset", "optimizer", "seed"], sort=True):
        expected = runner.EXPECTED[key[1]]
        if len(group) != expected or group.subset.nunique() != expected:
            raise ValueError(f"Nominal entity support changed: {key}")
        valid = bool(group.valid.all())
        p = float(group.practical_delta.median()) if valid else np.nan
        o = float(group.oracle_delta.median()) if valid else np.nan
        category = ("undefined" if not valid else "BI" if p > .005 and o > .005 else
                    "POG" if p > .005 else "OOG" if o > .005 else "NI")
        rows.append(dict(zip(["analysis", "dataset", "optimizer", "seed"], key)) |
                    {"entities_expected": expected, "entities_valid": int(group.valid.sum()),
                     "valid": valid, "practical_delta": p, "oracle_delta": o, "category": category,
                     "invalid_entities": ",".join(group.loc[~group.valid, "subset"].replace("", "__root__"))})
    return pd.DataFrame(rows)


def validate_existing(plan):
    _, pairs = collect(plan, [runner.EXISTING_OPT])
    cells = build_cells(pairs)
    old = pd.read_csv(runner.ROOT / "workspace/rebuttal_inputs/p3_iclr27_corrected_auprc/robustness_outcomes.csv")
    check = cells.merge(old, on=["analysis", "dataset", "seed"], validate="one_to_one")
    assert len(check) == 180 and check.valid.all()
    np.testing.assert_allclose(check.practical_delta, check.corrected_practical_delta, rtol=0, atol=1e-12)
    np.testing.assert_allclose(check.oracle_delta, check.corrected_oracle_delta, rtol=0, atol=1e-12)
    assert (check.category == check.corrected_category).all()
    # Ensure a single invalid entity does not become an available-subset median.
    partial = pairs[(pairs.analysis == "M2N2") & (pairs.dataset == "MSL") & (pairs.seed == 42)].copy()
    partial.loc[partial.index[0], "valid"] = False
    cell = build_cells(partial).iloc[0]
    assert cell.category == "undefined" and np.isnan(cell.practical_delta) and np.isnan(cell.oracle_delta)
    return {"old_45_cell_comparisons_reproduced": 4, "existing_cells_matched": len(check),
            "incomplete_nominal_cell_remains_undefined": True}


def main():
    plan = json.loads((runner.HERE / "plan.json").read_text())
    regression = validate_existing(plan)
    physical, pairs = collect(plan, [c.key for c in runner.load_protocol().tta_configs])
    cells = build_cells(pairs)
    assert len(cells) == 720 and len(pairs) == 9280
    assert int((~physical.reused).sum()) == 10440
    out = runner.HERE / "derived"
    out.mkdir(exist_ok=False)
    counts = pd.crosstab(cells.analysis, cells.category).reindex(
        columns=["BI", "OOG", "NI", "POG", "undefined"], fill_value=0)
    counts["nominal_cells"] = 180
    for name, frame in (("physical", physical), ("paired_entities", pairs), ("cells", cells)):
        frame.to_csv(out / f"{name}.csv", index=False)
    counts.to_csv(out / "outcome_counts.csv")
    runner.atomic_json(out / "summary.json", {"finished_at": runner.now(), "execution_finished": True,
        "valid_grid_complete": bool(cells.valid.all()), "support_policy": plan["support_policy"],
        "new_physical_terminal_records": int((~physical.reused).sum()),
        "new_physical_valid": int((~physical.reused & physical.valid).sum()),
        "existing_invalid_fpm_oracle_controls": plan["existing_invalid_fpm_oracle_controls"],
        "regression": regression, "counts": counts.reset_index().to_dict(orient="records")})
    print(counts.to_string(), flush=True)


if __name__ == "__main__":
    main()
