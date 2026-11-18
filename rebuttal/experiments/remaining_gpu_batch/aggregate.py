"""Complete-support aggregation; reused trajectories retain explicit provenance."""
from pathlib import Path
import argparse
import json
import experiment as runner
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CATEGORIES=("BI","OOG","NI","POG")


def category(p,o):
    if not np.isfinite([p,o]).all():
        raise ValueError("Nonfinite cell effects")
    return "BI" if p>.005 and o>.005 else "POG" if p>.005 else "OOG" if o>.005 else "NI"


def read_result(directory,base,optimizer,arm,width,labels,checkpoint_hash,mask=None):
    meta=json.loads((directory/"result.json").read_text())
    expected={"complete":True,"dataset":base["dataset"],"subset":base["subset"],"seed":base["seed"],
        "optimizer":optimizer,"arm":arm,"window_length":width,"metric_version":runner.controls.VERSION,
        "source_checkpoint_sha256":checkpoint_hash,"data_manifest":base["data_manifest"],
        "train_stride":width,"test_stride":width,"test_points":len(labels),"test_windows":len(labels)//width}
    if any(meta.get(k)!=v for k,v in expected.items()):
        raise ValueError(f"Result identity differs: {directory}")
    truth=np.load(directory/"true.npy",allow_pickle=False)
    scores=np.load(directory/"test_energy.npy",allow_pickle=False)
    selected=np.load(directory/"update_mask.npy",allow_pickle=False)
    if not np.array_equal(truth,labels) or truth.shape!=scores.shape or selected.shape!=truth.shape:
        raise ValueError("Result support differs")
    value=runner.controls.metric(truth,scores)
    if abs(value-meta["auprc"])>1e-12:
        raise ValueError("Saved PR-AUC does not reproduce")
    target=mask.reshape(-1) if mask is not None else truth==0 if arm=="oracle" else scores<meta["selector_tau"] if arm=="practical" else np.zeros(len(labels),bool)
    if not np.array_equal(selected,target) or selected.dtype!=bool:
        raise ValueError("Saved update mask differs")
    if meta["selected_points"]!=int(selected.sum()) or meta["selected_normal_points"]!=int((selected&(truth==0)).sum()):
        raise ValueError("Saved selection counts differ")
    if arm!="frozen":
        trace=pd.read_parquet(directory/"window_trace.parquet")
        if (len(trace)!=len(labels)//width or not trace.optimizer_step.all()
                or meta["optimizer_steps"]!=len(trace)
                or not np.array_equal(trace.selected_points.to_numpy(),selected.reshape(-1,width).sum(1))
                or not (trace.loc[trace.selected_points==0,"masked_loss"]==0).all()
                or not np.isfinite(trace[["masked_loss","parameter_step_l2"]].to_numpy()).all()
                or abs(trace.parameter_step_l2.sum()-meta["parameter_step_l2_sum"])>1e-8):
            raise ValueError("Window update trace differs")
    elif meta["optimizer_steps"]!=0:
        raise ValueError("Frozen stepped")
    return meta


def assert_complete_files(expected,root):
    actual={p.parent for p in (root/"runs").rglob("result.json")}
    if actual!=set(expected):
        raise ValueError(f"Incomplete/unexpected result files under {root}: {len(actual)} vs {len(expected)}")


def count_frames(plan):
    expected=[]
    for t in plan["tasks"]:
        if t["experiment"]=="count":
            b=plan["bases"][t["base_id"]]
            expected.extend(runner.count_directory(b,t["optimizer"],r) for r in t["executions"])
    if len(expected)!=plan["expected"]["count_unique_new_runs"]:
        raise ValueError("Expected unique run count differs")
    assert_complete_files(expected,runner.COUNT)
    physical,paired,executed=[],[],[]
    for task in (t for t in plan["tasks"] if t["experiment"]=="count"):
        base=plan["bases"][task["base_id"]]
        labels,old,_=runner.original_inputs(base,task["optimizer"],verify=True)
        masks=runner.matched_masks(old,task["task_id"])
        slots,executions=runner.deduplicate(old,masks)
        if slots!=task["slots"] or executions!=task["executions"]:
            raise ValueError("Declared exact-reuse plan differs")
        new={}
        for ref in executions:
            directory=runner.count_directory(base,task["optimizer"],ref)
            meta=read_result(directory,base,task["optimizer"],ref["arm"],12,labels,
                base["source_checkpoint_sha256"],masks[ref["sample_seed"],ref["arm"]])
            new[ref["sample_seed"],ref["arm"]]=meta
            executed.append({k:v for k,v in meta.items() if k!="data_manifest"})
        original_values={a:runner.controls.reference(base,a,task["optimizer"])["auprc"] for a in ("frozen",*runner.ARMS)}
        values={}
        for slot in slots:
            key=slot["sample_seed"],slot["arm"]
            ref=slot["reference"]
            mask=masks[key].reshape(-1)
            if ref["kind"]=="original":
                original=runner.controls.reference(base,ref["arm"],task["optimizer"])
                value=original["auprc"]
                source="original:"+original["physical_run_id"]
                displacement=None
            else:
                meta=new[ref["sample_seed"],ref["arm"]]
                value=meta["auprc"]
                source="new:"+str(Path(meta["output_path"]).relative_to(runner.ROOT))
                displacement=meta["parameter_step_l2_sum"]
            values[key]=value
            count=int(mask.sum()); normal=int((mask&(labels==0)).sum())
            physical.append({"dataset":base["dataset"],"subset":base["subset"],"seed":base["seed"],
                "optimizer":task["optimizer"],"sample_seed":key[0],"arm":key[1],"auprc":value,
                "frozen_auprc":original_values["frozen"],"delta":value-original_values["frozen"],
                "reference_kind":ref["kind"],"physical_source":source,"selected_points":count,
                "original_selected_points":int(old[key[1]].sum()),"selected_normal_points":normal,
                "selected_anomalous_points":count-normal,"normal_coverage":normal/int((labels==0).sum()),
                "selector_purity":normal/count if count else None,"optimizer_steps":len(labels)//12,
                "zero_selected_windows":int((mask.reshape(-1,12).sum(1)==0).sum()),
                "parameter_step_l2_sum":displacement,"parameter_trace_available":displacement is not None})
        for sample in runner.SAMPLE_SEEDS:
            paired.append({"dataset":base["dataset"],"subset":base["subset"],"seed":base["seed"],
                "optimizer":task["optimizer"],"sample_seed":sample,
                "practical_delta":values[sample,"practical"]-original_values["frozen"],
                "oracle_delta":values[sample,"oracle"]-original_values["frozen"],
                "original_practical_delta":original_values["practical"]-original_values["frozen"],
                "original_oracle_delta":original_values["oracle"]-original_values["frozen"]})
    physical,paired,executed=pd.DataFrame(physical),pd.DataFrame(paired),pd.DataFrame(executed)
    if len(physical)!=12720 or len(paired)!=6360:
        raise ValueError("Incomplete logical count support")
    grouped=paired.groupby(["sample_seed","dataset","optimizer","seed"],sort=True)
    effects=["practical_delta","oracle_delta","original_practical_delta","original_oracle_delta"]
    cells=grouped[effects].median().reset_index()
    cells["entities"]=grouped.size().values
    cells["category"]=[category(p,o) for p,o in zip(cells.practical_delta,cells.oracle_delta)]
    cells["original_category"]=[category(p,o) for p,o in zip(cells.original_practical_delta,cells.original_oracle_delta)]
    original=pd.read_csv(runner.ROOT/"workspace/rebuttal_inputs/p3_iclr27_corrected_auprc_common_support_cells.csv")
    for sample,frame in cells.groupby("sample_seed"):
        check=frame.merge(original,on=["dataset","optimizer","seed"],validate="one_to_one")
        if len(check)!=180 or not (check.original_category==check.corrected_auprc_category).all():
            raise ValueError("Canonical 180-cell regression failed")
        for arm in runner.ARMS:
            np.testing.assert_allclose(check[f"original_{arm}_delta"],check[f"corrected_auprc_{arm}_delta"],rtol=0,atol=1e-14)
    return physical,paired,cells,executed


def window_frames(plan):
    datasets=plan.get("window_datasets",runner.WINDOW_DATASETS)
    bases=[b for b in plan["bases"].values() if b["dataset"] in datasets]
    expected=[]
    for task in plan["tasks"]:
        if task["experiment"]=="window":
            b=plan["bases"][task["base_id"]]
            expected.extend(runner.window_directory(b,task["length"],o,a) for o,a in runner.window_runs(runner.window_protocol(task["length"])))
    successful={p.parent for p in (runner.WINDOW/"runs").rglob("result.json")}
    failed={p.parent for p in (runner.WINDOW/"runs").rglob("failure.json")}
    if successful & failed or successful | failed != set(expected) or len(expected)!=len(bases)*18:
        raise ValueError(f"Incomplete/unexpected window runs: {len(successful)} successful, {len(failed)} invalid, {len(expected)} expected")
    physical,source_rows=[],[]
    for base in bases:
        runner.check_base(base)
        n=base["data_manifest"]["test_rows"]//48*48
        labels=np.load(runner.controls.reference(base,"frozen","shared")["label_path"],allow_pickle=False).reshape(-1)[:n]
        for length in (12,*runner.LENGTHS):
            path=Path(base["checkpoint"]) if length==12 else runner.source_path(base,length)
            source=json.loads((path.parent/"complete.json").read_text())
            entity=runner.controls.Entity(base["dataset"],base["subset"],Path(base["directory"]))
            if not runner.controls._checkpoint_complete(path,runner.window_protocol(length),entity,base["seed"]):
                raise ValueError("Window source completion differs")
            if length!=12 and source["test_loss_logging_enabled"] is not False:
                raise ValueError("Window source training must not inspect test losses")
            source_rows.append({"dataset":base["dataset"],"subset":base["subset"],"seed":base["seed"],"window_length":length,
                "reused_original":length==12,"checkpoint":str(path),"epochs_completed":source["epochs_completed"],
                "elapsed_seconds":source["elapsed_seconds"],"test_loss_logging_enabled":source["test_loss_logging_enabled"]})
            source_hash=runner.controls.sha256_file(path)
            for optimizer,arm in runner.window_runs(runner.window_protocol(length)):
                if length==12:
                    ref=runner.controls.reference(base,arm,optimizer)
                    truth=np.load(ref["label_path"],allow_pickle=False).reshape(-1)[:n]
                    if not np.array_equal(truth,labels):
                        raise ValueError("Original common prefix differs")
                    scores=np.load(ref["score_path"],allow_pickle=False).reshape(-1)[:n]
                    record={"dataset":base["dataset"],"subset":base["subset"],"seed":base["seed"],"optimizer":optimizer,
                        "arm":arm,"window_length":12,"auprc":runner.controls.metric(labels,scores),"test_points":n,
                        "optimizer_steps":0 if arm=="frozen" else n//12,"test_rows_omitted":base["data_manifest"]["test_rows"]-n,
                        "reused_original":True,"physical_source":ref["score_path"],"valid":True}
                else:
                    directory=runner.window_directory(base,length,optimizer,arm)
                    if directory in failed:
                        meta=json.loads((directory/"failure.json").read_text())
                        identity={"dataset":base["dataset"],"subset":base["subset"],"seed":base["seed"],
                            "optimizer":optimizer,"arm":arm,"window_length":length,"complete":False,
                            "status":"invalid","source_checkpoint_sha256":source_hash,"data_manifest":base["data_manifest"],
                            "expected_test_points":n,"metric_version":runner.controls.VERSION}
                        if any(meta.get(k)!=v for k,v in identity.items()) or meta.get("error_type")!="FloatingPointError":
                            raise ValueError(f"Invalid failure provenance: {directory}")
                        physical.append({k:v for k,v in meta.items() if k!="data_manifest"} | {
                            "valid":False,"auprc":np.nan,"physical_source":str(directory),"reused_original":False})
                        continue
                    meta=read_result(directory,base,optimizer,arm,length,labels,source_hash)
                    if meta["selection_kind"]!="online":
                        raise ValueError("New window selection must be online")
                    record={k:v for k,v in meta.items() if k!="data_manifest"}
                    record.update(reused_original=False,physical_source=str(directory),valid=True)
                physical.append(record)
    frame=pd.DataFrame(physical)
    if len(frame)!=len(bases)*27 or len(source_rows)!=len(bases)*3:
        raise ValueError("Incomplete common-window grid")
    if (frame[(frame.window_length!=12) & frame.valid].groupby(
            ["dataset","subset","seed","window_length"]).selector_tau.nunique()!=1).any():
        raise ValueError("Matched window arms do not share their source-validation P95")
    paired,cells=window_pair_cells(frame)
    if len(cells)!=len(datasets)*20*3:
        raise ValueError("Unexpected logical window grid size")
    return frame,cells,pd.DataFrame(source_rows),paired


def window_pair_cells(frame):
    keys=["dataset","subset","seed","window_length"]
    if frame.duplicated(keys+["optimizer","arm"]).any():
        raise ValueError("Duplicate physical window results")
    frozen=frame[frame.arm=="frozen"][keys+["auprc","valid"]].rename(columns={"auprc":"frozen_auprc","valid":"frozen_valid"})
    paired=frame[frame.arm!="frozen"].merge(frozen,on=keys,validate="many_to_one")
    paired["pair_valid"]=paired.valid & paired.frozen_valid
    paired["delta"]=paired.auprc-paired.frozen_auprc
    rows=[]
    cell_keys=["dataset","seed","optimizer","window_length"]
    for key,group in paired.groupby(cell_keys):
        record=dict(zip(cell_keys,key))
        expected=runner.controls.EXPECTED_ENTITIES[record["dataset"]]
        supports=[]
        for arm in runner.ARMS:
            arm_rows=group[group.arm==arm]
            if len(arm_rows)!=expected or arm_rows.subset.nunique(dropna=False)!=expected:
                raise ValueError(f"Missing physical entity support: {key}, {arm}")
            supports.append(set(arm_rows.subset))
            valid=int(arm_rows.pair_valid.sum())
            record[f"{arm}_valid_entities"]=valid
            record[f"{arm}_delta"]=float(arm_rows.delta.median()) if valid==expected else np.nan
        if supports[0]!=supports[1]:
            raise ValueError(f"Unmatched F/P/O entities: {key}")
        record["expected_entities"]=expected
        record["valid"]=all(record[f"{arm}_valid_entities"]==expected for arm in runner.ARMS)
        record["category"]=category(record["practical_delta"],record["oracle_delta"]) if record["valid"] else "UNDEFINED"
        rows.append(record)
    cells=pd.DataFrame(rows)
    baseline=cells[cells.window_length==12][["dataset","seed","optimizer","category","oracle_delta"]].rename(
        columns={"category":"original_category","oracle_delta":"original_oracle_delta"})
    cells=cells.merge(baseline,on=["dataset","seed","optimizer"],validate="many_to_one")
    direction=lambda x: "undefined" if not np.isfinite(x) else "negative" if x<0 else "positive" if x>0 else "zero"
    cells["original_oracle_direction"]=cells.original_oracle_delta.map(direction)
    cells["oracle_direction"]=cells.oracle_delta.map(direction)
    cells["category_changed"]=(cells.category!=cells.original_category).where(
        (cells.category!="UNDEFINED") & (cells.original_category!="UNDEFINED"))
    cells["oracle_direction_changed"]=(cells.oracle_direction!=cells.original_oracle_direction).where(
        (cells.oracle_direction!="undefined") & (cells.original_oracle_direction!="undefined"))
    return paired,cells


def verify_original_window_cells(cells,plan):
    original=pd.DataFrame(plan["original_e17_cells"])
    keys=["dataset","seed","optimizer","window_length"]
    check=cells[cells.dataset.isin(runner.WINDOW_DATASETS)].merge(original,on=keys,suffixes=("","_saved"),validate="one_to_one")
    if len(check)!=180 or not check.valid.all():
        raise ValueError("Original E17 support changed")
    errors=[]
    for column in ("practical_delta","oracle_delta"):
        error=float(np.max(np.abs(check[column]-check[f"{column}_saved"])))
        errors.append(error)
        np.testing.assert_allclose(check[column],check[f"{column}_saved"],rtol=0,atol=1e-12)
    for column in ("category","original_category"):
        if not (check[column]==check[f"{column}_saved"]).all():
            raise ValueError("Original E17 categories changed")
    summary=check.groupby(["dataset","window_length"])[["practical_delta","oracle_delta"]].median().reset_index()
    saved=pd.DataFrame(plan["original_e17_dataset_summary"])
    joined=summary.merge(saved,on=["dataset","window_length"],validate="one_to_one",suffixes=("","_saved"))
    for column in ("practical_delta","oracle_delta"):
        np.testing.assert_allclose(joined[column],joined[f"{column}_saved"],rtol=0,atol=1e-12)
    return {"status":"passed","original_cells":len(check),"max_absolute_delta_error":max(errors)}


def save_count(plan,physical,paired,cells,executed):
    out=runner.COUNT/"derived"
    out.mkdir(exist_ok=False)
    for name,frame in (("physical_logical",physical),("paired_entities",paired),("cells",cells),("unique_executions",executed)):
        frame.to_csv(out/f"{name}.csv",index=False)
    counts=cells.groupby(["sample_seed","category"]).size().unstack(fill_value=0).reindex(columns=CATEGORIES,fill_value=0).reset_index()
    counts.to_csv(out/"outcome_counts.csv",index=False)
    cells.groupby(["sample_seed","original_category","category"]).size().rename("cells").reset_index().to_csv(out/"transitions.csv",index=False)
    ni=cells[cells.original_category=="NI"].groupby("sample_seed").agg(original_NI=("category","size"),retained_NI=("category",lambda x:int((x=="NI").sum()))).reset_index()
    ni.to_csv(out/"original_NI_retention.csv",index=False)
    summary=cells.groupby(["sample_seed","dataset"])[["practical_delta","oracle_delta"]].median().reset_index()
    summary.to_csv(out/"dataset_summary.csv",index=False)
    physical.groupby(["sample_seed","dataset","arm"]).agg(logical_runs=("auprc","size"),parameter_trace_available=("parameter_trace_available","sum"),
        median_parameter_step_sum=("parameter_step_l2_sum","median"),median_selected_points=("selected_points","median")).reset_index().to_csv(out/"update_diagnostics.csv",index=False)
    fig,ax=plt.subplots(figsize=(7,4),constrained_layout=True)
    for i,sample in enumerate(runner.SAMPLE_SEEDS):
        row=counts[counts.sample_seed==sample].iloc[0]
        ax.bar(np.arange(4)+.25*(i-1),[row[c] for c in CATEGORIES],width=.25,label=str(sample))
    ax.set(xticks=np.arange(4),xticklabels=CATEGORIES,ylabel="Logical cells (180 per sampling repeat)")
    ax.legend(title="Sampling seed",fontsize=8)
    for suffix in ("png","pdf"):
        fig.savefig(out/f"matched_count_outcomes.{suffix}",dpi=180)
    plt.close(fig)
    lines=["# Per-window count matching: complete results","",f"All 12,720 logical arm observations are complete, including {len(executed):,} unique new GPU trajectories. Original streams and identical matched masks are explicitly reused.",
        "Each of three sampling repetitions retains 180 cells. Model seeds 42-46 and sampling seeds are separate dimensions. Fixed-mask replay is not a rerun of the dynamic Practical selector.","",
        "| Sampling seed | BI | OOG | NI | POG | Original 90 NI retained |","|---|---:|---:|---:|---:|---:|"]
    retention=ni.set_index("sample_seed")
    for row in counts.itertuples():
        lines.append(f"| {row.sample_seed} | {row.BI} | {row.OOG} | {row.NI} | {row.POG} | {retention.loc[row.sample_seed,'retained_NI']} |")
    lines += ["","Both arms have identical per-window counts, full-window reduction, and step schedules. Gradient directions and actual displacements may still differ.",
        "Parameter displacements for reused results were not fabricated. update_diagnostics.csv reports available denominators. Repeated observations are not presented as independent GPU experiments or independent datasets.",
        "","[Cells](cells.csv), [entity pairs](paired_entities.csv), [provenance mapping](physical_logical.csv), [new runs](unique_executions.csv).",""]
    (out/"RESULTS.md").write_text("\n".join(lines))
    runner.controls.atomic_json(out/"summary.json",{"status":"complete","finished_at":runner.controls.now(),
        "logical_arm_observations":len(physical),"unique_new_runs":len(executed),"logical_cells":len(cells),
        "exact_reuse_accounting":plan["expected"],"outcome_counts":counts.to_dict(orient="records"),
        "original_NI_retention":ni.to_dict(orient="records"),"summed_new_worker_run_hours":float(executed.wall_seconds.sum()/3600)})


def save_window(physical,cells,sources,paired,*,plan=None):
    out=runner.WINDOW/"derived"
    preservation=verify_original_window_cells(cells,plan) if plan is not None else None
    if plan is not None:
        validation=json.loads((runner.WINDOW/"extension_validation.json").read_text())
        for path,digest in validation["protected_files_sha256"].items():
            if runner.controls.sha256_file(Path(path))!=digest:
                raise ValueError(f"Protected main artifact changed: {path}")
    out.mkdir(exist_ok=plan is not None)
    physical.to_csv(out/"physical.csv",index=False)
    cells.to_csv(out/"cells.csv",index=False)
    sources.to_csv(out/"sources.csv",index=False)
    paired.to_csv(out/"paired_entities.csv",index=False)
    physical[~physical.valid].to_csv(out/"invalid_runs.csv",index=False)
    columns=[*CATEGORIES,"UNDEFINED"]
    counts=cells.groupby(["window_length","category"]).size().unstack(fill_value=0).reindex(columns=columns,fill_value=0).reset_index()
    counts.to_csv(out/"outcome_counts.csv",index=False)
    dataset_counts=cells.groupby(["dataset","window_length","category"]).size().unstack(fill_value=0).reindex(columns=columns,fill_value=0).reset_index()
    dataset_counts.to_csv(out/"dataset_outcome_counts.csv",index=False)
    cells.groupby(["window_length","original_category","category"]).size().rename("cells").reset_index().to_csv(out/"transitions.csv",index=False)
    cells.groupby(["dataset","window_length","original_category","category"]).size().rename("cells").reset_index().to_csv(out/"dataset_transitions.csv",index=False)
    cells.groupby(["dataset","window_length","original_oracle_direction","oracle_direction"]).size().rename("cells").reset_index().to_csv(out/"oracle_direction_transitions.csv",index=False)
    rows=[]
    for (dataset,width),group in cells.groupby(["dataset","window_length"]):
        if len(group)!=20:
            raise ValueError("Each family/length requires four optimizers and five seeds")
        row={"dataset":dataset,"window_length":width,"expected_cells":20,"valid_cells":int(group.valid.sum())}
        for arm in runner.ARMS:
            row[f"{arm}_delta"]=float(group[f"{arm}_delta"].median()) if np.isfinite(group[f"{arm}_delta"]).all() else np.nan
        rows.append(row)
    summary=pd.DataFrame(rows)
    summary.to_csv(out/"dataset_summary.csv",index=False)
    supports=[]
    for width,group in physical.groupby("window_length"):
        selected_cells=cells[cells.window_length==width]
        entity_validity=group.groupby(["dataset","subset"]).valid.all()
        entity_pairs=paired[paired.window_length==width].groupby(["dataset","subset","seed","optimizer"]).pair_valid.all()
        supports.append({"window_length":width,"expected_logical_cells":len(selected_cells),
            "valid_logical_cells":int(selected_cells.valid.sum()),"undefined_logical_cells":int((~selected_cells.valid).sum()),
            "expected_entities":len(entity_validity),"entities_with_all_runs_valid":int(entity_validity.sum()),
            "expected_physical_runs":len(group),"valid_physical_runs":int(group.valid.sum()),
            "invalid_physical_runs":int((~group.valid).sum()),
            "expected_paired_entity_cells":len(entity_pairs),"valid_paired_entity_cells":int(entity_pairs.sum())})
    support_frame=pd.DataFrame(supports)
    support_frame.to_csv(out/"support_counts.csv",index=False)
    fig,ax=plt.subplots(figsize=(7,4),constrained_layout=True)
    for dataset,group in summary.groupby("dataset"):
        ax.plot(group.window_length,100*group.oracle_delta,"o-",label=dataset)
    ax.axhline(.5,color="grey",linestyle="--",linewidth=.7)
    ax.set(xticks=[12,24,48],xlabel="Window = train/test stride",ylabel="Median Oracle minus matched Frozen (pp)")
    ax.legend(ncol=3,fontsize=7)
    for suffix in ("png","pdf"):
        fig.savefig(out/f"window_length_effects.{suffix}",dpi=180)
    plt.close(fig)
    new_sources=int((sources.window_length!=12).sum())
    new_evaluations=int((physical.window_length!=12).sum())
    invalid=int((~physical.valid).sum())
    pp=lambda value: f"{100*value:.4f}" if np.isfinite(value) else "undefined"
    lines=["# E17 window-length sensitivity", "",
        f"{cells.dataset.nunique()} datasets; w=24/48 include {new_sources} source runs and {new_evaluations} evaluation streams ({invalid} invalid).",
        "Length-12 source models and original scores are reused read-only. Completed trajectories for the original three datasets are not rerun.",
        "Each physical entity uses the common prefix floor(N_test/48)*48. Length-12 PR-AUC is also recomputed on this prefix.",
        "Effects are paired at entity level, then summarized by medians within dataset x optimizer x seed. The table reports the median over 20 cells in each family.",
        "Family medians require all 20 cells in that arm to be valid; paired categories require both arms to be valid. Undefined cells are not reweighted over remaining optimizers/seeds.",
        "The metric is point-level, unadjusted, trapezoidal PR-AUC; clear gain is strictly delta>0.005. Any invalid entity run makes its full-support logical cell undefined. Entities are not dropped and invalid cells are not counted as NI.","",
        "| Dataset | Window length | Practical-F (pp) | Oracle-F (pp) | Valid paired cells/20 |","|---|---:|---:|---:|---:|"]
    lines += [f"| {r.dataset} | {r.window_length} | {pp(r.practical_delta)} | {pp(r.oracle_delta)} | {r.valid_cells}/20 |" for r in summary.itertuples()]
    lines += ["","| Window length | BI | OOG | NI | POG | Undefined |","|---|---:|---:|---:|---:|---:|"]
    lines += [f"| {r.window_length} | {r.BI} | {r.OOG} | {r.NI} | {r.POG} | {r.UNDEFINED} |" for r in counts.itertuples()]
    if invalid:
        lines += ["","Numerical failures (full error locations in invalid_runs.csv; no finite-truncation metrics were computed):",""]
        for (dataset,subset,width,optimizer,arm),group in physical[~physical.valid].groupby(
                ["dataset","subset","window_length","optimizer","arm"]):
            seeds=", ".join(str(int(seed)) for seed in sorted(group.seed))
            lines += [f"- {dataset}/{subset or '__root__'}，w={width}，{optimizer}，{arm}：seeds {seeds}。"]
    if preservation is not None:
        lines += ["",f"All 180 cells and nine family x window summaries for the original three E17 families were reproduced; maximum absolute cell-effect error {preservation['max_absolute_delta_error']:.3g}."]
    lines += ["","This changes context length, AE capacity under the original scaling rule, and updates per unit time simultaneously. It neither isolates a causal effect of length nor selects windows by system period.",
        "Optimizer/seed cells are repeated configurations within each family, not independent datasets. Original w=12 main-text results and frozen evidence are unchanged.",
        "","[Cells](cells.csv), [paired entities](paired_entities.csv), [original stream references](physical.csv), [source training records](sources.csv),",
        "[Valid support](support_counts.csv), [family categories](dataset_outcome_counts.csv), [category changes](dataset_transitions.csv),",
        "[Oracle direction changes](oracle_direction_transitions.csv), [invalid runs](invalid_runs.csv).",""]
    (out/"RESULTS.md").write_text("\n".join(lines))
    runner.controls.atomic_json(out/"summary.json",{"status":"complete" if invalid==0 else "complete_with_invalid_runs","finished_at":runner.controls.now(),
        "new_source_checkpoints":new_sources,"new_evaluations":new_evaluations,
        "reused_original_evaluations":int((physical.window_length==12).sum()),"logical_cells":len(cells),
        "valid_logical_cells":int(cells.valid.sum()),"invalid_physical_runs":invalid,
        "original_e17_preservation":preservation,"support_counts":supports,
        "extension_plan":str(runner.WINDOW/"extension_plan.json") if plan is not None else None,
        "outcome_counts":counts.to_dict(orient="records"),
        "summed_new_worker_evaluation_hours":float(physical.loc[~physical.reused_original,"wall_seconds"].sum()/3600)})


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-only",action="store_true")
    args=parser.parse_args()
    runner.controls.configure_threads()
    if args.window_only:
        plan=json.loads((runner.WINDOW/"extension_plan.json").read_text())
        window=window_frames(plan)
        save_window(*window,plan=plan)
        print("COMPLETE: E17 nine-family grid, retained support, common-prefix metrics, masks and steps verified",flush=True)
    else:
        plan=json.loads((runner.HERE/"plan.json").read_text())
        count=count_frames(plan)
        window=window_frames(plan)
        save_count(plan,*count)
        save_window(*window)
        print("COMPLETE: both prespecified grids, exact reuse, labels, metrics and step traces verified",flush=True)
