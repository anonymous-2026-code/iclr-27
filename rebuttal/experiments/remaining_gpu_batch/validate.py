"""Finite checks before launching the fixed four-GPU experiment grid."""
import json
from pathlib import Path
import time
import experiment as runner
import numpy as np
import pandas as pd
import torch


def validate(plan):
    if (runner.HERE/"validation.json").exists() or (runner.HERE/"validation").exists():
        raise FileExistsError("Finite validation was already attempted")
    started=time.perf_counter()
    out=runner.HERE/"validation"
    out.mkdir()
    torch.cuda.set_device(0)
    p=np.array([[True]*12,[True]*9+[False]*3,[True]*11+[False],[False]*12])
    o=np.array([[True]*12,[True]*7+[False]*5,[False]*12,[True]*12])
    original={"practical":p,"oracle":o}
    before=torch.get_rng_state().clone()
    numpy_before=np.random.get_state()
    masks=runner.matched_masks(original,17)
    repeated=runner.matched_masks(original,17)
    np.testing.assert_array_equal(torch.get_rng_state().numpy(),before.numpy())
    numpy_after=np.random.get_state()
    np.testing.assert_array_equal(numpy_before[1],numpy_after[1])
    assert numpy_before[0]==numpy_after[0] and numpy_before[2:]==numpy_after[2:]
    for key,mask in masks.items():
        np.testing.assert_array_equal(mask,repeated[key])
        np.testing.assert_array_equal(mask.sum(1),np.minimum(p.sum(1),o.sum(1)))
        assert not np.any(mask & ~original[key[1]])
    same={"practical":o,"oracle":o.copy()}
    slots,executions=runner.deduplicate(same,runner.matched_masks(same,17))
    assert len(slots)==6 and not executions and all(s["reference"]["kind"]=="original" for s in slots)
    assert any(not np.array_equal(masks[runner.SAMPLE_SEEDS[0],a],masks[runner.SAMPLE_SEEDS[1],a]) for a in runner.ARMS)

    base_id,base=next((k,b) for k,b in plan["bases"].items() if b["dataset"]=="Daphnet" and b["seed"]==42)
    runner.check_base(base)
    series=runner.controls.prepare_series(Path(base["directory"]),runner.controls.load_protocol().train_fraction)
    parity=[]
    completed_count=[]
    for optimizer in ("sgd_ttlr0.001","adamw_ttlr0.0001"):
        labels,old,tau=runner.original_inputs(base,optimizer,verify=True)
        for arm in runner.ARMS:
            meta=runner.execute(base,runner.controls.load_protocol(),base["checkpoint"],series,
                out/"original_replay"/optimizer/arm,optimizer,arm,labels,fixed_mask=old[arm],expected_tau=tau,
                parity_reference=runner.controls.reference(base,arm,optimizer))
            trace=pd.read_parquet(Path(meta["output_path"])/"window_trace.parquet")
            assert trace.optimizer_step.all() and len(trace)==len(labels)//12
            assert (trace.loc[trace.selected_points==0,"masked_loss"]==0).all()
            parity.append({"optimizer":optimizer,"arm":arm,"max_score_error":meta["max_score_parity_error"],
                "auprc_error":abs(meta["auprc"]-runner.controls.reference(base,arm,optimizer)["auprc"]),
                "zero_selected_windows":int((trace.selected_points==0).sum()),"seconds":meta["wall_seconds"]})
        task=next(t for t in plan["tasks"] if t["experiment"]=="count" and t["base_id"]==base_id and t["optimizer"]==optimizer)
        selection=runner.matched_masks(old,task["task_id"])
        reference=next(e for e in task["executions"] if e["arm"]=="oracle")
        result=runner.execute(base,runner.controls.load_protocol(),base["checkpoint"],series,
            runner.count_directory(base,optimizer,reference),optimizer,reference["arm"],labels,
            fixed_mask=selection[reference["sample_seed"],reference["arm"]],expected_tau=tau)
        assert result["optimizer_steps"]==len(labels)//12
        completed_count.append({"task_id":task["task_id"],"sample_seed":reference["sample_seed"],"arm":reference["arm"]})
    window_task=next(t for t in plan["tasks"] if t["experiment"]=="window" and t["base_id"]==base_id and t["length"]==24)
    runner.run_window_task(plan,window_task)
    checkpoint=runner.source_path(base,24)
    source=json.loads((checkpoint.parent/"complete.json").read_text())
    assert source["test_loss_logging_enabled"] is False and source["protocol"]["sequence_length"]==24
    values=[]
    from aggregate import read_result
    for optimizer,arm in runner.window_runs(runner.window_protocol(24)):
        path=runner.window_directory(base,24,optimizer,arm)
        result=json.loads((path/"result.json").read_text())
        truth=np.load(path/"true.npy",allow_pickle=False)
        np.testing.assert_array_equal(truth,series.test_labels[:window_task["common_points"]])
        assert result["optimizer_steps"]==(0 if arm=="frozen" else len(truth)//24)
        read_result(path,base,optimizer,arm,24,truth,runner.controls.sha256_file(checkpoint))
        values.append({"optimizer":optimizer,"arm":arm,"points":len(truth),"seconds":result["wall_seconds"]})
    # Lower-cost shape and prefix checks also cover the other declared length.
    from section5_repro.model import WindowAutoencoder
    model=WindowAutoencoder(48,9)
    with torch.no_grad():
        assert model(torch.zeros(1,48,9)).shape==(1,48,9)
    assert all(t["common_points"]%48==0 for t in plan["tasks"] if t["experiment"]=="window")
    for completed in completed_count:
        task=next(t for t in plan["tasks"] if t["task_id"]==completed["task_id"])
        labels,original,_=runner.original_inputs(base,task["optimizer"])
        mask=runner.matched_masks(original,task["task_id"])[completed["sample_seed"],completed["arm"]]
        read_result(runner.count_directory(base,task["optimizer"],completed),base,task["optimizer"],
                    completed["arm"],12,labels,base["source_checkpoint_sha256"],mask)
    result={"status":"passed","finished_at":runner.controls.now(),"matching_count_subset_rng_and_exact_reuse":"passed",
        "original_full_stream_parity":parity,"new_window24_full_task":values,
        "window48_shape_and_all_common_prefixes":"passed","aggregation_artifact_checks":"passed","completed_count_runs":completed_count,
        "completed_window_task_ids":[window_task["task_id"]],"wall_seconds":time.perf_counter()-started}
    runner.controls.atomic_json(runner.HERE/"validation.json",result)
    print(json.dumps(result),flush=True)


def validate_window_extension(plan):
    """Reuse the audited E17 GPU preflight; check the extended support on CPU."""
    destination=runner.WINDOW/"extension_validation.json"
    if destination.exists():
        raise FileExistsError("Window extension preflight already exists")
    old_validation=json.loads((runner.HERE/"validation.json").read_text())
    if old_validation["status"]!="passed" or json.loads((runner.HERE/"status.json").read_text())["status"]!="complete":
        raise ValueError("The audited original E17 batch must be complete")
    from aggregate import window_frames, window_pair_cells, verify_original_window_cells, category
    old_plan=json.loads((runner.HERE/"plan.json").read_text())
    physical,cells,sources,_=window_frames(old_plan)
    preservation=verify_original_window_cells(cells,plan)
    old_physical=pd.read_csv(runner.WINDOW/"derived/physical.csv",keep_default_na=False)
    keys=["dataset","subset","seed","optimizer","arm","window_length"]
    check=physical.merge(old_physical,on=keys,validate="one_to_one",suffixes=("","_saved"))
    if len(check)!=405 or not (check.physical_source==check.physical_source_saved).all():
        raise ValueError("Existing E17 physical provenance changed")
    np.testing.assert_allclose(check.auprc,check.auprc_saved,rtol=0,atol=1e-12)
    base_protocol=runner.window_protocol(12).as_dict()
    for width in (24,48):
        new=runner.window_protocol(width).as_dict()
        changed={key for key in new if new[key]!=base_protocol[key]}
        if changed!={"sequence_length","train_stride","test_stride","protocol_id"}:
            raise ValueError(f"Unintended protocol changes: {changed}")
        if not new["tta_test_batch_size"]==1 or not new["source_learning_rate"]==1e-4:
            raise ValueError("Source/adaptation configuration differs")
    # Meaningful aggregation regression: median(P-F) differs from median(P)-median(F).
    synthetic=[]
    for i in range(25):
        frozen=.1 if i<12 else .8
        practical=.3 if i<12 else .9 if i<24 else .2
        for arm,value in (("frozen",frozen),("practical",practical),("oracle",frozen-.02)):
            synthetic.append({"dataset":"MSL","subset":str(i),"seed":42,"window_length":12,
                "optimizer":"shared" if arm=="frozen" else "sgd_ttlr0.001","arm":arm,"auprc":value,"valid":True})
    synthetic=pd.DataFrame(synthetic)
    _,regression=window_pair_cells(synthetic)
    assert len(regression)==1 and abs(regression.iloc[0].practical_delta-.1)<1e-12
    assert regression.iloc[0].category=="POG" and regression.iloc[0].expected_entities==25
    synthetic.loc[(synthetic.subset=="0") & (synthetic.arm=="oracle"),["valid","auprc"]]=[False,np.nan]
    _,invalid=window_pair_cells(synthetic)
    assert invalid.iloc[0].category=="UNDEFINED" and np.isnan(invalid.iloc[0].oracle_delta)
    assert invalid.iloc[0].oracle_valid_entities==24
    assert category(.005,.005)=="NI" and category(np.nextafter(.005,np.inf),np.nextafter(.005,np.inf))=="BI"
    raw_runs,max_error=0,0.0
    for index,base in enumerate(plan["bases"].values()):
        runner.check_base(base)
        n=base["data_manifest"]["test_rows"]//48*48
        full_labels=np.load(runner.controls.reference(base,"frozen","shared")["label_path"],allow_pickle=False).reshape(-1)
        if len(full_labels)!=base["data_manifest"]["test_rows"]//12*12 or n<=0 or set(np.unique(full_labels[:n]))!={0,1}:
            raise ValueError("Invalid common-prefix reference")
        truth=pd.read_parquet(Path(base["directory"])/"test.parquet",columns=["label"]).label.to_numpy()
        if not np.array_equal(full_labels,truth[:len(full_labels)]):
            raise ValueError("Original point alignment differs from processed labels")
        for optimizer,arm in runner.window_runs(runner.window_protocol(12)):
            ref=runner.controls.reference(base,arm,optimizer)
            labels=np.load(ref["label_path"],allow_pickle=False).reshape(-1)
            scores=np.load(ref["score_path"],allow_pickle=False).reshape(-1)
            if not np.array_equal(labels,full_labels):
                raise ValueError("Original F/P/O point alignment differs")
            meta=json.loads(Path(ref["manifest_path"]).read_text())
            if meta["status"]!="complete" or meta["data_manifest"]!=base["data_manifest"] or meta["source_checkpoint_hash"]!=base["source_checkpoint_sha256"]:
                raise ValueError("Original F/P/O source/data provenance differs")
            error=abs(runner.controls.metric(labels,scores)-ref["auprc"])
            if error>1e-12:
                raise ValueError("Original full-prefix trapezoidal PR-AUC does not reproduce")
            runner.controls.metric(labels[:n],scores[:n])
            max_error=max(max_error,error)
            raw_runs+=1
        if (index+1)%100==0:
            print(f"VALIDATED {index+1}/530 retained entity/seed bases",flush=True)
    protected=[runner.ROOT/"configs/section5_ae.toml",runner.controls.PHYSICAL,runner.controls.SUPPORT,
        runner.ROOT/"workspace/rebuttal_inputs/p3_iclr27_corrected_auprc_common_support_cells.csv",
        runner.ROOT/"rebuttal/P3_ICLR27_REBUTTAL_EVIDENCE_FROZEN_20261113.md"]
    result={"status":"passed","finished_at":runner.controls.now(),"original_e17_preservation":preservation,
        "audited_gpu_preflight_reused":str(runner.HERE/"validation.json"),
        "new_gpu_preflight_runs":0,"raw_streams_verified":raw_runs,"max_raw_pr_auc_error":max_error,
        "original_e17_physical_streams_verified":len(physical),"original_e17_sources_verified":len(sources),
        "paired_entity_median_and_strict_gain_checks":"passed","invalid_entity_preserves_undefined_cell":"passed",
        "original_label_alignment_and_common_prefixes":"passed","length_protocol_only_declared_changes":"passed",
        "protected_files_sha256":{str(path):runner.controls.sha256_file(path) for path in protected},
        "expected":plan["expected"]}
    runner.controls.atomic_json(destination,result)
    print(json.dumps(result),flush=True)
