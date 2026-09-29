"""Fresh final population for minimal-reward retraining (seeds 2026090711--13).

Each output records its own generator seed. Ordered worker results make scene
identity independent of worker completion order. Physics/initial-graspability
qualification remains a separate, policy-independent step before final freezing.
"""
from pathlib import Path
import argparse
import hashlib
import json
import multiprocessing as mp
import time
import tempfile
import build_dataset as build
from scene_qa import audit_scene


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def worker(payload):
    tier,index,seed,args=payload
    rejection=0
    while True:
        draw_seed=seed+rejection*1000000007
        objs,blocked=build._gen_one((draw_seed,index,args))
        with tempfile.TemporaryDirectory(prefix="retrieval_scene_qa_") as d:
            path=Path(d)/"scene.txt"
            build.write_scene(objs,str(path))
            qa=audit_scene(str(path))
        if qa["max_overlap_cm2"]<=.30:
            return tier,index,draw_seed,rejection,objs,blocked,qa
        rejection+=1


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--output",required=True)
    ap.add_argument("--per-tier",type=int,default=200);ap.add_argument("--workers",type=int,default=2)
    ap.add_argument("--resume",action="store_true")
    a=ap.parse_args();out=Path(a.output).resolve()
    if out.exists() and not a.resume: raise FileExistsError("Use --resume for an interrupted generation")
    out.mkdir(parents=True,exist_ok=True)
    existing={sha(p) for p in (Path(__file__).resolve().parents[1]/"test-cases").rglob("*.txt")}
    definitions=[("easy",2026090711,3.,1.,12,[2,3],0.,.34,32.),
                 ("medium",2026090712,1.5,.5,14,[3,5],.2,.34,30.),
                 ("hard",2026090713,.75,.5,16,[4,6],.5,.33,30.)]
    payloads=[]
    configs={}
    for tier,seed,gap,tgap,blocked,big,motif,void,pocket in definitions:
        args=dict(num=a.per_tier,num_objects=[11,11],gap_mm=gap,target_gap_mm=tgap,
            min_blocked=blocked,big_range=big,motif_prob=motif,max_void_frac=void,
            max_pocket_mm=pocket,max_t4c_cm=20.,max_t3nn_mm=4.,concave_frac=.3,
            center_jitter=.03,yaw_mode="axis",compact_sweeps=2)
        configs[tier]=args
        for i in range(a.per_tier): payloads.append((tier,i,seed*1000003+i,args))
    previous=json.loads((out/"meta.json").read_text()) if a.resume else None
    meta=previous["scenes"] if previous else []
    if previous and previous["configurations"]!=configs: raise ValueError("Generation settings changed")
    for row in meta:
        if sha(out/row["scene"])!=row["sha256"]: raise ValueError("Previously accepted scene changed")
    for p in out.glob("*.txt"):
        if int(p.stem)>=len(meta):
            rejected=out/"rejected_static";rejected.mkdir(exist_ok=True)
            p.rename(rejected/p.name)
    new_hashes={r["sha256"] for r in meta};started=time.time()
    def save(state):
        payload=dict(status=state,role="independent_final_candidates_not_policy_evaluated",
            qualification="Static gates only; physics stability and initial graspability pending",
            source_sha256={p:sha(Path(__file__).with_name(p)) for p in ["prepare_minimal_final_suite.py","build_dataset.py","scene_qa.py"]},
            configurations=configs,scenes=meta,started_unix=started,updated_unix=time.time())
        tmp=out/"meta.json.tmp";tmp.write_text(json.dumps(payload,indent=2)+"\n");tmp.replace(out/"meta.json")
    save("generating")
    with mp.Pool(a.workers) as pool:
        for tier,index,seed,rejections,objs,blocked,qa in pool.imap(worker,payloads[len(meta):]):
            path=out/f"{len(meta):06d}.txt"
            build.write_scene(objs,str(path))
            h=sha(path)
            if h in existing or h in new_hashes: raise RuntimeError("Generated duplicate scene")
            new_hashes.add(h)
            meta.append(dict(scene=path.name,tier=tier,generator_seed=seed,tier_index=index,
                sha256=h,blocked16=blocked,static_qa=qa,static_rejected_draws=rejections,
                tightness=build.tightness_metrics(objs),layout=build.layout_metrics(objs)))
            save("generating")
            if len(meta)%10==0: print(f"Generated {len(meta)}/{len(payloads)} unique candidates ({time.time()-started:.1f}s)",flush=True)
    save("static_generation_complete")
    print(f"COMPLETE: {len(meta)} independent candidates; not yet physically qualified",flush=True)


if __name__=="__main__": main()
