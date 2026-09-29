"""Sequential, bounded GPU campaign with trace validation and live status.

Nominal planning and every execution actor run in separate fresh processes.
Never silently resume under changed code or overwrite an existing experiment.
"""
from pathlib import Path
import argparse
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from validate_retrieval_run import validate,digest
from teacher_reference import teacher_path

ROOT=Path(__file__).resolve().parents[1]


def write_json(path,payload):
    tmp=Path(str(path)+".tmp");tmp.write_text(json.dumps(payload,indent=2)+"\n");tmp.replace(path)


def wait_for_gpu_memory(gpu,root,status,minimum_mib=6144,timeout=3600):
    """Wait for headroom; an OOM race is handled separately with bounded retries."""
    deadline=time.monotonic()+timeout
    while True:
        free=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),
            '--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True,timeout=15).strip())
        if free>=minimum_mib:
            status.update(state='running',gpu_free_mib=free);write_json(root/'status.json',status)
            return
        status.update(state='waiting_for_gpu_memory',gpu_free_mib=free,required_free_mib=minimum_mib,updated_unix=time.time())
        write_json(root/'status.json',status)
        if time.monotonic()>=deadline: raise RuntimeError(f'GPU {gpu} memory unavailable for {timeout}s')
        time.sleep(10)


def launch_with_resource_retries(spec_path,out,root,gpu,status,max_retries=3):
    log_path=root/(out.name+'.log')
    for attempt in range(max_retries+1):
        wait_for_gpu_memory(gpu,root,status)
        with log_path.open('w') as log:
            p=subprocess.Popen(['bash',str(ROOT/'tools/run_retrieval_evaluation.sh'),str(spec_path),str(gpu)],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            status['active_pid']=p.pid;write_json(root/'status.json',status)
            try: rc=p.wait(timeout=1800)
            except BaseException:
                os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=30);raise
        if rc==0: return
        content=log_path.read_text(errors='replace').lower()
        oom='cuda out of memory' in content or 'cuda error: out of memory' in content
        if not oom or (out/'complete.json').exists() or attempt==max_retries:
            raise RuntimeError(f'{out.name} exited {rc}; see {log_path}')
        # Preserve technical failures outside the accepted batch namespace.
        # Retry exactly the same scene/seed/spec, never a scored task failure.
        archive=root/'infrastructure_failures'/f'{out.name}_attempt{attempt+1}'
        archive.parent.mkdir(exist_ok=True)
        if out.exists(): out.rename(archive)
        else: archive.mkdir()
        log_path.rename(archive/'process.log')
        status.setdefault('resource_retries',[]).append(dict(job=out.name,attempt=attempt+1,
            reason='CUDA out of memory',gpu=gpu,archive=str(archive),spec_sha256=digest(spec_path),time_unix=time.time()))
        write_json(root/'status.json',status)


def run_job(spec,root,gpu,status):
    frozen=json.loads((root/'frozen_inputs.json').read_text())
    if any(digest(p)!=h for p,h in frozen.items()):
        raise RuntimeError('Source/configuration/checkpoint changed since campaign launch')
    out=Path(spec["output"])
    spec_path=root/(out.name+".spec.json")
    write_json(spec_path,spec)
    if (out/"complete.json").exists():
        old=json.loads((out/"provenance.json").read_text())
        assert old["spec"]==spec,"Refusing resume with changed specification"
        assert all(digest(ROOT/p)==h for p,h in old["source_sha256"].items()),"Source changed; use new campaign"
        return validate(out)
    if out.exists(): raise RuntimeError(f"Incomplete previous job at {out}; inspect it and use a fresh campaign")
    status.update(state="running",active=out.name,updated_unix=time.time())
    write_json(root/"status.json",status)
    launch_with_resource_retries(spec_path,out,root,gpu,status)
    report=validate(out)
    if any(digest(p)!=h for p,h in frozen.items()):
        raise RuntimeError('Source/configuration/checkpoint changed during job')
    status["completed"].append(out.name)
    status.update(updated_unix=time.time(),active_pid=None)
    write_json(root/"status.json",status)
    print(out.name,json.dumps(report["cases"]),flush=True)
    return report


def paired_starts(outputs):
    reference=None;diffs={}
    for out,case in outputs:
        with np.load(out/(case+".npz")) as d:
            current=(d["initial_state"].copy(),d["initial_eef"].copy(),d["initial_q"].copy(),
                     d["requested_initial_state"].copy())
        if reference is None: reference=current
        np.testing.assert_array_equal(current[3],reference[3])
        diffs[case]={name:float(np.max(np.abs(a-b))) for name,a,b in zip(
            ["state","eef","q"],current[:3],reference[:3])}
    return dict(differences=diffs,matched=all(max(x.values())<=1e-5 for x in diffs.values()))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--teacher-checkpoint")
    ap.add_argument("--reward-recipe",choices=["legacy","graspability_only"],default=os.environ.get("TRACE_REWARD_RECIPE","legacy"))
    ap.add_argument("--manifest",required=True)
    ap.add_argument("--output",required=True)
    ap.add_argument("--cases",required=True,help="JSON list of actor specifications")
    ap.add_argument("--gpu",type=int,default=4)
    ap.add_argument("--count",type=int,default=12)
    ap.add_argument("--batch-size",type=int,default=64)
    ap.add_argument("--horizon",type=int,default=120)
    ap.add_argument("--seeds",type=int,nargs="+",default=[7])
    ap.add_argument("--pos-noise",type=float,default=.015)
    ap.add_argument("--yaw-noise",type=float,default=10.)
    a=ap.parse_args()
    # Detached/sandboxed shells may resolve `python` to a different major
    # version than the interpreter running this supervisor. Isaac Gym Preview 4
    # is ABI-bound to Python 3.8, so propagate this exact executable.
    os.environ.setdefault('TRACE_PYTHON',sys.executable)
    root=Path(a.output).resolve();root.mkdir(parents=True,exist_ok=True)
    teacher=teacher_path(a.teacher_checkpoint)
    a.teacher_checkpoint=str(teacher)
    cases=json.loads(Path(a.cases).read_text())
    manifest=Path(a.manifest).resolve()
    available=len(json.loads(manifest.read_text())["scenes"])
    if not 0<a.count<=available: raise ValueError("Requested count exceeds unique manifest scenes")
    watched={Path(__file__).resolve(),Path(a.cases).resolve(),manifest,
        ROOT/'tools/evaluate_retrieval.py',ROOT/'tools/run_retrieval_evaluation.sh',
        ROOT/'tools/validate_retrieval_run.py',ROOT/'tools/_net_compat.py',ROOT/'tools/_ckpt_compat.py',
        teacher,ROOT/'tools/teacher_reference.py',
        ROOT/'logs_grasp/grasp_model-89.pth',ROOT/'logs_grasp/snapshot-post-020000.reinforcement.pth'}
    for folder in ['tasks','learning','open_loop','utils']:
        watched.update((ROOT/folder).rglob('*.py'))
    watched.update((ROOT/'cfg').rglob('*.yaml'))
    watched.update((ROOT.parent/'assets/urdf/more/blocks-more').glob('*.obj'))
    for folder in ('ur5e_simplified','ur5e'):
        watched.update(p for p in (ROOT.parent/'assets/urdf/more'/folder).rglob('*')
                       if p.suffix.lower() in ('.urdf','.stl','.obj','.dae'))
    for case in cases:
        if case.get('checkpoint'):
            path=Path(case['checkpoint']).resolve()
            if path.is_dir(): path=next(p for p in [path/'student.pt',path/'student_best.pth'] if p.is_file())
            watched.add(path)
        if case.get('policy_stop_checkpoint'):
            path=Path(case['policy_stop_checkpoint']).resolve()
            if path.is_dir(): path=path/'policy_stop.pt'
            watched.add(path)
    watched.update((ROOT.parent/'cluster').glob('*.py'))
    watched.update((ROOT.parent/'cluster').glob('*.sh'))
    snapshot=root/'frozen_inputs.json'
    current={str(p):digest(p) for p in sorted(watched)}
    if snapshot.exists():
        if json.loads(snapshot.read_text())!=current: raise RuntimeError('Changed campaign inputs; use a new campaign')
    else: write_json(snapshot,current)
    status=dict(state="starting",pid=os.getpid(),started_unix=time.time(),completed=[],arguments=vars(a))
    write_json(root/"status.json",status)
    try:
        for seed in a.seeds:
            for offset in range(0,a.count,a.batch_size):
                size=min(a.batch_size,a.count-offset)
                tag=f"s{seed}_b{offset:04d}"
                plan_out=root/(tag+"_plan")
                common=dict(manifest=str(manifest),offset=offset,batch_size=size,horizon=a.horizon,
                    seed=seed,pos_noise=a.pos_noise,yaw_noise_deg=a.yaw_noise,cpu_threads=4,
                    teacher_checkpoint=str(teacher),reward_recipe=a.reward_recipe)
                run_job(dict(common,output=str(plan_out),cases=[]),root,a.gpu,status)
                paired=[]
                for case in cases:
                    out=root/(tag+"_"+case["name"])
                    spec=dict(common,output=str(out),cases=[case],plan_source=str(plan_out/"nominal.json"))
                    run_job(spec,root,a.gpu,status)
                    paired.append((out,case["name"]))
                starts=paired_starts(paired)
                write_json(root/(tag+"_paired_starts.json"),starts)
                if not starts["matched"]:
                    raise RuntimeError("Fresh-process initial states differ; inspect paired_starts before accepting paired comparisons")
        status.update(state="complete",active=None,active_pid=None,updated_unix=time.time())
    except BaseException as e:
        status.update(state="failed",error=repr(e),updated_unix=time.time())
        write_json(root/"status.json",status)
        raise
    write_json(root/"status.json",status)
    print("CAMPAIGN COMPLETE",root,flush=True)


if __name__=="__main__": main()
