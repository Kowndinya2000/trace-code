"""Freeze prescribed checkpoints, qualify untouched scenes, then evaluate once.

No checkpoint ranking or selection uses qualification/final policy outcomes.
Main comparison uses three perturbation seeds. Occlusion stress uses one draw
and all three independent training pipelines, disclosed separately.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import signal
import subprocess
import time
from collections import Counter
import numpy as np
from run_evaluation_campaign import ROOT,write_json
from teacher_reference import teacher_path

PY=Path(sys.executable)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_qualification(directory):
    directory=Path(directory)
    result=json.loads((directory/'complete.json').read_text())
    assert result['complete'] and result['policy_queries']==0 and result['reset_count_during_hold']==0
    assert sha(directory/'qualification.npz')==result['trace_sha256']
    rules=result['rules']
    with np.load(directory/'qualification.npz') as d:
        s,q,o=d['state'],d['q'],d['oow']
        assert len(s)==rules['hold_decisions']+1 and len(result['rows'])==s.shape[1]
        for i,row in enumerate(result['rows']):
            assert sha(row['scene']['path'])==row['scene']['sha256']
            # Independent acceptance calculation from the saved physical trace.
            xy=float(np.linalg.norm(s[:,i,:,:2]-s[0,i,:,:2],axis=-1).max())
            tilt=np.rad2deg(np.arccos(np.clip(1-2*(s[:,i,:,3:5]**2).sum(-1),-1,1))).max()
            good=(np.isfinite(s[:,i]).all() and np.isfinite(q[:,i]).all() and
                  not o[:,i].any() and (q[:,i]>=0).all() and (q[:,i]<=rules['max_graspability']).all() and
                  xy<=rules['max_xy_motion_m'] and tilt<=rules['max_tilt_deg'] and
                  np.linalg.norm(s[-1,i,:,7:10],axis=-1).max()<=rules['max_final_linear_speed_m_s'] and
                  np.linalg.norm(s[-1,i,:,10:13],axis=-1).max()<=rules['max_final_angular_speed_rad_s'])
            assert bool(good)==row['accepted']
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--after',required=True)
    ap.add_argument('--candidates',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--gpu',type=int,default=4)
    ap.add_argument('--teacher-checkpoint')
    ap.add_argument('--reward-recipe',choices=['legacy','graspability_only'],default=os.environ.get('TRACE_REWARD_RECIPE','legacy'))
    ap.add_argument('--prepare-only',action='store_true',help='Freeze recipes and qualify scenes; leave final rollouts to sharded jobs')
    a=ap.parse_args()
    a.teacher_checkpoint=str(teacher_path(a.teacher_checkpoint))
    os.environ["TRACE_TEACHER_CHECKPOINT"]=a.teacher_checkpoint
    os.environ['TRACE_REWARD_RECIPE']=a.reward_recipe
    root=Path(a.output).resolve()
    if root.exists(): raise FileExistsError(root)
    root.mkdir(parents=True)
    prerequisite=Path(a.after).resolve();learning=prerequisite.parent
    candidates=Path(a.candidates).resolve()
    status=dict(state='waiting_for_prerequisite',pid=os.getpid(),arguments=vars(a),completed=[],started_unix=time.time())
    watched=[Path(a.teacher_checkpoint),ROOT/'tools/teacher_reference.py',Path(__file__).resolve(),ROOT/'tools/qualify_retrieval_scenes.py',ROOT/'tools/run_scene_qualification.sh',
        ROOT/'tools/run_evaluation_campaign.py',ROOT/'tools/evaluate_retrieval.py',ROOT/'tools/run_retrieval_evaluation.sh',
        ROOT/'tools/validate_retrieval_run.py',ROOT/'open_loop/evaluation_core.py',ROOT/'open_loop/student_obs.py',
        ROOT/'tasks/more_evaluation.py',ROOT/'tasks/more_open_loop.py',ROOT/'tasks/more_teacher.py',
        ROOT/'tasks/more_robust.py',ROOT/'tasks/more.py',ROOT/'tasks/base/vec_task.py',
        ROOT/'learning/student_net.py',ROOT/'learning/student_ablate.py',candidates]
    watched += list((ROOT.parent/'cluster').glob('*.py')) + list((ROOT.parent/'cluster').glob('*.sh'))
    frozen={str(p):sha(p) for p in watched};status['source_sha256']=frozen
    def save(): status['updated_unix']=time.time();write_json(root/'status.json',status)
    save()
    def run(name,cmd,timeout=172800):
        if any(sha(p)!=h for p,h in frozen.items()): raise RuntimeError('Source changed while campaign queued/running')
        status.update(state='running',active=name);save()
        with (root/(name+'.log')).open('w') as log:
            child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            status['active_pid']=child.pid;save()
            try: rc=child.wait(timeout=timeout)
            except BaseException:
                os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=30);raise
        status['active_pid']=None
        if rc: raise RuntimeError(f'{name} failed: exit {rc}; see {root/name}.log')
        status['completed'].append(name);save()
    try:
        while True:
            if prerequisite.exists():
                state=json.loads(prerequisite.read_text()).get('state')
                if state=='failed': raise RuntimeError('Corrected learning prerequisite failed')
                if state=='complete': break
            time.sleep(30)
        from isaacgymenvs.open_loop.evaluation_core import read_manifest
        raw=read_manifest(candidates)
        if raw['role']!='independent_final_candidates': raise ValueError('Unexpected candidate role')
        # Fixed recipe: final scheduled optimizer update, full round-3 aggregate;
        # include matched-label control separately. No best-seed/checkpoint search.
        cases=[dict(name=x,actor=x) for x in ['teacher','replay','hold']]
        stress=[];checkpoint_hashes={}
        seeds=json.loads(prerequisite.read_text())['arguments']['fit_seeds']
        for seed in seeds:
            recipes=dict(bc=f'fit{seed}_round0_student',student=f'fit{seed}_round3_student',
                         bc_matched=f'fit{seed}_expert_matched',dagger_matched=f'fit{seed}_dagger_matched')
            for label,folder in recipes.items():
                path=learning/folder/'student.pt'
                done=json.loads((path.parent/'complete.json').read_text())
                assert done['complete'] and done['checkpoint_sha256']==sha(path)
                checkpoint_hashes[str(path)]=sha(path)
                cases.append(dict(name=f'{label}_s{seed}',actor='student',checkpoint=str(path)))
                if label in ('bc','student'):
                    for tag,p in [('d90',.9),('d100',1.)]:
                        stress.append(dict(name=f'{label}_s{seed}_{tag}',actor='student',checkpoint=str(path),p_drop=p))
        write_json(root/'frozen_recipes.json',dict(frozen_unix=time.time(),teacher_checkpoint=a.teacher_checkpoint,
            teacher_sha256=sha(a.teacher_checkpoint),reward_recipe=os.environ.get('TRACE_REWARD_RECIPE','legacy'),checkpoint_sha256=checkpoint_hashes,
            main_cases=cases,stress_cases=stress,main_perturbation_seeds=[7,19,37],stress_perturbation_seeds=[7],
            candidate_manifest_sha256=sha(candidates),selection='Prespecified final update of all independent fits; no outcome selection'))
        qualified=[]
        # Small qualification smoke precedes the full candidate pool. It never
        # evaluates a retrieval policy and does not change qualification rules.
        for offset,size,name in [(0,6,'qualification_smoke')]+[
                (i,min(64,len(raw['scenes'])-i),f'qualification_{i:04d}') for i in range(0,len(raw['scenes']),64)]:
            directory=root/name;spec=root/(name+'.spec.json')
            write_json(spec,dict(manifest=str(candidates),offset=offset,batch_size=size,output=str(directory)))
            run(name,['bash',str(ROOT/'tools/run_scene_qualification.sh'),str(spec),str(a.gpu)],1800)
            report=validate_qualification(directory)
            if name!='qualification_smoke': qualified+=report['rows']
        groups={}
        for row in qualified:
            if row['accepted']: groups.setdefault(row['scene']['tier'],[]).append(row['scene'])
        tiers=sorted(set(r['tier'] for r in raw['scenes']))
        per_tier=min(len(groups.get(t,[])) for t in tiers)
        write_json(root/'qualification_summary.json',dict(total_candidates=len(qualified),
            accepted_by_tier={t:len(groups.get(t,[])) for t in tiers},balanced_per_tier=per_tier,
            rejection_reasons=dict(Counter(x for r in qualified for x in r['reasons'])),
            rules=report['rules'],policy_queries=0))
        if per_tier<100: raise RuntimeError('Fewer than 100 physically qualified scenes per tier; preserve results and extend generation without consulting final policy outcomes')
        final_rows=[groups[t][i] for i in range(per_tier) for t in tiers]
        manifest=root/'final_manifest.json'
        write_json(manifest,dict(role='independent_final',scenes=final_rows,
            source_candidate_manifest=str(candidates),qualification_summary=str(root/'qualification_summary.json'),
            selection='First accepted candidates in the pre-shuffled manifest, balanced by tier; original files unchanged'))
        read_manifest(manifest)
        write_json(root/'main.cases.json',cases)
        write_json(root/'occlusion.cases.json',stress)
        if a.prepare_only:
            status.update(state='ready_for_evaluation',active=None);save()
            print('FINAL SUITE QUALIFIED AND RECIPES FROZEN',root,flush=True)
            return
        for name,selected,draws in [('main',cases,[7,19,37]),('occlusion',stress,[7])]:
            if any(sha(p)!=h for p,h in checkpoint_hashes.items()): raise RuntimeError('Frozen checkpoint changed')
            casefile=root/(name+'.cases.json');write_json(casefile,selected)
            directory=root/name
            run(name,[str(PY),'-u',str(ROOT/'tools/run_evaluation_campaign.py'),
                '--manifest',str(manifest),'--output',str(directory),'--cases',str(casefile),
                '--count',str(len(final_rows)),'--batch-size','64','--seeds',*[str(s) for s in draws],
                '--gpu',str(a.gpu)])
            if json.loads((directory/'status.json').read_text())['state']!='complete': raise RuntimeError('Incomplete final evaluation')
        status.update(state='complete',active=None);save()
        print('INDEPENDENT FINAL CAMPAIGN COMPLETE',root,flush=True)
    except BaseException as e:
        status.update(state='failed',error=repr(e));save();raise


if __name__=='__main__': main()
