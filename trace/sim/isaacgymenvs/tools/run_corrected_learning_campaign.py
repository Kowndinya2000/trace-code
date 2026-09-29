"""Gated recollection, independent DAgger pipelines, and matched controls.

Uses one GPU sequentially. Every simulation subprocess is trace-validated by
run_evaluation_campaign. A failed prerequisite stops this queue; it never
substitutes legacy data or checkpoints for a missing corrected result.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import numpy as np
from run_evaluation_campaign import write_json
from teacher_reference import teacher_path

ROOT=Path(__file__).resolve().parents[1]
PY=Path(sys.executable)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--teacher-checkpoint')
    ap.add_argument('--reward-recipe',choices=['legacy','graspability_only'],default=os.environ.get('TRACE_REWARD_RECIPE','legacy'))
    ap.add_argument('--output',required=True);ap.add_argument('--after',required=True)
    ap.add_argument('--train-manifest',required=True);ap.add_argument('--development-manifest',required=True)
    ap.add_argument('--count',type=int,default=512);ap.add_argument('--eval-count',type=int,default=128)
    ap.add_argument('--updates',type=int,default=10000);ap.add_argument('--gpu',type=int,default=4)
    ap.add_argument('--round-updates',type=int,nargs=4,help='Prespecified BC/r1/r2/r3 updates; matched and representation fits use r3, head comparison uses BC')
    ap.add_argument('--fit-seeds',type=int,nargs='+',default=[0,1,2])
    a=ap.parse_args();root=Path(a.output).resolve()
    a.teacher_checkpoint=str(teacher_path(a.teacher_checkpoint))
    if a.round_updates and (min(a.round_updates)<=0 or a.updates!=a.round_updates[-1]):
        raise ValueError('--updates must equal the final round budget')
    os.environ["TRACE_TEACHER_CHECKPOINT"]=a.teacher_checkpoint
    os.environ['TRACE_REWARD_RECIPE']=a.reward_recipe
    if root.exists(): raise FileExistsError('Use a fresh campaign directory: '+str(root))
    root.mkdir(parents=True)
    watched=[Path(a.teacher_checkpoint),ROOT/'tools/teacher_reference.py',Path(__file__).resolve(),ROOT/'tools/train_student_repaired.py',
        ROOT/'tools/run_evaluation_campaign.py',ROOT/'tools/evaluate_retrieval.py',
        ROOT/'tools/run_retrieval_evaluation.sh',ROOT/'tools/validate_retrieval_run.py',
        ROOT/'open_loop/evaluation_core.py',ROOT/'open_loop/student_obs.py',
        ROOT/'learning/student_net.py',ROOT/'learning/student_ablate.py',
        ROOT/'tasks/more_evaluation.py',ROOT/'tasks/more_open_loop.py',
        ROOT/'tasks/more_teacher.py',ROOT/'tasks/more.py',
        Path(a.train_manifest).resolve(),Path(a.development_manifest).resolve()]
    def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
    watched += list((ROOT.parent/'cluster').glob('*.py')) + list((ROOT.parent/'cluster').glob('*.sh'))
    frozen={str(p):digest(p) for p in watched}
    status=dict(state='waiting_for_prerequisite',pid=os.getpid(),arguments=vars(a),completed=[],
                started_unix=time.time(),source_sha256=frozen)
    def save(): status['updated_unix']=time.time();write_json(root/'status.json',status)
    save()
    def run(name,cmd,completion):
        if any(digest(Path(p))!=h for p,h in frozen.items()):
            raise RuntimeError('Campaign source or manifest changed after launch; inspect before starting a new campaign')
        if Path(completion).exists(): raise FileExistsError(completion)
        status.update(state='running',active=name);save()
        with (root/(name+'.log')).open('w') as log:
            child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            status['active_pid']=child.pid;save()
            try: rc=child.wait(timeout=43200)
            except BaseException:
                os.killpg(child.pid,signal.SIGTERM)
                child.wait(timeout=30)
                raise
        status['active_pid']=None
        if rc: raise RuntimeError(f'{name} failed ({rc}); see {root/name}.log')
        if not Path(completion).exists(): raise RuntimeError(f'{name} produced no completion marker')
        payload=json.loads(Path(completion).read_text())
        if not (payload.get('complete') is True or payload.get('state')=='complete'):
            raise RuntimeError(f'{name} did not complete: {completion}')
        status['completed'].append(name);save()
    def collection(seed,round_index,previous):
        name=f'fit{seed}_round{round_index}_collection'
        directory=root/name
        cases=[dict(name='expert',actor='teacher',collect=True)]
        if previous is not None:
            cases.append(dict(name='dagger',actor='student',checkpoint=str(previous),collect=True))
        casefile=root/(name+'.cases.json');write_json(casefile,cases)
        cmd=[str(PY),'-u',str(ROOT/'tools/run_evaluation_campaign.py'),
            '--manifest',a.train_manifest,'--output',str(directory),'--cases',str(casefile),
            '--count',str(a.count),'--batch-size','64','--seeds',str(1000+100*seed+round_index),
            '--gpu',str(a.gpu)]
        run(name,cmd,directory/'status.json')
        expert=sorted(p for p in directory.glob('s*_b*_expert') if p.is_dir())
        dagger=sorted(p for p in directory.glob('s*_b*_dagger') if p.is_dir())
        if not expert or (previous is not None and not dagger): raise RuntimeError('Missing collected data')
        return expert,dagger
    def fit(name,data,seed,extra=(),updates=None):
        directory=root/name
        cmd=[str(PY),'-u',str(ROOT/'tools/train_student_repaired.py'),'--data',*[str(p) for p in data],
            '--manifest',a.train_manifest,'--output',str(directory),'--seed',str(seed),
            '--updates',str(updates or a.updates),'--device',f'cuda:{a.gpu}',*extra]
        run(name,cmd,directory/'complete.json')
        return directory
    def evaluate(name,cases):
        directory=root/name;casefile=root/(name+'.cases.json');write_json(casefile,cases)
        cmd=[str(PY),'-u',str(ROOT/'tools/run_evaluation_campaign.py'),
            '--manifest',a.development_manifest,'--output',str(directory),'--cases',str(casefile),
            '--count',str(a.eval_count),'--batch-size','64','--seeds','7','--gpu',str(a.gpu)]
        run(name,cmd,directory/'status.json')
    try:
        while True:
            prerequisite=Path(a.after)
            if prerequisite.exists():
                state=json.loads(prerequisite.read_text()).get('state')
                if state=='failed': raise RuntimeError(f'Prerequisite failed: {prerequisite}')
                if state=='complete': break
            time.sleep(30)
        # Import only once execution is ready; this uses no GPU by itself.
        from train_student_repaired import load_data,grouped_split
        def labels(data):
            d,_=load_data(data,a.train_manifest)
            ti,_=grouped_split(d['scene_hash'])
            return int(d['valid'][:,ti].sum())
        for seed in a.fit_seeds:
            expert_data=[];dagger_data=[];previous=None;bc=None
            for round_index in range(4):
                expert,dagger=collection(seed,round_index,previous)
                expert_data+=expert
                dagger_data+=expert if round_index==0 else dagger
                previous=fit(f'fit{seed}_round{round_index}_student',dagger_data,seed,
                    updates=a.round_updates[round_index] if a.round_updates else a.updates)
                if round_index==0: bc=previous
                evaluate(f'fit{seed}_round{round_index}_development',[
                    dict(name='student',actor='student',checkpoint=str(previous))])
            # Same valid-label budget and optimizer-update count; both datasets
            # use the same initial scenes and perturbation seeds each round.
            budget=min(labels(expert_data),labels(dagger_data))
            matched_bc=fit(f'fit{seed}_expert_matched',expert_data,seed,['--label-budget',str(budget)])
            matched_dagger=fit(f'fit{seed}_dagger_matched',dagger_data,seed,['--label-budget',str(budget)])
            cases=[dict(name='bc_initial',actor='student',checkpoint=str(bc)),
                dict(name='bc_matched',actor='student',checkpoint=str(matched_bc)),
                dict(name='dagger_matched',actor='student',checkpoint=str(matched_dagger)),
                dict(name='dagger_full',actor='student',checkpoint=str(previous))]
            for label,checkpoint in [('bc',bc),('dagger',previous)]:
                for tag,p in [('d90',.9),('d100',1.)]:
                    cases.append(dict(name=label+'_'+tag,actor='student',checkpoint=str(checkpoint),p_drop=p))
            evaluate(f'fit{seed}_matched_and_occlusion',cases)
            # Primary representation tests are repeated across independent
            # collection pipelines; all use the selected zero-value-loss recipe.
            ablation_cases=[]
            for ablation in ['no_plan','no_vis','no_gru']:
                checkpoint=fit(f'fit{seed}_{ablation}',dagger_data,seed,['--ablate',ablation])
                for suffix,p in [('default',.15),('d90',.9)]:
                    ablation_cases.append(dict(name=ablation+'_'+suffix,actor='student',
                        checkpoint=str(checkpoint),ablation=ablation,p_drop=p))
            evaluate(f'fit{seed}_representation',ablation_cases)
            # Head comparison shares expert data, value coefficient (zero),
            # minibatch size, and optimizer update budget with initial BC.
            initial_data=sorted(p for p in (root/f'fit{seed}_round0_collection').glob('s*_b*_expert') if p.is_dir())
            categorical=fit(f'fit{seed}_categorical_kl',initial_data,seed,['--head','categorical_kl'],
                updates=a.round_updates[0] if a.round_updates else a.updates)
            evaluate(f'fit{seed}_action_head',[
                dict(name='xy',actor='student',checkpoint=str(bc)),
                dict(name='categorical_kl',actor='student',checkpoint=str(categorical))])
        status.update(state='complete',active=None);save()
        print('CORRECTED LEARNING CAMPAIGN COMPLETE',root,flush=True)
    except BaseException as e:
        status.update(state='failed',error=repr(e));save();raise


if __name__=='__main__': main()
