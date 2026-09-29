"""Record disjoint policy/occlusion cases on a bounded set of local GPUs."""
from pathlib import Path
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import queue
import subprocess
import sys
import threading
import time
import shutil
import numpy as np
from validate_retrieval_run import validate, digest

TOOLS = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def compare_control(root):
    paths = [root/'runs/bc_default', root/'runs/bc_default_control']
    for path in paths:
        validate(path)
    differences={}
    with np.load(paths[0]/'bc_default.npz') as a, np.load(paths[1]/'bc_default.npz') as b:
        for key in a.files:
            if a[key].dtype.kind in 'bui':
                np.testing.assert_array_equal(a[key], b[key], err_msg=key)
                differences[key]=0.
            else:
                np.testing.assert_allclose(a[key], b[key], atol=1e-5, rtol=0, err_msg=key)
                differences[key]=float(np.max(np.abs(a[key]-b[key])))
    result=dict(valid=True, simulator_batch_size=64, recorded_envs=[0,1,2],
                checked='All recorded control arrays, including actions, object trajectories, observations, visibility and outcomes',
                max_absolute_differences=differences)
    (root/'recording_control_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path)
    parser.add_argument('--gpus',type=int,nargs='+',required=True)
    args=parser.parse_args();root=args.root.resolve()
    compare_control(root)
    jobs=queue.Queue();spec=read(root/'gallery_spec.json')
    for entry in spec['entries']:jobs.put(entry)
    lock=threading.Lock();status=dict(state='recording',started_unix=time.time(),active={},completed=[])
    frozen={str(p):digest(p) for p in [TOOLS/'record_student_eval.py',TOOLS/'evaluate_retrieval.py',
        root/'gallery_spec.json',root/'manifest.json',root/'fixed_plan/nominal.json']}
    sources=root/'recorder_sources';sources.mkdir(exist_ok=True)
    current_hash=frozen[str(TOOLS/'record_student_eval.py')]
    shutil.copyfile(TOOLS/'record_student_eval.py',sources/(current_hash+'.py'))

    def save():
        status['updated_unix']=time.time()
        temp=root/'video_status.json.tmp';temp.write_text(json.dumps(status,indent=2)+'\n')
        temp.replace(root/'video_status.json')

    def worker(gpu):
        while True:
            try:entry=jobs.get_nowait()
            except queue.Empty:return
            name=entry['name'];directory=root/'runs'/name
            if any(digest(p)!=h for p,h in frozen.items()):
                raise ValueError('Recorder inputs changed during the suite')
            with lock:status['active'][str(gpu)]=name;save()
            if not (directory/'complete.json').exists():
                if directory.exists():raise ValueError('Inspect incomplete recording: '+str(directory))
                case_spec=read(entry['spec'])
                command=[sys.executable,'-u',str(TOOLS/'record_student_eval.py'),
                    'task=MoreEvaluation','train=MoreOpenLoopSetSCPPO','test=True','headless=True',
                    'force_render=False','wandb_activate=False',f'sim_device=cuda:{gpu}',
                    f'rl_device=cuda:{gpu}',f'graphics_device_id={gpu}',
                    'checkpoint='+case_spec['teacher_checkpoint'],
                    *['task.env.teacher.'+k+'=0' for k in ['lambdaC','lambdaC3','lambdaArc','lambdaB','lambdaEef','lambdaDisturb']],
                    '+evaluation_spec='+entry['spec']]
                with (root/(name+'.log')).open('w') as log:
                    subprocess.run(command,cwd=TOOLS.parent,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800)
            validate(directory)
            recording=read(directory/'video_raw/recording.json')
            recorder_source=sources/(recording['recorder_sha256']+'.py')
            if not recorder_source.exists() or digest(recorder_source)!=recording['recorder_sha256']:
                raise ValueError('Missing exact source for a recorded clip')
            if recording['simulator_batch_size']!=64:
                raise ValueError('Changed simulator batch')
            with lock:
                status['completed'].append(name);status['active'].pop(str(gpu),None);save()
            print('RECORDED',name,flush=True)

    try:
        save()
        with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
            futures=[pool.submit(worker,gpu) for gpu in args.gpus]
            for future in futures:future.result()
        reference=None;differences={}
        for entry in spec['entries']:
            with np.load(root/'runs'/entry['name']/(entry['name']+'.npz')) as data:
                current={k:data[k].copy() for k in ['initial_state','initial_eef','initial_q','requested_initial_state']}
            if reference is None:reference=current
            differences[entry['name']]={k:float(np.max(np.abs(current[k]-reference[k]))) for k in current}
            if max(differences[entry['name']].values())>1e-5:
                raise ValueError('Recorded actors have different initial states')
        (root/'matched_starts.json').write_text(json.dumps(dict(valid=True,differences=differences),indent=2)+'\n')
        with lock:status['state']='encoding';save()
        subprocess.run([sys.executable,str(TOOLS/'make_student_video_gallery.py'),str(root),'--montages'],check=True)
        with lock:status['state']='complete';save()
    except BaseException as error:
        with lock:status.update(state='failed',error=repr(error));save()
        raise


if __name__=='__main__':main()
