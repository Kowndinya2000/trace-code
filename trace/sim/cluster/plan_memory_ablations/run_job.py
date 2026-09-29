"""Execute one job from jobs.json. Idempotent and safe to requeue.

    python cluster/plan_memory_ablations/run_job.py JOB_ID --out RUN_DIR [--gpu 0]

A completed, validated output is skipped. An incomplete output (preemption,
crash) is archived under RUN_DIR/failed_attempts/ and rerun from scratch; it
is never scored. CUDA out-of-memory is retried up to three times.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (PKG, REPO, check_frozen, complete_eval, complete_fit, layout_from, python_executable,  # noqa: E402
                    read_json, write_json)


def archive(lay, path, reason):
    path = Path(path)
    if not path.exists():
        return
    target = lay.out / 'failed_attempts' / f'{path.name}_{time.strftime("%Y%m%dT%H%M%S")}_{reason}'
    target.parent.mkdir(parents=True, exist_ok=True)
    path.rename(target)


def environment(gpu):
    env = dict(os.environ, TRACE_PYTHON=python_executable(), CUBLAS_WORKSPACE_CONFIG=':4096:8')
    env['PYTHONPATH'] = f'{PKG}:{PKG / "tools"}:{REPO / "cluster"}:{REPO}:' + env.get('PYTHONPATH', '')
    return env


def run_fit(lay, job, gpu, log):
    out = Path(job['output'])
    if complete_fit(out):
        return 'skipped'
    archive(lay, out, 'incomplete')
    command = [python_executable(), '-u', str(PKG / 'tools/train_student_ablation.py'),
               '--data', *map(str, lay.collections), '--manifest', str(lay.training_manifest),
               '--output', str(out), '--device', f'cuda:{gpu}', *job['args']]
    code = subprocess.call(command, cwd=PKG, stdout=log, stderr=subprocess.STDOUT, env=environment(gpu))
    if code or not complete_fit(out):
        raise RuntimeError(f'fit {job["model"]} exited {code}')
    return 'trained'


def run_eval(lay, job_id, job, gpu, log):
    from validate_retrieval_run import validate
    out = Path(job['output'])
    if complete_eval(out):
        return 'skipped'
    spec_path = lay.out / 'specs' / (out.name + '.spec.json')
    write_json(spec_path, job['spec'])
    for attempt in range(4):
        archive(lay, out, 'incomplete')
        with open(lay.logs / (job_id.replace(':', '__') + '.eval.log'), 'w') as eval_log:
            code = subprocess.call(['bash', str(PKG / 'tools/run_retrieval_evaluation.sh'), str(spec_path), str(gpu)],
                                   cwd=PKG, stdout=eval_log, stderr=subprocess.STDOUT, env=environment(gpu))
        text = Path(eval_log.name).read_text(errors='replace').lower()
        if code == 0 and (out / 'complete.json').exists():
            report = validate(out)
            if not report.get('valid'):
                raise RuntimeError(f'{job_id} failed validation')
            if job['kind'] == 'case' and not report.get('matched_initial_states', True):
                raise RuntimeError(f'{job_id} initial states differ')
            return 'evaluated'
        if 'out of memory' not in text or attempt == 3:
            raise RuntimeError(f'{job_id} exited {code}; see {eval_log.name}')
        log.write(f'CUDA OOM on attempt {attempt + 1}; retrying\n')
        log.flush()
        time.sleep(30)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('job_id')
    ap.add_argument('--data')
    ap.add_argument('--out')
    ap.add_argument('--gpu', type=int, default=int(os.environ.get('TRACE_GPU', 0)))
    args = ap.parse_args()
    if not args.data and not os.environ.get('TRACE_DATA') and args.out:
        args.data = read_json(Path(args.out) / 'run_config.json')['data']
    lay = layout_from(args)
    jobs = read_json(lay.jobs)
    job = jobs[args.job_id]
    lay.logs.mkdir(parents=True, exist_ok=True)
    check_frozen(lay)
    for dep in job['deps']:
        dep_job = jobs[dep]
        ready = complete_fit(dep_job['output']) if dep_job['kind'] == 'fit' else complete_eval(dep_job['output'])
        if not ready:
            raise SystemExit(f'Dependency {dep} is not complete')
    started = time.time()
    with open(lay.logs / (args.job_id.replace(':', '__') + '.log'), 'a') as log:
        log.write(f'START {args.job_id} gpu={args.gpu} host={os.uname().nodename} {time.ctime()}\n')
        log.flush()
        result = (run_fit(lay, job, args.gpu, log) if job['kind'] == 'fit'
                  else run_eval(lay, args.job_id, job, args.gpu, log))
        check_frozen(lay)
        log.write(f'DONE {args.job_id} {result} {time.time() - started:.1f}s\n')
    print(args.job_id, result, f'{time.time() - started:.1f}s', flush=True)


if __name__ == '__main__':
    main()
