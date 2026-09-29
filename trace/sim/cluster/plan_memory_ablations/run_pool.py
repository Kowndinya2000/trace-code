"""Dependency-aware job pool for one multi-GPU machine (no scheduler needed).

    python cluster/plan_memory_ablations/run_pool.py --out RUN_DIR --gpus 0 1 2 3 --per-gpu 2

Fits and plan batches start immediately; each case starts once its fit and its
plan batch are complete. Failed jobs are reported and their dependents skipped;
rerunning the pool resumes (completed jobs are skipped by run_job).
Use --per-gpu 1 on GPUs with less than ~16 GB. Each evaluation needs ~6 GB GPU
memory and ~4 CPU threads; each fit ~4-8 GB GPU memory and ~10 GB RAM.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import complete_eval, complete_fit, python_executable, read_json  # noqa: E402

HERE = Path(__file__).resolve().parent


def done(job):
    return complete_fit(job['output']) if job['kind'] == 'fit' else complete_eval(job['output'])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--gpus', nargs='+', type=int, required=True)
    ap.add_argument('--per-gpu', type=int, default=2)
    ap.add_argument('--kinds', nargs='*', default=['fit', 'plan', 'case'])
    args = ap.parse_args()
    jobs = read_json(args.out / 'jobs.json')
    selected = {k: v for k, v in jobs.items() if v['kind'] in args.kinds}
    finished = {k for k, v in jobs.items() if done(v)}
    failed, running = set(), {}
    slots = {g: 0 for g in args.gpus}
    (args.out / 'logs').mkdir(exist_ok=True)
    order = sorted(selected, key=lambda k: ({'fit': 0, 'plan': 1, 'case': 2}[selected[k]['kind']], k))
    print(f'{len(selected)} jobs selected, {len(finished & set(selected))} already complete', flush=True)
    while True:
        for job_id, (process, gpu, started) in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            slots[gpu] -= 1
            del running[job_id]
            if code == 0:
                finished.add(job_id)
            else:
                failed.add(job_id)
                print(f'FAILED {job_id} (exit {code}); see {args.out}/logs', flush=True)
        blocked = {k for k in order if any(d in failed for d in jobs[k]['deps'])}
        pending = [k for k in order if k not in finished and k not in failed and k not in running and k not in blocked]
        for job_id in pending:
            if not all(d in finished for d in jobs[job_id]['deps']):
                continue
            free = [g for g in args.gpus if slots[g] < args.per_gpu]
            if not free:
                break
            gpu = min(free, key=lambda g: slots[g])
            command = [python_executable(), str(HERE / 'run_job.py'), job_id, '--out', str(args.out), '--gpu', str(gpu)]
            stream = open(args.out / 'logs' / (job_id.replace(':', '__') + '.pool.log'), 'a')
            running[job_id] = (subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT), gpu, time.time())
            slots[gpu] += 1
        remaining = [k for k in order if k not in finished and k not in failed and k not in blocked]
        if not running and not [k for k in remaining if all(d in finished for d in jobs[k]['deps'])]:
            break
        complete = len(finished & set(selected))
        print(f'{time.strftime("%H:%M:%S")} complete {complete}/{len(selected)} running {len(running)} '
              f'failed {len(failed)} blocked {len(blocked)}', flush=True)
        time.sleep(15)
    unfinished = [k for k in order if k not in finished]
    print(f'POOL DONE complete {len(finished & set(selected))}/{len(selected)} failed {len(failed)} '
          f'not run {len(unfinished) - len(failed)}', flush=True)
    sys.exit(1 if unfinished else 0)


if __name__ == '__main__':
    main()
