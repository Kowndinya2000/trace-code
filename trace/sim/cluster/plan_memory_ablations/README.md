# Plan-horizon and memory ablations

Each variant is an offline refit on the exact aggregate behind the released DAgger R3 student --
the expert demonstrations plus DAgger rounds 1-3 -- using the same loss, recovery weighting,
optimizer, update count and validation-loss checkpoint selection. Only the listed trainer flag
changes. The evaluation is paired: every case of one evaluation seed reuses the same nominal-plan
job per batch, so variants are compared on identical plans and starts.

| File | Purpose |
|---|---|
| `experiments.py` | The fits (trainer flags), evaluation conditions and stages |
| `prepare.py` | Check inputs, copy the scene manifest, freeze inputs, write `jobs.json` |
| `run_job.py` | Run one idempotent job (`fit:*`, `plan:*`, `case:*`) |
| `run_pool.py` | Dependency-aware scheduler for one multi-GPU node |
| `slurm_submit.sh` | Three SLURM arrays: fits, then plans, then cases |
| `summarize.py` | Paired per-stage tables with scene-bootstrap intervals |

## Run

```bash
export TRACE_PYTHON=/path/to/isaacgym-python        # Python 3.8 with Isaac Gym Preview 4
export TRACE_DATA=$PWD/data TRACE_RUNS=$PWD/runs
python scripts/download_data.py --only collections  # the label sets the refits need

R=trace/sim/cluster/plan_memory_ablations
$TRACE_PYTHON $R/prepare.py  --data $TRACE_DATA --out $TRACE_RUNS/ablations
$TRACE_PYTHON $R/run_pool.py --out $TRACE_RUNS/ablations --gpus 0 --per-gpu 2
$TRACE_PYTHON $R/summarize.py --out $TRACE_RUNS/ablations
```

`--stages` limits the grid; `--use-bundle-fits` evaluates the shipped checkpoints instead of
refitting. A fit is about 75 seconds on an RTX 3090 (plus ~25 s to load the aggregate, ~10 GB
RAM); a plan job about 5 minutes cold; a case about 30 seconds.

Jobs are idempotent: rerunning skips validated outputs and archives incomplete ones under
`failed_attempts/`. `run_job.py` refuses to run if any frozen input -- evaluator, validator,
configs, assets, checkpoints -- changed since `prepare.py`.

Model code the ablations touch: `isaacgymenvs/open_loop/obs_variants.py`,
`isaacgymenvs/tools/train_student_ablation.py`, and small backward-compatible hooks in
`evaluate_retrieval.py`, `validate_retrieval_run.py` and `learning/student_ablate.py`.
