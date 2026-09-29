# PMBS 511-scene simulation evaluation

This directory is an isolated copy of the parallel-MCTS PMBS implementation
from `the PMBS release parallel_mcts`.  The source folder is
treated as read-only.  `evaluate_manifest.py` adds resumable evaluation over
the exact ordered 511-scene teacher development manifest, per-scene JSON
records, and continuously updated aggregate statistics.

The copied baseline uses its original 10 cm push, grasp networks, simulated
grasp-and-lift success check, parallel Isaac Gym MCTS, and 15-action cap.  The
default 5 second search budget and 500 environments match the active commands
in the upstream `mcts_parallel_run.sh`.

Run from this directory in the `pmbs` environment:

```bash
python evaluate_manifest.py \
  --output_dir "$TRACE_RUNS/baseline_mcts_parallel" \
  --num_envs 500 --time_limit 5 --max_actions 15 --headless
```

Results are written under `$TRACE_RUNS`. `scripts/reproduce_sim.py baselines` runs this
command and the serial one below with the reported settings, then verifies both.

Use `--limit 1` for a smoke test.  Re-running the same command resumes from
completed `scenes/NNNN.json` files; protocol changes require a new output
directory.

## Serial-MCTS baseline

`--search serial` swaps the tree search for `mcts_serial/`, an isolated copy of
the upstream `parallel_mcts/mcts` (the single-simulation-environment MCTS that the
parallel search is measured against).  Everything else is held fixed: the same
manifest, grasp networks, action sampler, 10 cm push primitive, 5 s search budget
per push and 15-action cap.  `mcts_serial` deviates from the source in two ways
only: the intra-package imports are renamed, and `best_action` returns `None` for
an unexpanded root instead of indexing an empty child list, which the harness
would otherwise record as an infrastructure error and retry forever.

Serial needs `--num_envs 2`: environment 0 is the real scene and environment 1 is
the single simulation environment the search steps through.  Because the search
budget is wall clock, serial and parallel are compared at equal real time per
push, so the cost of serial search shows up as more pushes and more planning time
per scene, not as a longer budget.  Run it with `--search serial --num_envs 2`, writing to
`$TRACE_RUNS/baseline_mcts_serial`.

The two campaigns must never share a GPU, because wall-clock timing is part of the
result: run the parallel campaign to completion first, then the serial one.
`scripts/reproduce_sim.py baselines` does exactly that.

The runner is resumable. It restarts after a process or GPU failure and continues at
the first uncommitted scene. An exception is written under `failures/` and deliberately
not committed as a scene result, so an infrastructure failure cannot silently become an
evaluation failure. Pass `--max_scenes_per_process 4` to recycle the process every four
scenes, which releases the GPU memory that repeated Isaac Gym simulator creation
retains.
After scene 511, `verify_results.py` independently verifies manifest and scene
hashes, indices, action/trace counts, success verdicts, timing fields, and every
aggregate before writing `results.csv` and `verification.json`.  A failed final
verification is an error, not a result: fix the cause and re-run.
