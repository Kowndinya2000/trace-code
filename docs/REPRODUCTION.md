# Reproducing the reported numbers

Every table is evaluated on the same 511 scenes. Each episode starts from the same perturbed
state for every method -- the perturbation is drawn per evaluation seed, not per method -- so
comparisons within a table are paired. Success means the grasp-quality network clears 0.9 on the
target in the simulator. Exhausting the step or travel budget is a failure, as is pushing any
object out of the workspace.

Learned policies are reported as the mean over three training seeds (0, 1, 2) at evaluation
seed 7. The scene manifest is `data/scenes/development.json`, named for the role it played during
model selection; it is the 511-scene set the paper reports.

The scene payload contains `training_fit.json` (1,783 scenes), `validation.json` (314 scenes),
and `test.json` (the same 511 scenes as `development.json`, not an independent blind holdout).
The fit/validation partition is grouped by scene hash with split seed 20260907.
`training.json` retains their combined 2,097-scene pool: pass this combined manifest to
the trainers, which reconstruct the same validation partition internally.

## Simulation comparison

```bash
python scripts/reproduce_sim.py main --gpu 0        # learned policies and the privileged teacher
python scripts/reproduce_sim.py baselines --gpu 0   # PMBS, serial MCTS, spiral, straight line
```

| Method | Online state | Success (%) | 95% CI | OOW (%) | Budget (%) |
|---|---|---:|---|---:|---:|
| Teacher Replay | none | 43.4 | [39.1, 48.0] | 6.3 | 50.3 |
| TRACE-BC | partial | 66.7 | [63.7, 69.8] | 15.3 | 18.0 |
| **TRACE** | partial | **90.7** | [88.9, 92.6] | 7.2 | 2.1 |
| PMBS | complete | 88.1 | [85.1, 90.8] | — | 11.9 |
| Serial MCTS | complete | 44.0 | [40.1, 48.1] | — | 56.0 |
| Spiral | target position | 80.2 | [76.9, 83.8] | 19.6 | 0.2 |
| Straight line | target position | 0.4 | [0.0, 1.0] | 0.0 | 99.6 |
| Online Teacher (privileged reference) | complete | 96.7 | [95.1, 98.0] | 3.3 | 0.0 |

Intervals are 95% stratified scene-bootstrap intervals over 2,000 resamples. The three
nominal-rollout rows share scene identities, perturbations and the teacher-relative budget. The
planning and heuristic baselines use their own action and termination protocols on the same scene
identities, so they are not strictly paired; their unsuccessful episodes are reported under
Budget, and OOW is not defined for the MCTS baselines, which reject out-of-workspace pushes when
sampling.

## Student-state supervision

```bash
python scripts/reproduce_sim.py dagger --gpu 0
```

Rounds are cumulative: round *k* trains from scratch on the expert demonstrations plus rounds
1..*k*. The two matched-budget blocks separate the effect of *where* labels are collected from the
effect of *how many* there are.

| Training data | Labels | Success (%) | Δ |
|---|---:|---:|---|
| Expert BC | 26,373 | 66.7 | — |
| DAgger R1 | 57,125 | 87.1 | +20.4 [17.5, 23.2] |
| DAgger R2 | 84,429 | 89.0 | +2.0 [−0.1, 4.0] |
| **TRACE (DAgger R3)** | 111,555 | **90.7** | +1.7 [−0.2, 3.5] |
| Student-state BC, matched labels | 26,373 | 87.8 | +21.1 [18.3, 23.9] |
| Expert BC, full data | 111,555 | 78.8 | — |

Δ is against the preceding round inside the aggregation block, and against Expert BC in the
matched-label block. The last row is Expert BC given the same 111,555-label budget as R3, which
reaches 78.8% against R3's 90.7% (+11.9 [9.5, 14.4]).

## Plan horizon and recurrent memory

```bash
python scripts/reproduce_sim.py ablations --gpu 0
```

Each ablated variant is refit on the fixed R3 aggregate with the same loss, optimizer, update
count and validation-based checkpoint selection; only the listed input or architecture changes.
The TRACE (K = 4) row uses the deployed checkpoints at the same three training seeds as the main
table. Each campaign regenerates its nominal plans, so detailed outcomes can vary between
campaigns. Both Δ and its interval use this deployed reference within the ablation campaign,
paired on per-scene means over a row's seeds. Steps is the mean over successful episodes.
Differences are calculated before rounding the displayed percentages.

| Variant | Success (%) | Δ vs TRACE (pp, 95% CI) | OOW (%) | Budget (%) | Steps |
|---|---:|---|---:|---:|---:|
| **TRACE (K = 4)** | **90.7** | — | 7.2 | 2.0 | 14.7 |
| w/o GRU | 85.8 | −5.0 [−6.8, −3.0] | 8.0 | 6.2 | 15.0 |
| K = 1 | 88.5 | −2.3 [−4.0, −0.5] | 8.4 | 3.1 | 14.6 |
| K = 2 | 89.9 | −0.8 [−2.7, +0.9] | 7.8 | 2.2 | 14.6 |
| K = 8 | 89.3 | −1.4 [−3.2, +0.3] | 7.7 | 3.0 | 14.7 |
| Full plan | 90.0 | −0.7 [−2.5, +1.1] | 7.3 | 2.6 | 14.4 |

The table consistently uses the deployed TRACE reference. The K = 1 interval excludes zero,
while the K = 2, K = 8 and full-plan intervals include zero.
Budget counts step/travel timeouts only; K = 2 and full plan each also have one target-out-of-view
failure (0.07%), recorded separately in `summaries/reported_table.json`.

A plan-free controller is not included: removing the nominal rollout also removes the
scene-specific execution horizon, which would change the termination policy rather than isolate
the plan as an input.

## Retraining instead of downloading

Every student in the tables can be refit from the shipped label sets. One fit is 10,000 updates
and takes about 75 seconds on an RTX 3090, plus roughly 25 seconds to load the aggregate.

```bash
python scripts/download_data.py --only collections     # the expert and DAgger labels
python scripts/reproduce_sim.py ablations --gpu 0 --refit
```

`--refit` retrains the fifteen ablation checkpoints instead of using the shipped ones; the
reference row is the deployed student and is always used as released.

`collections/collect_r0` is the expert demonstration set and is shared by every row. The
remaining directories are what distinguishes one checkpoint from another.

**The order of `--data` is part of the recipe.** The trainer aggregates shards in the order the
collections are given, and the batch stream — hence the weights — depends on it. The orders below
are the ones the shipped checkpoints were trained with; reproducing them exactly needs the same
order, and a different order trains an equally valid but different student (for round 3 it moved
success from 91.4% to 90.2%).

| Checkpoint | `--data`, in order | Extra flag |
|---|---|---|
| Expert BC, any seed | `collect_r0` | — |
| DAgger R*k*, seed 0 | `collect_r1 .. collect_r`*k*` collect_r0` | — |
| DAgger R*k*, seed 1 or 2 | `collect_r0 seed`*n*`/collect_r1 .. seed`*n*`/collect_r`*k* | — |
| Student-state BC (matched labels) | `collect_r1 collect_r2 collect_r3 collect_r0` | `--label-budget 26373` |
| Expert BC, full data | `collect_r0 expert_full/collect_e1 .. e4` | `--label-budget 110433` |
| Ablation variants | `collect_r1 collect_r2 collect_r3 collect_r0` | the flag in `experiments.py` |

The seed-0 rows put the DAgger rounds before the expert demonstrations because that is how the
original run's directories happened to sort; the seed-1 and seed-2 rows did not sort that way.
`scripts/reproduce_sim.py ablations --refit` and the ablation job graph already use the right
order, so this table only matters when invoking the trainer directly.

The recipe is otherwise fixed. Every shipped student was fit with exactly these flags, and a
refit that omits any of them will not match:

```bash
python trace/sim/isaacgymenvs/tools/train_student_repaired.py \
  --data $TRACE_DATA/collections/collect_r1 ... collect_r3 collect_r0 \
  --manifest $TRACE_DATA/scenes/training.json --output runs/refit --device cuda:0 \
  --seed 0 --split-seed 20260907 --updates 10000 --batch-seqs 32 --lr 3e-4 \
  --head categorical_kl --select-best-validation --allow-oracle-dagger \
  --require-geometric-visibility --teacher-relative-budget --smoothness-coef 0.05 \
  --recovery-travel-m 0.12
```

Training seed is `--seed`, independent of the data: Expert BC differs across seeds only by that
flag, while the DAgger rounds also differ by which collection the teacher was queried on.

The label budget of the full-data control is matched per seed to that seed's R3 total: 110,433,
113,558 and 110,675 for seeds 0, 1 and 2, which average the 111,555 the paper reports.

Both trainers are deterministic, and the aggregate order no longer depends on where the data is
installed, so a refit is reproducible on any machine given the same command. Verified against the
shipped weights, tensor by tensor: Expert BC seed 1, the deployed DAgger R3, and the K = 2
ablation all come back bit-identical. A different GPU model or driver changes the weights slightly
and success by a few tenths of a point.

The two trainers are not interchangeable: `train_student_ablation.py` rebuilds the observation
around the ablated input, and refitting the unmodified recipe through it does not return the
deployed weights. That is why the ablation table's reference row uses the released checkpoints
rather than a refit of itself.

## Notes on exactness

- Nominal plans are regenerated per campaign and are **not** bit-identical across simulator
  processes, so compare methods within one campaign rather than across two.
- Scene-bootstrap intervals use 2,000 resamples stratified by difficulty tier, seeded for
  repeatability.
- The grasp-quality network is fixed for every method, including the baselines and the privileged
  reference; it is part of the evaluation protocol, not a component being compared. See NOTICE.
