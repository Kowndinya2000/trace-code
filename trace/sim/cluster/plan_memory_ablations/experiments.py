"""Single source of truth for the plan-horizon and recurrent-memory ablations.

Every fit is an offline refit on the exact aggregate behind the released DAgger R3 student
(expert demonstrations plus DAgger rounds 1-3), with the same loss, recovery weighting,
optimizer, update count and validation-loss checkpoint selection. With no flags the trainer
changes nothing but the listed flag. A refit is not bit-identical to the released R3, which
came through the DAgger pipeline's own trainer, so the reference row uses the released
checkpoints and every ablated row is compared against them.

Three training seeds per row. The K=4 reference row is the deployed student at the three
seeds of the main table (`r3`, `r3s1`, `r3s2` -- a full DAgger chain per seed, not a refit);
each ablated row is the unsuffixed refit plus its `s1` / `s2` refits.

Evaluation: the 511-scene set, perturbation seed 7, link-union arm occlusion (10 mm), 10%
detection dropout, horizon-uniform 5-step blackout, simulator-oracle graspability above 0.9,
and the teacher-relative execution budget (L+20 decisions, nominal travel + 0.15 m).
"""

R3_NAME = 'r3'

FITS = {
    # recurrent memory
    'nogru': ['--ablate', 'no_gru'],
    # nominal-plan look-ahead window (end-effector poses); object centres stay at the current index
    'k1': ['--plan-k', '1'],
    'k2': ['--plan-k', '2'],
    'k8': ['--plan-k', '8'],
    'fullplan': ['--plan-mode', 'full_static'],
}
RELEASED_SEEDS = ('r3', 'r3s1', 'r3s2')   # resolved to shipped checkpoints by common.Layout
for _seed in (1, 2):
    for _base in ('nogru', 'k1', 'k2', 'k8', 'fullplan'):
        FITS[f'{_base}s{_seed}'] = FITS[_base] + ['--seed', str(_seed)]


def ablation(model):
    """Architecture/observation ablation name the evaluator must be told."""
    args = FITS.get(model, [])
    if '--ablate' in args:
        return args[args.index('--ablate') + 1]
    return 'none'


CONDITIONS = {
    'default': dict(p_drop=.1, blackout_len=5, blackout_schedule='horizon_uniform'),
}

# The reported table: one row per variant, three training seeds each. Keys are the row
# labels used in the paper; the first row is the reference every difference is against.
ROWS = {
    'TRACE (K=4)': ('r3', 'r3s1', 'r3s2'),
    'w/o GRU': ('nogru', 'nogrus1', 'nogrus2'),
    'K=1': ('k1', 'k1s1', 'k1s2'),
    'K=2': ('k2', 'k2s1', 'k2s2'),
    'K=8': ('k8', 'k8s1', 'k8s2'),
    'Full plan': ('fullplan', 'fullplans1', 'fullplans2'),
}
REFERENCE_ROW = 'TRACE (K=4)'

STAGES = {
    # the reported table: TRACE (K=4), w/o GRU, K=1, K=2, K=8, full plan -- three seeds each
    'architecture': (list(RELEASED_SEEDS) + sorted(FITS), ['default']),
}

for _stage, (_models, _conditions) in STAGES.items():
    assert all(m in RELEASED_SEEDS or m in FITS for m in _models), _stage
    assert all(c in CONDITIONS for c in _conditions), _stage

assert set(sum(ROWS.values(), ())) == set(RELEASED_SEEDS) | set(FITS)
assert REFERENCE_ROW in ROWS
