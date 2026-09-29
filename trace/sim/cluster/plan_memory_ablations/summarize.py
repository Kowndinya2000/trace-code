"""Paired stage tables from a completed run directory.

    python cluster/plan_memory_ablations/summarize.py --out RUN_DIR [--stages ...] [--allow-partial]

Writes RUN_DIR/summaries/<stage>.json and RUN_DIR/RESULTS.md. Every cell is a
policy x condition over all evaluated scenes; paired differences are against the
released R3 under the same condition, with 95% tier-stratified scene-bootstrap
intervals (2,000 resamples). Also verifies identical initial states per batch.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import SEED, complete_eval, read_json, write_json  # noqa: E402
from experiments import CONDITIONS, REFERENCE_ROW, ROWS  # noqa: E402
from report_retrieval_campaign import interval  # noqa: E402
from run_evaluation_campaign import paired_starts  # noqa: E402


def load_case(run, name, offsets, allow_partial):
    rows = {}
    for offset in offsets:
        path = run / 'eval' / f's{SEED}_b{offset:04d}_{name}'
        if not complete_eval(path):
            if allow_partial:
                continue
            raise SystemExit(f'Incomplete: {path}')
        for row in read_json(path / f'{name}.json')['rows']:
            rows[row['scene']['sha256']] = row
    return rows


def stage_table(run, stage, models, conditions, offsets, allow_partial):
    names = [f'{m}__{c}' for c in conditions for m in models]
    for offset in offsets:
        outputs = [(run / 'eval' / f's{SEED}_b{offset:04d}_{n}', n) for n in names
                   if complete_eval(run / 'eval' / f's{SEED}_b{offset:04d}_{n}')]
        if outputs and not paired_starts(outputs)['matched']:
            raise SystemExit(f'Initial states differ in batch {offset} of {stage}')
    per = {n: load_case(run, n, offsets, allow_partial) for n in names}
    scenes = sorted(set.intersection(*(set(v) for v in per.values())))
    if not scenes:
        return {}, 0
    tiers = [per[names[0]][s]['scene']['tier'] for s in scenes]
    table = {}
    for condition in conditions:
        reference = np.array([per[f'r3__{condition}'][s]['success'] for s in scenes], float)
        for model in models:
            rows = [per[f'{model}__{condition}'][s] for s in scenes]
            success = np.array([r['success'] for r in rows], float)
            reasons = {}
            for r in rows:
                reasons[r['reason']] = reasons.get(r['reason'], 0) + 1
            diff = success - reference
            table[f'{model}__{condition}'] = dict(
                model=model, condition=condition, **CONDITIONS[condition], scenes=len(scenes),
                successes=int(success.sum()), success_pct=100 * success.mean(),
                scene_bootstrap_95ci=interval(success, tiers),
                minus_r3_pp=100 * diff.mean(),
                minus_r3_paired_95ci=None if model == 'r3' else interval(diff, tiers),
                only_this=int(((success == 1) & (reference == 0)).sum()),
                only_r3=int(((success == 0) & (reference == 1)).sum()), reasons=reasons,
                mean_decisions=float(np.mean([r['terminal_step'] for r in rows])),
                episodes_with_blackout=int(sum(r['blackout_steps_exposed'] > 0 for r in rows)),
                mean_visible_fraction=float(np.mean([r['visible_fraction'] for r in rows
                                                     if r['visible_fraction'] is not None])))
    return table, len(scenes)


def reported_table(run, models, offsets, allow_partial, condition='default'):
    """Recomputed release table: one row per variant, pooled over three seeds.

    A row's success is pooled over all (seed, scene) episodes. The paired difference is taken
    on per-scene means over the row's seeds, so the interval reflects scene difficulty rather
    than seed count, matching the main table's protocol.
    """
    per = {m: load_case(run, f'{m}__{condition}', offsets, allow_partial) for m in models}
    usable = {m: v for m, v in per.items() if v}
    rows = {label: [m for m in seeds if m in usable] for label, seeds in ROWS.items()}
    rows = {label: seeds for label, seeds in rows.items() if seeds}
    if REFERENCE_ROW not in rows:
        return {}, 0
    scenes = sorted(set.intersection(*(set(usable[m]) for m in sum(rows.values(), []))))
    if not scenes:
        return {}, 0
    tiers = [usable[rows[REFERENCE_ROW][0]][s]['scene']['tier'] for s in scenes]

    def scene_means(seeds):
        return np.mean([[float(usable[m][s]['success']) for s in scenes] for m in seeds], axis=0)

    reference = scene_means(rows[REFERENCE_ROW])
    table = {}
    for label, seeds in rows.items():
        episodes = [usable[m][s] for m in seeds for s in scenes]
        success = np.array([e['success'] for e in episodes], float)
        reasons = {}
        for e in episodes:
            reasons[e['reason']] = reasons.get(e['reason'], 0) + 1
        budget = sum(v for k, v in reasons.items() if k.endswith('_timeout'))
        solved = [e['terminal_step'] for e in episodes if e['success']]
        means = scene_means(seeds)
        diff = means - reference
        table[label] = dict(
            seeds=list(seeds), episodes=len(episodes), scenes=len(scenes),
            success_pct=100 * success.mean(), scene_bootstrap_95ci=interval(means, tiers),
            oow_pct=100 * reasons.get('oow', 0) / len(episodes), budget_pct=100 * budget / len(episodes),
            mean_steps_over_successes=float(np.mean(solved)) if solved else None,
            delta_pp=None if label == REFERENCE_ROW else 100 * diff.mean(),
            delta_paired_95ci=None if label == REFERENCE_ROW else interval(diff, tiers),
            reasons=reasons)
    return table, len(scenes)


def reported_markdown(table, scenes):
    lines = ['## Reported table', '',
             f'{scenes} scenes, three training seeds per row, evaluation seed {SEED}.', '',
             '| Variant | Success (%) | Delta vs reference (pp, 95% CI) | OOW (%) | Budget (%) | Steps |',
             '|---|---:|---|---:|---:|---:|']
    for label, r in table.items():
        if r['delta_pp'] is None:
            delta = '---'
        else:
            lo, hi = r['delta_paired_95ci']
            delta = f"{r['delta_pp']:+.1f} [{lo:+.1f}, {hi:+.1f}]"
        steps = '---' if r['mean_steps_over_successes'] is None else f"{r['mean_steps_over_successes']:.1f}"
        lines.append(f"| {label} | {r['success_pct']:.1f} | {delta} | {r['oow_pct']:.1f} | "
                     f"{r['budget_pct']:.1f} | {steps} |")
    return '\n'.join(lines + [''])


def markdown(stage, models, conditions, table, scenes):
    lines = [f'## {stage} ({scenes} scenes)', '', '| model | ' + ' | '.join(conditions) + ' |',
             '|---|' + '---:|' * len(conditions)]
    for model in models:
        cells = []
        for condition in conditions:
            r = table[f'{model}__{condition}']
            cell = f"{r['success_pct']:.1f}"
            if model != 'r3':
                lo, hi = r['minus_r3_paired_95ci']
                cell += f" ({r['minus_r3_pp']:+.1f} [{lo:+.1f}, {hi:+.1f}])"
            cells.append(cell)
        lines.append(f'| {model} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines + [''])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--stages', nargs='*')
    ap.add_argument('--allow-partial', action='store_true')
    args = ap.parse_args()
    config = read_json(args.out / 'run_config.json')
    stages = read_json(args.out / 'stages.json')
    offsets = list(range(0, config['count'], config['batch_size']))
    report = ['# Plan/memory ablation results', '',
              'Success % (paired difference vs released R3 under the same condition, pp, 95% scene-bootstrap CI).',
              'Conditions: ' + '; '.join(f'{k}={v}' for k, v in CONDITIONS.items()), '']
    all_models = sorted({m for s in stages.values() for m in s['models']})
    reported, reported_scenes = reported_table(args.out, all_models, offsets, args.allow_partial)
    if reported:
        write_json(args.out / 'summaries' / 'reported_table.json',
                   dict(scenes=reported_scenes, rows=reported))
        report.append(reported_markdown(reported, reported_scenes))
    for stage in args.stages or list(stages):
        models, conditions = stages[stage]['models'], stages[stage]['conditions']
        table, scenes = stage_table(args.out, stage, models, conditions, offsets, args.allow_partial)
        if not table:
            report.append(f'## {stage}: no complete cases yet\n')
            continue
        write_json(args.out / 'summaries' / f'{stage}.json', dict(stage=stage, models=models,
                                                                  conditions=conditions, results=table))
        report.append(markdown(stage, models, conditions, table, scenes))
    (args.out / 'RESULTS.md').write_text('\n'.join(report) + '\n')
    print('\n'.join(report))


if __name__ == '__main__':
    main()
