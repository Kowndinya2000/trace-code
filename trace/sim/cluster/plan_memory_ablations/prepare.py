"""Check the inputs, copy the scene manifest, freeze inputs, write the job graph.

    python cluster/plan_memory_ablations/prepare.py --data $TRACE_DATA --out $TRACE_RUNS/ablations \
        [--stages A1_memory B_window ...] [--use-bundle-fits] [--count 511] [--skip-checksums]

Job graph (jobs.json):
  fit:<model>                 one refit, ~2 min on an RTX 3090 (+ ~25 s CPU loading, ~10 GB RAM)
  plan:<offset>               nominal teacher rollouts for one 64-scene batch, shared by every case
  case:<model>__<cond>:<off>  one policy x condition on one batch (~30 s), depends on its fit and plan
Plans are shared across ALL stages, so every case is paired on identical plans and starts.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (BATCH, HORIZON, REPO, SCENES, SEED, R3_SHA, TEACHER_SHA,  # noqa: E402
                    complete_fit, eval_dir, frozen_paths, layout_from, read_json, sha256,
                    write_json)
from experiments import CONDITIONS, FITS, STAGES, ablation  # noqa: E402


def verify_inputs(lay, skip_checksums, with_collections=True):
    """Every input the grid depends on must be present, and the four canonical ones unchanged."""
    missing = [p for p in lay.required(with_collections) if not Path(p).exists()]
    if missing:
        raise SystemExit('Missing inputs (run scripts/download_data.py; refitting also needs '
                         '--only collections):\n  '
                         + '\n  '.join(str(p) for p in missing))
    if not skip_checksums:
        # Checkpoints are pinned by content. The manifests are not: they carry absolute scene
        # paths after download, so every scene is verified against its own hash instead
        # (download_data.py at install time, relocate_manifest below on every run).
        for path, expected in ((lay.r3, R3_SHA), (lay.teacher, TEACHER_SHA)):
            if sha256(path) != expected:
                raise SystemExit(f'Hash mismatch for {path}')
    return len(lay.required(with_collections))


def relocate_manifest(lay):
    """Copy the manifest into the run directory after checking every scene against its hash."""
    manifest = read_json(lay.development_manifest_original)
    for row in manifest['scenes']:
        scene = Path(row['path'])
        if not scene.is_absolute():
            scene = (lay.scenes / scene).resolve()
        if not scene.exists() or sha256(scene) != row['sha256']:
            raise SystemExit(f'Scene missing or changed: {scene}')
        row['path'] = str(scene)
    write_json(lay.development_manifest, manifest)
    return len(manifest['scenes'])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data')
    ap.add_argument('--out')
    ap.add_argument('--stages', nargs='*', default=list(STAGES))
    ap.add_argument('--use-bundle-fits', action='store_true',
                    help='Copy the RTX 3090 fits from the bundle instead of refitting (pairs with local results)')
    ap.add_argument('--count', type=int, default=SCENES, help='Scenes to evaluate (smoke tests use fewer)')
    ap.add_argument('--batch-size', type=int, default=BATCH)
    ap.add_argument('--skip-checksums', action='store_true')
    args = ap.parse_args()
    lay = layout_from(args)
    unknown = [s for s in args.stages if s not in STAGES]
    if unknown:
        raise SystemExit(f'Unknown stages {unknown}; choices {list(STAGES)}')
    lay.out.mkdir(parents=True, exist_ok=True)
    files = verify_inputs(lay, args.skip_checksums, with_collections=not args.use_bundle_fits)
    scenes = relocate_manifest(lay)
    if not 0 < args.count <= scenes:
        raise SystemExit('Invalid --count')

    models = sorted({m for s in args.stages for m in STAGES[s][0]})
    cases = sorted({(m, c) for s in args.stages for m in STAGES[s][0] for c in STAGES[s][1]})
    fits = [m for m in models if m in FITS]        # released checkpoints are never refitted
    if args.use_bundle_fits:
        for model in fits:
            source = lay.bundle_fits / model
            target = lay.fits / model
            if complete_fit(source) and not target.exists():
                shutil.copytree(source, target)
                if sha256(target / 'student.pt') != read_json(target / 'complete.json')['checkpoint_sha256']:
                    raise SystemExit(f'Bundle fit {model} checkpoint hash mismatch')

    budget = read_json(lay.budget)['teacher_relative_budget']
    common = dict(manifest=str(lay.development_manifest), horizon=HORIZON, seed=SEED, pos_noise=.015,
                  yaw_noise_deg=10., cpu_threads=4, teacher_checkpoint=str(lay.teacher),
                  reward_recipe='graspability_only')
    jobs = {}
    for model in fits:
        jobs[f'fit:{model}'] = dict(kind='fit', model=model, args=FITS[model], deps=[],
                                    output=str(lay.fits / model))
    for offset in range(0, args.count, args.batch_size):
        size = min(args.batch_size, args.count - offset)
        plan = eval_dir(lay, offset, 'plan')
        jobs[f'plan:{offset}'] = dict(kind='plan', deps=[], output=str(plan),
                                      spec=dict(common, offset=offset, batch_size=size, output=str(plan), cases=[]))
        for model, condition in cases:
            name = f'{model}__{condition}'
            case = dict(name=name, checkpoint=str(lay.checkpoint(model)), occlusion_mode='link_union',
                        occlusion_margin_m=.01, actor='student', stopping='oracle_graspability',
                        diagnostic_oracle=True, teacher_relative_budget=budget, **CONDITIONS[condition])
            if ablation(model) != 'none':
                case['ablation'] = ablation(model)
            out = eval_dir(lay, offset, name)
            # Released TRACE checkpoints (r3/r3s1/r3s2) are evaluated as-is;
            # only ablation variants produced by this harness depend on a fit job.
            deps = [f'plan:{offset}'] + ([f'fit:{model}'] if model in FITS else [])
            jobs[f'case:{name}:{offset}'] = dict(
                kind='case', model=model, condition=condition, deps=deps, output=str(out),
                spec=dict(common, offset=offset, batch_size=size, output=str(out), cases=[case],
                          plan_source=str(plan / 'nominal.json')))
    write_json(lay.jobs, jobs)
    write_json(lay.out / 'stages.json', {s: dict(models=STAGES[s][0], conditions=STAGES[s][1]) for s in args.stages})
    write_json(lay.out / 'run_config.json', dict(data=str(lay.data), out=str(lay.out), count=args.count,
                                                 batch_size=args.batch_size, stages=args.stages,
                                                 use_bundle_fits=args.use_bundle_fits, repo=str(REPO)))
    for kind in ('fit', 'plan', 'case'):
        ids = [j for j in jobs if j.startswith(kind + ':')]
        (lay.out / f'joblist_{kind}.txt').write_text('\n'.join(ids) + ('\n' if ids else ''))
    write_json(lay.frozen, {str(p): sha256(p) for p in frozen_paths(lay)})
    pending_fits = [m for m in fits if not complete_fit(lay.fits / m)]
    print(json.dumps(dict(bundle_files=files, scenes=scenes, stages=args.stages, fits=len(fits),
                          fits_to_train=len(pending_fits), plans=sum(k.startswith('plan:') for k in jobs),
                          cases=len(cases), case_jobs=sum(k.startswith('case:') for k in jobs),
                          out=str(lay.out)), indent=2))


if __name__ == '__main__':
    main()
