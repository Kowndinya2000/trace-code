"""Paths, hashes and small helpers shared by prepare / run_job / run_pool / summarize."""
import hashlib
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PKG = REPO / 'isaacgymenvs'
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(PKG / 'tools'), str(REPO), str(PKG)]

R3_SHA = 'dda9db7dc06f30bce5aa9ab301e05fc6781641c0f3ae450af3e50948c7ef9d30'
TEACHER_SHA = '8bb5a5d6d3503e34c5b92ac1d3bff54cae3cdbac111d5fcd71959d8c3d1056b9'
SEED, HORIZON, BATCH, SCENES = 7, 120, 64, 511


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(payload, indent=2) + '\n')
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


class Layout:
    """Locations inside the downloaded data ($TRACE_DATA) and the run directory."""

    def __init__(self, data, out):
        self.data = Path(data).resolve()
        self.out = Path(out).resolve()
        # Aggregate order is part of the recipe: the trainer now follows the order given here,
        # and this is the order the shipped fits were aggregated in (DAgger rounds, then the
        # expert demonstrations). Passing r0 first trains a different, equally valid student.
        self.collections = [self.data / 'collections' / f'collect_r{k}' for k in (1, 2, 3, 0)]
        self.scenes = self.data / 'scenes'
        self.training_manifest = self.scenes / 'training.json'
        self.development_manifest_original = self.scenes / 'development.json'
        self.teacher = self.data / 'checkpoints/teacher_ep210.pth'
        self.r3 = self.data / 'checkpoints/trace_r3.pt'
        self.budget = self.data / 'budget.json'
        self.bundle_fits = self.data / 'checkpoints/ablations/fits'
        self.development_manifest = self.out / 'inputs/development_relocated.json'
        self.fits = self.out / 'fits'
        self.eval = self.out / 'eval'
        self.logs = self.out / 'logs'
        self.jobs = self.out / 'jobs.json'
        self.frozen = self.out / 'frozen_inputs.json'

    def required(self, with_collections=True):
        """Everything prepare() needs before it will write a job graph.

        The label collections are only needed to refit; evaluating the shipped ablation
        checkpoints does not touch them.
        """
        core = [self.training_manifest, self.development_manifest_original, self.teacher,
                self.budget] + [self.data / 'checkpoints' / name for name in self.RELEASED.values()]
        return core + (self.collections if with_collections else [])

    # The reported K=4 reference row is the deployed student itself, at the same three
    # training seeds as the main table (a DAgger chain per seed), not a refit.
    RELEASED = {'r3': 'trace_r3.pt', 'r3s1': 'trace_r3_seed1.pt', 'r3s2': 'trace_r3_seed2.pt'}

    def checkpoint(self, model):
        if model in self.RELEASED:
            return self.data / 'checkpoints' / self.RELEASED[model]
        return self.fits / model / 'student.pt'


def layout_from(args):
    data = getattr(args, 'data', None) or os.environ.get('TRACE_DATA')
    out = getattr(args, 'out', None) or os.environ.get('TRACE_RUNS')
    if not data or not out:
        raise SystemExit('Set --data/--out, or TRACE_DATA and TRACE_RUNS')
    return Layout(data, out)


def python_executable():
    return os.environ.get('TRACE_PYTHON', sys.executable)


def frozen_paths(lay):
    """Everything whose change would silently alter a result."""
    paths = [PKG / 'tools' / name for name in (
        'evaluate_retrieval.py', 'run_retrieval_evaluation.sh', 'validate_retrieval_run.py',
        'train_student_ablation.py', 'train_student_repaired.py', '_net_compat.py', '_ckpt_compat.py',
        'teacher_reference.py')]
    for folder in ('open_loop', 'learning', 'tasks', 'utils'):
        paths += sorted((PKG / folder).rglob('*.py'))
    paths += sorted((PKG / 'cfg').rglob('*.yaml'))
    paths += [PKG / 'logs_grasp/grasp_model-89.pth', PKG / 'logs_grasp/snapshot-post-020000.reinforcement.pth',
              REPO / 'cluster/runtime_env.sh', HERE / 'experiments.py', HERE / 'common.py', HERE / 'run_job.py',
              lay.teacher, lay.r3, lay.training_manifest, lay.development_manifest, lay.budget]
    paths += sorted((REPO / 'assets/urdf/more/blocks-more').glob('*.obj'))
    return [p for p in paths if p.exists()]


def check_frozen(lay):
    frozen = read_json(lay.frozen)
    changed = [p for p, h in frozen.items() if not Path(p).exists() or sha256(p) != h]
    if changed:
        raise RuntimeError(f'Frozen inputs changed since prepare: {changed[:5]}')


def eval_dir(lay, offset, name):
    return lay.eval / f's{SEED}_b{offset:04d}_{name}'


def complete_eval(path):
    path = Path(path)
    if not (path / 'complete.json').exists() or not (path / 'validation.json').exists():
        return False
    return bool(read_json(path / 'validation.json').get('valid'))


def complete_fit(path):
    return (Path(path) / 'complete.json').exists() and (Path(path) / 'student.pt').exists()
