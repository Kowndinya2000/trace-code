"""CPU-testable contracts for the versioned, single-attempt evaluation.

No simulator imports: manifests, random streams, and terminal accounting can
be checked independently of CUDA. Historical evaluators remain unchanged.
"""
from pathlib import Path
import hashlib
import json
import numpy as np

PROTOCOL = "retrieval-expert-graspability-v4"


def execution_stopping(actor, requested=None, collect=False, diagnostic_oracle=False):
    """Keep expert demonstrations independent of nominal rollout length."""
    mode = requested or ('expert_graspability' if actor == 'teacher' else
                         'learned_score' if actor == 'student' and collect else
                         'nominal_action_budget')
    if mode not in ('expert_graspability', 'nominal_action_budget', 'learned_score',
                    'oracle_graspability', 'budget_rollout', 'budget_then_grasp'):
        raise ValueError('Unknown stopping rule: ' + mode)
    if mode == 'expert_graspability' and actor != 'teacher':
        raise ValueError('Expert stopping requires the privileged teacher')
    if mode == 'learned_score' and actor != 'student':
        raise ValueError('Only a student predicts stopping')
    if mode == 'oracle_graspability' and (actor != 'student' or not diagnostic_oracle):
        raise ValueError('Oracle student stopping requires an explicitly labeled simulator diagnostic')
    if mode == 'budget_rollout' and (actor != 'student' or not collect or not diagnostic_oracle):
        raise ValueError('Budget rollouts are simulator-only student stopping data')
    if mode == 'budget_then_grasp' and (actor != 'student' or collect or not diagnostic_oracle):
        raise ValueError('Budget-then-grasp is a simulator evaluation of a student without stopping')
    permitted_collection = (('expert_graspability',) if actor == 'teacher' else
                            ('learned_score', 'oracle_graspability', 'budget_rollout'))
    if collect and mode not in permitted_collection:
        raise ValueError('Demonstrations cannot use nominal-length stopping')
    return mode


def action_supervision_mask(executed, q, invalid=None):
    """Once the teacher would stop, teach graspability, not an arbitrary push."""
    executed, q = np.asarray(executed, bool), np.asarray(q)
    if executed.shape != q.shape: raise ValueError('Action/score shape mismatch')
    mask = executed & np.isfinite(q) & (q <= .9) & (q != -2.)
    if invalid is not None: mask &= ~np.asarray(invalid, bool)
    return mask


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scene_seed(scene_hash, seed, stream):
    """Stable across actor choice, scene ordering, and simulation batch size."""
    key = f"{scene_hash}:{int(seed)}:{stream}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "little")


def read_manifest(path):
    payload = json.loads(Path(path).read_text())
    rows = payload["scenes"]
    if not rows:
        raise ValueError("Empty scene manifest")
    seen = set()
    for row in rows:
        if not Path(row["path"]).is_absolute():
            raise ValueError("Manifest scene paths must be absolute")
        digest = sha256(row["path"])
        if digest != row["sha256"]:
            raise ValueError(f"Scene changed since manifest creation: {row['path']}")
        if digest in seen:
            raise ValueError("Duplicate scene content in manifest")
        seen.add(digest)
    return payload


def make_manifest(directory, output, limit=None, seed=20260907, role="development"):
    """Stratified deterministic ordering, preserving source scene identities."""
    directory = Path(directory).resolve()
    meta_path = directory / "meta.json"
    meta = {r["scene"]: r for r in json.loads(meta_path.read_text())["scenes"]} if meta_path.exists() else {}
    groups = {}
    for p in sorted(directory.glob("*.txt")):
        row = dict(path=str(p), sha256=sha256(p),
                   tier=meta.get(p.name, {}).get("tier", "unspecified"),
                   source_metadata=meta.get(p.name, {}))
        groups.setdefault(row["tier"], []).append(row)
    rng = np.random.default_rng(seed)
    for group in groups.values():
        rng.shuffle(group)
    rows = []
    for i in range(max((len(g) for g in groups.values()), default=0)):
        for tier in sorted(groups):
            if i < len(groups[tier]): rows.append(groups[tier][i])
    if limit is not None: rows = rows[:limit]
    out = Path(output)
    if out.exists(): raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(role=role, seed=seed, source=str(directory), scenes=rows), indent=2)+"\n")
    read_manifest(out)
    return rows


class FirstAttempt:
    """Latch one verdict; later physics and resets cannot change it."""
    def __init__(self, n, horizon, threshold=0.9):
        self.horizon, self.threshold = horizon, threshold
        self.reason = np.full(n, "running", dtype="U24")
        self.step = np.full(n, -1, dtype=np.int64)
        self.terminal_q = np.full(n, np.nan)

    @property
    def live(self):
        return self.reason == "running"

    def update(self, step, q, oow, invalid=None, exhausted=None):
        q, oow = np.asarray(q), np.asarray(oow, dtype=bool)
        n = len(self.reason)
        if q.shape != (n,) or oow.shape != (n,): raise ValueError("Batch mismatch")
        bad = ~np.isfinite(q)
        if invalid is not None: bad |= np.asarray(invalid, dtype=bool)
        reason = np.full(n, "running", dtype="U24")
        if step >= self.horizon: reason[:] = "horizon"
        if exhausted is not None: reason[np.asarray(exhausted, dtype=bool)] = "plan_exhausted"
        reason[q > self.threshold] = "success"
        reason[q == -2.] = "out_of_view"
        reason[bad] = "invalid_state"
        reason[oow] = "oow"  # containment failure wins over simultaneous success
        newly = self.live & (reason != "running")
        self.reason[newly], self.step[newly] = reason[newly], step
        self.terminal_q[newly] = q[newly]
        return np.flatnonzero(newly)

    def rows(self):
        if self.live.any(): raise RuntimeError("Cannot summarize unfinished attempts")
        return [dict(reason=str(r), terminal_step=int(t),
                     terminal_q=float(q) if np.isfinite(q) else None,
                     success=bool(r == "success"))
                for r, t, q in zip(self.reason, self.step, self.terminal_q)]


class NominalBudgetAttempt(FirstAttempt):
    """Execute each stored plan's action budget; score its final state only.

    Current graspability never retires a live episode before its budget.
    Invalid states and physical task failures remain absorbing failures.
    This is an evaluation baseline, never the expert demonstration rule.
    """
    def __init__(self, budgets, horizon, threshold=0.9):
        raw = np.asarray(budgets)
        limits = raw.astype(np.int64)
        if (raw.ndim != 1 or not len(raw) or not np.array_equal(raw, limits)
                or (limits < 0).any() or (limits > horizon).any()):
            raise ValueError('Invalid nominal action budgets')
        super().__init__(len(limits), horizon, threshold)
        self.budgets = limits

    def update(self, step, q, oow, invalid=None, exhausted=None):
        q, oow = np.asarray(q), np.asarray(oow, dtype=bool)
        n = len(self.reason)
        if q.shape != (n,) or oow.shape != (n,): raise ValueError('Batch mismatch')
        if (self.live & (step > self.budgets)).any():
            raise RuntimeError('A live episode exceeded its nominal action budget')
        due = step >= self.budgets
        if exhausted is not None and not np.array_equal(np.asarray(exhausted, dtype=bool), due):
            raise ValueError('Replay exhaustion differs from nominal action budget')
        bad = ~np.isfinite(q)
        if invalid is not None: bad |= np.asarray(invalid, dtype=bool)
        reason = np.full(n, 'running', dtype='U24')
        reason[due] = 'budget_exhausted'
        reason[due & (q > self.threshold)] = 'success'
        reason[q == -2.] = 'out_of_view'
        reason[bad] = 'invalid_state'
        reason[oow] = 'oow'
        newly = self.live & (reason != 'running')
        self.reason[newly], self.step[newly] = reason[newly], step
        self.terminal_q[newly] = q[newly]
        return np.flatnonzero(newly)

    def rows(self):
        return [dict(row, step_budget=int(budget))
                for row, budget in zip(super().rows(), self.budgets)]


class LearnedStopAttempt(FirstAttempt):
    """Only the student's stop decision can succeed; timeout is a failure.

    With stop=None, check physical failures only (after a completed push).
    With stop supplied, judge the current decision BEFORE executing a push.
    Graspability labels never cause stopping by themselves.
    """
    def update(self, step, q, oow, invalid=None, exhausted=None, stop=None):
        q, oow = np.asarray(q), np.asarray(oow,dtype=bool)
        if q.shape!=self.reason.shape or oow.shape!=self.reason.shape:
            raise ValueError('Batch mismatch')
        if step>self.horizon and self.live.any(): raise RuntimeError('Stop policy exceeded timeout')
        bad=~np.isfinite(q)
        if invalid is not None: bad |= np.asarray(invalid,dtype=bool)
        reason=np.full(len(q),'running',dtype='U24')
        if stop is not None:
            stop=np.asarray(stop,dtype=bool)
            if stop.shape!=q.shape: raise ValueError('Stop batch mismatch')
            if step>=self.horizon: reason[:]='timeout'
            reason[stop]='premature_stop'
            reason[stop & (q>self.threshold)]='success'
        reason[q==-2.]='out_of_view'
        reason[bad]='invalid_state'
        reason[oow]='oow'
        newly=self.live & (reason!='running')
        self.reason[newly],self.step[newly]=reason[newly],step
        self.terminal_q[newly]=q[newly]
        return np.flatnonzero(newly)


def token_order(clutter_permutation):
    p = np.asarray(clutter_permutation, dtype=np.int64)
    if sorted(p.tolist()) != list(range(len(p))):
        raise ValueError("Invalid clutter permutation")
    return np.r_[0, p + 1]


def aligned_student_obs(so, teacher_obs, centers, vis_world, age_world,
                        permutation, prev_action, plan_xy, t, plan_actions,
                        plan_objects):
    order = token_order(permutation)
    return so.build_student_obs(teacher_obs, centers,
        np.asarray(vis_world)[order], np.asarray(age_world)[order],
        prev_action, plan_xy, t, plan_actions, plan_obj_xy=plan_objects)


def primitive_vectors(total=0.04):
    s = total / 2.
    v = np.zeros((16, 2), np.float32)
    v[:4] = np.array([[0, total], [total, 0], [0, -total], [-total, 0]])
    for i, signs in enumerate([(1,1),(-1,1),(1,-1),(-1,-1)]):
        v[4+3*i] = np.array(signs) * total / np.sqrt(2)
        v[5+3*i] = v[6+3*i] = np.array(signs) * s
    return v
