"""Teacher-relative action budget and recovery-aware imitation weighting.

This module is deliberately independent of Isaac Gym.  The action policy does
not predict stopping: simulator graspability can end an evaluation successfully,
while the first teacher-relative step or travel limit is a timeout failure.
"""
import numpy as np


PROTOCOL = "teacher-relative-action-budget-v1"


def config(extra_steps=20, extra_length_m=.15, primitive_length_m=.04,
           risk_start=.70, risk_temperature=.10, max_imitation_weight=3.):
    values = np.asarray([extra_length_m, primitive_length_m, risk_start,
                         risk_temperature, max_imitation_weight], dtype=float)
    if (not isinstance(extra_steps, int) or isinstance(extra_steps, bool)
            or extra_steps < 1 or not np.isfinite(values).all()
            or extra_length_m <= 0 or primitive_length_m <= 0
            or not 0 < risk_start < 1 or risk_temperature <= 0
            or max_imitation_weight < 1):
        raise ValueError("Invalid teacher-relative budget configuration")
    return dict(protocol=PROTOCOL, extra_steps=int(extra_steps),
                extra_length_m=float(extra_length_m),
                primitive_length_m=float(primitive_length_m),
                risk_start=float(risk_start),
                risk_temperature=float(risk_temperature),
                max_imitation_weight=float(max_imitation_weight),
                termination="first of step or planar TCP-travel cap",
                success="simulator graspability only; budget exhaustion is timeout")


def validate(settings):
    if not isinstance(settings, dict):
        raise ValueError("Teacher-relative budget must be a dictionary")
    expected = config(settings.get("extra_steps"), settings.get("extra_length_m"),
                      settings.get("primitive_length_m"), settings.get("risk_start"),
                      settings.get("risk_temperature"),
                      settings.get("max_imitation_weight"))
    if settings != expected:
        raise ValueError("Unknown or modified teacher-relative budget")
    return expected


def nominal_length_m(step_budget, settings):
    settings = validate(settings)
    budget = np.asarray(step_budget)
    if (budget < 0).any():
        raise ValueError("Negative teacher step budget")
    return budget.astype(np.float64) * settings["primitive_length_m"]


def recovery_weight_numpy(obs, step_budget, settings, plan_start=132):
    """Return [time, trajectory] weights from deployable state variables.

    Risk combines three continuous signals instead of waiting for a late clock:
    nominal-plan progress, EEF drift from the teacher path, and remaining travel
    slack after reserving the nominal teacher-to-go.  The latter two respond as
    soon as the student diverts.  All quantities are already available to the
    policy/controller; no future success label or privileged object state enters.
    """
    settings = validate(settings)
    obs = np.asarray(obs, dtype=np.float32)
    budget = np.asarray(step_budget, dtype=np.int64)
    if obs.ndim != 3 or budget.shape != (obs.shape[1],) or (budget < 0).any():
        raise ValueError("Expected observations [time, trajectory, feature] and one budget each")
    if obs.shape[-1] < plan_start + 10 or not np.isfinite(obs).all():
        raise ValueError("Observation does not contain the teacher-plan drift block")
    t = np.arange(obs.shape[0], dtype=np.float64)[:, None]
    n = budget.astype(np.float64)[None, :]
    progress = np.clip(t / np.maximum(n, 1.), 0., 2.)
    drift = np.linalg.norm(obs[..., plan_start + 8:plan_start + 10], axis=-1)

    # Decision-to-decision planar EEF arc is observable and deterministic from
    # the stored global EEF coordinates.  The physical runtime ledger measures
    # at controller substeps; this decision-level quantity is only a smooth
    # training signal, never the hard enforcement mechanism.
    eef = obs[..., 110:112].astype(np.float64)
    delta = np.linalg.norm(np.diff(eef, axis=0, prepend=eef[:1]), axis=-1)
    used_length = np.cumsum(delta, axis=0)
    nominal_remaining = np.maximum(n - t, 0.) * settings["primitive_length_m"]
    allowed = n * settings["primitive_length_m"] + settings["extra_length_m"]
    distance_slack = allowed - used_length - nominal_remaining - drift

    correction_steps = drift / settings["primitive_length_m"]
    step_slack = (n + settings["extra_steps"] - t
                  - np.maximum(n - t, 0.) - correction_steps)

    def sigmoid(x):
        return 1. / (1. + np.exp(-np.clip(x, -30., 30.)))

    clock_risk = sigmoid((progress - settings["risk_start"])
                         / settings["risk_temperature"])
    step_risk = sigmoid((1. - step_slack) / 1.)
    distance_risk = sigmoid((settings["primitive_length_m"] - distance_slack)
                            / settings["primitive_length_m"])
    # Probabilistic union is smooth, monotone in each source, and stays [0,1].
    risk = 1. - (1. - clock_risk) * (1. - step_risk) * (1. - distance_risk)
    return (1. + (settings["max_imitation_weight"] - 1.) * risk).astype(np.float32)


class TeacherRelativeBudgetAttempt:
    """First-attempt oracle ledger with strict teacher-relative hard timeouts."""
    def __init__(self, step_budget, horizon, settings, threshold=.9):
        self.settings = validate(settings)
        raw = np.asarray(step_budget)
        self.step_budget = raw.astype(np.int64)
        if (raw.ndim != 1 or not len(raw) or not np.array_equal(raw, self.step_budget)
                or (self.step_budget < 0).any()):
            raise ValueError("Invalid teacher step budgets")
        self.step_limit = np.minimum(self.step_budget + self.settings["extra_steps"], horizon)
        self.length_limit_m = nominal_length_m(self.step_budget, self.settings) + self.settings["extra_length_m"]
        self.horizon, self.threshold = int(horizon), float(threshold)
        self.reason = np.full(len(raw), "running", dtype="U24")
        self.step = np.full(len(raw), -1, dtype=np.int64)
        self.terminal_q = np.full(len(raw), np.nan)
        self.terminal_travel_m = np.full(len(raw), np.nan)

    @property
    def live(self):
        return self.reason == "running"

    def update(self, step, q, oow, invalid=None, travelled_m=None,
               reserve_next_m=0.):
        q = np.asarray(q)
        oow = np.asarray(oow, dtype=bool)
        travelled = np.asarray(travelled_m, dtype=np.float64)
        n = len(self.reason)
        if (q.shape != (n,) or oow.shape != (n,) or travelled.shape != (n,)
                or not np.isfinite(travelled).all() or (travelled < 0).any()
                or not np.isfinite(reserve_next_m) or reserve_next_m < 0):
            raise ValueError("Invalid teacher-relative budget update")
        bad = ~np.isfinite(q)
        if invalid is not None:
            bad |= np.asarray(invalid, dtype=bool)
        step_due = step >= self.step_limit
        distance_due = travelled >= self.length_limit_m - 1e-9
        if reserve_next_m:
            distance_due |= travelled + reserve_next_m > self.length_limit_m + 1e-9
        reason = np.full(n, "running", dtype="U24")
        reason[step_due] = "step_timeout"
        reason[distance_due] = "travel_timeout"
        reason[q > self.threshold] = "success"
        reason[q == -2.] = "out_of_view"
        reason[bad] = "invalid_state"
        reason[oow] = "oow"
        newly = self.live & (reason != "running")
        self.reason[newly], self.step[newly] = reason[newly], int(step)
        self.terminal_q[newly] = q[newly]
        self.terminal_travel_m[newly] = travelled[newly]
        return np.flatnonzero(newly)

    def rows(self):
        if self.live.any():
            raise RuntimeError("Cannot summarize unfinished attempts")
        return [dict(reason=str(reason), terminal_step=int(step),
                     terminal_q=float(q) if np.isfinite(q) else None,
                     terminal_travel_m=float(travel), success=reason == "success",
                     step_budget=int(budget), step_limit=int(step_limit),
                     length_limit_m=float(length_limit))
                for reason, step, q, travel, budget, step_limit, length_limit in zip(
                    self.reason, self.step, self.terminal_q, self.terminal_travel_m,
                    self.step_budget, self.step_limit, self.length_limit_m)]


class TeacherRelativeRolloutAttempt(TeacherRelativeBudgetAttempt):
    """Collect frozen-policy transitions to the cap without q-based stopping."""
    def update(self, step, q, oow, invalid=None, travelled_m=None,
               reserve_next_m=0.):
        q = np.asarray(q)
        oow = np.asarray(oow, dtype=bool)
        travelled = np.asarray(travelled_m, dtype=np.float64)
        n = len(self.reason)
        if (q.shape != (n,) or oow.shape != (n,) or travelled.shape != (n,)
                or not np.isfinite(travelled).all() or (travelled < 0).any()
                or not np.isfinite(reserve_next_m) or reserve_next_m < 0):
            raise ValueError("Invalid teacher-relative rollout update")
        bad = ~np.isfinite(q)
        if invalid is not None:
            bad |= np.asarray(invalid, dtype=bool)
        reason = np.full(n, "running", dtype="U24")
        reason[step >= self.step_limit] = "step_timeout"
        distance_due = travelled >= self.length_limit_m - 1e-9
        if reserve_next_m:
            distance_due |= travelled + reserve_next_m > self.length_limit_m + 1e-9
        reason[distance_due] = "travel_timeout"
        reason[q == -2.] = "out_of_view"
        reason[bad] = "invalid_state"
        reason[oow] = "oow"
        newly = self.live & (reason != "running")
        self.reason[newly], self.step[newly] = reason[newly], int(step)
        self.terminal_q[newly] = q[newly]
        self.terminal_travel_m[newly] = travelled[newly]
        return np.flatnonzero(newly)


class BudgetThenGraspAttempt(TeacherRelativeBudgetAttempt):
    """Execute to a travel budget without graspability stopping, then score the final state.

    The student acts until its planar TCP travel reaches L * primitive_length_m + tolerance_m
    (or the next primitive would exceed it), or until the evaluation horizon. Only then is
    simulator graspability read: q > threshold is a success, otherwise budget_ungraspable.
    Object out-of-workspace, invalid state and target out of view still end the episode early
    as failures. L is the teacher plan length, or a fixed length for plan-free budgets.
    """
    def __init__(self, step_budget, horizon, settings, tolerance_m, threshold=.9):
        super().__init__(step_budget, horizon, settings, threshold)
        if not np.isfinite(tolerance_m) or tolerance_m < 0:
            raise ValueError("Invalid budget tolerance")
        self.tolerance_m = float(tolerance_m)
        self.step_limit = np.full(len(self.step_budget), int(horizon), dtype=np.int64)
        self.length_limit_m = nominal_length_m(self.step_budget, self.settings) + self.tolerance_m

    def update(self, step, q, oow, invalid=None, travelled_m=None,
               reserve_next_m=0.):
        q = np.asarray(q)
        oow = np.asarray(oow, dtype=bool)
        travelled = np.asarray(travelled_m, dtype=np.float64)
        n = len(self.reason)
        if (q.shape != (n,) or oow.shape != (n,) or travelled.shape != (n,)
                or not np.isfinite(travelled).all() or (travelled < 0).any()
                or not np.isfinite(reserve_next_m) or reserve_next_m < 0):
            raise ValueError("Invalid budget-then-grasp update")
        bad = ~np.isfinite(q)
        if invalid is not None:
            bad |= np.asarray(invalid, dtype=bool)
        due = (step >= self.step_limit) | (travelled >= self.length_limit_m - 1e-9)
        if reserve_next_m:
            due |= travelled + reserve_next_m > self.length_limit_m + 1e-9
        reason = np.full(n, "running", dtype="U24")
        reason[due] = np.where(q[due] > self.threshold, "success", "budget_ungraspable")
        reason[q == -2.] = "out_of_view"
        reason[bad] = "invalid_state"
        reason[oow] = "oow"
        newly = self.live & (reason != "running")
        self.reason[newly], self.step[newly] = reason[newly], int(step)
        self.terminal_q[newly] = q[newly]
        self.terminal_travel_m[newly] = travelled[newly]
        return np.flatnonzero(newly)
