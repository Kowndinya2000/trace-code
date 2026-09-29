"""Expert-success demonstrations and deployable student stopping (v4).

Example: python tools/evaluate_retrieval.py task=MoreEvaluation
  train=MoreOpenLoopSetSCPPO test=True headless=True force_render=False
  checkpoint=... +evaluation_spec=/absolute/path/spec.json

The spec selects an explicit hashed manifest, batch, perturbation seed, and
actors. Each actor gets the same requested initial state. Per-scene traces,
initial-state differences, reset counters, and source/config hashes are saved.
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import isaacgym  # must precede torch
import tools._net_compat  # noqa
import tools._ckpt_compat  # noqa
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path
import json
import shutil
import time
import platform
import numpy as np
import torch
from isaacgymenvs.open_loop import student_obs as so
from isaacgymenvs.open_loop import obs_variants as ov
from isaacgymenvs.open_loop.arm_occlusion import SimulatorArmOcclusion, visibility_settings
from isaacgymenvs.open_loop.evaluation_core import (
    PROTOCOL, FirstAttempt, NominalBudgetAttempt, LearnedStopAttempt, aligned_student_obs, read_manifest, scene_seed,
    execution_stopping, action_supervision_mask,
    sha256, primitive_vectors)
from isaacgymenvs.open_loop.teacher_relative_budget import (
    TeacherRelativeBudgetAttempt, TeacherRelativeRolloutAttempt, BudgetThenGraspAttempt,
    validate as validate_teacher_relative_budget)
from isaacgymenvs.learning.policy_stop import BudgetedLearnedStopAttempt

ROOT = Path(__file__).resolve().parents[1]


def cpu(t):
    return t.detach().cpu().numpy().copy()


def teacher_forward(player, obs):
    inputs = dict(is_train=False, prev_actions=None, obs=player._preproc_obs(obs),
                  rnn_states=player.states)
    with torch.no_grad(): out = player.model(inputs)
    player.states = out["rnn_states"]
    logits = out["logits"]
    return (logits[0] if isinstance(logits, (tuple,list)) else logits), out["values"].flatten()


def zero_retired_teacher(player, live):
    if getattr(player, "is_rnn", False):
        for state in player.states: state[:, ~torch.as_tensor(live, device=state.device)] = 0


def load_student(case, device):
    from isaacgymenvs.learning.student_net import StudentNet
    from isaacgymenvs.learning.student_ablate import obs_mask, strip_recurrence
    path = Path(case["checkpoint"]).resolve()
    if path.is_dir():
        path = next((path / name for name in ("student.pt", "student_best.pth")
                     if (path/name).is_file()), None)
    if path is None: raise FileNotFoundError(case["checkpoint"])
    ck = torch.load(path, map_location="cpu")
    sd = ck.get("model", ck)
    n_out = sd["pi.weight"].shape[0]
    embed = sd["embed.0.weight"].shape[0]
    hidden = sd["gru.weight_ih_l0"].shape[0]//3 if "gru.weight_ih_l0" in sd else sd["pi.weight"].shape[1]
    n_extra = sd["trunk.0.weight"].shape[1] - 2 * embed
    net = StudentNet(n_actions=n_out, embed=embed, gru_hidden=hidden, n_extra=n_extra,
                     termination_head="stop.weight" in sd).to(device)
    ablation = case.get("ablation", "none")
    if ck.get("ablate") is not None and ck.get("obs_layout") is not None and ck["ablate"] != ablation:
        raise ValueError("Case ablation differs from the ablation checkpoint")
    if ablation in ("no_gru", "no_plan_no_gru"): net = strip_recurrence(net)
    if ablation == "no_gru_wide": net = strip_recurrence(net, sd["gru.net.0.weight"].shape[0])
    net.obs_layout = ov.layout(ck.get("obs_layout"))
    if ov.obs_dim(net.obs_layout) != so.N_TOKENS * so.TOKEN_DIM + n_extra:
        raise ValueError("Checkpoint observation layout does not match its trunk width")
    constraints = ck.get('plan_constraints')
    cartesian = ck.get('cartesian_policy')
    teacher_relative_budget = ck.get('teacher_relative_budget')
    if teacher_relative_budget is not None:
        teacher_relative_budget = validate_teacher_relative_budget(teacher_relative_budget)
    if constraints is not None and cartesian is not None:
        raise ValueError('Checkpoint cannot require two action decoders')
    if constraints is not None:
        from isaacgymenvs.open_loop.plan_constraint import config as constraint_config
        if constraints != constraint_config(constraints['radius_m'], constraints['correction_step_m']):
            raise ValueError('Invalid checkpoint path constraints')
        if n_out != 16 or net.stop is not None or ablation != 'none':
            raise ValueError('Constrained pilot requires action-only categorical student')
        if 'plan_constraint_required' not in sd or int(sd['plan_constraint_required']) != 1:
            raise ValueError('Missing required decoder marker')
        net.register_buffer('plan_constraint_required',torch.tensor(1,dtype=torch.int64,device=device))
        net.plan_constraints = constraints
    if cartesian is not None:
        from isaacgymenvs.open_loop.teacher_relative_cartesian import validate as validate_cartesian
        cartesian = validate_cartesian(cartesian)
        if n_out != cartesian['output_dim'] or ck.get('head') != cartesian['head']:
            raise ValueError('Cartesian checkpoint head does not match its decoder contract')
        if net.stop is not None or ablation != 'none':
            raise ValueError('Cartesian pilot requires an action-only, non-ablated student')
        net.cartesian_policy = cartesian
    net.load_state_dict(sd)
    net.teacher_relative_budget = teacher_relative_budget
    net.eval()
    net.initial_scene = ablation == "initial_scene"
    if net.initial_scene and not ov.is_default(net.obs_layout):
        raise ValueError("Initial-scene control requires the released 166-D layout")
    mask = obs_mask("none" if ablation == "no_gru_wide" else ablation, device)
    net.plan_drop = ov.validate_drop(ck.get("plan_drop"))
    component = ov.component_mask(net.obs_layout, net.plan_drop)
    if component is not None:
        if mask is not None and mask.shape[0] != component.shape[0]:
            raise ValueError("Observation mask and plan-component mask disagree in width")
        component = torch.tensor(component, device=device)
        mask = component if mask is None else mask * component
    return net, mask, n_out, str(path), sha256(path)


def load_policy_stop(case, device, action_checkpoint_sha256):
    path = Path(case['policy_stop_checkpoint']).resolve()
    if path.is_dir():
        path = path / 'policy_stop.pt'
    if not path.is_file():
        raise FileNotFoundError(path)
    from isaacgymenvs.learning.policy_stop import load_checkpoint
    model, payload = load_checkpoint(path, action_checkpoint_sha256, device)
    return model, payload, str(path), sha256(path)


def report_attempt(ledger, rows):
    verdicts = ledger.rows()
    for v, row in zip(verdicts, rows): v.update(scene=row)
    return verdicts


def update_ledger(ledger, env, step, exhausted=None, initial=False, stop=None):
    oow = env.physical_oow() | env.step_mesh_oow
    if initial: oow = oow | env.requested_initial_oow
    if isinstance(ledger, BudgetedLearnedStopAttempt):
        ids = ledger.update(step, cpu(env.grasp_q_parallel_values), cpu(oow),
                            cpu(env.invalid_state()),
                            travelled_m=cpu(env.evaluation_tcp_arc),
                            reserve_next_m=ledger.settings['primitive_length_m'],
                            stop=stop)
        if len(ids): env.freeze(torch.as_tensor(ids, device=env.device))
        return ids
    if isinstance(ledger, TeacherRelativeRolloutAttempt):
        ids = ledger.update(step, cpu(env.grasp_q_parallel_values), cpu(oow),
                            cpu(env.invalid_state()),
                            travelled_m=cpu(env.evaluation_tcp_arc),
                            reserve_next_m=ledger.settings['primitive_length_m'])
        if len(ids): env.freeze(torch.as_tensor(ids, device=env.device))
        return ids
    if isinstance(ledger, TeacherRelativeBudgetAttempt):
        ids = ledger.update(step, cpu(env.grasp_q_parallel_values), cpu(oow),
                            cpu(env.invalid_state()),
                            travelled_m=cpu(env.evaluation_tcp_arc),
                            reserve_next_m=ledger.settings['primitive_length_m'])
        if len(ids): env.freeze(torch.as_tensor(ids, device=env.device))
        return ids
    extra = dict(stop=stop) if isinstance(ledger, LearnedStopAttempt) else {}
    ids = ledger.update(step, cpu(env.grasp_q_parallel_values), cpu(oow),
                        cpu(env.invalid_state()), exhausted, **extra)
    if len(ids): env.freeze(torch.as_tensor(ids, device=env.device))
    return ids


def begin(player, initial, permutations):
    obs = player.obs_to_torch(player.env.begin_attempt(initial, permutations))
    player.get_batch_size(obs, 1)
    if getattr(player, "is_rnn", False): player.init_rnn()
    return obs


def nominal_rollout(player, initial, permutations, horizon):
    env = player.env
    obs = begin(player, initial, permutations)
    ledger = FirstAttempt(env.num_envs, horizon)
    resets = cpu(env.explicit_resets)
    initial_snapshot=dict(block_state=cpu(env.block_state).tolist(),eef=cpu(env.gripper_pos).tolist(),
        joints=cpu(env.ur5e_dof_pos).tolist(),q=cpu(env.grasp_q_parallel_values).tolist())
    xy0, obj0 = cpu(env.gripper_pos[:,:2]), cpu(env.block_state[:,:,:2])
    q0 = cpu(env.grasp_q_parallel_values)
    plans = [dict(xy=[xy0[i].tolist()], objects=[obj0[i].tolist()],
                  graspability=[float(q0[i])], actions=[])
             for i in range(env.num_envs)]
    update_ledger(ledger, env, 0, initial=True)
    for step in range(1, horizon+1):
        live = ledger.live.copy()
        if not live.any(): break
        if cpu(env.plan_active)[live].any():
            raise RuntimeError("Nominal decision arrived before the previous primitive completed")
        action = player.get_action(obs, is_deterministic=True).reshape(-1)
        av = cpu(action)
        for i in np.flatnonzero(live): plans[i]["actions"].append(int(av[i]))
        obs, _, _, _ = player.env_step(env, action)
        xy, obj = cpu(env.gripper_pos[:,:2]), cpu(env.block_state[:,:,:2])
        q = cpu(env.grasp_q_parallel_values)
        for i in np.flatnonzero(live):
            plans[i]["xy"].append(xy[i].tolist())
            plans[i]["objects"].append(obj[i].tolist())
            plans[i]["graspability"].append(float(q[i]))
        update_ledger(ledger, env, step)
        zero_retired_teacher(player, ledger.live)
    if not np.array_equal(resets, cpu(env.explicit_resets)):
        raise RuntimeError("Reset during nominal plan generation")
    return plans, ledger, initial_snapshot


def run_case(player, case, initial, permutations, plans, rows, spec, out_dir):
    env, device = player.env, player.device
    n, horizon = env.num_envs, int(spec.get("horizon",120))
    seed = int(spec.get("seed",7))
    actor = case["actor"]
    if actor not in ("student", "teacher", "replay", "hold"): raise ValueError(actor)
    collect = bool(case.get("collect", False))
    stopping = execution_stopping(actor, case.get('stopping'), collect,
                                  case.get('diagnostic_oracle',False))
    stop_threshold=float(case.get('stop_threshold',.9))
    if not 0<stop_threshold<1: raise ValueError('Invalid predicted-score threshold')
    visibility_config = visibility_settings(case, horizon)
    p_drop, bl = visibility_config["p_drop"], visibility_config["blackout_len"]
    arm_occlusion = SimulatorArmOcclusion(env)
    age_mode = case.get("age_mode", "current")
    if age_mode not in ("current", "lagged"): raise ValueError(age_mode)
    rngs = [np.random.default_rng(scene_seed(r["sha256"],seed,"visibility")) for r in rows]
    blackout_schedule = case.get("blackout_schedule", "horizon_uniform")
    blackouts = [ov.blackout_start(blackout_schedule, r["sha256"], seed, bl, horizon, len(p["actions"]), rng)
                 for r, p, rng in zip(rows, plans, rngs)]
    net = mask = stop_net = stop_payload = None
    checkpoint = checkpoint_hash = None
    stop_checkpoint = stop_checkpoint_hash = None
    if actor == "student":
        net, mask, n_out, checkpoint, checkpoint_hash = load_student(case, device)
        if case.get('policy_stop_checkpoint'):
            stop_net, stop_payload, stop_checkpoint, stop_checkpoint_hash = load_policy_stop(
                case, device, checkpoint_hash)
            calibrated_threshold = float(stop_payload['threshold'])
            if ('stop_threshold' in case
                    and abs(float(case['stop_threshold']) - calibrated_threshold) > 1e-12):
                raise ValueError('Runtime threshold differs from policy-stop calibration')
            stop_threshold = calibrated_threshold
        if stopping=='learned_score' and net.stop is None and stop_net is None and not case.get('image_grasp_checkpoint'):
            raise ValueError('Learned stopping requires a trained graspability head')
        if stop_net is not None and (stopping != 'learned_score' or net.stop is not None
                                     or case.get('image_grasp_checkpoint')):
            raise ValueError('The separate policy stop model is the sole learned stopping source')
    constraints = getattr(net,'plan_constraints',None)
    cartesian = getattr(net,'cartesian_policy',None)
    cartesian_recovery_actor = case.get('cartesian_recovery_actor', 'student')
    if cartesian_recovery_actor not in ('student', 'teacher'):
        raise ValueError('Unknown Cartesian recovery actor')
    if cartesian_recovery_actor == 'teacher':
        if (actor != 'student' or cartesian is None or collect
                or stopping != 'oracle_graspability' or not case.get('diagnostic_oracle')):
            raise ValueError('Teacher recovery is a non-collection Cartesian simulator-oracle diagnostic')
    extension=case.get('plan_extension')
    if extension is not None:
        from isaacgymenvs.open_loop.plan_constraint import extension_config
        if constraints is None or extension!=extension_config(extension['max_steps'],extension['max_length_m'],extension['step_length_m'],extension.get('shorten_to_bounds',False)):
            raise ValueError('Extension requires a constrained checkpoint and explicit validated limits')
    if cartesian is not None and extension is not None:
        raise ValueError('Cartesian recovery limits are checkpoint-bound; omit legacy plan_extension')
    teacher_relative_budget = case.get('teacher_relative_budget')
    if teacher_relative_budget is not None:
        teacher_relative_budget = validate_teacher_relative_budget(teacher_relative_budget)
        oracle_mode = stopping == 'oracle_graspability' and case.get('diagnostic_oracle')
        learned_mode = stopping == 'learned_score' and stop_net is not None
        rollout_mode = (stopping == 'budget_rollout' and collect
                        and case.get('diagnostic_oracle'))
        rollout_mode = rollout_mode or (stopping == 'budget_then_grasp' and not collect
                                        and case.get('diagnostic_oracle'))
        if (actor != 'student' or not (oracle_mode or learned_mode or rollout_mode)
                or constraints is not None or cartesian is not None):
            raise ValueError('Teacher-relative action budgets require an unbounded categorical '
                             'student with simulator training collection, explicit oracle '
                             'scoring, or its policy-matched learned stopping model')
        checkpoint_budget=getattr(net,'teacher_relative_budget',None)
        if checkpoint_budget != teacher_relative_budget and not case.get('diagnostic_budget_wrapper',False):
            raise ValueError('Runtime teacher-relative budget differs from checkpoint contract')
        if stop_payload is not None and stop_payload['teacher_relative_budget'] != teacher_relative_budget:
            raise ValueError('Stopping checkpoint budget differs from the action policy')
    elif getattr(net,'teacher_relative_budget',None) is not None:
        raise ValueError('Budget-aware checkpoint requires its teacher-relative runtime limits')
    bounded_collection_protocol = None
    cartesian_collection_protocol = None
    if constraints is not None and stopping != 'oracle_graspability':
        raise ValueError('Constrained policy is simulator-oracle only; no learned/hardware stopping')
    if constraints is not None and collect:
        if extension is None or actor != 'student' or not case.get('diagnostic_oracle'):
            raise ValueError('Bounded DAgger requires student oracle collection with explicit extension limits')
        bounded_collection_protocol = 'bounded_collection_v1'
    if cartesian is not None and collect:
        if actor != 'student' or stopping != 'oracle_graspability' or not case.get('diagnostic_oracle'):
            raise ValueError('Cartesian DAgger collection requires explicit simulator-oracle stopping')
        cartesian_collection_protocol = 'teacher-relative-cartesian-collection-v1'
    if constraints is not None and abs(float(env.cfg['env']['pushDistanceM'])-.04)>1e-9:
        raise ValueError('Constrained decoder assumes existing 4-cm primitives')
    P = torch.tensor(primitive_vectors(float(env.cfg["env"]["pushDistanceM"])), device=device)
    obs = begin(player, initial, permutations)
    if teacher_relative_budget is not None:
        env.enable_evaluation_tcp_arc()
    resets = cpu(env.explicit_resets)
    initial_state, initial_eef = cpu(env.block_state), cpu(env.gripper_pos)
    initial_q = cpu(env.grasp_q_parallel_values)
    initial_oow = cpu(env.physical_oow() | env.step_mesh_oow | env.requested_initial_oow)
    budgets = np.asarray([len(p['actions']) for p in plans], dtype=np.int64)
    fixed_plan_length = case.get('fixed_plan_length')
    if fixed_plan_length is not None:
        # Plan-free execution limits: every scene gets the step/travel budget of one fixed
        # nominal length instead of its own teacher plan length.
        if teacher_relative_budget is None or isinstance(fixed_plan_length, bool) or int(fixed_plan_length) != fixed_plan_length \
                or not 0 < int(fixed_plan_length) <= horizon:
            raise ValueError('fixed_plan_length must be a positive integer used with the teacher-relative budget')
        budgets = np.full(len(plans), int(fixed_plan_length), dtype=np.int64)
    if stopping == 'budget_then_grasp':
        tolerance_m = case.get('budget_then_grasp_tolerance_m')
        if teacher_relative_budget is None or tolerance_m is None:
            raise ValueError('budget_then_grasp requires the teacher-relative budget and a travel tolerance')
    ledger = (BudgetThenGraspAttempt(budgets,horizon,teacher_relative_budget,float(case['budget_then_grasp_tolerance_m']))
              if stopping == 'budget_then_grasp' else
              TeacherRelativeRolloutAttempt(budgets,horizon,teacher_relative_budget)
              if teacher_relative_budget is not None and stopping == 'budget_rollout' else
              BudgetedLearnedStopAttempt(budgets,horizon,teacher_relative_budget)
              if teacher_relative_budget is not None and stopping == 'learned_score' else
              TeacherRelativeBudgetAttempt(budgets,horizon,teacher_relative_budget)
              if teacher_relative_budget is not None else
              LearnedStopAttempt(n,horizon) if stopping=='learned_score' else
              FirstAttempt(n,horizon) if stopping in ('expert_graspability','oracle_graspability') else
              NominalBudgetAttempt(budgets,horizon))
    update_ledger(ledger,env,0,initial=True)
    if actor == "hold": env.freeze(torch.arange(n, device=env.device))
    prev = np.full(n,-1,np.int64)
    age = np.zeros((n,so.N_TOKENS),np.float32)
    extension_used=np.zeros(n,np.float64)
    plan_progress=np.zeros(n,np.int64)
    recovery_used=np.zeros(n,np.float64)
    h = None
    stop_h = None
    last_lookup_q=np.full(n,np.nan,np.float32)
    last_lookup_step=np.full(n,-1,np.int64)
    last_lookup_travel=np.zeros(n,np.float64)
    image_observer=None
    if spec.get('grasp_images',{}).get('enabled',False):
        from isaacgymenvs.open_loop.grasp_image_observer import ImageGraspObserver
        image_observer=ImageGraspObserver(env,rows,plans,case,spec,out_dir,
            terminal_failed=(~ledger.live)&(ledger.reason!='success'))
    traces = {k:[] for k in ["q","mesh_oow","center_oow","legacy_corner_oow","block_state","eef",
                             "action","live_before","visibility_world","student_obs"]}
    states = {k:[] for k in ['decision_obs','decision_valid','decision_q','decision_bad',
                             'predicted_graspability','stop_requested','stop_action_values',
                             'stop_eligible','query_budget_before','lookup_context',
                             'predicted_current_q','predicted_q_change',
                             'predicted_near_term_crossing','action_valid',
                             'constraint_reason','commanded_plan_offset','raw_student_action',
                             'action_scale','extension_travel_used','plan_progress',
                             'raw_cartesian_output','executed_cartesian_path','recovery_action',
                             'measured_recovery_travel_used','tcp_travel_m','student_cartesian_output',
                             'cartesian_action_source','teacher_recovery_action']}
    # Keep privileged geometry ONLY in the expert archive, never the student
    # network input. This makes new detection-dropout masks reproducible offline.
    clean_observations, geometric_masks = [], []
    labels, values, teacher_actions = [], [], []
    t0 = time.monotonic()
    # There are T executed actions and T+1 observations. Keeping the endpoint
    # observation supplies positive supervision when the last push succeeds.
    for t in range(horizon+1):
        observed = ledger.live | (ledger.step==t)
        if not observed.any(): break
        tobs, centers, eef = cpu(obs), cpu(env.block_state[:,:,:2]), cpu(env.gripper_pos[:,:2])
        if cartesian is not None:
            from isaacgymenvs.open_loop.teacher_relative_cartesian import advance_progress
            for i in np.flatnonzero(ledger.live):
                plan_progress[i]=advance_progress(eef[i],plans[i]['xy'],plan_progress[i],
                                                  cartesian['progress_window'])
        visible = np.zeros((n,so.N_TOKENS),np.float32)
        geometric_visibility = arm_occlusion.visibility(visibility_config)
        xrows = []
        for i in range(n):
            if observed[i]:
                v = geometric_visibility[i]
                visible[i] = so.apply_random_occlusion(v,rngs[i],p_drop,
                                      blackout=(bl > 0 and blackouts[i] <= t < blackouts[i]+bl))
                next_age = so.step_staleness(age[i],visible[i])
                if age_mode == "current": age[i] = next_age
            obs_progress=int(plan_progress[i]) if cartesian is not None else t
            xrow=aligned_student_obs(so,tobs[i],centers[i],visible[i],age[i],
                permutations[i],prev[i],plans[i]["xy"],obs_progress,plans[i]["actions"],plans[i]["objects"])
            if net is not None and not ov.is_default(net.obs_layout):
                xrow=ov.convert(xrow,plans[i]["xy"],obs_progress,net.obs_layout,plans[i]["objects"])
            xrows.append(xrow)
            if observed[i] and age_mode == "lagged": age[i] = next_age
        xnp = np.stack(xrows)
        if collect:
            clean_observations.append(np.stack([aligned_student_obs(
                so,tobs[i],centers[i],np.ones(so.N_TOKENS),np.zeros(so.N_TOKENS),
                permutations[i],prev[i],plans[i]['xy'],int(plan_progress[i]) if cartesian is not None else t,
                plans[i]['actions'],plans[i]['objects'])
                for i in range(n)]))
            geometric_masks.append(geometric_visibility.copy())
        bad=cpu(env.invalid_state() | env.physical_oow() | env.step_mesh_oow)
        if t==0: bad |= initial_oow
        q=cpu(env.grasp_q_parallel_values)
        image_score=None
        if image_observer is not None:
            image_score=image_observer.observe(t,observed,prev,eef,q,bad)
        score=np.full(n,np.nan,np.float32)
        policy_stop_choice=np.zeros(n,bool)
        stop_action_values=np.full((n,2),np.nan,np.float32)
        stop_eligible=(ledger.live & (ledger.remaining_queries > 0)
                       if isinstance(ledger, BudgetedLearnedStopAttempt)
                       else ledger.live.copy())
        query_budget_before=(ledger.remaining_queries.copy()
                             if isinstance(ledger, BudgetedLearnedStopAttempt)
                             else np.full(n,-1,np.int64))
        lookup_context=np.zeros((n,6),np.float32)
        has_lookup=np.isfinite(last_lookup_q)
        current_travel=cpu(env.evaluation_tcp_arc) if teacher_relative_budget is not None else np.zeros(n)
        if isinstance(ledger,BudgetedLearnedStopAttempt):
            lookup_context[:,4]=np.minimum(t/np.maximum(ledger.step_limit,1),1.)
            lookup_context[:,5]=np.minimum(
                current_travel/np.maximum(ledger.length_limit_m,1e-6),1.)
        if has_lookup.any():
            lookup_context[has_lookup,0]=1.
            lookup_context[has_lookup,1]=last_lookup_q[has_lookup]
            lookup_context[has_lookup,2]=np.minimum(
                (t-last_lookup_step[has_lookup])/np.maximum(ledger.step_limit[has_lookup],1),1.)
            lookup_context[has_lookup,3]=np.minimum(
                (current_travel[has_lookup]-last_lookup_travel[has_lookup])
                /np.maximum(ledger.length_limit_m[has_lookup],1e-6),1.)
        predicted_current_q=np.full(n,np.nan,np.float32)
        predicted_q_change=np.full(n,np.nan,np.float32)
        predicted_near_term_crossing=np.full(n,np.nan,np.float32)
        stop=np.zeros(n,bool)
        constraint_reason=np.full(n,'',dtype='U24')
        commanded_offset=np.full(n,np.nan,np.float32)
        raw_student_action=np.full(n,-1,np.int64)
        action_scale=np.ones(n,np.float32)
        extension_length=np.zeros(n,np.float64)
        raw_cartesian=np.full((n,n_out if actor=='student' else 1),np.nan,np.float32)
        student_cartesian=np.full((n,n_out if actor=='student' else 1),np.nan,np.float32)
        executed_cartesian=np.full((n,2,2),np.nan,np.float32)
        recovery_action=np.zeros(n,bool)
        cartesian_action_source=np.full(n,'',dtype='U24')
        teacher_recovery_action=np.full(n,-1,np.int64)
        if collect or actor=="teacher" or cartesian_recovery_actor=='teacher':
            logits, value = teacher_forward(player,obs)
            ta = logits.argmax(-1)
            if collect:
                labels.append(cpu(logits)); values.append(cpu(value)); teacher_actions.append(cpu(ta))
        if actor=="student":
            # The stored decision observation stays the released 166-D row; only the
            # network input of the initial-scene control is substituted.
            xnet=(np.stack([ov.initial_scene_context(xnp[i],plans[i]["objects"]) for i in range(n)])
                  if getattr(net,"initial_scene",False) else xnp)
            x=torch.tensor(xnet,device=device).unsqueeze(0)
            if mask is not None: x=x*mask
            with torch.no_grad():
                if net.stop is not None:
                    out,_,stop_logit,h=net.forward_with_stop(x,h)
                    score=cpu(stop_logit[0].sigmoid())
                else: out,_,h=net(x,h)
                if stop_net is not None:
                    if stop_payload.get('lookup_feedback') == 'numeric_q':
                        context_tensor=torch.as_tensor(lookup_context,device=device).unsqueeze(0)
                        stop_values,stop_auxiliary,stop_h=stop_net(x,context_tensor,stop_h)
                        predicted_current_q=cpu(stop_auxiliary[0,:,0])
                        predicted_q_change=cpu(stop_auxiliary[0,:,1])
                        predicted_near_term_crossing=cpu(stop_auxiliary[0,:,2].sigmoid())
                    else:
                        stop_values,stop_h=stop_net(x,stop_h)
                    all_stop_values=cpu(stop_values[0]).astype(np.float32)
                    selected_budget=np.clip(query_budget_before-1,0,all_stop_values.shape[1]-1)
                    stop_action_values=all_stop_values[np.arange(n),selected_budget]
                    score=cpu(torch.softmax(torch.as_tensor(stop_action_values),-1)[:,1])
                    policy_stop_choice=stop_action_values[:,1]>stop_action_values[:,0]
            if image_score is not None:score=image_score
            if stopping=='learned_score':
                if not np.isfinite(score[ledger.live]).all(): raise RuntimeError('Non-finite stop prediction')
                stop=((policy_stop_choice if stop_net is not None else score>stop_threshold)
                      & stop_eligible)
                update_ledger(ledger,env,t,stop=stop)
                if stop_payload is not None and stop_payload.get('lookup_feedback') == 'numeric_q':
                    negative=stop & (q<=ledger.threshold) & ledger.live
                    last_lookup_q[negative]=q[negative]
                    last_lookup_step[negative]=t
                    last_lookup_travel[negative]=current_travel[negative]
            if cartesian is not None:
                endpoint=out[0][...,-2:]
                action=torch.cdist(endpoint,P).argmin(-1)
                raw_cartesian=cpu(out[0]).astype(np.float32)
                student_cartesian=raw_cartesian.copy()
                teacher_action_np=cpu(ta).astype(np.int64) if cartesian_recovery_actor=='teacher' else None
                from isaacgymenvs.open_loop.teacher_relative_cartesian import (
                    controller_proposal, decode as decode_cartesian)
                for i in np.flatnonzero(ledger.live):
                    raw_cartesian[i],cartesian_action_source[i]=controller_proposal(
                        student_cartesian[i],
                        teacher_action_np[i] if teacher_action_np is not None else -1,
                        plan_progress[i],len(plans[i]['actions']),cartesian,
                        cartesian_recovery_actor)
                    if cartesian_action_source[i]=='teacher_recovery':
                        teacher_recovery_action[i]=teacher_action_np[i]
                        action[i]=int(teacher_action_np[i])
                    path,details=decode_cartesian(raw_cartesian[i],eef[i],plans[i],plan_progress[i],
                                                  recovery_used[i],cartesian)
                    if path is None:
                        constraint_reason[i]=details['reason']
                        ledger.reason[i]=details['reason'];ledger.step[i]=t;ledger.terminal_q[i]=q[i]
                        env.freeze(torch.as_tensor([i],device=env.device))
                        action[i]=0
                    else:
                        executed_cartesian[i]=path
                        recovery_action[i]=details['recovery']
                        commanded_offset[i]=details['commanded_max_teacher_residual_m']
            else:
                action=torch.cdist(out[0],P).argmin(-1) if n_out==2 else out[0].argmax(-1)
            raw_student_action=cpu(action).astype(np.int64)
            if constraints is not None:
                from isaacgymenvs.open_loop.plan_constraint import select as constrained_select
                logits_np=cpu(out[0])
                for i in np.flatnonzero(ledger.live):
                    chosen,details=constrained_select(logits_np[i],eef[i],plans[i],t,constraints,extension,extension_used[i])
                    if chosen < 0:
                        constraint_reason[i]=details['reason']
                        ledger.reason[i]=details['reason'];ledger.step[i]=t;ledger.terminal_q[i]=q[i]
                        env.freeze(torch.as_tensor([i],device=env.device))
                        action[i]=0  # frozen environment; never a scored action
                    else:
                        action[i]=chosen
                        commanded_offset[i]=details['commanded_max_offset_m']
                        action_scale[i]=details.get('action_scale',1.)
                        extension_length[i]=details.get('extension_length_m',0.)
        elif actor=="teacher": action=ta
        else:
            av=[p['actions'][t] if actor=='replay' and t<len(p['actions']) else 0 for p in plans]
            action=torch.tensor(av,device=env.device)
        live=ledger.live.copy()
        for key,value in dict(decision_obs=xnp,decision_valid=observed,decision_q=q,decision_bad=bad,
                              predicted_graspability=score,stop_requested=stop,
                              stop_action_values=stop_action_values,stop_eligible=stop_eligible,
                              query_budget_before=query_budget_before,
                              lookup_context=lookup_context,
                              predicted_current_q=predicted_current_q,
                              predicted_q_change=predicted_q_change,
                              predicted_near_term_crossing=predicted_near_term_crossing,
                              action_valid=live,
                              constraint_reason=constraint_reason,commanded_plan_offset=commanded_offset,
                              raw_student_action=raw_student_action,action_scale=action_scale,
                              extension_travel_used=extension_used.copy(),plan_progress=plan_progress.copy(),
                              raw_cartesian_output=raw_cartesian,executed_cartesian_path=executed_cartesian,
                              recovery_action=recovery_action,
                              measured_recovery_travel_used=recovery_used.copy(),
                              tcp_travel_m=cpu(env.evaluation_tcp_arc),
                              student_cartesian_output=student_cartesian,
                              cartesian_action_source=cartesian_action_source,
                              teacher_recovery_action=teacher_recovery_action).items():
            states[key].append(value)
        if not live.any(): break
        if t>=horizon: raise RuntimeError('Live episode reached hard cap without retiring')
        if actor!='hold' and cpu(env.plan_active)[live].any():
            raise RuntimeError('Execution decision would be discarded by an unfinished primitive')
        av=cpu(action).reshape(n)
        traces['student_obs'].append(xnp)
        traces['visibility_world'].append(visible)
        traces['action'].append(np.where(live,av,-1))
        traces['live_before'].append(live)
        zero_retired_teacher(player,ledger.live)
        if h is not None: h[:,~torch.as_tensor(ledger.live,device=h.device)]=0
        if extension is not None:env.set_evaluation_action_scale(action_scale)
        arc_before=cpu(env.evaluation_tcp_arc) if cartesian is not None else None
        if cartesian is not None:
            env.set_evaluation_cartesian_paths(np.nan_to_num(executed_cartesian),live)
        obs,_,_,_=player.env_step(env,action.to(env.device))
        extension_used[live]+=extension_length[live]
        if cartesian is not None:
            measured=cpu(env.evaluation_tcp_arc)-arc_before
            recovery_used[live & recovery_action]+=measured[live & recovery_action]
        traces['q'].append(cpu(env.grasp_q_parallel_values))
        traces['mesh_oow'].append(cpu(env.physical_oow() | env.step_mesh_oow))
        traces['center_oow'].append(cpu(env.oow_violation()))
        _,bdist=env._phi_terms()
        traces['legacy_corner_oow'].append(cpu(((bdist<0)&~env.oow_exempt.T).any(dim=0)))
        traces['block_state'].append(cpu(env.block_state))
        traces['eef'].append(cpu(env.gripper_pos))
        update_ledger(ledger,env,t+1)
        prev[live]=av[live]
        if not np.array_equal(resets,cpu(env.explicit_resets)):
            raise RuntimeError('Reset inside a scored attempt')
        if (t+1)%30==0:
            print(f"[{case['name']}] step={t+1} active={ledger.live.sum()} success={(ledger.reason=='success').sum()}/{n}",flush=True)
    elapsed=time.monotonic()-t0
    arrays={k:np.stack(v) if v else np.empty((0,n),np.float32) for k,v in traces.items()}
    arrays['live_before']=arrays['live_before'].astype(bool)
    arrays['action']=arrays['action'].astype(np.int64)
    if not traces['action']:
        arrays['student_obs']=np.empty((0,n,ov.obs_dim(getattr(net,'obs_layout',None))),np.float32)
        arrays['visibility_world']=np.empty((0,n,so.N_TOKENS),np.float32)
    arrays.update({k:np.stack(v) for k,v in states.items()})
    arrays.update(initial_state=initial_state,initial_eef=initial_eef,initial_q=initial_q,
        initial_oow=initial_oow,requested_initial_state=cpu(initial),permutation=permutations,step_budget=budgets)
    trace_path=out_dir/(case['name']+'.npz')
    np.savez_compressed(trace_path,**arrays)
    valid=arrays['live_before']
    per_scene=report_attempt(ledger,rows)
    for i,row in enumerate(per_scene):
        row.update(step_budget=int(budgets[i]),initial_q=float(initial_q[i]),initial_oow=bool(initial_oow[i]),
            stop_score=float(arrays['predicted_graspability'][ledger.step[i],i]) if net is not None and (net.stop is not None or stop_net is not None or (image_observer is not None and image_observer.net is not None)) else None,
            graspable_before_stop=bool(np.any((arrays['decision_q'][:,i]>.9)&arrays['decision_valid'][:,i]
                                             &(np.arange(len(states['decision_q']))<ledger.step[i]))),
            ever_oracle_eligible_graspable=bool(np.any(
                (arrays['decision_q'][:,i]>.9)&arrays['decision_valid'][:,i]
                &~arrays['decision_bad'][:,i])),
            blackout_start=blackouts[i],
            visible_fraction=float(arrays['visibility_world'][:,i][valid[:,i]].mean()) if valid[:,i].any() else None,
            blackout_steps_exposed=sum(1 for t in range(len(valid)) if valid[t,i] and bl>0 and blackouts[i]<=t<blackouts[i]+bl),
            final_plan_progress=int(arrays['plan_progress'][-1,i]) if cartesian is not None else None,
            measured_recovery_travel_m=float(arrays['measured_recovery_travel_used'][:,i].max()) if cartesian is not None else None,
            max_commanded_teacher_residual_m=float(np.nanmax(arrays['commanded_plan_offset'][:,i]))
                if cartesian is not None and np.isfinite(arrays['commanded_plan_offset'][:,i]).any() else None,
            explicit_resets_during_rollout=int(cpu(env.explicit_resets)[i]-resets[i]))
        row['query_timing_failure'] = bool(
            not row['success'] and row['ever_oracle_eligible_graspable'])
    if image_observer is not None:image_observer.finish(per_scene)
    if collect:
        data_dir=out_dir/'collection';data_dir.mkdir(exist_ok=True)
        stop_valid=arrays['decision_valid'] & np.isfinite(arrays['decision_q']) & np.isfinite(arrays['decision_obs']).all(-1)
        stop_target=np.where(arrays['decision_bad'],0,np.clip(arrays['decision_q'],0,1))
        stop_target=np.nan_to_num(stop_target)
        collection_payload=dict(obs=np.nan_to_num(arrays['decision_obs'],nan=0,posinf=0,neginf=0),
            teacher_logits=np.nan_to_num(np.stack(labels),nan=0,posinf=0,neginf=0),
            teacher_value=np.nan_to_num(np.stack(values),nan=0,posinf=0,neginf=0),
            valid=action_supervision_mask(arrays['action_valid'],arrays['decision_q'],arrays['decision_bad']).astype(np.float32),
            action_executed=arrays['action_valid'].astype(np.float32),actor=np.array(actor),
            graspability=stop_target,stop_valid=stop_valid.astype(np.float32),
            scene_hash=np.array([r['sha256'] for r in rows]),teacher_action=np.stack(teacher_actions),
            step_budget=budgets,terminal_step=ledger.step,terminal_reason=ledger.reason,
            stopping=np.array(stopping),execution_limit=np.array(horizon),protocol=np.array(PROTOCOL),
            diagnostic_oracle=np.array(bool(case.get('diagnostic_oracle',False))),
            capture_schema=np.array('clean-observation-and-arm-mask-v1'),
            clean_obs=np.stack(clean_observations),geometric_visibility_world=np.stack(geometric_masks),
            decision_observed=arrays['decision_valid'],permutation=permutations,
            visibility_seed=np.array(seed),age_mode=np.array(age_mode),
            visibility_config=np.array(json.dumps(visibility_config,sort_keys=True)))
        if teacher_relative_budget is not None:
            collection_payload.update(
                teacher_relative_budget_protocol=np.array(teacher_relative_budget['protocol']),
                teacher_relative_budget=np.array(json.dumps(teacher_relative_budget,sort_keys=True)),
                tcp_travel_m=arrays['tcp_travel_m'])
        if bounded_collection_protocol is not None:
            collection_payload.update(
                bounded_collection_protocol=np.array(bounded_collection_protocol),
                plan_constraints=np.array(json.dumps(constraints,sort_keys=True)),
                plan_extension=np.array(json.dumps(extension,sort_keys=True)),
                executed_action_scale=arrays['action_scale'],
                extension_travel_used=arrays['extension_travel_used'])
        if cartesian_collection_protocol is not None:
            collection_payload.update(
                cartesian_collection_protocol=np.array(cartesian_collection_protocol),
                cartesian_policy=np.array(json.dumps(cartesian,sort_keys=True)),
                raw_cartesian_output=arrays['raw_cartesian_output'],
                executed_cartesian_path=arrays['executed_cartesian_path'],
                plan_progress=arrays['plan_progress'],recovery_action=arrays['recovery_action'],
                measured_recovery_travel_used=arrays['measured_recovery_travel_used'])
        np.savez_compressed(data_dir/(case['name']+'.npz'),**collection_payload)
    result=dict(protocol=PROTOCOL,complete=True,case=case,seed=seed,termination=stopping,
        plan_constraints=constraints,plan_extension=extension,
        teacher_relative_budget=teacher_relative_budget,
        bounded_collection_protocol=bounded_collection_protocol,
        cartesian_policy=cartesian,cartesian_collection_protocol=cartesian_collection_protocol,
        cartesian_recovery_actor=cartesian_recovery_actor,
        visibility_config=visibility_config,blackout_schedule=blackout_schedule,
        obs_layout=getattr(net,'obs_layout',None),plan_drop=getattr(net,'plan_drop',None),
        stop_threshold=stop_threshold,success_scored_at='final_state_only',simulated_retraction=False,
        privileged_stopping=stopping in ('expert_graspability','oracle_graspability'),
        diagnostic_only=stopping in ('oracle_graspability','budget_rollout','budget_then_grasp'),
        checkpoint=checkpoint,checkpoint_sha256=checkpoint_hash,
        policy_stop_checkpoint=stop_checkpoint,policy_stop_checkpoint_sha256=stop_checkpoint_hash,
        policy_stop_protocol=stop_payload['protocol'] if stop_payload is not None else None,
        stop_decision_rule=stop_payload.get('decision_rule') if stop_payload is not None else None,
        max_queries=stop_payload.get('max_queries') if stop_payload is not None else None,
        lookup_feedback=stop_payload.get('lookup_feedback') if stop_payload is not None else None,
        unique_scenes=n,horizon=horizon,age_mode=age_mode,rows=per_scene,
        success_pct=100*float((ledger.reason=='success').mean()),
        reasons={str(r):int((ledger.reason==r).sum()) for r in np.unique(ledger.reason)},
        initial_graspable=int((initial_q>.9).sum()),initial_oow=int(initial_oow.sum()),elapsed_rollout_seconds=elapsed,
        trace=str(trace_path),trace_sha256=sha256(trace_path),source_snapshot='provenance.json')
    initial_success=int(((ledger.reason=='success')&(ledger.step==0)).sum())
    result['initial_successes']=initial_success
    if case.get('image_grasp_checkpoint'):
        result['image_grasp_checkpoint_sha256']=sha256(case['image_grasp_checkpoint'])
    result['success_after_action_pct']=100*int(((ledger.reason=='success')&(ledger.step>0)).sum())/max(n-initial_success,1) if n>initial_success else None
    (out_dir/(case['name']+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(f"RESULT {case['name']}: {result['success_pct']:.2f}% {result['reasons']} time={elapsed:.1f}s",flush=True)
    return initial_state


@hydra.main(config_name="config",config_path="../cfg")
def main(cfg: DictConfig):
    from isaacgymenvs.utils.reformat import omegaconf_to_dict
    from isaacgymenvs.utils.utils import set_seed
    from rl_games.torch_runner import Runner
    from isaacgymenvs.utils.rlgames_utils import RLGPUEnv,RLGPUAlgoObserver
    from rl_games.common import env_configurations,vecenv
    import isaacgymenvs
    spec_path = Path(to_absolute_path(str(cfg.evaluation_spec)))
    spec = json.loads(spec_path.read_text())
    if spec.get('grasp_images',{}).get('enabled',False):
        OmegaConf.update(cfg,'task.env.graspImages.enabled',True,force_add=True)
    elif any(c.get('image_grasp_checkpoint') for c in spec['cases']):
        raise ValueError('Image classifier requires grasp_images.enabled')
    if len(spec["cases"])>1 and not spec.get("diagnostic_reuse_simulator",False):
        raise ValueError("Use one actor per fresh simulator; reuse is diagnostic only")
    manifest = read_manifest(spec["manifest"])
    offset, size = int(spec.get("offset",0)), int(spec.get("batch_size",64))
    rows = manifest["scenes"][offset:offset+size]
    if not rows: raise ValueError("Empty manifest batch")
    out_dir = Path(spec["output"]).resolve()
    out_dir.mkdir(parents=True,exist_ok=True)
    if (out_dir/"provenance.json").exists(): raise FileExistsError("Use a fresh output directory")
    scene_dir = out_dir/"scenes"
    scene_dir.mkdir()
    for i,row in enumerate(rows): shutil.copyfile(row["path"],scene_dir/f"{i:06d}.txt")
    cfg.num_envs = len(rows)
    cfg.task.env.numEnvs = len(rows)
    cfg.task.env.test_cases.scene_root_dir = os.path.relpath(scene_dir.parent,ROOT)
    cfg.task.env.test_cases.difficulty_choice = scene_dir.name
    cfg.task.env.test_cases.sceneOffset = 0
    if cfg.task.name != "MoreEvaluation": raise ValueError("Requires task=MoreEvaluation")
    cfg.seed=int(spec.get("seed",7))
    torch.cuda.set_device(torch.device(cfg.rl_device))
    # Fix inference kernel selection. Global deterministic-algorithm mode in
    # torch 1.13 breaks the legacy task's broadcast indexed reset assignment;
    # that assignment uses unique indices, so it needs no atomic reduction.
    set_seed(cfg.seed,False)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_num_threads(int(spec.get("cpu_threads",4)))
    def thunk(**kw):
        return isaacgymenvs.make(cfg.seed,cfg.task_name,cfg.test,cfg.task.env.numEnvs,
            cfg.sim_device,cfg.rl_device,cfg.graphics_device_id,cfg.headless,
            cfg.multi_gpu,cfg.capture_video,cfg.force_render,cfg,**kw)
    vecenv.register("RLGPU",lambda name,num,**kw:RLGPUEnv(name,num,**kw))
    env_configurations.register("rlgpu",dict(vecenv_type="RLGPU",env_creator=thunk))
    runner=Runner(RLGPUAlgoObserver());runner.load(omegaconf_to_dict(cfg.train));runner.reset()
    player=runner.create_player();player.restore(to_absolute_path(cfg.checkpoint))
    env=player.env
    if env.num_envs != len(rows): raise RuntimeError("Scene/slot count mismatch")
    tracked = ["tools/evaluate_retrieval.py","open_loop/evaluation_core.py","open_loop/student_obs.py","open_loop/obs_variants.py",
        "open_loop/arm_occlusion.py","open_loop/teacher_relative_cartesian.py",
        "tasks/more_evaluation.py","tasks/more_open_loop.py","tasks/more_teacher.py",
        "tasks/more_robust.py","tasks/more.py","tasks/base/vec_task.py",
        "learning/student_net.py","learning/student_ablate.py","learning/policy_stop.py"]
    if spec.get('grasp_images',{}).get('enabled',False):
        tracked += ['open_loop/grasp_image_observer.py','open_loop/grasp_image_obs.py','learning/recurrent_grasp.py']
    # Freeze geometry assets too: changing a mesh changes the observation model.
    tracked += [os.path.relpath(p, ROOT) for p in SimulatorArmOcclusion(env).assets]
    provenance=dict(protocol=PROTOCOL,spec=spec,manifest_sha256=sha256(spec["manifest"]),
        image_grasp_checkpoint_sha256={c['name']:sha256(c['image_grasp_checkpoint']) for c in spec['cases'] if c.get('image_grasp_checkpoint')},
        policy_stop_checkpoint_sha256={c['name']:sha256(c['policy_stop_checkpoint']) for c in spec['cases'] if c.get('policy_stop_checkpoint')},
        scene_rows=rows,source_sha256={p:sha256(ROOT/p) for p in tracked},
        teacher_sha256=sha256(to_absolute_path(cfg.checkpoint)),
        configuration=OmegaConf.to_container(cfg,resolve=True),
        torch_version=torch.__version__,cuda_version=torch.version.cuda,
        deterministic_torch_algorithms=False,deterministic_cudnn=True,cudnn_benchmark=False,tf32=False,
        gpu=torch.cuda.get_device_name(),host=platform.node(),
        workspace=[list(env.ws_x),list(env.ws_y)],containment="all transformed collision-mesh vertices; no exemptions",
        initialization_hold_frames=env.control_freq_inv,
        notes="Expert demonstrations stop at actual execution graspability; students use predicted-score stopping. Nominal length is a separate evaluation baseline. No simulated retraction. No physical grasp-and-lift claim.")
    (out_dir/"provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")
    seed=int(spec.get("seed",7))
    permutations=np.stack([np.random.default_rng(scene_seed(r["sha256"],seed,"permutation")).permutation(10) for r in rows])
    nominal=env.pristine_block_state.clone()
    nominal[:,:,7:]=0
    if spec.get("plan_source"):
        source=Path(spec["plan_source"])
        plan_data=json.loads(source.read_text())
        if [r["scene"]["sha256"] for r in plan_data["rows"]] != [r["sha256"] for r in rows]:
            raise ValueError("Nominal plan scene manifest differs from execution")
        source_provenance=json.loads((source.parent/"provenance.json").read_text())
        if source_provenance["teacher_sha256"]!=provenance["teacher_sha256"]:
            raise ValueError("Nominal teacher checkpoint differs")
        if source_provenance["spec"].get("horizon",120)!=spec.get("horizon",120):
            raise ValueError("Nominal horizon differs")
        provenance["plan_source_sha256"]=sha256(source)
        (out_dir/"provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")
    else:
        plans,plan_ledger,plan_initial=nominal_rollout(player,nominal,permutations,int(spec.get("horizon",120)))
        plan_data=dict(plans=plans,rows=report_attempt(plan_ledger,rows),initial=plan_initial)
    plans=plan_data["plans"]
    (out_dir/"nominal.json").write_text(json.dumps(plan_data,indent=1)+"\n")
    initial=nominal.clone()
    pn,yn=float(spec.get("pos_noise",.015)),float(spec.get("yaw_noise_deg",10.))
    for i,row in enumerate(rows):
        rng=np.random.default_rng(scene_seed(row["sha256"],seed,"pose"))
        delta=rng.uniform(-pn,pn,(env.num_objects,2)).astype(np.float32)
        yaw_delta=rng.uniform(-np.deg2rad(yn),np.deg2rad(yn),env.num_objects)
        if pn>0:
            initial[i,:,:2]+=torch.tensor(delta,device=env.device)
        if yn>0:
            yaw=2*torch.atan2(initial[i,:,5],initial[i,:,6])
            yaw=yaw+torch.tensor(yaw_delta,device=env.device)
            initial[i,:,3:5]=0
            initial[i,:,5],initial[i,:,6]=torch.sin(yaw/2),torch.cos(yaw/2)
    reference=None
    initial_deltas={}
    for case in spec["cases"]:
        if not case["name"].replace("_","").replace("-","").isalnum(): raise ValueError("Unsafe case name")
        actual=run_case(player,case,initial,permutations,plans,rows,spec,out_dir)
        if reference is None: reference=actual
        initial_deltas[case["name"]]=float(np.abs(actual-reference).max())
    summary=dict(complete=True,protocol=PROTOCOL,initial_state_max_abs_difference=initial_deltas)
    torch.cuda.synchronize(env.device)
    if any(sha256(ROOT/p)!=h for p,h in provenance["source_sha256"].items()):
        raise RuntimeError("Source changed during execution; output is not certified")
    if any(sha256(c['image_grasp_checkpoint'])!=provenance['image_grasp_checkpoint_sha256'][c['name']]
           for c in spec['cases'] if c.get('image_grasp_checkpoint')):
        raise RuntimeError('Image grasp checkpoint changed during execution')
    (out_dir/"complete.json").write_text(json.dumps(summary,indent=2)+"\n")
    print("EVALUATION COMPLETE",json.dumps(summary),flush=True)
    # Isaac Gym's old binary bindings can segfault during interpreter teardown
    # on this host after successful runs. Synchronize CUDA and finish all files
    # above, then let process exit release GPU resources without those destructors.
    # Exceptions never reach this success-only exit; the supervisor also verifies
    # complete.json and independently validates the saved terminal traces.
    sys.stdout.flush();sys.stderr.flush()
    os._exit(0)


if __name__=="__main__": main()
