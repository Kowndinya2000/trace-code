"""Fit a separate conservative recurrent stop model for one frozen policy."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from torch.nn import functional as F

from isaacgymenvs.learning.policy_stop import (
    HAZARD_WINDOW_PROTOCOL, MAX_QUERIES, PROTOCOL, Q_ANCHORED_PROTOCOL,
    HazardWindowPolicyStopNet, PolicyStopNet, QAnchoredPolicyStopNet)
from isaacgymenvs.open_loop.evaluation_core import PROTOCOL as DATA_PROTOCOL, read_manifest, sha256
from isaacgymenvs.open_loop.teacher_relative_budget import validate as validate_budget


def write(path, value):
    path = Path(path)
    temporary = Path(str(path) + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def load_sequences(directories, manifest, action_sha256):
    allowed = {row['sha256'] for row in read_manifest(manifest)['scenes']}
    files = sorted({path.resolve() for directory in directories
                    for path in Path(directory).rglob('collection/*.npz')})
    if not files:
        raise ValueError('No stopping collections found')
    sequences = []
    visibility = budget_identity = None
    for path in files:
        result_path = path.parent.parent / (path.stem + '.json')
        result = json.loads(result_path.read_text())
        if result.get('checkpoint_sha256') != action_sha256:
            raise ValueError('Collection used a different action checkpoint: ' + str(path))
        with np.load(path) as data:
            stopping = str(data['stopping'])
            if (str(data['protocol']) != DATA_PROTOCOL or str(data['actor']) != 'student'
                    or stopping not in ('oracle_graspability', 'budget_rollout')
                    or not bool(data['diagnostic_oracle'])):
                raise ValueError('Stopping data must be a frozen-policy simulator collection')
            current_visibility = str(data['visibility_config'])
            current_budget = str(data['teacher_relative_budget'])
            if visibility is not None and current_visibility != visibility:
                raise ValueError('Mixed visibility configurations')
            if budget_identity is not None and current_budget != budget_identity:
                raise ValueError('Mixed teacher-relative budget configurations')
            visibility, budget_identity = current_visibility, current_budget
            obs = data['obs']
            q = data['graspability']
            tcp_travel = data['tcp_travel_m'] if 'tcp_travel_m' in data else np.zeros_like(q)
            valid = data['stop_valid'].astype(bool)
            hashes = data['scene_hash']
            terminal_step = data['terminal_step']
            terminal_reason = data['terminal_reason']
            if (obs.shape[:2] != q.shape or q.shape != valid.shape or tcp_travel.shape != q.shape
                    or obs.shape[-1] != 166 or len(hashes) != obs.shape[1]):
                raise ValueError('Invalid stopping collection shapes')
            for index, scene in enumerate(hashes.tolist()):
                if scene not in allowed:
                    raise ValueError('Stopping data contains a non-training scene')
                locations = np.flatnonzero(valid[:, index])
                if not len(locations):
                    continue
                if not np.array_equal(locations, np.arange(locations[-1] + 1)):
                    raise ValueError('Stop-valid observations must form a contiguous prefix')
                n = locations[-1] + 1
                current_obs = obs[:n, index].copy()
                current_q = q[:n, index].copy()
                current_travel = tcp_travel[:n, index].copy()
                if (not np.isfinite(current_obs).all() or not np.isfinite(current_q).all()
                        or not np.isfinite(current_travel).all()
                        or (current_q < 0).any() or (current_q > 1).any()):
                    raise ValueError('Invalid stopping observations or labels')
                label = current_q > .9
                if stopping == 'oracle_graspability' and label[:-1].any():
                    raise ValueError('Oracle collection continued after current graspability')
                budget_settings = validate_budget(json.loads(current_budget))
                budget = int(data['step_budget'][index])
                sequences.append(dict(obs=current_obs, q=current_q, tcp_travel=current_travel,
                    label=label, terminal_reason=str(terminal_reason[index]),
                    scene_sha256=scene, source=str(path),
                    terminal_step=int(terminal_step[index]),
                    step_limit=min(budget + budget_settings['extra_steps'], result['horizon']),
                    length_limit_m=(budget * budget_settings['primitive_length_m']
                                    + budget_settings['extra_length_m'])))
    if len({row['scene_sha256'] for row in sequences}) != len(sequences):
        raise ValueError('Each stopping trajectory must have a unique training scene')
    if len(sequences) < 1000:
        raise ValueError('Insufficient policy-specific stopping trajectories')
    return sequences, files, json.loads(visibility), validate_budget(json.loads(budget_identity))


def split_sequences(rows, seed=20260911):
    order = np.arange(len(rows))
    np.random.default_rng(seed).shuffle(order)
    n_cal = max(1, int(.15 * len(order)))
    n_val = max(1, int(.15 * len(order)))
    calibration = [rows[i] for i in order[:n_cal]]
    validation = [rows[i] for i in order[n_cal:n_cal + n_val]]
    training = [rows[i] for i in order[n_cal + n_val:]]
    identities = [set(row['scene_sha256'] for row in part)
                  for part in (training, validation, calibration)]
    if identities[0] & identities[1] or identities[0] & identities[2] or identities[1] & identities[2]:
        raise ValueError('Scene leakage across stopping splits')
    return training, validation, calibration


def optimal_stopping_targets(row, gamma=.99, continue_cost=.01,
                             success_reward=1., failure_reward=-1., query_cost=.05,
                             max_queries=MAX_QUERIES):
    """Bellman values for CONTINUE/QUERY at each remaining query budget.

    Simulator graspability is reduced to the binary terminal reward.  It is
    never a model input or regression target.  STOP is evaluable at every live
    state; CONTINUE uses the transition actually produced by frozen R3.  A
    failed query consumes budget, returns only a negative verdict, and forces
    the next frozen-R3 push.  No counterfactual transition is invented.
    """
    labels = np.asarray(row['label'], dtype=bool)
    n = len(labels)
    if n == 0:
        raise ValueError('Empty stopping trajectory')
    if max_queries != MAX_QUERIES:
        raise ValueError('Unknown query budget')
    values = np.zeros((n, max_queries, 2), np.float32)  # [remaining-1, continue/query]
    available = np.zeros_like(values, dtype=bool)
    available[:-1, :, :] = True
    no_query = np.zeros(n, np.float32)
    no_query[-1] = failure_reward
    for step in range(n - 2, -1, -1):
        no_query[step] = -continue_cost + gamma * no_query[step + 1]
    terminal_reason = row.get('terminal_reason', 'success' if labels[-1] else 'step_timeout')
    if terminal_reason == 'success':
        if not labels[-1]:
            raise ValueError('Successful terminal state lacks positive stop reward')
        values[-1, :, 1] = success_reward - query_cost
        available[-1, :, 1] = True
        next_value = values[-1].max(-1)
    elif terminal_reason in ('step_timeout', 'travel_timeout'):
        # The endpoint is recorded for auditing, but the hard cap has already
        # retired the attempt. Neither a query nor another push is available.
        next_value = np.full(max_queries, failure_reward, np.float32)
    else:
        # The final observation records an already absorbing physical failure;
        # neither meta-action is available there.
        next_value = np.full(max_queries, failure_reward, np.float32)
    for step in range(n - 2, -1, -1):
        values[step, :, 0] = -continue_cost + gamma * next_value
        if labels[step]:
            values[step, :, 1] = success_reward - query_cost
        else:
            values[step, 0, 1] = -query_cost + no_query[step]
            for remaining in range(2, max_queries + 1):
                # Negative lookup, then the mandatory CONTINUE action using
                # one fewer future lookup.
                values[step, remaining - 1, 1] = (
                    -query_cost + values[step, remaining - 2, 0])
        next_value = values[step].max(-1)
    if not np.isfinite(values).all():
        raise ValueError('Invalid optimal-stopping return targets')
    return values, available


def make_batch(rows, device, target_settings=None):
    target_settings = target_settings or {}
    length = max(len(row['obs']) for row in rows)
    obs = torch.zeros(length, len(rows), 166, device=device)
    labels = torch.zeros(length, len(rows), device=device)
    targets = torch.zeros(length, len(rows), MAX_QUERIES, 2, device=device)
    available = torch.zeros_like(targets)
    for index, row in enumerate(rows):
        n = len(row['obs'])
        obs[:n, index] = torch.as_tensor(row['obs'], device=device)
        labels[:n, index] = torch.as_tensor(row['label'], device=device)
        row_targets, row_available = optimal_stopping_targets(row, **target_settings)
        targets[:n, index] = torch.as_tensor(row_targets, device=device)
        available[:n, index] = torch.as_tensor(row_available, device=device)
    return obs, labels, targets, available


def make_q_anchored_batch(rows, device, rng, target_settings=None,
                          lookup_q_noise=.03, hazard_window=False,
                          policy_histories=None, policy_history_probability=.75):
    """Create reachable lookup histories without inventing scene transitions."""
    obs, labels, targets, available = make_batch(rows, device, target_settings)
    shape=tuple(obs.shape[:2])
    contexts=np.zeros((*shape,6),np.float32)
    auxiliary_dim = 9 if hazard_window else 3
    auxiliary_targets=np.zeros((*shape,auxiliary_dim),np.float32)
    auxiliary_available=np.zeros_like(auxiliary_targets)
    budget_mask=np.zeros((*shape,MAX_QUERIES,1),np.float32)
    for index, row in enumerate(rows):
        n = len(row['obs'])
        history = (policy_histories or {}).get(row['scene_sha256'])
        use_policy = history is not None and rng.random() < policy_history_probability
        if use_policy:
            contexts[:n,index] = history['context']
            budget_mask[:n,index] = history['budget_mask']
        else:
            candidates = np.flatnonzero(~row['label'][:-1])
            count = min(int(rng.integers(0, MAX_QUERIES)), len(candidates))
            events = set(rng.choice(candidates, size=count, replace=False).tolist()) if count else set()
            remaining = MAX_QUERIES
            observed_last_q = last_step = last_travel = None
            for step in range(n):
                contexts[step,index,4] = min(step / max(row['step_limit'],1),1.)
                contexts[step,index,5] = min(
                    row['tcp_travel'][step] / max(row['length_limit_m'],1e-6),1.)
                if observed_last_q is not None:
                    contexts[step, index, :4] = [
                        1., observed_last_q,
                        min((step - last_step) / max(row['step_limit'], 1), 1.),
                        min((row['tcp_travel'][step] - last_travel)
                            / max(row['length_limit_m'], 1e-6), 1.)]
                if remaining:
                    budget_mask[step,index,remaining-1]=1.
                if step in events and remaining > 1:
                    observed_last_q = float(np.clip(
                        row['q'][step] + rng.normal(0., lookup_q_noise), 0., 1.))
                    last_step = step
                    last_travel = float(row['tcp_travel'][step])
                    remaining -= 1
        for step in range(n):
            auxiliary_targets[step, index, 0] = float(row['q'][step])
            auxiliary_available[step, index, 0] = 1.
            if contexts[step,index,0] > .5:
                auxiliary_targets[step,index,1] = float(
                    row['q'][step] - contexts[step,index,1])
                auxiliary_available[step,index,1] = 1.
            label = np.asarray(row['label'], dtype=bool)
            auxiliary_targets[step,index,2] = float(label[step:min(step+4,n)].any())
            auxiliary_available[step,index,2] = 1.
            if hazard_window:
                auxiliary_targets[step,index,3] = float(label[step])
                for target_index, horizon in enumerate((1,2,4,8), start=4):
                    auxiliary_targets[step,index,target_index] = float(
                        label[step:min(step+horizon+1,n)].any())
                auxiliary_targets[step,index,8] = float(
                    label[step] and (step + 1 >= n or not label[step + 1]))
                auxiliary_available[step,index,3:] = 1.
    contexts=torch.as_tensor(contexts,device=device)
    auxiliary_targets=torch.as_tensor(auxiliary_targets,device=device)
    auxiliary_available=torch.as_tensor(auxiliary_available,device=device)
    anchored_available=available*torch.as_tensor(budget_mask,device=device)
    return (obs,labels,targets,anchored_available,contexts,
            auxiliary_targets,auxiliary_available)


@torch.no_grad()
def replay_q_anchored_histories(model, rows, device, batch_size=128):
    """Collect the current policy's reachable negative-lookup histories."""
    model.eval()
    result = {}
    for start in range(0, len(rows), batch_size):
        selected = rows[start:start + batch_size]
        lengths = np.asarray([len(row['obs']) for row in selected])
        count, maximum = len(selected), int(lengths.max())
        observations = torch.zeros(maximum, count, 166, device=device)
        contexts = [np.zeros((length,6),np.float32) for length in lengths]
        masks = [np.zeros((length,MAX_QUERIES,1),np.float32) for length in lengths]
        for index,row in enumerate(selected):
            observations[:lengths[index],index] = torch.as_tensor(row['obs'],device=device)
        remaining=np.full(count,MAX_QUERIES,np.int64)
        active=np.ones(count,bool)
        last_q=np.full(count,np.nan,np.float32)
        last_step=np.full(count,-1,np.int64)
        last_travel=np.zeros(count,np.float32)
        hidden=None
        for step in range(maximum):
            batch_context=np.zeros((count,6),np.float32)
            valid=step<lengths
            for index in np.flatnonzero(valid):
                row=selected[index]
                batch_context[index,4]=min(step/max(row['step_limit'],1),1.)
                batch_context[index,5]=min(row['tcp_travel'][step]
                    /max(row['length_limit_m'],1e-6),1.)
                if np.isfinite(last_q[index]):
                    batch_context[index,:4]=[1.,last_q[index],
                        min((step-last_step[index])/max(row['step_limit'],1),1.),
                        min((row['tcp_travel'][step]-last_travel[index])
                            /max(row['length_limit_m'],1e-6),1.)]
                contexts[index][step]=batch_context[index]
                terminal = (step == lengths[index]-1 and
                            row.get('terminal_reason') != 'success')
                if active[index] and remaining[index] and not terminal:
                    masks[index][step,remaining[index]-1]=1.
            values,_,hidden=model(observations[step:step+1],
                torch.as_tensor(batch_context,device=device)[None],hidden)
            current=values[0].cpu().numpy()
            for index in np.flatnonzero(valid & active & (remaining > 0)):
                row=selected[index]
                if (step == lengths[index]-1 and
                        row.get('terminal_reason') != 'success'):
                    active[index]=False
                    continue
                query=current[index,remaining[index]-1,1]>current[index,remaining[index]-1,0]
                if query and row['label'][step]:
                    active[index]=False
                elif query:
                    last_q[index]=row['q'][step]
                    last_step[index]=step
                    last_travel[index]=row['tcp_travel'][step]
                    remaining[index]-=1
                    if not remaining[index]:
                        active[index]=False
        for row,context,mask in zip(selected,contexts,masks):
            result[row['scene_sha256']]=dict(context=context,budget_mask=mask)
    return result


def average_precision(scores, labels):
    scores, labels = np.asarray(scores), np.asarray(labels, dtype=bool)
    positives = int(labels.sum())
    if not positives:
        return 0.
    order = np.argsort(-scores, kind='stable')
    ranked = labels[order]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(precision[ranked].sum() / positives)


def sequence_metrics(predictions, rows, threshold):
    correct = premature = missed = 0
    triggered = []
    for scores, row in zip(predictions, rows):
        indices = np.flatnonzero(scores > threshold)
        if len(indices):
            index = int(indices[0])
            triggered.append(float(scores[index]))
            if row['label'][index]:
                correct += 1
            else:
                premature += 1
        elif row['label'].any():
            missed += 1
    possible = sum(row['label'].any() for row in rows)
    attempts = correct + premature
    return dict(threshold=float(threshold), sequences=len(rows),
        possible_successes=int(possible), correct_stops=correct,
        premature_stops=premature, missed_successes=missed,
        precision=float(correct / attempts) if attempts else 1.,
        recall=float(correct / possible) if possible else 0.,
        stop_rate=float(attempts / len(rows)),
        minimum_triggered_score=min(triggered) if triggered else None)


def calibrate(predictions, rows, minimum_precision=.98, maximum_premature_fraction=.01):
    values = np.concatenate(predictions)
    thresholds = np.unique(np.r_[np.linspace(.05, .9999, 951),
                                 np.quantile(values, np.linspace(0, 1, 1001))])
    table = [sequence_metrics(predictions, rows, threshold) for threshold in thresholds]
    max_premature = max(1, int(np.floor(maximum_premature_fraction * len(rows))))
    eligible = [row for row in table if row['correct_stops'] > 0
                and row['precision'] >= minimum_precision
                and row['premature_stops'] <= max_premature]
    if not eligible:
        raise ValueError('No nontrivial threshold satisfies the false-positive constraints')
    chosen = max(eligible, key=lambda row: (row['correct_stops'],
                 -row['premature_stops'], row['threshold']))
    return dict(chosen=chosen, minimum_precision=minimum_precision,
        maximum_premature_fraction=maximum_premature_fraction,
        maximum_premature_count=max_premature, sweep=table,
        selection='maximize correct sequence-level first stops subject to precision and premature-stop limits')


def operating_point(predictions, rows, minimum_precision=.98,
                    maximum_premature_fraction=.01):
    """Validation-only achievable operating point; never chooses calibration."""
    values = np.concatenate(predictions)
    thresholds = np.unique(np.r_[np.linspace(.05, .9999, 951),
                                 np.quantile(values, np.linspace(0, 1, 1001))])
    table = [sequence_metrics(predictions, rows, threshold) for threshold in thresholds]
    max_premature = max(1, int(np.floor(maximum_premature_fraction * len(rows))))
    eligible = [row for row in table if row['correct_stops'] > 0
                and row['precision'] >= minimum_precision
                and row['premature_stops'] <= max_premature]
    if eligible:
        chosen = max(eligible, key=lambda row: (row['correct_stops'],
                     -row['premature_stops'], row['threshold']))
        return dict(chosen, gate_met=True)
    chosen = max(table, key=lambda row: (row['precision'] if row['correct_stops'] else 0.,
               row['correct_stops'], -row['premature_stops'], row['threshold']))
    return dict(chosen, gate_met=False)


@torch.no_grad()
def predict(model, rows, device, batch_size=128):
    model.eval()
    result = []
    for start in range(0, len(rows), batch_size):
        selected = rows[start:start + batch_size]
        obs, _, _, _ = make_batch(selected, device)
        # Frame diagnostics use the policy with the full query budget. Sequence
        # evaluation below decrements the budget after each negative lookup.
        scores = model(obs)[0][..., -1, :].softmax(-1)[..., 1].cpu().numpy()
        result.extend(scores[:len(row['obs']), index].copy()
                      for index, row in enumerate(selected))
    return result


def frame_report(predictions, rows):
    scores = np.concatenate(predictions)
    labels = np.concatenate([row['label'] for row in rows])
    return dict(frames=len(scores), positives=int(labels.sum()),
                average_precision=average_precision(scores, labels))


@torch.no_grad()
def predict_values(model, rows, device, batch_size=128):
    """Return unpadded [time, remaining-query-budget, action] values."""
    model.eval()
    result = []
    for start in range(0, len(rows), batch_size):
        selected = rows[start:start + batch_size]
        obs, _, _, _ = make_batch(selected, device)
        values = model(obs)[0].cpu().numpy()
        result.extend(values[:len(row['obs']), index].copy()
                      for index, row in enumerate(selected))
    return result


@torch.no_grad()
def predict_q_anchored_values(model, rows, device):
    """Replay lookup feedback so every decision sees its reachable q anchor."""
    model.eval()
    count = len(rows)
    lengths = np.asarray([len(row['obs']) for row in rows])
    observations = torch.zeros(int(lengths.max()), count, 166, device=device)
    result = []
    for index,row in enumerate(rows):
        observations[:lengths[index],index] = torch.as_tensor(row['obs'],device=device)
        sequence=np.zeros((lengths[index],MAX_QUERIES,2),np.float32)
        sequence[...,0]=1.
        result.append(sequence)
    remaining=np.full(count,MAX_QUERIES,np.int64)
    active=np.ones(count,bool)
    last_q=np.full(count,np.nan,np.float32)
    last_step=np.full(count,-1,np.int64)
    last_travel=np.zeros(count,np.float32)
    hidden=None
    for step in range(int(lengths.max())):
        context=np.zeros((count,6),np.float32)
        valid=step<lengths
        for index in np.flatnonzero(valid):
            row=rows[index]
            context[index,4]=min(step/max(row['step_limit'],1),1.)
            context[index,5]=min(row['tcp_travel'][step]
                /max(row['length_limit_m'],1e-6),1.)
            if np.isfinite(last_q[index]):
                context[index,:4]=[1.,last_q[index],
                    min((step-last_step[index])/max(row['step_limit'],1),1.),
                    min((row['tcp_travel'][step]-last_travel[index])
                        /max(row['length_limit_m'],1e-6),1.)]
        values,_,hidden=model(observations[step:step+1],
            torch.as_tensor(context,device=device)[None],hidden)
        current=values[0].cpu().numpy()
        for index in np.flatnonzero(valid & active):
            result[index][step]=current[index]
            row=rows[index]
            if step==lengths[index]-1 and row.get('terminal_reason')!='success':
                active[index]=False
                continue
            if not remaining[index]:
                continue
            query=current[index,remaining[index]-1,1]>current[index,remaining[index]-1,0]
            if query and row['label'][step]:
                active[index]=False
            elif query:
                last_q[index]=row['q'][step]
                last_step[index]=step
                last_travel[index]=row['tcp_travel'][step]
                remaining[index]-=1
    return result


def query_sequence_metrics(predictions, rows, target_settings=None,
                           max_queries=MAX_QUERIES):
    """Replay a fixed-argmax query policy with recoverable negative lookups."""
    settings = dict(gamma=.99, continue_cost=.01, success_reward=1.,
                    failure_reward=-1., query_cost=.05)
    if target_settings:
        settings.update(target_settings)
    gamma = float(settings['gamma'])
    successes = total_lookups = failed_lookups = exhausted = 0
    possible = 0
    returns = []
    lookup_histogram = {str(index): 0 for index in range(max_queries + 1)}
    for values, row in zip(predictions, rows):
        labels = np.asarray(row['label'], dtype=bool)
        values = np.asarray(values)
        if values.shape != (len(labels), max_queries, 2):
            raise ValueError('Invalid query-value prediction shape')
        possible += int(labels.any())
        remaining = max_queries
        lookups = 0
        discounted_return = 0.
        discount = 1.
        succeeded = False
        terminal_without_decision = row.get('terminal_reason') not in (None, 'success')
        for step in range(len(labels)):
            if terminal_without_decision and step == len(labels) - 1:
                discounted_return += discount * settings['failure_reward']
                break
            query = (remaining > 0 and
                     values[step, remaining - 1, 1] > values[step, remaining - 1, 0])
            if query:
                lookups += 1
                total_lookups += 1
                discounted_return -= discount * settings['query_cost']
                if labels[step]:
                    discounted_return += discount * settings['success_reward']
                    successes += 1
                    succeeded = True
                    break
                failed_lookups += 1
                remaining -= 1
            if step == len(labels) - 1:
                discounted_return += discount * settings['failure_reward']
                break
            discounted_return -= discount * settings['continue_cost']
            discount *= gamma
        if not succeeded and remaining == 0:
            exhausted += 1
        lookup_histogram[str(lookups)] += 1
        returns.append(discounted_return)
    sequences = len(rows)
    return dict(sequences=sequences, possible_successes=int(possible),
        successes=int(successes), missed_successes=int(possible - successes),
        total_lookups=int(total_lookups), failed_lookups=int(failed_lookups),
        successful_lookups=int(successes),
        lookup_precision=float(successes / total_lookups) if total_lookups else 1.,
        recall=float(successes / possible) if possible else 0.,
        success_rate=float(successes / sequences) if sequences else 0.,
        mean_lookups=float(total_lookups / sequences) if sequences else 0.,
        exhausted_query_budget=int(exhausted), lookup_histogram=lookup_histogram,
        mean_task_return=float(np.mean(returns)) if returns else 0.)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', nargs='+', required=True)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--action-checkpoint', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--split-seed', type=int, default=20260911)
    parser.add_argument('--head-updates', type=int, default=1500)
    parser.add_argument('--finetune-updates', type=int, default=2500)
    parser.add_argument('--batch-seqs', type=int, default=32)
    parser.add_argument('--positive-weight', type=float, default=1.)
    parser.add_argument('--decision-loss-weight', type=float, default=.5)
    parser.add_argument('--gamma', type=float, default=.99)
    parser.add_argument('--continue-cost', type=float, default=.01)
    parser.add_argument('--success-reward', type=float, default=1.)
    parser.add_argument('--failure-reward', type=float, default=-1.)
    parser.add_argument('--query-cost', type=float, default=.05)
    parser.add_argument('--lookup-feedback', choices=('binary', 'numeric_q', 'hazard_window'),
                        default='binary')
    parser.add_argument('--q-aux-weight', type=float, default=.1)
    parser.add_argument('--lookup-q-noise', type=float, default=.03)
    args = parser.parse_args()
    if args.q_aux_weight < 0 or args.lookup_q_noise < 0:
        raise ValueError('Auxiliary weight and lookup-q noise must be non-negative')
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    started = time.monotonic()
    status = dict(state='loading', started_unix=time.time(), configuration={
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()})
    write(output / 'status.json', status)
    source = Path(__file__).resolve()
    action_path = args.action_checkpoint.resolve()
    action_sha = sha256(action_path)
    action = torch.load(action_path, map_location='cpu')
    if action.get('termination_head') or action.get('head') != 'categorical_kl':
        raise ValueError('Expected an action-only categorical policy')
    settings = validate_budget(action.get('teacher_relative_budget'))
    rows, files, visibility, collected_settings = load_sequences(
        args.data, args.manifest.resolve(), action_sha)
    if collected_settings != settings:
        raise ValueError('Collection budget differs from action checkpoint')
    training, validation, calibration = split_sequences(rows, args.split_seed)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    if args.device.startswith('cuda'):
        torch.cuda.set_device(torch.device(args.device))
    device = torch.device(args.device)
    hazard_window = args.lookup_feedback == 'hazard_window'
    q_anchored = args.lookup_feedback in ('numeric_q', 'hazard_window')
    model = (HazardWindowPolicyStopNet(action['model']) if hazard_window else
             QAnchoredPolicyStopNet(action['model']) if q_anchored else
             PolicyStopNet(action['model'])).to(device)
    model.set_backbone_trainable(False)
    rng = np.random.default_rng(args.seed)
    target_settings = dict(gamma=args.gamma, continue_cost=args.continue_cost,
        success_reward=args.success_reward, failure_reward=args.failure_reward,
        query_cost=args.query_cost,
        max_queries=MAX_QUERIES)
    checkpoint_protocol = (HAZARD_WINDOW_PROTOCOL if hazard_window else
                           Q_ANCHORED_PROTOCOL if q_anchored else PROTOCOL)
    provenance = dict(protocol=checkpoint_protocol,
        target='continue/query Bellman action values under the frozen R3 policy',
        decision_rule='argmax(Q_continue, Q_query) with three-query budget; no calibrated threshold',
        reward_contract=target_settings,
        privileged_training_signal=(
            'simulator q defines terminal reward and auxiliary q/window-hazard targets; '
            'only q obtained at an actual lookup is a runtime input' if q_anchored else
            'simulator q is used only to define terminal reward q > 0.9'),
        lookup_feedback=('last verified numeric q, lookup age, and travel since lookup'
                         if q_anchored else 'binary negative verdict via remaining-query budget'),
        auxiliary_training_signal=(
            'current q, q change, current/1/2/4/8-step graspability hazards, and '
            'next-push window-loss risk' if hazard_window else
            'current simulator q and change from last lookup' if q_anchored else None),
        action_checkpoint=str(action_path), action_checkpoint_sha256=action_sha,
        teacher_relative_budget=settings, visibility_config=visibility,
        training_manifest=str(args.manifest.resolve()), training_manifest_sha256=sha256(args.manifest),
        configuration=status['configuration'], data_sha256={str(path): sha256(path) for path in files},
        training_scenes=sorted(row['scene_sha256'] for row in training),
        validation_scenes=sorted(row['scene_sha256'] for row in validation),
        calibration_scenes=sorted(row['scene_sha256'] for row in calibration),
        split_counts=dict(training=len(training), validation=len(validation), calibration=len(calibration)),
        positive_labels={name: int(sum(row['label'].sum() for row in part))
                         for name, part in [('training', training), ('validation', validation),
                                            ('calibration', calibration)]},
        source_sha256={str(source): sha256(source),
            str(Path(__file__).parents[1] / 'learning/policy_stop.py'):
                sha256(Path(__file__).parents[1] / 'learning/policy_stop.py')},
        action_policy_modified=False, development_scenes_used=False,
        progress_indicator='deterministic max(step_fraction, measured_TCP_travel_fraction); not a learned stop score')
    write(output / 'provenance.json', provenance)

    training_histories = None

    def loss(selected, context_rng=None, policy_histories=None):
        auxiliary = auxiliary_targets = auxiliary_available = None
        if q_anchored:
            batch = make_q_anchored_batch(selected, device,
                context_rng or np.random.default_rng(0), target_settings,
                args.lookup_q_noise, hazard_window=hazard_window,
                policy_histories=policy_histories)
            obs, labels, targets, available, contexts, auxiliary_targets, auxiliary_available = batch
            values, auxiliary, _ = model(obs, contexts)
        else:
            obs, labels, targets, available = make_batch(selected, device, target_settings)
            values, _ = model(obs)
        # Each trajectory contributes equally even though teacher rollouts have
        # different lengths.  This is value fitting, not q/graspability fitting.
        value_error = F.smooth_l1_loss(values, targets, reduction='none') * available
        value_loss = (value_error.sum((0, 2, 3))
                      / available.sum((0, 2, 3)).clamp(min=1.)).mean()
        choices = targets.argmax(-1)
        choice_error = F.cross_entropy(values.reshape(-1, 2), choices.reshape(-1),
                                       reduction='none').view_as(choices)
        decision_valid = available.any(-1)
        weights = torch.where(labels > .5,
            torch.full_like(labels, args.positive_weight), torch.ones_like(labels)).unsqueeze(-1)
        decision_loss = (choice_error * weights * decision_valid).sum() / (
            weights * decision_valid).sum().clamp(min=1.)
        total = value_loss + args.decision_loss_weight * decision_loss
        if auxiliary is not None:
            regression_error = F.smooth_l1_loss(auxiliary[...,:2],
                auxiliary_targets[...,:2],reduction='none')*auxiliary_available[...,:2]
            regression_loss = (regression_error.sum(0)
                / auxiliary_available[...,:2].sum(0).clamp(min=1.)).mean()
            hazard_logits = auxiliary[...,2:]
            hazard_targets = auxiliary_targets[...,2:]
            hazard_available = auxiliary_available[...,2:]
            if hazard_window:
                positives=(hazard_targets*hazard_available).sum((0,1))
                negatives=((1.-hazard_targets)*hazard_available).sum((0,1))
                positive_weight=(negatives/positives.clamp(min=1.)).clamp(min=1.,max=20.)
                hazard_error = F.binary_cross_entropy_with_logits(hazard_logits,
                    hazard_targets,reduction='none',pos_weight=positive_weight.detach())
            else:
                hazard_error = F.binary_cross_entropy_with_logits(hazard_logits,
                    hazard_targets,reduction='none')
            hazard_error = hazard_error * hazard_available
            hazard_loss = hazard_error.sum()/hazard_available.sum().clamp(min=1.)
            auxiliary_loss = regression_loss + hazard_loss
            total = total + args.q_aux_weight * auxiliary_loss
        return total

    history = []
    best = None
    best_rank = (-float('inf'), -1, -float('inf'), -1., -float('inf'))

    def assess(update, stage, train_loss):
        nonlocal best, best_rank
        value_predictions = (predict_q_anchored_values(model, validation, device)
                             if q_anchored else predict_values(model, validation, device))
        if q_anchored:
            predictions = []
            for values in value_predictions:
                shifted = values[:, -1] - values[:, -1].max(-1, keepdims=True)
                probabilities = np.exp(shifted)
                predictions.append(probabilities[:, 1] / probabilities.sum(-1))
        else:
            predictions = predict(model, validation, device)
        report = frame_report(predictions, validation)
        operating = query_sequence_metrics(value_predictions, validation, target_settings)
        with torch.no_grad():
            validation_loss = float(loss(validation, np.random.default_rng(0)))
        row = dict(update=update, stage=stage, train_loss=float(train_loss),
                   validation_loss=validation_loss, **report,
                   validation_successes=operating['successes'],
                   validation_total_lookups=operating['total_lookups'],
                   validation_failed_lookups=operating['failed_lookups'],
                   validation_mean_lookups=operating['mean_lookups'],
                   validation_lookup_precision=operating['lookup_precision'],
                   validation_recall=operating['recall'],
                   seconds=time.monotonic() - started)
        history.append(row)
        write(output / 'history.json', history)
        rank = ((operating['successes'], -operating['mean_lookups'],
                 operating['mean_task_return'], report['average_precision'], -validation_loss)
                if hazard_window else
                (operating['mean_task_return'], operating['successes'],
                 -operating['mean_lookups'], report['average_precision'], -validation_loss))
        if rank > best_rank:
            best_rank = rank
            best = {key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()}
            write(output / 'validation_selection.json', dict(
                rule=('maximum fixed-argmax validation successes, then fewer lookups, '
                      'task return, frame AP, and loss' if hazard_window else
                      'maximum fixed-argmax task return (including lookup and continuation '
                      'costs), then successes, fewer lookups, frame AP, and loss'),
                selected_update=update, selected_stage=stage, **report,
                operating_point=operating,
                validation_loss=validation_loss, calibration_not_used=True,
                fixed_argmax_decision=True,
                development_not_used=True))
        print(json.dumps(row), flush=True)

    update = 0
    status.update(state='running', active='head_only')
    write(output / 'status.json', status)
    head_parameters = [parameter for name, parameter in model.named_parameters()
                       if not name.startswith('backbone.')]
    optimizer = torch.optim.Adam(head_parameters, lr=5e-4, weight_decay=1e-5)
    for _ in range(args.head_updates):
        selected = [training[i] for i in rng.integers(0, len(training), args.batch_seqs)]
        model.train(); optimizer.zero_grad(set_to_none=True)
        current_loss = loss(selected, rng)
        current_loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step(); update += 1
        if update % 250 == 0 or update == args.head_updates:
            assess(update, 'head_only', current_loss)
    model.set_backbone_trainable(True)
    optimizer = torch.optim.Adam([
        {'params': head_parameters, 'lr': 1e-4},
        {'params': model.backbone.parameters(), 'lr': 3e-5}], weight_decay=1e-5)
    status.update(active='policy_conditioned_finetune')
    write(output / 'status.json', status)
    for _ in range(args.finetune_updates):
        if q_anchored and ((_ == 0) or (_ % 500 == 0)):
            training_histories = replay_q_anchored_histories(
                model, training, device)
        selected = [training[i] for i in rng.integers(0, len(training), args.batch_seqs)]
        model.train(); optimizer.zero_grad(set_to_none=True)
        current_loss = loss(selected, rng, training_histories)
        current_loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step(); update += 1
        if update % 250 == 0 or update == args.head_updates + args.finetune_updates:
            assess(update, 'policy_conditioned_finetune', current_loss)
    if best is None:
        raise RuntimeError('No finite validation candidate')
    model.load_state_dict(best, strict=True)
    threshold = .5
    final_predict = predict_q_anchored_values if q_anchored else predict_values
    validation_report = query_sequence_metrics(
        final_predict(model, validation, device), validation, target_settings)
    heldout_report = query_sequence_metrics(
        final_predict(model, calibration, device), calibration, target_settings)
    write(output / 'heldout_validation_query_metrics.json', heldout_report)
    checkpoint = dict(protocol=checkpoint_protocol, model=model.state_dict(), stop_hidden=128,
        architecture=('hazard_window_recurrent_v1' if hazard_window else
                      'q_anchored_recurrent_v1' if q_anchored else
                      'target_preserving_recurrent_v2'),
        embed=model.backbone.embed[0].out_features,
        gru_hidden=model.backbone.gru_hidden,
        target='continue_and_query_action_values', decision_rule='argmax_continue_query',
        max_queries=MAX_QUERIES,
        graspability_threshold=.9,
        lookup_feedback='numeric_q' if q_anchored else 'binary',
        training_variant=args.lookup_feedback,
        lookup_context=('has_lookup,last_q,push_fraction_since_lookup,travel_fraction_since_lookup,'
                        'overall_step_fraction,overall_travel_fraction'
                        if q_anchored else None),
        q_aux_weight=args.q_aux_weight if q_anchored else 0.,
        auxiliary_layout=(['current_q','q_change_from_last_lookup',
            'graspable_within_3_decisions','graspable_now','graspable_within_1_push',
            'graspable_within_2_pushes','graspable_within_4_pushes',
            'graspable_within_8_pushes','window_lost_after_next_push']
            if hazard_window else None),
        lookup_q_training_noise=args.lookup_q_noise if q_anchored else 0.,
        threshold=threshold, threshold_source='fixed_argmax_equivalent',
        reward_contract=target_settings,
        action_checkpoint_sha256=action_sha, teacher_relative_budget=settings,
        visibility_config=visibility, calibration=None,
        progress_indicator=dict(kind='deterministic_teacher_relative_budget_fraction',
            formula='max(step/(teacher_steps+20), tcp_travel/(0.04*teacher_steps+0.15))'),
        action_policy_modified=False)
    torch.save(checkpoint, output / 'policy_stop.pt')
    if sha256(action_path) != action_sha:
        raise RuntimeError('Frozen action checkpoint changed during stop fitting')
    if any(sha256(path) != digest for path, digest in provenance['data_sha256'].items()):
        raise RuntimeError('Stopping data changed during fitting')
    if any(sha256(path) != digest for path, digest in provenance['source_sha256'].items()):
        raise RuntimeError('Stopping trainer changed during fitting')
    complete = dict(complete=True, checkpoint=str(output / 'policy_stop.pt'),
        checkpoint_sha256=sha256(output / 'policy_stop.pt'), action_checkpoint_sha256=action_sha,
        updates=update, threshold=threshold, threshold_source='fixed_argmax_equivalent',
        calibration=None, selection_validation=validation_report,
        heldout_validation=heldout_report, seconds=time.monotonic() - started)
    write(output / 'complete.json', complete)
    status.update(state='complete', active=None, complete=True,
        checkpoint_sha256=complete['checkpoint_sha256'], finished_unix=time.time())
    write(output / 'status.json', status)
    print('POLICY STOP FIT COMPLETE', json.dumps(complete), flush=True)


if __name__ == '__main__':
    main()
