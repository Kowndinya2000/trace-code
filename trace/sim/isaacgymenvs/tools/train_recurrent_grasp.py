"""Fine-tune the clear-view depth checkpoint on causal occluded-image sequences.

Scene identities, including all actor trajectories for a scene, stay on one
side of the fitting/calibration split. No development/test scene trains the
model or selects its threshold. Offline stopping metrics are diagnostics;
closed-loop simulator evaluation is required separately.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from torch.nn import functional as F
from isaacgymenvs.learning.recurrent_grasp import RecurrentGrasp

PROTOCOL = 'occluded-image-grasp-v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def load_sequences(directories, training_manifest):
    allowed = {r['sha256'] for r in json.loads(Path(training_manifest).read_text())['scenes']}
    sequences = []
    manifests = sorted({p.resolve() for d in directories for p in Path(d).rglob('grasp_images/manifest.json')})
    for path in manifests:
        manifest = json.loads(path.read_text())
        if not manifest['complete'] or manifest['protocol'] != PROTOCOL:
            raise ValueError('Incomplete/wrong image collection')
        for item in manifest['sequences']:
            file = path.parent / item['path']
            if sha(file) != item['sha256'] or item['scene_sha256'] not in allowed:
                raise ValueError('Changed data or non-training scene')
            with np.load(file) as data:
                row = {k: data[k].copy() for k in ['images', 'masks', 'meta', 'graspability', 'success_label', 'decision']}
                if str(data['protocol']) != PROTOCOL or str(data['scene_hash']) != item['scene_sha256']:
                    raise ValueError('Sequence identity mismatch')
            n = len(row['images'])
            if n != item['frames'] or not np.array_equal(row['decision'], np.arange(n)):
                raise ValueError('Truncated/misaligned recurrent sequence')
            if row['images'].shape != (n, 2, 112, 112) or row['masks'].shape != (n, 2, 2, 112, 112):
                raise ValueError('Unexpected image/mask shape')
            if not np.array_equal(row['success_label'], row['graspability'] > .9):
                raise ValueError('Wrong classifier supervision')
            if not all(np.isfinite(row[k]).all() for k in ['images', 'meta', 'graspability']):
                raise ValueError('Nonfinite classifier data')
            row.update(scene_sha256=item['scene_sha256'], path=str(file), sha256=item['sha256'])
            sequences.append(row)
    if not sequences:
        raise ValueError('No validated image sequences')
    return sequences


def split_sequences(rows, seed=20260909):
    identities = sorted({r['scene_sha256'] for r in rows})
    np.random.default_rng(seed).shuffle(identities)
    if len(identities) < 10:
        raise ValueError('Need at least 10 independent training scenes')
    calibration = set(identities[:max(2, len(identities)//5)])
    train = [r for r in rows if r['scene_sha256'] not in calibration]
    val = [r for r in rows if r['scene_sha256'] in calibration]
    return train, val


def stopping_metrics(predictions, rows, threshold):
    correct = premature = missed = 0
    for scores, row in zip(predictions, rows):
        above = np.flatnonzero(scores > threshold)
        if len(above):
            if row['success_label'][above[0]]:
                correct += 1
            else:
                premature += 1
        elif row['success_label'].any():
            missed += 1
    return dict(sequences=len(rows), correct_stops=correct, premature_stops=premature,
                missed_successful_endpoints=missed, correct_stop_pct=100*correct/len(rows),
                premature_stop_pct=100*premature/len(rows), threshold=float(threshold))


def calibrate(predictions, rows):
    # Choose on calibration scenes only: maximize detected successful episodes
    # subject to <=5% premature stops on these recorded trajectories.
    thresholds = np.unique(np.r_[np.linspace(.05, .99, 95), .995, .999, .9999, .99999])
    table = [stopping_metrics(predictions, rows, t) for t in thresholds]
    eligible = [r for r in table if r['premature_stop_pct'] <= 5.]
    if not eligible:
        raise ValueError('No threshold satisfies calibration premature-stop bound')
    chosen = max(eligible, key=lambda r: (r['correct_stops'], -r['premature_stops'], r['threshold']))
    strata = {}
    for name, visible in [('target_visible', True), ('target_hidden', False)]:
        scores, labels = [], []
        for p, row in zip(predictions, rows):
            mask = (row['meta'][:, 20] > .5) == visible
            scores.extend(p[mask]); labels.extend(row['success_label'][mask])
        labels, scores = np.asarray(labels, bool), np.asarray(scores)
        selected = scores > chosen['threshold']
        strata[name] = dict(frames=len(labels), positives=int(labels.sum()),
            true_positive=int((selected & labels).sum()), false_positive=int((selected & ~labels).sum()),
            false_negative=int((~selected & labels).sum()))
    return dict(chosen=chosen, sweep=table, strata=strata,
                interpretation='Offline recorded-trajectory diagnostic, not closed-loop retrieval success')


@torch.no_grad()
def cache_features(net, rows, device):
    net.eval()
    for row in rows:
        features = []
        for start in range(0, len(row['images']), 32):
            x = torch.tensor(row['images'][start:start+32], device=device, dtype=torch.float32)
            masks = torch.tensor(row['masks'][start:start+32], device=device, dtype=torch.uint8)
            features.append(net.image_features(x,masks).cpu())
        row['features'] = torch.cat(features)


def batch(rows, device, images=False):
    size = len(rows)
    length = max(len(r['images']) for r in rows)
    key = 'images' if images else 'features'
    shape = (2,112,112) if images else (rows[0]['features'].shape[-1],)
    xs = torch.zeros((length, size) + shape, device=device)
    masks = torch.zeros((length,size,2,2,112,112), device=device, dtype=torch.uint8)
    meta = torch.zeros((length,size,26), device=device)
    labels = torch.zeros((length,size), device=device)
    valid = torch.zeros_like(labels)
    for i, row in enumerate(rows):
        n = len(row['images'])
        xs[:n,i] = torch.as_tensor(row[key], device=device, dtype=torch.float32)
        masks[:n,i] = torch.as_tensor(row['masks'], device=device)
        meta[:n,i] = torch.as_tensor(row['meta'], device=device)
        labels[:n,i] = torch.as_tensor(row['success_label'].astype(np.float32), device=device)
        valid[:n,i] = 1
    return xs, masks, meta, labels, valid


@torch.no_grad()
def predict(net, rows, device):
    net.eval()
    result = []
    for row in rows:
        x, masks, meta, _, _ = batch([row], device)
        logits, _ = net.forward_features(x, masks, meta)
        result.append(logits[:,0].sigmoid().cpu().numpy())
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', nargs='+', required=True)
    parser.add_argument('--training-manifest', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--warmup-updates', type=int, default=600)
    parser.add_argument('--finetune-epochs', type=int, default=4)
    args = parser.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    status = dict(state='loading', started_unix=time.time(), configuration=vars(args))
    write(out/'status.json', status)
    try:
        torch.set_num_threads(4)
        torch.manual_seed(args.seed); np.random.seed(args.seed)
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
        rng = np.random.default_rng(args.seed)
        rows = load_sequences(args.data, args.training_manifest)
        train, val = split_sequences(rows)
        pos = sum(r['success_label'].sum() for r in train)
        total = sum(len(r['images']) for r in train)
        if pos < 20:
            raise ValueError('Insufficient positive endpoints; collect more scenes')
        net = RecurrentGrasp(args.checkpoint).to(args.device)
        for p in net.encoder.parameters(): p.requires_grad=False
        pos_weight = torch.tensor(min((total-pos)/pos, 15.), device=args.device)
        optim = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=3e-4)
        provenance = dict(protocol=PROTOCOL, configuration=vars(args),
            initial_checkpoint_sha256=sha(args.checkpoint),
            fitting_scene_hashes=sorted({r['scene_sha256'] for r in train}),
            calibration_scene_hashes=sorted({r['scene_sha256'] for r in val}),
            data=[dict(path=r['path'],sha256=r['sha256']) for r in rows],
            training_frames=int(total), training_positives=int(pos), positive_weight=float(pos_weight),
            training_hidden_target_frames=int(sum((r['meta'][:,20]<.5).sum() for r in train)),
            training_hidden_target_positives=int(sum(((r['meta'][:,20]<.5)&r['success_label']).sum() for r in train)),
            encoder_training='Frozen-feature warmup followed by last-block/head fine-tuning; BN statistics fixed',
            labels='Binary clear-view q>0.9, invalid/physical-failure frames negative',
            source_sha256={str(p):sha(p) for p in [Path(__file__),Path(__file__).parents[1]/'learning/recurrent_grasp.py',Path(__file__).parents[1]/'open_loop/grasp_image_obs.py']})
        write(out/'provenance.json', provenance)
        status.update(state='running',active='cache_pretrained_features'); write(out/'status.json',status)
        cache_features(net, rows, args.device)
        # Unmodified pretrained local-crop classifier on the SAME occluded
        # inputs: reports a starting point, without using hidden target poses.
        with torch.no_grad():
            baseline=[net.encoder._fc(r['features'][:,:net.feature_dim].to(args.device)).sigmoid().flatten().cpu().numpy() for r in val]
        try:
            baseline_calibration=calibrate(baseline,val)
        except ValueError as error:
            baseline_calibration=dict(eligible=False,error=str(error))
        write(out/'original_classifier_calibration.json',dict(
            fixed_threshold=stopping_metrics(baseline,val,.9),
            calibrated=baseline_calibration,
            note='Original single-frame CNN on last-detected-target crops; offline trajectory diagnostic'))
        best = (-1, -100000)

        def evaluate_and_save(stage):
            nonlocal best
            try:
                report = calibrate(predict(net, val, args.device), val)
            except ValueError as error:
                report=dict(eligible=False,error=str(error),stage=stage)
                write(out/(stage+'_calibration.json'),report)
                print(stage,json.dumps(report),flush=True)
                return report
            report['stage'] = stage
            write(out/(stage+'_calibration.json'), report)
            metric = report['chosen']
            rank = (metric['correct_stops'], -metric['premature_stops'])
            if rank > best:
                best = rank
                torch.save(dict(protocol=PROTOCOL,model=net.state_dict(),hidden=net.hidden,
                    threshold=metric['threshold'],initial_checkpoint_sha256=sha(args.checkpoint),
                    calibration=report,provenance_sha256=sha(out/'provenance.json')), out/'classifier.pt')
            print(stage, json.dumps(metric), flush=True)
            return report

        for update in range(args.warmup_updates):
            net.train()
            chosen = [train[i] for i in rng.integers(0,len(train),8)]
            x, masks, meta, y, valid = batch(chosen, args.device)
            optim.zero_grad(set_to_none=True)
            logits, _ = net.forward_features(x,masks,meta)
            loss = (F.binary_cross_entropy_with_logits(logits,y,reduction='none',pos_weight=pos_weight)*valid).sum()/valid.sum()
            loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);optim.step()
            if (update+1)%50==0:
                status.update(active='recurrent_warmup',update=update+1,loss=float(loss),updated_unix=time.time())
                write(out/'status.json',status)
                print('warmup',update+1,float(loss),flush=True)
            if (update+1)%200==0 or update+1==args.warmup_updates:
                evaluate_and_save('warmup_%04d'%(update+1))
        net.finetune_last_stage()
        optim=torch.optim.Adam([p for p in net.parameters() if p.requires_grad],lr=3e-5)
        for epoch in range(args.finetune_epochs):
            order=rng.permutation(len(train))
            for start in range(0,len(order),4):
                net.train()
                chosen=[train[i] for i in order[start:start+4]]
                x,masks,meta,y,valid=batch(chosen,args.device,images=True)
                hidden=None
                for t in range(0,len(x),8):
                    sl=slice(t,t+8);v=valid[sl]
                    if not v.any():continue
                    optim.zero_grad(set_to_none=True)
                    logits,hidden=net(x[sl],masks[sl],meta[sl],hidden)
                    loss=(F.binary_cross_entropy_with_logits(logits,y[sl],reduction='none',pos_weight=pos_weight)*v).sum()/v.sum()
                    loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);optim.step()
                    hidden=hidden.detach()
                status.update(active='encoder_finetune',epoch=epoch+1,sequences=start+len(chosen),updated_unix=time.time())
                write(out/'status.json',status)
            cache_features(net,val,args.device)
            evaluate_and_save('finetune_%02d'%(epoch+1))
        if any(sha(r['path'])!=r['sha256'] for r in rows):raise ValueError('Data changed during fitting')
        if any(sha(p)!=h for p,h in provenance['source_sha256'].items()):raise ValueError('Training source changed')
        status.update(state='complete',active=None,complete=True,checkpoint_sha256=sha(out/'classifier.pt'),finished_unix=time.time())
    except BaseException as error:
        status.update(state='failed',error=repr(error),updated_unix=time.time())
        write(out/'status.json',status)
        raise
    write(out/'status.json',status)


if __name__=='__main__':main()
