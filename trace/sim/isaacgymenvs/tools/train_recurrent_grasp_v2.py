"""Expanded-data grasp fitting with explicit precision/recall calibration.

The v1 checkpoint format and causal deployment inputs are unchanged. This
trainer never reads development/test scenes and never adjusts pushing weights.
"""
from pathlib import Path
import argparse
import json
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from torch.nn import functional as F
from isaacgymenvs.learning.recurrent_grasp import RecurrentGrasp
from isaacgymenvs.tools.train_recurrent_grasp import (
    PROTOCOL,load_sequences,cache_features,predict,batch,write,sha,stopping_metrics)


def split_from_manifest(rows,split):
    fitting=set(split['fitting']);calibration=set(split['calibration'])
    if fitting & calibration:raise ValueError('Fitting/calibration overlap')
    present={r['scene_sha256'] for r in rows}
    if present != fitting|calibration:raise ValueError('Split does not cover data exactly')
    return ([r for r in rows if r['scene_sha256'] in fitting],
            [r for r in rows if r['scene_sha256'] in calibration])


def confusion(scores,labels,threshold):
    selected=np.asarray(scores)>threshold;labels=np.asarray(labels,bool)
    tp=int((selected&labels).sum());fp=int((selected&~labels).sum())
    fn=int((~selected&labels).sum());tn=int((~selected&~labels).sum())
    precision=tp/max(tp+fp,1);recall=tp/max(tp+fn,1)
    return dict(frames=len(labels),positives=tp+fn,tp=tp,fp=fp,fn=fn,tn=tn,
        accuracy=(tp+tn)/max(len(labels),1),precision=precision,recall=recall,
        f05=1.25*precision*recall/max(.25*precision+recall,1e-12))


def metrics(predictions,rows,threshold):
    p=np.concatenate(predictions);y=np.concatenate([r['success_label'] for r in rows])
    hidden=np.concatenate([r['meta'][:,20]<.5 for r in rows])
    first=stopping_metrics(predictions,rows,threshold)
    first['precision']=first['correct_stops']/max(first['correct_stops']+first['premature_stops'],1)
    first['recall']=first['correct_stops']/max(sum(bool(r['success_label'].any()) for r in rows),1)
    return dict(threshold=float(threshold),all=confusion(p,y,threshold),
        target_hidden=confusion(p[hidden],y[hidden],threshold),
        target_visible=confusion(p[~hidden],y[~hidden],threshold),first_stop=first)


def rank_metrics(m):
    # Explicitly avoid selecting a nearly-always-negative model just because
    # fewer than 5% of all episodes contain a false stop. Precision's
    # denominator is actual positive predictions/stop requests.
    passed=(m['all']['precision']>=.90 and m['first_stop']['precision']>=.90
        and m['all']['recall']>=.30 and m['target_hidden']['recall']>=.20)
    if passed:
        return (1,(m['first_stop']['recall']+m['target_hidden']['recall'])/2,
            m['all']['f05'])
    p,r=m['first_stop']['precision'],m['first_stop']['recall']
    first_f05=1.25*p*r/max(.25*p+r,1e-12)
    # If no candidate clears the gate, retain an explicitly unqualified
    # diagnostic candidate rather than pretending the target was achieved.
    return (0,(m['all']['f05']+first_f05)/2,m['target_hidden']['recall'])


def calibrate_v2(predictions,rows):
    thresholds=np.unique(np.r_[np.linspace(.05,.99,95),.995,.999,.9999,.99999])
    sweep=[metrics(predictions,rows,t) for t in thresholds]
    chosen=max(sweep,key=lambda m:(*rank_metrics(m),m['threshold']))
    return dict(chosen=chosen,rank=list(rank_metrics(chosen)),
        offline_gate_pass=bool(rank_metrics(chosen)[0]),sweep=sweep,
        selection='Require frame/first-stop precision>=0.90, frame recall>=0.30, hidden recall>=0.20; otherwise diagnostic F0.5 fallback',
        interpretation='Calibration histories only; neither independent test nor closed-loop retrieval success')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',nargs='+',required=True)
    ap.add_argument('--training-manifest',required=True)
    ap.add_argument('--split',required=True)
    ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--output',required=True)
    ap.add_argument('--device',default='cuda:1')
    ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--positive-weight-cap',type=float,default=4.)
    ap.add_argument('--warmup-updates',type=int,default=2000)
    ap.add_argument('--finetune-epochs',type=int,default=4)
    ap.add_argument('--architecture', choices=['recurrent_v1','residual_gru_v1','memory_residual_gru_v1','spatial_memory_gru_v1'], default='recurrent_v1')
    ap.add_argument('--weight-decay',type=float,default=0.)
    ap.add_argument('--soft-targets',action='store_true')
    a=ap.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    status=dict(state='loading',started_unix=time.time(),configuration=vars(a))
    write(out/'status.json',status)
    try:
        torch.set_num_threads(4);torch.manual_seed(a.seed);np.random.seed(a.seed)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        rng=np.random.default_rng(a.seed)
        rows=load_sequences(a.data,a.training_manifest)
        if a.architecture in ('memory_residual_gru_v1','spatial_memory_gru_v1'):
            from isaacgymenvs.open_loop.grasp_image_memory import transform_sequence
            for row in rows:
                row['images'],row['masks']=transform_sequence(row['images'],row['masks'],row['meta'])
        train,val=split_from_manifest(rows,json.loads(Path(a.split).read_text()))
        pos=sum(r['success_label'].sum() for r in train);total=sum(len(r['images']) for r in train)
        if pos<20:raise ValueError('Too few positive frames')
        weight=min(np.sqrt((total-pos)/pos),a.positive_weight_cap)
        pos_weight=torch.tensor(weight,device=a.device)
        net=RecurrentGrasp(a.checkpoint,architecture=a.architecture).to(a.device)
        for p in net.encoder.parameters():p.requires_grad=False
        provenance=dict(protocol='occluded-grasp-improvement-v2',configuration=vars(a),
            initial_checkpoint_sha256=sha(a.checkpoint),split_sha256=sha(a.split),
            data=[dict(path=r['path'],sha256=r['sha256']) for r in rows],
            fitting_scene_hashes=sorted({r['scene_sha256'] for r in train}),
            calibration_scene_hashes=sorted({r['scene_sha256'] for r in val}),
            training_frames=int(total),training_positives=int(pos),positive_weight=float(weight),
            training_hidden_target_positives=int(sum(((r['meta'][:,20]<.5)&r['success_label']).sum() for r in train)),
            loss=('Masked BCE with clear-GN soft probability targets' if a.soft_targets else
                  'Masked per-frame binary cross entropy; positive weight=min(sqrt(Nnegative/Npositive), cap)'),
            image_preprocessing=('Causal workspace-aligned local depth memory from recorded observations only'
                if a.architecture in ('memory_residual_gru_v1','spatial_memory_gru_v1') else 'Original local crop and workspace overview'),
            source_sha256={str(p):sha(p) for p in [Path(__file__),Path(__file__).with_name('train_recurrent_grasp.py'),Path(__file__).parents[1]/'learning/recurrent_grasp.py']})
        write(out/'provenance.json',provenance)
        status.update(state='running',active='cache_features',training_frames=int(total),training_positives=int(pos))
        write(out/'status.json',status);cache_features(net,rows,a.device)
        optim=torch.optim.Adam([p for p in net.parameters() if p.requires_grad],lr=3e-4,weight_decay=a.weight_decay)
        best=None

        def evaluate(stage):
            nonlocal best
            report=calibrate_v2(predict(net,val,a.device),val);report['stage']=stage
            write(out/(stage+'_calibration.json'),report)
            rank=tuple(report['rank'])
            if best is None or rank>best:
                best=rank
                torch.save(dict(protocol=PROTOCOL,model=net.state_dict(),hidden=net.hidden,architecture=a.architecture,
                    threshold=report['chosen']['threshold'],initial_checkpoint_sha256=sha(a.checkpoint),
                    calibration=report,provenance_sha256=sha(out/'provenance.json')),out/'classifier.pt')
                write(out/'selected.json',dict(stage=stage,rank=list(rank),threshold=report['chosen']['threshold'],
                    offline_gate_pass=report['offline_gate_pass'],metrics=report['chosen']))
            print(stage,json.dumps(report['chosen']),flush=True)

        def targets(chosen, binary):
            if not a.soft_targets:
                return binary
            soft = torch.zeros_like(binary)
            for i,row in enumerate(chosen):
                soft[:len(row['graspability']),i] = torch.as_tensor(row['graspability'],device=a.device)
            return soft

        if a.architecture != 'recurrent_v1':
            evaluate('initial_stock_prior')
        for update in range(a.warmup_updates):
            net.train();chosen=[train[i] for i in rng.integers(0,len(train),8)]
            x,masks,meta,y,valid=batch(chosen,a.device)
            y=targets(chosen,y)
            optim.zero_grad(set_to_none=True)
            logits,_=net.forward_features(x,masks,meta)
            loss=(F.binary_cross_entropy_with_logits(logits,y,reduction='none',pos_weight=pos_weight)*valid).sum()/valid.sum()
            loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);optim.step()
            if (update+1)%50==0:
                status.update(active='recurrent_warmup',update=update+1,loss=float(loss),updated_unix=time.time());write(out/'status.json',status)
            if (update+1)%500==0 or update+1==a.warmup_updates:evaluate('warmup_%04d'%(update+1))
        net.finetune_last_stage()
        optim=torch.optim.Adam([p for p in net.parameters() if p.requires_grad],lr=3e-5,weight_decay=a.weight_decay)
        for epoch in range(a.finetune_epochs):
            order=rng.permutation(len(train))
            for start in range(0,len(order),4):
                net.train();chosen=[train[i] for i in order[start:start+4]]
                x,masks,meta,y,valid=batch(chosen,a.device,images=True);hidden=None
                y=targets(chosen,y)
                for t in range(0,len(x),8):
                    sl=slice(t,t+8);v=valid[sl]
                    if not v.any():continue
                    optim.zero_grad(set_to_none=True)
                    logits,hidden=net(x[sl],masks[sl],meta[sl],hidden)
                    loss=(F.binary_cross_entropy_with_logits(logits,y[sl],reduction='none',pos_weight=pos_weight)*v).sum()/v.sum()
                    loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);optim.step();hidden=hidden.detach()
                if start%40==0:
                    status.update(active='encoder_finetune',epoch=epoch+1,sequences=start+len(chosen),loss=float(loss),updated_unix=time.time());write(out/'status.json',status)
            cache_features(net,val,a.device);evaluate('finetune_%02d'%(epoch+1))
        state=torch.load(out/'classifier.pt',map_location='cpu');net.load_state_dict(state['model']);net.eval()
        final={}
        for name,data in [('training',train),('calibration',val)]:
            cache_features(net,data,a.device);pred=predict(net,data,a.device)
            final[name]=dict(scenes=len({r['scene_sha256'] for r in data}),histories=len(data),
                operating=metrics(pred,data,state['threshold']),threshold_05=metrics(pred,data,.5))
        write(out/'frame_metrics.json',final)
        if any(sha(r['path'])!=r['sha256'] for r in rows):raise ValueError('Training data changed')
        if any(sha(p)!=h for p,h in provenance['source_sha256'].items()):raise ValueError('Training source changed')
        status.update(state='complete',active=None,complete=True,checkpoint_sha256=sha(out/'classifier.pt'),finished_unix=time.time())
    except BaseException as error:
        status.update(state='failed',error=repr(error),updated_unix=time.time());write(out/'status.json',status);raise
    write(out/'status.json',status)


if __name__=='__main__':main()
