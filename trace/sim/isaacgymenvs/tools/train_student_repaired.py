"""Joint action/graspability fitting from expert-success and DAgger episodes."""
from pathlib import Path
import argparse
import hashlib
import json
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from isaacgymenvs.learning.student_net import StudentNet
from isaacgymenvs.learning.student_ablate import obs_mask,strip_recurrence
from isaacgymenvs.open_loop.evaluation_core import PROTOCOL,primitive_vectors,sha256,read_manifest
from isaacgymenvs.open_loop.arm_occlusion import VISIBILITY_PROTOCOL, visibility_settings


def reduce_masked(values, mask, trajectory_balanced=False, weights=None):
    """Reduce a time-by-trajectory loss without overweighting long failures."""
    if values.shape != mask.shape or values.ndim != 2:
        raise ValueError('Expected matching [time, trajectory] loss and mask tensors')
    if weights is None:
        weights=torch.ones_like(mask)
    if weights.shape != mask.shape or not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError('Expected finite non-negative loss weights')
    effective=mask*weights
    if not trajectory_balanced:
        return (values*effective).sum()/effective.sum().clamp(min=1.)
    counts=effective.sum(0)
    active=counts>0
    if not bool(active.any()):
        return (values*mask).sum()*0.
    per_trajectory=(values*effective).sum(0)/counts.clamp(min=1.)
    return per_trajectory[active].mean()


def load_data(directories,manifest,allow_oracle_dagger=False,require_geometric_visibility=False,
              require_bounded_dagger=False,require_cartesian_dagger=False,
              teacher_relative_budget=None):
    allowed={r["sha256"] for r in read_manifest(manifest)["scenes"]}
    # Aggregate order follows the --data arguments and each shard's path *within* its own
    # collection, never the absolute path: otherwise where the data happens to sit on disk
    # changes the batch stream, and the same command gives different weights on another machine.
    files=[];seen=set()
    for directory in directories:
        root=Path(directory).resolve()
        for shard in sorted(root.rglob("collection/*.npz"),key=lambda q:q.relative_to(root).as_posix()):
            if shard not in seen:seen.add(shard);files.append(shard)
    if not files: raise ValueError("No corrected collection shards")
    datasets=[]
    visibility_identity = None
    bounded_identity = None
    cartesian_identity = None
    budget_identity = None
    if teacher_relative_budget is not None:
        from isaacgymenvs.open_loop.teacher_relative_budget import validate as validate_budget
        teacher_relative_budget=validate_budget(teacher_relative_budget)
    for path in files:
        with np.load(path) as d:
            if str(d["protocol"])!=PROTOCOL: raise ValueError("Legacy/mixed collection protocol")
            visibility = str(d['visibility_config']) if 'visibility_config' in d else 'legacy'
            if require_geometric_visibility or allow_oracle_dagger:
                if visibility == 'legacy': raise ValueError('Recollect data with current geometric visibility')
                settings = json.loads(visibility)
                if settings.get('protocol') != VISIBILITY_PROTOCOL:
                    raise ValueError('Training requires the current geometric visibility protocol')
                checked = visibility_settings(dict(occlusion_mode=settings['mode'],occlusion_margin_m=settings['margin_m'],
                                                   p_drop=settings['p_drop'],blackout_len=settings['blackout_len']))
                if settings != checked:
                    raise ValueError('Invalid visibility configuration')
            if visibility_identity is not None and visibility != visibility_identity:
                raise ValueError('Mixed visibility configurations; recollect under one observation model')
            visibility_identity = visibility
            row={k:d[k].copy() for k in ["obs","teacher_logits","teacher_value","valid","scene_hash","graspability","stop_valid","step_budget"]}
            budget,steps,reasons=d['step_budget'],d['terminal_step'],d['terminal_reason']
            mode,actor=str(d['stopping']),str(d['actor'])
            if teacher_relative_budget is not None and actor == 'student':
                if str(d.get('teacher_relative_budget_protocol','')) != teacher_relative_budget['protocol']:
                    raise ValueError('Student DAgger shard lacks teacher-relative budget evidence')
                collected=json.loads(str(d['teacher_relative_budget']))
                if collected != teacher_relative_budget:
                    raise ValueError('Mixed teacher-relative budget configurations')
                identity=json.dumps(collected,sort_keys=True)
                if budget_identity is not None and budget_identity != identity:
                    raise ValueError('Mixed teacher-relative budget configurations')
                budget_identity=identity
            permitted = [('teacher','expert_graspability')]
            if allow_oracle_dagger:
                if 'diagnostic_oracle' in d and bool(d['diagnostic_oracle']):
                    permitted.append(('student','oracle_graspability'))
            else:
                permitted.append(('student','learned_score'))
            if (actor,mode) not in permitted:
                raise ValueError('Training requires expert-success demonstrations and the explicitly selected DAgger stopping mode')
            if require_bounded_dagger and actor == 'student':
                from isaacgymenvs.open_loop.plan_constraint import config, extension_config
                if str(d.get('bounded_collection_protocol','')) != 'bounded_collection_v1':
                    raise ValueError('Student DAgger shard lacks bounded_collection_v1 evidence')
                constraints=json.loads(str(d['plan_constraints']))
                extension=json.loads(str(d['plan_extension']))
                if constraints != config(constraints['radius_m'],constraints['correction_step_m']):
                    raise ValueError('Invalid bounded DAgger path constraints')
                if extension != extension_config(extension['max_steps'],extension['max_length_m'],
                                                  extension['step_length_m'],extension.get('shorten_to_bounds',False)):
                    raise ValueError('Invalid bounded DAgger extension limits')
                identity=json.dumps(dict(plan_constraints=constraints,plan_extension=extension),sort_keys=True)
                if bounded_identity is not None and identity != bounded_identity:
                    raise ValueError('Mixed bounded DAgger configurations')
                bounded_identity=identity
                scale=d['executed_action_scale']
                if scale.shape != d['action_executed'].shape or not np.isfinite(scale).all() or (scale<=0).any() or (scale>1).any():
                    raise ValueError('Invalid bounded DAgger action scales')
                t=np.arange(len(scale))[:,None]
                extra=d['action_executed'].astype(bool)&(t>=np.asarray(d['step_budget'])[None,:])
                if (scale[extra]*.04>extension['step_length_m']+1e-7).any():
                    raise ValueError('Bounded DAgger per-step extension exceeded')
                if ((scale*.04*extra).sum(0)>extension['max_length_m']+1e-7).any():
                    raise ValueError('Bounded DAgger total extension exceeded')
            if require_cartesian_dagger and actor == 'student':
                from isaacgymenvs.open_loop.teacher_relative_cartesian import validate as validate_cartesian
                if str(d.get('cartesian_collection_protocol','')) != 'teacher-relative-cartesian-collection-v1':
                    raise ValueError('Student DAgger shard lacks Cartesian execution evidence')
                settings = validate_cartesian(json.loads(str(d['cartesian_policy'])))
                identity = json.dumps(settings, sort_keys=True)
                if cartesian_identity is not None and identity != cartesian_identity:
                    raise ValueError('Mixed Cartesian policy configurations')
                cartesian_identity = identity
                executed = d['executed_cartesian_path']
                if executed.shape != (*d['action_executed'].shape, 2, 2):
                    raise ValueError('Invalid executed Cartesian path shape')
                if not np.isfinite(executed[d['action_executed'].astype(bool)]).all():
                    raise ValueError('Non-finite executed Cartesian path')
            limit=int(d['execution_limit'])
            if (steps>limit).any() or (steps<0).any(): raise ValueError('Invalid episode action budget')
            if (steps>=len(row['obs'])).any(): raise ValueError('Missing terminal observation for graspability supervision')
            expected=np.arange(len(row['obs']))[:,None]<=steps[None,:]
            if (row['stop_valid']>expected).any(): raise ValueError('Stop supervision after termination')
            if not np.isfinite(row['graspability']).all() or ((row['graspability']<0)|(row['graspability']>1)).any():
                raise ValueError('Invalid graspability targets')
            executed=d['action_executed']
            if not np.array_equal(executed.sum(0),steps): raise ValueError('Executed mask does not match decisions')
            expected_actions=np.arange(len(row['obs']))[:,None]<steps[None,:]
            if not np.array_equal(executed,expected_actions): raise ValueError('Executed mask is not a contiguous episode')
            if (row['valid']>executed).any() or (row['valid'][row['graspability']>.9]>0).any():
                raise ValueError('Action labels include a stopped or already-graspable state')
            if actor=='teacher' or mode=='oracle_graspability':
                if not np.array_equal(row['valid'],executed): raise ValueError('Expert action labels missing')
                if ((row['graspability']>.9)&expected_actions).any():
                    raise ValueError('Expert continued after graspability')
            for i,step in enumerate(steps):
                if reasons[i]=='success' and (row['stop_valid'][step,i]!=1 or row['graspability'][step,i]<=.9):
                    raise ValueError('Successful terminal observation lacks positive graspability supervision')
        if not set(row["scene_hash"]).issubset(allowed): raise ValueError("Shard contains a scene outside the approved fitting manifest")
        if not np.isfinite(row["obs"]).all(): raise ValueError("Non-finite observation")
        # A learner can miss a stop and push a graspable scene back into clutter;
        # its action-supervision mask can resume, but execution never resumes after terminal.
        datasets.append(row)
    T=max(d["obs"].shape[0] for d in datasets);N=sum(d["obs"].shape[1] for d in datasets)
    obs_dims={d['obs'].shape[-1] for d in datasets}
    if obs_dims != {166}: raise ValueError('Teacher-relative categorical policy requires the stable 166-D observation')
    out=dict(obs=np.zeros((T,N,166),np.float32),teacher_logits=np.zeros((T,N,16),np.float32),
             teacher_value=np.zeros((T,N),np.float32),valid=np.zeros((T,N),np.float32),
             graspability=np.zeros((T,N),np.float32),stop_valid=np.zeros((T,N),np.float32))
    hashes=[];budgets=[];offset=0
    for d in datasets:
        t,n=d["obs"].shape[:2]
        for key in out: out[key][:t,offset:offset+n]=d[key]
        hashes.extend(d["scene_hash"].tolist());budgets.extend(d['step_budget'].tolist());offset+=n
    out["scene_hash"]=np.array(hashes)
    out['step_budget']=np.asarray(budgets,dtype=np.int64)
    out["visibility_config"] = visibility_identity
    if require_bounded_dagger and bounded_identity is None:
        raise ValueError('No bounded student DAgger trajectories found')
    if require_cartesian_dagger and cartesian_identity is None:
        raise ValueError('No Cartesian student DAgger trajectories found')
    out['bounded_dagger_config'] = json.loads(bounded_identity) if bounded_identity is not None else None
    out['cartesian_dagger_config'] = json.loads(cartesian_identity) if cartesian_identity is not None else None
    out['teacher_relative_budget_config'] = teacher_relative_budget
    return out,files


def grouped_split(hashes,split_seed=20260907,fraction=.15):
    unique=np.unique(hashes)
    if len(unique)<2: raise ValueError("Need at least two source scenes")
    perm=np.random.default_rng(split_seed).permutation(unique)
    nv=max(1,min(len(unique)-1,int(len(unique)*fraction)))
    vi=np.flatnonzero(np.isin(hashes,perm[:nv]))
    ti=np.flatnonzero(~np.isin(hashes,perm[:nv]))
    assert not set(hashes[vi])&set(hashes[ti])
    return ti,vi


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--data",nargs="+",required=True)
    ap.add_argument("--manifest",required=True);ap.add_argument("--output",required=True)
    ap.add_argument("--seed",type=int,default=0);ap.add_argument("--split-seed",type=int,default=20260907)
    ap.add_argument("--updates",type=int,default=10000);ap.add_argument("--batch-seqs",type=int,default=32)
    ap.add_argument("--select-best-validation",action="store_true")
    ap.add_argument("--head",choices=["xy","categorical_kl","categorical_hard",
                                      "eef_endpoint","eef_waypoints"],default="xy")
    ap.add_argument("--value-coef",type=float,default=0.);ap.add_argument("--lr",type=float,default=3e-4)
    ap.add_argument("--label-budget",type=int,default=0)
    ap.add_argument("--termination-head",action="store_true")
    ap.add_argument("--allow-oracle-dagger",action="store_true",
                    help="Explicit simulator-only experiment; accepts oracle-terminated DAgger, no learned stop head")
    ap.add_argument("--require-geometric-visibility",action="store_true",
                    help="Require the current arm-footprint model and one consistent visibility/dropout configuration")
    ap.add_argument("--require-bounded-dagger",action="store_true",
                    help="Require consistent, trace-backed runtime bounds on every student DAgger shard")
    ap.add_argument("--require-cartesian-dagger",action="store_true",
                    help="Require trace-backed continuous Cartesian execution on student DAgger shards")
    ap.add_argument('--teacher-relative-budget',action='store_true',
                    help='Recovery-aware KL weighting for the action-only policy; stopping remains separate')
    ap.add_argument('--budget-extra-steps',type=int,default=20)
    ap.add_argument('--budget-extra-length-m',type=float,default=.15)
    ap.add_argument('--budget-risk-start',type=float,default=.70)
    ap.add_argument('--budget-risk-temperature',type=float,default=.10)
    ap.add_argument('--budget-max-imitation-weight',type=float,default=3.)
    ap.add_argument("--smoothness-coef",type=float,default=.05,
                    help="For Cartesian heads, temporal penalty on changes in teacher-relative prediction error")
    ap.add_argument("--trajectory-balanced-action-loss",action="store_true",
                    help="Give each trajectory equal action/smoothness weight instead of overweighting long failures")
    ap.add_argument("--recovery-travel-m",type=float,default=.12,
                    help="Checkpoint-bound cumulative Cartesian recovery travel in metres")
    ap.add_argument("--stop-coef",type=float,default=1.)
    ap.add_argument("--stop-label-budget",type=int,default=0)
    ap.add_argument("--ablate",default="none");ap.add_argument("--device",default="cuda:0")
    ap.add_argument("--smoke",action="store_true",help="Explicitly label a non-production fitting check")
    a=ap.parse_args()
    if a.allow_oracle_dagger and a.termination_head:
        ap.error('--allow-oracle-dagger is an action-only experiment; omit --termination-head')
    if a.smoothness_coef < 0:
        ap.error('--smoothness-coef must be non-negative')
    if not np.isfinite(a.recovery_travel_m) or a.recovery_travel_m <= 0:
        ap.error('--recovery-travel-m must be positive and finite')
    if not a.head.startswith('eef_') and a.recovery_travel_m != .12:
        ap.error('--recovery-travel-m applies only to Cartesian eef heads')
    if a.require_bounded_dagger and a.require_cartesian_dagger:
        ap.error('Categorical and Cartesian DAgger contracts are mutually exclusive')
    budget_config=None
    if a.teacher_relative_budget:
        from isaacgymenvs.open_loop.teacher_relative_budget import config as make_budget_config
        budget_config=make_budget_config(a.budget_extra_steps,a.budget_extra_length_m,.04,
                                         a.budget_risk_start,a.budget_risk_temperature,
                                         a.budget_max_imitation_weight)
        if a.head != 'categorical_kl' or a.termination_head or a.require_bounded_dagger or a.require_cartesian_dagger:
            ap.error('--teacher-relative-budget is only for an action-only categorical-KL policy')
    out=Path(a.output).resolve()
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    manifest=read_manifest(a.manifest)
    if not a.smoke and manifest.get("role")!="training":
        raise ValueError("Production fitting requires a training-role manifest")
    d,files=load_data(a.data,a.manifest,a.allow_oracle_dagger,a.require_geometric_visibility,
                      a.require_bounded_dagger,a.require_cartesian_dagger,budget_config)
    if a.require_cartesian_dagger:
        collected=d['cartesian_dagger_config']
        if collected is None or collected['max_recovery_travel_m'] != a.recovery_travel_m:
            raise ValueError('Requested recovery budget does not match collected Cartesian DAgger data')
    ti,vi=grouped_split(d["scene_hash"],a.split_seed)
    mk=d["valid"].copy()
    available=int(mk[:,ti].sum())
    if a.label_budget:
        if a.label_budget>available: raise ValueError("Insufficient valid labels for requested budget")
        train=np.zeros_like(mk);train[:,ti]=mk[:,ti]
        indexes=np.flatnonzero(train)
        keep=np.random.default_rng(a.seed+271828).choice(indexes,a.label_budget,replace=False)
        train[:]=0;train.flat[keep]=1
        mk[:,ti]=train[:,ti]
    sm=d['stop_valid'].copy()
    stop_available=int(sm[:,ti].sum())
    if a.stop_label_budget:
        if a.stop_label_budget>stop_available: raise ValueError('Insufficient graspability labels')
        train=np.zeros_like(sm);train[:,ti]=sm[:,ti]
        keep=np.random.default_rng(a.seed+314159).choice(np.flatnonzero(train),a.stop_label_budget,replace=False)
        train[:]=0;train.flat[keep]=1;sm[:,ti]=train[:,ti]
    active=mk+(sm if a.termination_head else 0)
    ti=ti[active[:,ti].sum(0)>0];vi=vi[active[:,vi].sum(0)>0]
    if not len(ti) or not len(vi): raise ValueError("No valid train or validation sequences")
    torch.set_num_threads(4)
    torch.manual_seed(a.seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    if str(a.device).startswith("cuda"): torch.cuda.set_device(torch.device(a.device))
    output_dims=dict(xy=2,eef_endpoint=2,eef_waypoints=4,categorical_kl=16,categorical_hard=16)
    net=StudentNet(n_actions=output_dims[a.head],termination_head=a.termination_head).to(a.device)
    if a.ablate=="no_gru": net=strip_recurrence(net)
    X=torch.tensor(d["obs"],device=a.device)
    mask=obs_mask(a.ablate,a.device)
    if mask is not None: X=X*mask
    L=torch.tensor(d["teacher_logits"],device=a.device)
    V=torch.tensor(d["teacher_value"],device=a.device)
    M=torch.tensor(mk,device=a.device)
    if budget_config is not None:
        from isaacgymenvs.open_loop.teacher_relative_budget import recovery_weight_numpy
        weight_np=recovery_weight_numpy(d['obs'],d['step_budget'],budget_config)
        W=torch.tensor(weight_np,device=a.device)
    else:
        W=torch.ones_like(M)
    S=torch.tensor(sm,device=a.device)
    G=torch.tensor(d['graspability'],device=a.device)
    P=torch.tensor(primitive_vectors(.04),device=a.device)
    if a.head == 'eef_waypoints':
        from isaacgymenvs.open_loop.plan_constraint import primitive_paths
        paths=torch.tensor(primitive_paths(.04)[:,1:].reshape(16,4),device=a.device,dtype=X.dtype)
        Y=paths[L.argmax(-1)]
    else:
        Y=P[L.argmax(-1)]
    opt=torch.optim.Adam(net.parameters(),lr=a.lr)
    ti=torch.tensor(ti,device=a.device);vi=torch.tensor(vi,device=a.device)
    source=Path(__file__).resolve()
    provenance=dict(protocol=PROTOCOL,configuration=vars(a),source_sha256=sha256(source),
        visibility_config=d['visibility_config'],simulator_oracle_experiment=a.allow_oracle_dagger,
        bounded_dagger_config=d['bounded_dagger_config'],
        cartesian_dagger_config=d['cartesian_dagger_config'],
        teacher_relative_budget_config=budget_config,
        manifest_sha256=sha256(a.manifest),data_sha256={str(p):sha256(p) for p in files},
        training_scenes=sorted(set(d["scene_hash"][ti.cpu().numpy()])),
        validation_scenes=sorted(set(d["scene_hash"][vi.cpu().numpy()])),
        available_training_labels=available,used_training_labels=int(M[:,ti].sum().item()),
        validation_labels=int(M[:,vi].sum().item()),trajectories=len(d["scene_hash"]),
        available_stop_labels=stop_available,used_stop_labels=int(S[:,ti].sum().item()) if a.termination_head else 0,
        training_positive_stop_labels=int(((G[:,ti]>.9)*S[:,ti]).sum().item()) if a.termination_head else 0,
        stop_loss='MSE of sigmoid prediction and clipped current graspability' if a.termination_head else None,
        stop_threshold=.9 if a.termination_head else None,
        action_loss_units=('teacher Cartesian waypoint MSE / 0.04^2 plus temporal teacher-relative-error smoothness'
                           if a.head in ('eef_endpoint','eef_waypoints') else
                           'squared displacement divided by 0.04^2 when training graspability jointly; categorical KL otherwise'),
        action_loss_reduction=('equal mean per trajectory' if a.trajectory_balanced_action_loss
                               else 'mean over valid action labels'),
        action_loss_weighting=('smooth union of nominal progress, teacher-path drift, and remaining travel/step slack'
                               if budget_config is not None else 'uniform'),
        action_weight_min=float(W[M>0].min().item()) if bool((M>0).any()) else None,
        action_weight_mean=float(W[M>0].mean().item()) if bool((M>0).any()) else None,
        action_weight_max=float(W[M>0].max().item()) if bool((M>0).any()) else None,
        gpu=torch.cuda.get_device_name() if str(a.device).startswith("cuda") else "cpu")
    (out/"provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")
    print(f"FIT {a.head}: {len(ti)} train / {len(vi)} validation sequences; {provenance['used_training_labels']} train labels; {a.updates} optimizer updates",flush=True)
    def loss(sel):
        if a.termination_head:
            pred,val,sg,_=net.forward_with_stop(X[:,sel])
            stop_loss=((sg.sigmoid()-G[:,sel]).square()*S[:,sel]).sum()/S[:,sel].sum().clamp(min=1.)
        else:
            pred,val,_=net(X[:,sel]);stop_loss=X.new_tensor(0.)
        m=M[:,sel];den=m.sum().clamp(min=1.)
        if a.head in ("xy","eef_endpoint","eef_waypoints"):
            per=(pred-Y[:,sel]).square().sum(-1)
            if a.termination_head or a.head.startswith('eef_'): per=per/(.04**2)
        elif a.head=="categorical_hard":
            per=-torch.log_softmax(pred,-1).gather(-1,L[:,sel].argmax(-1,keepdim=True)).squeeze(-1)
        else:
            lp=torch.log_softmax(L[:,sel],-1)
            per=(lp.exp()*(lp-torch.log_softmax(pred,-1))).sum(-1)
        act=reduce_masked(per,m,a.trajectory_balanced_action_loss,W[:,sel])
        if a.head.startswith('eef_') and pred.shape[0] > 1 and a.smoothness_coef:
            # Penalize changes in prediction ERROR, not changes in the teacher's
            # desired path. Turns in the nominal primitive sequence remain free;
            # only the learned Cartesian correction is encouraged to be smooth.
            pair=m[1:]*m[:-1]
            error=pred-Y[:,sel]
            smooth=reduce_masked((error[1:]-error[:-1]).square().sum(-1),pair,
                                 a.trajectory_balanced_action_loss)/(.04**2)
            act=act+a.smoothness_coef*smooth
        val_loss=((val-V[:,sel]).square()*m).sum()/den
        return act+a.value_coef*val_loss+a.stop_coef*stop_loss,act,val_loss,stop_loss
    started=time.monotonic();cursor=len(ti);order=ti
    history=[];best_loss=float('inf');best_state=None;best_update=None
    for update in range(a.updates):
        if cursor>=len(order): order=ti[torch.randperm(len(ti),device=a.device)];cursor=0
        sel=order[cursor:cursor+a.batch_seqs];cursor+=len(sel)
        net.train();total,act,vl,sl=loss(sel)
        if not torch.isfinite(total): raise RuntimeError("Non-finite fitting loss")
        opt.zero_grad();total.backward();opt.step()
        if (update+1)%500==0 or update+1==a.updates:
            net.eval()
            with torch.no_grad(): _,validation,vv,vs=loss(vi)
            row=dict(update=update+1,train_action=float(act),validation_action=float(validation),
                validation_value=float(vv),train_graspability=float(sl),validation_graspability=float(vs),seconds=time.monotonic()-started)
            history.append(row);print(json.dumps(row),flush=True)
            (out/"history.json").write_text(json.dumps(history,indent=2)+"\n")
            if a.select_best_validation and float(validation)<best_loss:
                best_loss=float(validation);best_update=update+1
                best_state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
    if sha256(source)!=provenance["source_sha256"]: raise RuntimeError("Trainer changed during fitting")
    if a.select_best_validation:
        if best_state is None: raise RuntimeError('No finite validation checkpoint')
        net.load_state_dict(best_state)
    (out/'selection.json').write_text(json.dumps(dict(
        rule='minimum grouped training-scene validation action loss' if a.select_best_validation else 'last update',
        update=best_update if a.select_best_validation else a.updates,
        validation_loss=best_loss if a.select_best_validation else history[-1]['validation_action'],
        development_used=False),indent=2)+'\n')
    cartesian_policy=None
    if a.head.startswith('eef_'):
        from isaacgymenvs.open_loop.teacher_relative_cartesian import config as cartesian_config
        cartesian_policy=cartesian_config(a.head,recovery_travel_m=a.recovery_travel_m)
    torch.save(dict(model=net.state_dict(),obs_dim=166,head=a.head,protocol=PROTOCOL,
                    ablate=a.ablate,seed=a.seed,smoke=a.smoke,
                    visibility_config=d['visibility_config'],simulator_oracle_experiment=a.allow_oracle_dagger,
                    bounded_dagger_config=d['bounded_dagger_config'],
                    teacher_relative_budget=budget_config,
                    cartesian_policy=cartesian_policy,
                    termination_head=a.termination_head,stop_threshold=.9 if a.termination_head else None,
                    stop_target="current_graspability" if a.termination_head else None),out/"student.pt")
    (out/"complete.json").write_text(json.dumps(dict(complete=True,updates=a.updates,
        checkpoint_sha256=sha256(out/"student.pt"),seconds=time.monotonic()-started),indent=2)+"\n")
    print("FIT COMPLETE",out,flush=True)


if __name__=="__main__": main()
