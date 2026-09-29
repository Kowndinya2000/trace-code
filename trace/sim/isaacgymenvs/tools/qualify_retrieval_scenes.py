"""Policy-independent stationary-arm physics qualification of candidate scenes.

No teacher or student policy is created or queried. The depth graspability
classifier defines the same task proxy as evaluation. Candidate files remain
unchanged; the accepted manifest selects original, hashed files.
"""
import os
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import isaacgym  # must precede torch
import tools._net_compat
import tools._ckpt_compat
import hydra
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
import json
import shutil
import numpy as np
import torch
from isaacgymenvs.open_loop.evaluation_core import read_manifest,sha256

ROOT=Path(__file__).resolve().parents[1]
RULES=dict(hold_decisions=30,max_xy_motion_m=.002,max_tilt_deg=5.,
           max_final_linear_speed_m_s=.02,max_final_angular_speed_rad_s=.2,
           max_graspability=.9,require_containment_all_hold_frames=True)


def cpu(x): return x.detach().cpu().numpy().copy()


@hydra.main(config_name="config",config_path="../cfg")
def main(cfg):
    import isaacgymenvs
    from isaacgymenvs.utils.utils import set_seed
    spec=json.loads(Path(to_absolute_path(str(cfg.evaluation_spec))).read_text())
    manifest=read_manifest(spec['manifest'])
    if manifest['role']!='independent_final_candidates': raise ValueError('Requires independent candidates')
    rows=manifest['scenes'][spec['offset']:spec['offset']+spec['batch_size']]
    if not rows: raise ValueError('Empty batch')
    out=Path(spec['output']).resolve()
    if out.exists(): raise FileExistsError(out)
    scene_dir=out/'scenes';scene_dir.mkdir(parents=True)
    for i,row in enumerate(rows): shutil.copyfile(row['path'],scene_dir/f'{i:06d}.txt')
    cfg.num_envs=len(rows);cfg.task.env.numEnvs=len(rows)
    cfg.task.env.test_cases.scene_root_dir=os.path.relpath(out,ROOT)
    cfg.task.env.test_cases.difficulty_choice='scenes';cfg.task.env.test_cases.sceneOffset=0
    if cfg.task.name!='MoreEvaluation': raise ValueError('Requires MoreEvaluation')
    cfg.seed=20260907
    torch.cuda.set_device(torch.device(cfg.rl_device));torch.set_num_threads(4)
    set_seed(cfg.seed,False)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    env=isaacgymenvs.make(cfg.seed,cfg.task_name,cfg.test,cfg.task.env.numEnvs,
        cfg.sim_device,cfg.rl_device,cfg.graphics_device_id,cfg.headless,
        cfg.multi_gpu,cfg.capture_video,cfg.force_render,cfg)
    sources=['tools/qualify_retrieval_scenes.py','tasks/more_evaluation.py',
             'tasks/more_open_loop.py','tasks/more_teacher.py','tasks/more_robust.py',
             'tasks/more.py','tasks/base/vec_task.py','open_loop/evaluation_core.py']
    hashes={p:sha256(ROOT/p) for p in sources}
    initial=env.pristine_block_state.clone();initial[:,:,7:]=0
    env.begin_attempt(initial,np.tile(np.arange(10),(len(rows),1)))
    env.freeze(torch.arange(len(rows),device=env.device))
    resets=cpu(env.explicit_resets)
    states=[cpu(env.block_state)];qualities=[cpu(env.grasp_q_parallel_values)]
    oow=[cpu(env.physical_oow()|env.step_mesh_oow|env.requested_initial_oow)]
    for _ in range(RULES['hold_decisions']):
        env.step(torch.zeros(len(rows),dtype=torch.long,device=env.device))
        states.append(cpu(env.block_state));qualities.append(cpu(env.grasp_q_parallel_values))
        oow.append(cpu(env.physical_oow()|env.step_mesh_oow))
    if not np.array_equal(resets,cpu(env.explicit_resets)): raise RuntimeError('Reset during qualification')
    states=np.stack(states);q=np.stack(qualities);oow=np.stack(oow)
    xy_motion=np.linalg.norm(states[:,:,:,:2]-states[0:1,:,:,:2],axis=-1).max(axis=(0,2))
    # Angle between each object's local z axis and the world z axis.
    tilt=np.rad2deg(np.arccos(np.clip(1-2*np.square(states[:,:,:,3:5]).sum(-1),-1,1))).max(axis=(0,2))
    speed=np.linalg.norm(states[-1,:,:,7:10],axis=-1).max(1)
    spin=np.linalg.norm(states[-1,:,:,10:13],axis=-1).max(1)
    decisions=[]
    for i,row in enumerate(rows):
        reasons=[]
        if not np.isfinite(states[:,i]).all() or not np.isfinite(q[:,i]).all(): reasons.append('invalid_state')
        if oow[:,i].any(): reasons.append('workspace_violation')
        if (q[:,i]<0).any(): reasons.append('target_out_of_view')
        if (q[:,i]>RULES['max_graspability']).any(): reasons.append('graspable_without_action')
        if xy_motion[i]>RULES['max_xy_motion_m']: reasons.append('spontaneous_xy_motion')
        if tilt[i]>RULES['max_tilt_deg']: reasons.append('nonplanar_object')
        if speed[i]>RULES['max_final_linear_speed_m_s'] or spin[i]>RULES['max_final_angular_speed_rad_s']:
            reasons.append('unsettled_velocity')
        decisions.append(dict(scene=row,accepted=not reasons,reasons=reasons,
            initial_q=float(q[0,i]),max_q=float(q[:,i].max()),max_xy_motion_m=float(xy_motion[i]),
            max_tilt_deg=float(tilt[i]),final_max_linear_speed=float(speed[i]),final_max_angular_speed=float(spin[i])))
    trace=out/'qualification.npz';np.savez_compressed(trace,state=states,q=q,oow=oow)
    result=dict(complete=True,policy_queries=0,rules=RULES,rows=decisions,trace_sha256=sha256(trace),
        source_sha256=hashes,manifest_sha256=sha256(spec['manifest']),spec=spec,
        configuration=OmegaConf.to_container(cfg,resolve=True),reset_count_during_hold=0,
        gpu=torch.cuda.get_device_name(),torch_version=torch.__version__)
    torch.cuda.synchronize(env.device)
    if any(sha256(ROOT/p)!=h for p,h in hashes.items()): raise RuntimeError('Source changed during qualification')
    (out/'complete.json').write_text(json.dumps(result,indent=2)+'\n')
    print('QUALIFICATION COMPLETE',sum(r['accepted'] for r in decisions),'/',len(rows),flush=True)
    sys.stdout.flush();sys.stderr.flush();os._exit(0)


if __name__=='__main__': main()
