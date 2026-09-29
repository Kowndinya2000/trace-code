"""Train a fresh minimal-reward teacher and save the prescribed final update."""
import os
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import isaacgym  # Preview 4 must be imported before torch.
import torch
import numpy as np
import hydra
from omegaconf import OmegaConf,open_dict
import json
import shutil
import time
import platform
from isaacgymenvs.open_loop.evaluation_core import read_manifest,sha256
from isaacgymenvs.utils.utils import set_seed
from isaacgymenvs.utils.reformat import omegaconf_to_dict
from isaacgymenvs.utils.rlgames_utils import RLGPUEnv,RLGPUAlgoObserver
from rl_games.common import vecenv,env_configurations
from rl_games.torch_runner import Runner
import tools._net_compat
import isaacgymenvs
from minimal_inference import ChunkedGraspInference

ROOT=Path(__file__).resolve().parents[1]
ZERO_TERMS=['lambdaC','lambdaC3','lambdaArc','lambdaB','lambdaEef','lambdaDisturb']


@hydra.main(version_base='1.1',config_name='config',config_path='../cfg')
def main(cfg):
    spec=json.loads(Path(cfg.teacher_training_spec).read_text())
    out=Path(spec['output']).resolve()
    if out.exists(): raise FileExistsError('Fresh teacher run required: '+str(out))
    manifest=read_manifest(spec['manifest'])
    if manifest['role']!='training': raise ValueError('Teacher requires a training-role manifest')
    smoke=bool(spec.get('smoke',False))
    n=int(spec.get('num_envs',1024));epochs=int(spec.get('epochs',450));seed=int(spec['seed'])
    if not smoke and (n!=1024 or epochs!=450): raise ValueError('Production budget is 1024 environments and 450 PPO epochs')
    if cfg.checkpoint: raise ValueError('Teacher must train from scratch')
    rows=sorted(manifest['scenes'],key=lambda r:int(Path(r['path']).stem))[:n]
    if len(rows)!=n: raise ValueError('Insufficient unique teacher training scenes')
    out.mkdir(parents=True);scenes=out/'scenes';scenes.mkdir()
    for i,row in enumerate(rows): shutil.copyfile(row['path'],scenes/f'{i:06d}.txt')
    with open_dict(cfg):
        cfg.seed=seed;cfg.test=False;cfg.wandb_activate=False
        cfg.task.env.numEnvs=n;cfg.task.env.test_cases.scene_root_dir=os.path.relpath(out,ROOT)
        cfg.task.env.test_cases.difficulty_choice='scenes';cfg.task.env.test_cases.sceneOffset=0
        cfg.train.params.seed=seed;cfg.train.params.load_checkpoint=False;cfg.train.params.load_path=''
        config=cfg.train.params.config
        config.device=cfg.rl_device;config.name=f'graspability_teacher_s{seed}'
        config.full_experiment_name='ppo';config.train_dir=str(out)
        config.num_actors=n;config.max_epochs=epochs;config.save_frequency=15
        if smoke:
            config.horizon_length=32;config.minibatch_size=n*32;config.mini_epochs=1;config.seq_len=32
    tc=cfg.task.env.teacher
    if any(float(tc[k])!=0 for k in ZERO_TERMS): raise ValueError('Excluded reward term has a nonzero coefficient')
    if float(tc.lambdaG)!=2 or float(tc.lambdaStep)!=.1: raise ValueError('Unexpected reward coefficients')
    if (float(tc.rSuccess),float(tc.rOow),float(tc.rOutOfView))!=(10.,12.,11.): raise ValueError('Unexpected terminal rewards')
    if float(tc.gamma)!=float(cfg.train.params.config.gamma): raise ValueError('PPO/potential discount mismatch')
    if (float(tc.arcAlignProb),float(cfg.task.env.robust.poseNoisePos),
        float(cfg.task.env.robust.poseNoiseYawDeg),float(cfg.train.params.config.entropy_coef))!=(0.,.012,10.,.02):
        raise ValueError('Preserve the saved original teacher settings outside the reward')
    (out/'configuration.yaml').write_text(OmegaConf.to_yaml(cfg,resolve=True))
    watched={Path(__file__).resolve(),ROOT/'tools/minimal_inference.py'}
    for folder in ['tasks','learning','open_loop','utils']:
        watched.update((ROOT/folder).rglob('*.py'))
    watched.update((ROOT/'cfg').rglob('*.yaml'))
    watched.update((ROOT.parent/'assets/urdf/more/blocks-more').glob('*.obj'))
    watched.update([ROOT/'logs_grasp/grasp_model-89.pth',ROOT/'logs_grasp/snapshot-post-020000.reinforcement.pth'])
    frozen={str(p):sha256(p) for p in sorted(watched)}
    (out/'frozen_inputs.json').write_text(json.dumps(frozen,indent=2)+'\n')
    set_seed(seed,False)
    torch.set_num_threads(4);torch.cuda.set_device(torch.device(cfg.rl_device))
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(False)  # legacy reset assignment, as in the audited evaluator
    created=[]
    def make_env(**kwargs):
        env=isaacgymenvs.make(seed=seed,task=cfg.task_name,test=False,num_envs=n,
            sim_device=cfg.sim_device,rl_device=cfg.rl_device,graphics_device_id=cfg.graphics_device_id,
            headless=True,multi_gpu=False,virtual_screen_capture=False,force_render=False,cfg=cfg)
        created.append(env)
        return env
    vecenv.register('RLGPU',lambda name,num,**kw:RLGPUEnv(name,num,**kw))
    env_configurations.register('rlgpu',dict(vecenv_type='RLGPU',env_creator=make_env))
    runner=Runner(RLGPUAlgoObserver());runner.load(omegaconf_to_dict(cfg.train));runner.reset()
    agent=runner.algo_factory.create(runner.algo_name,base_name='run',params=runner.params)
    env=created[0]
    env.mcts_helper.grasp_eval_model=ChunkedGraspInference(env.mcts_helper.grasp_eval_model,
        out/'inference_validation.json',batch_size=64)
    attributes=['t_lam_c','t_lam_c3','t_lam_arc','t_lam_b','t_lam_eef','t_lam_disturb']
    if any(getattr(env,k)!=0 for k in attributes): raise ValueError('Runtime reward differs from declared minimal reward')
    phi,_=env._phi_terms()
    torch.testing.assert_close(phi,2*env.grasp_q_parallel_values.clamp(0,1))
    provenance=dict(protocol='graspability-only-teacher-v1',spec=spec,source_sha256=frozen,
        scene_rows=rows,manifest_sha256=sha256(spec['manifest']),gpu=torch.cuda.get_device_name(),
        host=platform.node(),torch_version=torch.__version__,cuda_version=torch.version.cuda,
        reward=OmegaConf.to_container(tc,resolve=True),selection='Final scheduled epoch; fresh initialization; no score selection',
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),graphics_device_id=int(cfg.graphics_device_id),
        grasp_inference_batch_size=64)
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    started=time.monotonic();print('MINIMAL REWARD VERIFIED; STARTING PPO',flush=True)
    agent.train()
    if int(agent.epoch_num)!=epochs: raise RuntimeError('Teacher did not reach the prescribed final epoch')
    if any(sha256(p)!=h for p,h in frozen.items()): raise RuntimeError('Teacher inputs changed during training')
    checkpoint=out/'teacher.pth';torch.save(agent.get_full_state_weights(),checkpoint)
    done=dict(complete=True,smoke=smoke,seed=seed,epochs=epochs,checkpoint_sha256=sha256(checkpoint),
        seconds=time.monotonic()-started,reward='graspability progress + step cost + terminal events')
    (out/'complete.json').write_text(json.dumps(done,indent=2)+'\n')
    torch.cuda.synchronize();print('TEACHER TRAINING COMPLETE',json.dumps(done),flush=True)
    sys.stdout.flush();sys.stderr.flush();os._exit(0)


if __name__=='__main__': main()
