"""Collect paired occluded images/clear labels or run a recurrent image judge."""
from pathlib import Path
import json
import hashlib
import numpy as np
import torch
from isaacgymenvs.open_loop.grasp_image_obs import TargetHistory,augment_missing
from isaacgymenvs.open_loop.evaluation_core import scene_seed
from isaacgymenvs.learning.recurrent_grasp import load_recurrent_grasp


class ImageGraspObserver:
    def __init__(self,env,rows,plans,case,spec,out_dir,terminal_failed=None):
        self.env,self.rows,self.plans=env,rows,plans
        self.options=spec.get('grasp_images',{})
        self.directory=Path(out_dir)/'grasp_images'
        self.directory.mkdir(parents=True)
        self.buffers=[[] for _ in rows]
        self.rngs=[np.random.default_rng(scene_seed(r['sha256'],int(spec['seed']),
            self.options.get('rng_stream','grasp_image'))) for r in rows]
        self.blackouts=[int(rng.integers(1,max(2,min(len(p['actions'])+1,21)))) for rng,p in zip(self.rngs,plans)]
        self.hidden=None
        self.net=self.checkpoint=None
        if case.get('image_grasp_checkpoint'):
            self.net,self.checkpoint=load_recurrent_grasp(case['image_grasp_checkpoint'],env.device)
        self.initial_depth,self.initial_seg=self.capture(clear=True)
        self.terminal_failed=(np.zeros(len(rows),bool) if terminal_failed is None else np.asarray(terminal_failed,bool))
        if self.terminal_failed.shape!=(len(rows),):raise ValueError('Initial failure mask has wrong shape')
        self.initial_unavailable=np.array([not np.any(s==255) for s in self.initial_seg])
        if np.any(self.initial_unavailable&~self.terminal_failed):
            raise ValueError('A clean initial target detection is required for a live scene')
        # An already-failed scene remains in the denominator and has a single
        # negative terminal record. It gets no invented target location or
        # live recurrent memory; all active scenes still require a detection.
        self.trackers=[None if missing else TargetHistory(d,s==255)
            for d,s,missing in zip(self.initial_depth,self.initial_seg,self.initial_unavailable)]
        self.image_memories=None
        if self.net is not None and self.net.architecture in ('memory_residual_gru_v1','spatial_memory_gru_v1'):
            from isaacgymenvs.open_loop.grasp_image_memory import CausalCropMemory,CANVAS_SIZE
            if self.initial_depth.shape[-2:]!=(CANVAS_SIZE,CANVAS_SIZE):
                raise ValueError('Depth-memory canvas resolution differs from training')
            self.image_memories=[None if missing else CausalCropMemory() for missing in self.initial_unavailable]
        self.natural_target_hidden=0
        self.missing_target_frames=0
        self.valid_frames=0
        self.max_robot_fraction=0.
        self.case=case
        # Extra recording streams follow the SAME physical trajectory, but each
        # has its own causal last-detection tracker. Masking already-cropped
        # images would leak target locations observed only in another stream.
        self.variants=[]
        for j,options in enumerate(self.options.get('record_variants',[])):
            variant_spec=dict(spec,grasp_images=dict(options,enabled=True,
                rng_stream='grasp_image_variant_%d'%j))
            variant_case={k:v for k,v in case.items() if k!='image_grasp_checkpoint'}
            self.variants.append(ImageGraspObserver(env,rows,plans,variant_case,
                variant_spec,Path(out_dir)/('augmentation_%d'%j),terminal_failed=self.terminal_failed))

    def capture(self,clear=False):
        env=self.env
        depths=env.cam_depth_tensors if clear else env.occluded_depth_tensors
        segments=env.cam_segm_tensors if clear else env.occluded_seg_tensors
        if len(depths)!=len(self.rows):raise ValueError('Occluded camera was not enabled')
        env.gym.start_access_image_tensors(env.sim)
        try:
            depth=torch.stack(depths).detach().cpu().numpy().copy()
            seg=torch.stack(segments).detach().cpu().numpy().copy()
        finally:env.gym.end_access_image_tensors(env.sim)
        return depth,seg

    def observe(self,t,valid,previous_action,eef,q,bad,captured=None):
        depth,seg=captured if captured is not None else ((self.initial_depth,self.initial_seg) if t==0 else self.capture())
        for variant in self.variants:
            variant.observe(t,valid,previous_action,eef,q,bad,captured=(depth,seg))
        images,masks,metas=[],[],[]
        for i,(tracker,d,s) in enumerate(zip(self.trackers,depth,seg)):
            robot=s==1
            if tracker is None:
                if valid[i] and (t!=0 or not (bad[i] or q[i]==-2. or not np.isfinite(q[i]))):
                    raise ValueError('An unavailable initial target may only record an already-failed endpoint')
                image=np.zeros((2,112,112),np.float32)
                mask=np.zeros((2,2,112,112),np.uint8);mask[:,1]=1
                meta=np.zeros(26,np.float32);meta[:2]=eef[i]
            else:
                unknown=augment_missing(robot,tracker.center,self.rngs[i],t,self.blackouts[i],
                    int(self.options.get('blackout_length',5)),float(self.options.get('target_hide_probability',.2)))
                image,mask,meta=tracker.encode(d,s==255,unknown,np.asarray(eef[i]),int(previous_action[i]),t,self.plans[i]['xy'])
            images.append(image);masks.append(mask);metas.append(meta)
            if valid[i]:
                target=float(np.clip(q[i],0,1)) if np.isfinite(q[i]) and not bad[i] else 0.
                self.buffers[i].append(dict(images=image.astype(np.float16),masks=mask.astype(np.uint8),meta=meta,
                    graspability=target,success_label=bool(target>.9),decision=t))
                self.valid_frames+=1
                self.missing_target_frames+=int(meta[20]==0)
                self.natural_target_hidden+=int(not np.any(s==255))
                self.max_robot_fraction=max(self.max_robot_fraction,float(robot.mean()))
        if self.net is None:return None
        if self.image_memories is not None:
            remembered=[(i,m) if memory is None else memory.observe(i,m,s)
                for memory,i,m,s in zip(self.image_memories,images,masks,metas)]
            images=[item[0] for item in remembered];masks=[item[1] for item in remembered]
        with torch.no_grad():
            logits,self.hidden=self.net(torch.tensor(np.stack(images),device=self.env.device)[None],
                torch.tensor(np.stack(masks),device=self.env.device)[None],
                torch.tensor(np.stack(metas),device=self.env.device)[None],self.hidden)
        return logits[0].sigmoid().cpu().numpy()

    def finish(self,terminal_rows):
        for variant in self.variants:variant.finish(terminal_rows)
        output=[]
        for i,(frames,row,terminal) in enumerate(zip(self.buffers,self.rows,terminal_rows)):
            if len(frames)!=terminal['terminal_step']+1:raise ValueError('Missing image endpoint/history')
            if self.initial_unavailable[i] and (terminal['terminal_step']!=0 or
                    terminal['reason'] not in ('oow','out_of_view','invalid_state')):
                raise ValueError('Unavailable initial target was not an absorbing initial failure')
            path=self.directory/(row['sha256']+'.npz')
            arrays={k:np.stack([frame[k] for frame in frames]) for k in frames[0]}
            np.savez_compressed(path,**arrays,scene_hash=np.array(row['sha256']),
                protocol=np.array('occluded-image-grasp-v1'),terminal_reason=np.array(terminal['reason']))
            output.append(dict(path=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                scene_sha256=row['sha256'],frames=len(frames),positives=int(arrays['success_label'].sum()),
                initial_target_unavailable_terminal=bool(self.initial_unavailable[i])))
        report=dict(complete=True,protocol='occluded-image-grasp-v1',sequences=output,
            image_options=self.options,
            valid_frames=self.valid_frames,missing_target_frames=self.missing_target_frames,
            natural_target_hidden_frames=self.natural_target_hidden,max_robot_pixel_fraction=self.max_robot_fraction,
            initial_view='Unoccluded initial observation only',
            current_inputs='Robot-visible depth/target detection, explicit missing pixels, last-seen target center, action/EEF history, nominal EEF trajectory',
            supervision='Original frozen clear-view classifier; never an image-network input',
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        (self.directory/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
