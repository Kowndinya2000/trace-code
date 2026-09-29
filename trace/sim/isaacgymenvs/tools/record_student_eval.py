"""Record a fresh retrieval evaluation using the existing simulator color view.

Wrap only rendering around existing evaluator/physics calls. The original
evaluation still writes its complete trace, first-terminal ledger and provenance.
Color frames do not feed the student; classifier depth/segmentation stay unchanged.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import isaacgym  # Must precede torch.
import evaluate_retrieval as evaluation
import cv2
import json
import numpy as np
import hashlib
import hydra
import os


class Recorder:
    def __init__(self, env, folder, indices):
        self.env = env
        self.folder = folder
        folder.mkdir()
        self.enabled = False
        self.decision = -1
        self.frames = []
        self.writers = []
        self.indices = indices
        self.tensors = [env.cam_color_tensors[i] for i in indices]
        for i, tensor in enumerate(self.tensors):
            height, width = tensor.shape[:2]
            writer = cv2.VideoWriter(str(folder / f'scene{i:02d}.mp4'),
                                    cv2.VideoWriter_fourcc(*'mp4v'), 15, (width, height))
            if not writer.isOpened():
                raise RuntimeError('Could not open video writer')
            self.writers.append(writer)
        if len(self.writers) != len(indices):
            raise ValueError('Every selected environment requires a color camera')

    def capture(self, endpoint=False):
        if not self.enabled:
            return
        env = self.env
        # Once every filmed scene has reached its first terminal event, finish
        # the other evaluation environments without recording redundant frames.
        if bool(env.frozen[self.indices].all()):
            return
        env.gym.fetch_results(env.sim, True)
        env.gym.step_graphics(env.sim)
        env.gym.render_all_camera_sensors(env.sim)
        env.gym.start_access_image_tensors(env.sim)
        try:
            images = [tensor.clone().cpu().numpy()[:, :, :3] for tensor in self.tensors]
        finally:
            env.gym.end_access_image_tensors(env.sim)
        for writer, image in zip(self.writers, images):
            writer.write(np.ascontiguousarray(image[:, :, ::-1]))
        self.frames.append(dict(decision=self.decision, endpoint=endpoint,
            eef={str(i):env.gripper_pos[i,:2].detach().cpu().tolist() for i in self.indices}))

    def close(self, case, rows, spec):
        for writer in self.writers:
            writer.release()
        payload = dict(case=case, scenes=[rows[i] for i in self.indices], indices=self.indices,
            simulator_batch_size=self.env.num_envs, frames=self.frames, fps=15,
            physics_dt=float(self.env.cfg['sim']['dt']),
            control_frequency_inv=int(self.env.control_freq_inv),
            camera=dict(resolution=list(self.tensors[0].shape[:2]),
                        role='simulator top view; color pixels are never supplied to the student'),
            recorder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            notes='Fresh checkpoint evaluation; existing color view recorded at physics substeps and decision endpoints. '
                  'Playback is slowed for inspection and is not wall-clock execution time. '
                  'Detection dropout is applied to object tokens, not rendered pixels.')
        (self.folder / 'recording.json').write_text(json.dumps(payload, indent=2) + '\n')


original_run_case = evaluation.run_case


def recorded_case(player, case, initial, permutations, plans, rows, spec, out_dir):
    env = player.env
    indices = list(spec.get('video_indices', [0,1,2]))
    recorder = Recorder(env, out_dir / 'video_raw', indices)
    original_begin = evaluation.begin
    original_step = player.env_step
    original_substep = env._apply_plan_target_one_substep

    def begin(*args, **kwargs):
        result = original_begin(*args, **kwargs)
        recorder.enabled = True
        recorder.capture(endpoint=True)
        return result

    def substep(*args, **kwargs):
        result = original_substep(*args, **kwargs)
        recorder.capture()
        return result

    def step(*args, **kwargs):
        recorder.decision += 1
        result = original_step(*args, **kwargs)
        recorder.capture(endpoint=True)
        return result

    evaluation.begin = begin
    player.env_step = step
    env._apply_plan_target_one_substep = substep
    try:
        return original_run_case(player, case, initial, permutations, plans, rows, spec, out_dir)
    finally:
        evaluation.begin = original_begin
        player.env_step = original_step
        env._apply_plan_target_one_substep = original_substep
        recorder.close(case, rows, spec)


@hydra.main(version_base='1.1', config_name='config', config_path=str(evaluation.ROOT / 'cfg'))
def main(cfg):
    cfg.task.env.videoLog.enabled = True
    cfg.task.env.videoLog.firstStart = 10**9
    cfg.task.env.videoLog.vizRes = 0
    evaluation.run_case = recorded_case
    evaluation.main.__wrapped__(cfg)


if __name__ == '__main__':
    # The task's inherited classifier paths are relative to the package root.
    os.chdir(evaluation.ROOT)
    main()
