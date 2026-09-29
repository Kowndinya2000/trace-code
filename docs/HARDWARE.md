# Running on a robot

Simulation needs none of this. The pipeline is included so that a lab with the same setup can
run the method and reproduce the recordings.

## Bill of materials

| Item | Notes |
|---|---|
| UR5e arm | reachable workspace covering a 0.448 m square in front of the base |
| Robotiq 2F-85 gripper | stock fingertips; the TCP is set at the fingertip plane |
| Intel RealSense D455 | mounted looking straight down, about 0.64 m above the table |
| Webcam (optional) | third-person view for review videos only |
| Matte black mat | defines the workspace; the controllers stop if an object crosses its edge |
| Eleven printed blocks | six shapes; the target is a coloured cylinder. URDFs and meshes ship with the data payload |
| ChArUco board | 4×4 legacy pattern, 25 mm squares, 18.75 mm markers, rigidly bolted to the gripper for calibration |

The camera sees the whole mat and the robot never occludes it entirely; that partial visibility is
exactly what the student is trained for.

## 1. Calibration (eye-to-hand)

For the legacy ChArUco interpolation helpers, replace the simulation OpenCV wheel with
the matching contrib wheel (do not keep both installed):

```bash
python -m pip uninstall -y opencv-python
python -m pip install --no-deps opencv-contrib-python==4.10.0.84
```

This adds the legacy calibration functions while retaining the modern ArUco detector API.
Keep this replacement as the final environment-installation step on the hardware machine.

Bolt the ChArUco board to the gripper, clear the mat, and run:

```bash
python trace/hardware/pmbs_calibrate_eye_to_hand.py --plan        # show the sampled poses, no motion
python trace/hardware/pmbs_calibrate_eye_to_hand.py --collect     # 36 poses, ~8 min
```

The robot visits a grid above the mat, detects the board in each frame, solves hand-eye and then
refines the camera pose and the board offset jointly. It writes a candidate calibration and a
report; it never overwrites the deployed one. Inspect pixel reprojection error, board-origin
error, and orientation error on both fitting and held-out poses before deployment.

Held-out error close to training error means the fit is not overfitting. Deploy it by pointing
`TRACE_CALIB` at the candidate matrix, then remove the board from the gripper.

`--collect --manual` records operator-positioned poses instead, and `--replay DATASET` re-solves a
saved set without touching the robot.

## 2. Workspace geometry

The simulator frame and the robot base frame differ by a rotation: `real_x = sim_y`,
`real_y = −sim_x`. The workspace is a 0.448 m square centred 0.5 m in front of the base, and the
mat is slightly larger than that square. Perception rasterises the scene into a 320 px canvas at
2 mm per pixel. These constants live in `trace/sim/isaacgymenvs/open_loop/frames.py` and must match the
physical setup.

## 3. Running a controller

```bash
export TRACE_ROBOT_IP=192.168.1.102     # your arm
export TRACE_D455_SERIAL=XXXXXXXXXXXX   # top-down perception camera
# export TRACE_D415_SERIAL=XXXXXXXXXXXX # optional third-person RealSense
export TRACE_CALIB=/path/to/camera_to_base.txt
export TRACE_MASKRCNN=/path/to/maskrcnn.pth # train for your blocks with train_maskrcnn.py
export TRACE_RUNS=/path/to/output

trace/hardware/run_student_demo.sh   $TRACE_RUNS/trial --execute   # TRACE, closed loop
trace/hardware/run_real_demo.sh      $TRACE_RUNS/trial --execute   # teacher plan, open-loop replay
trace/hardware/run_spiral_demo.sh    $TRACE_RUNS/trial --execute   # spiral baseline
trace/hardware/run_pmbs_demo.sh      $TRACE_RUNS/trial --execute   # parallel-search baseline
trace/hardware/run_teacher_full_obs_demo.sh $TRACE_RUNS/trial --execute   # privileged reference
```

Each launcher starts the cameras, perceives the scene once, solves the twin where the method needs
it, runs the controller, evaluates the grasp and parks the arm. A trial directory holds the
per-step perception records, the phase log, the timing record and both camera recordings.

Safety notes, all enforced in code: pushes are issued as guarded moves that stop on contact force,
the run aborts if any object's footprint crosses the mat edge, and the result is written before the
arm returns home so an interrupted homing cannot lose it. Check `goto_corner.py HOME` brings the
arm to a safe parked pose before the first run.

## 4. Review videos

Launchers produce `review.mp4` automatically after the cameras stop and the twin is rendered.
Set `SKIP_POSTPROCESS=1` to defer composition; for TRACE/Teacher Replay, also set this when
using `SKIP_RENDER=1`. Install system `ffmpeg`/`ffprobe` and the `fonts-lato` package first.
Set `TRACE_WEBCAM=/dev/videoN` to select a webcam. A configured D415 is the fallback;
without either camera the third-person panel is explicitly labeled as not connected.
The release contains no recordings, camera dumps, calibration measurements, or trial logs.
Calibration and segmentation weights must be supplied for the receiving lab's setup.

To fit a local segmenter, annotate still images with single-channel instance masks and
create a JSON list such as `[{"image":"rgb.png","mask":"instances.png","labels":{"1":2}}]`.
Here instance 1 is a cube (class 2); the full class map is `CLASS_ID_TO_NAME` in
`open_loop/perceive_scene.py`. Run:

```bash
python trace/hardware/train_maskrcnn.py --manifest /path/to/train.json \
  --output /path/to/maskrcnn.pth
```

Use `--initialize /path/to/compatible.pth` for fine-tuning. Validate segmentation on separate
local images before robot use; this utility exports the final training epoch, not a
validation-selected model. Point `TRACE_MASKRCNN` to the resulting checkpoint.

```bash
V=trace/hardware/video_postprocess
python $V/compose_closed_loop_memory.py  <trial> <out.mp4>   # TRACE
python $V/compose_teacher_closed_loop.py <trial> <out.mp4>   # Online Teacher
python $V/compose_open_loop.py           <trial> <out.mp4>   # Teacher Replay
python $V/compose_spiral_closed_loop.py  <trial> <out.mp4>   # Spiral and PMBS (auto-detected)
```

`compose_closed_loop.py` is the shared layout module the four composers import, not an entry
point of its own.

Each composer builds a single annotated view: the overhead camera with the nominal plan and the
executed path drawn on it, what the controller actually observed, the digital twin, the
third-person view and a phase dashboard. Camera panels are exposure-corrected by default
(`--exposure measured`, the transfer in `exposure.py`): the gamma is fixed per camera and a
per-clip gain is measured once from sampled frames, so brightness is uniform over the whole
video and highlights roll off smoothly instead of clipping. `--exposure fixed` selects the older
constant-gamma path and is only there for comparison.
