"""Encode annotated evaluation videos and a local, synchronized comparison gallery."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from open_loop.evaluation_core import token_order

LABELS = {'teacher': 'Teacher', 'replay': 'Open-loop replay', 'bc': 'BC', 'dagger_r1': 'DAgger round 1', 'dagger_r2': 'DAgger round 2', 'dagger_r3': 'DAgger round 3'}
CONDITIONS = {'default': '15% extra dropout (default)', 'd90': '90% extra dropout',
              'd100': '100% extra dropout', 'd0': '0% extra dropout'}
REASONS = {'success': 'SUCCESS: target graspable', 'oow': 'FAILURE: outside workspace',
           'horizon': 'TIMEOUT: 120 decisions', 'out_of_view': 'FAILURE: invalid state'}
WIDTH, HEIGHT, FPS = 896, 688, 15
BACKGROUND = (29, 23, 17)
WHITE, MUTED, CYAN, GOLD = (237, 232, 220), (160, 151, 136), (235, 206, 87), (70, 195, 245)
MASKED = (114, 130, 227)


def object_id(world_index):
    return 'T' if world_index == 0 else str(world_index)


def verify_object_mapping(arrays, index):
    """Check IDs against both independent world masks and visible geometry."""
    order = token_order(arrays['permutation'][index])
    tokens = arrays['student_obs'][:, index, :110].reshape(-1, 11, 10)
    np.testing.assert_array_equal(tokens[:, :, 8], arrays['visibility_world'][:, index][:, order])
    eef = np.concatenate([arrays['initial_eef'][None, index, :2], arrays['eef'][:-1, index, :2]])
    centers = np.concatenate([arrays['initial_state'][None, index, :, :2],
                              arrays['block_state'][:-1, index, :, :2]])[:, order]
    visible = tokens[:, :, 8] > .5
    decoded = tokens[:, :, :8].reshape(-1, 11, 4, 2).mean(axis=2) + eef[:, None, :]
    np.testing.assert_allclose(decoded[visible], centers[visible], atol=1e-5, rtol=0)
    return [dict(cell=cell, world_index=int(world), label=object_id(world))
            for cell, world in enumerate(order)]


def read(path):
    return json.loads(Path(path).read_text())


def text(image, value, xy, scale=.50, color=WHITE):
    cv2.putText(image, value, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def badge(canvas, label, anchor, color, occupied, bounds):
    """Small ID badges with leader lines when dense clutter requires an offset."""
    width = 22 if len(label) > 1 else 17
    height = 18
    x0, y0, x1, y1 = bounds
    candidates = [(0, 0), (0, -21), (0, 21), (-24, 0), (24, 0),
                  (-24, -21), (24, -21), (-24, 21), (24, 21), (0, -42), (0, 42)]
    for dx, dy in candidates:
        x = min(max(anchor[0] + dx - width//2, x0), x1-width)
        y = min(max(anchor[1] + dy - height//2, y0), y1-height)
        box = (x, y, x+width, y+height)
        if not any(x < b[2]+2 and x+width > b[0]-2 and y < b[3]+2 and y+height > b[1]-2 for b in occupied):
            break
    occupied.append(box)
    cv2.line(canvas, anchor, (x+width//2, y+height//2), color, 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (x, y), (x+width, y+height), BACKGROUND, -1)
    cv2.rectangle(canvas, (x, y), (x+width, y+height), color, 1)
    text(canvas, label, (x+3, y+13), .38, color)


def annotated(image, entry, scene, frame, arrays, index, plan, workspace, verdict, trail=None):
    canvas = np.full((HEIGHT, WIDTH, 3), BACKGROUND, np.uint8)
    canvas[72:648, :576] = cv2.resize(image, (576,576), interpolation=cv2.INTER_NEAREST)
    if trail is not None and len(trail) > 1:
        # EEF path so far (--eef-trace): every recorded frame, substeps included.
        cv2.polylines(canvas, [np.asarray(trail, np.int32)], False, (222,100,245), 2, cv2.LINE_AA)
    if 'eef' in frame:
        eef_xy = frame['eef'][str(index)]
        location = (int((eef_xy[1]+.32)/.64*576), 72+int((eef_xy[0]-.18)/.64*576))
        cv2.circle(canvas, location, 6, (222,100,245), 2, cv2.LINE_AA)
    tier = scene['tier'].replace('test-', '').capitalize()
    title = LABELS[entry['policy']]
    text(canvas, title, (18, 29), .76)
    text(canvas, CONDITIONS[entry['condition']], (290, 29), .53)
    text(canvas, f"{tier} {Path(scene['path']).stem} | Simulator top view | Pink: EEF", (18, 57), .45, MUTED)
    text(canvas, 'Actual policy detections', (594, 31), .56)
    text(canvas, 'Plan, memory and proprioception also available', (594, 57), .33, MUTED)
    terminal = int(verdict['terminal_step'])
    decision = min(max(frame['decision'], 0), max(len(arrays['student_obs']) - 1, 0))
    endpoint = frame['endpoint'] and frame['decision'] >= 0
    completed = max(0, frame['decision'] + int(endpoint))
    ended = terminal == 0 or completed >= terminal
    if len(arrays['student_obs']):
        tokens = arrays['student_obs'][decision, index, :110].reshape(11, 10)
        eef = arrays['initial_eef'][index, :2] if decision == 0 else arrays['eef'][decision - 1, index, :2]
        visibility = tokens[:, 8] > .5
    else:
        tokens = np.zeros((11, 10)); eef = arrays['initial_eef'][index, :2]; visibility = np.zeros(11, bool)
    order = token_order(arrays['permutation'][index])
    world_visibility = np.zeros(11, bool)
    world_visibility[order] = visibility
    # Raw videos include substeps, but trace positions are exact at decision
    # boundaries. Hold the latest recorded positions between those boundaries.
    state = arrays['initial_state'][index]
    if completed > 0:
        state = arrays['block_state'][min(completed-1, len(arrays['block_state'])-1), index]
    occupied = []
    for world, xy in enumerate(state[:, :2]):
        location = (int((xy[1]+.32)/.64*576), 72+int((xy[0]-.18)/.64*576))
        color = (GOLD if world == 0 else CYAN) if world_visibility[world] else MASKED
        badge(canvas, object_id(world), location, color, occupied, (4, 76, 570, 600))
    text(canvas, 'Object IDs: T = target, 1-10 = clutter', (18, 94), .43)
    text(canvas, 'ID positions update at decision boundaries', (18, 114), .36, MUTED)
    # Exact pre-action token geometry: zeroed/missing objects are not imputed.
    left, top, size = 594, 116, 284
    xbounds, ybounds = workspace
    x0, x1 = xbounds[0]-.035, xbounds[1]+.035
    y0, y1 = ybounds[0]-.035, ybounds[1]+.035

    def pixel(xy):
        x, y = xy
        return (int(left + (y-y0) / (y1-y0) * size), int(top + (x-x0) / (x1-x0) * size))

    cv2.rectangle(canvas, (left, top), (left+size, top+size), (43, 36, 27), -1)
    cv2.rectangle(canvas, pixel((xbounds[0], ybounds[0])), pixel((xbounds[1], ybounds[1])), MUTED, 1)
    points = np.asarray([pixel(xy[:2]) for xy in plan['xy']], np.int32)
    if len(points) > 1:
        cv2.polylines(canvas, [points], False, (111, 103, 90), 1, cv2.LINE_AA)
    detection_labels = []
    for token_index, token in enumerate(tokens):
        if not visibility[token_index]:
            continue
        corners = token[:8].reshape(4, 2) + eef
        poly = np.asarray([pixel(xy) for xy in corners], np.int32)
        # The target token is the teacher's grasp-clearance footprint.
        cv2.polylines(canvas, [poly], True, GOLD if token_index == 0 else CYAN, 2, cv2.LINE_AA)
        detection_labels.append((object_id(order[token_index]), pixel(corners.mean(axis=0)),
                                 GOLD if token_index == 0 else CYAN))
    occupied = []
    for label, anchor, color in detection_labels:
        badge(canvas, label, anchor, color, occupied, (left+2, top+2, left+size-2, top+size-2))
    cv2.circle(canvas, pixel(eef), 4, WHITE, -1, cv2.LINE_AA)
    text(canvas, 'Last decision input; missing objects hidden', (594, 100), .38, MUTED)
    text(canvas, f'Visible tokens: {int(visibility.sum())} / 11', (594, 430), .57)
    for i, visible in enumerate(visibility):
        x = 596 + i*25
        color = (GOLD if order[i] == 0 else CYAN) if visible else MASKED
        cv2.rectangle(canvas, (x, 447), (x+21, 468), color if visible else BACKGROUND, -1)
        cv2.rectangle(canvas, (x, 447), (x+21, 468), color, 1)
        label = object_id(order[i])
        text(canvas, label, (x+(3 if len(label)>1 else 6), 462), .38, BACKGROUND if visible else color)
        if not visible:
            text(canvas, 'x', (x+7, 483), .34, MASKED)
    text(canvas, 'Filled = detected    x = masked', (594, 503), .43)
    text(canvas, 'Same object IDs in scene, outlines and cells', (594, 524), .35, MUTED)
    text(canvas, 'Cells retain the actual shuffled token order', (594, 544), .35, MUTED)
    text(canvas, 'Gray: plan  |  White dot: end effector', (594, 565), .36, MUTED)
    text(canvas, 'Arm shadow + scheduled blackouts', (594, 594), .39, MUTED)
    text(canvas, 'remain active at every dropout setting.', (594, 615), .36, MUTED)
    q = float(arrays['initial_q'][index])
    if len(arrays['q']) and completed > 0:
        q = float(arrays['q'][min(completed-1, len(arrays['q'])-1), index])
    text(canvas, f'Decision {min(completed, terminal):03d}  |  Grasp score {q:.2f}  |  Slowed playback', (18, 675), .46)
    if ended:
        good = verdict['reason'] == 'success'
        color = (110, 221, 126) if good else (94, 131, 249)
        cv2.rectangle(canvas, (18, 605), (550, 637), BACKGROUND, -1)
        text(canvas, REASONS.get(verdict['reason'], verdict['reason']), (30, 628), .55, color)
    return canvas


def encode_entry(root, entry, refresh=False, trace=False):
    folder = root / 'runs' / entry['name']
    if not (folder / 'complete.json').exists():
        raise ValueError('Recording incomplete: ' + str(folder))
    raw = read(folder / 'video_raw/recording.json')
    result = read(folder / (entry['name'] + '.json'))
    provenance = read(folder / 'provenance.json')
    nominal = read(folder / 'nominal.json')
    clips = root / 'clips'; clips.mkdir(exist_ok=True)
    thumbs = root / 'thumbnails'; thumbs.mkdir(exist_ok=True)
    records = []
    renderer_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with np.load(folder / (entry['name'] + '.npz')) as archive:
        # NpzFile indexing decompresses on every access; decode each input once
        # instead of repeatedly decompressing a full rollout for every frame.
        arrays = {key:archive[key] for key in ['student_obs','initial_eef','eef','initial_q','q',
                  'initial_state','block_state','permutation','visibility_world']}
        for slot, scene in enumerate(raw['scenes']):
            index = raw.get('indices', list(range(len(raw['scenes']))))[slot]
            slug = scene['tier'].replace('test-', '') + '_' + Path(scene['path']).stem
            filename = slug + '_' + entry['name'] + '.mp4'
            output = clips / filename
            metrics = output.with_suffix('.json')
            mapping = verify_object_mapping(arrays, index)
            if metrics.exists() and not refresh:
                existing = read(metrics)
                if existing.get('renderer_sha256') != renderer_hash:
                    raise ValueError('Renderer changed; use --refresh: ' + str(output))
                if not output.exists():
                    raise ValueError('Missing encoded video: ' + str(output))
                records.append(existing); continue
            if output.exists() and not refresh:
                raise ValueError('Incomplete encode: ' + str(output))
            temporary_output = output.with_suffix('.rendering.mp4')
            capture = cv2.VideoCapture(str(folder / f'video_raw/scene{slot:02d}.mp4'))
            verdict = result['rows'][index]
            encoder = subprocess.Popen(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-y',
                '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{WIDTH}x{HEIGHT}', '-r', str(FPS), '-i', '-',
                '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '22', '-threads', '2',
                '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(temporary_output)], stdin=subprocess.PIPE)
            count = 0
            cover = None
            last = None
            trail = [] if trace else None
            try:
                for frame_index, frame in enumerate(raw['frames']):
                    ok, image = capture.read()
                    if not ok:
                        raise ValueError('Truncated raw recording')
                    if frame['decision'] >= int(verdict['terminal_step']):
                        break
                    if trail is not None and 'eef' in frame:
                        ex, ey = frame['eef'][str(index)]
                        trail.append((int((ey+.32)/.64*576), 72+int((ex-.18)/.64*576)))
                    last = annotated(image, entry, scene, frame, arrays, index,
                                     nominal['plans'][index], provenance['workspace'], verdict, trail)
                    repeat = FPS if frame_index == 0 else 1
                    for _ in range(repeat):
                        encoder.stdin.write(last.tobytes()); count += 1
                    if frame_index == min(35, max(0, int(verdict['terminal_step'])*2)):
                        cover = last.copy()
                if last is None:
                    # A scene terminal at initialization still gets a labeled frame.
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, image = capture.read()
                    if not ok:
                        raise ValueError('Missing initial frame')
                    last = annotated(image, entry, scene, raw['frames'][0], arrays, index,
                                     nominal['plans'][index], provenance['workspace'], verdict)
                for _ in range(FPS*2):
                    encoder.stdin.write(last.tobytes()); count += 1
            finally:
                capture.release()
                encoder.stdin.close()
            if encoder.wait() != 0:
                raise RuntimeError('Video encoder failed')
            temporary_output.replace(output)
            thumbnail = thumbs / filename.replace('.mp4', '.jpg')
            cv2.imwrite(str(thumbnail), cover if cover is not None else last)
            record = dict(entry=entry['name'], policy=entry['policy'], condition=entry['condition'],
                p_drop=entry['p_drop'], scene=scene, scene_slug=slug, reason=verdict['reason'],
                success=verdict['success'], decisions=verdict['terminal_step'], duration=count/FPS,
                path='clips/'+filename, thumbnail='thumbnails/'+thumbnail.name,
                checkpoint_sha256=result['checkpoint_sha256'], trace_sha256=result['trace_sha256'],
                renderer_sha256=renderer_hash, token_object_mapping=mapping,
                mapping_checked_against_world_visibility_and_geometry=True)
            metrics.write_text(json.dumps(record, indent=2)+'\n')
            records.append(record)
            print('ENCODED', filename, verdict['reason'], flush=True)
    return records


def gallery(root, make_montages=False, refresh=False):
    records = [read(p) for p in sorted((root/'clips').glob('*.json'))]
    spec = read(root/'gallery_spec.json')
    scenes = []
    for scene in spec['scenes']:
        slug = scene['tier'].replace('test-', '')+'_'+Path(scene['path']).stem
        scenes.append(dict(slug=slug, label=scene['tier'].replace('test-', '').capitalize()+' '+Path(scene['path']).stem))
    comparisons=[]
    if make_montages:
        directory=root/'comparisons';directory.mkdir(exist_ok=True)
        for scene in scenes:
            for condition in CONDITIONS:
                chosen=[next((r for r in records if r['scene_slug']==scene['slug'] and r['condition']==condition and r['policy']==policy),None) for policy in LABELS]
                if any(r is None for r in chosen):
                    continue
                name=scene['slug']+'_'+condition+'.mp4';output=directory/name
                if refresh or not output.exists():
                    temporary_output = output.with_suffix('.rendering.mp4')
                    command=['ffmpeg','-hide_banner','-loglevel','error','-nostdin','-y','-filter_complex_threads','1']
                    for record in chosen:
                        command+=['-threads','1','-i',str(root/record['path'])]
                    filters=';'.join(f'[{i}:v]tpad=stop_mode=clone:stop_duration=60[v{i}]' for i in range(4))
                    filters+=';[v0][v1][v2][v3]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0,scale=1280:-2[out]'
                    command+=['-filter_complex',filters,'-map','[out]','-t',str(max(r['duration'] for r in chosen)),
                        '-an','-c:v','libx264','-preset','fast','-crf','23','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(temporary_output)]
                    subprocess.run(command,check=True)
                    temporary_output.replace(output)
                comparisons.append(dict(scene=scene['slug'],condition=condition,path='comparisons/'+name))
    payload=dict(scenes=scenes,records=records,comparisons=comparisons,labels=LABELS,conditions=CONDITIONS)
    (root/'gallery.json').write_text(json.dumps(payload,indent=2)+'\n')
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Epoch 210: student evaluation videos</title><style>
body{margin:0;background:#101720;color:#e8edf3;font:16px/1.55 system-ui,sans-serif}main{max-width:1450px;margin:auto;padding:30px}
h1{font-size:30px;line-height:1.2}p{max-width:1000px;color:#b5c3d0}a{color:#8cd7ff}label{margin-right:16px}select,button{font:inherit;padding:9px 13px;margin:5px 8px 5px 0;border-radius:7px;border:1px solid #526273;background:#203142;color:white;cursor:pointer}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}article{background:#192330;padding:12px;border-radius:12px}h2{margin:0;font-size:18px}video{width:100%;display:block;margin:10px 0;background:black}small{color:#b5c3d0}.controls{margin:22px 0}.links{margin:20px 0}table{border-collapse:collapse;width:100%;max-width:950px}td,th{padding:8px;border-bottom:1px solid #344252;text-align:left}@media(max-width:780px){.grid{grid-template-columns:1fr}main{padding:18px}}</style>
<main><h1>BC and DAgger: matched evaluation videos</h1>
<p>Saved student checkpoints trained from the simplified-reward teacher at epoch 210. Three fixed development scenes, one per difficulty tier; failures are included. These fresh qualitative runs are not a success-rate benchmark.</p>
<p>Left in each video: simulator top view, with the end effector marked in pink. Right: the exact object tokens supplied to the policy at the last decision, with missing objects hidden. Color pixels are for viewing only. Detection dropout removes tokens, not image pixels; geometric arm shadow and scheduled five-decision blackouts also remain active, including at 0% extra dropout.</p>
<p><strong>Object IDs:</strong> T identifies the target; 1–10 identify clutter objects. The same IDs label the scene, detected outlines and visibility cells. Filled cells mean detected; outlined cells with an x mean masked. Cells retain the policy's shuffled token order. Scene ID positions update at recorded decision boundaries; mask status refers to the last policy input.</p>
<div class="controls"><label>Scene <select id="scene"></select></label><label>Extra dropout <select id="condition"></select></label><button id="play">Play all from start</button><button id="pause">Pause all</button><button id="reset">Reset</button></div>
<p><small>All panels use the same initial scene and perturbation draw. Playback is slowed and synchronized by decisions; each clip stops at its first terminal event. Success means a graspable target with no workspace violation, not a physical grasp and lift.</small></p>
<div class="grid" id="grid"></div><div class="links" id="download"></div>
<details><summary>All comparison videos and example outcomes</summary><div id="all"></div></details>
<p><a href="gallery.json">Video manifest and outcomes</a> · <a href="README.md">Protocol and validation notes</a></p></main>
<script>const D=__DATA__;const scene=document.getElementById('scene'),condition=document.getElementById('condition');
D.scenes.forEach(s=>scene.add(new Option(s.label,s.slug)));Object.entries(D.conditions).forEach(([k,v])=>condition.add(new Option(v,k)));
function show(){const grid=document.getElementById('grid');grid.replaceChildren();Object.entries(D.labels).forEach(([policy,label])=>{const r=D.records.find(r=>r.scene_slug===scene.value&&r.condition===condition.value&&r.policy===policy);const a=document.createElement('article');const h=document.createElement('h2');h.textContent=label;a.append(h);if(r){const v=document.createElement('video');v.controls=true;v.muted=true;v.playsInline=true;v.preload='metadata';v.src=r.path;v.poster=r.thumbnail;a.append(v);const s=document.createElement('small');s.textContent=(r.success?'Success':r.reason==='oow'?'Outside workspace':r.reason)+' · '+r.decisions+' decisions';a.append(s);const link=document.createElement('a');link.href=r.path;link.textContent=' Open video';a.append(link)}grid.append(a)});const c=D.comparisons.find(c=>c.scene===scene.value&&c.condition===condition.value);document.getElementById('download').innerHTML=c?'<a href="'+c.path+'">Open the four-policy comparison as one MP4</a>':''}
scene.onchange=condition.onchange=show;document.getElementById('play').onclick=()=>{document.querySelectorAll('video').forEach(v=>{v.currentTime=0;v.play().catch(()=>{})})};document.getElementById('pause').onclick=()=>document.querySelectorAll('video').forEach(v=>v.pause());document.getElementById('reset').onclick=()=>document.querySelectorAll('video').forEach(v=>{v.pause();v.currentTime=0});
const all=document.getElementById('all');D.comparisons.forEach(c=>{const p=document.createElement('p'),a=document.createElement('a');a.href=c.path;a.textContent=c.scene+' — '+D.conditions[c.condition];p.append(a);all.append(p)});show();</script></html>'''
    page=page.replace('__DATA__',json.dumps(payload).replace('</','<\\/'))
    (root/'index.html').write_text(page)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path)
    parser.add_argument('--entry');parser.add_argument('--montages',action='store_true')
    parser.add_argument('--refresh',action='store_true',help='Regenerate annotations and comparisons from existing recordings')
    parser.add_argument('--eef-trace',action='store_true',help='Draw the EEF path so far on the simulator view')
    args=parser.parse_args();root=args.root.resolve()
    spec=read(root/'gallery_spec.json')
    for entry in spec['entries']:
        if args.entry is None or entry['name']==args.entry:
            encode_entry(root,entry,args.refresh,args.eef_trace)
    gallery(root,args.montages,args.refresh)
