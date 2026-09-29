"""Conservative top-down arm occlusion; observation masking, NOT collision checking.

Use the union of padded minimum-area rectangles enclosing projected URDF
collision-body bounds. A whole-arm hull is optional. This is the near-
orthographic simulator camera model, not a calibrated perspective projection
for the real D455. All arm and object poses must use the SAME coordinate frame.
"""
from functools import lru_cache
from itertools import product
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np

VISIBILITY_PROTOCOL = "arm-link-footprints-v1"
DEFAULT_MARGIN_M = .010
DEFAULT_P_DROP = .05
DEFAULT_BLACKOUT_LEN = 5
MODES = ("link_union", "convex_hull")


def visibility_settings(options=None, horizon=120):
    options = options or {}
    result = dict(protocol=VISIBILITY_PROTOCOL,
                  mode=options.get("occlusion_mode", "link_union"),
                  margin_m=float(options.get("occlusion_margin_m", DEFAULT_MARGIN_M)),
                  p_drop=float(options.get("p_drop", DEFAULT_P_DROP)),
                  blackout_len=int(options.get("blackout_len", DEFAULT_BLACKOUT_LEN)))
    if result["mode"] not in MODES:
        raise ValueError("Unknown arm occlusion mode")
    if not np.isfinite(result["margin_m"]) or result["margin_m"] < 0:
        raise ValueError("Occlusion margin must be finite and nonnegative")
    if not 0 <= result["p_drop"] <= 1 or not 0 <= result["blackout_len"] <= horizon:
        raise ValueError("Invalid stress-test dropout/blackout")
    return result


def transform_points(points, poses):
    """Broadcast (..., V, 3) points by (..., 7) poses, quaternion XYZW."""
    points, poses = np.asarray(points, np.float64), np.asarray(poses, np.float64)
    q = poses[..., None, 3:7]
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if not np.isfinite(poses).all() or np.any(norm < 1e-8):
        raise ValueError("Invalid body pose for occlusion")
    q = q / norm
    cross = 2 * np.cross(q[..., :3], points)
    return points + q[..., 3:] * cross + np.cross(q[..., :3], cross) + poses[..., None, :3]


def projected_regions(body_points, margin_m=DEFAULT_MARGIN_M, mode="link_union"):
    """(L,V,2 or 3) body bounds -> padded rectangles (or their convex hull)."""
    if mode not in MODES or not np.isfinite(margin_m) or margin_m < 0:
        raise ValueError("Invalid arm projection settings")
    points = np.asarray(body_points, np.float32)
    if points.ndim != 3 or not len(points) or points.shape[1] < 1 or points.shape[2] not in (2, 3):
        raise ValueError("Current arm body points are required; no base-to-EEF fallback")
    if not np.isfinite(points).all():
        raise ValueError("Non-finite arm geometry")
    regions = []
    for body in points:
        center, size, angle = cv2.minAreaRect(np.ascontiguousarray(body[:, :2]))
        # Nonzero extent also handles vertical links with point-like projection.
        size = tuple(max(float(s) + 2 * margin_m, 1e-7) for s in size)
        regions.append(cv2.boxPoints((center, size, angle)))
    if mode == "convex_hull":
        return [cv2.convexHull(np.concatenate(regions)).reshape(-1, 2)]
    return regions


def polygons_intersect(a, b, tolerance=1e-7):
    """Separating-axis test on convex polygons; touching/containment both count."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if np.any(a.max(0) < b.min(0) - tolerance) or np.any(b.max(0) < a.min(0) - tolerance):
        return False
    edges = np.concatenate([np.roll(a, -1, axis=0) - a, np.roll(b, -1, axis=0) - b])
    axes = np.stack([-edges[:, 1], edges[:, 0]], axis=-1)
    lengths = np.linalg.norm(axes, axis=1)
    axes = axes[lengths > 1e-12] / lengths[lengths > 1e-12, None]
    pa, pb = a @ axes.T, b @ axes.T
    return not np.any((pa.max(0) < pb.min(0) - tolerance) | (pb.max(0) < pa.min(0) - tolerance))


def footprint_visibility(object_points, regions):
    """Hide the entire token on ANY overlap with its convex mesh footprint.

    Convexifying concave object meshes is deliberately conservative. An object
    containing an arm rectangle is hidden too, even if no object corner is in it.
    """
    visible = np.ones(len(object_points), np.float32)
    for i, points in enumerate(object_points):
        points = np.asarray(points, np.float32)
        if not np.isfinite(points).all():
            visible[i] = 0.
            continue
        hull = cv2.convexHull(np.ascontiguousarray(points[:, :2])).reshape(-1, 2)
        if len(hull) < 3 or any(polygons_intersect(hull, region) for region in regions):
            visible[i] = 0.
    return visible


def _box(lo, hi):
    return np.array(list(product(*zip(lo, hi))), np.float64)


@lru_cache(maxsize=8)
def collision_bounds(urdf_path):
    """Load each physical body's local enclosing box once, retaining asset paths.

    Collision meshes provide stable metric geometry without visual COLLADA
    up-axis conversions. The configurable workspace padding covers modest
    visual/collision and calibration discrepancies; it is not a safety margin.
    """
    import trimesh
    from scipy.spatial.transform import Rotation
    path = Path(urdf_path).resolve()
    names, bounds, assets = [], [], {path}
    for link in ET.parse(path).getroot().findall("link"):
        points = []
        for collision in link.findall("collision"):
            geometry = collision.find("geometry")
            mesh, box = geometry.find("mesh"), geometry.find("box")
            cylinder, sphere = geometry.find("cylinder"), geometry.find("sphere")
            if mesh is not None:
                filename = mesh.attrib["filename"]
                mesh_path = (path.parent.parent / filename[len("package://"):]
                             if filename.startswith("package://") else path.parent / filename).resolve()
                shape = trimesh.load(str(mesh_path), force="mesh", process=False)
                vertices = np.asarray(shape.vertices) * np.fromstring(mesh.get("scale", "1 1 1"), sep=" ")
                vertices = _box(vertices.min(0), vertices.max(0))
                assets.add(mesh_path)
            elif box is not None:
                half = np.fromstring(box.attrib["size"], sep=" ") / 2
                vertices = _box(-half, half)
            elif cylinder is not None or sphere is not None:
                node = cylinder if cylinder is not None else sphere
                r = float(node.attrib["radius"])
                half = np.array([r, r, float(node.attrib["length"]) / 2 if cylinder is not None else r])
                vertices = _box(-half, half)
            else:
                raise ValueError("Unsupported collision geometry: " + link.attrib["name"])
            origin = collision.find("origin")
            if origin is not None:
                vertices = Rotation.from_euler("xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")).apply(vertices)
                vertices += np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
            points.append(vertices)
        if points:
            points = np.concatenate(points)
            if not np.isfinite(points).all():
                raise ValueError("Non-finite collision geometry")
            names.append(link.attrib["name"])
            bounds.append(_box(points.min(0), points.max(0)))
    if not bounds:
        raise ValueError("Robot has no collision bodies")
    return tuple(names), np.stack(bounds), tuple(sorted(assets))


class SimulatorArmOcclusion:
    """Read-only adapter: current rigid-body poses, no IK or motion changes."""
    def __init__(self, env):
        from isaacgym import gymapi
        import torch
        self.env = env
        asset = Path(__file__).resolve().parents[2] / "assets/urdf/more"
        if env.robot_urdf_option == 0:
            robot = asset / "ur5e_simplified/ur5e_simplified_gripper_no_eoh.urdf"
        elif env.robot_urdf_option == 1:
            robot = asset / "ur5e/ur5e_gripper.urdf"
        else:
            raise ValueError("Unsupported robot asset for occlusion")
        self.names, self.bounds, self.assets = collision_bounds(str(robot))
        ids = [[env.gym.find_actor_rigid_body_index(e, a, name, gymapi.DOMAIN_SIM)
                for name in self.names] for e, a in zip(env.envs, env.ur5es)]
        if np.any(np.asarray(ids) < 0):
            raise ValueError("URDF body missing from simulator; cannot compute arm mask")
        self.ids = torch.tensor(ids, dtype=torch.long, device=env.device)
        # Actual object meshes, NOT teacher target/gripper-clearance rectangles.
        meshes, paths = [], []
        for name in ("concave", "cylinder", "cube", "half-cube", "rect", "triangle"):
            path = asset / "blocks-more" / (name + ".obj")
            vertices = [[float(x) for x in line.split()[1:4]]
                        for line in path.read_text().splitlines() if line.startswith("v ")]
            meshes.append(np.asarray(vertices)); paths.append(path)
        nv = max(map(len, meshes))
        meshes = np.stack([np.pad(p, ((0, nv-len(p)), (0, 0)), mode="edge") for p in meshes])
        self.object_points = meshes[env.all_block_name_ids.long().cpu().numpy()]
        self.assets = tuple(sorted(set(self.assets) | set(paths)))

    def visibility(self, settings):
        env = self.env
        # Both tensors use the simulator's same environment coordinate convention.
        poses = env.rb_states[self.ids, :7].detach().cpu().numpy().copy()
        object_poses = env.block_state[..., :7].detach().cpu().numpy().copy()
        arm_valid = np.isfinite(poses).all(-1) & (np.linalg.norm(poses[..., 3:7], axis=-1) > 1e-8)
        object_valid = np.isfinite(object_poses).all(-1) & (np.linalg.norm(object_poses[..., 3:7], axis=-1) > 1e-8)
        # Invalid terminal states still need a recorded failure, not an exception
        # that discards the batch. Sanitize only for arithmetic, then mask unknowns.
        poses[~arm_valid] = [0,0,0,0,0,0,1]
        object_poses[~object_valid] = [0,0,0,0,0,0,1]
        bodies = transform_points(self.bounds, poses)
        objects = transform_points(self.object_points, object_poses)
        visible = np.stack([footprint_visibility(obj, projected_regions(arm, settings["margin_m"], settings["mode"]))
                            for obj, arm in zip(objects, bodies)])
        visible[~arm_valid.all(-1)] = 0.
        visible[~object_valid] = 0.
        return visible
