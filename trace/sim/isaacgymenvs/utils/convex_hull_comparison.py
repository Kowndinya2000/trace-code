import torch

def flatten_points(points):
    # points: (11, x, 4, 2) → (x, 44, 2)
    return points.permute(1, 0, 2, 3).reshape(points.shape[1], -1, 2)

def cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
def lexsort_2d(points):
    """
    points: (N,2)
    returns: lexicographically sorted points by (x, then y)
    """
    idx = torch.argsort(points[:, 1], stable=True)
    points = points[idx]

    idx = torch.argsort(points[:, 0], stable=True)
    points = points[idx]

    return points

def convex_hull_single(points):
    """
    points: (N,2)
    returns: (M,2) CCW hull
    """
    if points.shape[0] <= 2:
        return points

    pts = lexsort_2d(points)

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return torch.stack(lower[:-1] + upper[:-1])
def batched_convex_hulls(points):
    """
    points: (x, N, 2)
    returns: list of x hull tensors, variable length
    """
    hulls = []
    for i in range(points.shape[0]):
        hulls.append(convex_hull_single(points[i]))
    return hulls

def point_in_hull_single(hull, q):
    """
    hull: (M,2) CCW
    q: (2,)
    """
    M = hull.shape[0]
    prev = None

    for i in range(M):
        a = hull[i]
        b = hull[(i + 1) % M]
        cp = cross(a, b, q)
        s = torch.sign(cp)

        if s == 0:
            continue
        if prev is None:
            prev = s
        elif s != prev:
            return False

    return True

def batched_point_in_convex_hull(points, queries):
    """
    points: (11, x, 4, 2)
    queries: (x, 2)
    returns: (x, 1) bool
    """
    pts = flatten_points(points)          # (x, 44, 2)
    hulls = batched_convex_hulls(pts)

    result = torch.zeros((pts.shape[0], 1), dtype=torch.bool, device=points.device)

    for i in range(pts.shape[0]):
        result[i, 0] = point_in_hull_single(hulls[i], queries[i])

    return result

def point_in_convex_hull_gpu(points, queries, eps=1e-6):
    """
    Fully GPU-vectorized convex hull containment test
    (no explicit hull construction)

    points:  (11, x, 4, 2)
    queries: (x, 2)
    returns: (x, 1) bool
    """

    # (x, 44, 2)
    P = points.permute(1, 0, 2, 3).reshape(points.shape[1], -1, 2)
    Q = queries[:, None, :]  # (x, 1, 2)

    # All edge directions (pairwise differences)
    # (x, 44, 44, 2)
    edges = P[:, :, None, :] - P[:, None, :, :]

    # Perpendicular normals
    normals = torch.stack(
        [-edges[..., 1], edges[..., 0]], dim=-1
    )  # (x, 44, 44, 2)

    # Normalize to avoid scale issues
    norm = torch.linalg.norm(normals, dim=-1, keepdim=True)
    normals = normals / (norm + eps)

    # Project points and query
    # (x, 44, 44)
    proj_P = torch.einsum("xijk,xmk->xijm", normals, P)

    # (x, 44, 44)
    proj_Q = torch.einsum("xijk,xlk->xijl", normals, Q).squeeze(-1)

    # Max over points for each normal
    max_proj_P = proj_P.max(dim=-1).values

    # Containment condition
    inside = (proj_Q <= max_proj_P + eps).all(dim=(1, 2))

    return inside.unsqueeze(1)


num_envs = 32
points = torch.rand(11, num_envs, 4, 2, device="cuda")
queries = torch.rand(num_envs, 2, device="cuda")

import time 
start_time = time.time()
inside = point_in_convex_hull_gpu(points, queries)
print(inside.shape, "time (ms):", int((time.time()-start_time)*1000))  # (32, 1)

start_time = time.time()
inside_m2 = batched_point_in_convex_hull(points, queries)
print(inside_m2.shape, "time (ms):", int((time.time()-start_time)*1000))  

print("Difference:", torch.sum(inside != inside_m2))