# -*- coding: utf-8 -*-
"""Chain GigaSLAM segments using nearest keyframe pairs within +/-2 frames."""
import glob
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
BASE = os.environ.get("SELF_ROOT", "<PROJECT_ROOT>")
ROOT = os.environ.get("GIGASLAM_ROOT", BASE + "/gigaslam_osdar")
SEGS = [(0, 349), (300, 649), (600, 949), (900, 1249), (1200, 1549),
        (1500, 1849), (1800, 2149), (2100, 2449), (2400, 2749), (2700, 3043)]
OUT = os.environ.get("GIGASLAM_MERGED", ROOT + "/gigaslam_chain_tum.txt")
TOL = 2


def load_seg(s, e):
    dirs = sorted(glob.glob("%s/results/*self_%d_%d*/*/" % (ROOT, s, e)))
    dirs = [d for d in dirs if os.path.isfile(os.path.join(d, "poses_est.txt"))]
    if not dirs:
        return None
    d = dirs[-1]
    idx = np.loadtxt(os.path.join(d, "poses_idx.txt"), dtype=int).reshape(-1)
    M = np.loadtxt(os.path.join(d, "poses_est.txt")).reshape(-1, 4, 4)
    n = min(len(idx), len(M))
    o = np.argsort(idx[:n])
    return s + idx[:n][o], M[:n][o]


def sim3(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    U, sv, Vt = np.linalg.svd(S.T @ D)
    R = Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
    s = sv.sum() / (S ** 2).sum()
    return s, R, mu_d - s * R @ mu_s


merged_ids = []
merged_M = None
for s, e in SEGS:
    got = load_seg(s, e)
    if got is None:
        print("missing", s, e)
        continue
    ids, M = got
    if not merged_ids:
        merged_ids, merged_M = list(ids), M.copy()
        print("base", s, e, len(ids))
        continue
    arr = np.array(sorted(merged_ids))
    pairs, used = [], set()
    for k, gid in enumerate(ids):
        j = int(np.argmin(np.abs(arr - gid)))
        if abs(arr[j] - gid) <= TOL and arr[j] not in used:
            pairs.append((k, arr[j]))
            used.add(arr[j])
    if len(pairs) < 6:
        print("skip %d-%d pairs %d" % (s, e, len(pairs)))
        continue
    idx_new = np.array([p[0] for p in pairs])
    idx_old = np.array([merged_ids.index(int(p[1])) for p in pairs])
    P_new = M[idx_new, :3, 3]
    P_old = np.array(merged_M)[idx_old, :3, 3]
    sc, R, t = sim3(P_new, P_old)
    res = np.linalg.norm((sc * (R @ P_new.T)).T + t - P_old, axis=1)
    M2 = M.copy()
    M2[:, :3, 3] = (sc * (R @ M[:, :3, 3].T)).T + t
    for i in range(len(M2)):
        M2[i, :3, :3] = R @ M[i, :3, :3]
    print("chain %d-%d pairs %d scale %.3f rms %.3f" % (s, e, len(pairs), sc, np.sqrt((res ** 2).mean())))
    keep = [k for k in range(len(ids)) if ids[k] not in used]
    merged_M = np.concatenate([np.array(merged_M), M2[keep]], 0)
    merged_ids = merged_ids + [int(ids[k]) for k in keep]

merged_M = np.array(merged_M)
ids = np.array(merged_ids)
o = np.argsort(ids)
ids, M = ids[o], merged_M[o]
t = ids / 15.0
q = Rot.from_matrix(M[:, :3, :3]).as_quat()
out = np.column_stack([t, M[:, :3, 3], q])
np.savetxt(OUT, out, fmt="%.6f %.6f %.6f %.6f %.9f %.9f %.9f %.9f")
print("wrote", OUT, out.shape, "ids", ids[0], ids[-1])
