# -*- coding: utf-8 -*-
"""Build a full-length MonoGS trajectory from segmented runs.

Each 350-frame segment is an independent monocular run with its own arbitrary
scale and origin. Overlapping keyframes are too sparse to chain all ten
segments (1-4 per link), so each segment is Sim(3)-aligned to the FAST-LIVO2
reference on its own keyframes and the aligned segments are concatenated.
Reported ATE is therefore a segment-wise Sim(3)-aligned ATE.
"""
import glob
import json
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
BASE = os.environ.get("SELF_ROOT", "<PROJECT_ROOT>")
ROOT = os.environ.get("MONOGS_ROOT", BASE + "/monogs_run")
SEGS = [(0, 349), (300, 649), (600, 949), (900, 1249), (1200, 1549),
        (1500, 1849), (1800, 2149), (2100, 2449), (2400, 2749), (2700, 3043)]
OUT = os.environ.get("MONOGS_MERGED", ROOT + "/merged_monogs_tum.txt")
T0 = 1677730093.022842


def umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    U, sv, Vt = np.linalg.svd(S.T @ D)
    R = Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
    s = sv.sum() / (S ** 2).sum()
    return s, R, mu_d - s * R @ mu_s


ref = np.loadtxt(os.environ.get("SELF_REF", BASE + "/self_livo/livo_traj_tum.txt"))
sw = np.loadtxt(os.environ.get("SELF_SCANS_WALL", BASE + "/self_livo/scans_wall.txt"))
k = len(sw) - len(ref)
t_ref = sw[k:k + len(ref), 0]
P_ref = ref[:, 1:4]

all_t, all_M = [], []
for s, e in SEGS:
    pat = "%s/results/dataset_self_%d_%d/*/plot/trj_final.json" % (ROOT, s, e)
    files = sorted(glob.glob(pat))
    if not files:
        print("missing", s, e)
        continue
    d = json.load(open(files[-1]))
    ids = np.array(d["trj_id"], dtype=int) + s
    M = np.array(d["trj_est"], dtype=float)
    t = T0 + ids / 15.0
    m = (t >= t_ref[0]) & (t <= t_ref[-1])
    if m.sum() < 3:
        print("skip", s, e, "outside reference")
        continue
    ids, M, t = ids[m], M[m], t[m]
    refp = np.stack([np.interp(t, t_ref, P_ref[:, i]) for i in range(3)], 1)
    sc, R, tr = umeyama(M[:, :3, 3], refp)
    M2 = M.copy()
    M2[:, :3, 3] = (sc * (R @ M[:, :3, 3].T)).T + tr
    for i in range(len(M2)):
        M2[i, :3, :3] = R @ M[i, :3, :3]
    res = np.linalg.norm(M2[:, :3, 3] - refp, axis=1)
    print("seg %4d-%4d kf %3d scale %.3f rmse %.3f max %.3f" %
          (s, e, len(ids), sc, np.sqrt((res ** 2).mean()), res.max()))
    all_t.append(t)
    all_M.append(M2)

t = np.concatenate(all_t)
M = np.concatenate(all_M, 0)
o = np.argsort(t)
t, M = t[o], M[o]
q = Rot.from_matrix(M[:, :3, :3]).as_quat()
out = np.column_stack([t - T0, M[:, :3, 3], q])
np.savetxt(OUT, out, fmt="%.6f %.6f %.6f %.6f %.9f %.9f %.9f %.9f")
print("wrote", OUT, out.shape)
