# -*- coding: utf-8 -*-
"""ATE of the merged MonoGS trajectory against the FAST-LIVO2 reference."""
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
BASE = os.environ.get("SELF_ROOT", "<PROJECT_ROOT>")
ROOT = os.environ.get("MONOGS_ROOT", BASE + "/monogs_run")
T0 = 1677730093.022842


def umeyama(src, dst, scale):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    U, sv, Vt = np.linalg.svd(S.T @ D)
    R = Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
    s = sv.sum() / (S ** 2).sum() if scale else 1.0
    return s, R, mu_d - s * R @ mu_s


ref = np.loadtxt(os.environ.get("SELF_REF", BASE + "/self_livo/livo_traj_tum.txt"))
sw = np.loadtxt(os.environ.get("SELF_SCANS_WALL", BASE + "/self_livo/scans_wall.txt"))
k = len(sw) - len(ref)
t_ref = sw[k:k + len(ref), 0]
P_ref = ref[:, 1:4]

m = np.loadtxt(os.environ.get("MONOGS_MERGED", ROOT + "/merged_monogs_tum.txt"))
tq = T0 + m[:, 0]
P = m[:, 1:4]
keep = (tq >= t_ref[0]) & (tq <= t_ref[-1])
tq, P = tq[keep], P[keep]
ref_p = np.stack([np.interp(tq, t_ref, P_ref[:, i]) for i in range(3)], 1)
mile = np.linalg.norm(np.diff(ref_p, axis=0), axis=1).sum()

for sc in (False, True):
    s, R, t = umeyama(P, ref_p, sc)
    e = np.linalg.norm((s * (R @ P.T)).T + t - ref_p, axis=1)
    print("%s s=%.3f n=%d rmse=%.3f mean=%.3f max=%.3f final=%.3f ref_mile=%.1f ratio=%.4f" %
          ("sim3" if sc else "se3", s, len(tq), np.sqrt((e ** 2).mean()), e.mean(), e.max(), e[-1],
           mile, np.sqrt((e ** 2).mean()) / mile))
