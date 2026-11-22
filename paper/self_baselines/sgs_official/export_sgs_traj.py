# -*- coding: utf-8 -*-
"""Export SGS-SLAM estimated camera centers to TUM for one chunk."""
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = os.environ.get("SGS_EXP_ROOT", "<PROJECT_ROOT>/sgs_official_run/experiments/Railway_self_official")
run, start, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
out = sys.argv[4]
d = np.load(os.path.join(ROOT, run, "params.npz"))
q = np.asarray(d["cam_unnorm_rots"])[0].T  # (N,4) wxyz
t = np.asarray(d["cam_trans"])[0].T        # (N,3)
N = min(len(q), len(t), n)


def q2R(qq):
    qq = qq / max(np.linalg.norm(qq), 1e-12)
    w, x, y, z = qq
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


rows = []
for i in range(N):
    R = q2R(q[i])
    C = -R.T @ t[i]
    rows.append((1677730093.022842 + (start + i) / 15.0, C[0], C[1], C[2], 0.0, 0.0, 0.0, 1.0))
np.savetxt(out, np.array(rows), fmt="%.6f %.6f %.6f %.6f %.9f %.9f %.9f %.9f")
print("wrote", out, N)
