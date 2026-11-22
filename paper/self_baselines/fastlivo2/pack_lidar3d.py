# -*- coding: utf-8 -*-
"""Pack lidar_all npz scans into one binary, parallel reads.

Binary layout: uint32 n_scans, then per scan: int32 n_points, float64 stamp,
n_points * 3 float32 (x, y, z).
"""
import csv
import os
import struct
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = os.environ.get("SELF_ROOT", "<PROJECT_ROOT>")
LID = os.environ.get("SELF_LIDAR3D", ROOT + "/selfcollected/lidar_all")
FMAP = os.environ.get("SELF_LMAP", LID + "/frame_lidar_map.csv")
OUT = os.environ.get("SELF_LIDAR3D_BIN", ROOT + "/self_livo/lidar3d.bin")

seen, rows = set(), []
with open(FMAP, newline="") as f:
    for r in csv.DictReader(f):
        ts = round(float(r["lidar_t"]), 6)
        if ts in seen:
            continue
        seen.add(ts)
        rows.append((float(r["lidar_t"]), os.path.join(LID, r["file"])))
rows.sort()
print("unique scans", len(rows), flush=True)


def one(item):
    ts, path = item
    try:
        return np.load(path)["xyz"].astype("<f4")
    except Exception:
        return np.zeros((0, 3), "<f4")


with open(OUT, "wb") as f, ThreadPoolExecutor(max_workers=16) as ex:
    f.write(struct.pack("<I", len(rows)))
    for k, (item, xyz) in enumerate(zip(rows, ex.map(one, rows))):
        f.write(struct.pack("<Id", len(xyz), item[0]))
        xyz.tofile(f)
        if (k + 1) % 200 == 0:
            print(k + 1, flush=True)
print("DONE", os.path.getsize(OUT))
