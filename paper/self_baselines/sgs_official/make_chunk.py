# -*- coding: utf-8 -*-
"""Assemble an SGS-SLAM chunk dataset (frames/depths/valid/semantics/traj)."""
import argparse
import os
from pathlib import Path

SELF = Path(os.environ.get("SELF_ROOT", "<PROJECT_ROOT>"))
FRAMES = SELF / "selfcollected/yolo_input/frames"
SEM_ID = SELF / "selfcollected/yolo_full/semantic_ids"
SEM_COL = SELF / "selfcollected/yolo_full/semantic_colors"
BASE = Path(os.environ.get("SGS_DATA", str(SELF / "SGS-SLAM/data")))

ap = argparse.ArgumentParser()
ap.add_argument("--start", type=int, required=True)
ap.add_argument("--end", type=int, required=True)
ap.add_argument("--depth", required=True, help="depth_pipeline output dir")
args = ap.parse_args()
s, e = args.start, args.end
name = "chunk_%d_%d" % (s, e)
out = BASE / name
for sub in ("frames", "depths", "valid", "semantic_ids", "semantic_colors"):
    (out / sub).mkdir(parents=True, exist_ok=True)

n = 0
for i in range(s, e + 1):
    pairs = [
        (FRAMES / ("frame%06d.jpg" % i), out / "frames" / ("frame%06d.png" % i)),
        (Path(args.depth) / "depths" / ("depth%06d.npy" % i), out / "depths" / ("depth%06d.npy" % i)),
        (Path(args.depth) / "valid" / ("valid%06d.png" % i), out / "valid" / ("valid%06d.png" % i)),
        (SEM_ID / ("semantic_id%06d.png" % i), out / "semantic_ids" / ("semantic_id%06d.png" % i)),
        (SEM_COL / ("semantic_color%06d.png" % i), out / "semantic_colors" / ("semantic_color%06d.png" % i)),
    ]
    for src, dst in pairs:
        if not src.exists():
            raise SystemExit("missing %s" % src)
        if not dst.exists():
            os.symlink(src, dst)
    n += 1

traj = ["1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"] * n
(out / "traj.txt").write_text("\n".join(traj) + "\n")
print("built", out, "frames", n)
