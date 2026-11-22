# -*- coding: utf-8 -*-
"""Build a MonoGS TUM-style dataset for a frame segment of the self sequence."""
import argparse
import os
from pathlib import Path

SELF = Path(os.environ.get("SELF_ROOT", "<PROJECT_ROOT>"))
FRAMES = Path(os.environ.get("SELF_FRAMES", str(SELF / "selfcollected/yolo_input/frames")))
ROOT = Path(os.environ.get("MONOGS_ROOT", str(SELF / "monogs_run")))

ap = argparse.ArgumentParser()
ap.add_argument("--start", type=int, required=True)
ap.add_argument("--end", type=int, required=True)
args = ap.parse_args()
start, end = args.start, args.end
name = "self_%d_%d" % (start, end)
out = ROOT / "dataset" / name
(out / "rgb").mkdir(parents=True, exist_ok=True)
(out / "depth").mkdir(parents=True, exist_ok=True)

rgb_lines, depth_lines, pose_lines = [], [], ["# timestamp tx ty tz qx qy qz qw"]
for i in range(start, end + 1):
    ts = i / 15.0
    src = FRAMES / ("frame%06d.jpg" % i)
    dst = out / "rgb" / ("frame%06d.jpg" % i)
    if not dst.exists():
        os.symlink(src, dst)
    rgb_lines.append("%.6f rgb/frame%06d.jpg" % (ts, i))
    depth_lines.append("%.6f depth/frame%06d.png" % (ts, i))
    pose_lines.append("%.6f 0 0 0 0 0 0 1" % ts)

(out / "rgb.txt").write_text("\n".join(rgb_lines) + "\n")
(out / "depth.txt").write_text("\n".join(depth_lines) + "\n")
(out / "pose.txt").write_text("\n".join(pose_lines) + "\n")
print("built", out, "frames", end - start + 1)
