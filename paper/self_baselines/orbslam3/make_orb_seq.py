# -*- coding: utf-8 -*-
"""Build a TUM-style monocular sequence folder for one self-sequence chunk."""
import argparse
import os
from pathlib import Path

SELF = Path(os.environ.get("SELF_ROOT", "<PROJECT_ROOT>"))
FRAMES = Path(os.environ.get("SELF_FRAMES", str(SELF / "selfcollected/yolo_input/frames")))
ROOT = Path(os.environ.get("ORB_SEQ_ROOT", str(SELF / "benchmarks/repos/orb_self")))
T0 = 1677730093.022842

ap = argparse.ArgumentParser()
ap.add_argument("--start", type=int, required=True)
ap.add_argument("--end", type=int, required=True)
args = ap.parse_args()
s, e = args.start, args.end
out = ROOT / ("chunk_%d_%d" % (s, e))
(out / "rgb").mkdir(parents=True, exist_ok=True)

lines = []
for i in range(s, e + 1):
    name = "%06d.jpg" % i
    dst = out / "rgb" / name
    if not dst.exists():
        os.symlink(FRAMES / ("frame%06d.jpg" % i), dst)
    lines.append("%.9f rgb/%s" % (T0 + i / 15.0, name))
(out / "rgb.txt").write_text("\n".join(lines) + "\n")
print("built", out, len(lines))
