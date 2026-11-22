# -*- coding: utf-8 -*-
"""Build a GigaSLAM KITTI-style frame folder and config for one self-sequence segment."""
import argparse
import os
from pathlib import Path

SELF = Path(os.environ.get("SELF_ROOT", "<PROJECT_ROOT>"))
FRAMES = Path(os.environ.get("SELF_FRAMES", str(SELF / "selfcollected/yolo_input/frames")))
ROOT = Path(os.environ.get("GIGASLAM_ROOT", str(SELF / "gigaslam_osdar")))
SNAPSHOT = os.environ.get("GIGASLAM_SNAPSHOT", str(ROOT / "snapshot_dir/1"))

ap = argparse.ArgumentParser()
ap.add_argument("--start", type=int, required=True)
ap.add_argument("--end", type=int, required=True)
ap.add_argument("--motion", type=float, default=2.0)
ap.add_argument("--loop", type=int, default=0)
args = ap.parse_args()
s, e = args.start, args.end
name = "self_%d_%d" % (s, e)
out = ROOT / name
out.mkdir(parents=True, exist_ok=True)
for i in range(s, e + 1):
    src = FRAMES / ("frame%06d.jpg" % i)
    dst = out / ("%06d.jpg" % i)
    if not dst.exists():
        os.symlink(src, dst)

cfg = (ROOT / "configs" / (name + ".yaml"))
cfg.write_text("""inherit_from: "configs/base_config.yaml"

Results:
  save_trj: True
  eval_rendering: False

Dataset:
  type: 'kitti'
  sensor_type: 'mono'
  color_path: "%s"
  Calibration:
    fx: 2371.3075
    fy: 2371.3075
    cx: 700.2272
    cy: 364.5387
    k1: 0.022928
    k2: 0.0
    p1: 0.0
    p2: 0.0
    k3: 0.0
    width: 1280
    height: 720
    distorted: True

SLAM:
  viz: False
  loop_closure: %s
  motion_thresh: %.3f
  min_kf_gap: 3
  max_kf_gap: 10
  loop_detect_thresh: 0.034
  3d2d_thread: 20
  2d2d_thread: 20
  n_max: 5000

DepthModel:
  from_huggingface: False
  huggingface:
    model_name: 'lpiccinelli/unidepth-v2-vitl14'
    commit_hash: '1d0d3c52f60b5164629d279bb9a7546458e6dcc4'
  local_snapshot_path: '%s'

Hierarchical:
  voxel_size_lis: [0.1, 0.25, 1, 5, 25]
  distance_lis: [20.0, 40.0, 80.0, 160.0]
  n_offsets: 24
  point_ratio: 26
  appearance_dim: 64
  color_refinement_iter: 200
  rendering_width: 480
  upsampling_method: 'bicubic'
""" % (str(out), "True" if args.loop else "False", args.motion, SNAPSHOT))
print("built", out, "frames", e - s + 1, "config", cfg)
