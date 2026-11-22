# -*- coding: utf-8 -*-
"""Create the SGS-SLAM official config for the self-collected 100-frame chunk."""
import io
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = os.environ.get("SGS_RUN_ROOT", "<PROJECT_ROOT>/sgs_official_run")
SGS_DATA = os.environ.get("SGS_DATA", "<PROJECT_ROOT>/SGS-SLAM/data")
src = os.path.join(ROOT, "configs/railway/d8_31_l4_official.py")
dst = os.path.join(ROOT, "configs/railway/self_100_official.py")

s = io.open(src, encoding="utf-8").read()
rep = {
    'scene_name = "osdar23_3_fire_site_3.1"': 'scene_name = "selfcollected_100"',
    'run_name = "d8_31_l4_official"': 'run_name = "self_100_official"',
    'group_name = "Railway_d"': 'group_name = "Railway_self_official"',
    'basedir="./data"': 'basedir=%r' % SGS_DATA,
    'gradslam_data_cfg="./configs/data/railway_osdar_official.yaml"':
        'gradslam_data_cfg="./configs/data/railway_self_100.yaml"',
    'desired_image_height=1878': 'desired_image_height=720',
    'desired_image_width=3084': 'desired_image_width=1280',
    'num_semantic_classes=29': 'num_semantic_classes=19',
}
for a, b in rep.items():
    assert a in s, a
    s = s.replace(a, b)
io.open(dst, "w", encoding="utf-8").write(s)

yaml = """dataset_name: 'railway'
camera_params:
  image_height: 720
  image_width: 1280
  fx: 2371.3075
  fy: 2371.3075
  cx: 700.2272
  cy: 364.5387
  png_depth_scale: 1.0
  crop_edge: 0
"""
io.open(os.path.join(ROOT, "configs/data/railway_self_100.yaml"), "w", encoding="utf-8").write(yaml)
print("wrote", dst)
