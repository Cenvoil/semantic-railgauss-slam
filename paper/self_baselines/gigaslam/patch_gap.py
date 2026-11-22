# -*- coding: utf-8 -*-
"""Add a bounded keyframe interval to GigaSLAM's flow gate (min/max gap in frames)."""
import os
import shutil
import sys

p = os.path.join(os.environ.get("GIGASLAM_ROOT", "<PROJECT_ROOT>/gigaslam_osdar"),
                 "utils/slam_frontend.py")
s = open(p).read()
if "min_kf_gap" in s:
    print("already patched")
    sys.exit(0)
shutil.copy(p, p + ".pregap")

old_init = "        self.motion_thresh = config['SLAM']['motion_thresh']\n"
new_init = ("        self.motion_thresh = config['SLAM']['motion_thresh']\n"
            "        self.min_kf_gap = int(config['SLAM'].get('min_kf_gap', 0) or 0)\n"
            "        self.max_kf_gap = int(config['SLAM'].get('max_kf_gap', 0) or 0)\n"
            "        self._last_kf_idx = 2\n")
assert old_init in s
s = s.replace(old_init, new_init)

old_gate = """                if cur_frame_idx > 2:
                    flow = self.classic_tracking.motion_flow_keyframe(viewpoint.image_ori, prev_kf=False)
                    Log(f'FLOW {flow:.2f}')
                    if flow < self.motion_thresh:
                        cur_frame_idx += 1

                        Log(f'Skip frame {cur_frame_idx}')
                        continue
"""
new_gate = """                if cur_frame_idx > 2:
                    flow = self.classic_tracking.motion_flow_keyframe(viewpoint.image_ori, prev_kf=False)
                    since = cur_frame_idx - self._last_kf_idx
                    skip = False
                    if self.min_kf_gap > 0 and since < self.min_kf_gap:
                        skip = True
                    elif flow < self.motion_thresh and (self.max_kf_gap <= 0 or since < self.max_kf_gap):
                        skip = True
                    if skip:
                        cur_frame_idx += 1
                        Log(f'Skip frame {cur_frame_idx}')
                        continue
                    self._last_kf_idx = cur_frame_idx
"""
assert old_gate in s
s = s.replace(old_gate, new_gate)
open(p, "w").write(s)
print("patched gap gate")
