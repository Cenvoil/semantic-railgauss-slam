# -*- coding: utf-8 -*-
"""Save GigaSLAM keyframe poses without running the rendering evaluation."""
import os
import shutil
import sys

ROOT = os.environ.get("GIGASLAM_ROOT", "<PROJECT_ROOT>/gigaslam_osdar")
E = os.path.join(ROOT, "utils/eval_utils.py")
S = os.path.join(ROOT, "slam.py")

s = open(E).read()
if "def save_trajectory" not in s:
    shutil.copy(E, E + ".pretraj")
    s += '''

def save_trajectory(frames, save_dir):
    """Write keyframe poses in the same format as eval_rendering."""
    def gen_pose_matrix(R, T):
        pose = np.eye(4)
        pose[0:3, 0:3] = R.cpu().numpy()
        pose[0:3, 3] = T.cpu().numpy()
        return pose

    with open(os.path.join(save_dir, "poses_idx.txt"), "w") as f:
        for idx in frames.keys():
            f.write(f"{idx}\\n")
    with open(os.path.join(save_dir, "poses_est.txt"), "w") as f:
        for idx in frames.keys():
            frame = frames[idx]
            pose_est = np.linalg.inv(gen_pose_matrix(frame.R, frame.T))
            f.write(" ".join(map(str, pose_est.flatten())) + "\\n")
'''
    open(E, "w").write(s)
    print("eval_utils patched")

s = open(S).read()
if "save_trajectory" in s:
    print("slam.py already patched")
    sys.exit(0)
shutil.copy(S, S + ".pretraj")
old = """        self.frontend.run()
        backend_queue.put(["pause"])

        end.record()
"""
new = """        self.frontend.run()
        backend_queue.put(["pause"])

        if self.config["Results"]["save_results"]:
            from utils.eval_utils import save_trajectory
            save_trajectory(self.frontend.cameras, self.save_dir)

        end.record()
"""
assert old in s, "anchor not found"
open(S, "w").write(s.replace(old, new))
print("slam.py patched")
