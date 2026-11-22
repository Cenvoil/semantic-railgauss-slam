# -*- coding: utf-8 -*-
"""Wrap MonoGS ATE evaluation so a degenerate monocular trajectory does not abort the run."""
import os
import shutil
import sys

p = os.path.join(os.environ.get("MONOGS_ROOT", "<PROJECT_ROOT>/monogs_run"),
                 "utils/eval_utils.py")
s = open(p).read()
if "eval_ate skipped" in s:
    print("already patched")
    sys.exit(0)
shutil.copy(p, p + ".preevalpatch")
old = """    ate = evaluate_evo(
        poses_gt=trj_gt_np,
        poses_est=trj_est_np,
        plot_dir=plot_dir,
        label=label_evo,
        monocular=monocular,
    )"""
new = """    try:
        ate = evaluate_evo(
            poses_gt=trj_gt_np,
            poses_est=trj_est_np,
            plot_dir=plot_dir,
            label=label_evo,
            monocular=monocular,
        )
    except Exception as exc:
        print("eval_ate skipped:", type(exc).__name__, flush=True)
        ate = float("nan")"""
assert old in s, "pattern not found"
open(p, "w").write(s.replace(old, new))
print("patched", p)
