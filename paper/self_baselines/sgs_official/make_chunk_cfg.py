# -*- coding: utf-8 -*-
"""Create an SGS official run config for a self-sequence chunk."""
import io
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = os.environ.get("SGS_RUN_ROOT", "<PROJECT_ROOT>/sgs_official_run")
src = os.path.join(ROOT, "configs/railway/self_100_official.py")
s, e = int(sys.argv[1]), int(sys.argv[2])
name = "chunk_%d_%d" % (s, e)
dst = os.path.join(ROOT, "configs/railway/self_chunk_%d_%d.py" % (s, e))
txt = io.open(src, encoding="utf-8").read()
txt = txt.replace('scene_name = "selfcollected_100"', 'scene_name = "%s"' % name)
txt = txt.replace('run_name = "self_100_official"', 'run_name = "self_chunk_%d_%d"' % (s, e))
io.open(dst, "w", encoding="utf-8").write(txt)
print("wrote", dst)
