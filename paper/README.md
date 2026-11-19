# Paper-specific add-ons

This folder contains the S2 semantic-head scripts and the CSV/trajectory files used
in the paper. The main SLAM code is the repository root (RailGauss-OSDaR23).

- `livogs_train.py`, `patch_s2.py` — per-Gaussian semantic head on LIVO-GS.
- `P3_实现与复现.md` — implementation/reproduction notes.
- `results/` — CSV versions of the main tables.
- `self_traj/` — derived trajectories on the self-collected sequence.

The self-collected raw sequence is not redistributed; contact the authors subject to
railway operator restrictions.

Paths in `patch_s2.py` use the placeholder `<PROJECT_ROOT>`; replace it with your
local checkout of LIVO-GS before running.
