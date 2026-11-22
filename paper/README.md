# Paper-specific add-ons

This folder contains the S2 semantic-head scripts, the self-collected baseline
tooling, and the CSV/trajectory files used in the paper. The main SLAM code is
the repository root (RailGauss-OSDaR23).

- `livogs_train.py`, `patch_s2.py` — per-Gaussian semantic head on LIVO-GS.
- `P3_实现与复现.md` — implementation/reproduction notes.
- `self_baselines/` — scripts for the self-collected sequence baselines
  (Table 13): FAST-LIVO2 reference, FAST-LIO2, MonoGS, GigaSLAM, SGS official,
  ORB-SLAM3, plus the shared ATE evaluator. See its `README.md` for the
  protocol, expected layout and commands.
- `results/` — CSV versions of the main tables. `self_baselines.csv` is the
  Table 13 source (FAST-LIVO2 reference, SE(3) unless noted; MonoGS is
  segment-wise Sim(3); ORB-SLAM3 is Sim(3)).
- `self_traj/` — derived trajectories on the self-collected sequence:
  `fastlivo2_ref_tum.txt` (reference, 2,033 poses, 1,107 m), `fastlio2_traj_tum.txt`,
  `monogs_full_tum.txt`, `gigaslam_full_tum.txt`, `orb_slam3_full_tum.txt`,
  the three `sgs_official_*.txt` chunks, and the older KISS/GenZ/pseudo-reference
  files retained for history.

The self-collected raw sequence is not redistributed; contact the authors subject to
railway operator restrictions.

Paths in `patch_s2.py` and `self_baselines/` use the placeholders `<PROJECT_ROOT>`,
`<CONDA_ROOT>` and `SELF_ROOT`; replace them with your local checkout before running.
