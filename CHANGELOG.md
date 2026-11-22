# Changelog

## 2026-11-22 — self-collected baseline update (main)

- Added `paper/self_baselines/` with the self-collected sequence baseline
  tooling: FAST-LIVO2 reference player/configs, FAST-LIO2 runner, MonoGS
  segmented runner and segment-wise Sim(3) merge, GigaSLAM bounded-keyframe
  patch and chained merge, SGS official 100-frame chunk builder/exporter,
  ORB-SLAM3 sequence builder/camera yaml/CMake shim, and `eval_tum_vs_ref.py`.
- Replaced `paper/results/self_baselines.csv` with the FAST-LIVO2-reference
  protocol (13 rows; previous file used the legacy pseudo-reference).
- Added derived trajectories to `paper/self_traj/`: `fastlivo2_ref_tum.txt`,
  `fastlio2_traj_tum.txt`, `monogs_full_tum.txt`, `gigaslam_full_tum.txt`,
  three `sgs_official_*.txt` chunks, and `orb_slam3_full_tum.txt`.
- Added the Table 13 summary to `README.md`, `PAPER.md` and
  `paper/P3_实现与复现.md`.
- No change to the method code, configs or the OSDaR23 results.

## 2026-11-19 — v1.0 (`p3-v1.0`, DOI 10.5281/zenodo.24364397)

- First public paper release: sanitized code snapshot, S2 semantic-head
  scripts, result CSVs, derived self-collected trajectories.
