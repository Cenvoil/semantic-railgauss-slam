# Changelog

## 2026-11-24 — v1.1 (paper rebuild)

- Added `paper/results/same_resolution_3.1.csv`: same-resolution 0.5× control on 3.1
  (ours 19.35 dB / 1.67 cm; SGS official 15.43 dB / 1563.64 cm).
- ATE labels for the paper's main results changed from w2c-t to **camera-center**
  (FAST-LIO2 on 5.1 is 9.85 cm in this convention). The L7 development table in
  `README.md` is retained as-is; see `PAPER.md` for the paper-aligned numbers.
- Added `paper/results/self_psnr_baselines.csv`.
- No change to the method code or configurations.

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
