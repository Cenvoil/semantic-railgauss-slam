# Self-collected sequence baselines (Table 13)

Scripts and protocol notes for the baseline comparison on the self-collected
Baoding–Tianjin railway sequence (3,044 frames, 1280×720, 15 Hz, ≈1.11 km).
The paper's Table 13 is generated from these runs; the numerical table is in
`../results/self_baselines.csv` and the trajectories are in `../self_traj/`.

The raw sequence is **not** redistributed. The scripts below assume the dataset
and the third-party repositories are already installed under a common root.

## 1. Reference trajectory

There is no INS/GNSS ground truth on this sequence. The reference is a
FAST-LIVO2 (LIVO-GS fork) run on the LiDAR-IMU data:

- 2,033 poses, 1,107 m, TUM format: `../self_traj/fastlivo2_ref_tum.txt`;
- the player also writes `scans_wall.txt` (scan stamp → wall-clock time), used
  to map estimator poses to scan stamps;
- ATE is reported after a single global SE(3) alignment (Sim(3) for monocular
  systems, see below), RMSE over all matched poses.

Reference mileage is 1,107 m; `ate_over_reference` in the CSV is ATE RMSE
divided by this mileage.

## 2. Protocol per method

| Method | Modality | How the full 0–3043 trajectory is obtained | Alignment |
|---|---|---|---|
| FAST-LIVO2 | LiDAR-IMU | single full run (reference) | — |
| KISS-ICP | LiDAR-only | single full run | SE(3) |
| GenZ-ICP | LiDAR-only | single full run | SE(3) |
| FAST-LIO2 | LiDAR-IMU | single full run | SE(3) |
| Ours | LiDAR-IMU + 3DGS | single full run | SE(3) |
| MonoGS | monocular RGB | 10 overlapping 350-frame runs; each segment Sim(3)-aligned to the reference, then concatenated | per-segment Sim(3) |
| GigaSLAM | monocular RGB | 10 overlapping 350-frame runs; chained through nearest keyframes (±2 frames, 9 links, scales 1.01–1.05) | chained Sim(3) links |
| SGS official | RGB-D (pseudo depth) | three 100-frame runs (600–699, 1200–1299, 2400–2499); SGS cannot initialize on long chunks | SE(3) per chunk |
| ORB-SLAM3 | monocular RGB | single full run, 992 keyframes, no tracking loss | Sim(3) |
| Ours (100-frame chunks) | LiDAR-IMU + 3DGS | same three chunks as SGS official | SE(3) per chunk |

Numbers (ATE RMSE / final error, metres):

| Method | Segment | ATE | Final |
|---|---|---:|---:|
| KISS-ICP | 0–3043 | 229.4 | 498.0 |
| GenZ-ICP | 0–3043 | 2.7 | 2.8 |
| FAST-LIO2 | 0–3043 | 1.1 | 2.1 |
| Ours | 0–3043 | 44.7 | 77.2 |
| MonoGS | 0–3043 | 34.4 | — |
| ORB-SLAM3 | 0–3043 | 3.5 | 8.4 |
| GigaSLAM | 0–3043 | 43.5 | 87.1 |
| SGS official | 600–699 / 1200–1299 / 2400–2499 | 0.7 / 2.2 / 30.5 | 0.8 / 2.4 / 47.8 |
| Ours (100-frame chunks) | 600–699 / 1200–1299 / 2400–2499 | 1.1 / 0.5 / 3.2 | 1.8 / 0.7 / 5.9 |

The three SGS chunks are the only same-input, same-granularity comparison with
the official SGS-SLAM configuration; the 100-frame rows show our system under
the identical chunking.

## 3. Directory layout

```
self_baselines/
├── eval_tum_vs_ref.py         # shared ATE against the FAST-LIVO2 reference
├── fastlivo2/                 # reference run: LiDAR-IMU player + configs
├── fastlio2/                  # LiDAR-IMU baseline runner
├── monogs/                    # segment builder, queue, merge, eval
├── gigaslam/                  # segment builder, gap patch, trajectory dump, chain merge
├── sgs_official/              # 100-frame chunk builder, config, trajectory export
└── orbslam3/                  # sequence builder, camera yaml, CMake build fix
```

## 4. Environment and expected layout

All scripts use `SELF_ROOT` (default `<PROJECT_ROOT>`), matching the server
layout used for the experiments:

```
$SELF_ROOT/
├── selfcollected/
│   ├── yolo_input/frames/frame%06d.jpg
│   ├── yolo_input/frame_map.csv
│   ├── yolo_full/semantic_ids/, semantic_colors/
│   └── lidar_all/                 # per-scan npz with key "xyz" + frame_lidar_map.csv
├── self_livo/                     # player, scan wall times, lidar3d.bin, FAST-LIVO2 output
├── monogs_run/                    # MonoGS checkout + dataset/ + configs/ + results/
├── gigaslam_osdar/                # GigaSLAM checkout + configs/ + results/
├── SGS-SLAM/                      # official SGS-SLAM checkout (data/ holds chunks)
├── sgs_official_run/              # patched SGS-SLAM run tree
└── benchmarks/repos/ORB_SLAM3/    # ORB-SLAM3 checkout
```

Shell runners also read `<CONDA_ROOT>` for the Python environment and
`<WORKSPACE>` for the catkin workspace; set them in the script header before
running.

## 5. Commands

### 5.1 FAST-LIVO2 reference

```bash
# 1) pack lidar_all npz scans into one binary (parallel reads)
python fastlivo2/pack_lidar3d.py

# 2) launch LIVO-GS + play scans (3D clouds, no per-point time)
PLAY_SPEED=1.0 bash fastlivo2/run_self_livo.sh
```

The player publishes `sensor_msgs/PointCloud2` in the Ouster field layout on
`/livox/lidar`, IMU on `/livox/imu` (accelerations already in m/s²), and records
the odometry topic to TUM. `scans_wall.txt` is required by the evaluator.

### 5.2 FAST-LIO2

```bash
bash fastlio2/run_self_lio.sh
```

### 5.3 MonoGS

```bash
# one queue per GPU; queue file: one "start end" pair per line
CUDA_VISIBLE_DEVICES=2 bash monogs/run_seg_queue.sh 2 queue_gpu2.txt a
CUDA_VISIBLE_DEVICES=3 bash monogs/run_seg_queue.sh 3 queue_gpu3.txt b

# merge the 10 segments (segment-wise Sim(3)) and evaluate
python monogs/merge_segwise.py
python monogs/eval_merged.py
```

`patch_eval.py` wraps MonoGS's own ATE call so a degenerate monocular segment
does not abort the run.

### 5.4 GigaSLAM

```bash
# apply the bounded-keyframe patch once to the GigaSLAM checkout
python gigaslam/patch_gap.py
python gigaslam/patch_traj.py

bash gigaslam/run_giga_queue.sh 2 queue_gpu2.txt a
bash gigaslam/run_giga_queue.sh 3 queue_gpu3.txt b

python gigaslam/merge_giga_chain.py
python eval_tum_vs_ref.py ../self_traj/gigaslam_full_tum.txt
```

`motion_thresh=2.0`, `min_kf_gap=3`, `max_kf_gap=10`, loop closure and rendering
evaluation disabled. `patch_traj.py` writes `poses_idx.txt` / `poses_est.txt`
without running `eval_rendering`.

### 5.5 SGS official

```bash
python sgs_official/make_cfg.py
python sgs_official/make_chunk.py --start 600 --end 699 --depth <depth_pipeline_out>
python sgs_official/make_chunk_cfg.py 600 699
# run the generated config inside the SGS-SLAM tree
python sgs_official/export_sgs_traj.py <run_name> 600 100 <out_tum>
```

Each chunk needs LiDAR-anchored pseudo depth (`depths/`, `valid/`) built with
UniDepthV2 plus the LiDAR anchoring script from `tools/data_prep/`.

### 5.6 ORB-SLAM3

```bash
# CMake >= 3.28 needs a policy shim for the vendored third-party projects
bash orbslam3/build_fix.sh

python orbslam3/make_orb_seq.py --start 0 --end 3043
./Examples/Monocular/mono_tum Vocabulary/ORBvoc.txt orbslam3/self_1280.yaml \
    <orb_root>/chunk_0_3043
python eval_tum_vs_ref.py ../self_traj/orb_slam3_full_tum.txt
```

`mono_tum` is patched to non-viewer mode; the camera yaml includes the
`Viewer.Viewpoint*` keys required by current ORB-SLAM3.

## 6. Caveats

- FAST-LIVO2 is a LiDAR-IMU odometry reference, not survey-grade truth; all
  errors are relative to it.
- MonoGS segments have arbitrary monocular scale; the per-segment Sim(3)
  alignment removes inter-segment drift, so the reported ATE is a
  segment-wise-aligned one.
- SGS official cannot complete the full sequence; only 100-frame chunks are
  reported. The high-speed chunk (2400–2499, 15–19 m/s) drifts strongly.
- The system runs offline at ≈64 s/frame on an RTX 3090; these baselines are
  offline evaluations, not real-time claims.
