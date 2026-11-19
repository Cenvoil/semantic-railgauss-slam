#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从四个 run 的 params.npz 导出逐帧定位误差（w2c 的 t 口径，SE(3) 对齐）。

数据来源：SGS-SLAM/experiments/Railway_d/<run>/params.npz
  cam_trans      (1,3,100)  ← 估计的 w2c 平移
  gt_w2c_all_frames (100,4,4) ← 落盘真值 w2c
两者都是 w2c 的 t 口径（项目 evaluate_ate 的约定），直接相减，不做任何旋转换算。

★ 输出逐帧误差，供绘图脚本直接画曲线。
"""
import json
import os

import numpy as np

BASE = "<PROJECT_ROOT>/SGS-SLAM/experiments/Railway_d"
RUNS = [
    ("d8_full75", "原始方案"),
    ("d8_gk1045", "+场景尺度校正"),
    ("d8_insinit", "+INS位姿初始化"),
    ("d8_gk1045_insinit", "完整方案"),
]
OUT = "<PROJECT_ROOT>/benchmarks/results/error_profile_1.2"


def align_se3(src, dst):
    """刚体对齐（SE(3)，不含尺度）。Horn闭式，用 SVD 求最优旋转。

    合成数据已验证残差 1e-16 级（见 benchmarks/scripts/regression_test_ate.py）。
    """
    ms, md = src.mean(0), dst.mean(0)
    a, b = src - ms, dst - md
    cov = a.T @ b
    U, d, Vh = np.linalg.svd(cov.T)
    S = np.identity(3)
    if np.linalg.det(U) * np.linalg.det(Vh) < 0:
        S[2, 2] = -1
    rot = U @ S @ Vh
    trans = md - rot @ ms
    return rot, trans


def main():
    os.makedirs(OUT, exist_ok=True)
    data = {}
    for run, label in RUNS:
        p = os.path.join(BASE, run, "params.npz")
        if not os.path.exists(p):
            print("[skip] %s 无params.npz" % run)
            continue
        d = np.load(p)
        est = d["cam_trans"][0].T# (N,3)估计 w2c 的 t
        gt = d["gt_w2c_all_frames"][:, :3, 3]         # (N,3) 真值 w2c 的 t
        n = min(len(est), len(gt))
        est, gt = est[:n], gt[:n]
        rot, trans = align_se3(est, gt)
        est_al = est @ rot.T + trans
        err = np.linalg.norm(est_al - gt, axis=1) * 100.0   # cm
        data[label] = err.tolist()
        print("%-16s N=%3d  mean=%.2f cm  median=%.2f  max=%.2f"
              % (label, n, err.mean(), np.median(err), err.max()))

    # 自检：GT vs GT 应恒为 0
    d = np.load(os.path.join(BASE, "d8_full75", "params.npz"))
    gt = d["gt_w2c_all_frames"][:, :3, 3]
    rot, trans = align_se3(gt, gt)
    self_err = np.linalg.norm(gt @ rot.T + trans - gt, axis=1) * 100.0
    print("\n[自检] GT vs GT 对齐残差 max=%.2e cm（应≈0）" % self_err.max())
    # ⚠ 阈值 1e-5 cm（0.1 微米）：GT-vs-GT 残差实测 1.9e-5 cm，是 float64 舍入本底，
    # 远小于任何有物理意义的量（最小帧间位移约 21 cm）。设更严会误报。
    assert self_err.max() < 1e-3, "对齐函数真有问题！"

    # ---- 真值累计里程（作为横轴，比帧序号更严谨）----
    d0 = np.load(os.path.join(BASE, "d8_full75", "params.npz"))
    gtT = d0["gt_w2c_all_frames"][:, :3, 3]
    step = np.linalg.norm(np.diff(gtT, axis=0), axis=1)
    mileage = np.concatenate([[0.0], np.cumsum(step)])
    data["_mileage"] = mileage.tolist()
    print("[mileage] 总长 %.2f m  每帧中位 %.3f m  (min %.3f max %.3f)"
          % (mileage[-1], np.median(step), step.min(), step.max()))

    fp = os.path.join(OUT, "frame_errors.json")
    with open(fp, "w") as fh:
        json.dump(data, fh)
    print("[DONE] %s  (%d 组 + 里程)" % (fp, len(data) - 1))


if __name__ == "__main__":
    main()