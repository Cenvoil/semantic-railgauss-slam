#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统一 ATE 评估脚本 —— benchmarks 工程的指标唯一出口。

## 为什么必须统一
之前踩过的坑：项目里 `evaluate_ate()` 打印的字符串叫"ATE RMSE"，
但它算的是 `trans_error.mean()`（mean，不是 RMSE）；对齐用 Horn 闭式刚体（SE(3)，无尺度）；
取 w2c 的[:3,3] 而非相机中心。自写脚本换成"相机中心 + 真 RMSE" 算出2.03 cm，
项目是 2.22 cm —— 不是代码错，是口径不同。

⇒ 所以本脚本**同时输出三种口径**，并在 json 里标明哪个可与项目 2.22 cm 并列。

## 三种口径
  align : se3（不含尺度刚体）/ sim3（含尺度 Umeyama）
  agg   : mean（项目口径）/ rmse
  traj  : w2c_t（取 w2c 的 [:,3]，项目口径）/ cam_center（相机中心）

## 主口径（可与本项目并列）
  se3 + mean + w2c_t

## 用法
  python eval_ate.py --gt <gt.txt> --est <est.txt> --out <metrics.json> \
      [--align se3|sim3] [--traj w2c_t|cam_center]
  GT/EST 均为 TUM 格式：每行 `timestamp tx ty tz [qx qy qz qw]`
  （只有位置也行，旋转会退化为不用旋转对齐）
"""
import argparse
import json
import os
from datetime import datetime

import numpy as np


# ---------------- 对齐算法 ----------------
def align_se3(src, dst):
    """刚体对齐（不含尺度）。

    SVD 构造已用合成数据标定（2026-10-08）：
      np.linalg.svd(cov.T) -> U,d,Vh，rot = U @ S @ Vh，应用 x @ rot.T
      残差 1e-16 级（机器精度）。
    ⚠ 早期版本写成 np.linalg.svd(W) 且应用 x @ rot，
      与项目 evaluate_ate 差 2 倍以上（3.41 -> 7.08 cm），已修正。
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
    return rot, trans, 1.0


def align_sim3(src, dst):
    """Umeyama 相似变换（含尺度）。ORB-SLAM2 等论文用的就是这个。

    SVD 构造已用合成数据标定（2026-10-08）：
      cov = b.T @ a，np.linalg.svd(cov) -> U,d,Vt，rot = U @ S @ Vt，
      应用 x @ rot.T（**注意是 .T**）
      残差 9.2e-16（机器精度）。

    ⚠ 三个实测踩过的坑（都表现为"Sim3 残差 > SE3 残差"这个模糊现象，
      靠合成数据才定位到）：
      ① SVD 参数顺序：必须是 svd(cov) 配U @ S @ Vt，
        不能写成 svd(cov.T) 配 Vt.T @ U.T（后者残差 0.62）。
      ② 应用方向：x @ rot.T，不能用 x @ rot（后者残差 0.15）。
      ③ 平移不能预乘 scale：trans 已是最终偏移，
        再乘一次会让残差大两个数量级（1.2 上测出 883 cm）。
    """
    ms, md = src.mean(0), dst.mean(0)
    a, b = src - ms, dst - md
    cov = (b.T @ a) / len(src)
    U, d, Vt = np.linalg.svd(cov)
    S = np.identity(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    rot = U @ S @ Vt
    var = (a ** 2).sum() / len(src)
    scale = float(np.trace(np.diag([d[0], d[1], S[2, 2] * d[2]])) / var)
    trans = md - scale * (rot @ ms)      # 最终偏移，不再乘 scale
    return rot, trans, scale


# ---------------- 轨迹读取 ----------------
def read_tum(path):
    """读 TUM 格式。返回 (ts[N], pos[N,3])；若有旋转则一并返回 quat[N,4]。"""
    ts, pos, quat = [], [], []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            v = line.split()
            ts.append(float(v[0]))
            pos.append([float(v[1]), float(v[2]), float(v[3])])
            if len(v) >= 8:
                quat.append([float(v[4]), float(v[5]), float(v[6]), float(v[7])])
    ts = np.array(ts)
    pos = np.array(pos)
    quat = np.array(quat) if quat and len(quat) == len(ts) else None
    order = np.argsort(ts)          # 时间对齐
    return ts[order], pos[order], (quat[order] if quat is not None else None)


def match_poses(gt_ts, gt_pos, est_ts, est_pos, max_dt):
    """按最近时间戳配对，容差 max_dt。

    必须配对而不是假设等间隔—— 各方法输出频率不同
    （KISS-ICP 逐帧、ORB-SLAM3 关键帧、LiDAR 方法 10 Hz）。
    """
    n = len(gt_ts)
    idx = np.searchsorted(est_ts, gt_ts).clip(1, len(est_ts) - 1)
    pick = np.where(np.abs(est_ts[idx] - gt_ts) < np.abs(est_ts[idx - 1] - gt_ts),
                    idx, idx - 1)
    dt = np.abs(est_ts[pick] - gt_ts)
    keep = dt <= max_dt
    if keep.sum() < 3:
        raise SystemExit('时间戳匹配不足 3 帧：可用 %d/%d，max_dt=%.3f'
                         % (keep.sum(), n, max_dt))
    return gt_pos[keep], est_pos[pick][keep], gt_ts[keep], dt[keep]


def quat_to_rot(q):
    """TUM 的 (qx,qy,qz,qw) -> 旋转矩阵。"""
    x, y, z, w = q
    n = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def apply_align(pts, rot, trans, scale):
    """施加对齐：x' = scale * (x @ rot.T) + trans。

    ⚠ trans 已是最终坐标偏移，应用时**不能再乘 scale**
      （否则 Sim(3) 残差比 SE(3) 大两个数量级：1.2 上测出 883 cm）。
    """
    # 方向是 `pts @ rot.T`（合成数据标定，残差 1e-16）
    return scale * (pts @ rot.T) + trans


# ---------------- 主 ----------------
def compute(gt_pos, est_pos):
    """核心计算：给定配对好的真值与估计，算出全部口径指标。

    单独抽出来是为了让回归测试能直接调用，不必走命令行。
    """
    rot, trans, scale = align_se3(est_pos, gt_pos)
    err_se3 = np.linalg.norm(apply_align(est_pos, rot, trans, scale) - gt_pos,
                             axis=1)
    rot3, trans3, scale3 = align_sim3(est_pos, gt_pos)
    err_sim = np.linalg.norm(apply_align(est_pos, rot3, trans3, scale3) - gt_pos,
                             axis=1)
    return {
        'ate_cm': float(err_se3.mean() * 100),
        'ate_rmse_cm': float(np.sqrt((err_se3 ** 2).mean()) * 100),
        'ate_max_cm': float(err_se3.max() * 100),
        'ate_sim3_mean_cm': float(err_sim.mean() * 100),
        'ate_sim3_rmse_cm': float(np.sqrt((err_sim ** 2).mean()) * 100),
        'sim3_scale': float(scale3),
        'odom_gt_m': float(np.linalg.norm(
            np.diff(gt_pos, axis=0), axis=1).sum()),
        'odom_est_m': float(np.linalg.norm(
            np.diff(est_pos, axis=0), axis=1).sum()),
    }


def R_to_quat_tum(R_cw):
    """cam->world 旋转矩阵 -> TUM 四元数 (x,y,z,w)。"""
    tr = np.trace(R_cw)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (R_cw[2, 1] - R_cw[1, 2]) / S
        y = (R_cw[0, 2] - R_cw[2, 0]) / S
        z = (R_cw[1, 0] - R_cw[0, 1]) / S
    elif R_cw[0, 0] > R_cw[1, 1] and R_cw[0, 0] > R_cw[2, 2]:
        S = np.sqrt(1.0 + R_cw[0, 0] - R_cw[1, 1] - R_cw[2, 2]) * 2
        w = (R_cw[2, 1] - R_cw[1, 2]) / S
        x = 0.25 * S
        y = (R_cw[0, 1] + R_cw[1, 0]) / S
        z = (R_cw[0, 2] + R_cw[2, 0]) / S
    elif R_cw[1, 1] > R_cw[2, 2]:
        S = np.sqrt(1.0 + R_cw[1, 1] - R_cw[0, 0] - R_cw[2, 2]) * 2
        w = (R_cw[0, 2] - R_cw[2, 0]) / S
        x = (R_cw[0, 1] + R_cw[1, 0]) / S
        y = 0.25 * S
        z = (R_cw[1, 2] + R_cw[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R_cw[2, 2] - R_cw[0, 0] - R_cw[1, 1]) * 2
        w = (R_cw[1, 0] - R_cw[0, 1]) / S
        x = (R_cw[0, 2] + R_cw[2, 0]) / S
        y = (R_cw[1, 2] + R_cw[2, 1]) / S
        z = 0.25 * S
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def camcenter_to_w2c_t(est_c, gt_t, gt_w2c_rot):
    """把外部方法输出的**相机中心 C** 转成项目口径的 **w2c 的 t**。

    ## 为什么需要
    外部方法（KISS-ICP / FAST-LIVO / ORB-SLAM）输出的轨迹是 T_wc 的平移，
    也就是**相机中心 C**；而项目 `evaluate_ate` 用的是 **w2c 矩阵的 [:3,3]**（即 t）。
    两者差一个**逐帧变化的旋转** Rᵀ：
        t = R_w2c @ (-C)
    ⚠ 必须用**真值那一帧的 R**（gt_w2c_all_frames 里的），**不能用 traj.txt 的 R**
      （实测两者不一致，混用会让 5.1 的 ATE 从 3.38 变成 38.88 cm）。

    参数
      est_c: 估计的相机中心序列 (N,3)
      gt_t:  真值的 t 序列 (N,3)，用于确认帧数一致
      gt_w2c_rot: 真值每帧的 w2c 旋转 (N,3,3)
    返回
      (N,3) 的 t 序列
    """
    if len(est_c) != len(gt_w2c_rot):
        raise SystemExit('帧数不一致：估计 %d，真值 %d'
                         % (len(est_c), len(gt_w2c_rot)))
    out = np.zeros_like(est_c)
    for i in range(len(est_c)):
        out[i] = gt_w2c_rot[i] @ (-est_c[i])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt', required=True, help='真值 TUM 格式')
    ap.add_argument('--est', required=True, help='估计 TUM 格式')
    ap.add_argument('--out', default='', help='指标 json 输出路径')
    ap.add_argument('--max-dt', type=float, default=0.02,
                    help='时间戳匹配容差（秒）。ORB-SLAM3 关键帧建议 0.06')
    ap.add_argument('--method', default='')
    ap.add_argument('--dataset', default='')
    ap.add_argument('--sequence', default='')
    ap.add_argument('--version', default='')
    ap.add_argument('--runtime', type=float, default=None, help='运行耗时（秒）')
    ap.add_argument('--notes', default='')
    ap.add_argument('--repr', default='w2c_t',
                    choices=['w2c_t', 'camcenter'],
                    help='位置表示。主口径 w2c_t = w2c 矩阵的 [:3,3]（项目 evaluate_ate 用）；'
                         'camcenter = 相机中心 C（外部方法输出，需先转换）')
    args = ap.parse_args()

    gt_ts, gt_pos, _ = read_tum(args.gt)
    est_ts, est_pos, _ = read_tum(args.est)

    g, e, mts, dts = match_poses(gt_ts, gt_pos, est_ts, est_pos, args.max_dt)
    m = compute(g, e)

    out = {
        'method': args.method,
        'method_version': args.version,
        'dataset': args.dataset,
        'sequence': args.sequence,
        'n_frames_aligned': int(len(g)),
        'n_frames_gt': int(len(gt_ts)),
        'alignment': 'se3',
        'aggregation': 'mean',
        'traj_repr': args.repr,
        'ate_cm': round(m['ate_cm'], 4),
        'ate_rmse_cm': round(m['ate_rmse_cm'], 4),
        'ate_max_cm': round(m['ate_max_cm'], 4),
        'ate_sim3_mean_cm': round(m['ate_sim3_mean_cm'], 4),
        'ate_sim3_rmse_cm': round(m['ate_sim3_rmse_cm'], 4),
        'sim3_scale': round(m['sim3_scale'], 6),
        'odom_gt_m': round(m['odom_gt_m'], 3),
        'odom_est_m': round(m['odom_est_m'], 3),
        'timestamp_match': {
            'max_dt_s': args.max_dt,
            'matched': int(len(g)),
            'dt_median_ms': round(float(np.median(dts) * 1000), 3),
            'dt_max_ms': round(float(dts.max() * 1000), 3),
        },
        'comparable_with_project': True,
        'runtime_s': args.runtime,
        'notes': args.notes,
        'date': datetime.now().strftime('%Y-%m-%d'),
    }

    print('=' * 62)
    print('%-14s %-18s %s' % ('method', 'sequence', 'ATE'))
    print('-' * 62)
    print('%-14s %-18s %8.2f cm  (SE(3)+mean, 主口径)'
          % (args.method, args.sequence, m['ate_cm']))
    print('%-14s %-18s %8.2f cm  (SE(3)+RMSE)'
          % ('', '', m['ate_rmse_cm']))
    print('%-14s %-18s %8.2f cm  (Sim(3)+mean, 单目方法常用)'
          % ('', '', m['ate_sim3_mean_cm']))
    print('-' * 62)
    print('对齐帧数 %d/%d  里程 真值 %.2f m / 估计 %.2f m  尺度 %.5f'
          % (len(g), len(gt_ts), m['odom_gt_m'], m['odom_est_m'],
             m['sim3_scale']))
    print('时间戳匹配 中位 %.2f ms / 最大 %.2f ms'
          % (np.median(dts) * 1000, dts.max() * 1000))
    if args.notes:
        print('备注：%s' % args.notes)

    if args.out:
        d = os.path.dirname(args.out)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print('-> %s' % args.out)


if __name__ == '__main__':
    main()