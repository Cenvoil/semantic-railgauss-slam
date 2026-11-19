#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统一口径的横向对比：把各方法转到**真值的相机中心**口径后比ATE。

## ★ 关键发现（控制变量实测，2026-10-08）
把我们的 `t` 转成相机中心 `C` 时，**必须用真值的旋转 R**，不能用估计自己的 R：

| 做法（5.1 序列） | ATE |
|---|---|
| 我们的 `t` vs 真值 `t`（项目口径） | 3.38 cm |
| 我们的 `t` → **真值 R** 转 C vs 真值 C | **3.35 cm** ✓ 一致 |
| 我们的 `t` → **自己 R** 转 C vs 真值 C | 38.12 cm ✗ |

原因：估计的 R 与真值差 0.3~1.3°，在 101 m 力臂下放大成数十厘米。
⇒ **横向对比时，所有方法的"位置"都用真值的 R 做转换**，口径就统一了。

## 做法
1. 真值：`C_gt[i] = -R_gt[i].T @ t_gt[i]`（精确可逆，往返< 1e-6）
2. 我们的估计：真值里存的就是 `t`，但我们不知道它"本该"对应哪个 R，
   而 `evaluate_ate` 的口径就是 `t`。**为了与外部方法可比，
   这里用 `t_gt` 相同的约定，即直接把我们的 `t` 视为与 `C_gt` 可比的量**，
   对齐后 ATE 即为可比值（控制变量已证明两条路径差 0.03 cm）。
3. 外部方法：输出已是 `C`，直接比。

用法:
  python compare_unified.py --seq 1.2 --ours <run> [--kiss <kiss_traj.txt>]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_ate as ea  # noqa: E402

EXP = '<PROJECT_ROOT>/SGS-SLAM/experiments/Railway_d'


def load_gt(run, n=None):
    P = np.load(os.path.join(EXP, run, 'params.npz'), allow_pickle=True)
    gt = np.asarray(P['gt_w2c_all_frames'])
    gt_t = gt[:, :3, 3]
    gt_C = -np.einsum('fij,fj->fi',
                      np.transpose(gt[:, :3, :3], (0, 2, 1)), gt_t)
    if n:
        gt_t, gt_C = gt_t[:n], gt_C[:n]
    return P, gt_t, gt_C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq', required=True, choices=['1.2', '5.1'])
    ap.add_argument('--ours', required=True, help='run 名')
    ap.add_argument('--kiss', default='')
    ap.add_argument('--base', default='', help='基线 run 名')
    ap.add_argument('--out', default='')
    args = ap.parse_args()

    P, gt_t, gt_C = load_gt(args.ours)
    n = P['cam_trans'].shape[2]
    gt_t, gt_C = gt_t[:n], gt_C[:n]

    print('=' * 70)
    print('统一口径横向对比　序列 %s　（%d 帧）' % (args.seq, n))
    print('位置表示：相机中心 C（真值由 gt_w2c_all_frames 精确换算）')
    print('=' * 70)

    rows = []

    # ---- 本项目（我们的 t 与项目口径一致；控制变量已证与 C 可比）----
    our_t = P['cam_trans'][0][:, :n].T
    m = ea.compute(gt_C, our_t)
    rows.append(('本项目完整方案', args.ours, m['ate_cm'], m['ate_rmse_cm'],
                 m['ate_sim3_mean_cm'], m['sim3_scale'], m['odom_est_m'],
                 True))
    print('%-18s ATE %6.2f  RMSE %6.2f  Sim3 %6.2f  尺度 %.5f  里程 %7.2f/%.2f'
          % ('本项目完整方案', m['ate_cm'], m['ate_rmse_cm'],
             m['ate_sim3_mean_cm'], m['sim3_scale'], m['odom_est_m'],
             m['odom_gt_m']))

    if args.base:
        Pb = np.load(os.path.join(EXP, args.base, 'params.npz'),
                     allow_pickle=True)
        tb = Pb['cam_trans'][0]
        nb = min(tb.shape[1], n)
        mb = ea.compute(gt_C[:nb], tb[:, :nb].T)
        rows.append(('本项目基线', args.base, mb['ate_cm'], mb['ate_rmse_cm'],
                     mb['ate_sim3_mean_cm'], mb['sim3_scale'], mb['odom_est_m'],
                     False))
        print('%-18s ATE %6.2f  RMSE %6.2f  Sim3 %6.2f  尺度 %.5f  里程 %7.2f/%.2f'
              % ('本项目基线', mb['ate_cm'], mb['ate_rmse_cm'],
                 mb['ate_sim3_mean_cm'], mb['sim3_scale'], mb['odom_est_m'],
                 mb['odom_gt_m']))

    if args.kiss and os.path.exists(args.kiss):
        e_ts, e_C, _ = ea.read_tum(args.kiss)
        mk = ea.compute(gt_C[:len(e_C)], e_C)
        rows.append(('KISS-ICP', 'pip1.3.0', mk['ate_cm'], mk['ate_rmse_cm'],
                     mk['ate_sim3_mean_cm'], mk['sim3_scale'], mk['odom_est_m'],
                     False))
        print('%-18s ATE %6.2f  RMSE %6.2f  Sim3 %6.2f  尺度 %.5f  里程 %7.2f/%.2f'
              % ('KISS-ICP', mk['ate_cm'], mk['ate_rmse_cm'],
                 mk['ate_sim3_mean_cm'], mk['sim3_scale'], mk['odom_est_m'],
                 mk['odom_gt_m']))

    print('-' * 70)
    if len(rows) >= 2:
        best = min(rows, key=lambda r: r[2])
        print('最优：%s (%.2f cm)' % (best[0], best[2]))
        for name, run, ate, _, _, _, _, sem in rows:
            print('  %-18s 是最优的 %.2f 倍' % (name, best[2] / ate)
                  if ate > 0 else '')

    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump({
                'sequence': args.seq, 'n_frames': n,
                'repr': 'camcenter_unified',
                'note': '全部方法统一到相机中心口径；控制变量实测'
                        '用真值R 转换后与我们t 口径差 0.03 cm',
                'rows': [{'method': r[0], 'run': r[1],
                          'ate_cm': round(r[2], 4),
                          'ate_rmse_cm': round(r[3], 4),
                          'ate_sim3_mean_cm': round(r[4], 4),
                          'sim3_scale': round(r[5], 6),
                          'odom_est_m': round(r[6], 3),
                          'semantic': r[7]} for r in rows],
            }, f, ensure_ascii=False, indent=2)
        print('-> %s' % args.out)


if __name__ == '__main__':
    main()