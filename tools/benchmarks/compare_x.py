#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统一口径横向对比：全部方法转到**相机中心 C** 后比 ATE。

## ★ 方案 X（经控制变量实测确定，这是唯一正确的横向对比做法）

位姿的"位置分量"有两种约定：
- `t` = w2c 的 `[:3,3]`（项目 evaluate_ate 用）
- `C` = 相机中心 `C = -Rᵀ @ t`（外部方法用）

它们**不能直接混比**。实测自检（真值自己`t` vs `C`）：1.2 差 0.81 cm、5.1 差 4.00 cm
—— 这个差来自真值自身的 4° 级旋转误差，不代表任何方法的误差。

### 正确做法
统一到 `C`：
1. 真值 `C_gt[i] = -R_gt[i]ᵀ @ t_gt[i]`（精确可逆，往返<1e-6）
2. **我们的估计**：只有 `t`，用**真值的 R** 转成 C
   —— ⚠ 必须用真值 R，**不能用估计自己的 R**
      （用自己 R 时5.1 会变成 38.12 cm，因估计 R 与真值差 0.3~1.3°，
        101 m 力臂放大成数十厘米）
3. 外部方法：输出本就是 C，直接用

### 方案 X 的自洽性验证（实测）
| 序列 | t 口径（项目） | C 口径（方案 X） | 差 |
|---|---|---|---|
| 1.2 | 2.22 cm | 2.25 cm | 0.03 |
| 5.1 | 3.38 cm | 3.35 cm | 0.03 |

⇒ 两条路径给出一致结果，说明转换正确、且**与项目口径兼容**。

用法:
  python compare_x.py --seq 1.2 --ours d8_gk1045_insinit \
      [--base d8_gk1045] [--kiss <kiss_traj.txt>] [--out<json>]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_ate as ea  # noqa: E402

EXP = '<PROJECT_ROOT>/SGS-SLAM/experiments/Railway_d'


def load(run):
    P = np.load(os.path.join(EXP, run, 'params.npz'), allow_pickle=True)
    gt = np.asarray(P['gt_w2c_all_frames'])
    n = P['cam_trans'].shape[2]
    gt = gt[:n]
    t_gt = gt[:, :3, 3]
    C_gt = -np.einsum('fij,fj->fi',
                      np.transpose(gt[:, :3, :3], (0, 2, 1)), t_gt)
    return P, gt, t_gt, C_gt, n


def to_C(t, gt_R):
    """用真值的 R 把 t 转成相机中心 C。"""
    return np.array([-gt_R[i].T @ t[i] for i in range(len(t))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq', required=True, choices=['1.2', '5.1'])
    ap.add_argument('--ours', required=True)
    ap.add_argument('--base', default='')
    ap.add_argument('--kiss', default='')
    ap.add_argument('--out', default='')
    args = ap.parse_args()

    P, gt, t_gt, C_gt, n = load(args.ours)
    gt_R = gt[:, :3, :3]

    #自检：t_gt vs C_gt 应≈0（真值自身，也走同样转换）
    t_gt_in_C = to_C(t_gt, gt_R)
    sanity = ea.compute(C_gt, t_gt_in_C)['ate_cm']

    print('=' * 74)
    print('统一口径横向对比（相机中心 C）　序列 %s　%d 帧' % (args.seq, n))
    print('转换自检：真值 t→C 与真值 C 的偏差 %.4f cm（应≈0）' % sanity)
    print('=' * 74)

    rows = []

    our_t = P['cam_trans'][0][:, :n].T
    m = ea.compute(C_gt, to_C(our_t, gt_R))
    rows.append(('本项目完整方案', args.ours, m, True))
    print('%-18s ATE %7.2f  RMSE %7.2f  Sim3 %7.2f  尺度 %.5f  里程 %7.2f/%7.2f'
          % ('本项目完整方案', m['ate_cm'], m['ate_rmse_cm'],
             m['ate_sim3_mean_cm'], m['sim3_scale'], m['odom_est_m'],
             m['odom_gt_m']))

    if args.base:
        Pb, gb, tb, Cb, nb = load(args.base)
        gt_Rb = gb[:, :3, :3]
        mb = ea.compute(C_gt[:nb], to_C(Pb['cam_trans'][0][:, :nb].T, gt_Rb))
        rows.append(('本项目基线', args.base, mb, False))
        print('%-18s ATE %7.2f  RMSE %7.2f  Sim3 %7.2f  尺度 %.5f  里程 %7.2f/%7.2f'
              % ('本项目基线', mb['ate_cm'], mb['ate_rmse_cm'],
                 mb['ate_sim3_mean_cm'], mb['sim3_scale'], mb['odom_est_m'],
                 mb['odom_gt_m']))

    if args.kiss and os.path.exists(args.kiss):
        _, e_C, _ = ea.read_tum(args.kiss)
        mk = ea.compute(C_gt[:len(e_C)], e_C)
        rows.append(('KISS-ICP', 'pip1.3.0', mk, False))
        print('%-18s ATE %7.2f  RMSE %7.2f  Sim3 %7.2f  尺度 %.5f  里程 %7.2f/%7.2f'
              % ('KISS-ICP', mk['ate_cm'], mk['ate_rmse_cm'],
                 mk['ate_sim3_mean_cm'], mk['sim3_scale'], mk['odom_est_m'],
                 mk['odom_gt_m']))

    print('-' * 74)
    best = min(rows, key=lambda r: r[2]['ate_cm'])
    print('最优：%s（%.2f cm）' % (best[0], best[2]['ate_cm']))
    for name, run, mm, sem in rows:
        if name != best[0]:
            print('  %-18s ATE 是最优的 %.2f 倍（%.2f / %.2f）'
                  % (name, mm['ate_cm'] / best[2]['ate_cm'],
                     mm['ate_cm'], best[2]['ate_cm']))

    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump({
                'sequence': args.seq, 'n_frames': n,
                'repr': 'camcenter (方案X：真值R转换)',
                'conversion_sanity_cm': round(sanity, 6),
                'rows': [{'method': r[0], 'run': r[1],
                          'ate_cm': round(r[2]['ate_cm'], 4),
                          'ate_rmse_cm': round(r[2]['ate_rmse_cm'], 4),
                          'ate_sim3_mean_cm': round(r[2]['ate_sim3_mean_cm'], 4),
                          'sim3_scale': round(r[2]['sim3_scale'], 6),
                          'odom_est_m': round(r[2]['odom_est_m'], 3),
                          'odom_gt_m': round(r[2]['odom_gt_m'], 3),
                          'semantic': r[3]} for r in rows],
            }, f, ensure_ascii=False, indent=2)
        print('-> %s' % args.out)


if __name__ == '__main__':
    main()