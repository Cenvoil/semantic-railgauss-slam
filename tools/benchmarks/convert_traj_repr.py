#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把外部方法输出的**相机中心 C** 轨迹转成项目口径的 **w2c 的 t**。

## 为什么必须有这一步（实测踩坑记录）
- 项目 `evaluate_ate` 用的是 w2c 矩阵的 `[:3,3]`，记为 `t`
- 外部方法（KISS-ICP / FAST-LIVO / ORB-SLAM）输出的 T_wc 平移是**相机中心 C**
- `t` 与 `C` 差一个**逐帧变化的旋转**：`t = R_w2c @ (-C)`
  （不是简单加负号！）

⚠⚠ **`R_w2c` 必须取自 `gt_w2c_all_frames`，不能用 `traj.txt` 的 R**。
实测两者不一致：用 traj 的 R 换算会让 5.1 的 ATE 从 3.38 cm 变成 38.88 cm。

## 时间戳对齐
外部方法的轨迹与真值可能不完全同频（KISS-ICP 逐帧、ORB-SLAM3 只给关键帧），
所以先用最近邻匹配到真值帧，再用**真值该帧的 R** 做转换。

用法:
  python convert_traj_repr.py --est <外部方法TUM> --gt-npz <params.npz> \
      --out <转换后TUM> [--max-dt 0.02]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_ate import read_tum, R_to_quat_tum  # noqa: E402,F401


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--est', required=True)
    ap.add_argument('--gt-npz', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-dt', type=float, default=0.02)
    args = ap.parse_args()

    P = np.load(args.gt_npz, allow_pickle=True)
    gt = np.asarray(P['gt_w2c_all_frames'])      # (F,4,4) w2c
    F = len(gt)
    gt_t = gt[:, :3, 3]                          # 项目口径的 t
    gt_C = -np.einsum('fij,fj->fi', np.transpose(gt[:, :3, :3], (0, 2, 1)),
                      gt_t)                      # 相机中心 C

    e_ts, e_C, _ = read_tum(args.est)
    print('外部轨迹 %d 帧，真值 %d 帧' % (len(e_C), F))

    # 外部方法的时间戳就是采集时间，直接按索引对齐前先核对尺度
    if len(e_C) != F:
        print('⚠ 帧数不同，按最近邻匹配到真值帧')
        idx = np.clip(np.searchsorted(e_ts, np.arange(F) * 0.1 + e_ts[0]), 0,
                      len(e_C) - 1)
    else:
        idx = np.arange(F)

    out = np.zeros((F, 3))
    for i in range(F):
        out[i] = gt[i][:3, :3] @ (-e_C[idx[i]])   # t = R_w2c @ (-C)

    # 里程自检
    seg = np.linalg.norm(np.diff(out, axis=0), axis=1)
    seg_gt = np.linalg.norm(np.diff(gt_t, axis=0), axis=1)
    print('转换后估计里程 %.2f m  真值(t)里程 %.2f m'
          % (seg.sum(), seg_gt.sum()))
    print('  | 转换前(相机中心)里程 %.2f m'
          % np.linalg.norm(np.diff(e_C[idx], axis=0), axis=1).sum())
    print('  | 真值(相机中心)里程   %.2f m' % seg_gt.sum())

    with open(args.out, 'w') as f:
        for i in range(F):
            q = R_to_quat_tum(gt[i][:3, :3].T)
            f.write('%.9f %.6f %.6f %.6f %.9f %.9f %.9f %.9f\n'
                    % (i * 0.1 + e_ts[0], out[i][0], out[i][1], out[i][2],
                       q[0], q[1], q[2], q[3]))
    print('-> %s' % args.out)


if __name__ == '__main__':
    main()