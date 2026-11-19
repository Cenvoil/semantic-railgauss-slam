#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 params.npz 内置的 gt_w2c_all_frames 生成**相机中心**真值 TUM。

## 为什么不用 traj.txt 换算（实测踩过的坑）
最初用 `C = -R^T @ t`，其中 R、t 都取自 `traj.txt`，
结果 5.1 上我们算出 **38.88 cm**（而正确值是 3.38 cm），完全失真。

追查发现：`traj.txt` 的旋转与 `gt_w2c_all_frames` 的旋转**不一致**
（往返误差 10.7 m），所以用 traj 的 R 去换算 traj 的 t 得到的"相机中心"是错的。

⇒ **正确做法**：真值必须来自 `gt_w2c_all_frames`（训练/评估时用的那份），
用**它自己的 R** 换算出相机中心。C = -R_i^T @ t_i，逐帧精确可逆。

## 口径
主口径（默认，对齐项目的 evaluate_ate）：输出 **w2c 的 t**
统一口径（`--repr camcenter`）：输出 **相机中心 C**，外部方法用它
"""
import argparse
import os

import numpy as np


def R_to_quat_tum(R_cw):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='<PROJECT_ROOT>/SGS-SLAM')
    ap.add_argument('--run', required=True)
    ap.add_argument('--unpacked', required=True, help='取 RGB 文件名里的真实时间戳')
    ap.add_argument('--out', required=True)
    ap.add_argument('--repr', default='w2c_t', choices=['w2c_t', 'camcenter'])
    args = ap.parse_args()

    P = np.load(os.path.join(args.root, 'experiments', 'Railway_d',
                             args.run, 'params.npz'), allow_pickle=True)
    gt = np.asarray(P['gt_w2c_all_frames'])          # (F,4,4)
    n = len(gt)

    # 真实时间戳
    import re
    cam_dir = os.path.join(args.unpacked, 'rgb_highres_center')
    files = sorted(f for f in os.listdir(cam_dir) if f.endswith('.png'))
    ts = np.array([float(re.search(r'(\d+\.\d+)', f).group(1))
                   for f in files[:n]])

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        for i in range(n):
            R = gt[i][:3, :3]
            t = gt[i][:3, 3]
            if args.repr == 'camcenter':
                pos = -R.T @ t              # 逐帧精确可逆
            else:
                pos = t
            q = R_to_quat_tum(R.T)
            f.write('%.9f %.6f %.6f %.6f %.9f %.9f %.9f %.9f\n'
                    % (ts[i], pos[0], pos[1], pos[2], q[0], q[1], q[2], q[3]))

    # 自检
    pos_all = np.array([(-gt[i][:3, :3].T @ gt[i][:3, 3])
                        if args.repr == 'camcenter' else gt[i][:3, 3]
                        for i in range(n)])
    seg = np.linalg.norm(np.diff(pos_all, axis=0), axis=1)
    print('%s  repr=%s  %d 帧  里程 %.2f m  首末直线 %.2f m  每帧中位 %.3f m'
          % (os.path.basename(args.out), args.repr, n, seg.sum(),
             np.linalg.norm(pos_all[-1] - pos_all[0]), np.median(seg)))

    # 往返自检：camcenter 再转回 w2c t 应精确还原
    if args.repr == 'camcenter':
        back = np.array([-gt[i][:3, :3].T @ pos_all[i] for i in range(n)])
        err = np.abs(back - np.array([gt[i][:3, 3] for i in range(n)])).max()
        print('  往返自检（C -> t）最大误差 %.2e m  %s'
              % (err, 'OK' if err < 1e-6 else '!! 异常'))


if __name__ == '__main__':
    main()