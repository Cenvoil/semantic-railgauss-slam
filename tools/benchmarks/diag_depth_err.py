#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诊断深度误差的结构：整体偏移 还是 随机误差？

## 为什么必须分清
平均误差 20.8 m 可能有两种完全不同的成因：
- **整体偏移**（systematic offset）：LiDAR 投影的 Z_OFF 没标准 ⇒ 误差集中在某一段
- **随机误差**（noise）：投影对了，是渲染/噪声 ⇒ 误差散布

两者的处置完全不同：前者改标定，后者改方法。
判据：误差的**中位数 vs 均值**、**分布形态**、**与深度的相关性**。

用法: python diag_depth_err.py --frame 29
"""
import argparse
import os

import cv2
import numpy as np

R = '<PROJECT_ROOT>/benchmarks/results'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frame', type=int, default=29)
    ap.add_argument('--zoff', type=float, default=None,
                    help='若给，则用该 Z_OFF 重算 LiDAR 投影')
    args = ap.parse_args()
    fi = args.frame

    if args.zoff is not None:
        import sys
        sys.path.insert(0, '<PROJECT_ROOT>/benchmarks/scripts')
        import lidar_ref_depth as L
        L.Z_OFF = args.zoff
        files = sorted(f for f in os.listdir(L.LIV) if f.endswith('.pcd'))
        pcd = L.read_pcd(os.path.join(L.LIV, files[fi]))
        u, v, z = L.project(pcd)
        m = (z >= 3.0) & (z <= 120.0)
        iu = np.round(u[m]).astype(int)
        iv = np.round(v[m]).astype(int)
        img = np.full((L.H, L.W), np.nan)
        np.minimum.at(img, (np.clip(iv, 0, L.H - 1),
                            np.clip(iu, 0, L.W - 1)), z[m])
        ld = img
        print('Z_OFF = %.2f 重算投影' % args.zoff)
    else:
        ld = np.load(os.path.join(R, 'frames12_lidar',
                                  'lidar_depth_%04d.npy' % fi))

    rd = np.load(os.path.join(R, 'frames12', 'renddepth_%04d.npy' % fi))
    ld_s = cv2.resize(ld, (rd.shape[1], rd.shape[0]),
                      interpolation=cv2.INTER_NEAREST)
    both = np.isfinite(ld_s) & (rd > 0)
    print('共同有效像素 %d (%.2f%%)' % (both.sum(), 100 * both.mean()))

    rl = ld_s[both]
    rr = rd[both]
    err = rr - rl                     # 正=渲染更深
    print()
    print('%-16s %10s %10s %10s' % ('', 'p10', 'p50', 'p90'))
    print('%-16s %10.2f %10.2f %10.2f' % ('LiDAR 深度', *np.percentile(rl, [10, 50, 90])))
    print('%-16s %10.2f %10.2f %10.2f' % ('渲染深度', *np.percentile(rr, [10, 50, 90])))
    print('%-16s %10.2f %10.2f %10.2f' % ('误差(渲-激)',
                                             *np.percentile(err, [10, 50, 90])))
    print()
    print('误差均值 %.2f  中位 %.2f  |均值-中位| %.2f'
          % (err.mean(), np.median(err), abs(err.mean() - np.median(err))))
    print('⇒ %s'
          % ('整体偏移为主（均值≈中位）⇒ 标定问题'
             if abs(err.mean() - np.median(err)) < abs(err.mean()) * 0.3
             else '分布有长尾⇒ 可能是离群点'))
    # 比例尺度：渲染/LiDAR
    r = rr / np.maximum(rl, 1e-6)
    print()
    print('深度比值 渲染/LiDAR: p10 %.3f p50 %.3f p90 %.3f'
          % tuple(np.percentile(r, [10, 50, 90])))
    print('⇒ %s'
          % ('纯尺度差异（比值集中）⇒ LiDAR 投影的深度尺度不对'
             if np.percentile(r, 90) / max(1e-6, np.percentile(r, 10)) < 1.6
             else '不是纯尺度问题'))

    # 分深度段看
    print()
    print('按 LiDAR 深度分段（看是否近处准、远处差）')
    qs = np.percentile(rl, [0, 20, 40, 60, 80, 100])
    for a, b in zip(qs[:-1], qs[1:]):
        m2 = (rl >= a) & (rl < b)
        if m2.sum() > 50:
            print('  LiDAR %6.1f~%6.1f m: 误差均值 %7.2f  中位 %7.2f  n=%d'
                  % (a, b, err[m2].mean(), np.median(err[m2]), m2.sum()))


if __name__ == '__main__':
    main()