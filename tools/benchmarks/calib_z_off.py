#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""扫描 LiDAR->相机的垂直偏移 Z_OFF，用「投影落图内比例」当判据自标定。

## 为什么必须自标定
LiDAR 是 6 台设备合并的、合并后 z 基准未知；
而 `calibration.txt` 的 LIDAR 外参是单位变换（position [0,0,0]），
没有给出与相机的相对高度。⇒ 只能从数据反推。

## 判据（不是"看起来对"，是硬指标）
地面点必须落在图像**下半部**（相机前视、轨顶面在地面上方 2.15 m），
所以：
1. 投影落图内比例要高（> 40%）
2. 地面点的 v 应该 > CY（1252），因为相机高于地面

用法: python calib_z_off.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lidar_ref_depth as L  # noqa: E402


def evaluate(off):
    L.Z_OFF = off
    files = sorted(f for f in os.listdir(L.LIV) if f.endswith('.pcd'))
    pcd = L.read_pcd(os.path.join(L.LIV, files[0]))
    xyz = pcd[:, :3]
    u, v, z = L.project(pcd)
    m = (z > 3) & (z < 120)
    iu = np.round(u[m]).astype(int)
    iv = np.round(v[m]).astype(int)
    inb = ((iu >= 0) & (iu < L.W) & (iv >= 0) & (iv < L.H))
    frac = 100 * inb.mean()

    # 地面点：z接近最低的那批（LiDAR 系）
    zmin = np.percentile(xyz[:, 2], 1)
    gm = xyz[:, 2] < zmin + 0.15
    ug, vg, zg = L.project(xyz[gm])
    mg = (zg > 5) & (zg < 60)
    vg = np.round(vg[mg]).astype(int)
    ok = (vg >= 0) & (vg < L.H)
    # 地面点 v 的中位数应该在 CY 附近或以下（地平线在中部，脚下在下方）
    vmed = np.median(vg[ok]) if ok.sum() else -1
    return frac, vmed, int(ok.sum())


def main():
    print('图 %dx%d  CY=%.0f（地面点应大致在 CY 或以下，脚下更靠下）'
          % (L.W, L.H, L.CY))
    print()
    print('%8s %12s %14s %12s' % ('Z_OFF', '落图内%', '地面点v中位', '地面点数'))
    print('-' * 52)
    best = None
    for off in np.arange(-3.0, 4.01, 0.25):
        frac, vmed, ng = evaluate(float(off))
        flag = ''
        # 判据：地面点 v 中位在 900~1700 之间（图像中下部）且落图内 >40%
        if 700 <= vmed <= 1900 and frac > 40:
            flag = '  <= 候选'
            if best is None:
                best = (off, frac, vmed)
        print('%8.2f %11.1f%% %14.0f %12d%s'
              % (off, frac, vmed, ng, flag))
    print()
    if best:
        print('★ 最佳Z_OFF = %.2f（落图内 %.1f%%，地面点 v 中位 %.0f）'
              % best)
    else:
        print('没有候选，需放宽判据或检查轴向约定')


if __name__ == '__main__':
    main()