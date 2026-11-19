#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""标定 LiDAR→相机投影，使它与渲染深度**近处对齐**。

## 为什么必须标定（实测诊断，2026-10-08）
第一次投影后，深度误差随距离单调增长：

| LiDAR 深度段 | 误差中位 |
|---|---|
| 3–16.5 m | 8.72 m |
| 16.5–18.8 m | 9.10 m |
| 18.8–29.1 m | 11.98 m |
| 29.1–44.9 m | 16.80 m |
| 44.9–120 m | 36.44 m |

⇒ 近处也差 8~9 m，**不是"远场离群"，而是整体的系统偏差**。
根因候选：俯仰角 PITCH 估错 / Z_OFF 不对 / 轴向约定有符号错。

## 标定方法（用几何约束，不靠试）
扫描 (PITCH, Z_OFF) 两个自由度，目标函数 =
**近处（< 25m）深度中位误差最小**（近处受遮挡影响小，最能反映标定是否对）。

用法:
  python calib_proj.py --frame 29          # 扫描并输出最优参数
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lidar_ref_depth as L  # noqa: E402

R = '<PROJECT_ROOT>/benchmarks/results'


def project_with(xyz, pitch, zoff):
    x, y, z = xyz[:, 0], xyz[:, 1], xyz  [:, 2]
    cp, sp = np.cos(-pitch), np.sin(-pitch)
    x2 = x * cp + z * sp
    z2 = -x * sp + z * cp
    zc = x2
    xc = -y
    yc = -(z2 + zoff)
    ok = zc > 0.5
    return (L.FX * xc[ok] / zc[ok] + L.CX,
            L.FY * yc[ok] / zc[ok] + L.CY, zc[ok])


def make_depth(xyz, u, v, z, H, W):
    """z-buffer 投影成深度图。

    ⚠ 坑：`np.minimum.at` 对 **np.nan初始化的数组无效**
      （nan 参与 min 比较结果仍是 nan）⇒ 先用 +inf 填充再转 nan。
    """
    m = (z >= 3.0) & (z <= 120.0)
    img = np.full((H, W), np.inf)
    np.minimum.at(img, (np.clip(np.round(v[m]).astype(int), 0, H - 1),
                        np.clip(np.round(u[m]).astype(int), 0, W - 1)), z[m])
    img[~np.isfinite(img)] = np.nan
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frame', type=int, default=29)
    args = ap.parse_args()
    fi = args.frame

    files = sorted(f for f in os.listdir(L.LIV) if f.endswith('.pcd'))
    pcd = L.read_pcd(os.path.join(L.LIV, files[fi]))
    xyz = pcd[:, :3]
    rd = np.load(os.path.join(R, 'frames12', 'renddepth_%04d.npy' % fi))
    Hr, Wr = rd.shape

    def score(pitch, zoff):
        u, v, z = project_with(xyz, pitch, zoff)
        ld = make_depth(xyz, u, v, z, L.H, L.W)
        ld_s = cv2.resize(ld, (Wr, Hr), interpolation=cv2.INTER_NEAREST)
        both = np.isfinite(ld_s) & (rd > 0)
        if both.sum() < 200:
            return None
        rl = ld_s[both]
        rr = rd[both]
        # ⚠ LiDAR 近处点很稀（<25m 常不足 100 个），
        #   所以用「深度比值」当判据 —— 它对稀疏更稳健：
        #   纯标定错误会体现为比值整体偏离 1。
        r = rr / np.maximum(rl, 1e-6)
        med_ratio = float(np.median(r))
        med_err = float(np.median(np.abs(rr - rl)))
        # 目标函数：比值偏离 1 的程度（越小越好）
        obj = abs(med_ratio - 1.0)
        return (obj, med_ratio, med_err, int(both.sum()))

    print('扫描 (俯仰角 pitch, 高度偏移 zoff)')
    print('目标函数：深度比值(渲染/LiDAR) 的中位数偏离 1 的程度，越小越好')
    print('%8s %8s %12s %14s %10s' % ('pitch°', 'zoff',
                                      '目标(偏离1)', '比值中位数', '共同像素'))
    print('-' * 56)
    best = None
    for pitch_d in np.arange(0.0, 7.01, 1.0):
        for zoff in np.arange(-4.0, 0.51, 0.5):
            r = score(np.deg2rad(pitch_d), float(zoff))
            if r is None:
                continue
            obj, ratio, merr, n = r
            mark = ''
            if best is None or obj < best[0]:
                best = (obj, pitch_d, float(zoff), ratio)
                mark = '  <= 当前最优'
            print('%8.1f %8.2f %12.3f %14.3f %10d%s'
                  % (pitch_d, zoff, obj, ratio, n, mark))
    print('-' * 56)
    print('★ 最优: pitch=%.1f°  zoff=%.2f  比值中位 %.3f（越接近1越好）'
          % (best[1], best[2], best[3]))
    print('  ⇒ 按此参数重建 LiDAR 投影后，深度比值应接近 1')


if __name__ == '__main__':
    main()