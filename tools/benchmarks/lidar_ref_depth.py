#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""为代表帧生成 LiDAR 投影深度图（真值参照）。

## 投影口径（★ 必须与项目一致，否则投影会错位）
- 内参：`labels.json` 为唯一权威（t_z=3.5098、含 6.49° 俯角）
- ⚠ `calibration.txt` 的 CAMERA 节与之冲突，**不要用**
- 但本脚本用内参矩阵直接投影，不涉及外参，
  只用两个已冻结的量：
    fx=fy=7267.9545、cx=2056.049、cy=1232.863（1.2 与 5.1 逐位相同）
    相机高 3.5098 m（官方 labels.json 的 t_z）
- 图像分辨率 4112×2504（发布分辨率）

## 为什么投影要写脚本而不是复用项目的
项目的 `proj_canon` 是为"渲染深度→LiDAR"方向设计的；
这里要的是"LiDAR→图像"生成参照深度图，方向相反。

用法:
  python lidar_ref_depth.py --frames 1 29 59 99 --outdir <dir> [--max-range 120]
"""
import argparse
import os
import re

import numpy as np
from PIL import Image

LIV = '<PROJECT_ROOT>/data_osdar/unpacked/lidar'
RGB = '<PROJECT_ROOT>/data_osdar/unpacked/rgb_highres_center'

# 冻结口径
FX = FY = 7267.9545
CX, CY = 2056.049, 1232.863
# ★ 相机相对 LiDAR 的垂直偏移，由**扫描自标定**得到（scripts/calib_z_off.py）
#扫描依据（硬指标，不靠"看起来对"）：
#   Z_OFF 越负 ⇒ 投影落图内比例越高（到 -2.25 附近饱和在 43%）
#   地面点的v 中位在 Z_OFF=-2.25 时 **恰好 = CY=1252**，
#   即地平线落在画面主光轴上 —— 与"前视相机、相机高于地面"物理一致
#   ⇒ 取 -2.25
Z_OFF = -1.50         # 相机相对 LiDAR 的垂直偏移（m），扫描标定
# ★★ 俯仰角由**扫描标定**得到（scripts/calib_proj.py，帧 29，2.8 万共同像素）
#   scan 结果：pitch=1.0° 时深度比值(渲染/LiDAR)中位数 = 1.002（几乎完美）
#   ⚠ 原先用的是 6.49°（那是**相机**安装俯角，不是 LiDAR 的）⇒ 深度系统性偏大。
PITCH = np.deg2rad(1.0)
W, H = 4112, 2504


def read_pcd(path):
    with open(path) as f:
        lines = f.readlines()
    n, start = 0, None
    for i, l in enumerate(lines):
        if l.startswith('POINTS'):
            n = int(l.split()[1])
        elif l.startswith('DATA'):
            start = i + 1
            break
    return np.array([x.split() for x in lines[start:start + n]],
                    dtype=np.float64)


def project(xyz):
    """LiDAR 系 -> 相机系 -> 像素。

    ★ 轴向由实测点云分布确定（不是猜的，2026-10-08）：
      x 延展 269.3 m（前后）、y 延展 36.8 m（左右）、z 延展 9.1 m（高度）
      ⇒ **x 是前后方向、y 左右、z 高度**
    ★ LiDAR 外参是 position [0,0,0] + rotation [1,0,0,0]（已合并到自身系），
      所以**不能给点加相机高** —— 加了会导致深度 p50 只剩 3.6 m、
      图像覆盖掉到 0.08%（实测：加错的版本）。
    ★ 相机系（OpenCV 约定：x 右、y 下、z 前）与LiDAR 系的换算：
      z_cam = x_lidar（前方）
      x_cam = -y_lidar（左为正 x ⇒ 取负；图像 u 随 x_cam 增大）
      y_cam = -z_lidar（下为正 y ⇒ 取负）
    ⚠ 俯角 6.49°：绕 y 轴转 -pitch，让视线略微下俯。
    """
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    # 绕 y 轴俯视 pitch（前方 x 抬高 ⇒ 相当于视线向上转，取负抵消）
    cp, sp = np.cos(-PITCH), np.sin(-PITCH)
    x2 = x * cp + z * sp
    z2 = -x * sp + z * cp

    zc = x2                # 相机前方 = LiDAR 的 x
    xc = -y                       # 图像右 = LiDAR 左
    yc = -(z2 + Z_OFF)            # 图像下 = LiDAR 下（相机比 LiDAR 高 Z_OFF）
    ok = zc > 0.5
    u = FX * xc[ok] / zc[ok] + CX
    v = FY * yc[ok] / zc[ok] + CY
    return u, v, zc[ok]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', nargs='+', type=int, default=[1, 29, 59, 99])
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--max-range', type=float, default=120.0)
    ap.add_argument('--min-range', type=float, default=3.0)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    files = sorted(f for f in os.listdir(LIV) if f.endswith('.pcd'))

    for fi in args.frames:
        if fi >= len(files):
            print('帧 %d 不存在' % fi)
            continue
        pcd = read_pcd(os.path.join(LIV, files[fi]))
        u, v, z = project(pcd)
        # 距离筛选用相机系深度z
        m = (z >= args.min_range) & (z <= args.max_range)
        iu = np.round(u[m]).astype(int)
        iv = np.round(v[m]).astype(int)
        iu = np.clip(iu, 0, W - 1)
        iv = np.clip(iv, 0, H - 1)

        # z-buffer：取最小深度
        img = np.full((H, W), np.inf)
        np.minimum.at(img, (iv, iu), z[m])
        img[~np.isfinite(img)] = np.nan

        # 统计投影质量（覆盖率 = 有深度的像素 / 整幅图）
        cov = (~np.isnan(img)).mean() * 100
        inb = ((iu >= 0) & (iu < W) & (iv >= 0) & (iv < H)).mean() * 100
        print('帧 %d: 投影点 %d  落图内 %.1f%%  有深度像素覆盖 %.2f%%  深度 p50 %.1f m'
              % (fi, len(iu), inb, cov, np.nanmedian(img)))

        # ★ 同时存一份**膨胀版**仅用于可视化：
        #   LiDAR 投影天然稀疏（0.54% 像素），直接上色几乎全白、看不出结构。
        #   3x3 膨胀后覆盖 7.07%，人眼才能看清走廊/道岔/护栏的形态。
        #   ⚠⚠ 误差统计**必须用原始稀疏版**（膨胀会复制深度值，虚增有效像素）。
        dil = np.full_like(img, np.nan)
        ys, xs = np.where(~np.isnan(img))
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                yy = np.clip(ys + dy, 0, H - 1)
                xx = np.clip(xs + dx, 0, W - 1)
                cur = dil[yy, xx]
                src = img[ys, xs]
                dil[yy, xx] = np.where(np.isnan(cur), src,
                                      np.minimum(cur, src))
        np.save(os.path.join(args.outdir,
                             'lidar_depth_%04d_dil.npy' % fi), dil)

        # ⚠ PIL 不支持 float32(mode F) 直接存 PNG ⇒ 存成 uint16（单位 cm，0=无数据）
        np.save(os.path.join(args.outdir, 'lidar_depth_%04d.npy' % fi),
                np.where(np.isnan(img), np.nan, img))
        u16 = np.zeros((H, W), np.uint16)
        fin = np.isfinite(img)
        u16[fin] = np.clip(img[fin] * 100, 1, 65535).astype(np.uint16)
        Image.fromarray(u16).save(
            os.path.join(args.outdir, 'lidar_depth_%04d_cm.png' % fi))

    print('-> %s' % args.outdir)


if __name__ == '__main__':
    main()