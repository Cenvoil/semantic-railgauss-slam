#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""代表帧对比图（以 1.2 为例，帧 1/29/59/99）。

## 每帧四个 panel（按"不加说明就能看懂"设计）
  ① 输入 RGB（真彩色）
  ② 本项目渲染 RGB（高斯重建结果）
  ③ 官方 LiDAR 投影深度（**物理测量的真值参照**）
  ④ 深度误差 |渲染深度 − LiDAR 深度|（蓝=小，青绿=中，红=大）

## 为什么第③ panel 用 LiDAR 而不是官方渲染
LiDAR 是**物理测量**，比"和官方渲染比"更硬 ——
渲染图本身就是被评估的对象，拿它当参照是循环论证。

## 为什么误差图用 LiDAR 而不是 GT 深度图
项目的 GT 深度图本身就是单目深度+LiDAR 锚定的产物，
与被评估的渲染同源 ⇒ 用它当参照也是循环论证。
★ LiDAR 投影是独立测量，唯一无偏。

用法:
  python frame_compare_12.py --frames 1 29 59 99 --outdir <dir>
"""
import argparse
import os
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

RGB_DIR = '<PROJECT_ROOT>/data_osdar/unpacked/rgb_highres_center'
RENDER = '<PROJECT_ROOT>/benchmarks/results/frames12'
LIDAR = '<PROJECT_ROOT>/benchmarks/results/frames12_lidar'

NAVY = (27, 58, 107)
GRAY = (90, 100, 116)
ORANGE = (194, 94, 0)
WHITE = (255, 255, 255)


def get_font(sz):
    r = subprocess.run(['fc-list', ':lang=zh', 'file'], capture_output=True,
                       text=True)
    for line in r.stdout.split('\n'):
        if 'NotoSansCJK' in line or 'wqy' in line.lower():
            try:
                return ImageFont.truetype(line.split(':')[0], sz)
            except Exception:
                pass
    return ImageFont.load_default()


def ramp(x, reverse=False):
    """分段色标：0=深蓝 → 0.5=青绿 → 1=红。"""
    x = np.clip(x, 0, 1)
    if reverse:
        x = 1 - x
    r = np.clip(1.6 * x - 0.35, 0, 1)
    g = np.clip(1.5 - np.abs(3.2 * x - 1.7), 0, 1)
    b = np.clip(1.4 - np.abs(3.4 * x - 0.4), 0, 1)
    return np.stack([r, g, b], -1)


def colorize(d, lo, hi, reverse=False):
    m = np.isfinite(d)
    out = np.ones(d.shape + (3,), np.uint8) * 255
    if m.sum() == 0:
        return Image.fromarray(out)
    x = (d[m] - lo) / max(1e-6, hi - lo)
    out[m] = (ramp(x, reverse) * 255).astype(np.uint8)
    return Image.fromarray(out)


def resize_h(img, h):
    sc = h / img.height
    return img.resize((max(1, int(img.width * sc)), h), Image.LANCZOS)


def resize_w(img, w):
    sc = w / img.width
    return img.resize((w, max(1, int(img.height * sc))), Image.LANCZOS)


def panel(img, title, pw, ph, fnt, note=None):
    """缩放并加标题栏（可选角标）。"""
    im = resize_w(img, pw)
    if im.height > ph:
        im = resize_h(im, ph)
    out = Image.new('RGB', (pw, ph + 46), WHITE)
    out.paste(im, ((pw - im.width) // 2, 46 + (ph - im.height) // 2))
    d = ImageDraw.Draw(out)
    d.rectangle([0, 0, pw, 45], fill=NAVY)
    d.text((10, 9), title, fill=WHITE, font=fnt)
    d.rectangle([0, 46, pw, ph + 46], outline=(198, 208, 220))
    if note:
        d.rectangle([0, 46, pw, 46 + 26], fill=(255, 255, 255))
        d.text((10, 50), note, fill=GRAY, font=fnt)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', nargs='+', type=int, default=[1, 29, 59, 99])
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--pw', type=int, default=860)
    # cv2 在panel 分支里用到，这里提前导入
    import cv2
    ap.add_argument('--ph', type=int, default=520)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    FNT = get_font(25)
    FNT_S = get_font(21)
    stats = []

    for fi in args.frames:
        # ⚠ OSDaR23 的 RGB 文件名是 `029_<timestamp>.png`（3 位帧号）
        rp = os.path.join(RGB_DIR, '%03d_' % fi +
                          [f for f in os.listdir(RGB_DIR)
                           if f.startswith('%03d_' % fi)][0].split('_', 1)[1])
        # 上面拼回完整名，更稳的写法：
        rp = os.path.join(RGB_DIR, [f for f in os.listdir(RGB_DIR)
                                    if f.startswith('%03d_' % fi)][0])
        gp = os.path.join(RENDER, 'gs_%04d.png' % fi)
        lp = os.path.join(LIDAR, 'lidar_depth_%04d.npy' % fi)
        if not (os.path.exists(rp) and os.path.exists(gp)):
            print('帧 %d: 缺输入或渲染' % fi)
            continue

        rgb = Image.open(rp).convert('RGB')
        ren = Image.open(gp).convert('RGB')

        # LiDAR 投影深度是 4112x2504，渲染图是 3084x1878 —— 下采样对齐
        ld = np.load(lp) if os.path.exists(lp) else None
        rd_path = os.path.join(RENDER, 'renddepth_%04d.npy' % fi)
        rd = np.load(rd_path) if os.path.exists(rd_path) else None
        if ld is not None and rd is not None:
            import cv2
            ld_s = cv2.resize(ld, (rd.shape[1], rd.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
        else:
            ld_s = rd_s = None

        panels = [
            panel(rgb, '① 输入 RGB（第 %d 帧）' % fi, args.pw, args.ph, FNT),
            panel(ren, '② 本项目渲染 RGB（5.17M 高斯）', args.pw, args.ph, FNT),
        ]

        # ③ LiDAR 深度（用**膨胀版**显示才看得清，统计仍用原始稀疏版）
        ld_dil_p = os.path.join(LIDAR, 'lidar_depth_%04d_dil.npy' % fi)
        ld_show = (np.load(ld_dil_p) if os.path.exists(ld_dil_p) else ld_s)
        if ld_show is not None:
            ld_show_s = cv2.resize(ld_show, (args.pw, args.ph),
                                   interpolation=cv2.INTER_NEAREST)
            v = ld_show[np.isfinite(ld_show)]
            if v.size:
                lo, hi = np.percentile(v, 2), np.percentile(v, 98)
                panels.append(panel(
                    colorize(ld_show_s, lo, hi),
                    '③ 官方 LiDAR 投影深度（真值参照）',
                    args.pw, args.ph, FNT,
                    '近 %.0fm ← → 远 %.0fm（物理测量）' % (lo, hi)))
        else:
            print('帧 %d: 无 LiDAR 深度' % fi)

        # ④ 深度误差
        if ld_s is not None and rd is not None:
            both = np.isfinite(ld_s) & (rd > 0)
            if both.sum() > 100:
                diff = np.where(both, np.abs(rd - ld_s), np.nan)
                dv = diff[both]
                #★ 统计只用「近处 + 中距离」（3~60 m）：
                #   远端（>60m）LiDAR 几乎打不到（点稀疏），
                #   少量错误配对会贡献巨大误差，把 p95 拉到 50m+，
                #   反而掩盖了近处的真实质量。
                good = both & (ld_s >= 3.0) & (ld_s <= 60.0)
                gd = diff[good]
                mae = float(np.median(gd))
                rmse = float(np.sqrt((gd ** 2).mean()))
                p95 = float(np.percentile(gd, 95))
                hi = max(1.0, float(np.percentile(gd, 95)))
                stats.append((fi, mae, rmse, p95, 100.0 * good.mean()))
                # 只在 good 区域上色，其余留白（更易读）
                dshow = np.where(good, diff, np.nan)
                panels.append(panel(
                    colorize(dshow, 0, hi, reverse=True),
                    '④ 深度误差 |渲染 − LiDAR|（3~60m）',
                    args.pw, args.ph, FNT,
                    '蓝=接近 红=大（上限 p95=%.0fm）  中位 %.2fm'
                    % (hi, mae)))
        else:
            print('帧 %d: 无渲染深度可比' % fi)

        W = args.pw * len(panels)
        sheet = Image.new('RGB', (W, args.ph + 46), WHITE)
        for k, im in enumerate(panels):
            sheet.paste(im, (k * args.pw, 0))
        out = os.path.join(args.outdir, 'frame_%03d_compare.png' % fi)
        sheet.save(out)
        print('帧 %d -> %s  (%d panel, %dx%d)'
              % (fi, os.path.basename(out), len(panels), sheet.width,
                 sheet.height))

    if stats:
        print('\n深度误差统计（vs 官方 LiDAR 投影，3~60 m 区间，单位 m）')
        print('%6s %10s %10s %10s %12s' % ('帧', '中位误差', 'RMSE', 'p95',
                                          '有效像素'))
        print('-' * 52)
        for fi, mae, rmse, p95, cov in stats:
            print('%6d %9.2f %10.2f %10.2f %11.2f%%'
                  % (fi, mae, rmse, p95, cov))
        mm = np.mean([s[1] for s in stats])
        rr = np.mean([s[2] for s in stats])
        print('-' * 52)
        print('%6s %9.2f %10.2f' % ('平均', mm, rr))


if __name__ == '__main__':
    main()