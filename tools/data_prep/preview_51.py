# -*- coding: utf-8 -*-
"""5.1 交付目检图生成：4 帧代表帧 × (RGB / 深度 / 语义 / 深度叠加 / 语义叠加)。

代表帧选取（不拍脑袋，按画面内容分层挑）：
  先算每帧的两个量 —— 语义覆盖率（信息量）与 LiDAR 锚像素数（几何信息量），
  然后取"覆盖率接近中位、但彼此分散"的 4 帧，避免 4 张都挑到最空/最满的帧。
  具体：按覆盖率排序后取 [5%, 35%, 65%, 95%] 分位的帧。

深度可视化（skill §6.2 的坑）：
  * 线性 45 m 标尺在铁路场景会大面积饱和（实测 28.5%），本脚本用**对数标尺**
    d0=8 m（≈p5）、dmax=120 m（= CAP），理由：5.1 深度 p50≈20 m、p90≈45 m，
    线性标尺会让 45 m 以上全部纯红、完全无信息。
  * 同时输出**带色标(colorbar)**的版本，目检时能读出绝对米数。
  * 关键：饱和像素占比应接近"GT 中超过 dmax 的比例"，作为自检。

语义配色：**info_semantic.json 的 color 是 BGR**，用前必须转 RGB（skill §18.2）。

用法: python preview_51.py <out_png_dir> [frames...]
"""
import glob
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

DS = '<PROJECT_ROOT>/SGS-SLAM/data/output_1.0_51'
OUT = sys.argv[1] if len(sys.argv) > 1 else '<PROJECT_ROOT>/tmp/d51/preview'
FR = [int(x) for x in sys.argv[2:]] if len(sys.argv) > 2 else None

info = json.load(open(os.path.join(DS, 'info_semantic.json')))
# ⚠ BGR -> RGB
lut = np.zeros((256, 3), np.uint8)
id2name = {}
for c in info['classes']:
    b, g, r = c['color'][:3]
    lut[c['id']] = np.array([r, g, b], np.uint8)
    id2name[c['id']] = c['name']
legend = [(c['id'], c['name'], tuple(int(x) for x in lut[c['id']])) for c in info['classes']]

ids = sorted(glob.glob(os.path.join(DS, 'semantic_ids', '*.png')))
assert len(ids) == 100
cov = []
for p in ids:
    a = np.array(Image.open(p))
    cov.append(float((a > 0).mean()))
cov = np.array(cov)
print('语义覆盖率: min=%.4f%% p25=%.4f%% 中位=%.4f%% p75=%.4f%% max=%.4f%%'
      % (100 * cov.min(), 100 * np.percentile(cov, 25), 100 * np.median(cov),
         100 * np.percentile(cov, 75), 100 * cov.max()))

if FR is None:
    order = np.argsort(cov)
    FR = sorted(int(order[int(len(order) * q)]) for q in (0.05, 0.35, 0.65, 0.95))
    print('按覆盖率分位自动选代表帧:', FR, ' 覆盖率:',
          ['%.2f%%' % (100 * cov[f]) for f in FR])

# ---- 深度色标（对数） ----
D0, DMAX = 8.0, 120.0
vmax_lin = 45.0


def depth_rgb(d, valid, mode='log'):
    out = np.zeros(d.shape + (3,), np.uint8)
    m = valid > 0
    if mode == 'log':
        t = (np.log(np.maximum(d, D0)) - np.log(D0)) / (np.log(DMAX) - np.log(D0))
    else:
        t = d / vmax_lin
    t = np.clip(t, 0, 1)
    # turbo 近似色带（暗蓝->青->绿->黄->红）
    r = np.clip(1.5 - np.abs(4 * t - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * t - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * t - 1), 0, 1)
    col = (np.stack([r, g, b], -1) * 255).astype(np.uint8)
    out[m] = col[m]
    out[~m] = np.array([40, 40, 40], np.uint8)
    return out


def colorbar(w, h, mode='log'):
    bar = np.zeros((h, w, 3), np.uint8)
    for i in range(w):
        t = i / (w - 1.0)
        if mode == 'log':
            dm = np.exp(np.log(D0) + t * (np.log(DMAX) - np.log(D0)))
        else:
            dm = t * vmax_lin
        t2 = (np.log(max(dm, D0)) - np.log(D0)) / (np.log(DMAX) - np.log(D0)) if mode == 'log' else dm / vmax_lin
        r = np.clip(1.5 - np.abs(4 * t2 - 3), 0, 1)
        g = np.clip(1.5 - np.abs(4 * t2 - 2), 0, 1)
        b = np.clip(1.5 - np.abs(4 * t2 - 1), 0, 1)
        bar[:, i] = (np.array([r, g, b]) * 255).astype(np.uint8)
    return Image.fromarray(bar)


os.makedirs(OUT, exist_ok=True)
TW = 1024                      # 缩略宽
font = ImageFont.load_default()

for f in FR:
    rgb = Image.open(os.path.join(DS, 'frames', 'frame%06d.png' % f)).convert('RGB')
    dep = np.load(os.path.join(DS, 'depths', 'depth%06d.npy' % f)).astype(np.float64)
    val = np.array(Image.open(os.path.join(DS, 'valid', 'valid%06d.png' % f)))
    # 语义层优先用 L2 SAM2 精修层（本次交付目标②的"完善"部分），
    # 没有则退回官方 L1。sem 可能是 id 图（uint8/uint16）——直接当 id 用，不要过调色板。
    import os as _os
    _l2 = '<PROJECT_ROOT>/tmp/d51/l2/mask_L2_%03d.npy' % f
    if _os.path.exists(_l2):
        sem = np.load(_l2)
        SEM_LAYER = 'L2 SAM2 精修'
    else:
        sem = np.array(Image.open(os.path.join(DS, 'semantic_ids', 'semantic_id%06d.png' % f)))
        SEM_LAYER = 'L1 官方几何'
    th = int(round(TW * rgb.size[1] / rgb.size[0]))

    dlog = Image.fromarray(depth_rgb(dep, val, 'log'))
    dlin = Image.fromarray(depth_rgb(dep, val, 'lin'))
    semc = Image.fromarray(lut[np.clip(sem, 0, 255)])
    ov_d = Image.blend(rgb, dlog, 0.55)
    ov_s = Image.blend(rgb, semc, 0.55)

    # 拼接：上排 RGB / 深度(对数) / 深度(线性45m)；下排 语义 / 深度叠加 / 语义叠加
    top = Image.new('RGB', (TW * 3, th), (20, 20, 20))
    for i, im in enumerate([rgb, dlog, dlin]):
        top.paste(im.resize((TW, th), Image.LANCZOS if im is rgb else Image.NEAREST), (TW * i, 0))
    bot = Image.new('RGB', (TW * 3, th), (20, 20, 20))
    for i, im in enumerate([semc, ov_d, ov_s]):
        bot.paste(im.resize((TW, th), Image.NEAREST if im is semc else Image.LANCZOS), (TW * i, 0))

    lab = Image.new('RGB', (TW * 3, 26), (0, 0, 0))
    dl = ImageDraw.Draw(lab)
    for i, t in enumerate(['frame %03d  RGB' % f,
                           '深度(对数标尺 %.0f-%.0fm)' % (D0, DMAX),
                           '深度(线性 %.0fm, 对照饱和)' % vmax_lin]):
        dl.text((TW * i + 8, 9), t, fill=(255, 255, 255), font=font)
    lab2 = Image.new('RGB', (TW * 3, 26), (0, 0, 0))
    dl2 = ImageDraw.Draw(lab2)
    for i, t in enumerate(['语义(%s, BGR->RGB)' % SEM_LAYER,
                           '深度叠加', '语义叠加']):
        dl2.text((TW * i + 8, 9), t, fill=(255, 255, 255), font=font)

    cv = Image.new('RGB', (TW * 3, th * 2 + 52 + 66), (20, 20, 20))
    cv.paste(lab, (0, 0)); cv.paste(top, (0, 26))
    cv.paste(lab2, (0, 26 + th)); cv.paste(bot, (0, 52 + th))
    # 色标条
    cb = colorbar(TW, 14)
    cv.paste(cb, (0, 52 + th * 2))
    ticks = [8, 12, 20, 35, 60, 120]
    for t in ticks:
        x = int((np.log(t) - np.log(D0)) / (np.log(DMAX) - np.log(D0)) * (TW - 1))
        dl2.text((min(x + 2, TW - 40), 52 + th * 2 + 18), '%dm' % t, fill=(200, 200, 200), font=font)
    # 图例（当前帧出现的类）
    present = sorted(int(x) for x in np.unique(sem) if 0 < x < 250)
    lx, ly = TW * 2, 52 + th * 2 + 40
    dl2.text((lx + 4, ly - 2), '本帧出现 %d 类:' % len(present), fill=(255, 255, 255), font=font)
    for i, cid in enumerate(present):
        px = lx + 8 + (i % 8) * 190
        py = ly + 16 + (i // 8) * 18
        dl2.rectangle([px, py, px + 14, py + 12], fill=tuple(int(x) for x in lut[cid]))
        dl2.text((px + 19, py + 2), '%s(%d)' % (id2name.get(cid, '?'), cid), fill=(230, 230, 230), font=font)

    vm = val > 0
    sat = float((dep[vm] >= DMAX).mean()) if vm.any() else 0.0
    dl2.text((8, 52 + th * 2 + 34),
             'valid=%.2f%%  LiDAR锚=%.3f%%  深度 p50=%.1fm p90=%.1fm  语义覆盖=%.2f%%  对数标尺饱和=%.4f%%'
             % (100 * vm.mean(), 100 * (val == 2).mean(),
                np.median(dep[vm]) if vm.any() else -1,
                np.percentile(dep[vm], 90) if vm.any() else -1,
                100 * cov[f], 100 * sat), fill=(255, 255, 0), font=font)
    p = os.path.join(OUT, '5.1_frame%03d_六联' % f)
    cv.save(p + '.png')
    print('saved %s.png  %s' % (p, cv.size))

# 类例总图
cl = Image.new('RGB', (760, 30 * ((len(legend) + 3) // 4) + 20), (20, 20, 20))
dl3 = ImageDraw.Draw(cl)
dl3.text((8, 6), 'OSDaR23 29 类配色（info_semantic.json 的 BGR 已转 RGB）', fill=(255, 255, 255), font=font)
for i, (cid, nm, col) in enumerate(legend):
    px = 8 + (i % 4) * 188
    py = 26 + (i // 4) * 28
    dl3.rectangle([px, py, px + 18, py + 20], fill=col)
    dl3.text((px + 24, py + 6), '%s (%d)' % (nm, cid), fill=(230, 230, 230), font=font)
cl.save(os.path.join(OUT, '5.1_语义图例_29类.png'))
print('saved legend', cl.size)
print('PREVIEW_51_DONE')
