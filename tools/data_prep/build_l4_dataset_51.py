#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""建 5.1 的 L4 版数据集 output_1.0_51_l4。

从 tmp/d1/build_l4_dataset.py（md5 73fffdb0c882f5cd76217a2199975eed，1.2 权威版）适配：
  1) SRC/DST/L4 三个路径换成 5.1；
  2) N_L4=100（5.1 的 L4 是全量 100 帧都跑了，不是 1.2 早期的前 8 帧）；
  3) 帧尺寸不硬编码（L4 npy 的 shape 自行决定，5.1 与 1.2 同为 4112x2504）。

结构与 1.2 版完全一致：
  frames/depths/valid/traj.txt/info_semantic.json 软链到 SRC（不复制 8GB）；
  semantic_ids/semantic_colors 全部 100 帧由 L4 npy 重新落盘。
  配色不假设通道顺序 —— 从 SRC 旧图逐类取中位色反查，缺的类回退 reversed(info 色)。
  先删旧软链再 save，防止穿透写坏只读源数据。
"""
import json
import os
from collections import defaultdict

import numpy as np
from PIL import Image

SRC = '<PROJECT_ROOT>/SGS-SLAM/data/output_1.0_51'
DST = '<PROJECT_ROOT>/SGS-SLAM/data/output_1.0_51_l4'
L4 = '<PROJECT_ROOT>/tmp/d51/l4'
N = 100
N_L4 = 100

assert os.path.isdir(SRC), 'SRC 不存在: %s' % SRC
assert os.path.isdir(L4), 'L4 不存在: %s' % L4
miss = [i for i in range(N_L4) if not os.path.exists(L4 + '/semantic_id_%03d.npy' % i)]
assert not miss, 'L4 缺帧: %s' % miss[:10]

os.makedirs(DST, exist_ok=True)
for n in ('frames', 'depths', 'valid', 'traj.txt', 'info_semantic.json'):
    d, s = os.path.join(DST, n), os.path.join(SRC, n)
    assert os.path.exists(s), 'SRC 缺 %s' % n
    if not os.path.exists(d):
        os.symlink(s, d)
for n in ('semantic_ids', 'semantic_colors'):
    os.makedirs(os.path.join(DST, n), exist_ok=True)

info = json.load(open(SRC + '/info_semantic.json'))
info_rgb = {c['id']: tuple(c['color'][:3]) for c in info['classes']}

# —— 从 SRC 旧图反查「类 → 颜色」（多帧取中位，不假设通道顺序）——
acc = defaultdict(list)
for i in range(10):
    sid = np.asarray(Image.open(SRC + '/semantic_ids/semantic_id%06d.png' % i)).astype(np.int32)
    sco = np.asarray(Image.open(SRC + '/semantic_colors/semantic_color%06d.png' % i).convert('RGB')).astype(np.int32)
    for cid in np.unique(sid):
        m = sid == cid
        if m.sum() > 200:
            acc[int(cid)].append(tuple(np.median(sco[m], axis=0).astype(int)))
cmap = {}
for cid, lst in acc.items():
    cmap[cid] = tuple(int(np.median(np.array(lst), axis=0)[k]) for k in range(3))

# —— 通道顺序实测裁决（不照抄 1.2 的 [::-1]）——
# 1.2 脚本对「旧图缺的类」用 info 色反转，但那只在旧图是 BGR 时成立。
# 5.1 实测：类 3 旧图 (0,0,142) == info RGB (0,0,142)；类 18 旧图 (220,220,0) == info RGB，
# 两类都唯一匹配 RGB（反转不匹配）⇒ 5.1 的 semantic_colors 是 RGB 序，**不反转**。
# 这里按实测自动判，而不是把结论写死：能匹配的类数 RGB 多就�� RGB，反之 BGR。
n_rgb = n_bgr = 0
for cid in acc:
    rgb = info_rgb.get(cid)
    if rgb is None:
        continue
    if cmap[cid] == rgb:
        n_rgb += 1
    elif cmap[cid] == rgb[::-1]:
        n_bgr += 1
print('通道裁决：与 info RGB 匹配 %d 类 / 与 info BGR 匹配 %d 类 ⇒ 用 %s'
      % (n_rgb, n_bgr, 'RGB(不反转)' if n_rgb >= n_bgr else 'BGR(反转)'))
REV = n_bgr > n_rgb
missing = [cid for cid in info_rgb if cid and cid not in cmap]
for cid in missing:
    cmap[cid] = tuple(info_rgb[cid][::-1]) if REV else tuple(info_rgb[cid])
print('从 SRC 旧图反查到的类色 %d 个；旧图缺、用回退的类 %s' % (len(acc), missing))
print('  回退色示例 %s' % {c: cmap[c] for c in missing[:4]})

all_ids = set()
SRC_MODE = Image.open(SRC + '/semantic_ids/semantic_id000000.png').mode
print('SRC 旧图 semantic_id 的 mode = %s（L4 npy dtype = %s）'
      % (SRC_MODE, np.load(L4 + '/semantic_id_000.npy').dtype))
for i in range(N):
    a = np.load(L4 + '/semantic_id_%03d.npy' % i)
    all_ids |= set(np.unique(a).tolist())
    assert np.issubdtype(a.dtype, np.integer), 'frame %d dtype=%s 非整型' % (i, a.dtype)
    assert a.shape == (2504, 4112), 'frame %d shape=%s 与 SRC 不符' % (i, a.shape)
    illegal = sorted(x for x in np.unique(a).tolist() if x not in info_rgb)
    assert not illegal, 'frame %d 非法 id %s' % (i, illegal)
    sid_d = os.path.join(DST, 'semantic_ids', 'semantic_id%06d.png' % i)
    sco_d = os.path.join(DST, 'semantic_colors', 'semantic_color%06d.png' % i)
    for p in (sid_d, sco_d):                    # 先删旧软链，防止 save 穿透写坏源数据
        if os.path.islink(p):
            os.remove(p)
    # 不做 1.2 版的 astype(uint8) 再存 L 模式：加载器 imageio.imread 两种模式都能读成整型，
    # 但 L 模式会把 16bit 容器降成 8bit。这里按 L4 原 dtype 存，模式与 SRC 旧图保持一致。
    Image.fromarray(a).save(sid_d)
    rgb = np.zeros(a.shape + (3,), np.uint8)
    for cid, c in cmap.items():
        rgb[a == cid] = c
    Image.fromarray(rgb).save(sco_d)
    if (i + 1) % 20 == 0:
        print('  写入 %d/%d' % (i + 1, N))
print('L4 出现的 id 合计 %d 个：%s' % (len(all_ids), sorted(all_ids)))

# —— 自检 ——
print()
print('自检（我们的 id ↔ color 是否自洽 + 与 L4 npy 是否逐像素一致）')
for i in (0, 3, 7, 50, 99):
    p = os.path.join(DST, 'semantic_ids', 'semantic_id%06d.png' % i)
    sid = np.asarray(Image.open(p))
    sco = np.asarray(Image.open(os.path.join(DST, 'semantic_colors', 'semantic_color%06d.png' % i)).convert('RGB'))
    l4 = np.load(L4 + '/semantic_id_%03d.npy' % i)
    bad = 0
    for cid in np.unique(sid):
        m = sid == cid
        if m.sum() < 100:
            continue
        if tuple(np.median(sco[m], axis=0).astype(int)) != cmap.get(int(cid)):
            bad += 1
    cover = float((sid > 0).mean())
    print('  frame %d: mode=%s dtype=%s 类数=%d 配色不符=%d | 与L4npy逐像素一致=%s | 非零覆盖率=%.2f%%'
          % (i, Image.open(p).mode, sid.dtype, len(np.unique(sid)), bad,
             bool((sid == l4).all()), cover * 100))
for n in ('frames', 'depths', 'valid', 'semantic_ids', 'semantic_colors'):
    d = os.path.join(DST, n)
    items = os.listdir(d)
    # 统计「目录本身是否是软链」用 os.path.islink(d)；os.listdir 沿软链列的是子项，
    # 数子项里的软链会得 0，2026-10-05 首次跑时就是这个口径错误。
    print('  %-16s %d 项 | 本目录是软链=%s' % (n, len(items), os.path.islink(d)))
for n in ('frames', 'depths', 'valid', 'traj.txt', 'info_semantic.json'):
    print('  顶层软链 %-20s -> %s' % (n, 'OK' if os.path.islink(os.path.join(DST, n)) else 'NOT_A_LINK'))
print('BUILD_L4_51_DONE')
