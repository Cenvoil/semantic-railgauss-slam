#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""建 L4 版数据集 output_1.0_d1_l4：frames/depths/valid/traj 用软链，
   semantic_ids / semantic_colors 的前 8 帧用我们的 L4 输出，其余 92 帧软链到旧图。
   配色：从旧图里逐类取中位色反查（不假设通道顺序），旧图少见的类回退用 reversed(info 色)。"""
import json
import os
from collections import defaultdict

import numpy as np
from PIL import Image

SRC = '<PROJECT_ROOT>/SGS-SLAM/data/output_1.0_d1'
DST = '<PROJECT_ROOT>/SGS-SLAM/data/output_1.0_d1_l4'
L4 = '<PROJECT_ROOT>/tmp/d1/l4'
N_L4 = 8

os.makedirs(DST, exist_ok=True)
for n in ('frames', 'depths', 'valid', 'traj.txt', 'info_semantic.json'):
    d, s = os.path.join(DST, n), os.path.join(SRC, n)
    if not os.path.exists(d):
        os.symlink(s, d)
for n in ('semantic_ids', 'semantic_colors'):
    os.makedirs(os.path.join(DST, n), exist_ok=True)

info = json.load(open(SRC + '/info_semantic.json'))
info_rgb = {c['id']: tuple(c['color'][:3]) for c in info['classes']}

# —— 从旧图反查「类 → 颜色」（多帧取中位）——
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
missing = [cid for cid in info_rgb if cid and cid not in cmap]
for cid in missing:
    cmap[cid] = tuple(info_rgb[cid][::-1])          # 旧图缺的类 → 按旧图的通道顺序（反向）
print('从旧图反查到的类色 %d 个；旧图缺、用回退的类 %s' % (len(acc), missing))

for i in range(100):
    sid_d = os.path.join(DST, 'semantic_ids', 'semantic_id%06d.png' % i)
    sco_d = os.path.join(DST, 'semantic_colors', 'semantic_color%06d.png' % i)
    if i < N_L4:
        a = np.load(L4 + '/semantic_id_%03d.npy' % i).astype(np.uint8)
        Image.fromarray(a).save(sid_d)              # 8 位 L 模式，与旧图一致
        rgb = np.zeros(a.shape + (3,), np.uint8)
        for cid, c in cmap.items():
            rgb[a == cid] = c
        Image.fromarray(rgb).save(sco_d)
    else:
        if not os.path.exists(sid_d):
            os.symlink(os.path.join(SRC, 'semantic_ids', 'semantic_id%06d.png' % i), sid_d)
        if not os.path.exists(sco_d):
            os.symlink(os.path.join(SRC, 'semantic_colors', 'semantic_color%06d.png' % i), sco_d)

# —— 自检 ——
print()
print('自检（前 8 帧：我们的 id ↔ color 是否自洽）')
for i in (0, 3, 7):
    sid = np.asarray(Image.open(os.path.join(DST, 'semantic_ids', 'semantic_id%06d.png' % i)))
    sco = np.asarray(Image.open(os.path.join(DST, 'semantic_colors', 'semantic_color%06d.png' % i)).convert('RGB'))
    bad = 0
    for cid in np.unique(sid):
        m = sid == cid
        if m.sum() < 100:
            continue
        got = tuple(np.median(sco[m], axis=0).astype(int))
        if got != cmap.get(int(cid)):
            bad += 1
    print('  frame %d: mode=%s dtype=%s 类数=%d 配色不符=%d' %
          (i, Image.open(os.path.join(DST, 'semantic_ids', 'semantic_id%06d.png' % i)).mode,
           sid.dtype, len(np.unique(sid)), bad))
for n in ('frames', 'depths', 'valid', 'semantic_ids', 'semantic_colors'):
    print('  %-16s %d 项' % (n, len(os.listdir(os.path.join(DST, n)))))
print('BUILD_L4_CKPT_DONE')
