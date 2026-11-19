#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""用 SLAM 自己的加载器读 5.1 的 L4 数据集，验证训练侧真的读到 L4（而不是官方原始投影）。

不自写读图逻辑 —— 直接实例化 railway.py 的 RailwayDataset（与 slam.py 同一路径），
把 slam.py 用到的那些字段都打出来，和 L4 npy / info_semantic.json 对账。
"""
import glob
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, '<PROJECT_ROOT>/SGS-SLAM')
from scripts.slam import get_dataset

BASE = '<PROJECT_ROOT>/SGS-SLAM'
L4 = '<PROJECT_ROOT>/tmp/d51/l4'
SEQ = 'output_1.0_51_l4'
W, H = 3084, 1878

# 调用形状照抄 slam.py:1092 get_dataset 的真实调用（按铁律⑫：不凭函数名假设）
ds = get_dataset(
    config_dict=dict(
        dataset_name='railway',
        camera_params=dict(
            image_height=2504, image_width=4112,
            fx=7267.95450880415, fy=7267.95450880415,
            cx=2056.049238502414, cy=1232.862908875167,
            png_depth_scale=1.0, crop_edge=0,
        ),
    ),
    basedir=os.path.join(BASE, 'data'),
    sequence=os.path.basename(SEQ),
    start=0, end=-1, stride=1,
    desired_height=H, desired_width=W,
    device='cuda:0', relative_pose=True,
    ignore_bad=False, use_train_split=False,
    load_semantics=True, num_semantic_classes=29,
)

print('帧数 =', len(ds.color_paths), '| 语义路径齐 =',
      len(ds.semantic_id_paths) == len(ds.color_paths) == len(ds.semantic_color_paths))

ok_all = True
for idx in (0, 50, 99):
    item = ds[idx]
    color, depth, intr, pose, sid, scol = item[0], item[1], item[2], item[3], item[4], item[5]
    l4 = np.load(os.path.join(L4, 'semantic_id_%03d.npy' % idx))
    ids = torch.unique(sid).cpu().numpy()
    new_ids = sorted(set(ids.tolist()) - set(np.unique(l4).tolist()))
    l4s = cv2.resize(l4, (W, H), interpolation=cv2.INTER_NEAREST)
    # 实测 raw shape = (H, W, 1)，channels_first=False ⇒ 取 [..., 0] 得 (H, W)。
    # （2026-10-05 踩过：误用 [0] 得 (W,1)、误用 [:,0] 也错，raw 维序以实测为准。）
    sid_np = sid.cpu().numpy()[..., 0].astype(np.int64)
    assert sid_np.shape == l4s.shape, 'shape 不一致 sid=%s(raw %s) l4s=%s' % (
        sid_np.shape, tuple(sid.shape), l4s.shape)
    eq = float((sid_np == l4s.astype(np.int64)).mean())
    print('frame %2d | color %s depth %s sid %s scol %s' % (
        idx, tuple(color.shape), tuple(depth.shape), tuple(sid.shape), tuple(scol.shape)))
    print('         sid 取值 = %s' % ids.tolist())
    print('         L4 原始取值 = %s' % np.unique(l4).tolist())
    print('         降采样后与 L4 最近邻一致率 = %.4f | 非零覆盖 = %.4f'
          % (eq, float((sid_np > 0).mean())))
    print('         scol 值域 [%.3f, %.3f] | depth>0 比例 = %.4f | 深度中位 = %.2f m'
          % (float(scol.min()), float(scol.max()), float((depth > 0).float().mean()),
             float(depth[depth > 0].median())))
    if new_ids:
        print('         !! 出现 L4 里没有的 id: %s' % new_ids)
        ok_all = False
    if eq < 0.9999:
        print('         !! 与 L4 一致率不足')
        ok_all = False

print()
print('=== 语义色 vs info 色表（sRGB 序）===')
import json
info = json.load(open(os.path.join(BASE, 'data', SEQ, 'info_semantic.json')))
irgb = {c['id']: tuple(c['color'][:3]) for c in info['classes']}
item = ds[50]
sid, scol = item[4], item[5]
sid_np = item[4].cpu().numpy()[..., 0].astype(np.int64)
scol_np = item[5].cpu().numpy()[..., :]   # (H, W, 3) 实测维序，与 sid 同
assert scol_np.shape[:2] == sid_np.shape, 'scol shape %s vs sid %s' % (scol_np.shape, sid_np.shape)
for cid in np.unique(sid_np):
    m = sid_np == cid
    if m.sum() < 500:
        continue
    # 实测 scol 已是 0~255（normalize_color 在本配置下未生效，见下方 原样/255 双打）
    raw = np.median(scol_np[m], axis=0)
    got = tuple(int(round(x)) for x in raw)
    exp = irgb.get(int(cid))
    print('  cid %2d  px %8d  图上RGB %s  info %s  %s'
          % (cid, m.sum(), got, exp,
             'MATCH' if exp and max(abs(got[k] - exp[k]) for k in range(3)) <= 2 else 'DIFF'))
print('  （scol 原样值域 [%.1f, %.1f] —— >1.5 说明未归一化，判据按 0~255 比）'
      % (float(scol_np.min()), float(scol_np.max())))

print()
print('LOADER_CHECK_%s' % ('OK' if ok_all else 'FAILED'))
