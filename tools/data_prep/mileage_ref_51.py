# -*- coding: utf-8 -*-
"""5.1 里程基准的多静态物核对（替代 v4.1 的"单个 buffer_stop"）。

背景：make_depth_51.py 的断言正确拦住了 —— 5.1 的 3 个 buffer_stop 实例
**没有一个满 100 帧**，所以 v4.1 那套 `len(traj[oid])==100` 的取法在 5.1 上直接失效。
这不是 bug，是 1.2 的假设在 5.1 上不成立（MEMORY §8 已预警）。

做法（skill §24.2）：
  1. 对每个静态类实例，算它"逐帧 x 递减量"（若它在动，x 会变；若标注在当帧车体系，
     静态物的 x 应随车前进而单调减小，递减量 ≈ 行驶距离）；
  2. 用**多个实例的位移中位**估 travel，而不是取单个；
  3. 加一致性判据：各实例给出的总位移应接近，偏差大的剔除；
  4. 报告噪声本底，供后续阈值标定 —— 但**不做"改法"**（Gary 9-26：不主动立项）。

这里只输出事实表，供 make_depth_51 选基准。
"""
import collections
import glob
import json
import os
import sys

import numpy as np

UNP = sys.argv[1] if len(sys.argv) > 1 else '<PROJECT_ROOT>/data_osdar/unpacked_51'
LAB = glob.glob(os.path.join(UNP, '*_labels.json'))[0]
d = json.load(open(LAB))['openlabel']
objs, frames = d['objects'], d['frames']

# 收集每个实例的逐帧 cuboid 中心
traj = collections.defaultdict(list)
for oid, o in objs.items():
    for fk, f in frames.items():
        oe = (f.get('objects') or {}).get(oid)
        if not oe:
            continue
        for cu in (oe.get('object_data') or {}).get('cuboid', []):
            v = cu['val']
            traj[oid].append((int(fk), float(v[0]), float(v[1]), float(v[2]), o['type']))

STATIC = {'catenary_pole', 'signal', 'signal_pole', 'switch', 'track', 'buffer_stop',
          'building', 'road_vehicle', 'person', 'train', 'crowd'}

print('=== 逐实例：x 递减量（应 ≈ 行驶距离） ===')
print('%-14s %-22s %5s %8s %8s %8s %9s' % ('类', 'uuid', '帧数', 'x首', 'x末', '递减', 'x极差'))
rows = []
for oid, lst in sorted(traj.items(), key=lambda kv: objs[kv[0]]['type']):
    lst.sort()
    if objs[oid]['type'] not in STATIC or len(lst) < 20:
        continue
    xs = np.array([r[1] for r in lst])
    fks = [r[0] for r in lst]
    span = fks[-1] - fks[0] + 1
    dec = xs[0] - xs[-1]
    rng_ = xs.max() - xs.min()
    rows.append({'oid': oid, 'type': objs[oid]['type'], 'n': len(lst),
                 'f0': fks[0], 'f1': fks[-1], 'span': span,
                 'dec': float(dec), 'x_range': float(rng_),
                 'per_frame': float(dec) / max(span - 1, 1)})
    print('%-14s %-22s %5d %8.2f %8.2f %8.2f %9.2f' % (
        objs[oid]['type'], oid, len(lst), xs[0], xs[-1], dec, rng_))

print()
print('=== 候选"长跨度"基准（帧数 >= 60）===')
cand = [r for r in rows if r['span'] >= 60]
for r in sorted(cand, key=lambda z: -z['span']):
    print('  %-14s %-22s 帧 %3d~%3d (跨度%3d)  递减=%8.2f m  每帧=%6.3f m  x极差=%6.2f m'
          % (r['type'], r['oid'], r['f0'], r['f1'], r['span'], r['dec'], r['per_frame'], r['x_range']))

if cand:
    decs = np.array([r['dec'] for r in cand], float)
    rngs = np.array([r['x_range'] for r in cand], float)
    print()
    print('  递减量: 中位=%.2f  均值=%.2f  std=%.2f  极差=%.2f  n=%d'
          % (np.median(decs), decs.mean(), decs.std(), decs.max() - decs.min(), len(decs)))
    print('  x极差(标注噪声本底): 中位=%.2f  最大=%.2f' % (np.median(rngs), rngs.max()))
    med = float(np.median(decs))
    keep = [r for r in cand if abs(r['dec'] - med) < 0.15 * med]
    print('  与中位相差 <15%% 的实例: %d / %d' % (len(keep), len(cand)))
    print('  => 建议基准: 多个静态物 x 递减量取中位 = %.2f m（不用单个 buffer_stop）' % med)
else:
    print('  !! 无跨度 >= 60 的静态物')

print()
print('=== 与官方 INS 里程对照（独立量） ===')
rowsi = []
for p in sorted(glob.glob(os.path.join(UNP, 'novatel_oem7_inspva', '*.csv'))):
    with open(p) as f:
        L = [x for x in f.read().splitlines() if x.strip()]
    h = L[0].split(',')
    rowsi.append({k: float(v) for k, v in zip(h, L[1].split(','))})
rowsi.sort(key=lambda r: r['frame_idx'])
la0, lo0 = rowsi[0]['latitude'], rowsi[0]['longitude']
latr = np.deg2rad(la0)
m_lat = 111132.92 - 559.82 * np.cos(2 * latr) + 1.175 * np.cos(4 * latr)
m_lon = 111412.84 * np.cos(latr) - 93.5 * np.cos(3 * latr)
E = np.array([(r['longitude'] - lo0) * m_lon for r in rowsi])
N = np.array([(r['latitude'] - la0) * m_lat for r in rowsi])
U = np.array([r['height'] - rowsi[0]['height'] for r in rowsi])
xy = np.stack([E, N], 1)
d2 = np.linalg.norm(np.diff(xy, axis=0), axis=1)
print('  INS 平面总里程 = %.2f m  每帧位移 中位=%.3f m  高差=%.2f m'
      % (d2.sum(), float(np.median(d2)), float(U[-1] - U[0])))
print('  首末直线距离 = %.2f m  （与累计里程之差 = 弯道/转弯累积）'
      % float(np.linalg.norm(xy[-1] - xy[0])))
