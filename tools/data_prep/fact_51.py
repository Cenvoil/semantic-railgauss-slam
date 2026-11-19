# -*- coding: utf-8 -*-
"""5.1 官方内参/位姿/标注结构 事实核验（不推演，只读官方值）。

为什么要这个：
  1.2 的 labels.json 给 cy=1232.8629（已裁 4112x3008->4112x2504），
  同序列的 calibration.txt 却给 1486.86（未裁切），两者冲突。
  MEMORY 裁定"投影唯一权威 = labels.json"。5.1 的 zip **没有 calibration.txt**，
  所以 labels.json 是唯一来源 —— 但仍要实测它给的是裁切值还是原生值，
  以及 fx/cx 是否与 1.2 同（不同场景组可能是不同相机）。
"""
import collections
import glob
import json
import os
import sys

import numpy as np
from PIL import Image

DST = sys.argv[1] if len(sys.argv) > 1 else '<PROJECT_ROOT>/data_osdar/unpacked_51'
LAB = glob.glob(os.path.join(DST, '*_labels.json'))[0]
CAM = 'rgb_highres_center'
REF = '<PROJECT_ROOT>/data_osdar/ex/1_calibration_1.2_labels.json'

print('=== 0. 图像分辨率（5.1 实际发布） ===')
pngs = sorted(glob.glob(os.path.join(DST, CAM, '*.png')))
print('  帧数 =', len(pngs), ' 尺寸 =', Image.open(pngs[0]).size, ' 文件名例 =', os.path.basename(pngs[0]))

d = json.load(open(LAB))['openlabel']
dr = json.load(open(REF))['openlabel']

print('\n=== 1. 官方内参（labels.json -> streams.%s） ===' % CAM)
for tag, dd in (('5.1', d), ('1.2', dr)):
    sp = dd['streams'][CAM]['stream_properties']['intrinsics_pinhole']
    K = np.array(sp['camera_matrix'], float).reshape(3, 4)[:3, :3]
    D = sp.get('distortion_coeffs', [0] * 5)
    print('  [%s] fx=%.6f fy=%.6f cx=%.4f cy=%.4f' % (tag, K[0, 0], K[1, 1], K[0, 2], K[1, 2]))
    print('       dist=%s' % list(map(float, D)))
    sp2 = dd['streams'][CAM]['stream_properties']
    for k in sp2:
        if k not in ('intrinsics_pinhole',):
            print('       %s = %s' % (k, sp2[k]))
K51 = np.array(d['streams'][CAM]['stream_properties']['intrinsics_pinhole']['camera_matrix'], float).reshape(3, 4)[:3, :3]
K12 = np.array(dr['streams'][CAM]['stream_properties']['intrinsics_pinhole']['camera_matrix'], float).reshape(3, 4)[:3, :3]
print('  5.1 - 1.2 =', (K51 - K12).round(4).tolist())

print('\n=== 2. 相机安装外参（pose_wrt_parent） ===')
for tag, dd in (('5.1', d), ('1.2', dr)):
    p = dd['coordinate_systems'][CAM]['pose_wrt_parent']
    print('  [%s] translation = %s' % (tag, np.array(p['translation'], float).round(6).tolist()))
    print('       quaternion(xyzw) = %s' % np.array(p['quaternion'], float).round(8).tolist())
    print('       相机高 t_z = %.4f m' % p['translation'][2])

print('\n=== 3. 标注对象结构 ===')
objs, frames = d['objects'], d['frames']
print('  帧数 =', len(frames), '  实例数 =', len(objs))
tc = collections.Counter(o['type'] for o in objs.values())
print('  按类实例数:', dict(sorted(tc.items(), key=lambda kv: -kv[1])))
kinds = collections.Counter()
for f in frames.values():
    for od in (f.get('objects') or {}).values():
        for k in (od.get('object_data') or {}):
            kinds[k] += 1
print('  object_data 键名计数:', dict(kinds))
nfr = collections.Counter()
for oid, o in objs.items():
    n = sum(1 for f in frames.values() if oid in (f.get('objects') or {}))
    nfr[n] += 1
print('  每实例出现帧数分布（帧数:实例数）前 8:', dict(sorted(nfr.items())[:8]))

print('\n=== 4. 坐标系统清单 ===')
print(' ', list(d['coordinate_systems'].keys()))
print('  streams:', list(d['streams'].keys()))

print('\n=== 5. 动态类基准物（buffer_stop）可用性 ===')
cand = [(oid, sum(1 for f in frames.values() if oid in (f.get('objects') or {})))
        for oid, o in objs.items() if o['type'] == 'buffer_stop']
print('  buffer_stop 实例数 =', len(cand), ' 出现帧数 =', sorted(set(c[1] for c in cand)))
cand2 = [(oid, sum(1 for f in frames.values() if oid in (f.get('objects') or {})))
         for oid, o in objs.items() if o['type'] in ('catenary_pole', 'switch', 'track')]
print('  catenary/switch/track 实例数 =', len(cand2),
      ' 出现帧数 =', sorted(set(c[1] for c in cand2))[:6])

print('\n=== 6. INS 采样 ===')
ins = sorted(glob.glob(os.path.join(DST, 'novatel_oem7_inspva', '*.csv')))
print('  文件数 =', len(ins), ' 例 =', os.path.basename(ins[0]))
with open(ins[0]) as f:
    L = [x for x in f.read().splitlines() if x.strip()]
hdr = L[0].split(',')
print('  列名 =', hdr)
print('  数据行数 =', len(L) - 1)
print('  首行值 =', dict(zip(hdr, L[1].split(','))))
