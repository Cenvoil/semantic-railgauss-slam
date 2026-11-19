# -*- coding: utf-8 -*-
"""投影口径最终裁定：直接复用 1.2 已验证的约定（M 矩阵轴重排），在 5.1 上实证。

背景（自检实测，勿再重推）：
  * pose_wrt_parent 的 R@ez ≈ [0.11, 0.01, 0.99] 沿 base 的 Z（上）⇒ 该 R 的列/行
    与"相机 z 前"不对应。实测 R@ex 让 80% 轨面点落在相机前方。
  * 但 1.2 的 build_l1v9.py 用 M=[[0,0,1],[-1,0,0],[0,-1,0]] 做重排，且**产出过
    正确的 1.2 官方几何掩膜**（100 帧、门③召回 1.0000）。那个约定是验证过的。

所以本脚本不重新发明约定，而是：
  1) 把 build_l1v9 的约定原样搬来（R = (quat2R(q) @ M).T, tv = -R @ t）；
  2) 在 5.1 上跑，判据 = 轨面点投影后应落在图像下半部、v 随 z 收敛到 cy、深度合理；
  3) 同时用**官方 poly3d 轨线**做第二判据（1.2 已用它验过投影自洽）。

若两个判据一致 → 口径确立，深度锚定与语义投影都用它。
"""
import glob
import json
import os
import sys

import numpy as np
from PIL import Image

DST = sys.argv[1] if len(sys.argv) > 1 else '<PROJECT_ROOT>/data_osdar/unpacked_51'
LAB = glob.glob(os.path.join(DST, '*_labels.json'))[0]
CAM = 'rgb_highres_center'
d = json.load(open(LAB))['openlabel']
sp = d['streams'][CAM]['stream_properties']['intrinsics_pinhole']
K = np.array(sp['camera_matrix'], float).reshape(3, 4)[:3, :3]
DSTC = np.array(sp.get('distortion_coeffs', [0] * 5), float)
pose = d['coordinate_systems'][CAM]['pose_wrt_parent']
t = np.array(pose['translation'], float)
q = np.array(pose['quaternion'], float)
M = np.array([[0, 0, 1.0], [-1.0, 0, 0], [0, -1.0, 0]])   # 与 1.2 相同
FX, FY, CX, CY = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
W, H = Image.open(sorted(glob.glob(os.path.join(DST, CAM, '*.png')))[0]).size


def quat2R(qq):
    x, y, z, w = qq
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


R = (quat2R(q) @ M).T
tv = -R @ t
print('R =\n', R.round(6))
print('tv =', tv.round(6))
print('图像 %dx%d  cy=%.4f  相机高 t_z=%.4f' % (W, H, CY, t[2]))


def distort(xn, yn):
    k1, k2, p1, p2, k3 = DSTC
    r2 = xn * xn + yn * yn
    f = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 ** 3
    return (xn * f + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn * xn),
            yn * f + p1 * (r2 + 2 * yn * yn) + 2 * p2 * xn * yn)


def project(P):
    """返回 (uv, z) —— 已滤掉 z<=0.3，uv 与 z 形状一致。"""
    Pc = (R @ np.asarray(P, float).T).T + tv
    z = Pc[:, 2]
    m = z > 0.3
    xn, yn = Pc[m, 0] / z[m], Pc[m, 1] / z[m]
    xd, yd = distort(xn, yn)
    u = FX * xd + CX
    v = FY * yd + CY
    return np.stack([u, v], 1), z[m]


def load_pcd(path):
    with open(path) as fh:
        h = {}
        while True:
            line = fh.readline(); p = line.split()
            if not p:
                continue
            if p[0] in ('FIELDS', 'WIDTH', 'POINTS', 'DATA'):
                h[p[0]] = p[1:]
            if p[0] == 'DATA':
                break
        return np.loadtxt(fh, max_rows=int(h['POINTS'][0]))


pcds = sorted(glob.glob(os.path.join(DST, 'lidar', '*.pcd')))
print('\n=== 判据 1：LiDAR 轨面点（10 帧）===')
U, V, Z, nrail, nin = [], [], [], 0, 0
for pp in pcds[:10]:
    P = load_pcd(pp)[:, :3].astype(float)
    rail = P[(np.abs(P[:, 2]) < 0.05) & (P[:, 0] > 6) & (P[:, 0] < 150)]
    nrail += len(rail)
    uv, z = project(rail)
    inb = (uv[:, 0] >= 0) & (uv[:, 0] < W - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < H - 1)
    U.append(uv[inb, 0]); V.append(uv[inb, 1]); Z.append(z[inb]); nin += int(inb.sum())
u = np.concatenate(U); v = np.concatenate(V); z = np.concatenate(Z)
print('  轨面点 %d -> 入画 %d (%.1f%%)' % (nrail, nin, 100.0 * nin / max(nrail, 1)))
if nin:
    print('  u: p5=%.0f p50=%.0f p95=%.0f  (图宽 %d)' % (np.percentile(u, 5), np.percentile(u, 50), np.percentile(u, 95), W))
    print('  v: p5=%.0f p50=%.0f p95=%.0f  (图高 %d, cy=%.0f)' % (np.percentile(v, 5), np.percentile(v, 50), np.percentile(v, 95), H, CY))
    print('  z: p5=%.1f p50=%.1f p95=%.1f m' % (np.percentile(z, 5), np.percentile(z, 50), np.percentile(z, 95)))
    print('  v/cy 中位 = %.2f  （>1 = 在主点下方 = 钢轨在脚下 = 正确）' % float(np.median(v) / CY))

print('\n=== 判据 1b：地面自洽性 cy_obs = v - fy*H/z（H=3.5098）===')
Hc = 3.50982
for lo, hi in [(6, 15), (15, 30), (30, 60), (60, 150)]:
    k = (z >= lo) & (z < hi)
    if k.sum() < 30:
        print('    z∈[%3d,%3d) n=%5d (不足)' % (lo, hi, int(k.sum())))
        continue
    cy_obs = float(np.median(v[k] - FY * Hc / z[k]))
    print('    z∈[%3d,%3d) n=%6d  cy_obs=%9.2f  与CY差=%+8.2f  残差std=%6.2f'
          % (lo, hi, int(k.sum()), cy_obs, cy_obs - CY, float(np.std(v[k] - FY * Hc / z[k]))))
print('  （各档 cy_obs 近似常数 = 地面是平面、投影自洽；单调漂移 = 约定错）')

print('\n=== 判据 2：官方 poly3d 轨线投影 ===')
frames, objs = d['frames'], d['objects']
cnt = 0
for fk in ('0', '10', '30', '50', '70', '90'):
    fr = frames.get(fk) or frames.get(int(fk))
    if fr is None:
        continue
    for oid, od in (fr.get('objects') or {}).items():
        src = od.get('object_data') or objs.get(oid, {}).get('object_data', {})
        for a in (src.get('poly3d') or []):
            v3 = np.array(a['val'], float).reshape(-1, 3)
            if len(v3) < 2:
                continue
            uv, zz = project(v3)
            inb = (uv[:, 0] >= 0) & (uv[:, 0] < W - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < H - 1)
            cls = objs.get(oid, {}).get('type')
            if inb.sum() >= 2:
                cnt += 1
                if cnt <= 8:
                    print('  f%-3s %-14s 顶点%2d 入画%2d  u[%6.0f,%6.0f] v[%6.0f,%6.0f] z[%5.1f,%5.1f]'
                          % (fk, cls, len(v3), int(inb.sum()),
                             uv[inb, 0].min(), uv[inb, 0].max(),
                             uv[inb, 1].min(), uv[inb, 1].max(),
                             zz.min(), zz.max()))
print('  poly3d 段入画数 =', cnt)

print('\n=== 判据 3：官方 bbox/poly2d 与投影的粗对账（同一对象） ===')
hit = 0
tot = 0
for fk in ('0', '30', '60', '90'):
    fr = frames.get(fk) or frames.get(int(fk))
    if fr is None:
        continue
    for oid, od in (fr.get('objects') or {}).items():
        src = od.get('object_data') or objs.get(oid, {}).get('object_data', {})
        bb = [a for a in (src.get('bbox') or []) if a.get('coordinate_system') == CAM]
        cub = src.get('cuboid') or []
        if not bb or not cub:
            continue
        bu, bv, bw, bh = bb[0]['val'][:4]
        ctr = np.array(cub[0]['val'][:3], float)
        uv, _ = project(ctr.reshape(1, 3))
        if not len(uv):
            continue
        tot += 1
        pu, pv = float(uv[0, 0]), float(uv[0, 1])
        # cuboid 中心通常在框内（允许框外一点，轨道类/远场例外）
        inside = (bu - 0.3 * bw <= pu <= bu + 1.3 * bw) and (bv - 0.3 * bh <= pv <= bv + 1.3 * bh)
        hit += int(inside)
        if tot <= 10:
            print('  f%-3s %-14s bbox=(%.0f,%.0f,%.0f,%.0f) 投影中心=(%.0f,%.0f) %s'
                  % (fk, objs.get(oid, {}).get('type'), bu, bv, bw, bh, pu, pv,
                     'IN' if inside else 'OUT'))
print('  cuboid 中心落在 bbox 内: %d / %d' % (hit, tot))
print('  （这是 1.2 已用过的粗对账；高比例 IN 即投影自洽）')
