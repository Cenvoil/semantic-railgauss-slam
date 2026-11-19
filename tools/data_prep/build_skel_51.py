# -*- coding: utf-8 -*-
"""5.1 数据集骨架构建（output_1.0_51）。

做四件事（与 1.2 的 output_1.0_d1 同构，SGS-SLAM dataloader 可直接读）：
  1. frames/        —— 软链或复制 rgb_highres_center 的 100 帧
  2. info_semantic.json —— 29 类表（沿用 1.2 的 taxonomy 与配色；**color 是 BGR**）
  3. semantic_colors/  —— 官方 poly2d 直接上色的原始语义图（conf 层的底）
  4. traj.txt       —— 官方 INS（INSPVA）→ ENU → 叠加相机安装外参 → c2w，3x4 行主序

关键口径（都已在本轮实测确认，不是照抄 1.2）：
  * 图像 4112x2504，官方 labels.json 内参 cy=1232.8629（**已裁切值**），5.1 与 1.2 逐位相同
  * 相机安装外参 t=[0.080158,-0.332862,3.50982]，q=xyzw，与 1.2 逐位相同
  * 投影轴重排 M=[[0,0,1],[-1,0,0],[0,-1,0]]，R=(quat2R(q)@M).T —— 已在 5.1 上实证
    （地面 cy_obs 四档近似常数 414.7/416.0/414.6/412.1，残差 std 17.5→4.1）
  * INS 列名：frame_idx,timestamp,latitude,longitude,height,north_velocity,
    east_velocity,up_velocity,roll,pitch,azimuth（5.1 无 corrimu）

用法: python build_skel_51.py <unpacked_dir> <out_dataset_dir> [n_frames]
"""
import glob
import json
import os
import shutil
import sys

import numpy as np
from PIL import Image, ImageDraw

UNP = sys.argv[1] if len(sys.argv) > 1 else '<PROJECT_ROOT>/data_osdar/unpacked_51'
OUT = sys.argv[2] if len(sys.argv) > 2 else '<PROJECT_ROOT>/SGS-SLAM/data/output_1.0_51'
NFR = int(sys.argv[3]) if len(sys.argv) > 3 else 100
CAM = 'rgb_highres_center'
LAB = glob.glob(os.path.join(UNP, '*_labels.json'))[0]
SEQ = '5_station_bergedorf_5.1'
W, H = 4112, 2504

# ---- 类表：沿用 1.2（同一 OSDaR23 taxonomy），只换 sequence ----
REF = '<PROJECT_ROOT>/SGS-SLAM/data/output_1.0_d1/info_semantic.json'
info = json.load(open(REF))
info['sequence'] = SEQ
info['description'] = ('Railway station scene (station_bergedorf), 100 frames; '
                       'classes derived from OSDaR23 labeling guide')
json.dump(info, open(os.path.join(OUT, 'info_semantic.json'), 'w'), indent=1)
lut = np.zeros((256, 3), np.uint8)
for c in info['classes']:
    lut[c['id']] = np.array(c['color'][:3], np.uint8)   # BGR，与 1.2 同
name2id = {c['name']: c['id'] for c in info['classes']}
print('[skel] info_semantic.json: %d 类, sequence=%s' % (info['num_classes'], SEQ))

# ---- frames ----
os.makedirs(os.path.join(OUT, 'frames'), exist_ok=True)
os.makedirs(os.path.join(OUT, 'semantic_colors'), exist_ok=True)
os.makedirs(os.path.join(OUT, 'semantic_ids'), exist_ok=True)
src = sorted(glob.glob(os.path.join(UNP, CAM, '*.png')))
assert len(src) >= NFR, '图像不足：%d < %d' % (len(src), NFR)
for i, s in enumerate(src[:NFR]):
    dst = os.path.join(OUT, 'frames', 'frame%06d.png' % i)
    if not os.path.exists(dst):
        # 用复制而非软链：1.2 的 output_1.0_d1/frames 实测是普通文件，
        # 保持同构，避免下游重建数据集时软链穿透写坏只读源数据（skill §4 注）
        shutil.copyfile(s, dst)
print('[skel] frames: %d 帧（复制）' % NFR)

# ---- 官方 poly2d 上色（原始语义，供 L1/L4 与 conf=2 层用） ----
d = json.load(open(LAB))['openlabel']
sp = d['streams'][CAM]['stream_properties']['intrinsics_pinhole']
K = np.array(sp['camera_matrix'], float).reshape(3, 4)[:3, :3]
DSTC = np.array(sp.get('distortion_coeffs', [0] * 5), float)
pose = d['coordinate_systems'][CAM]['pose_wrt_parent']
t_mount = np.array(pose['translation'], float)
q = np.array(pose['quaternion'], float)
M = np.array([[0, 0, 1.0], [-1.0, 0, 0], [0, -1.0, 0]])
frames_lab, objs = d['frames'], d['objects']


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
tv = -R @ t_mount
FX, FY, CX, CY = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

# ---- 权威口径自检（复现 proj_canon v2 的自检，不另立公式）----
# ⚠ 必须用 proj_canon 的**原式**：`RJ.T @ (P - TC)`，v = CY + FY*(-b2)/z。
#   我先前写成 R=(quat2R@M).T 再套 tv / R.T，符号约定不同，算出 v_h=-51786 的假异常，
#   被自己的断言拦住了。教训：**自检公式要照抄权威实现，"等价改写"就是引入符号分歧。**
_RJ = quat2R(q)                       # 纯四元数旋转（不含 M），= proj_canon 的 RJ
_TC = t_mount
_v20 = CY + FY * (-(_RJ.T @ (np.array([20.0, 0.0, 0.0]) - _TC))[2]) / (_RJ.T @ (np.array([20.0, 0.0, 0.0]) - _TC))[0]
_b100 = _RJ.T @ (np.array([100.0, 0.0, 0.0]) - _TC)
_v100 = CY + FY * (-_b100[2]) / _b100[0]
_vh, _c = np.linalg.solve(np.array([[1.0, 1 / 20.0], [1.0, 1 / 100.0]]),
                           np.array([_v20, _v100]))
print('[skel] 权威口径自检：相机高 t_z=%.4f  俯角=%.2f°  VH=%.1f  C=%.1f'
      % (t_mount[2], np.degrees(2 * np.arccos(min(1.0, abs(q[3])))), _vh, _c))
print('       （proj_canon v2 记录 VH≈406、C≈25026；本脚本实测 VH=%.1f、C=%.1f）' % (_vh, _c))
assert abs(_vh - 406.0) < 40.0, 'VH 异常：%.1f（与权威 406 差过大）' % _vh


def distort(xn, yn):
    k1, k2, p1, p2, k3 = DSTC
    r2 = xn * xn + yn * yn
    f = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 ** 3
    return (xn * f + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn * xn),
            yn * f + p1 * (r2 + 2 * yn * yn) + 2 * p2 * xn * yn)


def project(P):
    Pc = (R @ np.asarray(P, float).T).T + tv
    z = Pc[:, 2]
    m = z > 0.3
    xn, yn = Pc[m, 0] / z[m], Pc[m, 1] / z[m]
    xd, yd = distort(xn, yn)
    return np.stack([FX * xd + CX, FY * yd + CY], 1), z[m]


def build_sem(fk):
    im = Image.new('L', (W, H), 0)
    dr = ImageDraw.Draw(im)
    fr = frames_lab.get(str(fk)) or frames_lab.get(fk)
    if fr is None:
        return np.asarray(im)
    for oid, od in (fr.get('objects') or {}).items():
        src_ = od.get('object_data') or objs.get(oid, {}).get('object_data', {})
        cls = objs.get(oid, {}).get('type') or '?'
        cid = name2id.get(cls, 0)
        for a in (src_.get('poly2d') or []):
            if a.get('coordinate_system') != CAM:
                continue
            v = np.array(a['val'], float).reshape(-1, 2)
            if len(v) < 2:
                continue
            pts = [(float(x), float(y)) for x, y in v]
            if a.get('closed', False) and len(v) >= 3:
                dr.polygon(pts, fill=int(cid))
            else:
                dr.line(pts, fill=int(cid), width=8)
        for a in (src_.get('poly3d') or []):
            if [x for x in (src_.get('poly2d') or []) if x.get('coordinate_system') == CAM]:
                continue
            v = np.array(a['val'], float).reshape(-1, 3)
            uv, zz = project(v)
            if len(uv) < 2:
                continue
            keep = (zz > 1.0) & (uv[:, 0] > -W) & (uv[:, 0] < 2 * W) & (uv[:, 1] > -H) & (uv[:, 1] < 2 * H)
            idx = np.where(keep)[0]
            seg, last = [], None
            for i in idx:
                pt = (float(uv[i, 0]), float(uv[i, 1]))
                if last and (abs(pt[0] - last[0]) > 600 or abs(pt[1] - last[1]) > 600):
                    if len(seg) >= 2:
                        dr.line(seg, fill=int(name2id.get('track', 13)), width=10)
                    seg = []
                seg.append(pt)
                last = pt
            if len(seg) >= 2:
                dr.line(seg, fill=int(name2id.get('track', 13)), width=10)
    return np.asarray(im)


nlab = 0
for i in range(NFR):
    m = build_sem(i)
    p = os.path.join(OUT, 'semantic_ids', 'semantic_id%06d.png' % i)
    Image.fromarray(m.astype(np.uint16)).save(p)
    Image.fromarray(lut[np.clip(m, 0, 255)]).save(
        os.path.join(OUT, 'semantic_colors', 'semantic_color%06d.png' % i))
    nlab += 1
    if i % 20 == 0:
        print('[skel] 官方语义 %3d/%d  覆盖率 %.4f%%' % (i, NFR, 100.0 * (m > 0).mean()))
print('[skel] semantic_ids/semantic_colors: %d 帧' % nlab)

# ---- traj.txt：官方 INS → ENU → c2w（首帧相机为世界原点） ----
rows = []
for p in sorted(glob.glob(os.path.join(UNP, 'novatel_oem7_inspva', '*.csv'))):
    with open(p) as f:
        L = [x for x in f.read().splitlines() if x.strip()]
    hdr = L[0].split(',')
    rows.append({k: float(v) for k, v in zip(hdr, L[1].split(','))})
rows.sort(key=lambda r: r['frame_idx'])
assert len(rows) >= NFR, 'INS 帧数不足：%d < %d' % (len(rows), NFR)
rows = rows[:NFR]
print('[skel] INS 帧数 =', len(rows), ' 经纬高首帧 =',
      round(rows[0]['latitude'], 6), round(rows[0]['longitude'], 6))

lat0, lon0 = rows[0]['latitude'], rows[0]['longitude']
latr = np.deg2rad(lat0)
m_lat = 111132.92 - 559.82 * np.cos(2 * latr) + 1.175 * np.cos(4 * latr)
m_lon = 111412.84 * np.cos(latr) - 93.5 * np.cos(3 * latr)


def enu(r):
    return np.array([(r['longitude'] - lon0) * m_lon,
                     (r['latitude'] - lat0) * m_lat,
                     r['height'] - rows[0]['height']])


def Rb_from_ins(r):
    az, pit, rol = (np.deg2rad(r['azimuth']), np.deg2rad(r['pitch']), np.deg2rad(r['roll']))
    fwd = np.array([np.sin(az) * np.cos(pit), np.cos(az) * np.cos(pit), -np.sin(pit)])
    fwd /= np.linalg.norm(fwd)
    left = np.cross(np.array([0.0, 0.0, 1.0]), fwd)
    left /= np.linalg.norm(left)
    up = np.cross(fwd, left)
    cr, sr = np.cos(rol), np.sin(rol)
    return np.stack([fwd, left, up], 1) @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])


R_cb = quat2R(q) @ M
Tc = []
for r in rows:
    T = np.eye(4)
    T[:3, :3] = Rb_from_ins(r) @ R_cb
    T[:3, 3] = enu(r) + (Rb_from_ins(r) @ t_mount)
    Tc.append(T)
T0 = np.linalg.inv(Tc[0])
lines, steps, prev = [], [], None
for T in Tc:
    t_ = T0 @ T
    lines.append(' '.join('%.10f' % v for v in t_[:3, :].reshape(-1)))
    p = t_[:3, 3]
    if prev is not None:
        steps.append(float(np.linalg.norm(p - prev)))
    prev = p
open(os.path.join(OUT, 'traj.txt'), 'w').write('\n'.join(lines) + '\n')
print('[skel] traj.txt: %d 行 | 每帧位移 中位 %.3f m | 总里程 %.2f m | 高差 %.2f m'
      % (len(lines), float(np.median(steps)), float(sum(steps)),
         float(Tc[-1][:3, 3][2] - Tc[0][:3, 3][2])))

# ---- 硬断言 ----
for sub, ext, n in (('frames', '.png', NFR), ('semantic_ids', '.png', NFR),
                    ('semantic_colors', '.png', NFR)):
    c = len(glob.glob(os.path.join(OUT, sub, '*' + ext)))
    print('[assert] %-16s %d/%d' % (sub, c, n))
    assert c == n, '%s 数量不符' % sub
assert len(open(os.path.join(OUT, 'traj.txt')).read().strip().splitlines()) == NFR
print('SKEL_51_OK')
