# -*- coding: utf-8 -*-
"""投影约定轴向自检（skill §20：先在最短链路上钉死方向，再谈别的）。

关键问题：pose_wrt_parent 给的 R 是「相机 → base」还是「base → 相机」？
以及相机的光轴在 base 系里是第几维？

自检方法（不依赖任何下游量）：
  取相机系光轴方向 d_cv = (0,0,1)（optional 约定：z 前 / x 右 / y 下）。
  若 R 是「相机→base」，则 base 系里光轴的表示 = R @ (0,0,1) = R[:,2]。
  反之若 R 是「base→相机」，光轴在 base 系里应是 R.T @ (0,0,1)。

判据（物理）：相机高 3.5098 m 装在车上，且安装外参 t = [0.0802, -0.3329, 3.5098]，
  第三维是"上"（base 系 Z 上，MEMORY 实证：poly3d 轨道点沿第一维 21→117 m = X 前）。
  所以 base 系 = (X 前, Y 左, Z 上)。
  相机朝前 ⇒ 光轴在 base 系应主要沿 **+X**。
  → 哪一列/行主要落在 X 分量，哪种约定就是对的。

另外用 LiDAR 轨面点做交叉判据：轨面点 |z|<0.05，在 base 系沿 X 分布 6~150 m。
  投到图像后应落在**图像下半部**（v > cy），且 v 随距离增大趋向 cy。
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
pose = d['coordinate_systems'][CAM]['pose_wrt_parent']
t = np.array(pose['translation'], float)
q = np.array(pose['quaternion'], float)
sp = d['streams'][CAM]['stream_properties']['intrinsics_pinhole']
K = np.array(sp['camera_matrix'], float).reshape(3, 4)[:3, :3]
D = np.array(sp.get('distortion_coeffs', [0] * 5), float)
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


R = quat2R(q)
print('R(quat) =\n', R.round(6))
print('t =', t.round(6), ' 相机高(t_z) = %.4f' % t[2])
print()
print('=== 自检 1：光轴方向（相机 z 前）===')
e_z = np.array([0, 0, 1.0])
e_y = np.array([0, 1.0, 0])
e_x = np.array([1.0, 0, 0])
a1 = R @ e_z
a2 = R.T @ e_z
print('  R @ (0,0,1)   = %s   <- 若 R 是 cam->base，这是光轴在 base 系' % a1.round(5))
print('  R.T @ (0,0,1) = %s   <- 若 R 是 base->cam，这是光轴在 base 系' % a2.round(5))
print('  base 系 = (X 前, Y 左, Z 上) ⇒ 光轴应主要沿 +X')
for tag, v in (('R@ez', a1), ('R.T@ez', a2)):
    lab = ['X前', 'Y左', 'Z上']
    dom = int(np.argmax(np.abs(v)))
    print('    %-9s = %s  主分量=%s(%+.4f)  %s'
          % (tag, v.round(4).tolist(), lab[dom], v[dom],
             'OK 朝前' if dom == 0 and v[0] > 0 else 'NOT 朝前'))
print('  另：若相机系其实是 (x后 y左 z上) 或 base 系 Z 才是前，则需换轴序 —— 见自检 4')

print('\n=== 自检 1b：四元数两种顺序都试 ===')
for order in ('xyzw', 'wxyz'):
    Rq = quat2R(q if order == 'xyzw' else np.array([q[3], q[0], q[1], q[2]]))
    v = Rq @ e_z
    lab = ['X前', 'Y左', 'Z上']
    dom = int(np.argmax(np.abs(v)))
    print('    order=%-5s R@ez=%s  主=%s(%+.4f)  %s'
          % (order, v.round(4).tolist(), lab[dom], v[dom],
             'OK 朝前' if dom == 0 and v[0] > 0 else 'NOT 朝前'))

print('\n=== 自检 2：相机"下"方向在 base 系（y 下）===')
for tag, v in (('R@ey(下)', R @ e_y), ('R.T@ey', R.T @ e_y)):
    lab = ['X前', 'Y左', 'Z上']
    dom = int(np.argmax(np.abs(v)))
    print('    %-9s = %s  主分量=%s(%+.4f)  %s'
          % (tag, v.round(4).tolist(), lab[dom], v[dom],
             'OK 指向下' if dom == 2 and v[2] < 0 else 'NOT 指向下'))

print('\n=== 自检 3：LiDAR 轨面点的 base 系分布（确认 base 轴义） ===')


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
P = load_pcd(pcds[0])[:, :3]
print('  帧0 点数 =', len(P))
for i, nm in enumerate(['X', 'Y', 'Z']):
    q_ = np.percentile(P[:, i], [1, 50, 99])
    print('    %s: p1=%8.2f p50=%8.2f p99=%8.2f  跨度=%8.2f' % (nm, q_[0], q_[1], q_[2], q_[2] - q_[0]))
rail = P[np.abs(P[:, 2]) < 0.05]
print('  轨面点 |Z|<0.05: n=%d  X∈[%.1f,%.1f] Y∈[%.1f,%.1f]'
      % (len(rail), rail[:, 0].min(), rail[:, 0].max(),
         rail[:, 1].min(), rail[:, 1].max()))

print('\n=== 自检 4：多种候选轴向下的光轴深度（决定性） ===')
print('  P_base 的点在"相机前向"上的分量 = dot(P - t, 光轴_base)')
cands = [('R@ez  (R=cam->base)', R @ e_z),
         ('R.T@ez (R=base->cam)', R.T @ e_z),
         ('R@ex  (相机 x 前?)', R @ e_x),
         ('R.T@ex', R.T @ e_x),
         ('R@ey  (相机 y 前?)', R @ e_y),
         ('R.T@ey', R.T @ e_y)]
for tag, axis in cands:
    depth = (P - t) @ axis
    dm = depth[np.abs(P[:, 2]) < 0.05]
    print('    %-22s 全云深>0.5: %6d/%d (%5.1f%%)  轨面点深>0.5: %6d/%d  深p50=%7.1f'
          % (tag, int((depth > 0.5).sum()), len(P), 100.0 * (depth > 0.5).mean(),
             int((dm > 0.5).sum()), len(dm), np.percentile(depth, 50)))
print('  正确约定应使"轨面点深>0.5"接近 100%（轨面在相机前方）且深度范围合理（几 m~150 m）')
