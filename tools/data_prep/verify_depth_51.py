# -*- coding: utf-8 -*-
"""5.1 深度图独立核验（三条腿，skill §14）。

为什么必须独立核验：
  make_depth_51.py 的 holdout absrel 用的是"LiDAR 定标用的那批点"的对偶 20%，
  属同源校验。本脚本改用**三条互相独立的腿**：

  腿1 解析地面：地面像素应满足 v = VH + C/z（内容定律）。
     深度图里"地面类"像素（valid>0 且位于轨道面上）反推出的 z 应与深度值自洽。
     假设：地面是平面、VH/C 已知。⚠ 只在近场可信（≤60 m），远场小角形式失效。

  腿2 地面像素反推相机高：h_imp = d·(v − VH)/fy。
     对平面深度，h_imp 应恒等于相机高（3.5098 m）。这条**只用深度图 + 内参 + 地平线行**，
     完全绕开 LiDAR 外参，是最廉价的第三方。
     数学上 h_imp = h_true / r（r = z_LiDAR/d）⇒ h_imp 偏大 ⇔ 深度偏小。

  腿3 深度直方图与 CAP：检查是否有 clip 堆积（人造上限）。

  另：与 1.2 的深度分布并列对照（注意 5.1 每帧走 1.01 m、1.2 走 0.22 m，
  近场内容不同，不能直接比 p50，但量级应一致）。
"""
import glob
import json
import os
import sys

import numpy as np
from PIL import Image

DS = sys.argv[1] if len(sys.argv) > 1 else '<PROJECT_ROOT>/SGS-SLAM/data/output_1.0_51'
N = int(sys.argv[2]) if len(sys.argv) > 2 else 20
sys.path.insert(0, '<PROJECT_ROOT>/tmp/d51')
import proj_canon as pc                                    # noqa: E402

FX, FY, CX, CY = pc.FX, pc.FY, pc.CX, pc.CY
VH, C = pc.VH_LAW, pc.C_LAW
H_TRUE = 3.50982
print('权威口径：VH=%.1f C=%.1f 相机高=%.4f' % (VH, C, H_TRUE))

# 轨面在图像里的行范围：内容定律 v = VH + C/z，z 取 8~80 m
# 注意方向：v = VH + C/z，z 越大 v 越小（越靠地平线）。取 z∈[8,80] m 并裁到图内。
V_ZMIN, V_ZMAX = 8.0, 80.0
V_NEAR = min(VH + C / V_ZMIN, 2503)   # 近处（z 小）行号大
V_FAR = max(VH + C / V_ZMAX, 0)       # 远处（z 大）行号小
print('轨面行范围（z=%.0f~%.0f m）: v = %.0f ~ %.0f  （图像高 2504）'
      % (V_ZMIN, V_ZMAX, V_FAR, V_NEAR))

ids = sorted(glob.glob(os.path.join(DS, 'depths', '*.npy')))
sel = [ids[i] for i in np.linspace(0, len(ids) - 1, N).astype(int)]
print('抽样 %d 帧\n' % N)

h1, h2, allz, hist_edges = [], [], [], np.arange(0, 130, 5.0)
for p in sel:
    fi = int(os.path.basename(p)[5:11])
    d = np.load(p).astype(np.float64)
    v = np.array(Image.open(os.path.join(DS, 'valid', 'valid%06d.png' % fi)))
    # 只取"轨面行带内 + 有效 + 非 LiDAR 锚"（锚像素是写入值，用它验等于循环论证）
    band = np.zeros_like(v, bool)
    band[int(max(V_FAR, 0)):int(min(V_NEAR, 2503))] = True
    m = (v == 1) & band & (d > 1.0) & (d < 120.0)
    if m.sum() < 500:
        continue
    dv = d[m]
    # 取每个像素的行号
    rows = np.broadcast_to(np.arange(2504)[:, None], d.shape)[m]
    # 腿1：内容定律反解轨面距离，与深度值比
    z_law = C / np.maximum(rows - VH, 1.0)
    h1.append(np.median(z_law / dv))
    # 腿2：地面像素反推相机高
    h2.append(np.median(dv * (rows - VH) / FY))
    allz.append(dv)
    hist_edges  # noqa

allz = np.concatenate(allz)
print('=== 腿1：轨面像素，内容定律距离 / 深度图值 ===')
print('  比值中位 = %.4f   （1.0 = 深度图的轨面距离与几何自洽）' % float(np.median(h1)))
print('  逐帧: %s' % ' '.join('%.3f' % x for x in h1))

print('\n=== 腿2：轨面像素反推相机高 h_imp = d·(v−VH)/fy ===')
h2 = np.array(h2)
print('  h_imp 中位 = %.4f m   真实相机高 = %.4f m   比值 = %.4f'
      % (float(np.median(h2)), H_TRUE, float(np.median(h2)) / H_TRUE))
print('  逐帧: %s' % ' '.join('%.2f' % x for x in h2))
print('  ⇒ h_imp/h_true = h_true/r ⇒ 隐含 r = %.4f'
      % (H_TRUE / float(np.median(h2))))

print('\n=== 腿3：深度直方图（查 clip 堆积）===')
hh, _ = np.histogram(allz, bins=hist_edges)
for i in range(len(hh)):  # 25 档
    pct = 100.0 * hh[i] / allz.size
    bar = '#' * int(pct * 1.5)
    print('  %5.0f-%5.0f m  %8d  %6.3f%%  %s' % (hist_edges[i], hist_edges[i + 1], hh[i], pct, bar))
print('  p50=%.2f p90=%.2f p99=%.2f max=%.2f  (>=119.5m 的占比 %.4f%% = CAP 堆积)'
      % (np.percentile(allz, 50), np.percentile(allz, 90), np.percentile(allz, 99),
         allz.max(), 100.0 * (allz >= 119.5).mean()))
print('  （CAP 堆积应接近 0；若 >1% 说明大量像素被 clip 到上限，是人造远场壳）')

print('\n=== 对照 1.2（同口径，用 1.2 的 VH/C 与相机高）===')
try:
    D12 = '<PROJECT_ROOT>/SGS-SLAM/data/output_1.0_d1_l4'
    ids12 = sorted(glob.glob(os.path.join(D12, 'depths', '*.npy')))
    z12 = []
    for p in [ids12[i] for i in np.linspace(0, len(ids12) - 1, N).astype(int)]:
        d = np.load(p).astype(np.float64)
        v = np.array(Image.open(os.path.join(D12, 'valid', 'valid%06d.png' % int(os.path.basename(p)[5:11]))))
        m = (v == 1) & (d > 1.0) & (d < 120.0)
        if m.sum() > 500:
            z12.append(d[m])
    z12 = np.concatenate(z12)
    print('  1.2: p50=%.2f p90=%.2f p99=%.2f  n=%d' % (
        np.percentile(z12, 50), np.percentile(z12, 90), np.percentile(z12, 99), z12.size))
    print('  5.1: p50=%.2f p90=%.2f p99=%.2f  n=%d' % (
        np.percentile(allz, 50), np.percentile(allz, 90), np.percentile(allz, 99), allz.size))
except Exception as e:
    print('  对照失败：%r' % e)
print('\n注：5.1 每帧走 1.01 m、1.2 走 0.22 m，近场内容不同，p50 不宜直接比；看量级与形状。')
print('VERIFY_DEPTH_51_DONE')
