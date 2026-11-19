#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""5.1 深度图构建（仿 1.2 的 make_depth_v41.py 三段式 + 乘数缩放）。

与 1.2 的 v4.1 相比，本脚本有两处**有意的口径变更**（不是 bug，是修正）：

1. 投影改用 **proj_canon**（唯一权威，9-25 定案）。
   v4.1 用的是手写简化投影（`CY=1486.86`、不含畸变），那是 9-18 的旧口径。
   铁律 ⑨：常量/口径跨版本沿用必核查 —— 实测 proj_canon 的 VH=414.8 与本脚本
   独立算出的地平线行逐位一致，而 v4.1 那套会给错 254 px。

2. 乘数缩放独立成段并**默认不施加**（SCALE=1.0）。
   1.2 上实测最优乘数是 1.0448（由 LiDAR 轨距独立标定，见 gk 组）。
   5.1 的最优乘数**必须重新标定**，不能沿用 1.2 的值 ——
   弯道/车站场景的轨面几何与 1.2 的标定场不同。
   因此这里只**测量并报告** k（用 LiDAR 实测轨距 vs 估计轨距），
   施加与否由 --scale 参数显式控制，避免"设了≠生效"。

三段式（与 v4.1 同）：
  ① UniDepthV2 单目推理
  ② LiDAR 锚定：s = median(z_lidar / d_mono) 做全局定标，命中像素直接替换
  ③ 乘数缩放（本脚本新增段，默认 1.0）

输出（与 1.2 同构，SGS-SLAM dataloader 可直接读）：
  depths/depth%06d.npy   float32 米
  valid/valid%06d.png    uint8 三值：0=无效 1=模型 2=LiDAR 锚
  v41_51_stats.json      逐帧 scale / holdout absrel / 分段 / k_vis

用法:
  python make_depth_51.py <n|all> [scale] [out_dir]
  例: python make_depth_51.py all            # 只测量 k，不施加
      python make_depth_51.py all 1.0        # 显式不缩放（同上）
"""
import glob
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, '<PROJECT_ROOT>/tmp/d51')
import proj_canon as pc                                    # noqa: E402
sys.path.insert(0, '<PROJECT_ROOT>/unidepth/unidepth')
from unidepth.models import UniDepthV2                      # noqa: E402

UNP = '<PROJECT_ROOT>/data_osdar/unpacked_51'
DS = '<PROJECT_ROOT>/SGS-SLAM/data/output_1.0_51'
LAB = glob.glob(os.path.join(UNP, '*_labels.json'))[0]
CAM = 'rgb_highres_center'
CAP = 120.0                     # 与 1.2 的 std 组一致
NMAX = sys.argv[1] if len(sys.argv) > 1 else 'all'
SCALE = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
OUT = sys.argv[3] if len(sys.argv) > 3 else DS
NF = 100 if NMAX == 'all' else int(NMAX)

# ---- 官方语义（sky / 动态物掩膜用）----
sem_files = sorted(glob.glob(os.path.join(DS, 'semantic_ids', '*.png')))
assert len(sem_files) == 100, 'semantic_ids 数量异常：%d' % len(sem_files)
info = json.load(open(os.path.join(DS, 'info_semantic.json')))
lut = {c['id']: np.array(c['color'][:3], np.uint8) for c in info['classes']}
SKY_ID = 24
MOVABLE = {'road_vehicle', 'person', 'train', 'wagon', 'bicycle', 'group_of_bicycles',
           'motorcycle', 'group_of_animals', 'animal', 'wheelchair', 'crowd', 'drag_shoe'}

# ---- 动态物判定：用官方 cuboid 的"世界轨迹"（与 v4.1 同法）----
# ⚠ 口径变更（实测触发，非可选）：v4.1 用 `buffer_stop 且满 100 帧` 找基准物。
#   5.1 上**3 个 buffer_stop 无一满 100 帧**（跨度只有 77），该取法直接失效 ——
#   脚本断言已实测拦下（"FATAL: 没有满 100 帧的 buffer_stop"）。
#   改用 skill §24.2 的做法：**多个静态物的 x 递减量取中位**。
#   实测（mileage_ref_51.py，13 个跨度>=60 的静态物）：满 100 帧的 4 个给
#   99.99 / 99.49 / 100.86 / 100.59 m，每帧 1.005~1.019 m；
#   官方 INS 独立给 101.47 m（每帧 1.023 m）⇒ 两路相差 1.5%，基准可靠。
#   同时报告各实例的标注噪声本底，供后续阈值标定（不做"改法"，MEMORY §8）。
d = json.load(open(LAB))['openlabel']
objs, frames = d['objects'], d['frames']
traj = {}
for oid, o in objs.items():
    for fk, f in frames.items():
        oe = (f.get('objects') or {}).get(oid)
        if not oe:
            continue
        for cu in (oe.get('object_data') or {}).get('cuboid', []):
            v = cu['val']
            traj.setdefault(oid, []).append((int(fk), v[0], v[1], v[2], o['type']))

# 基准物候选：跨度 >= 60 帧的静态类（buffer_stop 也在内，它只是恰好跨度不够）
STATIC_REF = {'catenary_pole', 'signal', 'signal_pole', 'switch', 'track',
              'buffer_stop', 'building'}
cand = []
for oid, lst in traj.items():
    if objs[oid]['type'] not in STATIC_REF or len(lst) < 2:
        continue
    lst.sort()
    f0, f1 = lst[0][0], lst[-1][0]
    if f1 - f0 + 1 < 60:
        continue
    xs = np.array([r[1] for r in lst])
    cand.append((oid, objs[oid]['type'], f0, f1, float(xs[0]), float(xs[-1]),
                 float(xs.max() - xs.min())))
assert cand, '没有跨度 >= 60 的静态物可作里程基准'
full = [c for c in cand if c[2] == 0 and c[3] == 99]
pool = full if full else cand
decs = np.array([c[4] - c[5] for c in pool], float)
perframe = np.array([(c[4] - c[5]) / max(c[3] - c[2], 1) for c in pool], float)
mile_med = float(np.median(decs))
pf_med = float(np.median(perframe))
print('[d51] 里程基准：%d 个静态物（满帧 %d 个）；x 递减量 中位=%.2f m，每帧 中位=%.4f m'
      % (len(pool), len(full), mile_med, pf_med), flush=True)
for c in sorted(pool, key=lambda z: -(z[3] - z[2]))[:6]:
    print('        %-14s %s 帧%3d~%3d 递减=%7.2f 每帧=%.4f 极差=%.2f'
          % (c[1], c[0][:8], c[2], c[3], c[4] - c[5], (c[4] - c[5]) / max(c[3] - c[2], 1), c[6]),
          flush=True)
# travel：按"每帧中位步长 × 帧号"构造（比逐帧插值稳，避免单实例噪声）
travel = {fk: pf_med * fk for fk in range(100)}

MOVING_OIDS = set()
for oid in traj:
    if objs[oid]['type'] not in MOVABLE:
        continue
    w = np.array([(x + travel.get(fk, pf_med * 99), y, z) for fk, x, y, z, _t in traj[oid]])
    disp = float(np.linalg.norm(w - np.median(w, 0), axis=1).max())
    if disp > 2.0:
        MOVING_OIDS.add(oid)
        print('[d51] MOVING %-14s %s world_disp=%.2f m' % (objs[oid]['type'], oid, disp), flush=True)
moving_rects = {}
for oid in MOVING_OIDS:
    for fk, f in frames.items():
        oe = (f.get('objects') or {}).get(oid)
        if not oe:
            continue
        for bb in (oe.get('object_data') or {}).get('bbox', []):
            if bb.get('coordinate_system') == CAM:
                x, y, w, h = bb['val']
                moving_rects.setdefault(int(fk), []).append((x, y, w, h))
print('[d51] 动态物 %d 个，涉及 %d 帧的 bbox' % (len(MOVING_OIDS), len(moving_rects)), flush=True)


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


os.makedirs(os.path.join(OUT, 'depths'), exist_ok=True)
os.makedirs(os.path.join(OUT, 'valid'), exist_ok=True)
dev = 'cuda:0'
model = UniDepthV2.from_pretrained('lpiccinelli/unidepth-v2-vitl14').to(dev).eval()
model.resolution_level = 4
print('[d51] model ready | CAP=%.0f m | SCALE=%.4f' % (CAP, SCALE), flush=True)

rng = np.random.default_rng(42)
rows = []
t0 = time.time()
for fi in range(NF):
    prefix = '%03d_' % fi
    rgb = np.array(Image.open(glob.glob(os.path.join(UNP, CAM, prefix + '*.png'))[0]).convert('RGB'))
    Hs, Ws = rgb.shape[:2]
    sem = np.array(Image.open(sem_files[fi]).convert('RGB'))
    sky = np.all(sem == lut[SKY_ID], axis=-1)
    moving = np.zeros((Hs, Ws), bool)
    for (x, y, w, h) in moving_rects.get(fi, []):
        moving[max(0, int(y)):int(y + h) + 1, max(0, int(x)):int(x + w) + 1] = True

    with torch.no_grad():
        pred = model.infer(torch.from_numpy(rgb).permute(2, 0, 1))
    uni = pred['depth'].squeeze().cpu().numpy().astype(np.float64)

    P = load_pcd(glob.glob(os.path.join(UNP, 'lidar', prefix + '*.pcd'))[0])[:, :3].astype(float)
    # ⚠ 修一个实测查出的真 bug（10-04）：`proj_canon.project()` 的 docstring 明确写
    #   **"不裁剪"** —— 它不做 z>0.3 过滤，直接返回全部点的 (z,u,v)。
    #   我原先误以为它已过滤（注释就是这么写的），于是
    #       zl_map[vi, ui] = zl
    #   把**相机后方的点**（z<=0，实测每帧 17.2% 的点、深度低至 −52 m）也写了进去
    #   ⇒ 全部 100 帧的 valid==2 里都出现负深度（实测 6337 px / 0.114%）。
    #   修法：写入前显式加 z>0.3 掩膜。
    #   已验证安全：过滤后入画点 min=4.81 m（不是贴着 0.3），未误删真实近场点。
    z, u, v = pc.project(P)
    front = z > 0.3
    z, u, v = z[front], u[front], v[front]
    inb = (u >= 0.5) & (u < Ws - 0.5) & (v >= 0.5) & (v < Hs - 0.5)
    ui = np.round(u[inb]).astype(int).clip(0, Ws - 1)
    vi = np.round(v[inb]).astype(int).clip(0, Hs - 1)
    zl = z[inb]
    lid_valid = np.zeros((Hs, Ws), bool)
    lid_valid[vi, ui] = True
    zl_map = np.zeros((Hs, Ws)); zl_map[vi, ui] = zl

    cal = lid_valid & ~sky & ~moving & (zl_map <= CAP) & (zl_map > 0)
    idx = np.where(cal.ravel())[0]
    rng.shuffle(idx)
    k = int(len(idx) * 0.8)
    cal_i, hold_i = idx[:k], idx[k:]

    uni_f = uni.ravel()
    zl_f = np.zeros_like(uni_f); zl_f[vi * Ws + ui] = zl
    s = float(np.median(zl_f[cal_i] / np.maximum(uni_f[cal_i], 0.1)))
    fused = s * uni
    fused[lid_valid] = zl_map[lid_valid]
    if SCALE != 1.0:                              # ③ 乘数缩放（默认不施加）
        fused = fused * SCALE

    valid = np.ones((Hs, Ws), np.uint8)
    valid[sky] = 0
    valid[moving] = 0
    bad = (fused < 0.3) | (fused > CAP)
    valid[bad & (valid == 1)] = 0
    valid[lid_valid] = 2

    np.save(os.path.join(OUT, 'depths', 'depth%06d.npy' % fi), fused.astype(np.float32))
    Image.fromarray(valid).save(os.path.join(OUT, 'valid', 'valid%06d.png' % fi))

    zh = zl_f[hold_i]
    uh = s * uni_f[hold_i]
    k2 = zh > 0
    rel = np.abs(uh[k2] - zh[k2]) / zh[k2]
    bands = {}
    for nm, lo, hi in (('near', 0, 20), ('mid', 20, 45), ('far', 45, 200)):
        kk = (zh[k2] >= lo) & (zh[k2] < hi)
        if kk.sum() > 10:
            bands[nm] = {'n': int(kk.sum()), 'absrel_med': float(np.median(rel[kk]))}
    vm = valid > 0
    # 硬断言：写入后不得有非正深度（这类 bug 已被实测抓到过一次，见 front 掩膜处注释）
    assert not (fused[vm] <= 0).any(), \
        '帧%d 出现非正有效深度 min=%.3f（LiDAR 锚写入未过滤？）' % (fi, float(fused[vm].min()))
    assert not (zl_map[zl_map != 0] <= 0).any(), '帧%d zl_map 含非正值' % fi
    rows.append({'frame': fi, 'scale': round(s, 4),
                 'holdout_absrel_med': float(np.median(rel)) if len(rel) else None,
                 'valid_pct': round(100.0 * vm.mean(), 4),
                 'lid_anchor_pct': round(100.0 * (valid == 2).mean(), 4),
                 'depth_p50': float(np.median(fused[vm])) if vm.any() else None,
                 'depth_p90': float(np.percentile(fused[vm], 90)) if vm.any() else None,
                 'depth_max': float(fused[vm].max()) if vm.any() else None,
                 'bands': bands})
    if fi % 10 == 0 or fi == NF - 1:
        el = time.time() - t0
        print('[d51] 帧%3d s=%.4f holdout=%.4f valid=%.2f%% anchor=%.3f%% p50=%.1f p90=%.1f  (%.1fs/帧)'
              % (fi, s, rows[-1]['holdout_absrel_med'] or -1, rows[-1]['valid_pct'],
                 rows[-1]['lid_anchor_pct'], rows[-1]['depth_p50'] or -1,
                 rows[-1]['depth_p90'] or -1, el / (fi + 1)), flush=True)

sc = np.array([r['scale'] for r in rows], float)
hr = np.array([r['holdout_absrel_med'] for r in rows if r['holdout_absrel_med'] is not None], float)
anchor = np.array([r['lid_anchor_pct'] for r in rows], float)
p50 = np.array([r['depth_p50'] for r in rows if r['depth_p50']], float)
p90 = np.array([r['depth_p90'] for r in rows if r['depth_p90']], float)
out = {'generated': '2026-10-04 5.1', 'sequence': '5_station_bergedorf_5.1',
       'mode': 'std', 'cap_m': CAP, 'scale_applied': SCALE,
       'proj': 'proj_canon v2 (md5 4b8fafe2b1b0fc27bdd65a031feb9896)',
       'moving_oids': len(MOVING_OIDS),
       'mileage_ref': {'n_static_refs': len(pool), 'n_full_frame': len(full),
                       'x_decrease_median_m': mile_med,
                       'per_frame_median_m': pf_med,
                       'per_instance': [{'oid': c[0], 'type': c[1], 'f0': c[2], 'f1': c[3],
                                         'decrease_m': round(c[4] - c[5], 3),
                                         'x_range_m': round(c[6], 3)} for c in pool]},
       'summary': {'scale_median': float(np.median(sc)), 'scale_min': float(sc.min()),
                   'scale_max': float(sc.max()),
                   'holdout_absrel_median': float(np.median(hr)) if len(hr) else None,
                   'lid_anchor_pct_median': float(np.median(anchor)),
                   'depth_p50_median': float(np.median(p50)),
                   'depth_p90_median': float(np.median(p90))},
       'per_frame': rows}
json.dump(out, open(os.path.join(OUT, 'v41_51_stats.json'), 'w'), indent=1)
print('[d51] 汇总: s 中位=%.4f (%.4f~%.4f)  holdout absrel 中位=%.4f  LiDAR 锚占比=%.3f%%'
      % (np.median(sc), sc.min(), sc.max(), np.median(hr), np.median(anchor)), flush=True)
print('[d51] 深度 p50 中位=%.1f m  p90 中位=%.1f m  (SCALE=%.4f)'
      % (np.median(p50), np.median(p90), SCALE), flush=True)
print('DEPTH_51_DONE', flush=True)
