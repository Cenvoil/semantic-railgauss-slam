import os, glob, json, time
import numpy as np
from PIL import Image
import open3d as o3d

R = os.path.expanduser('~/haichuan')
SRC = R + '/SGS-SLAM/data/output_1.0_d1'
OUT = R + '/haichuan_depthA'
os.makedirs(OUT + '/depths', exist_ok=True)
os.makedirs(OUT + '/valid', exist_ok=True)
LID = R + '/data_osdar/unpacked/lidar'
Rc = np.array([[0.9999621719218765, -0.008527168633861885, -0.001715261021946117],
               [0.00853067694888426, 0.999961514770642, 0.002048542993746481],
               [0.001697726738170886, -0.00206309783896331, 0.9999964306692448]])
Tc = np.array([0.04421604784634205, 0.00122754853356305, 2.078480448887938])
FX = FY = 7267.95450880415
CX, CY = 2056.049238502414, 1232.862908875167
W, H = 4112, 2504
DYN = set(range(1, 13))
EDGES = [0, 9, 12, 15, 20, 25, 30, 40, 55, 80, 120, 200, 400, 1e9]
CTR = np.array([3, 10.4, 13.4, 17.3, 22.4, 27.4, 34.6, 46.9, 66.3, 98, 155, 283, 2000.0])
N = int(os.environ.get('A_N', '100'))

def project(f):
    P = np.asarray(o3d.io.read_point_cloud(sorted(glob.glob(LID + '/%03d_*.pcd' % f))[0]).points)
    b = (Rc.T @ (P - Tc[None, :]).T).T
    zc = b[:, 0]
    u = np.round(FX * (-b[:, 1]) / np.maximum(zc, 1e-6) + CX).astype(np.int64)
    v = np.round(FY * (-b[:, 2]) / np.maximum(zc, 1e-6) + CY).astype(np.int64)
    ok = (zc > 0.5) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    ui, vi, zz = u[ok], v[ok], zc[ok]
    flat = vi * W + ui
    o = np.argsort(-zz)
    zm = np.zeros(W * H, np.float32)
    zm[flat[o]] = zz[o]
    return zm.reshape(H, W)

def load(f):
    return (np.load(SRC + '/depths/depth%06d.npy' % f).astype(np.float32),
            np.array(Image.open(SRC + '/valid/valid%06d.png' % f)),
            np.array(Image.open(SRC + '/semantic_ids/semantic_id%06d.png' % f)))

# ---------- 第一遍：全局曲线 + 逐帧标量 ----------
pooled = [[] for _ in CTR]
af = {}
t0 = time.time()
for f in range(N):
    dep, val, sem = load(f)
    zm = project(f)
    g = np.zeros_like(dep)
    for sh in (3, -3):
        g = np.maximum(g, np.abs(dep - np.roll(dep, sh, 0)))
        g = np.maximum(g, np.abs(dep - np.roll(dep, sh, 1)))
    ok = (zm > 0) & (~np.isin(sem, list(DYN))) & ((g / np.maximum(dep, 1e-3)) < 0.15) & (dep > 0)
    r = np.where(ok, zm / np.maximum(dep, 1e-3), np.nan)
    for i, (lo, hi) in enumerate(zip(EDGES[:-1], EDGES[1:])):
        m = ok & (dep >= lo) & (dep < hi)
        if m.sum() >= 30:
            pooled[i].append(float(np.median(r[m])))
    if f % 20 == 0:
        print('  fit %3d/%d %.0fs' % (f + 1, N, time.time() - t0), flush=True)

cnt = np.array([len(p) for p in pooled])
curve = np.array([np.median(p) if len(p) >= 3 else np.nan for p in pooled])
print('\n各段样本帧数与中位 r：')
for i, (lo, hi) in enumerate(zip(EDGES[:-1], EDGES[1:])):
    print('   %4d-%5.0f m : n_frames=%3d  r=%s' % (lo, hi, cnt[i], ('%.2f' % curve[i]) if curve[i] == curve[i] else 'NA'))

# 只保留"可靠段"（>=25 帧有样本），其余靠插值/保持
reliable = np.where((cnt >= 25) & np.isfinite(curve))[0]
print('可靠段索引:', reliable.tolist())
c = np.full(len(CTR), np.nan)
c[reliable] = curve[reliable]
# 段内线性插值（按 log 深度），两端保持
logc = np.log(np.maximum(CTR, 1e-6))
valid = np.isfinite(c)
c = np.interp(logc, logc[valid], c[valid])
# 强制单调非减（物理上单目压缩随距离单调），再做轻微平滑
c = np.maximum.accumulate(c)
for _ in range(2):
    c[1:-1] = 0.5 * c[1:-1] + 0.25 * c[:-2] + 0.25 * c[2:]
c = np.maximum.accumulate(np.maximum(c, 0.5))
print('\n最终全局曲线（已单调化）：')
for ct, v in zip(CTR, c):
    print('   %6.1f m : %.2f' % (ct, v))

# 逐帧标量：a_f = 加权中位(r / R(d1))，限制在 [0.85, 1.25]
for f in range(N):
    dep, val, sem = load(f)
    zm = project(f)
    ok = (zm > 0) & (~np.isin(sem, list(DYN))) & (dep > 0)
    Rf = np.interp(np.log(np.maximum(dep, 1e-6)), logc, c)
    ratio = zm[ok] / np.maximum(dep[ok] * Rf[ok], 1e-6)
    a = float(np.clip(np.median(ratio[np.abs(np.log(ratio)) < np.log(2.0)]), 0.85, 1.25)) if ok.sum() > 500 else 1.0
    af[f] = a
vals = np.array([af[f] for f in range(N)])
print('\n逐帧标量 a_f: 中位 %.3f  p5 %.3f  p95 %.3f  min %.3f  max %.3f' % (
    np.median(vals), *np.percentile(vals, [5, 95]), vals.min(), vals.max()))

# ---------- 第二遍：写出新深度 ----------
rows = []
for f in range(N):
    dep, val, sem = load(f)
    zm = project(f)
    Rf = np.interp(np.log(np.maximum(dep, 1e-6)), logc, c) * af[f]
    dnew = dep * Rf
    hit = zm > 0
    with np.errstate(divide='ignore', invalid='ignore'):
        rat = np.where(hit, zm / np.maximum(dnew, 1e-3), 1.0)
    write = hit & (rat > 1 / 3.0) & (rat < 3.0)
    dnew = np.where(write, zm, dnew)
    dnew = np.clip(dnew, 0.5, 800.0).astype(np.float32)
    dnew[val == 0] = 0.0
    vm = np.where(write, 2, 1).astype(np.uint8)
    vm[val == 0] = 0
    np.save(OUT + '/depths/depth%06d.npy' % f, dnew)
    Image.fromarray(vm).save(OUT + '/valid/valid%06d.png' % f)
    rows.append({'frame': f, 'hit': int(hit.sum()), 'written': int(write.sum()), 'af': af[f],
                 'med_before': float(np.median(dep[val != 0])), 'med_after': float(np.median(dnew[val != 0]))})
    if f % 20 == 0:
        print('  write %3d/%d %.0fs' % (f + 1, N, time.time() - t0), flush=True)

json.dump({'edges': EDGES, 'ctr': CTR.tolist(), 'curve': c.tolist(), 'cnt': cnt.tolist(),
           'af_median': float(np.median(vals)), 'rows': rows},
          open(OUT + '/depthA_stats.json', 'w'), indent=1)
print('\n完成 %d 帧 -> %s' % (N, OUT))

# ---------- 校验 ----------
print('\n=== 地面模型校验（地面类像素的深度 / 解析地面深度）===')
for f in (0, 50, 99):
    dep, val, sem = load(f)
    new = np.load(OUT + '/depths/depth%06d.npy' % f).astype(np.float32)
    print(' 帧 %03d  %6s %9s %9s %8s %8s' % (f, '行', 'd1', '新深度', 'd1/解析', '新/解析'))
    for v0 in (1400, 1600, 1800, 2000, 2200, 2400):
        sl = (slice(v0, v0 + 60), slice(None))
        m = (sem[sl] == 26) & (val[sl] != 0)
        if m.sum() < 200:
            continue
        z_an = 2.0785 * FY / ((v0 + 30) - CY)
        print('        %6d %9.1f %9.1f %8.2f %8.2f' % (
            v0, np.median(dep[sl][m]), np.median(new[sl][m]),
            np.median(dep[sl][m]) / z_an, np.median(new[sl][m]) / z_an))

print('\n=== LiDAR 一致性（命中像素上新深度 vs 实测，应≈1）===')
for f in (0, 50):
    dep, val, sem = load(f)
    new = np.load(OUT + '/depths/depth%06d.npy' % f).astype(np.float32)
    zm = project(f)
    hit = zm > 0
    for lo, hi in [(12, 20), (20, 30), (30, 45), (45, 70), (70, 120)]:
        m = hit & (dep >= lo) & (dep < hi)
        if m.sum() < 200:
            continue
        print('  帧%03d %3d-%3d m: n=%6d  旧 d1/实测 中位 %.2f -> 新 中位 %.2f' % (
            f, lo, hi, m.sum(), np.median(dep[m] / zm[m]), np.median(new[m] / zm[m])))
