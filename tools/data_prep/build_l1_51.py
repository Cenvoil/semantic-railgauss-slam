#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""L1 v9 @ 5_station_bergedorf_5.1 —— 从 1.2 的 build_l1v9.py 适配而来。

适配点（全部为路径/规模差异，逻辑一字未改）：
  * LAB -> 5.1 的 labels.json（26 MB）
  * DATA -> output_1.0_51
  * OUT -> tmp/d51/l1
  * FRAMES -> 100（5.1 也是 100 帧，实测）
  * 删掉末尾针对 1.2 的 1:1 目检裁剪框（BOX 坐标是 1.2 画面上的远场位置，
    5.1 是车站场景，构图完全不同，沿用会看错区域）—— 改为整帧缩略拼图

口径沿用（已在 5.1 上独立验证，见 verify_proj_51b.py / build_skel_51.py）：
  * K 与外参全部从 5.1 自己的 labels.json 读，**不硬编码 1.2 的值**
    （实测两者 fx/fy/cx/cy 与 pose_wrt_parent 逐位相同，但仍然按 5.1 自己的文件读，
     避免"沿用别人结论"）
  * 轴重排 M=[[0,0,1],[-1,0,0],[0,-1,0]]，R=(quat2R(q)@M).T —— 与 proj_canon v2 同源
  * BBOX_MODE=fill / MARK_MODE=box（与 1.2 v9 一致）
"""
import glob
import json
import os

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.spatial import ConvexHull

LAB = "<PROJECT_ROOT>/data_osdar/unpacked_51/5_station_bergedorf_5.1_labels.json"
CAM = "rgb_highres_center"
DATA = "<PROJECT_ROOT>/SGS-SLAM/data/output_1.0_51"
OUT = "<PROJECT_ROOT>/tmp/d51/l1"
DIA = os.path.join(OUT, "diag")
FRAMES = list(range(100))
W, H = 4112, 2504
M = np.array([[0, 0, 1.0], [-1.0, 0, 0], [0, -1.0, 0]])
BBOX_MODE = os.environ.get("BBOX_MODE", "fill")   # v9：默认放弃收窄，回到官方 bbox 出框
MARK_MODE = os.environ.get("MARK_MODE", "box")    # 远场小目标标记：方框（非圆环）

d = json.load(open(LAB))["openlabel"]
sp = d["streams"][CAM]["stream_properties"]["intrinsics_pinhole"]
K = np.array(sp["camera_matrix"]).reshape(3, 4)[:3, :3]
DST = sp.get("distortion_coeffs", [0] * 5)
pose = d["coordinate_systems"][CAM]["pose_wrt_parent"]
t = np.array(pose["translation"], float)
q = np.array(pose["quaternion"], float)
info = json.load(open(os.path.join(DATA, "info_semantic.json")))
name2id = {c["name"]: c["id"] for c in info["classes"]}
id2name = {c["id"]: c["name"] for c in info["classes"]}
lut = np.zeros((256, 3), np.uint8)
for c in info["classes"]:
    if c["id"] < 256:
        lut[c["id"]] = np.array(c["color"][:3], np.uint8)


def quat2R(q):
    x, y, z, w = q
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


R = (quat2R(q) @ M).T
tv = -R @ t


def distort(xn, yn):
    k1, k2, p1, p2, k3 = DST
    r2 = xn * xn + yn * yn
    f = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 ** 3
    return xn * f + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn * xn), yn * f + p1 * (r2 + 2 * yn * yn) + 2 * p2 * xn * yn


def project(P):
    Pc = (R @ np.asarray(P, float).T).T + tv
    z = Pc[:, 2]
    m = z > 0.3
    xn = Pc[m, 0] / z[m]
    yn = Pc[m, 1] / z[m]
    xd, yd = distort(xn, yn)
    return np.stack([K[0, 0] * xd + K[0, 2], K[1, 1] * yd + K[1, 2]], 1), m, z[m]


frames, objs = d["frames"], d["objects"]
FALLBACK = []
ROI = {}
TRACK_ID = int(name2id.get("track", 13))


def apply_layer(im, dr, lay):
    k = lay[0]
    if k == "fill":
        dr.polygon(lay[1], fill=lay[2])
    elif k == "line":
        dr.line(lay[1], fill=lay[2], width=lay[3])
    elif k == "rect":
        dr.rectangle(lay[1], fill=lay[2], outline=255, width=4)
    elif k == "arr":
        arr = lay[1]
        im.paste(Image.fromarray(arr), (0, 0), Image.fromarray((arr > 0).astype(np.uint8) * 255))


def build(fk, stats):
    im = Image.new("L", (W, H), 0)
    dr = ImageDraw.Draw(im)
    fr = frames.get(str(fk)) or frames.get(fk)
    if fr is None:
        return np.asarray(im)
    coarse, fine = [], []
    for oid, od in fr.get("objects", {}).items():
        src = od.get("object_data") or objs.get(oid, {}).get("object_data", {})
        cls = objs.get(oid, {}).get("type") or "?"
        cid = name2id.get(cls, 0)
        p2c = [a for a in (src.get("poly2d") or []) if a.get("coordinate_system") == CAM]
        for a in p2c:
            v = np.array(a["val"], float).reshape(-1, 2)
            if len(v) < 2:
                continue
            pts = [(float(x), float(y)) for x, y in v]
            if a.get("closed", False) and len(v) >= 3:
                fine.append(("fill", pts, int(cid)))
            else:
                fine.append(("line", pts, int(cid), 8))
        if not p2c:
            for a in (src.get("poly3d") or []):
                v = np.array(a["val"], float).reshape(-1, 3)
                uv, m, zz = project(v)
                if len(uv) < 2:
                    continue
                keep = (zz > 1.0) & np.all(np.abs(uv - np.array([W / 2, H / 2])) < np.array([W, H]), axis=1)
                idx = np.where(keep)[0]
                seg, last = [], None
                for i in idx:
                    pt = (float(uv[i, 0]), float(uv[i, 1]))
                    if last and (abs(pt[0] - last[0]) > 600 or abs(pt[1] - last[1]) > 600):
                        if len(seg) >= 2:
                            coarse.append(("line", seg, TRACK_ID, 10))
                        seg = []
                    seg.append(pt)
                    last = pt
                if len(seg) >= 2:
                    coarse.append(("line", seg, TRACK_ID, 10))
        bb = [a for a in (src.get("bbox") or []) if a.get("coordinate_system") == CAM]
        bu = bv = bw = bh = None
        if bb:
            bu, bv, bw, bh = bb[0]["val"][:4]
        for a in (src.get("cuboid") or []):
            val = np.array(a["val"], float)
            if len(val) < 10 or cls in ("switch", "track") or p2c:
                continue
            ctr, qq, sz = val[:3], val[3:7], val[7:10]
            Rc = quat2R(qq)
            off = np.array([[sx, sy, szz] for sx in (-1, 1) for sy in (-1, 1) for szz in (-1, 1)]) * (sz / 2.0)
            corners = ctr + off @ Rc.T
            uv, m, zz = project(corners)
            # 修 1.2 原版的一处越界：project() 内部已按 z>0.3 过滤，
            # 返回的 uv 可能为空。原判据用 m.sum()（过滤前的点数），
            # 当 cuboid 中心落在相机后方时 m.sum()>0 但 uv 为空 -> IndexError。
            # 5.1 的 train 类 cuboid 触发了它（1.2 未触发）。
            # 正确判据是"返回值长度"，不是过滤前的 mask 和。
            _uv1, _m1, _z1 = project(ctr.reshape(1, 3))
            c3u, c3v = (float(_uv1[0, 0]), float(_uv1[0, 1])) if len(_uv1) else (None, None)
            use_bbox = False
            if bu is not None and c3u is not None:
                du, dv = float(c3u - bu), float(c3v - bv)
                thr = max(30.0, 0.35 * float(bw))
                if abs(du) > thr or abs(dv) > 3.0 * float(bh):
                    use_bbox = True
            if use_bbox:
                FALLBACK.append((fk, cls, float(du), float(dv)))
                if BBOX_MODE == "roi":
                    ROI.setdefault(fk, []).append(dict(
                        cls=cls, uuid=oid, bbox=[float(bu), float(bv), float(bw), float(bh)],
                        du=float(du), dv=float(dv), dist_m=float(np.mean(zz)) if len(zz) else -1.0))
                else:
                    x0, y0 = bu - bw / 2, bv - bh / 2
                    coarse.append(("rect", [x0, y0, x0 + bw, y0 + bh], int(cid)))
                continue
            # 同上：角点投影后可能只剩 <8 个（部分角点在相机后方/被滤掉），
            # 判据必须用过滤后的 uv 长度，否则 ConvexHull 会抛错。
            if len(uv) < 8:
                continue
            hull = ConvexHull(uv)
            pts = [(float(uv[i, 0]), float(uv[i, 1])) for i in hull.vertices]
            xa = np.array([p[0] for p in pts]); ya = np.array([p[1] for p in pts])
            ar = 0.5 * abs(np.dot(xa, np.roll(ya, 1)) - np.dot(np.roll(xa, 1), ya))
            if ar > 0.08 * W * H:
                continue
            if bu is not None:
                tmp = Image.new("L", (W, H), 0)
                ImageDraw.Draw(tmp).polygon(pts, fill=int(cid))
                arr = np.asarray(tmp).copy()
                x0 = max(0, int(bu - bw * 0.65)); x1 = min(W, int(bu + bw * 0.65))
                y0 = max(0, int(bv - bh * 0.65)); y1 = min(H, int(bv + bh * 1.65))
                clip = np.zeros_like(arr); clip[y0:y1, x0:x1] = 1
                arr = arr * clip
                if (arr > 0).any():
                    coarse.append(("arr", arr))
            else:
                coarse.append(("fill", pts, int(cid)))
    for lay in coarse:
        apply_layer(im, dr, lay)
    for lay in fine:
        apply_layer(im, dr, lay)
    mask = np.asarray(im).copy()
    np.save(os.path.join(OUT, "mask_%03d.npy" % fk), mask)
    cnt = {int(c): int((mask == c).sum()) for c in np.unique(mask) if c > 0 and c < 250}
    stats[fk] = sorted(cnt.items(), key=lambda kv: -kv[1])
    return mask


def small_marks(mask, lo=100, hi=800):
    """小连通域 → 返回原分辨率外接框 (u0, v0, u1, v1, 颜色, 类别, 面积)，预览图上画方框用。"""
    out = []
    for cid in [int(x) for x in np.unique(mask) if 0 < x < 250]:
        lab, n = ndimage.label(mask == cid)
        for i, sl in enumerate(ndimage.find_objects(lab)):
            a = int((lab[sl] == i + 1).sum())
            if lo <= a <= hi:
                out.append((sl[1].start, sl[0].start, sl[1].stop, sl[0].stop,
                            tuple(int(v) for v in lut[cid]), cid, a))
    return out


def dashed_rect(dr, box, col=(255, 255, 255), dash=16, w=3):
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    for x in range(x0, x1, dash * 2):
        dr.line([x, y0, min(x + dash, x1), y0], fill=col, width=w)
        dr.line([x, y1, min(x + dash, x1), y1], fill=col, width=w)
    for y in range(y0, y1, dash * 2):
        dr.line([x0, y, x0, min(y + dash, y1)], fill=col, width=w)
        dr.line([x1, y, x1, min(y + dash, y1)], fill=col, width=w)


stats = {}
tiles_rgb, tiles_sem, tiles_sem_pure, tiles_ov = [], [], [], []
marks_per_frame, roi_per_frame = [], []
TW = 1400
PAD = 7          # 预览坐标下的外扩像素（保证小目标框看得见）
flist = sorted(os.listdir(os.path.join(DATA, "frames")))
for f in FRAMES:
    mask = build(f, stats)
    marks_per_frame.append(small_marks(mask))
    roi_per_frame.append(ROI.get(f, []))
    rgb = Image.open(os.path.join(DATA, "frames", flist[f])).convert("RGB")
    sem = Image.fromarray(lut[np.clip(mask, 0, 255)])
    ov = Image.blend(rgb, sem, 0.45)
    th = int(round(TW * H / W))
    tiles_rgb.append(rgb.resize((TW, th), Image.LANCZOS))
    tiles_sem.append(sem.resize((TW, th), Image.NEAREST))
    tiles_sem_pure.append(sem.resize((TW, th), Image.NEAREST))
    tiles_ov.append(ov.resize((TW, th), Image.LANCZOS))
    print("frame %3d 掩膜 %.4f%% | 小目标框 %d | ROI(不填像素) %d"
          % (f, (mask > 0).mean() * 100, len(marks_per_frame[-1]), len(roi_per_frame[-1])))

print()
print("=== 覆盖率（v8, BBOX_MODE=%s）===" % BBOX_MODE)
for f in FRAMES:
    tot = W * H
    s = ", ".join("%s(id%d)=%.4f%%" % (id2name.get(k, "?"), k, 100.0 * v / tot) for k, v in stats[f][:10])
    print("frame %3d 类别构成: %s" % (f, s))
print("  掩膜总覆盖率: " + " / ".join("%.4f%%" % ((np.load(os.path.join(OUT, "mask_%03d.npy" % f)) > 0).mean() * 100) for f in FRAMES))

print()
print("=== 与 v7（实心矩形）差异 ===")
totpix = 0
for f in FRAMES:
    if not os.path.exists(os.path.join(OUT, "mask_v7_%03d.npy" % f)):
        print("  frame %3d  (无 v7 备份，跳过对比)" % f)
        continue
    a = np.load(os.path.join(OUT, "mask_v7_%03d.npy" % f)).astype(int)
    b = np.load(os.path.join(OUT, "mask_%03d.npy" % f)).astype(int)
    only7 = int(((a > 0) & (b == 0)).sum())
    only8 = int(((a == 0) & (b > 0)).sum())
    totpix += only7
    print("  frame %3d  v7 有 v8 无 %6d px | v8 有 v7 无 %6d px  覆盖率 %+.4f%%"
          % (f, only7, only8, ((b > 0).mean() - (a > 0).mean()) * 100))
    for cid in sorted(set(list(np.unique(a)) + list(np.unique(b)))):
        if cid == 0:
            continue
        A = int((a == cid).sum()); B = int((b == cid).sum())
        if A != B:
            print("      %-14s v7=%7d -> v8=%7d (%+d)" % (id2name.get(int(cid), "?"), A, B, B - A))
print("  合计被收窄掉的像素: %d px（4 帧，全部来自 bbox 回退对象）" % totpix)

print()
print("=== ROI 清单（写 roi_fallback.json）===")
for r in sorted(FALLBACK, key=lambda z: (z[0], z[2])):
    print("  frame %3d %-14s Δu=%7.1f Δv=%7.1f" % r)
if BBOX_MODE == "roi":
    with open(os.path.join(OUT, "roi_fallback.json"), "w") as fh:
        json.dump({str(k): v for k, v in ROI.items()}, fh, indent=1)
    print("  已写", os.path.join(OUT, "roi_fallback.json"))
else:
    print("  BBOX_MODE=fill：不写 roi_fallback.json（v8 的清单保留在 roi_fallback_v8.json）")

print()
print("=== 小目标框清单 ===")
for f, mk in zip(FRAMES, marks_per_frame):
    for (u0, v0, u1, v1, col, cid, a) in sorted(mk, key=lambda z: -z[6]):
        print("  frame %3d %-14s u%4d~%4d v%4d~%4d 面积%4d rgb=%s" % (f, id2name.get(cid, "?"), u0, u1, v0, v1, a, col))


def stitch(tiles, path, mk, roi, tag, sm=True):
    th = tiles[0].size[1]
    cv = Image.new("RGB", (TW * 2, th * 2), (20, 20, 20))
    dr = ImageDraw.Draw(cv)
    sx, sy = TW / float(W), th / float(H)
    for i, im in enumerate(tiles):
        ox, oy = TW * (i % 2), th * (i // 2)
        cv.paste(im, (ox, oy))
        if sm and MARK_MODE == "box":
            for (u0, v0, u1, v1, col, cid, a) in mk[i]:
                b = [ox + u0 * sx - PAD, oy + v0 * sy - PAD, ox + u1 * sx + PAD, oy + v1 * sy + PAD]
                dr.rectangle(b, outline=(0, 0, 0), width=5)
                dr.rectangle(b, outline=col, width=3)
        elif sm and MARK_MODE == "circle":
            for (u0, v0, u1, v1, col, cid, a) in mk[i]:
                px, py = ox + (u0 + u1) / 2.0 * sx, oy + (v0 + v1) / 2.0 * sy
                r = 9
                dr.ellipse([px - r, py - r, px + r, py + r], outline=(0, 0, 0), width=7)
                dr.ellipse([px - r, py - r, px + r, py + r], outline=col, width=3)
        if sm:
            used = []
            for r in roi[i]:
                bu, bv, bw, bh = r["bbox"]
                box = [ox + (bu - bw / 2) * sx, oy + (bv - bh / 2) * sy,
                       ox + (bu + bw / 2) * sx, oy + (bv + bh / 2) * sy]
                dashed_rect(dr, box)
                lab = "%s ROI" % r["cls"]
                tw = dr.textlength(lab) + 6
                ly = box[1] - 13
                for _ in range(6):
                    cand = [box[0], ly, box[0] + tw, ly + 12]
                    if not any(not (cand[2] < u[0] or cand[0] > u[2] or cand[3] < u[1] or cand[1] > u[3]) for u in used):
                        break
                    ly -= 13
                used.append([box[0], ly, box[0] + tw, ly + 12])
                dr.rectangle([box[0], ly, box[0] + tw, ly + 12], fill=(0, 0, 0))
                dr.text((box[0] + 3, ly + 1), lab, fill=(255, 255, 255))
        dr.rectangle([ox, oy, ox + 240, oy + 32], fill=(0, 0, 0))
        dr.text((ox + 8, oy + 8), "frame %03d" % FRAMES[i], fill=(255, 255, 255))
    cv.save(path)
    print("saved", path, tag)


stitch(tiles_rgb, os.path.join(OUT, "rgb_stitch.png"), marks_per_frame, roi_per_frame, "RGB")
stitch(tiles_sem, os.path.join(OUT, "sem_stitch.png"), marks_per_frame, roi_per_frame, "SEM")
stitch(tiles_ov, os.path.join(OUT, "overlay_stitch.png"), marks_per_frame, roi_per_frame, "OVERLAY")
stitch(tiles_sem_pure, os.path.join(OUT, "sem_pure_stitch.png"), marks_per_frame, roi_per_frame, "SEM_PURE", sm=False)

# ---- 整帧缩略拼图（1:1 裁剪框是 1.2 专用的，5.1 构图不同，不沿用）----
print()
print("L1_51_DONE mode=%s/%s" % (BBOX_MODE, MARK_MODE))
