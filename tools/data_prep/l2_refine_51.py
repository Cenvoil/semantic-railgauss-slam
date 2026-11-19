"""L2 v5 @ 5_station_bergedorf_5.1 —— 从 1.2 的 l2_refine_v5.py 适配而来。

适配点（仅路径；算法、闸门、救援策略一字未改）：
  LAB -> 5.1 labels.json / DATA -> output_1.0_51 / L1D -> tmp/d51/l1 / OUT -> tmp/d51/l2

口径说明：
  * K 与相机外参全部从 5.1 自己的 labels.json 读（不硬编码 1.2 值）。
  * 轴重排 M 与 1.2 相同，已在 5.1 上独立验证（verify_proj_51b.py：
    地面 cy_obs 四档近似常数 414.7/416.0/414.6/412.1，残差 std 17.5→4.1）。
  * L1 输入是 5.1 自己跑的 build_l1_51.py 产物（BBOX_MODE=fill/MARK_MODE=box）。
  * 5.1 的实例数 88（1.2 是 28），类别分布也不同：有 train(7)、crowd(1)，
    1.2 没有的类。SAM2 对"大面状目标"（列车）的行为与对细杆不同，
    闸门参数沿用 1.2，但**结论以 5.1 的实际统计为准**，不预设通过率。
"""
import os
import json
import math

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.spatial import ConvexHull

LAB = "<PROJECT_ROOT>/data_osdar/unpacked_51/5_station_bergedorf_5.1_labels.json"
CAM = "rgb_highres_center"
DATA = "<PROJECT_ROOT>/SGS-SLAM/data/output_1.0_51"
CKPT = "<PROJECT_ROOT>/checkpoints/sam2/sam2.1_hiera_large.pt"
CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
L1D = "<PROJECT_ROOT>/tmp/d51/l1"
OUT = "<PROJECT_ROOT>/tmp/d51/l2"
V4D = os.path.join(OUT, "v4_archive")
REV = os.path.join(OUT, "review_v5")
FRAMES = [int(x) for x in os.environ.get("L2_FRAMES", "0,1,2,3,4,5,6,7,29,59,99").split(",")]
W, H, TW = 4112, 2504, 1400
SKIP_SAM = {"switch", "track", "transition", "rail_infra"}
M = np.array([[0, 0, 1.0], [-1.0, 0, 0], [0, -1.0, 0]])
RG = dict(cont=0.60, bleed=0.05, area=120, ratio=0.25, cover=0.55)   # R-geo
RE = dict(area=200, cont=0.85, bleed=0.02, span=0.50)                # R-eat

d = json.load(open(LAB))["openlabel"]
sp = d["streams"][CAM]["stream_properties"]["intrinsics_pinhole"]
K = np.array(sp["camera_matrix"]).reshape(3, 4)[:3, :3]
DST = sp.get("distortion_coeffs", [0] * 5)
pose = d["coordinate_systems"][CAM]["pose_wrt_parent"]
t = np.array(pose["translation"], float); q = np.array(pose["quaternion"], float)
info = json.load(open(os.path.join(DATA, "info_semantic.json")))
name2id = {c["name"]: c["id"] for c in info["classes"]}
id2name = {c["id"]: c["name"] for c in info["classes"]}
lut = np.zeros((256, 3), np.uint8)
for c in info["classes"]:
    lut[c["id"]] = np.array(c["color"][:3], np.uint8)
frames, objs = d["frames"], d["objects"]
flist = sorted(os.listdir(os.path.join(DATA, "frames")))
ddir = os.path.join(DATA, "depths")
DP = sorted(os.listdir(ddir)) if os.path.isdir(ddir) else []
INST = {o: i + 1 for i, o in enumerate(sorted(objs.keys()))}
INST_REG = {str(v): {"uuid": k, "cls": objs[k].get("type")} for k, v in INST.items()}


def quat2R(q):
    x, y, z, w = q
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


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
    xn = Pc[m, 0] / z[m]; yn = Pc[m, 1] / z[m]
    xd, yd = distort(xn, yn)
    return np.stack([K[0, 0] * xd + K[0, 2], K[1, 1] * yd + K[1, 2]], 1), m, z[m]


def footprint(oid, src, cls):
    im = Image.new("L", (W, H), 0); dr = ImageDraw.Draw(im)
    cid = int(name2id.get(cls, 0))
    p2c = [a for a in (src.get("poly2d") or []) if a.get("coordinate_system") == CAM]
    for a in p2c:
        v = np.array(a["val"], float).reshape(-1, 2)
        if len(v) < 2:
            continue
        pts = [(float(x), float(y)) for x, y in v]
        if a.get("closed", False) and len(v) >= 3:
            dr.polygon(pts, fill=cid)
        else:
            dr.line(pts, fill=cid, width=8)
    if p2c:
        return np.asarray(im).copy(), "poly2d"
    bb = [a for a in (src.get("bbox") or []) if a.get("coordinate_system") == CAM]
    bu = bv = bw = bh = None
    if bb:
        bu, bv, bw, bh = bb[0]["val"][:4]
    for a in (src.get("cuboid") or []):
        val = np.array(a["val"], float)
        if len(val) < 10 or cls in ("switch", "track"):
            continue
        ctr, qq, sz = val[:3], val[3:7], val[7:10]
        Rc = quat2R(qq)
        off = np.array([[sx, sy, szz] for sx in (-1, 1) for sy in (-1, 1) for szz in (-1, 1)]) * (sz / 2.0)
        corners = ctr + off @ Rc.T
        uv, m, zz = project(corners)
        if m.sum() < 8:
            continue
        c3u, c3v = project(ctr.reshape(1, 3))[0][0]
        if bu is not None:
            du, dv = float(c3u - bu), float(c3v - bv)
            if abs(du) > max(30.0, 0.35 * float(bw)) or abs(dv) > 3.0 * float(bh):
                x0, y0 = bu - bw / 2, bv - bh / 2
                dr.rectangle([x0, y0, x0 + bw, y0 + bh], fill=cid)
                return np.asarray(im).copy(), "bbox(mismatch)"
        hull = ConvexHull(uv)
        pts = [(float(uv[i, 0]), float(uv[i, 1])) for i in hull.vertices]
        tmp = Image.new("L", (W, H), 0)
        ImageDraw.Draw(tmp).polygon(pts, fill=cid)
        arr = np.asarray(tmp).copy()
        if bu is not None:
            x0 = max(0, int(bu - bw * 0.65)); x1 = min(W, int(bu + bw * 0.65))
            y0 = max(0, int(bv - bh * 0.65)); y1 = min(H, int(bv + bh * 1.65))
            clip = np.zeros_like(arr); clip[y0:y1, x0:x1] = 1
            arr = arr * clip
        if (arr > 0).sum() < 4:
            if bu is not None:
                x0, y0 = bu - bw / 2, bv - bh / 2
                dr.rectangle([x0, y0, x0 + bw, y0 + bh], fill=cid)
                return np.asarray(im).copy(), "bbox(hull_empty)"
            return np.asarray(im).copy(), "empty"
        return np.maximum(np.asarray(im), arr), "hull"
    if bu is not None:
        x0, y0 = bu - bw / 2, bv - bh / 2
        dr.rectangle([x0, y0, x0 + bw, y0 + bh], fill=cid)
        return np.asarray(im).copy(), "bbox"
    return np.asarray(im).copy(), "empty"


def pick_points(pm, box, seed=0, far_ring=False):
    x0, y0, x1, y1 = box
    m = pm[y0:y1, x0:x1] > 0
    h, w = m.shape
    pos, neg = [], []
    if not m.any():
        return np.zeros((0, 2), np.float32), np.zeros(0, np.int32), 0, 0
    dt = ndimage.distance_transform_edt(m)
    ys, xs = np.where(m)
    bx0, bx1, by0, by1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    bw, bh = bx1 - bx0 + 1, by1 - by0 + 1
    if max(bw, bh) > 2.5 * min(bw, bh):
        for tt in (0.15, 0.5, 0.85):
            if bh >= bw:
                yy = int(by0 + tt * (bh - 1))
                win = dt[max(0, yy - 8):yy + 9, :]
                if win.size and win.max() > 0:
                    k = np.argmax(win); yy2, xx2 = np.unravel_index(k, win.shape)
                    pos.append([float(xx2), float(max(0, yy - 8) + yy2)])
            else:
                xx = int(bx0 + tt * (bw - 1))
                win = dt[:, max(0, xx - 8):xx + 9]
                if win.size and win.max() > 0:
                    k = np.argmax(win); yy2, xx2 = np.unravel_index(k, win.shape)
                    pos.append([float(max(0, xx - 8) + xx2), float(yy2)])
    else:
        rad = float(dt.max()); r = max(3.0, 0.5 * rad)
        d2 = dt.copy()
        for _ in range(4):
            if d2.max() <= 0:
                break
            yy, xx = np.unravel_index(np.argmax(d2), d2.shape)
            pos.append([float(xx), float(yy)])
            d2[max(0, int(yy - r)):int(yy + r) + 1, max(0, int(xx - r)):int(xx + r) + 1] = 0
    if not pos:
        pos = [[float((bx0 + bx1) / 2), float((by0 + by1) / 2)]]
    ccx, ccy = (bx0 + bx1) / 2.0, (by0 + by1) / 2.0
    R0 = 0.65 * max(bw, bh) + 6
    rng = np.random.RandomState(seed)
    ring = (1.15, 1.45, 1.8) if not far_ring else (1.15, 1.45, 1.8, 2.2, 2.6)
    for a in rng.uniform(0, 2 * np.pi, 16 if not far_ring else 24):
        for mul in ring:
            px = int(round(ccx + mul * R0 * np.cos(a)))
            py = int(round(ccy + mul * R0 * np.sin(a)))
            if 0 <= px < w and 0 <= py < h and not m[py, px]:
                neg.append([float(px), float(py)]); break
        if len(neg) >= (8 if not far_ring else 14):
            break
    for (px, py) in [(2, 2), (w - 3, 2), (2, h - 3), (w - 3, h - 3)]:
        if len(neg) < (10 if not far_ring else 18) and not m[py, px]:
            neg.append([float(px), float(py)])
    return np.array(pos + neg, np.float32), np.array([1] * len(pos) + [0] * len(neg), np.int32), len(pos), len(neg)


def clean(m, min_area=25, min_hole=250):
    lab, n = ndimage.label(m)
    if n > 1:
        sz = ndimage.sum(m, lab, range(1, n + 1))
        keep = np.zeros_like(m)
        for i, s in enumerate(sz):
            if s >= min_area:
                keep |= (lab == i + 1)
        m = keep
    filled = ndimage.binary_fill_holes(m)
    holes = filled & ~m
    lab, n = ndimage.label(holes)
    if n:
        sz = ndimage.sum(holes, lab, range(1, n + 1))
        for i, s in enumerate(sz):
            if s < min_hole:
                m |= (lab == i + 1)
    return m


def outline(im, mask, col, width=2):
    ed = Image.fromarray((mask * 255).astype(np.uint8)).filter(
        __import__("PIL.ImageFilter", fromlist=["ImageFilter"]).FIND_EDGES).point(lambda v: 255 if v > 20 else 0)
    im.paste(Image.new("RGB", im.size, col), (0, 0), ed)
    return im


def eval_candidate(m, pmc, a_pm, truth_c, dep, y0, x0, x1, y1):
    """候选掩膜：clean + 深度收边（同 v4 流程）→ 指标。"""
    m = clean(m)
    cut = 0
    if dep is not None:
        sub = dep[y0:y1, x0:x1]
        v = sub[np.logical_and(m, sub > 0)]
        if v.size > 50:
            med = float(np.median(v))
            keep = np.logical_and(m, np.logical_or(sub <= 0, np.abs(sub - med) <= 0.15 * med))
            cut = int(m.sum() - keep.sum()); m = keep
    r = max(3, int(0.25 * max(pmc.shape)))
    outside = (ndimage.distance_transform_edt(~pmc) > r)
    bleed = float((m & outside).sum()) / max(1, int(m.sum()))
    ratio = float(m.sum()) / max(1, a_pm)
    cont = float((m & pmc).sum()) / max(1, int(m.sum()))
    eat = 0
    for c2, tm in truth_c.items():
        ov = int((m & tm).sum())
        if ov > 40:
            eat = max(eat, ov)
    yy1, xx1 = np.where(pmc); yy2, xx2 = np.where(m)
    ang = cover = -1.0
    if len(xx1) > 8 and len(xx2) > 8:
        px_, py_ = np.ptp(xx1), np.ptp(yy1)
        mx_, my_ = np.ptp(xx2), np.ptp(yy2)
        if px_ > 3 * py_:
            ang = float(abs(np.log2(max(1e-3, mx_ / max(1.0, px_)))))
            cover = float(mx_ / max(1.0, px_))
        elif py_ > 3 * px_:
            ang = float(abs(np.log2(max(1e-3, my_ / max(1.0, py_)))))
            cover = float(my_ / max(1.0, py_))
    return m, dict(bleed=bleed, ratio=ratio, cont=cont, eat=eat, ang=ang, cover=cover,
                   area=int(m.sum()), cut=cut)


def axis_span(m, pmc):
    """掩膜沿 prompt 主轴的跨度占比（块状 prompt 用长短两边取大）。"""
    yy1, xx1 = np.where(pmc)
    yy2, xx2 = np.where(m)
    if len(xx1) < 8 or len(xx2) < 8:
        return 0.0
    sx = (np.ptp(xx2) + 1) / max(1.0, np.ptp(xx1) + 1)
    sy = (np.ptp(yy2) + 1) / max(1.0, np.ptp(yy1) + 1)
    if np.ptp(xx1) > 3 * np.ptp(yy1):
        return float(sx)
    if np.ptp(yy1) > 3 * np.ptp(xx1):
        return float(sy)
    return float(max(sx, sy))


def save_review(crop, pmc, m, title, path, res_color=(255, 70, 0), band=(255, 255, 0)):
    tile = Image.fromarray(crop.copy())
    tile = outline(tile, pmc, (0, 255, 0))
    t3 = Image.fromarray(np.where(m[..., None], np.array(res_color, np.uint8), crop)).copy()
    t3 = Image.blend(Image.fromarray(crop), t3, 0.55)
    t3 = outline(t3, pmc, (0, 255, 0))
    sc2 = max(1, int(300 / max(tile.size)))
    tile = tile.resize((tile.size[0] * sc2, tile.size[1] * sc2), Image.NEAREST)
    t3 = t3.resize(tile.size, Image.NEAREST)
    sheet = Image.new("RGB", (tile.size[0] * 2 + 8, tile.size[1] + 22), (25, 25, 25))
    sheet.paste(tile, (0, 22)); sheet.paste(t3, (tile.size[0] + 8, 22))
    d2 = ImageDraw.Draw(sheet)
    d2.rectangle([0, 0, sheet.size[0], 20], fill=(0, 0, 0))
    d2.text((4, 4), title, fill=band)
    sheet.save(path)


def main():
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    torch.backends.cuda.matmul.allow_tf32 = True
    model = build_sam2(CFG, CKPT, device="cuda:0")
    predictor = SAM2ImagePredictor(model)
    print("[load] SAM2.1-large ready", flush=True)
    os.makedirs(REV, exist_ok=True)

    review, stat = [], {}
    tiles4, regress_bad = [], 0
    for f in FRAMES:
        fr = frames.get(str(f)) or frames.get(f)
        rgb_np = np.asarray(Image.open(os.path.join(DATA, "frames", flist[f])).convert("RGB"))
        dep = np.asarray(np.load(os.path.join(ddir, DP[f])), np.float32) if DP else None
        l1 = np.load(os.path.join(L1D, "mask_%03d.npy" % f))
        v4p = os.path.join(V4D, "mask_L2_%03d.npy" % f)
        has_v4 = os.path.exists(v4p)      # 仅 11 个交付帧有 v4 归档；新帧跳过 v4 对照
        if has_v4:
            mask4 = np.load(v4p)
            conf4 = np.load(os.path.join(V4D, "conf_L2_%03d.npy" % f))
            inst4 = np.load(os.path.join(V4D, "inst_L2_%03d.npy" % f))
            # v4 的 ok 对象实例号：有 conf==1 像素者（fb/skip 对象只有 conf 0/2 像素）
            v4ok_ids = set(int(v) for v in np.unique(inst4[(conf4 == 1) & (inst4 > 0)]))
            prot_all = inst4 > 0
        else:
            mask4 = conf4 = inst4 = None
            v4ok_ids = set()
            prot_all = np.zeros((H, W), bool)
        out = l1.copy()
        conf = np.zeros((H, W), np.uint8)
        items = []
        for oid, od in (fr or {}).get("objects", {}).items():
            src = od.get("object_data") or objs.get(oid, {}).get("object_data", {})
            cls = objs.get(oid, {}).get("type") or "?"
            pm, kind = footprint(oid, src, cls)
            if (pm > 0).sum() < 20:
                continue
            items.append(dict(oid=oid, cls=cls, cid=int(name2id.get(cls, 0)), pm=pm, kind=kind))
        for it in items:
            pm = it["pm"] > 0
            conf[pm & (l1 == it["cid"])] = 2 if it["kind"] == "poly2d" else 0
        truth = {}
        LINE_CLS = {"track", "transition", "rail_infra", "cable_wire"}
        for it in items:
            if it["kind"] == "poly2d" and it["cls"] not in LINE_CLS:
                truth.setdefault(it["cid"], np.zeros((H, W), bool))
                truth[it["cid"]] |= (it["pm"] > 0)
        order = sorted(items, key=lambda z: (z["kind"] == "poly2d", -(z["pm"] > 0).sum()))
        n_ok = n_fb = n_skip = n_rescue = 0
        records = []
        for it in order:
            pm = it["pm"] > 0
            cid = it["cid"]
            idx = INST.get(it["oid"], 0)
            ys, xs = np.where(pm)
            bx0, bx1, by0, by1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
            a_pm = int(pm.sum())
            if it["cls"] in SKIP_SAM:
                n_skip += 1
                records.append(dict(state="skip", oid=it["oid"], idx=idx, cid=cid,
                                    kind=it["kind"], pm=pm))
                continue
            long_side = max(bx1 - bx0, by1 - by0)
            padf = 0.15 if long_side <= 320 else 0.35
            padx = max(6.0, padf * (bx1 - bx0)); pady = max(6.0, padf * (by1 - by0))
            side_min = int(min(640, max(96, long_side * 2.2))) if long_side <= 320 else 160
            side = max(bx1 - bx0 + 2 * padx, by1 - by0 + 2 * pady, side_min)
            cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
            x0 = int(max(0, cx - side / 2)); x1 = int(min(W, cx + side / 2))
            y0 = int(max(0, cy - side / 2)); y1 = int(min(H, cy + side / 2))
            box = [x0, y0, x1, y1]
            crop = rgb_np[y0:y1, x0:x1]
            pmc = pm[y0:y1, x0:x1]
            truth_c = {c2: tm[y0:y1, x0:x1] for c2, tm in truth.items() if c2 != cid}
            try:
                pts, lbl, npos, nneg = pick_points(it["pm"], box, seed=f)
                predictor.set_image(crop)
                masks, scores, _ = predictor.predict(point_coords=pts, point_labels=lbl, multimask_output=True)
                masks = [np.asarray(m).astype(bool) for m in masks]
                scores = np.asarray(scores, float)
                ious = np.array([np.logical_and(m, pmc).sum() / max(1, np.logical_or(m, pmc).sum()) for m in masks])
                k = int(np.argmax(ious)) if float(np.max(ious)) >= 0.30 else int(np.argmax(scores))
                m = clean(masks[k])
                cut = 0
                if dep is not None:
                    sub = dep[y0:y1, x0:x1]
                    v = sub[np.logical_and(m, sub > 0)]
                    if v.size > 50:
                        med = float(np.median(v))
                        keep = np.logical_and(m, np.logical_or(sub <= 0, np.abs(sub - med) <= 0.15 * med))
                        cut = int(m.sum() - keep.sum()); m = keep
                r = max(3, int(0.25 * max(pmc.shape)))
                outside = (ndimage.distance_transform_edt(~pmc) > r)
                bleed = float((m & outside).sum()) / max(1, int(m.sum()))
                ratio = float(m.sum()) / max(1, a_pm)
                cont = float((m & pmc).sum()) / max(1, int(m.sum()))
                eat = 0
                for c2, tm in truth_c.items():
                    ov = int((m & tm).sum())
                    if ov > 40:
                        eat = max(eat, ov)
                sc, iou = float(scores[k]), float(ious[k])
                ang = cover = -1.0
                yy1, xx1 = np.where(pmc); yy2, xx2 = np.where(m)
                if len(xx1) > 8 and len(xx2) > 8:
                    px_, py_ = np.ptp(xx1), np.ptp(yy1)
                    mx_, my_ = np.ptp(xx2), np.ptp(yy2)
                    if px_ > 3 * py_:
                        ang = float(abs(np.log2(max(1e-3, mx_ / max(1.0, px_)))))
                        cover = float(mx_ / max(1.0, px_))
                    elif py_ > 3 * px_:
                        ang = float(abs(np.log2(max(1e-3, my_ / max(1.0, py_)))))
                        cover = float(my_ / max(1.0, py_))
                bad = []
                if it["kind"] == "poly2d":
                    if ratio < 0.45: bad.append("ratio%.2f" % ratio)
                    if iou < 0.45: bad.append("iou%.2f" % iou)
                    if sc < 0.45: bad.append("score%.2f" % sc)
                    if eat: bad.append("eat%d" % eat)
                else:
                    if cont < 0.55: bad.append("cont%.2f" % cont)
                    if ratio > 2.5: bad.append("ratio%.2f" % ratio)
                    if bleed > 0.25: bad.append("bleed%.2f" % bleed)
                    if eat: bad.append("eat%d" % eat)
                    if sc < 0.40: bad.append("score%.2f" % sc)
                    if int(m.sum()) < 60: bad.append("tiny%d" % int(m.sum()))
                    if ang >= 0 and cover < 0.30 and ang > 1.0:
                        bad.append("frag axis%.2f/cov%.2f" % (ang, cover))
                if bad:
                    # ================= pass2：救援（仅非 poly2d）=================
                    # 目检否决（43→38→35 条三轮逐条过目）：
                    #   a8102c4c（bbox mismatch，框内主体是灌木）所有帧、所有救援模式的掩膜
                    #   全是灌木/树影碎片（9 条记录）→ 整对象禁用救援，保持 L1 官方 bbox 填充；
                    #   (f005, 5786b934) geo 掩膜抓到涂鸦墙、(f099, a7a4adb9) eat_sub 抓水泥柱边 → 该帧禁用。
                    rescue_off = (it["oid"][:8] == "a8102c4c"
                                  or it["oid"][:8] == "c0c723c3"
                                  or (f, it["oid"][:8]) in {(5, "5786b934"), (99, "a7a4adb9")})
                    # c0c723c3（9-19 盆地二分）：其救援掩膜（沿隔音墙大杆，3077~4191px）令 SLAM
                    # 优化在帧 2~4 出现双盆地（4 帧探针 26.13×2 vs 26.54/26.63；单独回退即单簇
                    # 26.58~26.79；8 帧下对应 20.50 坏盆）——语义上目检合格但动力学不稳，整对象回退。
                    resc = None
                    if it["kind"] != "poly2d" and not rescue_off:
                        cand = []
                        for ki in range(len(masks)):
                            mi, met = eval_candidate(masks[ki], pmc, a_pm, truth_c, dep, y0, x0, x1, y1)
                            cand.append((ki, mi, met, float(scores[ki]), float(ious[ki])))
                        # R-geo：官方几何一致优先于模型自评
                        best = None
                        for ki, mi, met, sk, io in cand:
                            okg = (met["cont"] >= RG["cont"] and met["bleed"] <= RG["bleed"]
                                   and met["eat"] == 0 and met["area"] >= RG["area"]
                                   and met["ratio"] >= RG["ratio"]
                                   and not (met["ang"] >= 0 and met["ang"] > 1.0 and met["cover"] < RG["cover"]))
                            if okg:
                                key = (round(io, 3), round(met["cont"], 3))
                                if best is None or key > best[0]:
                                    best = (key, ki, mi, met, sk, io)
                        if best is not None:
                            resc = dict(mode="geo", m=best[2], met=best[3], sc=best[4], iou=best[5])
                        # R-eat：减去他类官方像素后，剩余完整落壳内 + 主轴跨度足够
                        # 目检修正（43 条救援逐条过目后）：
                        #   (a) bbox(mismatch) 的框内碎片几乎全是灌木/背景（真身是框顶小红白标）→ 整类禁用
                        #       （f0/f1/f3/f6/f7 的 a8102c4c 五条全部否决）；
                        #   (b) 逐条目检否决表（frame, uuid8）：f005 5786b934 抓墙面竖纹、
                        #       f099 a7a4adb9 抓左侧水泥柱边 —— 这两条是 review 图上人眼确认的错掩膜。
                        eat_ok_kind = it["kind"] != "bbox(mismatch)"
                        eat_black = (f, it["oid"][:8]) in {(5, "5786b934"), (99, "a7a4adb9")}
                        if resc is None and eat_ok_kind and not eat_black:
                            tm_all = None
                            for c2, tm in truth_c.items():
                                tm_all = tm if tm_all is None else (tm_all | tm)
                            for ki, mi, met, sk, io in sorted(cand, key=lambda z: -z[4]):
                                mi2 = (mi & ~tm_all) if (tm_all is not None and (mi & tm_all).any()) else mi
                                mi2 = clean(mi2)
                                if int(mi2.sum()) < RE["area"]:
                                    continue
                                cont2 = float((mi2 & pmc).sum()) / max(1, int(mi2.sum()))
                                out2 = (ndimage.distance_transform_edt(~pmc) > max(3, int(0.25 * max(pmc.shape))))
                                bleed2 = float((mi2 & out2).sum()) / max(1, int(mi2.sum()))
                                eat2 = 0
                                for c2, tm in truth_c.items():
                                    ov = int((mi2 & tm).sum())
                                    if ov > 40:
                                        eat2 = max(eat2, ov)
                                span2 = axis_span(mi2, pmc)
                                if cont2 >= RE["cont"] and bleed2 <= RE["bleed"] and eat2 == 0 and span2 >= RE["span"]:
                                    resc = dict(mode="eat_sub", m=mi2,
                                                met=dict(bleed=bleed2, ratio=round(float(mi2.sum()) / max(1, a_pm), 3),
                                                         cont=cont2, eat=0, ang=met["ang"], cover=met["cover"],
                                                         area=int(mi2.sum()), cut=0),
                                                sc=sk, iou=io)
                                    break
                        # R-retry：换种子 + 2.2/2.6 倍强负点环重试
                        if resc is None:
                            pts2, lbl2, _, _ = pick_points(it["pm"], box, seed=f + 100, far_ring=True)
                            predictor.set_image(crop)
                            masks2, scores2, _ = predictor.predict(point_coords=pts2, point_labels=lbl2, multimask_output=True)
                            masks2 = [np.asarray(mm).astype(bool) for mm in masks2]
                            scores2 = np.asarray(scores2, float)
                            ious2 = np.array([np.logical_and(mm, pmc).sum() / max(1, np.logical_or(mm, pmc).sum()) for mm in masks2])
                            for ki in range(len(masks2)):
                                mi, met = eval_candidate(masks2[ki], pmc, a_pm, truth_c, dep, y0, x0, x1, y1)
                                sk, io = float(scores2[ki]), float(ious2[ki])
                                okg = (met["cont"] >= RG["cont"] and met["bleed"] <= RG["bleed"]
                                       and met["eat"] == 0 and met["area"] >= RG["area"]
                                       and met["ratio"] >= RG["ratio"]
                                       and not (met["ang"] >= 0 and met["ang"] > 1.0 and met["cover"] < RG["cover"]))
                                gate_ok = (met["cont"] >= 0.55 and met["ratio"] <= 2.5 and met["bleed"] <= 0.25
                                           and met["eat"] == 0 and sk >= 0.40 and met["area"] >= 60)
                                if okg or gate_ok:
                                    resc = dict(mode="retry", m=mi, met=met, sc=sk, iou=io)
                                    break
                    if resc is not None:
                        # 保护集：conf4>0 的像素全是他类 ok(SAM2)/官方(poly2d) 像素
                        # （救援对象必为 v4-fb，自身所有像素 conf4==0，故 conf4>0 即他类）
                        # 新帧无 v4 归档 → 用当前帧已完成写盘的 conf（早处理对象写入的 1/2）
                        prot_full = (conf4 if has_v4 else conf) > 0
                        mm = resc["m"].copy()
                        mm &= ~prot_full[y0:y1, x0:x1]
                        mm = clean(mm)
                        if int(mm.sum()) >= 60:
                            n_rescue += 1
                            n_ok += 1
                            erase = pm & (out == cid) & ((conf4 if has_v4 else conf) == 0)
                            out[erase] = 0; conf[erase] = 0
                            full = np.zeros((H, W), bool)
                            full[y0:y1, x0:x1] = mm
                            out[full] = cid; conf[full] = 1
                            records.append(dict(state="ok", oid=it["oid"], idx=idx, cid=cid, kind=it["kind"],
                                                pm=pm, box=box, m=mm, cut=resc["met"].get("cut", 0),
                                                score=round(resc["sc"], 3), iou=round(resc["iou"], 3),
                                                rescue=resc["mode"]))
                            review.append(dict(frame=f, cls=it["cls"], uuid=it["oid"][:8],
                                               reason="rescue:" + resc["mode"], v4reason="+".join(bad),
                                               score=round(resc["sc"], 3), iou=round(resc["iou"], 3),
                                               ratio=round(resc["met"]["ratio"], 2), bleed=round(resc["met"]["bleed"], 3),
                                               cont=round(resc["met"]["cont"], 3), ang=resc["met"]["ang"],
                                               cover=resc["met"]["cover"], eat=resc["met"]["eat"], kind=it["kind"],
                                               box=box, area_pm=a_pm, area_sam=int(mm.sum()),
                                               cut=resc["met"].get("cut", 0)))
                            save_review(crop, pmc, mm,
                                        "f%03d %s %s | RESCUE-%s (v4: %s) | prompt(绿)/SAM2(橙)" %
                                        (f, it["cls"], it["oid"][:8], resc["mode"], "+".join(bad)),
                                        os.path.join(REV, "rescue_f%03d_%s_%s.png" % (f, it["cls"], it["oid"][:8])))
                            continue
                    n_fb += 1
                    records.append(dict(state="fb", oid=it["oid"], idx=idx, cid=cid, kind=it["kind"],
                                        pm=pm, reason="+".join(bad)))
                    review.append(dict(frame=f, cls=it["cls"], uuid=it["oid"][:8], reason="v4fb:" + "+".join(bad),
                                       score=round(sc, 3), iou=round(iou, 3), ratio=round(ratio, 2),
                                       bleed=round(bleed, 3), cont=round(cont, 3),
                                       ang=round(ang, 2), cover=round(cover, 2), eat=eat, kind=it["kind"], box=box,
                                       area_pm=a_pm, area_sam=int(m.sum()), cut=cut))
                else:
                    n_ok += 1
                    erase = pm & (out == cid)
                    out[erase] = 0; conf[erase] = 0
                    full = np.zeros((H, W), bool)
                    full[y0:y1, x0:x1] = m
                    out[full] = cid; conf[full] = 1
                    records.append(dict(state="ok", oid=it["oid"], idx=idx, cid=cid, kind=it["kind"],
                                        pm=pm, box=box, m=m, cut=cut, score=round(sc, 3), iou=round(iou, 3)))
                    if it["kind"] == "poly2d":
                        out[pm] = cid; conf[pm] = 2
            except Exception as ex:
                n_fb += 1
                records.append(dict(state="fb", oid=it["oid"], idx=idx, cid=cid, kind=it["kind"],
                                    pm=pm, reason="exception"))
                review.append(dict(frame=f, cls=it["cls"], uuid=it["oid"][:8], reason="exception:%s" % str(ex)[:70],
                                   score=-1, iou=-1, ratio=-1, bleed=-1, eat=-1, kind=it["kind"], box=box,
                                   area_pm=a_pm, area_sam=-1, cut=0))
        inst = np.zeros((H, W), np.uint16)
        for rec in records:
            if rec["idx"] <= 0:
                continue
            if rec["state"] == "ok":
                x0, y0, x1, y1 = rec["box"]
                full = np.zeros((H, W), bool)
                full[y0:y1, x0:x1] = rec["m"]
                if rec["kind"] == "poly2d":
                    full |= rec["pm"]
                inst[full] = rec["idx"]
            else:
                inst[rec["pm"] & (out == rec["cid"])] = rec["idx"]
        np.save(os.path.join(OUT, "inst_L2_%03d.npy" % f), inst)
        np.save(os.path.join(OUT, "mask_L2_%03d.npy" % f), out)
        np.save(os.path.join(OUT, "conf_L2_%03d.npy" % f), conf)
        # ===== 回归对照：conf4>0 像素（ok SAM2 + 官方 poly2d）的类别不得变；conf2 不得变 =====
        if has_v4:
            pos4 = conf4 > 0
            cls_bad = int((pos4 & (out != mask4)).sum())
            inst_fix = int((pos4 & (inst != inst4)).sum())   # v4 fb 重放偷像素的修正量（不计回归）
            changed = int((mask4 != out).sum())
            conf2_same = bool(np.array_equal(conf4 == 2, conf == 2))
        else:
            cls_bad = inst_fix = changed = -1
            conf2_same = True   # 新帧无 v4 对照，仅本帧自洽（poly2d conf2 由 init+ok 重放保证）
        obj_px = int((out > 0).sum() - (out == 255).sum())
        stat[f] = dict(ok=n_ok, fallback=n_fb, skip=n_skip, rescue=n_rescue,
                       cov_L1=round(float((l1 > 0).mean() * 100), 4),
                       cov_L2=round(float((out > 0).mean() * 100), 4),
                       id255=int((out == 255).sum()),
                       obj_px=obj_px, inst_px=int((inst > 0).sum()),
                       n_inst=int(len(np.unique(inst)) - 1),
                       cls={id2name.get(int(c), str(c)): int((out == c).sum()) for c in np.unique(out) if c > 0},
                       regress_cls_px_diff=cls_bad, inst_attribution_fixed=inst_fix,
                       mask_vs_v4_diff=changed, conf2_unchanged=conf2_same)
        if has_v4:
            regress_bad += cls_bad
        layer = {}
        for rec in records:
            layer[rec["oid"][:8]] = rec["state"] + ("*%s" % rec["rescue"] if rec.get("rescue") else "")
        stat[f]["obj_layer"] = layer
        if f in (0, 29, 59, 99):
            tiles4.append((f, Image.fromarray(lut[np.clip(out, 0, 255)])))
        print("frame %3d ok=%d(含rescue%d) fb=%d skip=%d | L2 %.4f%% | 类别层回归 %s px | inst 修正 %s px | conf2 %s"
              % (f, n_ok, n_rescue, n_fb, n_skip, stat[f]["cov_L2"],
                 cls_bad if has_v4 else "n/a", inst_fix if has_v4 else "n/a",
                 "不变" if conf2_same else "变化!"), flush=True)

    # 交付用 4 帧(0/29/59/99) 拼接
    th = int(round(TW * H / W))
    cv = Image.new("RGB", (TW * 2, th * 2), (20, 20, 20))
    dr = ImageDraw.Draw(cv)
    for i, (f, im) in enumerate(tiles4):
        t = im.resize((TW, th), Image.NEAREST)
        cv.paste(t, (TW * (i % 2), th * (i // 2)))
        dr.rectangle([TW * (i % 2), th * (i // 2), TW * (i % 2) + 240, th * (i // 2) + 32], fill=(0, 0, 0))
        dr.text((TW * (i % 2) + 8, th * (i // 2) + 8), "frame %03d" % f, fill=(255, 255, 255))
    cv.save(os.path.join(OUT, "sem_L2_stitch_4frames.png"))
    print("saved", os.path.join(OUT, "sem_L2_stitch_4frames.png"), flush=True)

    json.dump({"stat": stat, "review": review, "inst_registry": INST_REG},
              open(os.path.join(OUT, "l2_stats_v5.json"), "w"), ensure_ascii=False, indent=1)
    resc_list = [r for r in review if r["reason"].startswith("rescue:")]
    fb_list = [r for r in review if r["reason"].startswith("v4fb:")]
    print("\n=== 救援成功 %d 条（逐条目检 review_v5/） ===" % len(resc_list))
    for r in resc_list:
        print("  f%03d %-14s %s %-8s v4[%s] score=%s iou=%s cont=%s ratio=%s pm=%d sam=%d" %
              (r["frame"], r["cls"], r["uuid"], r["reason"][7:], r["v4reason"], r["score"], r["iou"],
               r["cont"], r["ratio"], r["area_pm"], r["area_sam"]))
    print("\n=== 仍拦截 %d 条（保持 L1 几何 conf=0） ===" % len(fb_list))
    for r in fb_list:
        print("  f%03d %-14s %s %-30s score=%s iou=%s cont=%s pm=%d sam=%d" %
              (r["frame"], r["cls"], r["uuid"], r["reason"][5:], r["score"], r["iou"], r["cont"],
               r["area_pm"], r["area_sam"]))
    print("\n类别层回归差异合计 = %d px（应 = 0；conf4>0 像素的类别不变）" % regress_bad)
    print("L2REFINE_V5_DONE")


if __name__ == "__main__":
    main()
