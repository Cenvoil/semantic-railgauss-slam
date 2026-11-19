#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""L2 视频层 v4：首见注册（全帧扫描）+ 分段传播，绕开 dtype 崩溃。
崩溃根因（vm3/vm3b 定位）：SAM2 video predictor 从 frame0 传播时，"首见帧>0 才加掩膜"的
对象在起始帧无条件 → F.linear 报 BFloat16/Float。SAM2 官方交互式用法是
「加一批对象 → propagate(start_frame_idx=...)」，本脚本按首见帧分组做分段传播：
  段 i：给组 i 的对象在首见帧加掩膜 → 从该帧传播 → 下一组首见帧处 break；
  组 i+1 加掩膜后从其首见帧继续 —— 任何时刻传播起点上所有对象都有条件。
合成逻辑与 vm3 相同：官方上界钳制（poly2d+8px / bbox+10%）→ IoU<0.30 或空 → 官方足迹兜底；
无标注帧不输出。输出 vm4_{sem,inst}_%03d.npy + vm4_meta.json。
"""
import os
import sys
import json

import numpy as np
import torch
sys.path.insert(0, "<PROJECT_ROOT>/tmp/d1/l2")
import l2_refine as L2
import scipy.ndimage as ndi
from sam2.build_sam import build_sam2_video_predictor

JPG = "<PROJECT_ROOT>/tmp/d1/l2/frames_jpg"
OUT = "<PROJECT_ROOT>/tmp/d1/l2"
W, H = L2.W, L2.H
ANCHOR_EVERY = 10
IOU_MIN = 0.30
BBOX_PAD = 0.10
POLY_DIL = 8

n_jpg = len([p for p in os.listdir(JPG) if p.endswith(".jpg")])
print("JPEG 帧数 =", n_jpg, flush=True)

pred = build_sam2_video_predictor(L2.CFG, L2.CKPT, device="cuda:0")
state = pred.init_state(video_path=JPG, offload_video_to_cpu=True, offload_state_to_cpu=True)
print("[init] ok", flush=True)


def get_foot(a, oid):
    fra = L2.frames.get(str(a)) or L2.frames.get(a)
    od = (fra or {}).get("objects", {}).get(oid) if fra is not None else None
    if od is None:
        return None, None
    src = od.get("object_data") or L2.objs.get(oid, {}).get("object_data", {})
    cls = L2.objs.get(oid, {}).get("type") or "?"
    pm, kind = L2.footprint(oid, src, cls)
    if (pm > 0).sum() < 20:
        return None, None
    return pm, kind


def upper_bound(a, oid, kind):
    fra = L2.frames.get(str(a)) or L2.frames.get(a)
    od = (fra or {}).get("objects", {}).get(oid) if fra is not None else None
    if od is None:
        return None
    src = od.get("object_data") or L2.objs.get(oid, {}).get("object_data", {})
    if kind != "poly2d":
        bb = [a2 for a2 in (src.get("bbox") or []) if a2.get("coordinate_system") == L2.CAM]
        if bb:
            bu, bv, bw, bh = [float(x) for x in bb[0]["val"][:4]]
            ub = np.zeros((H, W), bool)
            x0 = max(0, int(bu - bw / 2 * (1 + 2 * BBOX_PAD))); x1 = min(W, int(bu + bw / 2 * (1 + 2 * BBOX_PAD)))
            y0 = max(0, int(bv - bh / 2 * (1 + 2 * BBOX_PAD))); y1 = min(H, int(bv + bh / 2 * (1 + 2 * BBOX_PAD)))
            ub[y0:y1, x0:x1] = True
            return ub
    pm, _ = get_foot(a, oid)
    if pm is None:
        return None
    return ndi.binary_dilation(pm > 0, iterations=POLY_DIL)


# —— 对象与首见帧（全帧扫描，不再限于锚定帧）——
order = []
for oid, od in L2.objs.items():
    cls = od.get("type") or "?"
    if cls in ("track", "switch", "transition", "cable_wire"):
        continue
    order.append((oid, cls))
first_frame = {}
for oid, cls in order:
    for a in range(n_jpg):
        pm, kind = get_foot(a, oid)
        if pm is not None:
            first_frame[oid] = a
            break
groups = {}
for oid, cls in order:
    if oid in first_frame:
        groups.setdefault(first_frame[oid], []).append(oid)
gframes = sorted(groups)
print("[objs] %d 个可跟踪对象，首见帧分组：%s" %
      (sum(len(v) for v in groups.values()), {g: len(groups[g]) for g in gframes}), flush=True)

objs_idx = {}
k = 0
for g in gframes:
    for oid in groups[g]:
        k += 1
        objs_idx[k] = (oid, L2.objs.get(oid, {}).get("type") or "?")

# —— 分段传播：每段起点 = 组首见帧；所有已注册对象都有条件 ——
oid2kk = {v[0]: kk for kk, v in objs_idx.items()}
stored = {}          # frame -> {kk: (ys, xs)}
n_anchor = 0
for gi, g in enumerate(gframes):
    nxt = gframes[gi + 1] if gi + 1 < len(gframes) else n_jpg
    for oid in groups[g]:
        pm, _ = get_foot(g, oid)
        pred.add_new_mask(state, g, oid2kk[oid], torch.from_numpy(pm > 0))
        n_anchor += 1
    for fidx, obj_ids, masks in pred.propagate_in_video(state, start_frame_idx=g):
        if fidx >= nxt:
            break
        d = stored.setdefault(fidx, {})
        for j, oid in enumerate(obj_ids):
            kk = int(oid)
            mj = masks[j]
            mk = ((mj > 0).cpu().numpy() if hasattr(mj, "cpu") else (np.asarray(mj) > 0)).astype(bool).reshape(H, W)
            ys, xs = np.nonzero(mk)
            d[kk] = (ys, xs)
    print("[seg] 首见帧 %d 段完成（至 %d），累计锚点 %d" % (g, min(nxt - 1, n_jpg - 1), n_anchor), flush=True)

# —— 逐帧合成（与 vm3 相同规则）——
meta = {"objs": {str(kk): {"uuid": v[0], "cls": v[1], "first": first_frame[v[0]]} for kk, v in objs_idx.items()},
        "per_frame": {}, "iou_min": IOU_MIN}
summary = {oid[:8]: dict(cls=L2.objs.get(oid, {}).get("type"), anno=0, cov=0) for oid, _ in order}
n_off = n_sam = 0
for fidx in range(n_jpg):
    sem = np.zeros((H, W), np.uint8)
    inst = np.zeros((H, W), np.uint16)
    prow = {}
    for kk, (oid, cls) in objs_idx.items():
        o, kind = get_foot(fidx, oid)
        if o is None:
            continue
        summary[oid[:8]]["anno"] += 1
        ub = upper_bound(fidx, oid, kind)
        m = stored.get(fidx, {}).get(kk)
        vc = np.zeros((H, W), bool)
        if m is not None and ub is not None:
            vc[m[0], m[1]] = True
            vc &= ub
        elif m is not None:
            vc[m[0], m[1]] = True
        inter = int((vc & (o > 0)).sum())
        union = int((vc | (o > 0)).sum())
        iou = inter / max(1, union)
        if int(vc.sum()) == 0 or iou < IOU_MIN:
            fin = o > 0
            src = "official"
        else:
            fin = vc
            src = "sam"
        if src == "official":
            n_off += 1
        else:
            n_sam += 1
        sem[fin] = int(L2.name2id.get(cls, 0))
        inst[fin] = kk
        summary[oid[:8]]["cov"] += 1
        prow[str(kk)] = "%s/%.2f" % (src, iou)
    np.save(os.path.join(OUT, "vm4_sem_%03d.npy" % fidx), sem)
    np.save(os.path.join(OUT, "vm4_inst_%03d.npy" % fidx), inst)
    meta["per_frame"][str(fidx)] = prow
    if fidx % 10 == 0 or fidx == n_jpg - 1:
        print("frame %3d 有身份对象 %2d/%d" % (fidx, len(prow), len(objs_idx)), flush=True)

meta["source"] = {"official": n_off, "sam": n_sam}
meta["summary"] = summary
json.dump(meta, open(os.path.join(OUT, "vm4_meta.json"), "w"), ensure_ascii=False)
full = [e for e in summary.values() if e["cov"] == e["anno"] and e["anno"] > 0]
print("\n=== 汇总 ===")
print("覆盖=标注帧数 的对象: %d / %d" % (len(full), len(summary)))
for oid8, e in summary.items():
    if e["anno"] > 0 and e["cov"] != e["anno"]:
        print("  未全覆盖: %s %s anno=%d cov=%d" % (oid8, e["cls"], e["anno"], e["cov"]))
print("source: official %d / sam %d" % (n_off, n_sam))
for t in ("1ec189f0", "98eaac6d", "b87392a3", "0e876d25", "206c41d1", "879a617b",
          "9b3c761a", "a2661202", "a269e69b", "b5e95a39", "fd74553a"):
    print(" ", t, summary.get(t))
print("VIDEO_ANCHOR_V4_DONE")
