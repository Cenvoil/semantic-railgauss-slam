#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""L4 仲裁合成 @ 5_station_bergedorf_5.1 —— 从 1.2 的 l4_compose_v5.py 适配。

L4 是"填充推理"的落点：把 L1(官方几何) / L2(SAM2 精修) / L3(stuff 填充) 三层
按优先级仲裁成最终四通道（semantic_id / instance_id / sem_conf / sem_route），
并用 route 通道记录**每个像素的来源**（便于事后审计哪层做了什么）。

仲裁优先级（1.2 定版，沿用）：
  对象层（L1/L2）> L3 stuff > 轨道几何规则 > 未标注
  route: 1=stuff 2=贴对象归并 3=thing映射 5=不认识→背景 6=轨道床
         7=走廊内降级 8=小块降级 4=仍未标注

适配点（全部是路径/规模差异，仲裁逻辑与开关一字未改）：
  LAB -> 5.1 labels.json / DATA -> output_1.0_51 / L2,L3,OUT -> tmp/d51/{l2,l3,l4}
  FRAMES: 100（5.1 也是 100 帧）；W/H 由 imagesize 动态取（原版硬编码 4112x2504）
  代表帧选取:原版硬编码 (0,29,59,99)（1.2 的冒烟集），改为**按最终语义覆盖率分位自动选**
  开关:TRACK_BED=0 / PERSON_NEEDS_OFFICIAL=1 / CORRIDOR_R=40 /
       RAILINFRA_MIN_AREA=600 —— 与 1.2 定版一致

5.1 相关注意：
  * 5.1 是**野外站场**（无站台、多股道），CS2OUR 映射不变（1.2 已把
    fence/wall → rail_infra，正是护栏/隔音墙，5.1 同样适用）。
  * 5.1 官方实例 88/7 类，含 train(7)/crowd(1)（1.2 无）⇒ CS2OUR_THING 里
    Cityscapes train(16) → 我们的 train(3) 这条在 5.1 上更有用。
"""

import json
import os
from collections import Counter

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

LAB = "<PROJECT_ROOT>/data_osdar/unpacked_51/5_station_bergedorf_5.1_labels.json"
CAM = "rgb_highres_center"
DATA = '<PROJECT_ROOT>/SGS-SLAM/data/output_1.0_51'
L2 = '<PROJECT_ROOT>/tmp/d51/l2'
L3 = '<PROJECT_ROOT>/tmp/d51/l3'
OUT = '<PROJECT_ROOT>/tmp/d51/l4'
FRAMES = list(range(100))
TW = 1400
DEV = 'cpu'

TRACK_BED = 0
PERSON_NEEDS_OFFICIAL = 1
PERSON_NEAR_R = 25
CORRIDOR_R = 40
CORRIDOR_SUPPRESS = {25, 27}      # building / rail_infra → ground
RAILINFRA_MIN_AREA = 600

CS_THING = [5, 6, 7, 11, 12, 13, 14, 15, 16, 17, 18]
CS_NAME = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
           'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
           'truck', 'bus', 'train', 'motorcycle', 'bicycle']
CS2OUR_THING = {5: 16, 11: 1, 12: 5, 13: 8, 14: 8, 15: 8, 16: 3, 17: 7, 18: 5}
CS_UNSURE = {6, 7}
CS_STRUCTURAL = {5, 6, 7}
LINE_CLS = {13, 14, 15, 28}

os.makedirs(OUT, exist_ok=True)
info = json.load(open(os.path.join(DATA, 'info_semantic.json')))
id2name = {c['id']: c['name'] for c in info['classes']}
lut = np.zeros((256, 3), np.uint8)
for c in info['classes']:
    lut[c['id']] = np.array(c['color'][:3], np.uint8)
flist = sorted(os.listdir(os.path.join(DATA, 'frames')))
l2stat = json.load(open(os.path.join(L2, 'l2_stats_v5.json')))  # 5.1 的 L2 stats
d = json.load(open(LAB))["openlabel"]
frames, objs = d["frames"], d["objects"]


def track_bed(f, HW=None):
    """官方每根轨道的两条钢轨折线之间填充 → 轨道床。返回 bool 图。"""
    fr = frames.get(str(f)) or frames.get(f)
    HH, WW = HW or (2504, 4112)
    acc = np.zeros((HH, WW), bool)
    n = 0
    for oid in (fr or {}).get("objects", {}):
        src = ((fr or {})["objects"][oid].get("object_data")
               or objs.get(oid, {}).get("object_data", {}))
        if objs.get(oid, {}).get("type") != "track":
            continue
        segs = [np.array(a["val"], float).reshape(-1, 2)
                for a in (src.get("poly2d") or []) if a.get("coordinate_system") == CAM]
        if len(segs) != 2:
            continue
        v1, v2 = segs
        if np.hypot(*(v2[0] - v1[0])) > np.hypot(*(v2[-1] - v1[0])):
            v2 = v2[::-1]
        im = Image.new("L", (WW, HH), 0)
        ImageDraw.Draw(im).polygon([(float(x), float(y)) for x, y in np.vstack([v1, v2])], fill=1)
        acc |= np.asarray(im) > 0
        n += 1
    return acc, n


stats = {'switches': {'TRACK_BED': TRACK_BED, 'PERSON_NEEDS_OFFICIAL': PERSON_NEEDS_OFFICIAL,
                      'PERSON_NEAR_R': PERSON_NEAR_R, 'CORRIDOR_R': CORRIDOR_R,
                      'CORRIDOR_SUPPRESS': sorted(CORRIDOR_SUPPRESS),
                      'RAILINFRA_MIN_AREA': RAILINFRA_MIN_AREA},
         'route_codes': '1=stuff 2=贴对象归并 3=thing映射 5=不认识→背景 6=轨道床 7=走廊内降级 8=小块降级 4=未标注',
         'frames': {}}
tiles = {'sem': [], 'ov': [], 'conf': [], 'inst': []}
rng = np.random.RandomState(20260918)
CONF_COL = {2: (228, 60, 60), 1: (240, 160, 40), 0: (70, 105, 145)}
UNLBL = (12, 12, 12)

for f in FRAMES:
    rgb = Image.open(os.path.join(DATA, 'frames', flist[f])).convert('RGB')
    W, H = rgb.size
    sem_o = np.load(os.path.join(L2, 'mask_L2_%03d.npy' % f))
    conf_o = np.load(os.path.join(L2, 'conf_L2_%03d.npy' % f))
    inst_o = np.load(os.path.join(L2, 'inst_L2_%03d.npy' % f))
    stuff = np.load(os.path.join(L3, 'l3_stuff_%03d.npy' % f))
    qual = np.load(os.path.join(L3, 'l3_qual_%03d.npy' % f))
    cs = np.load(os.path.join(L3, 'l3_cs_%03d.npy' % f))

    sem = sem_o.astype(np.uint16)
    sem[sem == 255] = 0
    n255 = int((sem_o == 255).sum())
    conf = conf_o.astype(np.uint8).copy()
    conf[sem_o == 255] = 0
    obj_ref = sem.copy()
    obj_mask = sem > 0
    objsem = obj_ref.copy()

    fill = sem == 0
    n_fill = int(fill.sum())
    thing = fill & np.isin(cs, CS_THING)
    plain = fill & ~thing

    sem[plain] = stuff[plain]
    conf[plain] = 0
    route = np.zeros((H, W), np.uint8)
    route[plain & (stuff > 0)] = 1

    # ---- 官方轨道几何：轨线 + 轨道床 + 走廊 ----
    bed, nbed = track_bed(f, (H, W))
    rail = np.isin(obj_ref, [13])                 # 官方轨线像素（对象层）
    corridor = ndimage.binary_dilation(rail | bed | np.isin(obj_ref, [15]),
                                       np.ones((2 * CORRIDOR_R + 1,) * 2, bool))
    # ③ 走廊内把 building / rail_infra 降级为 ground
    dg = (route == 1) & corridor & np.isin(sem, list(CORRIDOR_SUPPRESS))
    n_dg = int(dg.sum())
    sem[dg] = 26
    route[dg] = 7
    # ② rail_infra 孤立小块降级（隔音墙是大块，不会被误伤）
    if (sem == 27).any() and RAILINFRA_MIN_AREA > 0:
        lab, n = ndimage.label(sem == 27)
        sz = ndimage.sum(sem == 27, lab, range(1, n + 1))
        small = np.zeros_like(sem, bool)
        for i in range(n):
            if sz[i] < RAILINFRA_MIN_AREA:
                small |= (lab == i + 1)
        n_small = int(small.sum())
        sem[small] = 26
        route[small] = 8
    else:
        n_small = 0

    # ---- thing 处理 ----
    rest = thing.copy()
    if rest.any():
        tgt = obj_mask & ~np.isin(obj_ref, list(LINE_CLS))
        dd, (iy, ix) = ndimage.distance_transform_edt(~tgt, return_indices=True)
        near = rest & (dd <= 4) & np.isin(cs, list(CS_STRUCTURAL))
        sem[near] = objsem[iy[near], ix[near]]
        conf[near] = 0
        route[near] = 2
        rest &= ~near
    n_person_rej = 0
    if rest.any():
        for k, v in CS2OUR_THING.items():
            m = rest & (cs == k)
            if m.any():
                sem[m] = v
                conf[m] = 0
                route[m] = 3
        rest &= ~np.isin(cs, list(CS2OUR_THING))
    # ① person 需官方佐证：只在官方 person 对象 25px 内才留，否则按"不认识"走背景
    if PERSON_NEEDS_OFFICIAL:
        off_p = (inst_o > 0) & (obj_ref == 1)
        off_d = ndimage.binary_dilation(off_p, np.ones((2 * PERSON_NEAR_R + 1,) * 2, bool))
        pm = (route == 3) & (sem == 1)
        rej = pm & ~off_d
        n_person_rej = int(rej.sum())
        sem[rej] = 0
        route[rej] = 0
        rest = rest | rej
    if rest.any():
        left = rest & np.isin(cs, list(CS_UNSURE | ({11} if PERSON_NEEDS_OFFICIAL else set())))
        if left.any():
            src = (sem > 0) & ~np.isin(sem, list(LINE_CLS))
            d3, (ky, kx) = ndimage.distance_transform_edt(~src, return_indices=True)
            sem[left] = sem[ky[left], kx[left]]
            conf[left] = 0
            route[left] = 5
        rest &= ~left

    # ③ 轨道床（只覆盖"由 stuff / 不认识"填出来的像素，不动任何对象像素）
    if TRACK_BED:
        tb = bed & np.isin(route, [1, 5, 7, 8])
        n_bed = int(tb.sum())
        sem[tb] = 13
        conf[tb] = 0
        route[tb] = 6
    else:
        n_bed = 0
    route[fill & (sem == 0)] = 4

    inst = inst_o.copy()
    inst[sem == 0] = 0
    inst[fill] = 0

    tot = float(W * H)
    unl = sem == 0
    obj_changed = int(((sem_o > 0) & (sem_o != 255) & (sem != sem_o)).sum())
    cls_px = Counter(sem.ravel().tolist())
    conf_px = Counter(conf.ravel().tolist())
    rp = Counter(route.ravel().tolist())
    st = {
        'unlabeled_pct': round(100.0 * int(unl.sum()) / tot, 4),
        'conf_pct': {'2_official_poly2d': round(100.0 * conf_px.get(2, 0) / tot, 4),
                     '1_sam2': round(100.0 * conf_px.get(1, 0) / tot, 4),
                     '0_geom_rule_stuff': round(100.0 * conf_px.get(0, 0) / tot, 4)},
        'route_px': {k: rp.get(i, 0) for i, k in
                     ((1, 'stuff'), (2, 'edge_merge'), (3, 'thing_map'), (5, 'unsure_to_bg'),
                      (6, 'track_bed'), (7, 'corridor_downgrade'), (8, 'small_railinfra'), (4, 'unlabeled'))},
        'person_rejected_px': n_person_rej,
        'bed_objects': nbed,
        'obj_pixels_changed': obj_changed,
        'illegal_id255_px': n255,
        'cls_pct': {id2name.get(k, str(k)): round(100.0 * v / tot, 4)
                    for k, v in sorted(cls_px.items(), key=lambda z: -z[1]) if k > 0},
        'n_instance': int(len(np.unique(inst)) - 1),
        'l2_ok_fallback_skip': [l2stat['stat'][str(f)]['ok'], l2stat['stat'][str(f)]['fallback'],
                                l2stat['stat'][str(f)]['skip']],
    }
    stats['frames'][str(f)] = st
    print('[frame %03d] 未标注 %.3f%% | 床 %d px | 走廊降级 %d | 小块降级 %d | person 剔除 %d | 对象改动 %d'
          % (f, st['unlabeled_pct'], n_bed, n_dg, n_small, n_person_rej, obj_changed), flush=True)

    np.save(os.path.join(OUT, 'semantic_id_%03d.npy' % f), sem.astype(np.uint16))
    np.save(os.path.join(OUT, 'instance_id_%03d.npy' % f), inst.astype(np.uint16))
    np.save(os.path.join(OUT, 'sem_conf_%03d.npy' % f), conf)
    np.save(os.path.join(OUT, 'sem_route_%03d.npy' % f), route)
    Image.fromarray(sem.astype(np.uint16)).save(os.path.join(OUT, 'semantic_id_%03d.png' % f))
    Image.fromarray(inst.astype(np.uint16)).save(os.path.join(OUT, 'instance_id_%03d.png' % f))
    Image.fromarray(conf).save(os.path.join(OUT, 'sem_conf_%03d.png' % f))

    sc = TW / float(W)
    th = int(round(H * sc))
    sem_i = Image.fromarray(lut[np.clip(sem, 0, 255)].astype(np.uint8)).resize((TW, th), Image.NEAREST)
    tiles['sem'].append(sem_i)
    tiles['ov'].append(Image.blend(rgb.resize((TW, th), Image.LANCZOS), sem_i, 0.45))
    cc = np.zeros((H, W, 3), np.uint8)
    for k, c in CONF_COL.items():
        cc[conf == k] = c
    cc[unl] = UNLBL
    tiles['conf'].append(Image.fromarray(cc).resize((TW, th), Image.NEAREST))
    ic = np.zeros((H, W, 3), np.uint8)
    for u in np.unique(inst):
        if u:
            ic[inst == u] = rng.randint(60, 255, 3)
    tiles['inst'].append(Image.fromarray(ic).resize((TW, th), Image.NEAREST))


def stitch(ts, path, label, ids):
    import math
    tw, th = ts[0].size
    cols = 2
    rows = int(math.ceil(len(ts) / float(cols)))
    cv = Image.new('RGB', (tw * cols, th * rows), (18, 18, 18))
    dr = ImageDraw.Draw(cv)
    for i, t in enumerate(ts):
        x, y = (i % cols) * tw, (i // cols) * th
        cv.paste(t, (x, y))
        dr.rectangle([x, y, x + 340, y + 30], fill=(0, 0, 0))
        dr.text((x + 8, y + 9), '%s  frame %03d' % (label, ids[i]), fill=(255, 255, 255))
    cv.save(path)
    print('saved', path, cv.size, flush=True)


# 代表帧：按最终 semantic_id 的覆盖率分位自动选（1.2 是硬编码 0/29/59/99）
_cov = []
for f in FRAMES:
    _p = os.path.join(OUT, 'semantic_id_%03d.npy' % f)
    if os.path.exists(_p):
        _cov.append((f, float((np.load(_p) > 0).mean())))
    else:
        _cov.append((f, 0.0))
_order = sorted(range(len(_cov)), key=lambda i: _cov[i][1])
_pick = [int(round((len(_order) - 1) * q)) for q in (0.05, 0.35, 0.65, 0.95)]
sel4 = [_order[i] for i in sorted(_pick)]
f4 = [FRAMES[i] for i in sel4]
print('[L4] 代表帧(按覆盖率分位) = %s  覆盖率 = %s'
      % (f4, ['%.2f%%' % (100 * _cov[i][1]) for i in sel4]), flush=True)
stitch([tiles['sem'][i] for i in sel4], os.path.join(OUT, 'sem_L4_stitch.png'), 'L4v5 semantic_id', f4)
stitch([tiles['ov'][i] for i in sel4], os.path.join(OUT, 'overlay_L4_stitch.png'), 'L4v5 semantic over RGB', f4)
stitch([tiles['conf'][i] for i in sel4], os.path.join(OUT, 'conf_L4_stitch.png'), 'L4v5 sem_conf', f4)
stitch([tiles['inst'][i] for i in sel4], os.path.join(OUT, 'inst_L4_stitch.png'), 'L4v5 instance_id', f4)
stitch(tiles['sem'], os.path.join(OUT, 'sem_L4_stitch_11f.png'), 'L4v5 semantic_id', FRAMES)
json.dump(stats, open(os.path.join(OUT, 'l4_stats.json'), 'w'), ensure_ascii=False, indent=1)
print('L4_51_DONE', flush=True)
