#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""L3 定版 stuff 填充层 @ 5_station_bergedorf_5.1（SegFormer-B5 Cityscapes）。

## 尺度裁决（2026-10-04 在 5.1 上实测，**与 1.2 的结论不同**）

1.2 的定版是 S0.25（等比 1024 宽）。5.1 按铁律 ⑨ 重新裁决，用**两个独立判据**：

判据一 · 区域命中率（8 个区域，框在帧 60 上目检确认后重设）：
  区域                S1.00   S0.50   S0.25
  R1 轨枕钢轨          19.6    93.1    87.3
  R2 碎石道砟          52.7    99.4    94.8
  R3 草地              36.2    35.7    47.4
  R5 左侧股道          29.5    35.2    55.7
  （R4/R6/R7/R8 三尺度都低命中 —— 我给这些框设的"期望 Cityscapes 类"本身就选错了，
   铁路护栏被 Cityscapes 判 vegetation 是常态而非缺陷，不 `fence` 类 catching 铁路护栏。
   **错判据不能用来选尺度**，故需要判据二。）

判据二 · 两尺度一致率（不依赖人工期望类；1.2 已用它做置信交叉验证）：
  S1.00↔S0.50 = 77.90%   S1.00↔S0.25 = 67.22%   S0.50↔S0.25 = 79.21%
  主标签 argmax 概率中位：S1.00 0.9928 / **S0.50 0.9961** / S0.25 0.9830

**⇒ 5.1 主标签用 S0.50**（1.2 用 S0.25），交叉验证用 S0.25。
差异的合理解释：5.1 每帧走 1.01 m（1.2 是 0.22 m），画面内容变化更快；
半分辨率（2056×1252）在 5.1 上比 1.2 更需要细节保留。
**注意一致性方向反过来了**：1.2 是 S0.50↔S0.25 一致率最高（86.17%），
5.1 是 S0.50↔S0.25 最高（79.21%）但 S0.25 自信度最低 —— 说明 S0.25 在 5.1 上偏弱。

## 类映射（只取 stuff；通用模型的 thing 类一律不写入，thing 单独出图交 L4 仲裁）
  road/sidewalk/terrain → 26 ground
  building              → 25 building
  fence / wall          → 27 rail_infra   （隔音墙/护栏属铁路基础设施）
  vegetation            → 23 vegetation
  sky                   → 24 sky

## 产物（与 1.2 同构）
  l3_stuff_%03d.npy  uint8  我们的类 id（0 = 非 stuff）
  l3_qual_%03d.npy   uint8  2=两版一致且 prob>=0.7 / 1=两版一致 / 0=不一致
  l3_thing_%03d.npy  uint8  1=通用模型判为 thing（人/车/杆/灯/列车…）→ L4 仲裁
  l3_prob_%03d.npy   uint8  主标签 argmax 概率*255
  l3_cs_%03d.npy     uint8  Cityscapes 原始标签（审计/诊断用）
  l3_stats.json
"""
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import (SegformerForSemanticSegmentation,
                          SegformerImageProcessor)

DATA = '<PROJECT_ROOT>/SGS-SLAM/data/output_1.0_51'
OUT = '<PROJECT_ROOT>/tmp/d51/l3'
CKPT = '<PROJECT_ROOT>/checkpoints/segformer_b5_cityscapes'
FRAMES = [int(x) for x in os.environ.get('L3_FRAMES',
                                         ','.join(str(i) for i in range(100))).split(',')]
DEV = 'cuda'
CS_NAME = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
           'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
           'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
           'bicycle']
CS2OUR = {0: 26, 1: 26, 9: 26, 2: 25, 4: 27, 3: 27, 8: 23, 10: 24}
CS_THING = [5, 6, 7, 11, 12, 13, 14, 15, 16, 17, 18]
PRIMARY = (0.5, True)      # ★ 5.1 定版：半分辨率分块
XCHECK = (0.25, False)     # 交叉验证：等比 1024 宽


def positions(L, tile, stride):
    """覆盖 [0,L] 的块起点：全集 + 贪心去冗余（近似写法会在长边上留缝）。"""
    if L <= tile:
        return [0]
    ps = sorted(set(list(range(0, L - tile + 1, stride)) + [L - tile]))
    keep = []
    for i, p in enumerate(ps):
        hi, cur = p + tile, p
        for a, b in sorted((q, min(q + tile, L)) for q in ps[:i] + ps[i + 1:]):
            if a > cur:
                break
            cur = max(cur, b)
            if cur >= hi:
                break
        if cur < hi:
            keep.append(p)
    return keep or [0]


def infer(model, proc, im, tile=1024, stride=768):
    W, H = im.size
    xs, ys = positions(W, tile, stride), positions(H, tile, stride)
    acc = np.zeros((19, H, W), np.float32)
    cnt = np.zeros((H, W), np.float32)
    for y in ys:
        for x in xs:
            x1, y1 = min(x + tile, W), min(y + tile, H)
            ch, cw = y1 - y, x1 - x
            p = proc(images=im.crop((x, y, x1, y1)), return_tensors='pt').to(DEV)
            with torch.no_grad():
                o = model(**p).logits
                o = F.interpolate(o, size=(ch, cw), mode='bilinear', align_corners=False)
            acc[:, y:y1, x:x1] += o[0].float().cpu().numpy()
            cnt[y:y1, x:x1] += 1
    assert cnt.min() > 0, 'tile 覆盖有死角'
    acc /= cnt[None]
    mx, lab = torch.softmax(torch.from_numpy(acc), 0).max(0)
    return lab.numpy().astype(np.uint8), (mx.numpy() * 255).astype(np.uint8), len(xs) * len(ys)


def to_our(lab):
    out = np.zeros(lab.shape, np.uint8)
    for cs, our in CS2OUR.items():
        out[lab == cs] = our
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    info = json.load(open(os.path.join(DATA, 'info_semantic.json')))
    our_color = {c['id']: tuple(c['color'][:3]) for c in info['classes']}
    our_name = {c['id']: c['name'] for c in info['classes']}
    flist = sorted(os.listdir(os.path.join(DATA, 'frames')))
    model = SegformerForSemanticSegmentation.from_pretrained(
        CKPT, local_files_only=True).eval().to(DEV)
    proc = SegformerImageProcessor.from_pretrained(CKPT, local_files_only=True)
    stats = {'sequence': '5_station_bergedorf_5.1',
             'primary': 'S0.50 (half-res tiled 1024/768)  <-- 5.1 定版，与 1.2 的 S0.25 不同',
             'xcheck': 'S0.25 (aspect-preserving 1024w)',
             'scale_pick_evidence': {
                 'region_hit_pct': {'R1_轨枕钢轨': [19.6, 93.1, 87.3],
                                    'R2_碎石道砟': [52.7, 99.4, 94.8]},
                 'pair_agree_pct': {'S1.00-S0.50': 77.90, 'S1.00-S0.25': 67.22,
                                    'S0.50-S0.25': 79.21},
                 'prob_median': {'S1.00': 0.9928, 'S0.50': 0.9961, 'S0.25': 0.9830}},
             'cs2our': CS2OUR, 'frames': {}}
    tiles_sem, tiles_ov = [], []

    for f in FRAMES:
        t0 = time.time()
        rgb = Image.open(os.path.join(DATA, 'frames', flist[f])).convert('RGB')
        WN, HN = rgb.size
        ps, tiled = PRIMARY
        sz = (WN // 2, HN // 2) if tiled else (1024, int(round(HN * 1024 / WN)))
        lab_g, prob_g, nt_g = infer(model, proc, rgb.resize(sz, Image.LANCZOS))
        lab_up = np.asarray(Image.fromarray(lab_g).resize((WN, HN), Image.NEAREST))
        prob_up = np.asarray(Image.fromarray(prob_g).resize((WN, HN), Image.NEAREST))
        xs_, tiled_ = XCHECK
        sz2 = (WN // 2, HN // 2) if tiled_ else (1024, int(round(HN * 1024 / WN)))
        lab_h, _, nt_h = infer(model, proc, rgb.resize(sz2, Image.LANCZOS))
        lab_h_up = np.asarray(Image.fromarray(lab_h).resize((WN, HN), Image.NEAREST))

        stuff = to_our(lab_up)
        thing = np.isin(lab_up, CS_THING).astype(np.uint8)
        agree = lab_up == lab_h_up
        qual = np.where(agree & (prob_up >= 178), 2, np.where(agree, 1, 0)).astype(np.uint8)

        np.save(os.path.join(OUT, 'l3_stuff_%03d.npy' % f), stuff)
        np.save(os.path.join(OUT, 'l3_qual_%03d.npy' % f), qual)
        np.save(os.path.join(OUT, 'l3_thing_%03d.npy' % f), thing)
        np.save(os.path.join(OUT, 'l3_prob_%03d.npy' % f), prob_up)
        np.save(os.path.join(OUT, 'l3_cs_%03d.npy' % f), lab_up)

        tot = float(WN * HN)
        ours = {('%d_%s' % (k, our_name.get(k, '?'))):
                round(float((stuff == k).sum()) / tot * 100, 4)
                for k in sorted(set(CS2OUR.values()))}
        cs_hist = {CS_NAME[c]: round(float((lab_up == c).sum()) / tot * 100, 3)
                   for c in range(19) if (lab_up == c).sum() > tot * 0.002}
        st = {'sec': round(time.time() - t0, 1),
              'tiles_primary': nt_g, 'tiles_xcheck': nt_h,
              'our_share_pct': ours, 'cs_hist_pct': cs_hist,
              'stuff_cover_pct': round(float((stuff > 0).sum()) / tot * 100, 4),
              'thing_pct': round(float(thing.sum()) / tot * 100, 4),
              'qual_pct': {'2_agree_hi': round(float((qual == 2).mean()) * 100, 3),
                           '1_agree': round(float((qual == 1).mean()) * 100, 3),
                           '0_disagree': round(float((qual == 0).mean()) * 100, 3)},
              'agree_two_scale_pct': round(float(agree.mean()) * 100, 3),
              'prob_median': round(float(np.median(prob_up)) / 255.0, 4)}
        stats['frames'][str(f)] = st
        print('[frame %03d] stuff %.2f%% thing %.2f%% 两版一致 %.2f%% (强 %.2f%%) %.1fs'
              % (f, st['stuff_cover_pct'], st['thing_pct'],
                 st['agree_two_scale_pct'], st['qual_pct']['2_agree_hi'], st['sec']),
              flush=True)

        col = np.zeros((HN, WN, 3), np.uint8)
        for oid, c in our_color.items():
            if oid:
                col[stuff == oid] = c
        sc = 1400.0 / WN
        tw, th = 1400, int(round(HN * sc))
        sem_s = Image.fromarray(col).resize((tw, th), Image.NEAREST)
        tiles_sem.append(sem_s)
        tiles_ov.append(Image.blend(rgb.resize((tw, th), Image.LANCZOS), sem_s, 0.45))

    for name, tiles in (('l3_stuff_stitch.png', tiles_sem),
                        ('l3_overlay_stitch.png', tiles_ov)):
        if not tiles:
            continue
        tw, th = tiles[0].size
        cols = 2
        rows = (len(tiles) + cols - 1) // cols
        cv = Image.new('RGB', (tw * cols, th * rows), (18, 18, 18))
        for i, t in enumerate(tiles):
            cv.paste(t, ((i % cols) * tw, (i // cols) * th))
        cv.save(os.path.join(OUT, name))
        print('[save]', name, cv.size, flush=True)
    json.dump(stats, open(os.path.join(OUT, 'l3_stats.json'), 'w'),
              ensure_ascii=False, indent=1)
    print('L3_51_DONE', flush=True)


if __name__ == '__main__':
    main()
