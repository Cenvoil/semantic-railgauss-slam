#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证自定义渲染与项目 eval 目录的官方渲染是否一致（参数是否用对）。

## 为什么必须验
我自写 `render_frames.py` 渲代表帧，若内参/分辨率/位姿用错，
渲出来的图"看起来像"但其实不对，误差图就毫无意义。
⇒硬判据：**与官方 `eval/rendered_rgb` 的同帧图逐像素比对**。

若一致 → 我的渲染参数正确，后续代表帧可信。
若不一致 → 差在哪（内参？分辨率？位姿？）必须查清。

用法: python verify_render.py
"""
import os

import numpy as np
from PIL import Image

B = '<PROJECT_ROOT>/benchmarks'
EXP = '<PROJECT_ROOT>/SGS-SLAM'
RUN = 'd8_gk1045_insinit'


def main():
    mine_dir = os.path.join(B, 'results', 'frames12')
    off_rgb = os.path.join(EXP, 'experiments', 'Railway_d', RUN, 'eval',
                           'rendered_rgb')
    off_dep = os.path.join(EXP, 'experiments', 'Railway_d', RUN, 'eval',
                           'rendered_depth')

    print('官方渲染帧:', sorted(os.listdir(off_rgb)))
    print('我的渲染帧:', sorted(os.listdir(mine_dir)))
    print()

    for fi in [29, 59, 99]:
        mine = os.path.join(mine_dir, 'gs_%04d.png' % fi)
        off = os.path.join(off_rgb, 'gs_%04d.png' % fi)
        if not (os.path.exists(mine) and os.path.exists(off)):
            print('帧 %d: 文件缺失 mine=%s off=%s'
                  % (fi, os.path.exists(mine), os.path.exists(off)))
            continue
        a = np.array(Image.open(mine).convert('RGB')).astype(np.float64)
        b = np.array(Image.open(off).convert('RGB')).astype(np.float64)
        print('帧 %d: 我的 %s  官方 %s' % (fi, a.shape, b.shape))
        if a.shape != b.shape:
            print('  ★ 尺寸不同，无法逐像素比对')
            continue
        d = np.abs(a - b)
        print('  RGB 逐像素差: mean %.3f  p95 %.1f  max %.1f'
              % (d.mean(), np.percentile(d, 95), d.max()))
        # 深度
        mn = os.path.join(mine_dir, 'renddepth_%04d.npy' % fi)
        on = os.path.join(off_dep, 'gs_%04d.png' % fi)
        if os.path.exists(mn) and os.path.exists(on):
            dm = np.load(mn)
            im = Image.open(on)
            do = np.array(im)
            print('  官方深度: mode=%s dtype=%s 范围 %d..%d'
                  % (im.mode, do.dtype, do.min(), do.max()))
            if do.ndim == 3:      # 官方是上色图（RGB），不是原始深度
                do = do[..., 0].astype(np.float32)
            m = (dm > 0) & (do > 0)
            if m.sum() > 0:
                print('  深度共同有效 %d px  我 p50 %.2f  官方 p50 %.2f  '
                      '差 p50 %.3f m'
                      % (m.sum(), np.median(dm[m]), np.median(do[m]),
                         np.median(np.abs(dm[m] - do[m]))))
        print()


if __name__ == '__main__':
    main()