#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""渲染代表帧 —— **完全照抄项目 eval 的做法**（此前自写版漏了两步，渲出来不对）。

## ★ 踩坑记录（2026-10-08）
我第一版自写渲染，直接用 params.npz + 手工缩放内参 + 每帧位姿，
结果与官方 `eval/rendered_rgb/gs_0029.png` **逐像素差 mean 29.0/255**。

读 `utils/eval_helpers.py::eval` 后找到两个漏掉的关键步骤：

1. **相机位姿固定用第 0 帧的 `first_frame_w2c`**
   （源码里 `first_frame_w2c = torch.linalg.inv(pose)` 只在 `time_idx == 0` 算一次，
   之后所有帧都用它）—— 不是每帧用各自位姿。
   ★ 这正是 SGS-SLAM"高斯在世界系、相机在第 0 帧系"的实现方式。

2. **高斯必须按帧裁剪**：`transform_to_frame(final_params, time_idx, ...)`
   只保留该帧对应的关键帧高斯。跳过这步会渲出错误内容。

3. **内参用 `dataset[i]['intrinsics']`**（数据集自带、已按实际分辨率缩放），
   不是自己缩放原始内参。

⇒ 本脚本严格照抄。验收标准：与官方渲染逐像素差应 ≈ 0（用 verify_render.py 验）。

用法:
  python render_frames.py --run d8_gk1045_insinit --frames 1 29 59 99 --outdir <dir>
"""
import argparse
import os
import sys

import numpy as np
import torch

ROOT = '<PROJECT_ROOT>/SGS-SLAM'
sys.path.insert(0, ROOT)
os.chdir(ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='d8_gk1045_insinit')
    # ★ 用 run config.py 里的 scene_name，保持一致
    ap.add_argument('--scene', default='output_1.0_gk1045_l4')
    # ★ 官方 eval 的渲染分辨率是 3084x1878（= 4112x2504 的 0.75 倍，
    #   来自 dataset 的 downsample_rate）。按同样尺寸渲才能逐像素验收。
    ap.add_argument('--downsample', type=float, default=0.75)
    # ★★ 相机位姿用「当前帧」还是「第 0 帧」？
    #   官方 eval 用第 0 帧（first_frame_w2c），因为它渲的是"新视角"，
    #   目的是展示地图质量而非逐帧重建。
    #   ★ 但要与「同帧的 LiDAR 投影」比误差时，**必须用当前帧位姿**，
    #     否则渲染深度与 LiDAR 深度差一个基线（实测 p50 23.9 vs 37.9 m）。
    ap.add_argument('--cam-frame', type=int, default=-1,
                    help='-1=用第0帧(官方eval口径, 用于看地图质量); '
                         '>=0=用该帧位姿(用于与同帧 LiDAR 比误差)')
    ap.add_argument('--frames', nargs='+', type=int, default=[1, 29, 59, 99])
    ap.add_argument('--outdir', required=True)
    args = ap.parse_args()

    from datasets.gradslam_datasets.dataconfig import load_dataset_config
    from datasets.gradslam_datasets.railway import RailwayDataset
    from utils.recon_helpers import setup_camera
    from utils.slam_helpers import (transform_to_frame,
                                    transformed_params2rendervar,
                                    transformed_params2depthplussilhouette)
    from diff_gaussian_rasterization import GaussianRasterizer as Renderer

    os.makedirs(args.outdir, exist_ok=True)
    dev = 'cuda'

    # ★★ 完全照抄 d8_gk1045_insinit/config.py 里的 data 段：
    #    gradslam_data_cfg = "./configs/data/railway_osdar_official.yaml"
    #    basedir = "./data"，scene_name 决定用哪个数据集目录
    cfg = load_dataset_config(
        os.path.join(ROOT, 'configs', 'data', 'railway_osdar_official.yaml'))
    ds = RailwayDataset(cfg, os.path.join(ROOT, 'data'),
                        sequence=args.scene, stride=None,
                        desired_height=int(2504 * args.downsample),
                        desired_width=int(4112 * args.downsample),
                        load_semantics=False)
    print('scene=%s 已载入，共 %d 帧' % (args.scene, len(ds)))

    P = np.load(os.path.join(ROOT, 'experiments', 'Railway_d', args.run,
                             'params.npz'), allow_pickle=True)
    params = {}
    # ★ 必须包含 cam_unnorm_rots / cam_trans：
    #   transform_to_frame() 要用它们做坐标变换（源码 slam_helpers.py:341）
    for k in ('means3D', 'rgb_colors', 'unnorm_rotations', 'logit_opacities',
              'log_scales', 'cam_unnorm_rots', 'cam_trans'):
        params[k] = torch.tensor(np.asarray(P[k]), dtype=torch.float32,
                                 device=dev)
    print('高斯总数 %s' % format(params['means3D'].shape[0], ','))

    # ---- 第 0 帧定相机（照抄 eval）----
    color0, depth0, intr0, pose0 = ds[0]
    intr0 = intr0[:3, :3]
    first_w2c = np.linalg.inv(pose0.cpu().numpy().astype(np.float64))
    first_w2c_t = torch.tensor(first_w2c, dtype=torch.float32, device=dev)
    # ⚠⚠ dataset 返回的 color 是 **(H,W,3)**（实测 shape=(1878,3084,3)），
    #   所以 W/H 必须取 shape[1]/shape[0]。
    #   ★ 官方 eval_helpers.py 传的是 color.shape[2], color.shape[1]（即 3,1878），
    #     在官方路径里靠别处修正；我们这里直接用正确顺序，避免渲出 (3,3084,3) 的废图。
    H, W = color0.shape[0], color0.shape[1]
    cam0 = setup_camera(W, H, intr0.cpu().numpy(), first_w2c, device=dev)
    print('相机 %dx%d  内参 fx=%.1f fy=%.1f cx=%.1f cy=%.1f'
          % (W, H, intr0[0, 0], intr0[1, 1], intr0[0, 2], intr0[1, 2]))

    from PIL import Image
    for fi in args.frames:
        pts = transform_to_frame(params, fi, gaussians_grad=False,
                                 camera_grad=False, device=dev)
        rv = transformed_params2rendervar(params, pts, device=dev)
        # ★ 像素深度必须用 depthplussilhouette 变体：
        #   普通 Renderer 返回的 depth 是「每高斯一个」(5,169,530)，
        #   不是像素图。官方 eval 用的就是这个变体（eval_helpers.py:693）。
        # 按 --cam-frame 决定相机位姿
        if args.cam_frame >= 0:
            _, _, _, p_i = ds[args.cam_frame]
            w2c_i = np.linalg.inv(p_i.cpu().numpy().astype(np.float64))
            cam_i = setup_camera(W, H, intr0.cpu().numpy(), w2c_i,
                                 device=dev)
            w2c_i_t = torch.tensor(w2c_i, dtype=torch.float32, device=dev)
        else:
            cam_i, w2c_i_t = cam0, first_w2c_t

        rv_d = transformed_params2depthplussilhouette(
            params, w2c_i_t, pts, device=dev)
        with torch.no_grad():
            img, dep, _ = Renderer(raster_settings=cam_i)(**rv)
            dep_pix, _, _ = Renderer(raster_settings=cam_i)(**rv_d)
        # ⚠ img 是 (3,H,W)，必须先 permute 成 (H,W,3) 再存
        #   （漏了这步会存成 (3,H,W) 的错图，验收脚本直接报尺寸不匹配）
        arr = (img.permute(1, 2, 0).detach().cpu().numpy() * 255
               ).clip(0, 255).astype(np.uint8)
        Image.fromarray(arr).save(os.path.join(args.outdir,
                                               'gs_%04d.png' % fi))
        dd = dep_pix.detach().cpu().numpy()
        dd2 = dd[0] if dd.ndim == 3 else dd
        if dd2 is not None and dd2.shape == (H, W):
            np.save(os.path.join(args.outdir, 'renddepth_%04d.npy' % fi),
                    dd2.astype(np.float32))
            p50 = (np.percentile(dd2[dd2 > 0], 50) if (dd2 > 0).any() else -1)
            print('帧 %d: 用高斯 %s  深度有效 %.1f%%  p50 %.1f m'
                  % (fi, format(pts.shape[0], ','),
                     float((dd2 > 0).mean()) * 100, p50))
        else:
            print('帧 %d: depth 形状 %s 不符(H,W)=(%d,%d)，跳过存深度'
                  % (fi, dd.shape, H, W))

    print('-> %s' % args.outdir)


if __name__ == '__main__':
    main()