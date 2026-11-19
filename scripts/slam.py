import argparse
import os
import shutil
import sys
import time
import pandas as pd
from importlib.machinery import SourceFileLoader

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE_DIR)

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb

# P-42 状态（跨帧缓存上一帧的 INS 位移与最终估计位姿，用于漂移尖峰兜底）
P42_STATE = {'ins_t': None, 'est_t': None, 'n_clamp': 0}


def depth_local_outlier_mask(depth, ratio=0.5, k=9):
    """P-43: 找出"显著近于局部邻域"的孤立深度离群点。

    融合深度在远景/建筑区常出现孤立的假近值（实测 1.2 序列右上角：中位 65m，
    但 p10 只有 21.5m），这些像素能通过 <FAR 的有效性检查被种成高斯 → 永久浮空物。
    判据：d < (1-ratio) * 局部均值(window=k) 视为离群。
    注意：过大会误伤细长结构（电杆），ratio/kernel 需按序列调。
    """
    d2 = depth.reshape(-1, *depth.shape[-2:]).unsqueeze(1).float()
    valid = (d2 > 0)
    with torch.no_grad():
        s = torch.nn.functional.avg_pool2d(torch.where(valid, d2, torch.zeros_like(d2)),
                                           k, stride=1, padding=k // 2)
        c = torch.nn.functional.avg_pool2d(valid.float(), k, stride=1, padding=k // 2)
        loc = s / c.clamp(min=1e-6)
        out = valid & (d2 < loc * (1.0 - ratio))
    return out.reshape(-1)

from datasets.gradslam_datasets import (
    load_dataset_config,
    ICLDataset,
    ReplicaDataset,
    RailwayDataset,
    ReplicaV2Dataset,
    AzureKinectDataset,
    ScannetDataset,
    Ai2thorDataset,
    Record3DDataset,
    RealsenseDataset,
    TUMDataset,
    ScannetPPDataset,
    NeRFCaptureDataset
)
from utils.common_utils import seed_everything, save_params_ckpt, save_params
from utils.eval_helpers import report_loss, report_progress, eval
from utils.keyframe_selection import keyframe_selection_overlap
from utils.recon_helpers import setup_camera
from utils.slam_helpers import (
    transformed_params2rendervar, transformed_params2depthplussilhouette,
    transformed_semantics2rendervar, transform_to_frame, l1_loss_v1, matrix_to_quaternion,
    aniso_scales, sky_render
)
from utils.slam_external import calc_ssim, build_rotation, prune_gaussians, densify

from diff_gaussian_rasterization import GaussianRasterizer as Renderer


# --- v4 (E-017 / F1+F2): env-switchable geo mask, default OFF for backward compat ---
import os as _os
V4_GEO_MASK = _os.environ.get('V4_GEO_MASK', '0') == '1'
V4_FAR_DEPTH = float(_os.environ.get('V4_FAR_DEPTH', '45.0'))
# ---- 解耦开关（2026-09-20 新增；默认值 = 原 V4_GEO_MASK 行为，不改动任何既有 run）----
# V4_SKY_GATE=0 保留天空参与建图与损失；V4_BG_WHITE=0 渲染背景保持黑色
V4_SKY_GATE = _os.environ.get('V4_SKY_GATE', '1' if V4_GEO_MASK else '0') == '1'
V4_BG_WHITE = _os.environ.get('V4_BG_WHITE', '1' if V4_GEO_MASK else '0') == '1'
# ---- P-85：按语义类放宽远场门控（默认关闭，行为完全不变）----
V4_FAR_CLS_IDS = [int(x) for x in _os.environ.get('V4_FAR_CLS_IDS', '').split(',') if x.strip()]
V4_FAR_CLS_DEPTH = float(_os.environ.get('V4_FAR_CLS_DEPTH', '65.0'))


def v4_far_invalid(depth_t, semantic_t=None):
    """远场无效掩码。V4_FAR_CLS_IDS 中的语义类放宽到 V4_FAR_CLS_DEPTH。"""
    inv = (depth_t >= V4_FAR_DEPTH)
    if V4_FAR_CLS_IDS and semantic_t is not None:
        in_cls = torch.zeros_like(inv)
        for _c in V4_FAR_CLS_IDS:
            in_cls = in_cls | (semantic_t == _c)
        inv = inv & ~(in_cls & (depth_t < V4_FAR_CLS_DEPTH))
    return inv

# P-58 depth-loss recalibration (mapping only, default OFF)
V4_DW_Z0 = float(_os.environ.get('V4_DW_Z0', '18.0'))
V4_DW_Z1 = float(_os.environ.get('V4_DW_Z1', '36.0'))
V4_DW_MIN = float(_os.environ.get('V4_DW_MIN', '1.0'))
V4_ONE_SIDE_DEPTH = _os.environ.get('V4_ONE_SIDE_DEPTH', '0') == '1'
# P-60: 前表面（softmin）深度监督；beta=0 表示关闭（零开销）
V4_FRONT_DEPTH_BETA = float(_os.environ.get('V4_FRONT_DEPTH_BETA', '0.0'))
V4_FRONT_DEPTH_CAP = float(_os.environ.get('V4_FRONT_DEPTH_CAP', '10.0'))
# P-63: 远场单边深度约束（把 45m 内不该有的几何往外推）；W=0 表示关闭
V4_FAR_PUSH_W = float(_os.environ.get('V4_FAR_PUSH_W', '0.0'))
V4_FAR_PUSH_SIL = float(_os.environ.get('V4_FAR_PUSH_SIL', '0.5'))
# P-64: 沿射线深度展布正则（抑制多层堆叠凑均值）；W=0 表示关闭
V4_OVERDRAW_W = float(_os.environ.get('V4_OVERDRAW_W', '0.0'))
V4_OVERDRAW_SIL = float(_os.environ.get('V4_OVERDRAW_SIL', '0.9'))
_P64_LAST_T = [-1]
_P63_LAST_T = [-1]
_P60_LAST_T = [-1]
if V4_DW_MIN < 1.0 or V4_ONE_SIDE_DEPTH:
    print(f'[P-58] depth-loss recalibrated: dw=({V4_DW_Z0},{V4_DW_Z1},{V4_DW_MIN}) one_side={V4_ONE_SIDE_DEPTH}', flush=True)
V4_SKY_ID = int(_os.environ.get('V4_SKY_ID', '24'))
# P-69: 天空/远场背景模块（方向条件全景图）。默认关闭
V4_SKY_MODULE = _os.environ.get('V4_SKY_MODULE', '0') == '1'
V4_SKY_W = float(_os.environ.get('V4_SKY_W', '1.0'))
V4_SKY_HW = tuple(int(x) for x in _os.environ.get('V4_SKY_HW', '32x128').split('x'))
V4_SKY_FAR = _os.environ.get('V4_SKY_FAR', '0') == '1'   # 远场(>FAR_DEPTH)也交给 sky 解释
V4_SKY_COV_W = float(_os.environ.get('V4_SKY_COV_W', '0.0'))   # P-69 反覆盖正则权重（治幕布）
_P69_LAST_T = [-1]
_SKY_KEEP_DBG = [False]   # 2026-10-10: mapping keep2d 天空剔除的一次性打印（结构判据）
_SEM_SNAP_DBG = [False, False]  # 2026-10-11: 调色板吸附首帧前后距离打印（结构判据）
# ---- v5 长程治理（2026-10-11 Gary 批准方法层改动；默认全关=原行为，逐项 env 显式开启）----
V4_CHAIN_INIT = _os.environ.get('V4_CHAIN_INIT', '0') == '1'
V4_SEM_CLAMP = _os.environ.get('V4_SEM_CLAMP', '0') == '1'
V4_KF_REACT_EVERY = int(_os.environ.get('V4_KF_REACT_EVERY', '0'))
V4_KF_REACT_NUM = int(_os.environ.get('V4_KF_REACT_NUM', '4'))
V4_CHAIN_REANCHOR = int(_os.environ.get('V4_CHAIN_REANCHOR', '0'))
V4_SEM_SPARSE = _os.environ.get('V4_SEM_SPARSE', '0') == '1'
if V4_SEM_SPARSE:
    print('[V6] GT 稀疏语义臂 ON：seg 损失只计官方 LiDAR 命中像素（semantic_color 非黑）', flush=True)
# ---- v7 语义调色板吸附（2026-10-11 Gary 批准方法层；治多视角 alpha 混色的中间色）----
# 每个建图帧结束后，把每个高斯的 semantic_colors 软拉向 21 类调色板最近色。
# 渲染语义的解码口径就是最近调色板色 ⇒ 颜色留在调色板吸引域内 = 跨视角解码稳定。
V4_SEM_SNAP = _os.environ.get('V4_SEM_SNAP', '0') == '1'
V4_SEM_SNAP_A = float(_os.environ.get('V4_SEM_SNAP_A', '0.1'))
# KITTI 21 类高可分配色（与 build_kitti_ds.PALETTE_RGB 逐位同源；数据集适配项，1.2 不设此 env 不受影响）
_SEM_PAL_21 = torch.tensor([
 (0,0,0),(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),
 (0,255,255),(255,128,0),(128,0,255),(0,128,255),(128,255,0),(255,0,128),
 (0,255,128),(128,0,0),(0,0,128),(128,128,0),(128,0,128),(0,128,0),
 (0,128,128),(200,200,200),(255,255,255)], dtype=torch.float32) / 255.0
if V4_SEM_SNAP:
    print(f'[V7] semantic palette snap ON alpha={V4_SEM_SNAP_A}', flush=True)
# ---- v8 语义锐化渲染（治射线级跨类混合：seg 通道透明度 γ 次幂，前表面主导）----
V4_SEM_SHARP = float(_os.environ.get('V4_SEM_SHARP', '0'))
if V4_SEM_SHARP > 1.0:
    print(f'[V8] semantic sharpened render ON gamma={V4_SEM_SHARP}', flush=True)
# ---- v9 远场门控（2026-10-11 Gary 要求自行重写；历史遗留 V4_GEO_MASK 禁用不再使用）----
# 语义：每帧只在 V9_FAR_GATE_D 米内建图/监督：
#   ① 帧0 初始点云不生成 D 米外高斯 ② 深度/跟踪损失不计 D 米外像素
#   ③ 新高斯插入不在 D 米外播种 ④（可选 V9_FAR_GATE_LOSS=1）D 米外像素同时剔出 mapping im/seg 损失
V9_FAR_GATE = _os.environ.get('V9_FAR_GATE', '0') == '1'
V9_FAR_GATE_D = float(_os.environ.get('V9_FAR_GATE_D', '45.0'))
V9_FAR_GATE_LOSS = _os.environ.get('V9_FAR_GATE_LOSS', '0') == '1'
if V9_FAR_GATE:
    print(f'[V9] 远场门控 ON D={V9_FAR_GATE_D}m | im/seg损失剔除={V9_FAR_GATE_LOSS}', flush=True)
if V4_CHAIN_INIT:
    print('[V5] 链式跟踪初始化 ON（上一帧估计 ⊕ INS 增量）', flush=True)
if V4_SEM_CLAMP:
    print('[V5] semantic_colors 钳位 [0,1] ON', flush=True)
if V4_KF_REACT_EVERY > 0:
    print(f'[V5] 老关键帧再激活 ON every={V4_KF_REACT_EVERY} num={V4_KF_REACT_NUM}', flush=True)
_SKY = {'map': None, 'opt': None}


def _sky_init(device):
    if not V4_SKY_MODULE:
        return
    if _SKY['map'] is None:
        _SKY['map'] = torch.nn.Parameter(
            torch.full((3, V4_SKY_HW[0], V4_SKY_HW[1]), 0.5, device=device))
        _SKY['opt'] = torch.optim.Adam([_SKY['map']], lr=1e-3)
        print('[P-69] sky module ON: map=%s lr=1e-3 far=%s' %
              ((3, V4_SKY_HW[0], V4_SKY_HW[1]), V4_SKY_FAR), flush=True)


def _sky_step():
    if _SKY['opt'] is not None and _SKY['map'] is not None and _SKY['map'].grad is not None:
        _SKY['opt'].step()
        _SKY['opt'].zero_grad(set_to_none=True)


# P-68: 各向异性协方差（log_scales 标量 -> 三分量）；默认关闭 = 旧行为
V4_ANISO = _os.environ.get('V4_ANISO', '0') == '1'

# P-79: tracking 的 depth loss 原用 .sum()（比 mapping 的 .mean() 大约 5 个数量级）
# ⇒ RGB 在 tracking 中被完全淹没。改 mean 让 RGB 重新参与位姿优化。
V4_TRACK_DEPTH_MEAN = _os.environ.get('V4_TRACK_DEPTH_MEAN', '0') == '1'

# P-78: 初始尺度系数 —— projective 初值固定为 1 像素（depth/focal），从未被审查过。
# 放大 N 倍 ⇒ 初始高斯更大 ⇒ 需更少点即可覆盖（验证"少而大" vs "多而小"）。
V4_INIT_PX = float(_os.environ.get('V4_INIT_PX', '1.0'))

# P-78b: 按深度自适应的初始像素足迹（组合方案 A）
# 近场保持 1px 保细节，远场放大省点数。FAR==NEAR 时等同原行为。
V4_INIT_PX_NEAR = float(_os.environ.get('V4_INIT_PX_NEAR', '1.0'))
V4_INIT_PX_FAR = float(_os.environ.get('V4_INIT_PX_FAR', '1.0'))
V4_INIT_PX_Z0 = float(_os.environ.get('V4_INIT_PX_Z0', '10.0'))
V4_INIT_PX_Z1 = float(_os.environ.get('V4_INIT_PX_Z1', '45.0'))

# P-80: tracking iter 级 dump（默认 OFF，纯旁路记录不影响训练）——
# V4_TRACK_DUMP=<路径前缀> 时启用。每帧写 <前缀>_t<帧号>.npz：
# iters / losses / trans(T,3) / rots(T,4)，用于尺度收缩(shrinkage)的 iter 级机制定位。
V4_TRACK_DUMP = _os.environ.get('V4_TRACK_DUMP', '')
if V4_ANISO:
    print('[P-68] anisotropic covariance ON: log_scales -> (N,3)', flush=True)

# --- P5 (rail prior): track-line prior on tracking poses ---
RAIL_PRIOR = _os.environ.get('RAIL_PRIOR', '0') == '1'
# RAIL_LINES: {t: (c_t(3,), d_t(3,))} in camera-t frame; d_t oriented +z (forward)
RAIL_LINES = {}
RAIL_REF = -1
RAIL_W_DIR = 10.0
RAIL_W_PT = 10000.0


def get_dataset(config_dict, basedir, sequence, **kwargs):
    if config_dict["dataset_name"].lower() in ["icl"]:
        return ICLDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["railway"]:
        return RailwayDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["replica"]:
        return ReplicaDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["replicav2"]:
        return ReplicaV2Dataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["azure", "azurekinect"]:
        return AzureKinectDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["scannet"]:
        return ScannetDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["ai2thor"]:
        return Ai2thorDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["record3d"]:
        return Record3DDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["realsense"]:
        return RealsenseDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["tum"]:
        return TUMDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["scannetpp"]:
        return ScannetPPDataset(basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["nerfcapture"]:
        return NeRFCaptureDataset(basedir, sequence, **kwargs)
    else:
        raise ValueError(f"Unknown dataset name {config_dict['dataset_name']}")


def get_pointcloud(color, depth, intrinsics, w2c, transform_pts=True, mask=None,
                   compute_mean_sq_dist=False, mean_sq_dist_method="projective", device="cuda",
                   load_semantics=False, semantic_id=None, semantic_color=None):
    width, height = color.shape[2], color.shape[1]
    CX = intrinsics[0][2]
    CY = intrinsics[1][2]
    FX = intrinsics[0][0]
    FY = intrinsics[1][1]

    # Compute indices of pixels
    x_grid, y_grid = torch.meshgrid(torch.arange(width).to(device).float(), 
                                    torch.arange(height).to(device).float(),
                                    indexing='xy')
    xx = (x_grid - CX)/FX
    yy = (y_grid - CY)/FY
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)
    depth_z = depth[0].reshape(-1)

    # Initialize point cloud
    pts_cam = torch.stack((xx * depth_z, yy * depth_z, depth_z), dim=-1)
    if transform_pts:
        pix_ones = torch.ones(height * width, 1).to(device).float()
        pts4 = torch.cat((pts_cam, pix_ones), dim=1)
        c2w = torch.inverse(w2c)
        pts = (c2w @ pts4.T).T[:, :3]
    else:
        pts = pts_cam

    # Compute mean squared distance for initializing the scale of the Gaussians
    if compute_mean_sq_dist:
        if mean_sq_dist_method == "projective":
            # Projective Geometry (this is fast, farther -> larger radius)
            # P-78: 初始尺度系数（默认 1.0 = 原行为：1 像素）
            if V4_INIT_PX_FAR != V4_INIT_PX_NEAR:
                # P-78b: 深度越大 → 初始足迹越大（远场不需要 1px 小点）
                _t = ((depth_z - V4_INIT_PX_Z0) / max(V4_INIT_PX_Z1 - V4_INIT_PX_Z0, 1e-6)).clamp(0.0, 1.0)
                _px = V4_INIT_PX_NEAR + (V4_INIT_PX_FAR - V4_INIT_PX_NEAR) * _t
            else:
                _px = V4_INIT_PX
            scale_gaussian = _px * depth_z / ((FX + FY)/2)
            mean3_sq_dist = scale_gaussian**2
        else:
            raise ValueError(f"Unknown mean_sq_dist_method {mean_sq_dist_method}")
        
    # Colorize point cloud
    cols = torch.permute(color, (1, 2, 0)).reshape(-1, 3) # (C, H, W) -> (H, W, C) -> (H * W, C)
    point_cld = torch.cat((pts, cols), -1)
    
    # Concat semantic label if load_semantics=True
    if load_semantics:
        semantic_id = torch.permute(semantic_id, (1, 2, 0)).reshape(-1, 1) # (1, H, W) -> (H, W, 1) -> (H * W, 1)
        semantic_color = torch.permute(semantic_color, (1, 2, 0)).reshape(-1, 3) # (3, H, W) -> (H, W, 3) -> (H * W, 3)
        point_cld = torch.cat((point_cld, semantic_id, semantic_color), -1)

    # Select points based on mask
    if mask is not None:
        point_cld = point_cld[mask]
        if compute_mean_sq_dist:
            mean3_sq_dist = mean3_sq_dist[mask]

    if compute_mean_sq_dist:
        return point_cld, mean3_sq_dist
    else:
        return point_cld

def initialize_params(init_pt_cld, num_frames, mean3_sq_dist, device, load_semantics=False):
    num_pts = init_pt_cld.shape[0]
    # channel 0-2 for 3d axis
    means3D = init_pt_cld[:, :3]
    # channel 3-5 for rgb colors
    rgb_colors = init_pt_cld[:, 3:6]
    unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1)) # [num_gaussians, 3]
    logit_opacities = torch.zeros((num_pts, 1), dtype=torch.float, device=device)
    
    params = {
        'means3D': means3D,
        'rgb_colors': rgb_colors,
        'unnorm_rotations': unnorm_rots,
        'logit_opacities': logit_opacities,
        # P-68: V4_ANISO=1 时三分量；否则仍是标量（与旧配置逐位一致）
        'log_scales': torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None],
                                 (1, 3 if V4_ANISO else 1)),
    }

    params_opt_exclude = set()
    if load_semantics:
        # Exclude semantic_ids from gradient
        params_opt_exclude.add('semantic_ids')
        # channel =6 for semantic id
        params['semantic_ids'] = init_pt_cld[:, 6]
        # Channel 7-9 for semantic colors
        params['semantic_colors'] = init_pt_cld[:, 7:10]

    # Initialize a single gaussian trajectory to model the camera poses relative to the first frame
    cam_rots = np.tile([1, 0, 0, 0], (1, 1))
    cam_rots = np.tile(cam_rots[:, :, None], (1, 1, num_frames))
    params['cam_unnorm_rots'] = cam_rots
    params['cam_trans'] = np.zeros((1, 3, num_frames))
    
    for k, v in params.items():
        if k not in params_opt_exclude:
            # Check if value is already a torch tensor
            if not isinstance(v, torch.Tensor):
                params[k] = torch.nn.Parameter(torch.tensor(v).to(device).float().contiguous().requires_grad_(True))
            else:
                params[k] = torch.nn.Parameter(v.to(device).float().contiguous().requires_grad_(True))

    variables = {'max_2D_radius': torch.zeros(params['means3D'].shape[0]).to(device).float(),
                 'means2D_gradient_accum': torch.zeros(params['means3D'].shape[0]).to(device).float(),
                 'denom': torch.zeros(params['means3D'].shape[0]).to(device).float(),
                 'timestep': torch.zeros(params['means3D'].shape[0]).to(device).float()}

    return params, variables, params_opt_exclude


def initialize_optimizer(params, params_opt_exclude, lrs_dict, tracking):
    lrs = lrs_dict
    param_groups = [{'params': [v], 'name': k, 'lr': lrs[k]} for k, v in params.items() if k not in params_opt_exclude]
    if tracking:
        return torch.optim.Adam(param_groups)
    else:
        return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)


def initialize_first_timestep(dataset, num_frames, scene_radius_depth_ratio, mean_sq_dist_method, device="cuda",
                              densify_dataset=None, load_semantics=False, dc_dict=None):
    # Get RGB-D Data & Camera Parameters
    if load_semantics:
        color, depth, intrinsics, pose, semantic_id, semantic_color = dataset[0]
    else:
        color, depth, intrinsics, pose = dataset[0]

    # Process RGB-D Data
    color = color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
    depth = depth.permute(2, 0, 1) # (H, W, 1) -> (1, H, W)
    
    if load_semantics:
        semantic_id = semantic_id.permute(2, 0, 1) # (H, W, 1) -> (1, H, W)
        semantic_color = semantic_color.permute(2, 0, 1) / 255 # (H, W, 3) -> (3, H, W)
    else:
        semantic_id = None
        semantic_color = None
    # Process Camera Parameters
    intrinsics = intrinsics[:3, :3]
    w2c = torch.linalg.inv(pose)

    # Setup Camera
    cam = setup_camera(color.shape[2], color.shape[1], intrinsics.cpu().numpy(),
                       w2c.detach().cpu().numpy(), device=device)

    if densify_dataset is not None:
        # Get Densification RGB-D Data & Camera Parameters
        color, depth, densify_intrinsics, _ = densify_dataset[0]
        color = color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
        depth = depth.permute(2, 0, 1) # (H, W, 1) -> (1, H, W)
        densify_intrinsics = densify_intrinsics[:3, :3]
        densify_cam = setup_camera(color.shape[2], color.shape[1], densify_intrinsics.cpu().numpy(),
                                   w2c.detach().cpu().numpy(), device=device)
    else:
        densify_intrinsics = intrinsics

    # Get Initial Point Cloud (PyTorch CUDA Tensor)
    mask = (depth > 0) # Mask out invalid depth values
    # 2026-10-10 Gary: 天空掩膜与 GEO_MASK 解耦 —— V4_SKY_GATE 单独驱动（KITT154 适配，
    # 只保留天空剔除，不带远场剔除）。既有组合行为逐位不变：
    #   GEO_MASK=1&SKY_GATE=1 ⇒ 远场∪天空剔除（与原实现等价）；SKY_GATE=0 ⇒ 不含天空项。
    if V4_GEO_MASK:
        mask = mask & (~(depth >= V4_FAR_DEPTH))
    if load_semantics and semantic_id is not None and V4_SKY_GATE:
        mask = mask & (~(semantic_id == V4_SKY_ID))
    # v9 远场门控①：初始点云（含一次性结构判据打印）
    if V9_FAR_GATE:
        _v9m = (depth < V9_FAR_GATE_D)
        print(f'[V9] init: D={V9_FAR_GATE_D}m 外像素={int((~_v9m).sum())} '
              f'({100 * float((~_v9m).float().mean()):.1f}%) 被剔除', flush=True)
        mask = mask & _v9m
    mask = mask.reshape(-1)
    # P-43: 剔除孤立的假近值深度（远景/建筑区的浮空物来源）
    if dc_dict is not None and dc_dict.get('enable', False):
        _out = depth_local_outlier_mask(depth, ratio=dc_dict.get('ratio', 0.5),
                                        k=dc_dict.get('kernel', 9))
        print(f"[P43] init 剔除离群深度像素={int(_out.sum().item())} "
              f"(ratio={dc_dict.get('ratio', 0.5)}, k={dc_dict.get('kernel', 9)})", flush=True)
        mask = mask & (~_out)
    init_pt_cld, mean3_sq_dist = get_pointcloud(color, depth, densify_intrinsics,
                                                w2c, mask=mask, compute_mean_sq_dist=True, 
                                                mean_sq_dist_method=mean_sq_dist_method, device=device,
                                                load_semantics=load_semantics, semantic_id=semantic_id,
                                                semantic_color=semantic_color)

    # Initialize Parameters
    params, variables, params_opt_exclude = initialize_params(init_pt_cld, num_frames, mean3_sq_dist, device,
                                                              load_semantics)

    # Initialize an estimate of scene radius for Gaussian-Splatting Densification
    variables['scene_radius'] = torch.max(depth)/scene_radius_depth_ratio

    if densify_dataset is not None:
        return params, variables, intrinsics, w2c, cam, params_opt_exclude, densify_intrinsics, densify_cam
    else:
        return params, variables, intrinsics, w2c, cam, params_opt_exclude


def transformed_params2frontdepth(params, w2c, transformed_pts, device="cuda"):
    """P-60: 前表面（softmin）深度的渲染变量。

    color 通道承载 [d*e, e, 0]，e = exp(-beta*(d - d_ref))，d 为相机系深度。
    渲染后 front_depth = I0 / I1 = Σw·d·e / Σw·e，随 beta 增大趋向前表面。
    """
    pts4 = torch.cat((transformed_pts, torch.ones_like(transformed_pts[:, :1])), dim=-1)
    pts_in_cam = (w2c @ pts4.T).T
    d = pts_in_cam[:, 2].clamp(min=0.0)
    with torch.no_grad():
        d_ref = d.min()
    expo = torch.clamp(V4_FRONT_DEPTH_BETA * (d - d_ref), max=V4_FRONT_DEPTH_CAP)
    e = torch.exp(-expo)
    colors = torch.stack([d * e, e, torch.zeros_like(e)], dim=-1)
    return {
        'means3D': transformed_pts,
        'colors_precomp': colors,
        'rotations': F.normalize(params['unnorm_rotations']),
        'opacities': torch.sigmoid(params['logit_opacities']),
        'scales': aniso_scales(params),
        'means2D': torch.zeros_like(params['means3D'], requires_grad=True, device=device) + 0
    }


def get_loss(params, curr_data, variables, iter_time_idx, loss_weights, use_sil_for_loss, sil_thres,
             use_l1, ignore_outlier_depth_loss, tracking=False, mapping=False, do_ba=False, device="cuda",
             plot_dir=None, visualize_tracking_loss=False, tracking_iteration=None, load_semantics=False):
    # Initialize Loss Dictionary
    losses = {}

    if tracking:
        # Get current frame Gaussians, where only the camera pose gets gradient
        transformed_pts = transform_to_frame(params, iter_time_idx, gaussians_grad=False,
                                             camera_grad=True, device=device)
    elif mapping:
        if do_ba:
            # Get current frame Gaussians, where both camera pose and Gaussians get gradient
            transformed_pts = transform_to_frame(params, iter_time_idx, gaussians_grad=True,
                                                 camera_grad=True, device=device)
        else:
            # Get current frame Gaussians, where only the Gaussians get gradient
            transformed_pts = transform_to_frame(params, iter_time_idx, gaussians_grad=True,
                                                 camera_grad=False, device=device)
    else:
        # Get current frame Gaussians, where only the Gaussians get gradient
        transformed_pts = transform_to_frame(params, iter_time_idx, gaussians_grad=True,
                                             camera_grad=False, device=device)

    # Initialize Render Variables
    rendervar = transformed_params2rendervar(params, transformed_pts, device=device)
    depth_sil_rendervar = transformed_params2depthplussilhouette(params, curr_data['w2c'],
                                                                 transformed_pts, device=device)
    # RGB Rendering
    rendervar['means2D'].retain_grad()
    im, radius, _, = Renderer(raster_settings=curr_data['cam'])(**rendervar)
    variables['means2D'] = rendervar['means2D']  # Gradient only accum from colour render for densification

    # Depth & Silhouette Rendering
    depth_sil, _, _, = Renderer(raster_settings=curr_data['cam'])(**depth_sil_rendervar)
    depth = depth_sil[0, :, :].unsqueeze(0)
    silhouette = depth_sil[1, :, :]
    presence_sil_mask = (silhouette > sil_thres)
    depth_sq = depth_sil[2, :, :].unsqueeze(0)
    uncertainty = depth_sq - depth**2
    uncertainty = uncertainty.detach()

    # P-60: 前表面（softmin）深度渲染（env 门控，beta=0 时不渲染）
    front_depth = None
    if V4_FRONT_DEPTH_BETA > 0.0:
        _frv = transformed_params2frontdepth(params, curr_data['w2c'], transformed_pts, device=device)
        _fs, _, _, = Renderer(raster_settings=curr_data['cam'])(**_frv)
        front_depth = _fs[0, :, :].unsqueeze(0) / _fs[1, :, :].unsqueeze(0).clamp(min=1e-4)

    # Semantic colors Rendering
    if load_semantics:
        semantic_rendervar = transformed_semantics2rendervar(params, transformed_pts, device=device)
        if V4_SEM_SHARP > 1.0:
            # v8: 锐化 seg 通道透明度（只改语义渲染，不动 RGB/深度）：
            #   o^γ 后前表面高斯主导混合，射线级跨类平均被抑制；
            #   反向传播仍经过 o（可导），但来自 seg 的梯度只流向可见前表面。
            semantic_rendervar['opacities'] = semantic_rendervar['opacities'].clamp(1e-5, 1.0) ** V4_SEM_SHARP
        rendered_seg, _, _, = Renderer(raster_settings=curr_data['cam'])(**semantic_rendervar)

    # Mask with valid depth values (accounts for outlier depth values)
    nan_mask = (~torch.isnan(depth)) & (~torch.isnan(uncertainty))
    if ignore_outlier_depth_loss:
        depth_error = torch.abs(curr_data['depth'] - depth) * (curr_data['depth'] > 0)
        mask = (depth_error < 10*depth_error.median())
        mask = mask & (curr_data['depth'] > 0)
    else:
        mask = (curr_data['depth'] > 0)
    mask = mask & nan_mask
    # 2026-10-10 Gary: 天空掩膜与 GEO_MASK 解耦（同 init 处注释）
    if V4_GEO_MASK:
        v4_invalid = v4_far_invalid(curr_data['depth'],
            (curr_data.get('semantic_id') if load_semantics else None))
        mask = mask & (~v4_invalid)
    if load_semantics and ('semantic_id' in curr_data) and V4_SKY_GATE:
        mask = mask & (~(curr_data['semantic_id'] == V4_SKY_ID))
    # v9 远场门控②：深度/跟踪损失不计 D 米外像素
    if V9_FAR_GATE:
        mask = mask & (curr_data['depth'] < V9_FAR_GATE_D)
    # Mask with presence silhouette mask (accounts for empty space)
    if tracking and use_sil_for_loss:
        mask = mask & presence_sil_mask

    # Depth loss
    if use_l1:
        mask = mask.detach()
        if tracking:
            if V4_TRACK_DEPTH_MEAN:
                # P-79: 与 mapping 同口径，让 im 项重新有话语权
                losses['depth'] = torch.abs(curr_data['depth'] - depth)[mask].mean()
            else:
                losses['depth'] = torch.abs(curr_data['depth'] - depth)[mask].sum()
        elif V4_DW_MIN < 1.0:
            # P-58: far-depth downweight; w depends on GT only -> constant wrt params
            _w = torch.clamp((V4_DW_Z1 - curr_data['depth']) / (V4_DW_Z1 - V4_DW_Z0),
                             min=V4_DW_MIN, max=1.0).detach()
            losses['depth'] = (torch.abs(curr_data['depth'] - depth) * _w)[mask].sum() \
                / _w[mask].sum().clamp(min=1.0)
        elif V4_ONE_SIDE_DEPTH:
            # P-58b: penalize only rd>gt (map too far); no reward for dilution when too near
            losses['depth'] = torch.relu(depth - curr_data['depth'])[mask].mean()
        elif V4_FRONT_DEPTH_BETA > 0.0 and front_depth is not None:
            # P-60: 用 softmin（前表面）深度监督，切断"往后加点稀释均值"的捷径
            losses['depth'] = torch.abs(curr_data['depth'] - front_depth)[mask].mean()
            if _P60_LAST_T[0] != int(iter_time_idx) and int(mask.sum()) > 0:
                _P60_LAST_T[0] = int(iter_time_idx)
                print(f"[P60] t={int(iter_time_idx)} beta={V4_FRONT_DEPTH_BETA:.2f} "
                      f"gt={curr_data['depth'][mask].mean().item():.2f} "
                      f"front={front_depth[mask].mean().item():.2f} "
                      f"mean={depth[mask].mean().item():.2f} "
                      f"|e_front|={torch.abs(curr_data['depth'] - front_depth)[mask].mean().item():.2f} "
                      f"|e_mean|={torch.abs(curr_data['depth'] - depth)[mask].mean().item():.2f}", flush=True)
        else:
            losses['depth'] = torch.abs(curr_data['depth'] - depth)[mask].mean()
    
    # P-63: 远场单边深度约束
    # 只对「已知 45m 内没有东西」且「该像素已经有几何」的像素施加：只推走，不催生
    if (not tracking) and V4_GEO_MASK and V4_FAR_PUSH_W > 0.0:
        with torch.no_grad():
            _inv = (curr_data['depth'][0, :, :] >= V4_FAR_DEPTH)
            if load_semantics and ('semantic_id' in curr_data):
                _inv = _inv | (curr_data['semantic_id'][0, :, :] == V4_SKY_ID)
            _pm = _inv & (silhouette > V4_FAR_PUSH_SIL)
        _n63 = int(_pm.sum().item())
        if _n63 > 0:
            _dn = depth[0] / silhouette.clamp(min=1e-4)   # 归一化渲染深度 = Σw·d / Σw
            _viol = torch.relu(V4_FAR_DEPTH - _dn)[_pm]
            loss_weights.setdefault('far_push', 1.0)
            losses['far_push'] = V4_FAR_PUSH_W * _viol.mean()
            if _P63_LAST_T[0] != int(iter_time_idx):
                _P63_LAST_T[0] = int(iter_time_idx)
                _nv = int((_viol > 0).sum().item())
                print(f"[P63] t={int(iter_time_idx)} n={_n63} 违规率={_nv / max(_n63, 1):.2f} "
                      f"渲染深度中位={float(_dn[_pm].median().item()):.2f} "
                      f"违规深度中位={float(_dn[_pm][_viol > 0].median().item()) if _nv > 0 else -1:.2f} "
                      f"loss={float(_viol.mean().item()):.3f}", flush=True)

    # P-64: 沿射线深度展布正则（只在完全覆盖且 GT 有效的像素上施加）
    if (not tracking) and V4_OVERDRAW_W > 0.0:
        _gt2 = curr_data['depth'][0, :, :]
        with torch.no_grad():
            _cov = (silhouette > V4_OVERDRAW_SIL) & (_gt2 > 0)
            if V4_GEO_MASK:
                _cov = _cov & (_gt2 < V4_FAR_DEPTH)
        _n64 = int(_cov.sum().item())
        if _n64 > 0:
            _unc = uncertainty[0] if uncertainty.dim() == 3 else uncertainty
            _sp = torch.sqrt(torch.relu(_unc))[_cov]
            loss_weights.setdefault('overdraw', 1.0)
            losses['overdraw'] = V4_OVERDRAW_W * _sp.mean()
            if _P64_LAST_T[0] != int(iter_time_idx):
                _P64_LAST_T[0] = int(iter_time_idx)
                _k = max(int(0.9 * _n64), 1)
                print(f"[P64] t={int(iter_time_idx)} n={_n64} "
                      f"展布均值={float(_sp.mean().item()):.2f} "
                      f"p90={float(torch.kthvalue(_sp, _k)[0].item()):.2f} "
                      f"loss={float(_sp.mean().item()):.3f}", flush=True)

    # P-69: 天空/远场背景合成（只作用 mapping；tracking 路径逐位不变）
    _sky_im = None
    if V4_SKY_MODULE and (not tracking) and (_SKY['map'] is not None):
        _sky_im = sky_render(_SKY['map'], curr_data['im'].shape[1], curr_data['im'].shape[2],
                             curr_data['intrinsics'],
                             params['cam_unnorm_rots'][..., iter_time_idx].detach(), device=device)
        # fix1: 硬合成 —— 天空像素由 sky 模型解释（覆盖度已是 1.0，加法合成等于空转）
        with torch.no_grad():
            _m69 = (curr_data['semantic_id'][0, :, :] == V4_SKY_ID) if ('semantic_id' in curr_data) \
                else torch.zeros_like(curr_data['depth'][0], dtype=torch.bool)
            if V4_SKY_FAR:
                _m69 = _m69 | (curr_data['depth'][0, :, :] >= V4_FAR_DEPTH)
            _m69f = _m69.float().unsqueeze(0)
        im = im * (1.0 - _m69f) + _sky_im * _m69f

    # RGB Loss
    if tracking and (use_sil_for_loss or ignore_outlier_depth_loss):
        color_mask = torch.tile(mask, (3, 1, 1))
        color_mask = color_mask.detach()
        losses['im'] = torch.abs(curr_data['im'] - im)[color_mask].sum()
        if load_semantics:
            if V4_SEM_SPARSE:
                _ms = (curr_data['semantic_color'].sum(0) > 0)
                losses['seg'] = torch.abs(curr_data['semantic_color'] - rendered_seg)[:, _ms].sum()
            else:
                losses['seg'] = torch.abs(curr_data['semantic_color'] - rendered_seg)[color_mask].sum()
    elif tracking:
        losses['im'] = torch.abs(curr_data['im'] - im).sum()
        if load_semantics:
            if V4_SEM_SPARSE:
                _ms = (curr_data['semantic_color'].sum(0) > 0)
                losses['seg'] = torch.abs(curr_data['semantic_color'] - rendered_seg)[:, _ms].sum()
            else:
                losses['seg'] = torch.abs(curr_data['semantic_color'] - rendered_seg).sum()
    else:
        # 2026-10-10 Gary: 分支条件加 V4_SKY_GATE —— 天空像素可脱离 GEO_MASK 单独
        # 从 mapping 的 im/seg 损失中剔除（KITTI sky id=154 适配）。
        # SKY_GATE=0&GEO_MASK=0 仍走原 else（1.2 既有 run 逐位不变）。
        if V4_GEO_MASK or V4_SKY_GATE:
            v4_sky = (curr_data['semantic_id'] == V4_SKY_ID) if (V4_SKY_GATE and load_semantics and ('semantic_id' in curr_data)) \
                else torch.zeros_like(curr_data['depth'], dtype=torch.bool)
            v4_keep2d = (~v4_sky)[0]
            if V4_SKY_GATE and not _SKY_KEEP_DBG[0]:
                _SKY_KEEP_DBG[0] = True
                _n_sky = int(v4_sky.sum().item())
                print(f"[SKY] mapping keep2d 生效: V4_SKY_ID={V4_SKY_ID} 天空像素={_n_sky} "
                      f"({_n_sky / max(v4_sky.numel(), 1) * 100:.1f}%) 被排除出 im/seg 损失", flush=True)
            if V4_GEO_MASK and V4_SKY_MODULE and V4_SKY_FAR:
                # 远场像素交给 sky 模块解释 -> 几何不再被要求"画"出远处背景
                v4_keep2d = v4_keep2d & (curr_data['depth'][0] < V4_FAR_DEPTH)
            # v9 远场门控④（可选子开关）：D 米外像素剔出 im/seg 损失
            if V9_FAR_GATE and V9_FAR_GATE_LOSS:
                v4_keep2d = v4_keep2d & (curr_data['depth'][0] < V9_FAR_GATE_D)
            losses['im'] = 0.8 * torch.abs(im - curr_data['im'])[:, v4_keep2d].mean() \
                + 0.2 * (1.0 - calc_ssim(im, curr_data['im']))
            if load_semantics:
                if V4_SEM_SPARSE:
                    v4_keepseg = v4_keep2d & (curr_data['semantic_color'].sum(0) > 0)
                    losses['seg'] = 0.8 * torch.abs(rendered_seg - curr_data['semantic_color'])[:, v4_keepseg].mean()
                else:
                    losses['seg'] = 0.8 * torch.abs(rendered_seg - curr_data['semantic_color'])[:, v4_keep2d].mean() \
                        + 0.2 * (1.0 - calc_ssim(rendered_seg, curr_data['semantic_color']))
        else:
            losses['im'] = 0.8 * l1_loss_v1(im, curr_data['im']) + 0.2 * (1.0 - calc_ssim(im, curr_data['im']))
            if load_semantics:
                losses['seg'] = 0.8 * l1_loss_v1(rendered_seg, curr_data['semantic_color']) \
                    + 0.2 * (1.0 - calc_ssim(rendered_seg, curr_data['semantic_color']))

    # P-69 fix1: 反覆盖正则 —— 直接把「几何占用天空区」这件事罚掉（治远场幕布）
    if V4_SKY_COV_W > 0.0 and _sky_im is not None:
        loss_weights.setdefault('sky_cov', 1.0)
        losses['sky_cov'] = V4_SKY_COV_W * silhouette[_m69].mean()

    # P-69: 天空模块监督（只监督天空像素上的 sky_map，不通过 (1-Λ) 回流给几何）
    if _sky_im is not None and load_semantics and ('semantic_id' in curr_data):
        _skm = _m69   # fix1: 与合成用的掩码完全一致
        _nsk = int(_skm.sum().item())
        if _nsk > 0:
            loss_weights.setdefault('sky', 1.0)
            losses['sky'] = V4_SKY_W * torch.abs(_sky_im - curr_data['im'])[:, _skm].mean()
            if _P69_LAST_T[0] != int(iter_time_idx):
                _P69_LAST_T[0] = int(iter_time_idx)
                print('[P69] t=%d sky_px=%d (%.1f%%) coverage_in_sky=%.3f loss=%.4f' %
                      (int(iter_time_idx), _nsk, 100.0 * _nsk / _skm.numel(),
                       float(silhouette[_skm].mean().item()), float(losses['sky'].item())), flush=True)

    # Visualize the Diff Images
    if tracking and visualize_tracking_loss:
        fig, ax = plt.subplots(2, 4, figsize=(12, 6))
        weighted_render_im = im * color_mask
        weighted_im = curr_data['im'] * color_mask
        weighted_render_depth = depth * mask
        weighted_depth = curr_data['depth'] * mask
        diff_rgb = torch.abs(weighted_render_im - weighted_im).mean(dim=0).detach().cpu()
        diff_depth = torch.abs(weighted_render_depth - weighted_depth).mean(dim=0).detach().cpu()
        viz_img = torch.clip(weighted_im.permute(1, 2, 0).detach().cpu(), 0, 1)
        ax[0, 0].imshow(viz_img)
        ax[0, 0].set_title("Weighted GT RGB")
        viz_render_img = torch.clip(weighted_render_im.permute(1, 2, 0).detach().cpu(), 0, 1)
        ax[1, 0].imshow(viz_render_img)
        ax[1, 0].set_title("Weighted Rendered RGB")
        ax[0, 1].imshow(weighted_depth[0].detach().cpu(), cmap="jet", vmin=0, vmax=6)
        ax[0, 1].set_title("Weighted GT Depth")
        ax[1, 1].imshow(weighted_render_depth[0].detach().cpu(), cmap="jet", vmin=0, vmax=6)
        ax[1, 1].set_title("Weighted Rendered Depth")
        ax[0, 2].imshow(diff_rgb, cmap="jet", vmin=0, vmax=0.8)
        ax[0, 2].set_title(f"Diff RGB, Loss: {torch.round(losses['im'])}")
        ax[1, 2].imshow(diff_depth, cmap="jet", vmin=0, vmax=0.8)
        ax[1, 2].set_title(f"Diff Depth, Loss: {torch.round(losses['depth'])}")
        ax[0, 3].imshow(presence_sil_mask.detach().cpu(), cmap="gray")
        ax[0, 3].set_title("Silhouette Mask")
        ax[1, 3].imshow(mask[0].detach().cpu(), cmap="gray")
        ax[1, 3].set_title("Loss Mask")
        # Turn off axis
        for i in range(2):
            for j in range(4):
                ax[i, j].axis('off')
        # Set Title
        fig.suptitle(f"Tracking Iteration: {tracking_iteration}", fontsize=16)
        # Figure Tight Layout
        fig.tight_layout()
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, f"tmp.png"), bbox_inches='tight')
        plt.close()
        plot_img = cv2.imread(os.path.join(plot_dir, f"tmp.png"))
        cv2.imshow('Diff Images', plot_img)
        cv2.waitKey(1)
        ## Save Tracking Loss Viz
        # save_plot_dir = os.path.join(plot_dir, f"tracking_%04d" % iter_time_idx)
        # os.makedirs(save_plot_dir, exist_ok=True)
        # plt.savefig(os.path.join(save_plot_dir, f"%04d.png" % tracking_iteration), bbox_inches='tight')
        # plt.close()

    # P5: rail-line prior (tracking only). Lambda baked in; weights pass-through below.
    if tracking and RAIL_PRIOR and iter_time_idx in RAIL_LINES and iter_time_idx != RAIL_REF:
        _c_t, _d_t = RAIL_LINES[iter_time_idx]
        _R_t = build_rotation(F.normalize(params['cam_unnorm_rots'][..., iter_time_idx])).reshape(3, 3)
        _p_t = params['cam_trans'][..., iter_time_idx].reshape(3)
        # 9-25: params 为 w2c(R,t)，cam->world 正确变换 = R^T(x - t)（旧 R@x+p 为 c2w 约定，对 d 系列参数错误）
        _c_p = _R_t.T @ (_c_t.reshape(3) - _p_t)
        _d_p = _R_t.T @ _d_t.reshape(3)
        _R_r = build_rotation(F.normalize(params['cam_unnorm_rots'][..., RAIL_REF])).reshape(3, 3)
        _p_r = params['cam_trans'][..., RAIL_REF].reshape(3)
        _c_r, _d_r = RAIL_LINES[RAIL_REF]
        _c_ref = _R_r.T @ (_c_r.reshape(3) - _p_r)
        _d_ref = _R_r.T @ _d_r.reshape(3)
        losses['rail_dir'] = RAIL_W_DIR * (1.0 - (_d_p @ _d_ref) ** 2)
        _e = _c_p - _c_ref
        losses['rail_pt'] = RAIL_W_PT * ((_e - (_e @ _d_ref) * _d_ref) ** 2).sum()
        if tracking_iteration == 0:
            print('[P5] rail_dir=%.4f rail_pt=%.4f (frame %d)' %
                  (losses['rail_dir'].item(), losses['rail_pt'].item(), iter_time_idx), flush=True)
        loss_weights.setdefault('rail_dir', 1.0)
        loss_weights.setdefault('rail_pt', 1.0)

    weighted_losses = {k: v * loss_weights[k] for k, v in losses.items()}
    loss = sum(weighted_losses.values())

    seen = radius > 0
    variables['max_2D_radius'][seen] = torch.max(radius[seen], variables['max_2D_radius'][seen])
    variables['seen'] = seen
    weighted_losses['loss'] = loss

    return loss, variables, weighted_losses


def initialize_new_params(new_pt_cld, mean3_sq_dist, device, load_semantics=False,
                          params_opt_exclude=None):
    num_pts = new_pt_cld.shape[0]
    means3D = new_pt_cld[:, :3] # [num_gaussians, 3]
    unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1)) # [num_gaussians, 3]
    logit_opacities = torch.zeros((num_pts, 1), dtype=torch.float, device=device)
    params = {
        'means3D': means3D,
        'rgb_colors': new_pt_cld[:, 3:6],
        'unnorm_rotations': unnorm_rots,
        'logit_opacities': logit_opacities,
        # P-68: V4_ANISO=1 时三分量；否则仍是标量（与旧配置逐位一致）
        'log_scales': torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None],
                                 (1, 3 if V4_ANISO else 1)),
    }

    if load_semantics:
        params['semantic_ids'] = new_pt_cld[:, 6]
        params['semantic_colors'] = new_pt_cld[:, 7:10]

    for k, v in params.items():
        if k not in params_opt_exclude:
            # Check if value is already a torch tensor
            if not isinstance(v, torch.Tensor):
                params[k] = torch.nn.Parameter(torch.tensor(v).to(device).float().contiguous().requires_grad_(True))
            else:
                params[k] = torch.nn.Parameter(v.to(device).float().contiguous().requires_grad_(True))

    return params


_P62_LOG = {}


def _p62_stopped(add_dict, time_idx, path):
    """P-62：late_stop_frame 之后停止指定播种通路。默认关闭。"""
    if not add_dict:
        return False
    k = int(add_dict.get('late_stop_frame', -1))
    if k < 0 or time_idx < k:
        return False
    if path not in add_dict.get('late_stop_paths', ['far_seed']):
        return False
    key = (int(time_idx), path)
    if key not in _P62_LOG:
        _P62_LOG[key] = 1
        print(f"[P62] t={int(time_idx)} late_stop 命中 -> 跳过 {path} (frame>={k})", flush=True)
    return True



# ---------- P-66 点数预算（默认关闭）----------
_P66_BUDGET = int(os.environ.get("V4_POINT_BUDGET", "0"))   # 0 = 不启用
_P66_LOG = set()


def _p66_blocked(params, tag, t):
    """点数达到预算后停止一切新增（播种 + 致密化）。"""
    if _P66_BUDGET <= 0:
        return False
    n = int(params["means3D"].shape[0])
    if n >= _P66_BUDGET:
        key = (tag, t // 10)
        if key not in _P66_LOG:
            _P66_LOG.add(key)
            print("[P66] t=%d %s 点数=%d >= 预算=%d -> 跳过新增" % (t, tag, n, _P66_BUDGET))
        return True
    return False
# ---------- P-66 end ----------


def add_new_gaussians(params, params_opt_exclude, variables, curr_data, sil_thres, time_idx,
                      mean_sq_dist_method, device="cuda", load_semantics=False, add_dict=None,
                      dc_dict=None):
    # Silhouette Rendering
    transformed_pts = transform_to_frame(params, time_idx, gaussians_grad=False,
                                         camera_grad=False, device=device)
    depth_sil_rendervar = transformed_params2depthplussilhouette(params, curr_data['w2c'],
                                                                 transformed_pts, device=device)
    depth_sil, _, _, = Renderer(raster_settings=curr_data['cam'])(**depth_sil_rendervar)
    silhouette = depth_sil[1, :, :]
    non_presence_sil_mask = (silhouette < sil_thres)
    # Check for new foreground objects by using GT depth
    gt_depth = curr_data['depth'][0, :, :]
    render_depth = depth_sil[0, :, :]
    depth_error = torch.abs(gt_depth - render_depth) * (gt_depth > 0)
    non_presence_depth_mask = (render_depth > gt_depth) * (depth_error > 50*depth_error.median())
    # Determine non-presence mask
    non_presence_mask = non_presence_sil_mask | non_presence_depth_mask
    # Flatten mask
    # P-35: 绝对深度误差触发 —— silhouette 饱和（alpha 被海量高斯撑满）时仍能发现"地图没跟上眼前"
    if add_dict is not None and add_dict.get('enable', False) and not _p62_stopped(add_dict, time_idx, 'depth_err'):
        _gt = curr_data['depth'][0, :, :]
        _rd = depth_sil[0, :, :]
        _err = torch.abs(_gt - _rd)
        _thr = torch.clamp(add_dict.get('rel_thresh', 0.1) * _gt, min=add_dict.get('abs_thresh', 1.0))
        _m_all = (_err > _thr) & (_gt > 0)
        # P-57 单向触发：只在「地图比真实更远」(rd > gt，= 眼前有东西没建出来) 时补点。
        # 依据（09-13 取证）：q5 帧95 渲染深度中位 20.15m vs GT 25.04m（地图整体偏近 ~5m），
        # 而 P-35 原实现用 |gt-rd| 双向触发 → 每帧都在「已经建多了」的地方继续加 2.5 万个点，
        # 加在遮挡面之后的点无法改善该像素的外观（前表面主导），只能堆出冗余壳层
        # （帧99 过绘制 18.95 层、NEW_PTS 100 帧全程 40k~58k/帧、上限被打满）。
        if add_dict.get("one_sided", False):
            _tc = _m_all & ~(_rd > _gt)          # 「地图偏近」方向（= 均值深度稀释路径）
            _m = _m_all & (_rd > _gt)
            _n_tooclose = int(_tc.sum().item())
            _r = float(add_dict.get("too_close_ratio", 0.0))   # P-57b 剂量控制：保留比例
            if _r > 0 and _n_tooclose > 0:
                _keep = _tc & (torch.rand_like(_gt) < _r)
                _m = _m | _keep
        else:
            _m = _m_all
            _n_tooclose = -1
        if V4_GEO_MASK:
            _m = _m & (_gt < V4_FAR_DEPTH)
            if load_semantics and (curr_data.get('semantic_id') is not None):
                _m = _m & (curr_data['semantic_id'][0, :, :] != V4_SKY_ID)
        _idx = torch.nonzero(_m.reshape(-1), as_tuple=False).reshape(-1)
        _cap = add_dict.get('max_per_frame', 150000)
        _n_pre = int(_idx.numel())
        if _n_pre > _cap:
            _idx = _idx[torch.randperm(_n_pre, device=_idx.device)[:_cap]]
        _m2 = torch.zeros_like(_m)
        _m2.reshape(-1)[_idx] = True
        non_presence_mask = non_presence_mask | _m2
        print(f"[P35] t={time_idx} depth_err_pixels={_n_pre} capped_to={int(_m2.sum().item())} "
              f"one_sided={add_dict.get('one_sided', False)} dropped_too_close={_n_tooclose}", flush=True)

    non_presence_mask = non_presence_mask.reshape(-1)

    # P-34 debug: 记录新增高斯的每一步筛子
    _n_sil = int(non_presence_sil_mask.reshape(-1).sum().item())
    _n_dep = int(non_presence_depth_mask.reshape(-1).sum().item())
    _n_mask0 = int(non_presence_mask.sum().item())
    _gt_d = curr_data['depth'][0, :, :]
    _n_valid = int((_gt_d > 0).sum().item())
    _n_far = int((_gt_d >= V4_FAR_DEPTH).sum().item()) if V4_GEO_MASK else -1

    # Get the new frame Gaussians based on the Silhouette
    if torch.sum(non_presence_mask) > 0:
        # Get the new pointcloud in the world frame
        curr_cam_rot = torch.nn.functional.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
        curr_cam_tran = params['cam_trans'][..., time_idx].detach()
        curr_w2c = torch.eye(4).to(device).float()
        curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
        curr_w2c[:3, 3] = curr_cam_tran
        valid_depth_mask = (curr_data['depth'][0, :, :] > 0)
        non_presence_mask = non_presence_mask & valid_depth_mask.reshape(-1)
        # 2026-10-10 Gary: 天空项从 GEO_MASK 解耦（V4_SKY_GATE 单独驱动）
        if V4_GEO_MASK:
            v4_invalid = v4_far_invalid(
                curr_data['depth'][0, :, :],
                (curr_data['semantic_id'][0, :, :]
                 if (load_semantics and curr_data.get('semantic_id') is not None) else None))
            non_presence_mask = non_presence_mask & (~v4_invalid).reshape(-1)
        if load_semantics and (curr_data.get('semantic_id') is not None) and V4_SKY_GATE:
            _sky_m = (curr_data['semantic_id'][0, :, :] == V4_SKY_ID)
            non_presence_mask = non_presence_mask & (~_sky_m).reshape(-1)
        # v9 远场门控③：新高斯插入不在 D 米外播种
        if V9_FAR_GATE:
            non_presence_mask = non_presence_mask & (curr_data['depth'][0, :, :] < V9_FAR_GATE_D).reshape(-1)

        if load_semantics:
            semantic_id = curr_data['semantic_id']
            semantic_color = curr_data['semantic_color']
        else:
            semantic_id = None
            semantic_color = None

        # P-41: 远景播种 —— 无有效 GT 深度（>FAR 或缺失）的像素此前一律跳过，导致
        # 地图只覆盖相机周边十几米（实测到相机距离 p90=13.5m），远处站台/电杆/车辆永远不出现。
        # 这里给这些像素按固定远距离沿视线播种几何（天空语义除外）。
        _depth_for_pc = curr_data['depth']
        if add_dict is not None and add_dict.get('far_seed_enable', False) and not _p62_stopped(add_dict, time_idx, 'far_seed'):
            _d0 = curr_data['depth'][0, :, :]
            _inv = (_d0 <= 0)
            if V4_GEO_MASK:
                _inv = _inv | (_d0 >= V4_FAR_DEPTH)
                if load_semantics and (curr_data.get('semantic_id') is not None):
                    _inv = _inv & (curr_data['semantic_id'][0, :, :] != V4_SKY_ID)
            _idx = torch.nonzero(_inv.reshape(-1), as_tuple=False).reshape(-1)
            _cap = add_dict.get('far_seed_max_per_frame', 10000)
            _n = int(_idx.numel())
            if _n > _cap:
                _idx = _idx[torch.randperm(_n, device=_idx.device)[:_cap]]
            if _idx.numel() > 0:
                _fm = torch.zeros_like(non_presence_mask)
                _fm[_idx] = True
                non_presence_mask = non_presence_mask | _fm
                _dd = _d0.clone().reshape(-1)
                _dd[_idx] = add_dict.get('far_seed_depth', 80.0)
                _depth_for_pc = _dd.reshape(1, *_d0.shape)
            print(f"[P41] t={time_idx} far_seed={int(_idx.numel())} inv_px={_n}", flush=True)

        # P-45: 结构类强制播种（用真实 GT 深度，含 45m 以外）—— 解决"电杆/车辆/站台永远不出现"
        # 实测 1.2：FAR_DEPTH=45 排除掉 29% 像素（road_vehicle 中位 63.5m、building 65m 全部无效），
        # 电杆/信号机像素太少、误差触发也轮不到 → 这些目标从头到尾没有播种来源。
        if add_dict is not None and add_dict.get('struct_seed_enable', False) and not _p62_stopped(add_dict, time_idx, 'struct_seed'):
            with torch.no_grad():
                _sdmax = add_dict.get('struct_seed_depth_max', 120.0)
                _cls = add_dict.get('struct_seed_classes', [])
                _cap = add_dict.get('struct_seed_max_per_frame', 8000)
                _d0s = curr_data['depth'][0, :, :]
                _ok = (_d0s > 0) & (_d0s < _sdmax)
                _sid = curr_data.get('semantic_id') if load_semantics else None
                if _sid is not None and len(_cls) > 0:
                    _cm = torch.zeros_like(_d0s, dtype=torch.bool)
                    for _c in _cls:
                        _cm = _cm | (_sid[0, :, :] == _c)
                    _ok = _ok & _cm
                _idx2 = torch.nonzero(_ok.reshape(-1), as_tuple=False).reshape(-1)
                _n2 = int(_idx2.numel())
                if _n2 > _cap:
                    _idx2 = _idx2[torch.randperm(_n2, device=_idx2.device)[:_cap]]
                if _idx2.numel() > 0:
                    _fm2 = torch.zeros_like(non_presence_mask)
                    _fm2[_idx2] = True
                    non_presence_mask = non_presence_mask | _fm2
                print(f"[P45] t={time_idx} struct_seed={int(_idx2.numel())}/{_n2} "
                      f"(classes={_cls}, depth<{_sdmax})", flush=True)
                # P-47: 小目标（车辆/电杆/信号）单独足额播种，不与 building/track/ground 争抢同一配额
                _obj = add_dict.get('struct_seed_object_classes', [])
                _ocap = add_dict.get('object_seed_max_per_frame', 0)
                if _sid is not None and len(_obj) > 0 and _ocap > 0:
                    _cm3 = torch.zeros_like(_d0s, dtype=torch.bool)
                    for _c in _obj:
                        _cm3 = _cm3 | (_sid[0, :, :] == _c)
                    _ok3 = (_d0s > 0) & (_d0s < _sdmax) & _cm3
                    _idx3 = torch.nonzero(_ok3.reshape(-1), as_tuple=False).reshape(-1)
                    _n3 = int(_idx3.numel())
                    if _n3 > _ocap:
                        _idx3 = _idx3[torch.randperm(_n3, device=_idx3.device)[:_ocap]]
                    if _idx3.numel() > 0:
                        _fm3 = torch.zeros_like(non_presence_mask)
                        _fm3[_idx3] = True
                        non_presence_mask = non_presence_mask | _fm3
                    print(f"[P47] t={time_idx} object_seed={int(_idx3.numel())}/{_n3} "
                          f"(classes={_obj})", flush=True)

        # P-43: 新增时同样剔除"显著近于局部邻域"的孤立假近值（浮空物来源）
        if dc_dict is not None and dc_dict.get('enable', False):
            _out = depth_local_outlier_mask(_depth_for_pc, ratio=dc_dict.get('ratio', 0.5),
                                            k=dc_dict.get('kernel', 9))
            _nb = int(non_presence_mask.sum().item())
            non_presence_mask = non_presence_mask & (~_out)
            print(f"[P43] t={time_idx} 新增前{_nb} → 剔除离群{int((_out & ~non_presence_mask).sum().item())} "
                  f"→ {int(non_presence_mask.sum().item())}", flush=True)

        new_pt_cld, mean3_sq_dist = get_pointcloud(curr_data['im'], _depth_for_pc, curr_data['intrinsics'],
                                                   curr_w2c, mask=non_presence_mask, compute_mean_sq_dist=True,
                                                   mean_sq_dist_method=mean_sq_dist_method, device=device,
                                                   load_semantics=load_semantics, semantic_id=semantic_id,
                                                   semantic_color=semantic_color)
        new_params = initialize_new_params(new_pt_cld, mean3_sq_dist, device, load_semantics=load_semantics,
                                           params_opt_exclude=params_opt_exclude)
        for k, v in new_params.items():
            if k not in params_opt_exclude:
                params[k] = torch.nn.Parameter(torch.cat((params[k], v), dim=0).requires_grad_(True))
            else:
                params[k] = torch.cat((params[k], v), dim=0)
        num_pts = params['means3D'].shape[0]
        variables['means2D_gradient_accum'] = torch.zeros(num_pts, device=device).float()
        variables['denom'] = torch.zeros(num_pts, device=device).float()
        variables['max_2D_radius'] = torch.zeros(num_pts, device=device).float()
        new_timestep = time_idx*torch.ones(new_pt_cld.shape[0],device=device).float()
        variables['timestep'] = torch.cat((variables['timestep'],new_timestep),dim=0)
        print(f"[P34dbg] t={time_idx} sil={_n_sil} dep={_n_dep} mask0={_n_mask0} "
              f"valid={_n_valid} far(>={V4_FAR_DEPTH})={_n_far} AFTER_V4={int(non_presence_mask.sum().item())} "
              f"NEW_PTS={new_pt_cld.shape[0]}", flush=True)
    else:
        print(f"[P34dbg] t={time_idx} sil={_n_sil} dep={_n_dep} mask0=0 "
              f"valid={_n_valid} far(>={V4_FAR_DEPTH})={_n_far} NO_NEW_GAUSSIANS", flush=True)

    return params, variables


def initialize_camera_pose(params, curr_time_idx, forward_prop):
    with torch.no_grad():
        if curr_time_idx > 1 and forward_prop:
            # Initialize the camera pose for the current frame based on a constant velocity model
            # Rotation
            prev_rot1 = F.normalize(params['cam_unnorm_rots'][..., curr_time_idx-1].detach())
            prev_rot2 = F.normalize(params['cam_unnorm_rots'][..., curr_time_idx-2].detach())
            new_rot = F.normalize(prev_rot1 + (prev_rot1 - prev_rot2))
            params['cam_unnorm_rots'][..., curr_time_idx] = new_rot.detach()
            # Translation
            prev_tran1 = params['cam_trans'][..., curr_time_idx-1].detach()
            prev_tran2 = params['cam_trans'][..., curr_time_idx-2].detach()
            new_tran = prev_tran1 + (prev_tran1 - prev_tran2)
            params['cam_trans'][..., curr_time_idx] = new_tran.detach()
        else:
            # Initialize the camera pose for the current frame
            params['cam_unnorm_rots'][..., curr_time_idx] = params['cam_unnorm_rots'][..., curr_time_idx-1].detach()
            params['cam_trans'][..., curr_time_idx] = params['cam_trans'][..., curr_time_idx-1].detach()
    
    return params


def convert_params_to_store(params):
    params_to_store = {}
    for k, v in params.items():
        if isinstance(v, torch.Tensor):
            params_to_store[k] = v.detach().clone()
        else:
            params_to_store[k] = v
    return params_to_store


def rgbd_slam(config: dict):
    # Print Config
    print("Loaded Config:")
    if "use_depth_loss_thres" not in config['tracking']:
        config['tracking']['use_depth_loss_thres'] = False
        config['tracking']['depth_loss_thres'] = 100000
    if "visualize_tracking_loss" not in config['tracking']:
        config['tracking']['visualize_tracking_loss'] = False
    print(f"{config}")

    # Create Output Directories
    output_dir = os.path.join(config["workdir"], config["run_name"])
    eval_dir = os.path.join(output_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    
    # Init WandB
    if config['use_wandb']:
        wandb_time_step = 0
        wandb_tracking_step = 0
        wandb_mapping_step = 0
        wandb_run = wandb.init(project=config['wandb']['project'],
                               entity=config['wandb']['entity'],
                               group=config['wandb']['group'],
                               name=config['wandb']['name'],
                               config=config)

    # Get Device
    device = torch.device(config["primary_device"])
    if config["primary_device"].startswith("cuda:"):
        device_id = int(config["primary_device"].split(':')[1])
        torch.cuda.set_device(device_id)

    # Load Dataset
    print("Loading Dataset ...")
    dataset_config = config["data"]
    if "gradslam_data_cfg" not in dataset_config:
        gradslam_data_cfg = {}
        gradslam_data_cfg["dataset_name"] = dataset_config["dataset_name"]
    else:
        gradslam_data_cfg = load_dataset_config(dataset_config["gradslam_data_cfg"])
    if "ignore_bad" not in dataset_config:
        dataset_config["ignore_bad"] = False
    if "use_train_split" not in dataset_config:
        dataset_config["use_train_split"] = True
    if "densification_image_height" not in dataset_config:
        dataset_config["densification_image_height"] = dataset_config["desired_image_height"]
        dataset_config["densification_image_width"] = dataset_config["desired_image_width"]
        seperate_densification_res = False
    else:
        if dataset_config["densification_image_height"] != dataset_config["desired_image_height"] or \
            dataset_config["densification_image_width"] != dataset_config["desired_image_width"]:
            seperate_densification_res = True
        else:
            seperate_densification_res = False
    if "tracking_image_height" not in dataset_config:
        dataset_config["tracking_image_height"] = dataset_config["desired_image_height"]
        dataset_config["tracking_image_width"] = dataset_config["desired_image_width"]
        seperate_tracking_res = False
    else:
        if dataset_config["tracking_image_height"] != dataset_config["desired_image_height"] or \
            dataset_config["tracking_image_width"] != dataset_config["desired_image_width"]:
            seperate_tracking_res = True
        else:
            seperate_tracking_res = False
    if "load_semantics" not in dataset_config:
        load_semantics = False
        num_semantic_classes = 0
    else:
        load_semantics = dataset_config["load_semantics"]
        num_semantic_classes = dataset_config["num_semantic_classes"]
    # Poses are relative to the first frame
    dataset = get_dataset(
        config_dict=gradslam_data_cfg,
        basedir=dataset_config["basedir"],
        sequence=os.path.basename(dataset_config["sequence"]),
        start=dataset_config["start"],
        end=dataset_config["end"],
        stride=dataset_config["stride"],
        desired_height=dataset_config["desired_image_height"],
        desired_width=dataset_config["desired_image_width"],
        device=device,
        relative_pose=True,
        ignore_bad=dataset_config["ignore_bad"],
        use_train_split=dataset_config["use_train_split"],
        load_semantics=load_semantics,
        num_semantic_classes=num_semantic_classes,
    )
    num_frames = dataset_config["num_frames"]
    if num_frames == -1:
        num_frames = len(dataset)

    # Init seperate dataloader for densification if required
    if seperate_densification_res:
        densify_dataset = get_dataset(
            config_dict=gradslam_data_cfg,
            basedir=dataset_config["basedir"],
            sequence=os.path.basename(dataset_config["sequence"]),
            start=dataset_config["start"],
            end=dataset_config["end"],
            stride=dataset_config["stride"],
            desired_height=dataset_config["densification_image_height"],
            desired_width=dataset_config["densification_image_width"],
            device=device,
            relative_pose=True,
            ignore_bad=dataset_config["ignore_bad"],
            use_train_split=dataset_config["use_train_split"],
        )
        # Initialize Parameters, Canonical & Densification Camera parameters
        params, variables, intrinsics, first_frame_w2c, cam, params_opt_exclude, \
            densify_intrinsics, densify_cam = initialize_first_timestep(dataset, num_frames,
                                                                        config['scene_radius_depth_ratio'],
                                                                        config['mean_sq_dist_method'],
                                                                        device=device,
                                                                        densify_dataset=densify_dataset,
                                                                        load_semantics=load_semantics)                                                                                                                  
    else:
        # Initialize Parameters & Canoncial Camera parameters
        params, variables, intrinsics, first_frame_w2c, cam, \
            params_opt_exclude = initialize_first_timestep(dataset, num_frames, config['scene_radius_depth_ratio'],
                                                           config['mean_sq_dist_method'], device=device,
                                                           load_semantics=load_semantics,
                                                           dc_dict=config['mapping'].get('depth_consistency'))
    
    # Init seperate dataloader for tracking if required
    if seperate_tracking_res:
        tracking_dataset = get_dataset(
            config_dict=gradslam_data_cfg,
            basedir=dataset_config["basedir"],
            sequence=os.path.basename(dataset_config["sequence"]),
            start=dataset_config["start"],
            end=dataset_config["end"],
            stride=dataset_config["stride"],
            desired_height=dataset_config["tracking_image_height"],
            desired_width=dataset_config["tracking_image_width"],
            device=device,
            relative_pose=True,
            ignore_bad=dataset_config["ignore_bad"],
            use_train_split=dataset_config["use_train_split"],
        )
        tracking_color, _, tracking_intrinsics, _ = tracking_dataset[0]
        tracking_color = tracking_color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
        tracking_intrinsics = tracking_intrinsics[:3, :3]
        tracking_cam = setup_camera(tracking_color.shape[2], tracking_color.shape[1],
                                    tracking_intrinsics.cpu().numpy(),
                                    first_frame_w2c.detach().cpu().numpy(), device=device)
    
    # Initialize list to keep track of Keyframes
    keyframe_list = []
    keyframe_time_indices = []
    timestamp_keyframes = []
    
    # Init Variables to keep track of ground truth poses and runtimes
    gt_w2c_all_frames = []
    tracking_iter_time_sum = 0
    tracking_iter_time_count = 0
    mapping_iter_time_sum = 0
    mapping_iter_time_count = 0
    tracking_frame_time_sum = 0
    tracking_frame_time_count = 0
    mapping_frame_time_sum = 0
    mapping_frame_time_count = 0

    # Load Checkpoint
    if config['load_checkpoint']:
        checkpoint_time_idx = config['checkpoint_time_idx']
        print(f"Loading Checkpoint for Frame {checkpoint_time_idx}")
        ckpt_path = os.path.join(config['workdir'], config['run_name'], f"params{checkpoint_time_idx}.npz")
        params = dict(np.load(ckpt_path, allow_pickle=True))
        for k in params:
            if k not in params_opt_exclude:
                params[k] = torch.tensor(params[k]).to(device).float().requires_grad_(True)
            else:
                params[k] = torch.tensor(params[k]).to(device).float()

        variables['max_2D_radius'] = torch.zeros(params['means3D'].shape[0]).to(device).float()
        variables['means2D_gradient_accum'] = torch.zeros(params['means3D'].shape[0]).to(device).float()
        variables['denom'] = torch.zeros(params['means3D'].shape[0]).to(device).float()
        variables['timestep'] = torch.zeros(params['means3D'].shape[0]).to(device).float()
        # Load the keyframe time idx list
        keyframe_time_indices = np.load(os.path.join(config['workdir'], config['run_name'], f"keyframe_time_indices{checkpoint_time_idx}.npy"))
        keyframe_time_indices = keyframe_time_indices.tolist()
        # Update the ground truth poses list
        for time_idx in range(checkpoint_time_idx):
            # Load RGBD frames incrementally instead of all frames
            if load_semantics:
                color, depth, _, gt_pose, semantic_id, semantic_color = dataset[time_idx]
            else:
                color, depth, _, gt_pose = dataset[time_idx]
            # Process poses
            gt_w2c = torch.linalg.inv(gt_pose)
            gt_w2c_all_frames.append(gt_w2c)
            # Initialize Keyframe List
            if time_idx in keyframe_time_indices:
                # Get the estimated rotation & translation
                curr_cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
                curr_cam_tran = params['cam_trans'][..., time_idx].detach()
                curr_w2c = torch.eye(4).to(device).float()
                curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                curr_w2c[:3, 3] = curr_cam_tran
                # Initialize Keyframe Info
                color = color.permute(2, 0, 1) / 255
                depth = depth.permute(2, 0, 1)
                curr_keyframe = {'id': time_idx, 'est_w2c': curr_w2c, 'color': color, 'depth': depth}

                if load_semantics:
                    semantic_id = semantic_id.permute(2, 0, 1)
                    semantic_color = semantic_color.permute(2, 0, 1) / 255
                    curr_keyframe['semantic_id'] = semantic_id
                    curr_keyframe['semantic_color'] = semantic_color
                # Add to keyframe list
                keyframe_list.append(curr_keyframe)
    else:
        checkpoint_time_idx = 0
    
    # P5: load rail-line prior (tracking)
    if RAIL_PRIOR:
        global RAIL_LINES, RAIL_REF, RAIL_W_DIR, RAIL_W_PT
        import json as _json
        _rp = _json.load(open(config['tracking']['rail_prior_json']))
        RAIL_REF = int(_rp['ref'])
        RAIL_W_DIR = float(config['tracking'].get('rail_w_dir', 10.0))
        RAIL_W_PT = float(config['tracking'].get('rail_w_pt', 10000.0))
        for _k, _cd in _rp['lines'].items():
            RAIL_LINES[int(_k)] = (
                torch.tensor(_cd[0], device=device, dtype=torch.float),
                torch.tensor(_cd[1], device=device, dtype=torch.float))
        print('RAIL_PRIOR on: %d lines, ref=%d, w_dir=%.1f, w_pt=%.0f' %
              (len(RAIL_LINES), RAIL_REF, RAIL_W_DIR, RAIL_W_PT), flush=True)

    # Iterate over Scan
    for time_idx in tqdm(range(checkpoint_time_idx, num_frames)):
        # Load RGBD frames incrementally instead of all frames
        if load_semantics:
            color, depth, _, gt_pose, semantic_id, semantic_color = dataset[time_idx]
        else:
            color, depth, _, gt_pose = dataset[time_idx]
        # Process poses
        gt_w2c = torch.linalg.inv(gt_pose)
        # Process RGB-D Data
        color = color.permute(2, 0, 1) / 255
        depth = depth.permute(2, 0, 1)
        gt_w2c_all_frames.append(gt_w2c)
        curr_gt_w2c = gt_w2c_all_frames
        # Optimize only current time step for tracking
        iter_time_idx = time_idx
        # Initialize Mapping Data for selected frame
        curr_data = {'cam': cam, 'im': color, 'depth': depth, 'id': iter_time_idx, 'intrinsics': intrinsics,
                     'w2c': first_frame_w2c, 'iter_gt_w2c_list': curr_gt_w2c}
        
        if load_semantics:
            semantic_id = semantic_id.permute(2, 0, 1)
            semantic_color = semantic_color.permute(2, 0, 1) / 255
            curr_data['semantic_id'] = semantic_id
            curr_data['semantic_color'] = semantic_color
        
        # Initialize Data for Tracking
        if seperate_tracking_res:
            tracking_color, tracking_depth, _, _ = tracking_dataset[time_idx]
            tracking_color = tracking_color.permute(2, 0, 1) / 255
            tracking_depth = tracking_depth.permute(2, 0, 1)
            tracking_curr_data = {'cam': tracking_cam, 'im': tracking_color, 'depth': tracking_depth,
                                  'id': iter_time_idx, 'intrinsics': tracking_intrinsics,
                                  'w2c': first_frame_w2c,'iter_gt_w2c_list': curr_gt_w2c}
        else:
            tracking_curr_data = curr_data

        # Optimization Iterations
        num_iters_mapping = config['mapping']['num_iters']
        
        # Initialize the camera pose for the current frame
        if time_idx > 0:
            params = initialize_camera_pose(params, time_idx, forward_prop=config['tracking']['forward_prop'])
        # v4 F1: initialize from INS pose (init only; tracking refinement follows)
        if time_idx > 0 and config['tracking'].get('ins_init', False) and not config['tracking']['use_gt_poses']:
            with torch.no_grad():
                rel_w2c = curr_gt_w2c[-1]
                # v5 CHAIN_INIT (2026-10-11 Gary 批准方法层)：链式初始化
                #   T_init = T_est(t-1) @ inv(T_ins(t-1)) @ T_ins(t) —— 上一帧估计 ⊕ INS 增量。
                #   动机：绝对 INS 起步使估计恒贴初始化源（10-11 实证 |est-INS|≤0.1m），
                #   绝对定位误差无法被优化吸收；链式让误差沿时间链传播并被逐帧约束。
                # v5b 混合再锚定（V4_CHAIN_REANCHOR>0 时生效）：每逢该帧数间隔改用绝对 INS
                #   起步一次，把链式累积漂移拉回传感器本底，其余帧保持链式（增量可被优化修正）。
                _reanchor = (V4_CHAIN_REANCHOR > 0 and time_idx % V4_CHAIN_REANCHOR == 0)
                if V4_CHAIN_INIT and time_idx >= 1 and not _reanchor:
                    T_ins_curr = rel_w2c.detach().double()
                    T_ins_prev = gt_w2c_all_frames[time_idx - 1].detach().double()
                    T_inc = torch.linalg.inv(T_ins_prev) @ T_ins_curr
                    _qprev = F.normalize(params['cam_unnorm_rots'][..., time_idx - 1].detach(), dim=0)
                    T_est_prev = torch.eye(4, dtype=torch.float64, device=rel_w2c.device)
                    T_est_prev[:3, :3] = build_rotation(_qprev).double()[0]
                    T_est_prev[:3, 3] = params['cam_trans'][..., time_idx - 1].detach().double()
                    rel_w2c = (T_est_prev @ T_inc).float()
                rel_w2c_rot = rel_w2c[:3, :3].unsqueeze(0).detach()
                params['cam_unnorm_rots'][..., time_idx] = matrix_to_quaternion(rel_w2c_rot)
                params['cam_trans'][..., time_idx] = rel_w2c[:3, 3].detach()

        # Step 1: Tracking
        tracking_start_time = time.time()
        if time_idx > 0 and not config['tracking']['use_gt_poses']:
            # Reset Optimizer & Learning Rates for tracking
            optimizer = initialize_optimizer(params, params_opt_exclude, config['tracking']['lrs'], tracking=True)
            # Keep Track of Best Candidate Rotation & Translation
            candidate_cam_unnorm_rot = params['cam_unnorm_rots'][..., time_idx].detach().clone()
            candidate_cam_tran = params['cam_trans'][..., time_idx].detach().clone()
            current_min_loss = float(1e20)
            # Tracking Optimization
            iter = 0
            do_continue_slam = False
            num_iters_tracking = config['tracking']['num_iters']
            if V4_TRACK_DUMP:
                _td_iters, _td_losses = [], []
                _td_trans, _td_rots = [], []
            progress_bar = tqdm(range(num_iters_tracking), desc=f"Tracking Time Step: {time_idx}")
            while True:
                iter_start_time = time.time()
                # Loss for current frame
                loss, variables, losses = get_loss(params, tracking_curr_data, variables, iter_time_idx, config['tracking']['loss_weights'],
                                                   config['tracking']['use_sil_for_loss'], config['tracking']['sil_thres'],
                                                   config['tracking']['use_l1'], config['tracking']['ignore_outlier_depth_loss'],
                                                   tracking=True, device=device, plot_dir=eval_dir,
                                                   visualize_tracking_loss=config['tracking']['visualize_tracking_loss'],
                                                   tracking_iteration=iter, load_semantics=load_semantics)
                if V4_TRACK_DUMP:
                    _td_iters.append(iter)
                    _td_losses.append(float(loss.item()))
                    _td_trans.append(params['cam_trans'][..., time_idx].detach().cpu().numpy().copy())
                    _td_rots.append(params['cam_unnorm_rots'][..., time_idx].detach().cpu().numpy().copy())
                if config['use_wandb']:
                    # Report Loss
                    wandb_tracking_step = report_loss(losses, wandb_run, wandb_tracking_step, tracking=True, load_semantics=load_semantics)
                # Backprop
                loss.backward()
                # Optimizer Update
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                _sky_step()   # P-69
                with torch.no_grad():
                    # Save the best candidate rotation & translation
                    if loss < current_min_loss:
                        current_min_loss = loss
                        candidate_cam_unnorm_rot = params['cam_unnorm_rots'][..., time_idx].detach().clone()
                        candidate_cam_tran = params['cam_trans'][..., time_idx].detach().clone()
                    # Report Progress
                    if config['report_iter_progress']:
                        if config['use_wandb']:
                            report_progress(params, tracking_curr_data, iter+1, progress_bar, iter_time_idx, sil_thres=config['tracking']['sil_thres'],
                                            tracking=True, device=device, load_semantics=load_semantics, wandb_run=wandb_run, wandb_step=wandb_tracking_step,
                                            wandb_save_qual=config['wandb']['save_qual'])
                        else:
                            report_progress(params, tracking_curr_data, iter+1, progress_bar, iter_time_idx, sil_thres=config['tracking']['sil_thres'],
                                            tracking=True, device=device, load_semantics=load_semantics)
                    else:
                        progress_bar.update(1)
                # Update the runtime numbers
                iter_end_time = time.time()
                tracking_iter_time_sum += iter_end_time - iter_start_time
                tracking_iter_time_count += 1
                # Check if we should stop tracking
                iter += 1
                if iter == num_iters_tracking:
                    if losses['depth'] < config['tracking']['depth_loss_thres'] and config['tracking']['use_depth_loss_thres']:
                        break
                    elif config['tracking']['use_depth_loss_thres'] and not do_continue_slam:
                        do_continue_slam = True
                        progress_bar = tqdm(range(num_iters_tracking), desc=f"Tracking Time Step: {time_idx}")
                        num_iters_tracking = 2*num_iters_tracking
                        if config['use_wandb']:
                            wandb_run.log({"Tracking/Extra Tracking Iters Frames": time_idx,
                                        "Tracking/step": wandb_time_step})
                    else:
                        break

            progress_bar.close()
            if V4_TRACK_DUMP:
                np.savez(V4_TRACK_DUMP + '_t%03d.npz' % time_idx,
                         iters=np.asarray(_td_iters), losses=np.asarray(_td_losses),
                         trans=np.asarray(_td_trans), rots=np.asarray(_td_rots))
            # Copy over the best candidate rotation & translation
            with torch.no_grad():
                params['cam_unnorm_rots'][..., time_idx] = candidate_cam_unnorm_rot
                params['cam_trans'][..., time_idx] = candidate_cam_tran
            # P-42: INS 相对位移兜底 —— 5.1 实测逐帧相对误差(RPE)中位 16cm、尖峰 2.8m，
            # 漂移会让地图按错位姿态建出来 → 轨道重影/段宽不一致/画面混杂。
            # 仅当本帧相对位移与 INS 相对位移的偏差超过阈值时，才把该帧拉回 INS（并记录日志）。
            _ic = config['tracking'].get('ins_clamp', None)
            if _ic is not None and _ic.get('enable', False):
                with torch.no_grad():
                    _ins = curr_gt_w2c[-1]
                    _ins_t = _ins[:3, 3].detach()
                    _est_t = params['cam_trans'][..., time_idx].detach()
                    if P42_STATE['ins_t'] is not None and P42_STATE['est_t'] is not None:
                        d_est = _est_t - P42_STATE['est_t']
                        d_ins = _ins_t - P42_STATE['ins_t']
                        dev = float(torch.linalg.norm(d_est - d_ins).item())
                        thr = _ic.get('dev_thresh', 0.5)
                        if dev > thr:
                            params['cam_unnorm_rots'][..., time_idx] = matrix_to_quaternion(
                                _ins[:3, :3].unsqueeze(0).detach())
                            params['cam_trans'][..., time_idx] = _ins_t
                            P42_STATE['n_clamp'] += 1
                            print(f"[P42] t={time_idx} dev={dev:.2f}m > {thr} -> INS兜底 "
                                  f"(累计{P42_STATE['n_clamp']})", flush=True)
                    P42_STATE['ins_t'] = _ins_t.clone()
                    P42_STATE['est_t'] = params['cam_trans'][..., time_idx].detach().clone()
        elif time_idx > 0 and config['tracking']['use_gt_poses']:
            with torch.no_grad():
                # Get the ground truth pose relative to frame 0
                rel_w2c = curr_gt_w2c[-1]
                rel_w2c_rot = rel_w2c[:3, :3].unsqueeze(0).detach()
                rel_w2c_rot_quat = matrix_to_quaternion(rel_w2c_rot)
                rel_w2c_tran = rel_w2c[:3, 3].detach()
                # Update the camera parameters
                params['cam_unnorm_rots'][..., time_idx] = rel_w2c_rot_quat
                params['cam_trans'][..., time_idx] = rel_w2c_tran
        # Update the runtime numbers
        tracking_end_time = time.time()
        tracking_frame_time_sum += tracking_end_time - tracking_start_time
        tracking_frame_time_count += 1

        if time_idx == 0 or (time_idx+1) % config['report_global_progress_every'] == 0:
            try:
                # Report Final Tracking Progress
                progress_bar = tqdm(range(1), desc=f"Tracking Result Time Step: {time_idx}")
                with torch.no_grad():
                    if config['use_wandb']:
                        report_progress(params, tracking_curr_data, 1, progress_bar, iter_time_idx, sil_thres=config['tracking']['sil_thres'],
                                        tracking=True, device=device, load_semantics=load_semantics, wandb_run=wandb_run, wandb_step=wandb_time_step,
                                        wandb_save_qual=config['wandb']['save_qual'], global_logging=True)
                    else:
                        report_progress(params, tracking_curr_data, 1, progress_bar, iter_time_idx, sil_thres=config['tracking']['sil_thres'],
                                        tracking=True, device=device, load_semantics=load_semantics)
                progress_bar.close()
            except:
                ckpt_output_dir = os.path.join(config["workdir"], config["run_name"])
                save_params_ckpt(params, ckpt_output_dir, time_idx)
                print('Failed to evaluate trajectory.')

        # Step 2: Densification & KeyFrame-based Mapping
        if time_idx == 0 or (time_idx+1) % config['map_every'] == 0:
            # Densification
            if config['mapping']['add_new_gaussians'] and time_idx > 0 and not _p66_blocked(params, "seed", time_idx):
                # Setup Data for Densification
                if seperate_densification_res:
                    # Load RGBD frames incrementally instead of all frames
                    densify_color, densify_depth, _, _ = densify_dataset[time_idx]
                    densify_color = densify_color.permute(2, 0, 1) / 255
                    densify_depth = densify_depth.permute(2, 0, 1)
                    densify_curr_data = {'cam': densify_cam, 'im': densify_color, 'depth': densify_depth, 'id': time_idx, 
                                 'intrinsics': densify_intrinsics, 'w2c': first_frame_w2c, 'iter_gt_w2c_list': curr_gt_w2c}
                else:
                    densify_curr_data = curr_data

                # Add new Gaussians to the scene based on the Silhouette
                params, variables = add_new_gaussians(params, params_opt_exclude, variables, densify_curr_data, 
                                                      config['mapping']['sil_thres'], time_idx, config['mean_sq_dist_method'],
                                                      device, load_semantics=load_semantics,
                                                      add_dict=config['mapping'].get('add_depth_err_dict'),
                                                      dc_dict=config['mapping'].get('depth_consistency'))
                post_num_pts = params['means3D'].shape[0]
                if config['use_wandb']:
                    wandb_run.log({"Mapping/Number of Gaussians": post_num_pts,
                                   "Mapping/step": wandb_time_step})
            
            # Update keyframes for gaussian mapping
            with torch.no_grad():
                # Get the current estimated rotation & translation
                curr_cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
                curr_cam_tran = params['cam_trans'][..., time_idx].detach()
                curr_w2c = torch.eye(4).to(device).float()
                curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                curr_w2c[:3, 3] = curr_cam_tran
                # P-36: 相机世界坐标与运动方向（cam_trans 是 w2c 平移，需换算成相机中心）
                _Rc = build_rotation(curr_cam_rot)[0]
                _t = curr_cam_tran.reshape(3)
                cam_center = -(_Rc.t() @ _t)
                _pt = max(time_idx - 1, 0)
                _prot = F.normalize(params['cam_unnorm_rots'][..., _pt].detach())
                _ptran = params['cam_trans'][..., _pt].detach()
                _pcenter = -(build_rotation(_prot)[0].t() @ _ptran.reshape(3))
                _fwd = cam_center - _pcenter
                _fn = _fwd.norm()
                cam_fwd = (_fwd / _fn) if float(_fn) > 1e-6 else None
                # Select Keyframes for Mapping
                num_keyframes = config['mapping_window_size']-2
                selected_keyframes = keyframe_selection_overlap(depth, curr_w2c, intrinsics, keyframe_list[:-1],
                                                                num_keyframes, device=device)
                selected_time_idx = [keyframe_list[frame_idx]['id'] for frame_idx in selected_keyframes]
                if len(keyframe_list) > 0:
                    # Add last keyframe to the selected keyframes
                    selected_time_idx.append(keyframe_list[-1]['id'])
                    selected_keyframes.append(len(keyframe_list)-1)
                # Add current frame to the selected keyframes
                selected_time_idx.append(time_idx)
                selected_keyframes.append(-1)
                # Print the selected keyframes
                print(f"\nSelected Keyframes at Frame {time_idx}: {selected_time_idx}")
                timestamp_keyframes.append(selected_time_idx)
                # v5 KF_REACTIVATE (2026-10-11 Gary 批准方法层)：老关键帧再激活
                #   动机：mapping window 只含重叠选取的近期关键帧 ⇒ recency 效应（06 全量中段
                #   PSNR 9-10、老区域无人约束）。每 REACT_EVERY 帧把均匀采样的 REACT_NUM 个
                #   未入选老关键帧并入本帧建图集——内循环本就按帧随机采样，天然时间分摊，
                #   无显存增长（每迭代只渲染单个关键帧）。
                if V4_KF_REACT_EVERY > 0 and time_idx > 0 and (time_idx % V4_KF_REACT_EVERY == 0) \
                        and len(keyframe_list) > 2 * V4_KF_REACT_NUM:
                    _sel = set(int(x) for x in selected_keyframes if int(x) >= 0)
                    _cand = [i for i in range(len(keyframe_list) - 1) if i not in _sel]
                    if len(_cand) >= V4_KF_REACT_NUM and V4_KF_REACT_NUM > 1:
                        _pick = sorted({_cand[int(round(k * (len(_cand) - 1) / (V4_KF_REACT_NUM - 1)))]
                                        for k in range(V4_KF_REACT_NUM)})
                        for _i in _pick:
                            selected_keyframes.append(_i)
                            selected_time_idx.append(keyframe_list[_i]['id'])
                        print(f"[KF-REACT] t={time_idx} 再激活老关键帧 id={selected_time_idx[-len(_pick):]}",
                              flush=True)

            # Reset Optimizer & Learning Rates for Full Map Optimization
            optimizer = initialize_optimizer(params, params_opt_exclude, config['mapping']['lrs'], tracking=False) 

            # P-69: 占位（sky 参数独立于 params，避免混入致密化路径）
            _sky_init(device)

            # Mapping
            mapping_start_time = time.time()
            if num_iters_mapping > 0:
                progress_bar = tqdm(range(num_iters_mapping), desc=f"Mapping Time Step: {time_idx}")
            for iter in range(num_iters_mapping):
                iter_start_time = time.time()
                # Randomly select a frame until current time step amongst keyframes
                rand_idx = np.random.randint(0, len(selected_keyframes))
                selected_rand_keyframe_idx = selected_keyframes[rand_idx]
                if selected_rand_keyframe_idx == -1:
                    # Use Current Frame Data
                    iter_time_idx = time_idx
                    iter_color = color
                    iter_depth = depth
                else:
                    # Use Keyframe Data
                    iter_time_idx = keyframe_list[selected_rand_keyframe_idx]['id']
                    iter_color = keyframe_list[selected_rand_keyframe_idx]['color']
                    iter_depth = keyframe_list[selected_rand_keyframe_idx]['depth']
                iter_gt_w2c = gt_w2c_all_frames[:iter_time_idx+1]
                iter_data = {'cam': cam, 'im': iter_color, 'depth': iter_depth, 'id': iter_time_idx, 
                             'intrinsics': intrinsics, 'w2c': first_frame_w2c, 'iter_gt_w2c_list': iter_gt_w2c}
                # Add semantic id and colors
                if load_semantics:
                    if selected_rand_keyframe_idx == -1:
                        iter_data['semantic_id'] = semantic_id
                        iter_data['semantic_color'] = semantic_color
                    else:
                        iter_data['semantic_id'] = keyframe_list[selected_rand_keyframe_idx]['semantic_id']
                        iter_data['semantic_color'] = keyframe_list[selected_rand_keyframe_idx]['semantic_color']
                # Loss for current frame
                loss, variables, losses = get_loss(params, iter_data, variables, iter_time_idx, config['mapping']['loss_weights'],
                                                config['mapping']['use_sil_for_loss'], config['mapping']['sil_thres'],
                                                config['mapping']['use_l1'], config['mapping']['ignore_outlier_depth_loss'],
                                                mapping=True, device=device, load_semantics=load_semantics)
                if config['use_wandb']:
                    # Report Loss
                    wandb_mapping_step = report_loss(losses, wandb_run, wandb_mapping_step, mapping=True, load_semantics=load_semantics)
                # Backprop
                loss.backward()
                with torch.no_grad():
# Gaussian-Splatting's Gradient-based Densification
                    if config['mapping']['use_gaussian_splatting_densification']:
                        if not _p66_blocked(params, "densify", time_idx):
                            params, variables = densify(params, variables, optimizer, iter, config['mapping']['densify_dict'], params_opt_exclude, device=device)
                        if config['use_wandb']:
                            wandb_run.log({"Mapping/Number of Gaussians - Densification": params['means3D'].shape[0],
                                           "Mapping/step": wandb_mapping_step})
# Prune Gaussians
                    if config['mapping']['prune_gaussians']:
                        params, variables = prune_gaussians(params, params_opt_exclude, variables, optimizer, iter, config['mapping']['pruning_dict'], time_idx=time_idx, cam_center=cam_center, cam_fwd=cam_fwd)
                        if config['use_wandb']:
                            wandb_run.log({"Mapping/Number of Gaussians - Pruning": params['means3D'].shape[0],
                                           "Mapping/step": wandb_mapping_step})
                    # Optimizer Update
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    _sky_step()   # P-69
                    # P-40: 尺度下限/上限 —— 实测高斯中位尺度仅 1mm（反复分裂所致），
                    # 导致地图变成"毫米级不透明尘点"：MeshLab 里是灰噪声、远处没有连续面片。
                    # 强制尺度落在 [scale_min, scale_max]（米），让高斯成为真正的面片。
                    _smin = config['mapping'].get('scale_min', 0.0)
                    _smax = config['mapping'].get('scale_max', 0.0)
                    if _smin > 0 or _smax > 0:
                        with torch.no_grad():
                            _ls = params['log_scales']
                            if _smin > 0:
                                _ls.clamp_(min=float(np.log(_smin)))
                            if _smax > 0:
                                _ls.clamp_(max=float(np.log(_smax)))
                    # P-46: 深度自适应尺度窗 —— 尺度应由"投影到屏幕占多少像素"决定，而不是常数。
                    # 实测 fx≈2828、FOV 31.6°：常数 3cm 在 5m 处占 15 像素、2m 处占 34 像素 → 糊成圆盘；
                    # 而旧的 1mm 只有 0.5 像素 → 亚像素噪声。正确区间 ≈ (1.5~5) 像素 = (1.5~5) × 距离 / fx。
                    _pxmin = config['mapping'].get('scale_px_min', 0.0)
                    _pxmax = config['mapping'].get('scale_px_max', 0.0)
                    if (_pxmin > 0 or _pxmax > 0) and cam_center is not None:
                        with torch.no_grad():
                            _fx = float(curr_data['intrinsics'][0, 0])
                            # 注意：必须用「相机系深度 z」而不是 3D 距离。
                            # 屏幕半径 ∝ scale / z；离轴高斯的 dist/z 可达数十倍，
                            # 用距离做夹逼会放过一批屏幕尺寸严重超标的高斯
                            # （实测 q2 帧99：1.8 万个 σ>3px、最大 22px，占过绘制 9% → 泛白雾斑）。
                            _zcam = (params['means3D'] @ _Rc.t())[:, 2] + _t[2]
                            _dcam = _zcam.clamp(min=0.1)
                            # P-68: 各向异性时逐轴夹逼（等价于约束椭球三个主轴的屏幕尺寸）
                            _ls2 = (params['log_scales'] if params['log_scales'].shape[-1] == 3
                                    else params['log_scales'].reshape(-1, 1))
                            # P-52 语义自适应尺度窗：软类（植被/线缆）用更大的像素窗。
                            # 依据：q5 帧0 左侧黄绿薄纱区投影到 49 万个高斯，语义 = vegetation 53%
                            # + ground 42%，σ 仅 0.4px 且 α=0.97 → 大量"亚像素不透明小点"叠加成洗色。
                            # 植被本就该是蓬松的大尺度低密度表示，不该用密集小硬点。
                            _pxs = config['mapping'].get('scale_px_soft', None)
                            if _pxs and ('semantic_ids' in params.keys()):
                                _sid = params['semantic_ids'].reshape(-1)
                                _soft = torch.zeros_like(_sid, dtype=torch.bool)
                                for _c in config['mapping'].get('scale_px_soft_classes', [23, 28]):
                                    _soft = _soft | (_sid == _c)
                                _lo = torch.where(_soft, torch.full_like(_dcam, float(_pxs[0])),
                                                  torch.full_like(_dcam, float(_pxmin)))
                                _hi = torch.where(_soft, torch.full_like(_dcam, float(_pxs[1])),
                                                  torch.full_like(_dcam, float(_pxmax)))
                            else:
                                _lo = torch.full_like(_dcam, float(_pxmin))
                                _hi = torch.full_like(_dcam, float(_pxmax))
                            # P-83: 尺度上限按深度分区放宽（远场允许大高斯）
                            _sdw = config['mapping'].get('scale_px_depthwise', None)
                            if _sdw:
                                _z0, _z1, _p0, _p1 = [float(x) for x in _sdw]
                                _tt = ((_dcam - _z0) / max(_z1 - _z0, 1e-6)).clamp(0.0, 1.0)
                                _hifar = _p0 + (_p1 - _p0) * _tt
                                # 只放宽、不放紧：与逐类软窗取较大者
                                _hi = torch.maximum(_hi, _hifar)
                            if _pxmin > 0 or _pxs is not None:
                                _ls2.clamp_(min=torch.log(_lo.unsqueeze(-1) * _dcam.unsqueeze(-1) / _fx))
                            if _pxmax > 0 or _pxs is not None:
                                _ls2.clamp_(max=torch.log(_hi.unsqueeze(-1) * _dcam.unsqueeze(-1) / _fx))
                    # Report Progress
                    if config['report_iter_progress']:
                        if config['use_wandb']:
                            report_progress(params, iter_data, iter+1, progress_bar, iter_time_idx, sil_thres=config['mapping']['sil_thres'], 
                                            wandb_run=wandb_run, wandb_step=wandb_mapping_step, wandb_save_qual=config['wandb']['save_qual'],
                                            mapping=True, device=device, load_semantics=load_semantics, online_time_idx=time_idx)
                        else:
                            report_progress(params, iter_data, iter+1, progress_bar, iter_time_idx, sil_thres=config['mapping']['sil_thres'], 
                                            mapping=True, device=device, load_semantics=load_semantics, online_time_idx=time_idx)
                    else:
                        progress_bar.update(1)
                # Update the runtime numbers
                iter_end_time = time.time()
                mapping_iter_time_sum += iter_end_time - iter_start_time
                mapping_iter_time_count += 1
            if num_iters_mapping > 0:
                progress_bar.close()
            # v5 SEM_CLAMP (2026-10-11 Gary 批准方法层)：每个建图帧结束后把语义色钳回 [0,1]
            #   动机：KITTI 长序列上 semantic_colors 无约束漂移出表色域（06@f1100 达 61%），
            #   渲染语义不可解码 ⇒ mIoU 崩塌；钳位保留 in-gamut 优化自由度、截断出界漂移。
            if V4_SEM_CLAMP and load_semantics and 'semantic_colors' in params:
                params['semantic_colors'].data.clamp_(0, 1)
            # v7 调色板吸附：软拉向最近类色（一次结构判据打印：吸附前后到调色板平均距离）
            if V4_SEM_SNAP and load_semantics and 'semantic_colors' in params:
                with torch.no_grad():
                    _sc = params['semantic_colors'].data
                    _pal = _SEM_PAL_21.to(_sc.device)
                    if not _SEM_SNAP_DBG[0]:
                        _SEM_SNAP_DBG[0] = True
                        _d0 = float(torch.cdist(_sc, _pal).min(1).values.mean())
                        print(f'[V7] 首次吸附前 平均|色-最近类色|={_d0:.4f}', flush=True)
                    _tgt = _pal[torch.cdist(_sc, _pal).argmin(1)]
                    _sc.add_(_tgt - _sc, alpha=V4_SEM_SNAP_A)
                    if not _SEM_SNAP_DBG[1]:
                        _SEM_SNAP_DBG[1] = True
                        _d1 = float(torch.cdist(_sc, _pal).min(1).values.mean())
                        print(f'[V7] 首次吸附后 平均|色-最近类色|={_d1:.4f}', flush=True)
            # Update the runtime numbers
            mapping_end_time = time.time()
            mapping_frame_time_sum += mapping_end_time - mapping_start_time
            mapping_frame_time_count += 1

            if time_idx == 0 or (time_idx+1) % config['report_global_progress_every'] == 0:
                try:
                    # Report Mapping Progress
                    progress_bar = tqdm(range(1), desc=f"Mapping Result Time Step: {time_idx}")
                    with torch.no_grad():
                        if config['use_wandb']:
                            report_progress(params, curr_data, 1, progress_bar, time_idx, sil_thres=config['mapping']['sil_thres'], 
                                            wandb_run=wandb_run, wandb_step=wandb_time_step, wandb_save_qual=config['wandb']['save_qual'],
                                            mapping=True, device=device, load_semantics=load_semantics, online_time_idx=time_idx, global_logging=True)
                        else:
                            report_progress(params, curr_data, 1, progress_bar, time_idx, sil_thres=config['mapping']['sil_thres'], 
                                            mapping=True, device=device, load_semantics=load_semantics, online_time_idx=time_idx)
                    progress_bar.close()
                except:
                    ckpt_output_dir = os.path.join(config["workdir"], config["run_name"])
                    save_params_ckpt(params, ckpt_output_dir, time_idx)
                    print('Failed to evaluate trajectory.')
        
        # Add frame to keyframe list
        if ((time_idx == 0) or ((time_idx+1) % config['keyframe_every'] == 0) or \
                    (time_idx == num_frames-2)) and (not torch.isinf(curr_gt_w2c[-1]).any()) and (not torch.isnan(curr_gt_w2c[-1]).any()):
            with torch.no_grad():
                # Get the current estimated rotation & translation
                curr_cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
                curr_cam_tran = params['cam_trans'][..., time_idx].detach()
                curr_w2c = torch.eye(4).to(device).float()
                curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                curr_w2c[:3, 3] = curr_cam_tran
                # Initialize Keyframe Info
                curr_keyframe = {'id': time_idx, 'est_w2c': curr_w2c, 'color': color, 'depth': depth}
                if load_semantics:
                    curr_keyframe['semantic_id'] = semantic_id
                    curr_keyframe['semantic_color'] = semantic_color
                # Add to keyframe list
                keyframe_list.append(curr_keyframe)
                keyframe_time_indices.append(time_idx)
        
        # Checkpoint every iteration
        if time_idx % config["checkpoint_interval"] == 0 and config['save_checkpoints']:
            ckpt_output_dir = os.path.join(config["workdir"], config["run_name"])
            save_params_ckpt(params, ckpt_output_dir, time_idx)
            np.save(os.path.join(ckpt_output_dir, f"keyframe_time_indices{time_idx}.npy"), np.array(keyframe_time_indices))
        
        # Increment WandB Time Step
        if config['use_wandb']:
            wandb_time_step += 1

        torch.cuda.empty_cache()

    if config['save_timestamp_keyframes']:
        # Save keyframes selected at each timestamp
        max_length = max(len(inner) for inner in timestamp_keyframes)
        # Insert -1 for placeholder
        timestamp_keyframes_df = pd.DataFrame([inner + [-1 for _ in range(max_length - len(inner))] \
                                               for inner in timestamp_keyframes])
        timestamp_keyframes_df.to_csv(os.path.join(eval_dir, f"timestamp_keyframes.csv"), \
                                      index=False, header=False, na_rep='-1')

    # Compute Average Runtimes
    if tracking_iter_time_count == 0:
        tracking_iter_time_count = 1
        tracking_frame_time_count = 1
    if mapping_iter_time_count == 0:
        mapping_iter_time_count = 1
        mapping_frame_time_count = 1
    tracking_iter_time_avg = tracking_iter_time_sum / tracking_iter_time_count
    tracking_frame_time_avg = tracking_frame_time_sum / tracking_frame_time_count
    mapping_iter_time_avg = mapping_iter_time_sum / mapping_iter_time_count
    mapping_frame_time_avg = mapping_frame_time_sum / mapping_frame_time_count
    print(f"\nAverage Tracking/Iteration Time: {tracking_iter_time_avg*1000} ms")
    print(f"Average Tracking/Frame Time: {tracking_frame_time_avg} s")
    print(f"Average Mapping/Iteration Time: {mapping_iter_time_avg*1000} ms")
    print(f"Average Mapping/Frame Time: {mapping_frame_time_avg} s")
    if config['use_wandb']:
        wandb_run.log({"Final Stats/Average Tracking Iteration Time (ms)": tracking_iter_time_avg*1000,
                       "Final Stats/Average Tracking Frame Time (s)": tracking_frame_time_avg,
                       "Final Stats/Average Mapping Iteration Time (ms)": mapping_iter_time_avg*1000,
                       "Final Stats/Average Mapping Frame Time (s)": mapping_frame_time_avg,
                       "Final Stats/step": 1})
    
    # Evaluate Final Parameters
    with torch.no_grad():
        if config['use_wandb']:
            eval(dataset, params, num_frames, eval_dir, sil_thres=config['mapping']['sil_thres'],
                 wandb_run=wandb_run, wandb_save_qual=config['wandb']['eval_save_qual'],
                 mapping_iters=config['mapping']['num_iters'], add_new_gaussians=config['mapping']['add_new_gaussians'],
                 device=device, load_semantics=load_semantics, eval_every=config['eval_every'], save_frames=True)
        else:
            eval(dataset, params, num_frames, eval_dir, sil_thres=config['mapping']['sil_thres'],
                 mapping_iters=config['mapping']['num_iters'], add_new_gaussians=config['mapping']['add_new_gaussians'],
                 device=device, load_semantics=load_semantics, eval_every=config['eval_every'], save_frames=True)

    # Add Camera Parameters to Save them
    params['timestep'] = variables['timestep']
    params['intrinsics'] = intrinsics.detach().cpu().numpy()
    params['w2c'] = first_frame_w2c.detach().cpu().numpy()
    params['org_width'] = dataset_config["desired_image_width"]
    params['org_height'] = dataset_config["desired_image_height"]
    params['gt_w2c_all_frames'] = []
    for gt_w2c_tensor in gt_w2c_all_frames:
        params['gt_w2c_all_frames'].append(gt_w2c_tensor.detach().cpu().numpy())
    params['gt_w2c_all_frames'] = np.stack(params['gt_w2c_all_frames'], axis=0)
    params['keyframe_time_indices'] = np.array(keyframe_time_indices)

    if load_semantics:
        params['semantic_ids'] = params['semantic_ids'].type(torch.uint8)
    
    # Save Parameters
    save_params(params, output_dir)
    if V4_SKY_MODULE and _SKY['map'] is not None:
        _skp = os.path.join(output_dir, 'sky_map.npz')
        np.savez(_skp, sky_map=_SKY['map'].detach().cpu().numpy())
        print('[P-69] sky_map saved ->', _skp, flush=True)

    # Close WandB Run
    if config['use_wandb']:
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("experiment", type=str, help="Path to experiment file")

    args = parser.parse_args()

    experiment = SourceFileLoader(
        os.path.basename(args.experiment), args.experiment
    ).load_module()

    # Set Experiment Seed
    seed_everything(seed=experiment.config['seed'])
    
    # Create Results Directory and Copy Config
    results_dir = os.path.join(
        experiment.config["workdir"], experiment.config["run_name"]
    )
    if not experiment.config['load_checkpoint']:
        os.makedirs(results_dir, exist_ok=True)
        shutil.copy(args.experiment, os.path.join(results_dir, "config.py"))

    rgbd_slam(experiment.config)
