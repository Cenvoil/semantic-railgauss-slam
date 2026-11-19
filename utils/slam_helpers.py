import os

import torch
import torch.nn.functional as F
from utils.slam_external import build_rotation

# P-53b: 亚像素不透明度补偿（渲染期 EWA 低通）
# 取证 ss_overdraw.py（q5 帧0）：82% 高斯屏幕 σ<0.5px、95% 为 σ<1px 且 α>0.9 的"亚像素硬点"，
# 类 26/15/23 中位 σ 仅 0.02~0.09px 却 α≈1.0；本工程的 diff_gaussian_rasterization 不含
# 3DGS 的 compensation 分支 → 亚像素点仍以全额 α 光栅化到约 1 像素，百万级叠加 = 洗色/薄纱。
# 修法（mip-splatting 的 EWA 低通）：α ← α·min(1, σ_px)^p，σ 本身不动（不引入模糊）。
# 必须渲染期做：训练期尺度夹逼会把 σ_px 抬到 ≥1px，烘焙式补偿恒等于 1（已实测空转 mean_f=1.000）。
# 安全闸：因子下限 0.02，任何高斯都不会被完全抹掉。
_SUBPIX_POW = float(os.environ.get("V4_SUBPIX_POW", "0") or 0.0)
_SUBPIX_FX = float(os.environ.get("V4_FX", "2828.0") or 2828.0)
_P53B_REPORTED = [False]

# ---------- P-68 各向异性协方差（默认关闭，V4_ANISO=1 开启）----------
# 原实现：log_scales 是标量 (N,1)，builder 里 tile 成 (N,3) -> Sigma = s^2 I，
# 高斯恒为球 => unnorm_rotations 的梯度没有任何作用（球绕自身旋转不变），
# mapping 里给它的 lr=0.001 一直是空转。3DGS 原文用 Sigma = R S S^T R^T
# (S=diag(exp(s)))，本工程的 diff_gaussian_rasterization 原生接收
# (scales(N,3), rotations(N,4))，因此只需把存储从 1 分量扩到 3 分量，无需改 CUDA。
def aniso_scales(params):
    """返回渲染器要的 (N,3) 尺度。兼容旧存档：标量 (N,1) -> 广播成三分量。"""
    ls = params['log_scales']
    if ls.shape[-1] == 1:
        return torch.exp(torch.tile(ls, (1, 3)))
    return torch.exp(ls)



def _subpix_factor(params, transformed_pts):
    if _SUBPIX_POW <= 0.0:
        return 1.0
    with torch.no_grad():
        z = transformed_pts[:, 2].clamp(min=0.1).reshape(-1, 1)
        # P-68: 各向异性时用三分量的几何平均代表等效屏幕尺寸
        _ls_p68 = params['log_scales']
        _ls_p68 = (_ls_p68.reshape(-1, 1) if _ls_p68.shape[-1] == 1
                   else _ls_p68.mean(-1, keepdim=True))
        sig_px = torch.exp(_ls_p68) * (_SUBPIX_FX / z)
        f = (torch.clamp(sig_px, max=1.0) ** _SUBPIX_POW).clamp(min=0.02, max=1.0)
        if not _P53B_REPORTED[0]:
            _P53B_REPORTED[0] = True
            print("[P-53b] render-time subpix opacity pow=%.2f fx=%.1f N=%d frac_f<0.5=%.3f mean_f=%.4f"
                  % (_SUBPIX_POW, _SUBPIX_FX, sig_px.shape[0],
                     float((f < 0.5).float().mean()), float(f.mean())), flush=True)
    return f


# ---------- P-69 天空/远场背景模块（默认关闭）----------
# 可学习「方向条件全景图」S(d)：等距柱状 (3, Hs, Ws)，由像素射线的**世界方向**索引。
# 合成：C = C_gs + (1 - Λ)·S(d)（Λ = 渲染累积覆盖度，即 depth_sil 渲染的 silhouette 通道）。
_SKY_GRID = {}


def sky_uv(H, W, intrinsics, cam_quat_w2c, device="cuda"):
    """像素射线的等距柱状采样坐标 (1,H,W,2)。

    渲染在「当前帧相机系」进行（transform_to_frame 把高斯搬到该系），因此
    相机系方向 = K^-1·[u,v,1]；再乘 R_wc = build_rotation(q_cam)^T 转到世界方向。
    相机位姿在此 detach（梯度只给 sky_map，不去拽位姿）。
    """
    key = (H, W)
    d = _SKY_GRID.get(key)
    if d is None:
        fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
        cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
        v, u = torch.meshgrid(torch.arange(H, device=device).float(),
                              torch.arange(W, device=device).float(), indexing="ij")
        d = torch.stack([(u - cx) / fx, (v - cy) / fy, torch.ones_like(u)], dim=-1)
        d = d / d.norm(dim=-1, keepdim=True)
        _SKY_GRID[key] = d
    R = build_rotation(F.normalize(cam_quat_w2c.reshape(1, 4)))[0]   # world -> cam
    with torch.no_grad():
        dw = d.reshape(-1, 3) @ R                                    # = R^T d（逐行）
        lon = torch.atan2(dw[:, 1], dw[:, 0])                        # [-pi, pi]
        lat = torch.asin(dw[:, 2].clamp(-1.0, 1.0))                  # [-pi/2, pi/2]
        uu = lon / 3.141592653589793                                 # [-1, 1] 经度
        vv = 1.0 - 2.0 * (lat / 3.141592653589793 + 0.5)             # 上=+1 纬度
        uv = torch.stack([uu, vv], dim=-1).reshape(H, W, 2).unsqueeze(0)
    return uv


def sky_render(sky_map, H, W, intrinsics, cam_quat_w2c, device="cuda"):
    """按像素方向采样全景图，返回 (3, H, W)。"""
    uv = sky_uv(H, W, intrinsics, cam_quat_w2c, device=device)
    tex = sky_map.reshape(1, 3, sky_map.shape[-2], sky_map.shape[-1]).float()
    return F.grid_sample(tex, uv.float(), mode="bilinear", padding_mode="border",
                         align_corners=True)[0]

def l1_loss_v1(x, y):
    return torch.abs((x - y)).mean()


def l1_loss_v2(x, y):
    return (torch.abs(x - y).sum(-1)).mean()


def weighted_l2_loss_v1(x, y, w):
    return torch.sqrt(((x - y) ** 2) * w + 1e-20).mean()


def weighted_l2_loss_v2(x, y, w):
    return torch.sqrt(((x - y) ** 2).sum(-1) * w + 1e-20).mean()


def quat_mult(q1, q2):
    w1, x1, y1, z1 = q1.T
    w2, x2, y2, z2 = q2.T
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z]).T


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    Source: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#matrix_to_quaternion
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    Source: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#matrix_to_quaternion
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))

def project_points(points_3d, intrinsics):
    """
    Function to project 3D points to image plane.
    params:
    points_3d: [num_gaussians, 3]
    intrinsics: [3, 3]
    out: [num_gaussians, 2]
    """
    points_2d = torch.matmul(intrinsics, points_3d.transpose(0, 1))
    points_2d = points_2d.transpose(0, 1)
    points_2d = points_2d / points_2d[:, 2:]
    points_2d = points_2d[:, :2]
    return points_2d

def params2rendervar(params, device="cuda"):
    rendervar = {
        'means3D': params['means3D'],
        'colors_precomp': params['rgb_colors'],
        'rotations': F.normalize(params['unnorm_rotations']),
        'opacities': torch.sigmoid(params['logit_opacities']),
        'scales': aniso_scales(params),
        'means2D': torch.zeros_like(params['means3D'], requires_grad=True, device=device) + 0
    }
    return rendervar


def transformed_params2rendervar(params, transformed_pts, device="cuda"):
    rendervar = {
        'means3D': transformed_pts,
        'colors_precomp': params['rgb_colors'],
        'rotations': F.normalize(params['unnorm_rotations']),
        'opacities': torch.sigmoid(params['logit_opacities']) * _subpix_factor(params, transformed_pts),  # P-53b
        'scales': aniso_scales(params),
        'means2D': torch.zeros_like(params['means3D'], requires_grad=True, device=device) + 0
    }
    return rendervar

def semantics2rendervar(params, device="cuda"):
    rendervar = {
        'means3D': params['means3D'],
        'colors_precomp': params['semantic_colors'],
        'rotations': F.normalize(params['unnorm_rotations']),
        'opacities': torch.sigmoid(params['logit_opacities']),
        'scales': aniso_scales(params),
        'means2D': torch.zeros_like(params['means3D'], requires_grad=True, device=device) + 0
    }
    return rendervar

def transformed_semantics2rendervar(params, transformed_pts, device="cuda"):
    rendervar = {
        'means3D': transformed_pts,
        'colors_precomp': params['semantic_colors'],
        'rotations': F.normalize(params['unnorm_rotations']),
        'opacities': torch.sigmoid(params['logit_opacities']) * _subpix_factor(params, transformed_pts),  # P-53b
        'scales': aniso_scales(params),
        'means2D': torch.zeros_like(params['means3D'], requires_grad=True, device=device) + 0
    }
    return rendervar

def params2silhouette(params, device="cuda"):
    sil_color = torch.zeros_like(params['rgb_colors'])
    sil_color[:, 0] = 1.0
    rendervar = {
        'means3D': params['means3D'],
        'colors_precomp': sil_color,
        'rotations': F.normalize(params['unnorm_rotations']),
        'opacities': torch.sigmoid(params['logit_opacities']),
        'scales': aniso_scales(params),
        'means2D': torch.zeros_like(params['means3D'], requires_grad=True, device=device) + 0
    }
    return rendervar


def transformed_params2silhouette(params, transformed_pts, device="cuda"):
    sil_color = torch.zeros_like(params['rgb_colors'])
    sil_color[:, 0] = 1.0
    rendervar = {
        'means3D': transformed_pts,
        'colors_precomp': sil_color,
        'rotations': F.normalize(params['unnorm_rotations']),
        'opacities': torch.sigmoid(params['logit_opacities']),
        'scales': aniso_scales(params),
        'means2D': torch.zeros_like(params['means3D'], requires_grad=True, device=device) + 0
    }
    return rendervar

def get_depth_and_silhouette(pts_3D, w2c, device="cuda"):
    """
    Function to compute depth and silhouette for each gaussian.
    These are evaluated at gaussian center.
    """
    # Depth of each gaussian center in camera frame
    pts4 = torch.cat((pts_3D, torch.ones_like(pts_3D[:, :1])), dim=-1)
    pts_in_cam = (w2c @ pts4.transpose(0, 1)).transpose(0, 1)
    depth_z = pts_in_cam[:, 2].unsqueeze(-1) # [num_gaussians, 1]
    depth_z_sq = torch.square(depth_z) # [num_gaussians, 1]

    # Depth and Silhouette
    depth_silhouette = torch.zeros((pts_3D.shape[0], 3)).to(device).float()
    depth_silhouette[:, 0] = depth_z.squeeze(-1)
    depth_silhouette[:, 1] = 1.0
    depth_silhouette[:, 2] = depth_z_sq.squeeze(-1)
    
    return depth_silhouette


def params2depthplussilhouette(params, w2c, device="cuda"):
    rendervar = {
        'means3D': params['means3D'],
        'colors_precomp': get_depth_and_silhouette(params['means3D'], w2c),
        'rotations': F.normalize(params['unnorm_rotations']),
        'opacities': torch.sigmoid(params['logit_opacities']),
        'scales': aniso_scales(params),
        'means2D': torch.zeros_like(params['means3D'], requires_grad=True, device=device) + 0
    }
    return rendervar


def transformed_params2depthplussilhouette(params, w2c, transformed_pts, device="cuda"):
    rendervar = {
        'means3D': transformed_pts,
        'colors_precomp': get_depth_and_silhouette(transformed_pts, w2c),
        'rotations': F.normalize(params['unnorm_rotations']),
        'opacities': torch.sigmoid(params['logit_opacities']) * _subpix_factor(params, transformed_pts),  # P-53b
        'scales': aniso_scales(params),
        'means2D': torch.zeros_like(params['means3D'], requires_grad=True, device=device) + 0
    }
    return rendervar


def transform_to_frame(params, time_idx, gaussians_grad, camera_grad, device="cuda"):
    """
    Function to transform Isotropic Gaussians from world frame to camera frame.
    
    Args:
        params: dict of parameters
        time_idx: time index to transform to
        gaussians_grad: enable gradients for Gaussians
        camera_grad: enable gradients for camera pose
    
    Returns:
        transformed_pts: Transformed Centers of Gaussians
    """
    # Get Frame Camera Pose
    if camera_grad:
        cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx])
        cam_tran = params['cam_trans'][..., time_idx]
    else:
        cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
        cam_tran = params['cam_trans'][..., time_idx].detach()
    rel_w2c = torch.eye(4).to(device).float()
    rel_w2c[:3, :3] = build_rotation(cam_rot)
    rel_w2c[:3, 3] = cam_tran

    # Get Centers and norm Rots of Gaussians in World Frame
    if gaussians_grad:
        pts = params['means3D']
    else:
        pts = params['means3D'].detach()
    
    # Transform Centers and Unnorm Rots of Gaussians to Camera Frame
    pts_ones = torch.ones(pts.shape[0], 1).to(device).float()
    pts4 = torch.cat((pts, pts_ones), dim=1)
    transformed_pts = (rel_w2c @ pts4.T).T[:, :3]

    return transformed_pts