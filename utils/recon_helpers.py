import os as _os
import torch
from diff_gaussian_rasterization import GaussianRasterizationSettings as Camera

def setup_camera(w, h, k, w2c, near=0.01, far=None, device="cuda"):
    # CAM_FAR: 渲染远平面。原硬编码 far=100 源自 Replica 室内数据集（室内 <100 m），
    # 铁路场景远场远超 100 m；far 只用于 projmatrix 的 z 行，若 rasterizer 按 NDC z 裁剪，
    # 则 >far 的高斯会被剔除。默认仍取 100 以保持既有 run 的复现性（2026-09-21）。
    if far is None:
        far = float(_os.environ.get("CAM_FAR", "100.0"))
    fx, fy, cx, cy = k[0][0], k[1][1], k[0][2], k[1][2]
    w2c = torch.tensor(w2c).to(device).float()
    cam_center = torch.inverse(w2c)[:3, 3]
    w2c = w2c.unsqueeze(0).transpose(1, 2)
    opengl_proj = torch.tensor([[2 * fx / w, 0.0, -(w - 2 * cx) / w, 0.0],
                                [0.0, 2 * fy / h, -(h - 2 * cy) / h, 0.0],
                                [0.0, 0.0, far / (far - near), -(far * near) / (far - near)],
                                [0.0, 0.0, 1.0, 0.0]]).to(device).float().unsqueeze(0).transpose(1, 2)
    full_proj = w2c.bmm(opengl_proj)
    cam = Camera(
        image_height=h,
        image_width=w,
        tanfovx=w / (2 * fx),
        tanfovy=h / (2 * fy),
        bg=(torch.tensor([1, 1, 1], dtype=torch.float32, device=device) if _os.environ.get("V4_BG_WHITE", _os.environ.get("V4_GEO_MASK", "0")) == "1" else torch.tensor([0, 0, 0], dtype=torch.float32, device=device)),  # v4 F2: white bg for unmodeled sky/far
        scale_modifier=1.0,
        viewmatrix=w2c,
        projmatrix=full_proj,
        sh_degree=0,
        campos=cam_center,
        prefiltered=False
    )
    return cam
