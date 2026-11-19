#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""proj_canon —— LiDAR/3D -> rgb_highres_center 图像的唯一权威投影（2026-09-25 v2）。

⚠ 使用规则：任何 LiDAR 点云→图像、poly3d/cuboid→图像、深度图 vs LiDAR 的工作，
  一律 import 本模块，禁止自写投影。旧实现（compress_curve / render_lidar_cmp /
  lidar_rail_split / lidar_r_final / overlay_check）已封存 _sealed_r曲线_9-21to24/。

口径来源（v2 终版）：**官方 labels.json coordinate_systems['rgb_highres_center'].
pose_wrt_parent 原值**（四元数 xyzw，d 系列口径文档 9-18 自证）——
    t = (0.0801578, -0.332862, 3.50982)   # 相机离 base 原点（Z=0=轨顶面）3.51m
    q = (-0.00313306, 0.0562995, 0.00482918, 0.998397)  # 旋转 6.49° 绕 Y（下俯）
内参 cy=1232.862908875167（d 系列，原生 1486.863 − 254 裁切）+ 官方畸变。
零拟合参数。verify5 实测：poly3d↔poly2d 中位失配 5.3px；20 帧 r 表 nonrail
0.92-1.00、rail 1.00-1.03（见 out/r_canon/r_table.json）。

⚠ 官方数据存在两套不一致的 rgb_highres_center 外参：
    - labels.json coordinate_systems：3.510m + 6.49° 俯仰 —— 与图像内容自洽（本口径）；
    - calibration.txt CAMERA 节：2.078m + 无俯角 —— 与图像不符（伪影根源）。
  一切与图像对齐的工作以 labels.json 为准；引用 calibration.txt 相机外参必须注明
  「与图像不一致，禁用」。差异归属（口径不同/重装/笔误）待向数据方求证。

历史注记：9-25 凌晨曾用「内容定律拟合」（TZ=3.443, θ=6.49°, v1）独立逼近同一模型
（poly3d 失配 40.9px），v2 的 json 原值全面优于 v1，v1 作废。内容定律
（span(v)=0.438·(v−406) 等）保留为独立确认证据（审计文档 9-25 续审 F 节）。

已知局限：轨顶平面点的内容定律 v ≈ VH + C/z（VH≈406, C≈25026）为小角形式，
与本实现的偏差 ≤16px（远场）；近场 <15m 如需亚像素精度须重新拟合。
"""
import math

import numpy as np

# ---- 官方常量 ----
FX = FY = 7267.95450880415
CX = 2056.049238502414
CY = 1232.862908875167          # d 系列（原生 1486.863 − 254 裁切）
W, H = 4112, 2504
K1, K2, K3 = -0.0764891, -0.188057, 1.78279   # p1=p2=0（官方原值）

# ---- 官方 labels.json coordinate_systems['rgb_highres_center'] 原值（xyzw）----
Q_JSON = (-0.00313306, 0.0562995, 0.00482918, 0.998397)
T_JSON = (0.0801578, -0.332862, 3.50982)


def _quat_to_R(xyzw):
    x, y, z, w = xyzw
    n = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


RJ = _quat_to_R(Q_JSON)
TC = np.array(T_JSON)
ROT_ANGLE_DEG = math.degrees(2 * math.acos(min(1.0, abs(Q_JSON[3]))))   # ≈6.49


def _calib_law():
    zs = (20.0, 100.0)
    vs = []
    for z0 in zs:
        b = RJ.T @ (np.array([z0, 0.0, 0.0]) - TC)
        v0 = CY + FY * (-b[2]) / b[0]
        xn = 0.0
        k = 1.0
        vs.append(v0)
    # 解线性方程: v_i = VH + C/z_i（两点精确解）
    A = np.array([[1.0, 1.0 / zs[0]], [1.0, 1.0 / zs[1]]])
    vh, c = np.linalg.solve(A, np.array(vs))
    return vh, c


VH_LAW, C_LAW = _calib_law()


def project(P):
    """P: (N,3) 或 (3,) lidar/base 系点（官方移动参照系，Z=0=轨顶面）。
    返回 (z, u, v)：z=前向距离 Pc_x（米）；u,v=像素坐标（未取整，含畸变）。不裁剪。"""
    P = np.asarray(P, dtype=np.float64)
    single = (P.ndim == 1)
    if single:
        P = P[None, :]
    b = (RJ.T @ (P - TC[None, :]).T).T
    z = b[:, 0]
    u0 = CX + FX * (-b[:, 1]) / z
    v0 = CY + FY * (-b[:, 2]) / z
    xn = (u0 - CX) / FX
    yn = (v0 - CY) / FY
    r2 = xn * xn + yn * yn
    k = 1.0 + K1 * r2 + K2 * r2 * r2 + K3 * r2 * r2 * r2
    u = CX + xn * k * FX
    v = CY + yn * k * FY
    if single:
        return float(z[0]), float(u[0]), float(v[0])
    return z, u, v


def ground_row_depth(v):
    """轨顶平面上、图像行 v 处的前向距离 z（内容定律反函数，VH/C 由本模型自算）。
    仅用于轨顶/道床平面点的行号↔距离换算；精度受小角形式限制（≤16px 远场）。"""
    return C_LAW / (np.asarray(v, dtype=np.float64) - VH_LAW)


if __name__ == '__main__':
    for z0 in (12.1, 20.0, 40.0, 75.0, 110.0):
        _, u, v = project(np.array([z0, 0.0, 0.0]))
        print('z=%6.1f m -> u=%7.1f v=%7.1f   (内容定律 v=%.1f)'
              % (z0, u, v, VH_LAW + C_LAW / z0))
    print('旋转角=%.2f°  VH=%.1f  C=%.0f' % (ROT_ANGLE_DEG, VH_LAW, C_LAW))
