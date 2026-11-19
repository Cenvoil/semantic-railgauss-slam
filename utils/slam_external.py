"""
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file found here:
# https://github.com/graphdeco-inria/gaussian-splatting/blob/main/LICENSE.md
#
# For inquiries contact  george.drettakis@inria.fr

#######################################################################################################################
##### NOTE: CODE IN THIS FILE IS NOT INCLUDED IN THE OVERALL PROJECT'S MIT LICENSE #####
##### USE OF THIS CODE FOLLOWS THE COPYRIGHT NOTICE ABOVE #####
#######################################################################################################################
"""

import os

import numpy as np
import torch
import torch.nn.functional as func
from torch.autograd import Variable
from math import exp


def build_rotation(q, device="cuda"):
    norm = torch.sqrt(q[:, 0] * q[:, 0] + q[:, 1] * q[:, 1] + q[:, 2] * q[:, 2] + q[:, 3] * q[:, 3])
    q = q / norm[:, None]
    rot = torch.zeros((q.size(0), 3, 3), device=device)
    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
    rot[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rot[:, 0, 1] = 2 * (x * y - r * z)
    rot[:, 0, 2] = 2 * (x * z + r * y)
    rot[:, 1, 0] = 2 * (x * y + r * z)
    rot[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rot[:, 1, 2] = 2 * (y * z - r * x)
    rot[:, 2, 0] = 2 * (x * z - r * y)
    rot[:, 2, 1] = 2 * (y * z + r * x)
    rot[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return rot


def calc_mse(img1, img2):
    return ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)


def calc_psnr(img1, img2):
    mse = ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def calc_iou(img1, img2):
    pass

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def calc_ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = func.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = func.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = func.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = func.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = func.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def accumulate_mean2d_gradient(variables):
    variables['means2D_gradient_accum'][variables['seen']] += torch.norm(
        variables['means2D'].grad[variables['seen'], :2], dim=-1)
    variables['denom'][variables['seen']] += 1
    return variables


def update_params_and_optimizer(new_params, params, params_opt_exclude, optimizer):
    for k, v in new_params.items():
        if k in params_opt_exclude:
            params[k] = new_params[k]
            continue
        group = [x for x in optimizer.param_groups if x["name"] == k][0]
        stored_state = optimizer.state.get(group['params'][0], None)

        stored_state["exp_avg"] = torch.zeros_like(v)
        stored_state["exp_avg_sq"] = torch.zeros_like(v)
        del optimizer.state[group['params'][0]]

        group["params"][0] = torch.nn.Parameter(v.requires_grad_(True))
        optimizer.state[group['params'][0]] = stored_state
        params[k] = group["params"][0]
    return params


def cat_params_to_optimizer(new_params, params, params_opt_exclude, optimizer):
    for k, v in new_params.items():
        if k in params_opt_exclude:
            params[k] = torch.cat((params[k], v), dim=0)
            continue
        group = [g for g in optimizer.param_groups if g['name'] == k][0]
        stored_state = optimizer.state.get(group['params'][0], None)
        if stored_state is not None:
            stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(v)), dim=0)
            stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(v)), dim=0)
            del optimizer.state[group['params'][0]]
            group["params"][0] = torch.nn.Parameter(torch.cat((group["params"][0], v), dim=0).requires_grad_(True))
            optimizer.state[group['params'][0]] = stored_state
            params[k] = group["params"][0]
        else:
            group["params"][0] = torch.nn.Parameter(torch.cat((group["params"][0], v), dim=0).requires_grad_(True))
            params[k] = group["params"][0]
    return params


def remove_points(to_remove, params, params_opt_exclude, variables, optimizer):
    to_keep = ~to_remove
    keys = [k for k in params.keys() if k not in ['cam_unnorm_rots', 'cam_trans']]
    for k in keys:
        # Keys not in optimizer
        if k in params_opt_exclude:
            params[k] = params[k][to_keep]
            continue
        group = [g for g in optimizer.param_groups if g['name'] == k][0]
        stored_state = optimizer.state.get(group['params'][0], None)
        if stored_state is not None:
            stored_state["exp_avg"] = stored_state["exp_avg"][to_keep]
            stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][to_keep]
            del optimizer.state[group['params'][0]]
            group["params"][0] = torch.nn.Parameter((group["params"][0][to_keep].requires_grad_(True)))
            optimizer.state[group['params'][0]] = stored_state
            params[k] = group["params"][0]
        else:
            group["params"][0] = torch.nn.Parameter(group["params"][0][to_keep].requires_grad_(True))
            params[k] = group["params"][0]
    variables['means2D_gradient_accum'] = variables['means2D_gradient_accum'][to_keep]
    variables['denom'] = variables['denom'][to_keep]
    variables['max_2D_radius'] = variables['max_2D_radius'][to_keep]
    if 'timestep' in variables.keys():
        variables['timestep'] = variables['timestep'][to_keep]
    return params, variables


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


def prune_gaussians(params, params_opt_exclude, variables, optimizer, iter, prune_dict,
                    time_idx=None, cam_center=None, cam_fwd=None):
    if iter <= prune_dict['stop_after']:
        if (iter >= prune_dict['start_after']) and (iter % prune_dict['prune_every'] == 0):
            if iter == prune_dict['stop_after']:
                remove_threshold = prune_dict['final_removal_opacity_threshold']
            else:
                remove_threshold = prune_dict['removal_opacity_threshold']
            # Remove Gaussians with low opacity
            to_remove = (torch.sigmoid(params['logit_opacities']) < remove_threshold).squeeze()
            # Remove Gaussians that are too big
            if iter >= prune_dict['remove_big_after']:
                big_points_ws = torch.exp(params['log_scales']).max(dim=1).values > 0.1 * variables['scene_radius']
                to_remove = torch.logical_or(to_remove, big_points_ws)
            params, variables = remove_points(to_remove, params, params_opt_exclude, variables, optimizer)
            torch.cuda.empty_cache()
        
        # Reset Opacities for all Gaussians
        if iter > 0 and iter % prune_dict['reset_opacities_every'] == 0 and prune_dict['reset_opacities']:
            new_params = {'logit_opacities': inverse_sigmoid(torch.ones_like(params['logit_opacities']) * 0.01)}
            params = update_params_and_optimizer(new_params, params, optimizer)

    # P-34: 近场陈旧高斯剪枝（默认关闭，由 pruning_dict['near_stale_enable'] 开启）
    # 目标：堆在当前相机跟前、且已很久没被新观测支撑的"糊脸 billboard"高斯
    if prune_dict.get('near_stale_enable', False):
        if (time_idx is not None) and (curr_cam_tran is not None) and ('timestep' in variables.keys()):
            if (iter >= prune_dict.get('near_stale_start_after', 20)) and (iter % prune_dict.get('near_stale_every', 20) == 0):
                with torch.no_grad():
                    depth_thresh = prune_dict.get('near_stale_depth', 3.0)
                    max_age = prune_dict.get('near_stale_max_age', 25)
                    max_ratio = prune_dict.get('near_stale_max_ratio', 0.5)
                    num_pts = params['means3D'].shape[0]
                    dist = torch.linalg.norm(params['means3D'] - curr_cam_tran.reshape(1, 3), dim=1)
                    near_mask = dist < depth_thresh
                    age = time_idx - variables['timestep']
                    stale_mask = age > max_age
                    to_remove_ns = torch.logical_and(near_mask, stale_mask)
                    n_near = int(near_mask.sum().item())
                    n_near20 = int((dist < 20.0).sum().item())
                    n_stale = int(stale_mask.sum().item())
                    n_hit = int(to_remove_ns.sum().item())
                    if n_hit > max_ratio * num_pts:
                        print(f"[P34] t={time_idx} it={iter} near={n_near}(<{depth_thresh}m) near20={n_near20} stale={n_stale} HIT={n_hit} "
                              f"SKIP(>{max_ratio:.0%} of {num_pts}) age>{max_age}", flush=True)
                    else:
                        print(f"[P34] t={time_idx} it={iter} near={n_near}(<{depth_thresh}m) near20={n_near20} stale={n_stale} HIT={n_hit} "
                              f"(age>{max_age})", flush=True)
                        if n_hit > 0:
                            params, variables = remove_points(to_remove_ns, params, params_opt_exclude, variables, optimizer)
                            torch.cuda.empty_cache()

    # P-36: 相机身后陈旧高斯清扫（默认关闭，由 pruning_dict['behind_stale_enable'] 开启）
    # 判据：沿运动方向投影在相机平面之后 >margin 米，且创建超过 max_age 帧。
    # 线性行驶场景（铁路）不会回看，身后几何既不可见也不会再被观测，清除安全。
    if prune_dict.get('behind_stale_enable', False):
        if (time_idx is not None) and (cam_center is not None) and (cam_fwd is not None) \
                and ('timestep' in variables.keys()):
            if (iter >= prune_dict.get('behind_stale_start_after', 20)) and (iter % prune_dict.get('behind_stale_every', 20) == 0):
                with torch.no_grad():
                    margin = prune_dict.get('behind_stale_margin', 5.0)
                    m_age = prune_dict.get('behind_stale_max_age', 40)
                    m_ratio = prune_dict.get('behind_stale_max_ratio', 0.5)
                    n_pts = params['means3D'].shape[0]
                    fwd = cam_fwd.reshape(1, 3)
                    along = (params['means3D'] - cam_center.reshape(1, 3)).mul(fwd).sum(dim=1)
                    old_mask = (time_idx - variables['timestep']) > m_age
                    behind_mask = along < -margin
                    to_rm = torch.logical_and(old_mask, behind_mask)
                    n_rm = int(to_rm.sum().item())
                    if n_rm > m_ratio * n_pts:
                        print(f"[P36] t={time_idx} it={iter} behind&old={n_rm} SKIP(>{m_ratio:.0%} of {n_pts}) "
                              f"margin={margin} age>{m_age}", flush=True)
                    else:
                        print(f"[P36] t={time_idx} it={iter} behind&old={n_rm} SWEEP "
                              f"(margin={margin}, age>{m_age})", flush=True)
                        if n_rm > 0:
                            params, variables = remove_points(to_rm, params, params_opt_exclude, variables, optimizer)
                            torch.cuda.empty_cache()

    # P-44: 近相机剪枝（默认关闭，pruning_dict['near_cam_prune_dist'] 开启）
    # 实测 1.2：14% 的高斯距相机 <2m、1.4 万个 <0.5m 且全不透明 —— 相机几乎从它们中间穿过，
    # 渲染时形成一片虚焦圆斑（用户反馈的"浮空不明物"）。这类几何物理上不可能存在，删除无风险。
    _nc = prune_dict.get('near_cam_prune_dist', 0.0)
    if (_nc is not None) and (_nc > 0) and (cam_center is not None):
        if (iter >= prune_dict.get('near_cam_start_after', 20)) and (iter % prune_dict.get('near_cam_every', 20) == 0):
            with torch.no_grad():
                n_pts = params['means3D'].shape[0]
                _d = torch.linalg.norm(params['means3D'] - cam_center.reshape(1, 3), dim=1)
                to_rm_nc = _d < _nc
                n_nc = int(to_rm_nc.sum().item())
                m_ratio = prune_dict.get('near_cam_max_ratio', 0.05)
                if n_nc > m_ratio * n_pts:
                    print(f"[P44] t={time_idx} it={iter} near_cam={n_nc} SKIP(>{m_ratio:.0%} of {n_pts}) "
                          f"dist<{_nc}", flush=True)
                elif n_nc > 0:
                    print(f"[P44] t={time_idx} it={iter} near_cam={n_nc} PRUNED (dist<{_nc}m)", flush=True)
                    params, variables = remove_points(to_rm_nc, params, params_opt_exclude, variables, optimizer)
                    torch.cuda.empty_cache()

    return params, variables


def densify(params, variables, optimizer, iter, densify_dict, params_opt_exclude, device="cuda"):
    if iter <= densify_dict['stop_after']:
        variables = accumulate_mean2d_gradient(variables)
        grad_thresh = densify_dict['grad_thresh']
        if (iter >= densify_dict['start_after']) and (iter % densify_dict['densify_every'] == 0):
            grads = variables['means2D_gradient_accum'] / variables['denom']
            grads[grads.isnan()] = 0.0
            # P-82: densify 判据与 loss 权重解耦（im 权重只影响优化，不应改变增点节奏）
            _gs = float(os.environ.get('V4_DENSIFY_GRAD_SCALE', '1.0'))
            if _gs != 1.0:
                grads = grads * _gs
            to_clone = torch.logical_and(grads >= grad_thresh, (
                        torch.max(torch.exp(params['log_scales']), dim=1).values <= 0.01 * variables['scene_radius']))
            new_params = {k: v[to_clone] for k, v in params.items() if k not in ['cam_unnorm_rots', 'cam_trans']}
            params = cat_params_to_optimizer(new_params, params, params_opt_exclude, optimizer)
            if 'timestep' in variables:  # v5tstepfix: inherit parent timestep for cloned gaussians
                variables['timestep'] = torch.cat((variables['timestep'], variables['timestep'][to_clone]), dim=0)
            # P-77: 透射率保持克隆 (transmittance-preserving densification, arXiv 2608.14902)
            # 标准 3DGS 克隆原样复制父本 opacity：两份重合高斯等效遮挡 1-(1-a)^2 > a，
            # 造成 over-occlusion / opacity drift，也是「加点比动点便宜」的激励来源。
            # 修正：父子同设 a' = 1-(1-a)^p；p=0.5 完全保持透射率；p=1.0 原行为。
            _tpd_p = float(os.environ.get("V4_TPD_P", "1.0"))
            if _tpd_p < 1.0:
                _nc = int(to_clone.sum())
                if _nc > 0:
                    _n_old = to_clone.shape[0]
                    _flat_lo = params['logit_opacities'].reshape(-1)
                    _a = torch.sigmoid(_flat_lo[:_n_old][to_clone.reshape(-1)]).clamp(1e-6, 1.0 - 1e-6)
                    _ap = (1.0 - (1.0 - _a).pow(_tpd_p)).clamp(1e-6, 1.0 - 1e-6)
                    _nl = inverse_sigmoid(_ap)
                    _idx = torch.nonzero(to_clone.reshape(-1)).squeeze(-1)
                    with torch.no_grad():
                        _flat_lo[_idx] = _nl      # 父本降 opacity
                        _flat_lo[_n_old:] = _nl   # 新克隆副本（cat 顺序保证新增点在尾部）
                    if os.environ.get("V4_TPD_DEBUG", "0") == "1":
                        print("[P-77] iter=%d clone=%d alpha %.4f -> %.4f  (p=%.2f)" % (
                            iter, _nc, _a.mean().item(), _ap.mean().item(), _tpd_p))
            num_pts = params['means3D'].shape[0]

            padded_grad = torch.zeros(num_pts, device=device)
            padded_grad[:grads.shape[0]] = grads
            to_split = torch.logical_and(padded_grad >= grad_thresh,
                                         torch.max(torch.exp(params['log_scales']), dim=1).values > 0.01 * variables[
                                             'scene_radius'])
            n = densify_dict['num_to_split_into']  # number to split into
            new_params = {k: (v[to_split].repeat(n) if v.dim() == 1 else v[to_split].repeat(n, 1)) for k, v in params.items() if k not in ['cam_unnorm_rots', 'cam_trans']}  # v5densfix: 1-D keys
            # P-68: 各向异性时 log_scales 已是 (N,3) —— 只在标量情况下才把 1 分量铺成 3
            _sd = torch.exp(params['log_scales'])[to_split]
            stds = _sd.repeat(n, 3 // _sd.shape[-1])
            means = torch.zeros((stds.size(0), 3), device=device)
            samples = torch.normal(mean=means, std=stds)
            rots = build_rotation(params['unnorm_rotations'][to_split], device=device).repeat(n, 1, 1)
            new_params['means3D'] += torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
            new_params['log_scales'] = torch.log(torch.exp(new_params['log_scales']) / (0.8 * n))
            params = cat_params_to_optimizer(new_params, params, params_opt_exclude, optimizer)
            if 'timestep' in variables:  # v5tstepfix: inherit parent timestep for split gaussians
                variables['timestep'] = torch.cat((variables['timestep'], variables['timestep'][to_split].repeat(n)), dim=0)
            num_pts = params['means3D'].shape[0]

            variables['means2D_gradient_accum'] = torch.zeros(num_pts, device=device)
            variables['denom'] = torch.zeros(num_pts, device=device)
            variables['max_2D_radius'] = torch.zeros(num_pts, device=device)
            to_remove = torch.cat((to_split, torch.zeros(n * to_split.sum(), dtype=torch.bool, device=device)))
            params, variables = remove_points(to_remove, params, params_opt_exclude, variables, optimizer)

            if iter == densify_dict['stop_after']:
                remove_threshold = densify_dict['final_removal_opacity_threshold']
            else:
                remove_threshold = densify_dict['removal_opacity_threshold']
            to_remove = (torch.sigmoid(params['logit_opacities']) < remove_threshold).squeeze()
            if iter >= densify_dict['remove_big_after']:
                big_points_ws = torch.exp(params['log_scales']).max(dim=1).values > 0.1 * variables['scene_radius']
                to_remove = torch.logical_or(to_remove, big_points_ws)
            params, variables = remove_points(to_remove, params, params_opt_exclude, variables, optimizer)

            torch.cuda.empty_cache()

        # Reset Opacities for all Gaussians (This is not desired for mapping on only current frame)
        if iter > 0 and iter % densify_dict['reset_opacities_every'] == 0 and densify_dict['reset_opacities']:
            new_params = {'logit_opacities': inverse_sigmoid(torch.ones_like(params['logit_opacities']) * 0.01)}
            params = update_params_and_optimizer(new_params, params, params_opt_exclude, optimizer)

    return params, variables


def update_learning_rate(optimizer, means3D_scheduler, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in optimizer.param_groups:
            if param_group["name"] == "means3D":
                lr = means3D_scheduler(iteration)
                param_group['lr'] = lr
                return lr


def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper