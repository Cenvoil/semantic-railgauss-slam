#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S2 patch: semantic head (19 logits) + semantic rendering + CE loss into the LIVO-GS fork.

Backups: *.bak_s2 (created once). All replacements are asserted; nothing is
written unless every anchor matches.
"""
import os
import shutil

R = "<PROJECT_ROOT>/livo_gs/gaussian-splatting-using-LIVO"


def apply(rel, pairs):
    p = os.path.join(R, rel)
    s = open(p).read()
    s2 = s
    for old, new in pairs:
        assert old in s2, "anchor not found in %s: %r" % (rel, old[:120])
        s2 = s2.replace(old, new, 1)
    bak = p + ".bak_s2"
    if not os.path.exists(bak):
        shutil.copy(p, bak)
    open(p, "w").write(s2)
    print("patched", rel)


# ---------------- scene/gaussian_model.py ----------------
apply("scene/gaussian_model.py", [
    ("        self._opacity = torch.empty(0)\n        self.max_radii2D = torch.empty(0)",
     "        self._opacity = torch.empty(0)\n"
     "        self._sem_logits = torch.empty(0)   # S2: per-Gaussian 19-class logits\n"
     "        self.max_radii2D = torch.empty(0)"),

    ("            self.optimizer.state_dict(),\n            self.spatial_lr_scale,\n        )",
     "            self.optimizer.state_dict(),\n            self.spatial_lr_scale,\n"
     "            self._sem_logits,\n        )"),

    ("""    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)""",
     """    def restore(self, model_args, training_args):
        # S2: backward-compatible with pre-semantic checkpoints (12-tuple)
        sem_logits = None
        if len(model_args) == 13:
            (self.active_sh_degree,
             self._xyz,
             self._features_dc,
             self._features_rest,
             self._scaling,
             self._rotation,
             self._opacity,
             self.max_radii2D,
             xyz_gradient_accum,
             denom,
             opt_dict,
             self.spatial_lr_scale,
             sem_logits) = model_args
        else:
            (self.active_sh_degree,
             self._xyz,
             self._features_dc,
             self._features_rest,
             self._scaling,
             self._rotation,
             self._opacity,
             self.max_radii2D,
             xyz_gradient_accum,
             denom,
             opt_dict,
             self.spatial_lr_scale) = model_args
        if sem_logits is not None and sem_logits.numel() > 0:
            self._sem_logits = nn.Parameter(sem_logits.requires_grad_(True))
        elif self._sem_logits.numel() == 0:
            self._sem_logits = nn.Parameter(
                torch.zeros((self._xyz.shape[0], 19), device="cuda").requires_grad_(True))
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)"""),

    ("        self._opacity = nn.Parameter(opacities.requires_grad_(True))\n        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=\"cuda\")",
     "        self._opacity = nn.Parameter(opacities.requires_grad_(True))\n"
     "        self._sem_logits = nn.Parameter(torch.zeros((self.get_xyz.shape[0], 19), device=\"cuda\").requires_grad_(True))\n"
     "        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=\"cuda\")"),

    ("            {'params': [self._rotation], 'lr': training_args.rotation_lr, \"name\": \"rotation\"}\n        ]",
     "            {'params': [self._rotation], 'lr': training_args.rotation_lr, \"name\": \"rotation\"},\n"
     "            {'params': [self._sem_logits], 'lr': getattr(training_args, 'semantic_lr', 0.001), \"name\": \"sem\"}\n        ]"),

    ("        for i in range(self._rotation.shape[1]):\n            l.append('rot_{}'.format(i))\n        return l",
     "        for i in range(self._rotation.shape[1]):\n            l.append('rot_{}'.format(i))\n"
     "        for i in range(self._sem_logits.shape[1] if self._sem_logits.numel() else 19):\n"
     "            l.append('sem_{}'.format(i))\n        return l"),

    ("        rotation = self._rotation.detach().cpu().numpy()\n\n        dtype_full",
     "        rotation = self._rotation.detach().cpu().numpy()\n"
     "        sem = (self._sem_logits.detach().cpu().numpy() if self._sem_logits.numel()\n"
     "               else np.zeros((xyz.shape[0], 19), np.float32))\n\n        dtype_full"),

    ("        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)",
     "        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, sem), axis=1)"),

    ("        for idx, attr_name in enumerate(rot_names):\n            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])\n\n        self._xyz = nn.Parameter",
     "        for idx, attr_name in enumerate(rot_names):\n            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])\n\n"
     "        sem_names = [p.name for p in plydata.elements[0].properties if p.name.startswith(\"sem_\")]\n"
     "        if sem_names:\n"
     "            sem_names = sorted(sem_names, key=lambda x: int(x.split('_')[-1]))\n"
     "            sem = np.zeros((xyz.shape[0], len(sem_names)))\n"
     "            for idx, attr_name in enumerate(sem_names):\n"
     "                sem[:, idx] = np.asarray(plydata.elements[0][attr_name])\n"
     "        else:\n"
     "            sem = np.zeros((xyz.shape[0], 19))\n\n        self._xyz = nn.Parameter"),

    ("        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device=\"cuda\").requires_grad_(True))\n\n        self.active_sh_degree",
     "        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device=\"cuda\").requires_grad_(True))\n"
     "        self._sem_logits = nn.Parameter(torch.tensor(sem, dtype=torch.float, device=\"cuda\").requires_grad_(True))\n\n        self.active_sh_degree"),

    ("        self._rotation = optimizable_tensors[\"rotation\"]\n\n        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]",
     "        self._rotation = optimizable_tensors[\"rotation\"]\n"
     "        self._sem_logits = optimizable_tensors[\"sem\"]\n\n        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]"),

    ("""    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]""",
     """    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_sem_logits):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        "sem" : new_sem_logits}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._sem_logits = optimizable_tensors["sem"]"""),

    ("        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)\n\n        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)",
     "        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)\n"
     "        new_sem_logits = self._sem_logits[selected_pts_mask].repeat(N,1)\n\n"
     "        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_sem_logits)"),

    ("        new_rotation = self._rotation[selected_pts_mask]\n\n        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)",
     "        new_rotation = self._rotation[selected_pts_mask]\n        new_sem_logits = self._sem_logits[selected_pts_mask]\n\n"
     "        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_sem_logits)"),
])

# ---------------- gaussian_renderer/__init__.py ----------------
renderer_old_return = """    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}"""
render_sem_fn = '''

def render_sem(viewpoint_camera, pc : GaussianModel, pipe, scaling_modifier = 1.0):
    """S2: 20-channel semantic render (0..18 class logits, 19 = alpha/coverage)."""
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    feat = torch.cat([pc._sem_logits, torch.ones_like(pc._sem_logits[:, :1])], dim=1)

    raster_settings = SemRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=torch.zeros(feat.shape[1], device=feat.device),
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=0,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )
    rasterizer = SemRasterizer(raster_settings=raster_settings)

    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    out, radii = rasterizer(
        means3D = pc.get_xyz,
        means2D = torch.zeros_like(pc.get_xyz),
        shs = None,
        colors_precomp = feat,
        opacities = pc.get_opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    return out[:19], out[19:20], radii
'''

apply("gaussian_renderer/__init__.py", [
    ("from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer",
     "from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer\n"
     "from diff_semantic_rasterization import GaussianRasterizationSettings as SemRasterizationSettings, "
     "GaussianRasterizer as SemRasterizer"),
    (renderer_old_return, renderer_old_return + render_sem_fn),
])

# ---------------- train.py ----------------
apply("train.py", [
    ("import os\nimport torch", "import os\nimport numpy as np\nimport cv2\nimport torch"),
    ("from gaussian_renderer import render, network_gui",
     "from gaussian_renderer import render, render_sem, network_gui"),
    ("def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, sparse_num):",
     "def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, sparse_num, sem_cfg):"),

    ("    background = torch.tensor(bg_color, dtype=torch.float32, device=\"cuda\")\n",
     "    background = torch.tensor(bg_color, dtype=torch.float32, device=\"cuda\")\n\n"
     "    # S2: semantic teacher index (aligned YOLO zs maps, same stems as GS images)\n"
     "    sem_paths = {}\n"
     "    sem_cache = {}\n"
     "    if sem_cfg.lambda_seg > 0 and sem_cfg.semantic_dir:\n"
     "        import glob\n"
     "        for p in glob.glob(os.path.join(sem_cfg.semantic_dir, \"*.png\")):\n"
     "            sem_paths[os.path.splitext(os.path.basename(p))[0]] = p\n"
     "        print(\"[S2] semantic teacher frames: {}\".format(len(sem_paths)))\n\n"
     "    def get_sem(name):\n"
     "        if name in sem_cache:\n"
     "            return sem_cache[name]\n"
     "        p = sem_paths.get(name)\n"
     "        if p is None:\n"
     "            return None\n"
     "        a = cv2.imread(p, cv2.IMREAD_UNCHANGED)\n"
     "        if a.ndim == 3:\n"
     "            a = a[..., 0]\n"
     "        sem_cache[name] = a.astype(np.uint8)\n"
     "        return sem_cache[name]\n"),

    ("        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))\n        loss.backward()",
     "        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))\n\n"
     "        # S2: semantic head loss (CE against the aligned teacher maps)\n"
     "        sem_ce_val = 0.0\n"
     "        if sem_cfg.lambda_seg > 0:\n"
     "            sem_gt = get_sem(viewpoint_cam.image_name)\n"
     "            if sem_gt is not None:\n"
     "                sem_logits, sem_cov, _ = render_sem(viewpoint_cam, gaussians, pipe)\n"
     "                gt_sem_t = torch.from_numpy(sem_gt).long().to(sem_logits.device)\n"
     "                valid = (gt_sem_t < 19) & (sem_cov[0] > 0.05)\n"
     "                if valid.any():\n"
     "                    ce = F.cross_entropy(sem_logits, gt_sem_t, reduction='none')\n"
     "                    sem_ce = ce[valid].mean()\n"
     "                    w_seg = sem_cfg.lambda_seg * min(1.0, iteration / max(1, sem_cfg.seg_warmup))\n"
     "                    loss = loss + w_seg * sem_ce\n"
     "                    sem_ce_val = float(sem_ce.detach())\n        loss.backward()"),

    ("                progress_bar.update(10)",
     "                progress_bar.update(10)\n"
     "            if iteration % 200 == 0 and sem_cfg.lambda_seg > 0:\n"
     "                print(\"\\n[S2] iter {} sem_ce {:.4f}\".format(iteration, sem_ce_val))"),

    ("    parser.add_argument(\"--quiet\", action=\"store_true\")",
     "    parser.add_argument(\"--quiet\", action=\"store_true\")\n"
     "    parser.add_argument(\"--semantic_dir\", type=str, default=\"\")\n"
     "    parser.add_argument(\"--lambda_seg\", type=float, default=0.0)\n"
     "    parser.add_argument(\"--seg_warmup\", type=int, default=1000)"),

    ("    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.sparse_num)",
     "    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, "
     "args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.sparse_num, "
     "Namespace(semantic_dir=args.semantic_dir, lambda_seg=args.lambda_seg, seg_warmup=args.seg_warmup))"),
])

print("S2_PATCH_DONE")
