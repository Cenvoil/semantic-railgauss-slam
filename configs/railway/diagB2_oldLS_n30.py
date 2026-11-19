import os
from os.path import join as p_join

scenes = ["room0", "room1", "room2", "office0", "office1", "office2",
          "office3", "office4", "apartment_0", "apartment_1",
          "apartment_2", "frl_apartment_0", "frl_apartment_4"]

primary_device="cuda:0"
seed = 0
scene_name = "output_1.0_v6dilLS"

map_every = 1
keyframe_every = 2
mapping_window_size = 24
tracking_iters = 80
mapping_iters = 600 # P-65+P-66 组合：点数受限后给足优化时间

group_name = "Railway_diag"
run_name = "diagB2_oldLS_n30"

config = dict(
    workdir=f"./experiments/{group_name}",
    run_name=run_name,
    seed=seed,
    primary_device=primary_device,
    map_every=map_every, # Mapping every nth frame
    keyframe_every=keyframe_every, # Keyframe every nth frame
    mapping_window_size=mapping_window_size, # Mapping window size
    report_global_progress_every=500, # Report Global Progress every nth frame
    eval_every=5, # Evaluate every nth frame (at end of SLAM)
    scene_radius_depth_ratio=8, # Max First Frame Depth to Scene Radius Ratio (For Pruning/Densification)
    mean_sq_dist_method="projective", # ["projective", "knn"] (Type of Mean Squared Distance Calculation for Scale of Gaussians)
    report_iter_progress=False,
    load_checkpoint=False,
    checkpoint_time_idx=0,
    save_checkpoints=True, # Save Checkpoints
    checkpoint_interval=500, # Checkpoint Interval
    save_timestamp_keyframes=False,
    use_wandb=False,
    wandb=dict(
        entity="my-project", # Please change the entity name
        project="SGS-SLAM",
        group=group_name,
        name=run_name,
        save_qual=False,
        eval_save_qual=True,
    ),
    data=dict(
        basedir="./data",
        gradslam_data_cfg="./configs/data/railway_fused_v5.yaml",
        sequence=scene_name,
        desired_image_height=976,
        desired_image_width=1600,
        start=0,
        end=-1,
        stride=1,
        num_frames=30, # Set to -1 to use all frames
        load_semantics=True,
        num_semantic_classes=29
    ),
    tracking=dict(
        use_gt_poses=True, # Use GT Poses for Tracking
        ins_init=True, # v4 F1: init each frame from INS pose, then refine
        forward_prop=True, # Forward Propagate Poses
        num_iters=tracking_iters,
        use_sil_for_loss=True,
        sil_thres=0.99,
        use_l1=True,
        ignore_outlier_depth_loss=False,
        loss_weights=dict(
            im=0.5,
            depth=1.0,
            seg=0.05,
        ),
        lrs=dict(
            means3D=0.0,
            rgb_colors=0.0,
            unnorm_rotations=0.0,
            logit_opacities=0.0,
            log_scales=0.0,
            cam_unnorm_rots=0.0004,
            cam_trans=0.002,
            semantic_colors=0.0,
        ),
    ),
    mapping=dict(
        num_iters=mapping_iters,
        scale_px_min=1.0, # P-46 尺度下限=1.5像素(随距离)
        scale_px_max=3.0, # P-46 尺度上限=5像素 尺度下限(米)
        add_new_gaussians=True,
        sil_thres=0.5, # v6: 帷幕已由F2拆除，回退防走廊重复实例化（P-26诊断）
        use_l1=True,
        use_sil_for_loss=False,
        ignore_outlier_depth_loss=False,
        loss_weights=dict(
            im=0.5,
            depth=1.0,
            seg=1.0
        ),
        lrs=dict(
            means3D=0.0001,
            rgb_colors=0.0025,
            unnorm_rotations=0.001,
            logit_opacities=0.05,
            log_scales=0.001,
            cam_unnorm_rots=0.0000,
            cam_trans=0.0000,
            semantic_colors=0.0025,
        ),
        prune_gaussians=True, # Prune Gaussians during Mapping
        pruning_dict=dict( # Needs to be updated based on the number of mapping iterations
            start_after=0,
            remove_big_after=0,
            stop_after=300, # v6: prune 全周期: prune whole mapping cycle
            prune_every=20,
            removal_opacity_threshold=0.05, # v5 F4
            final_removal_opacity_threshold=0.2, # v5c: 末期清扫半透明漂浮物
            reset_opacities=False,
            reset_opacities_every=500, # Doesn't consider iter 0
        ),
        add_depth_err_dict=dict( # P-35: 1.2 同样只在头 17 帧长过地图
            enable=True,
            rel_thresh=0.15,
            abs_thresh=1.5,
            max_per_frame=25000,
            far_seed_enable=True,
            far_seed_depth=80.0,
            far_seed_max_per_frame=10000,
            struct_seed_enable=True,
            struct_seed_classes=[13,15,25,27],
            struct_seed_depth_max=120.0,
            struct_seed_max_per_frame=8000,
            struct_seed_object_classes=[8,16,17,18,19,20],
            object_seed_max_per_frame=20000,
        ),
        use_gaussian_splatting_densification=True, # v5 F4 # Use Gaussian Splatting-based Densification during Mapping
        densify_dict=dict( # Needs to be updated based on the number of mapping iterations
            start_after=20, # v5 F4
            remove_big_after=100, # v5 F4
            stop_after=240, # v6: densify 留60轮收敛
            densify_every=60, # v6
            grad_thresh=0.0002,
            num_to_split_into=2,
            removal_opacity_threshold=0.05, # v5 F4
            final_removal_opacity_threshold=0.2, # v5c
            reset_opacities_every=3000, # Doesn't consider iter 0
        ),
    ),
    viz=dict(
        render_mode='color', # ['color', 'depth', 'centers', 'semantic_color']
        offset_first_viz_cam=True, # Offsets the view camera back by 0.5 units along the view direction (For Final Recon Viz)
        show_sil=False, # Show Silhouette instead of RGB
        visualize_cams=False, # Visualize Camera Frustums and Trajectory
        viz_w=600, viz_h=340,
        viz_near=0.01, viz_far=100.0,
        view_scale=2,
        viz_fps=5, # FPS for Online Recon Viz
        enter_interactive_post_online=True, # Enter Interactive Mode after Online Recon Viz
        scene_name=scene_name,
        load_semantics=True, # Whether load semantic information
    ),
)