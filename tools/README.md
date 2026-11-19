# tools — 评测与对照（tools/eval, tools/benchmarks）

## tools/eval — 指标唯一出口

| 脚本 | 作用 |
|------|------|
| `regression_test_ate.py` | 口径地基自检（合成数据 Sim3/SE3 残差 1e-15 级 + 与项目 evaluate_ate 差 0.0000 cm）。**每次评测前必跑，PASS 才采信新数字** |
| `eval_ate.py` | ATE 计算（`--repr w2c_t|camcenter` 双口径；SE(3)/Sim(3)；均值口径）|

口径文档：[`docs/protocol_ATE_Sim3.md`](../docs/protocol_ATE_Sim3.md)、
[`docs/method_definition.md`](../docs/method_definition.md) §4（台账规则）。

## tools/benchmarks — 横向对照与可视化

| 脚本 | 作用 |
|------|------|
| `export_gt_from_npz.py` | 从 params.npz 导出真值 t / C（带往返自检，往返误差 <1e-6） |
| `convert_traj_repr.py` / `compare_unified.py` / `compare_x.py` | 轨迹表示换算与对照（**每个方法跟自己那份真值在同一约定下比，不做坐标换算**——w2c-t 与相机中心混用是本项目最大的口径坑，见 benchmarks 记录） |
| `calib_proj.py` / `calib_z_off.py` / `diag_depth_err.py` | LiDAR 投影标定（pitch=1.0°、zoff=−1.50 ⇒ 深度比值中位 1.002；只标 2 自由度的遗留限制见 docs） |
| `lidar_ref_depth.py` | 官方 LiDAR 投影深度参照（独立物理测量，**对照必须用 LiDAR 不用官方渲染**——渲染图是被评估对象，循环论证） |
| `render_frames.py` / `frame_compare_12.py` / `verify_render.py` | 帧渲染对比与官方渲染对齐验证（逐像素差 ≈0.9% 判通过） |
| `plot_frame_err*.py` / `export_frame_err.py` | 逐帧/逐段误差形态（真值近完美直线 ⇒ SE(3) 旋转不可观测，端点最易漂） |
| `install_deps.sh` | 依赖安装 |

**对照表纪律**：KISS-ICP 等外部方法不产出 PSNR/mIoU（无渲染分支）⇒ 填「—」；
不同口径的数字分块呈现，`comparable_with_project: false` 标注不可相减。
