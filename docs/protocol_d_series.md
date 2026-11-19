# d 系列口径（2026-09-18）

> 原则：**官方 OSDaR23 数据绝对正确**；凡与官方冲突，先假设「我方有变量被改过或没考虑到」。
> 代码回到 **SGS-SLAM 本体**（`configs/replica/slam.py` 模板），q 系列的一切口径作废、不再沿用。

## 一、数据来源（官方 OSDaR23 · sequence #1 'calibration' part 1.2）
- 文件：`~/haichuan/data_osdar/1_calibration_1.2.zip`，MD5 `71b49b3422d138031a739a7260dd06f3`
- 已解压：`~/haichuan/data_osdar/unpacked/` → `rgb_highres_center` / `rgb_left` / `rgb_right` / `lidar` / `novatel_oem7_inspva`（各 100 帧）
- 标注：`1_calibration_1.2_labels.json`，OpenLABEL `file_version = 1.2.0`
- 相机选择：`rgb_highres_center`（4112×2504），与 SGS-SLAM 现行配置一致

## 二、相机标定（官方 intrinsics_pinhole 原值）
| 项 | 官方值 | q 系列 yaml | 差异 |
|---|---|---|---|
| fx / fy | 7267.95450880415 | 7267.95450880415 | 同 |
| cx | 2056.049238502414 | 2056.049238502414 | 同 |
| **cy** | **1232.862908875167** | **1486.862908875167** | **+254.000000000 px** |
| 分辨率 | 4112×2504 | 4112×2504 | 同 |
| 畸变 | k1..k5 = -0.0764891, -0.188057, 0, 0, 1.78279 | 未使用 | 本体为针孔，暂不建模 |

- 官方外参（相机 wrt base，base=LiDAR）：t=[0.0801578, -0.332862, 3.50982]，q=[-0.00313306, 0.0562995, 0.00482918, 0.998397]
- base 系约定：X 前 / Y 左 / Z 上（由官方 poly3d 轨道点 21→117m 在第一维证实）
- **cy 的 254px 差异是 d 系列头号待查项**：K 在 `basedataset.py` 会按降采样比例缩放，因此 yaml 必须写原生分辨率的官方值；q 系列的 cy 是人为改动还是历史遗留，必须查清后归零。
  ✅ 9-25 追记：cy=1232.86 与 labels.json 外参（含 6.49° 俯仰）配对自洽（poly3d↔poly2d 失配 5.3px）；cy 两口径在给定消失点下简并。外参以 labels.json 为准，**calibration.txt CAMERA 节（2.078m/无俯角）与图像不符，对图像对齐禁用**（见 `投影口径_唯一权威_2026-09-25.md`）。
- 四元数顺序 / 轴约定尚未用官方数据自证（poly3d↔poly2d 是两条独立标注、点序不一一对应，无法直接做残差检验），改用 LiDAR 投影目检来定。
  ✅ 9-25 追记：poly3d↔poly2d 直接对账已实现（密集化最近点），json 外参原值失配 5.3px，xyzw 约定与外参自证完成。

## 三、代码口径（SGS-SLAM 本体）
新建 `configs/railway/d_series_base.py`，由 `configs/replica/slam.py` 复制，**diff 仅 7 行**（场景名 / 路径 / 分辨率 / 类数 / end）：

| 参数 | 本体值 | q 系列值（作废） |
|---|---|---|
| map_every | 1 | 1 |
| keyframe_every | 5 | 2 |
| mapping_window_size | 24 | 24 |
| tracking_iters | 40 | 80 |
| mapping_iters | 60 | 600 |
| scene_radius_depth_ratio | 3 | 8 |
| use_gt_poses | False | True（INS 初始化 ins_init / forward_prop 改装） |
| prune 阈值 | 0.005 / stop_after 20 | 0.05 & 0.2 / stop_after 300 |
| densification | OFF | ON（start 20 / stop 240） |
| seg 权重 | track 0.05 / map 0.1 | track 0.05 / map 1.0 |
| 额外结构 | 无 | scale_px_min/max、add_depth_err_dict（P-35 far_seed / struct_seed / object_seed） |

- **环境变量全部保持默认**（不设即关闭）：`V4_GEO_MASK=0`、`V4_FAR_DEPTH=45`、`V4_DW_*`、`V4_SKY_*`、`V4_OVERDRAW_*`、`V4_POINT_BUDGET=0`、**`RAIL_PRIOR=0`（拟合线先验关闭）**
- 新建 `configs/data/railway_osdar_official.yaml`（官方 K，png_depth_scale=1.0 配 float32 npy 深度）

## 四、数据口径
- 序列目录：`data/output_1.0_v4np`（v4.1 伪 GT：float32 npy 深度 + 三值 valid png + 语义 ids + 官方帧）
- 深度：UniDepth V2 逐帧 LiDAR 中位定标（scale 1.75-2.07），锚定区直接填 LiDAR 真值，无 65m 截断，上限 120m
- 无效区策略：天空 + 移动物体（按 cuboid 轨迹判定）= 不监督；LiDAR 锚定区全类保留
- 语义：29 类（官方 OSDaR 类表），**目前只有类级语义，无 SAM 实例掩码**（见目检包第 4 张图说明）
- 位姿：现行 `traj.txt` 是 9-11 的自产物（非官方）；官方 INS 在 `novatel_oem7_inspva/*.csv`（经纬高 + roll/pitch/azimuth），需做 ENU 转换后使用 —— **待拍板**

## 五、评测口径
- d 系列只与 d 系列内部比较，**不与 q 系列任何数字对照**
- 深度误差必须在**同一像素集合**上比较（本轮已写 `eval_valid_v4.py`，输出 dl1_new / dl1_common / dl1_old 三口径）
- ATE 用官方 INS 轨迹（转换后）为基准，不用自产 traj.txt

## 六、Gary 拍板（2026-09-18 12:45）与执行结果
| # | 拍板 | 执行 | 硬证据 |
|---|---|---|---|
| ① | 只用官方 cy，旧 cy 不再沿用，矛盾单点排查 | 已建 `configs/data/railway_osdar_official.yaml`（cy=1232.862908875167） | 与 labels.json 官方 intrinsics_pinhole 逐字段相同 |
| ② | 原生分辨率 | `d_series_base.py` 改为 2504×4112 | 3 帧冒烟跑通无 OOM，峰值显存 **15.1 GiB**（卡上限 ~20.9），params 正常保存 |
| ③ | 位姿用官方 | `data/output_1.0_d1/traj.txt`（100 行 c2w，由 novatel INS 的经纬高+roll/pitch/azimuth 经 ENU 转换 + 官方相机安装外参合成） | 总里程 **21.16 m** / 每帧中位位移 0.221 m / 高差 −0.06 m（与官方匀加速直线场景一致） |
| ④ | 深度暂沿用 v4.1 | `output_1.0_d1/` 硬链接 v4.1 的 100 帧 depths + valid + semantic，只换 traj | 文件数 100/100/100/100/100 |

### 轴约定自证（本轮新结论）
- 官方四元数顺序 = **xyzw**（w 在最后）；wxyz 会使全部轨道点落到相机背后（有效深度 0/6556）。
- 相机 OpenCV 轴 → base(X 前 / Y 左 / Z 上) 需右乘置换 **M = [[0,0,1],[-1,0,0],[0,-1,0]]**。
- 判定依据（官方 poly2d 真值比对）：投影后轨道中位 u=2617 vs 官方 poly2d 中位 u=2740（Δu=−122 px）；镜像方案 M2 为 1495（Δu=−1245 px，判负）。
- **残留 Δv=+303 px**：嫌疑为官方强畸变未建模（k1=−0.076, k2=−0.188, **k3=1.78**）。列为语义/深度重建的前置阻塞项（见《语义掩码方案》）。

## 七、遗留待查（1 项）
- 残留 Δv=303px 的来源：官方畸变建模（undistort）或俯仰符号约定 —— 必须在用官方投影生成深度/语义掩膜前解决。
