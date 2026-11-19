# OSDaR23 数据准备管线（tools/data_prep）

将 OSDaR23 官方发布包转换为 SLAM 训练所需的 `RailwayDataset` 布局。管线设计与核验记录的
权威文档见 [`docs/data_prep_5.1.md`](../../docs/data_prep_5.1.md)（以下只给地图，不重复口径细节）。

## 管线总览（以 5.1 为例；1.2 同构）

```
官方 zip ──① 解包盘点──▶ 传感器原始目录
        ──② 投影口径──▶ 权威投影 proj_canon.py（唯一权威，勿改写公式）
        ──③ 数据集骨架─▶ frames/ + traj.txt + 官方语义投影 + info_semantic.json
        ──④ 深度三段式─▶ depths/*.npy + valid/*.png（UniDepth → LiDAR 锚定 → 可选缩放）
        ──⑤ 语义 L1~L4 ─▶ L1 官方几何 → L2 SAM2 精修 → L3 stuff 填充 → L4 仲裁合成
        ──⑥ L4 版数据集─▶ output_1.0_51_l4/（软链+实体）+ loader 自检
        ──⑦ 目检        ─▶ 六联图 / 100 帧网格
```

## 脚本地图

| 阶段 | 脚本 | 说明 |
|------|------|------|
| 下载 | `dl_osdar_seq.py` | 官方站点下载（含 anubis SHA-256 PoW 处理）；官方忽略 Range 头 ⇒ 单流整包 |
| 投影口径 | `proj_canon.py` | **唯一权威投影**（轴重排 `M=[[0,0,1],[-1,0,0],[0,-1,0]]`、`R=(quat2R(q)@M).T`；`project()` 不做 z 裁剪——调用方必须自滤 `z>0.3`） |
| 事实核验 | `fact_51.py` / `axis_selfcheck_51.py` / `verify_proj_51b.py` | 内参/外参/标注结构核验、轴向自检、投影三判据（地面自恒 / poly3d / bbox 对账） |
| 骨架 | `build_skel_51.py` | 数据集骨架；**里程基准用多静态物 x 递减量中位**（`mileage_ref_51.py`；单 buffer_stop 取法在 5.1 会断言失败） |
| 深度 | `make_depth_51.py` | 三段式：①UniDepthV2（`resolution_level=4`）②LiDAR 锚定 `s=median(z_lidar/d_mono)` 全局定标+命中替换（**写入前滤 `z>0.3` + 硬断言无负深度**）③乘数缩放默认 1.0 |
| 深度核验 | `verify_depth_51.py` | 三腿独立核验（轨面内容定律 / 地面反推相机高 / 直方图无 CAP 堆积）；5.1 深度尺度无偏（比值 1.0005） |
| 语义 L1 | `build_l1_51.py` | 官方几何掩膜；判据必须用 `len(uv)` 而非 `m.sum()`（投影内部已过滤——1.2 从未触发、5.1 即崩的两个 bug） |
| 语义 L2 | `l2_refine_51.py` | SAM2 两遍法精修（5.1：2449 对象，rescue 823 / 拦截 1626，类别层回归 0 px） |
| 语义 L3 | `l3_final_51.py` | stuff 五类填充（5.1 用 S0.50；1.2 是 S0.25；road/sidewalk/terrain→ground 等） |
| 语义 L4 | `l4_compose_51.py` | 四通道仲裁合成（semantic_id / sem_conf / instance_id / sem_route） |
| 数据集 | `build_l4_dataset_51.py` | L4 版训练数据集（frames/depths/traj 走软链，semantic_ids/colors 实体落盘） |
| 自检 | `loader_check_51.py` | **用 slam.py 自己的 `get_dataset` 验证**训练侧真正读到的东西（sid 逐像素一致率=1.0000 才算过） |
| 目检 | `preview_51.py` | 六联图 + 29 类图例 + 100 帧网格（诊断图必须分段着色，勿用连续色标） |
| 1.2 留档 | `make_depth_A.py` / `build_l4_dataset.py` / `l2_video_anchor_v4.py` | 1.2 时代管线（投影为旧口径 `make_depth_v41` 系，已被 proj_canon v2 取代，仅存档勿用于新序列） |

## 必须知道的坑（完整版见 docs/data_prep_5.1.md §9）

1. **官方两套相机外参冲突**：`labels.json`（3.5098 m + 6.49° 俯角，与图像自洽）✅ vs `calibration.txt` CAMERA 节（2.078 m）❌。一律以 `labels.json` 为准。
2. **通道序逐数据集**：1.2 旧图 BGR、5.1 为 RGB——按实测计数裁决，不要写死。
3. **L4 npy 是 uint16**（`I;16` 模式）：按原 dtype 存，勿 assert uint8。
4. **5.1 的 zip 无 calibration.txt**：标定只能从 `labels.json` 取。
5. **softlink 自检**：`os.listdir` 沿软链列子项恒为 0，要查 `os.path.islink(目录本身)`。
6. 深度 CAP=120 m 沿用 std 组口径；5.1 的 p99=72.6 m，实际未触发。
