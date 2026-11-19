> **Paper release.** This repository accompanies *SemanticRailGauss-SLAM: A Rail-Specific Semantic Gaussian Splatting SLAM System with LiDAR-Anchored Monocular Depth* by Hai-Chuan Gan and Dan Tao (Beijing Jiaotong University). See `paper/` for paper-specific add-ons and results.
>
> **Numbers.** The paper-aligned numbers are in `paper/results/`; the L7 table below is an earlier development configuration and must not be compared directly with the paper. See `PAPER.md`.
>
> Paper main results (v11, Horn SE(3), w2c-t, mean): **1.2 = 1.84 cm; 3.1 = 5.45 cm; 5.1 = 2.94 cm**; no-INS: **1.2 = 4.27 cm, 5.1 = 2280.88 cm**. The 2×2 factorial gives 23.45 / 3.41 / 2.57 / 2.22 cm.

# RailGauss-OSDaR23 — 铁路场景语义高斯 SLAM（OSDaR23 数据集专用）

**RailGauss-SLAM**（铁路语义高斯 SLAM）的 **OSDaR23 数据集专用代码库**。
基于 [SGS-SLAM (ECCV 2024)](https://github.com/ShuhongLL/SGS-SLAM) 改造，面向铁路室外、长大距离、稀疏量测场景，
针对上游（纯室内 RGB-D）在铁路数据上的失效环节提出系统性适配。

> **English.** RailGauss-SLAM is a semantic Gaussian-splatting SLAM for railway scenes, built on top of
> [SGS-SLAM (ECCV 2024)](https://github.com/ShuhongLL/SGS-SLAM) (BSD-3-Clause). This repository is the
> **OSDaR23-specific** branch of the project: it contains the modified SLAM core (scene-adaptation switch layer,
> monocular depth + LiDAR scale anchoring, SAM-level semantic masks + SegFormer filling, INS initialization,
> rail prior, sky mask), the OSDaR23 dataset loaders/configs, the full data-preparation pipeline, and the frozen
> evaluation protocols (ATE / projection / semantic-mask). See `docs/` for the authoritative protocol documents
> (Chinese). Results below are reported strictly under the protocols defined in `docs/method_definition.md`.

---

## 1. 「本方法」的定义（九项，缺一不可）

权威文件：[`docs/method_definition.md`](docs/method_definition.md)（含逐项验收标准与例外条款）。

| # | 组成项 | 一句话说明 |
|---|--------|-----------|
| A1 | SGS-SLAM 主干 | 官方 ECCV 2024 多通道（im/depth/seg）优化框架，损失权重与官方逐位一致 |
| A2 | V4 系列场景适配开关 | `scripts/slam.py` 内 30 个 `V4_*` 环境变量（上游 `os.environ` 计数 = 0） |
| A3 | RGB 单目深度估计 | UniDepthV2 单目推理（非 LiDAR 投影） |
| A4 | LiDAR 尺度锚定 | 用 LiDAR 对单目深度全局定标 + 命中像素替换（holdout absrel **0.4621 → 0.0293**，降 16×） |
| A5 | 标注 SAM 级掩膜 | 官方 2D/cuboid 标注升级为实例级掩膜（SAM2 精修） |
| A6 | SegFormer 填充推理 | 掩膜外区域语义填充（stuff 类五通道 + L4 仲裁合成） |
| A7 | INS 初始化 | 用真实 INS/GNSS 位姿逐帧初始化（**无 INS 数据集可缺 A7，须标注 `INS: 无`**） |
| A8 | 轨道先验 | 轨道几何先验（只罚垂距；直线序列必然中性） |
| A9 | 天空掩码 | 天空区域排除，防深度/语义异常污染 |

凡不满足全部条件者，不得称「本方法」，只能称「基线」「消融配置 X」或「官方 SGS-SLAM」。

## 2. 相对上游 SGS-SLAM 的改动

上游是纯室内 RGB-D 系统（Replica/ScanNet/ScanNet++，官方数据集白名单仅 11 个室内集，论文零提及户外/铁路），
在 OSDaR23 铁路数据上直接运行会因以下缺口失效。本仓库的改造（= 本项目贡献）：

1. **场景适配开关层（A2）**：`V4_*` 环境变量组 —— 天空(7)/几何掩膜(4)/深度门控(4)/前表面(2)/远场正则(4)/初始化像素(4)/各向异性等；
2. **单目深度 + LiDAR 尺度锚定（A3/A4）**：铁路场景无稠密深度传感器，UniDepthV2 推理 + LiDAR 全局定标；
3. **语义管线（A5/A6）**：官方稀疏标注（1.2 仅 11~13% 像素覆盖）→ L1 官方几何 → L2 SAM2 精修 → L3 stuff 填充 → L4 四通道仲裁合成；
4. **INS 初始化（A7）**：1.2/5.1 的真实 INS/GNSS 序列作为合法先验（ATE 2280.88 cm → 3.38 cm 的关键）；
5. **轨道先验与天空掩码（A8/A9）**。

主干未改：`loss_weights`（tracking `im0.5 / depth1.0 / seg0.05`）与官方 Replica 配置**逐位一致**。

## 3. 数据集

[OSDaR23 — Open Sensor Data for Rail 2023](https://data.fid-move.de)（DZSF / 德铁联合研究项目，[arXiv:2305.03001](https://arxiv.org/abs/2305.03001)）：
多传感器铁路开放数据集（RGB / LiDAR / RADAR / INS 等）。

本仓库使用其中两个 100 帧序列：

| 序列 | 名称 | 里程 | 每帧位移 | 特点 |
|------|------|------|---------|------|
| **1.2** | `1_calibration_1.2` | 21.16 m | 0.221 m | 标定场，近直线（直线性 1.0000） |
| **5.1** | `5_station_bergedorf_5.1` | 101.47 m | 1.023 m | 车站场景，标注 88 实例/7 类（1.2 为 28 实例/17 类） |

数据集获取、下载（官方站点 anubis 挑战处理脚本 `tools/data_prep/dl_osdar_seq.py`）、解包与已知坑
（**官方两套相机外参冲突时以 `labels.json` 为准**、5.1 无 `calibration.txt` 等）见
[`docs/data_prep_5.1.md`](docs/data_prep_5.1.md) §1 与 [`tools/data_prep/README.md`](tools/data_prep/README.md)。

**数据布局**（`RailwayDataset` 接口，见 `datasets/gradslam_datasets/railway.py`）：

```
data/<sequence>/
├── frames/frame%06d.png          # RGB
├── depths/depth%06d.npy          # float32 米（亦兼容 .png 16bit）
├── valid/valid%06d.png           # 0=无效 / 1=模型 / 2=LiDAR 锚
├── semantic_ids/semantic_id*.png # 类 id（×11 编码）
├── semantic_colors/semantic_color*.png
├── traj.txt                      # 3×4 行主序 c2w（真实 INS/GNSS 序列）
└── info_semantic.json            # 官方 29 类色表
```

## 4. 结果摘要（口径必读）

> **口径规则**（详见 [`docs/method_definition.md`](docs/method_definition.md) §4）：
> ① ATE 一律 w2c 平移 `t` 口径 + SE(3) 对齐 + 均值（引用必带口径标签）；
> ② 数字只来自实际执行，跨口径比较须同集合+同协议；
> ③ OSDaR23 与 KITTI（城市道路，长 176×）数字**不可直接相减**；
> ④ 本方法 = 九项全开（1.2/5.1 含真实 INS 先验，A7 合法）。

| 序列 | 配置 | ATE (w2c-t, SE(3)) | PSNR | Depth L1 | mIoU | 备注 |
|------|------|--------------------|------|----------|------|------|
| 1.2 | **L7** = `d8_gk1045_insinit`（本方法） | **2.22 / 2.16 cm**（两跑） | 21.45 | 0.815 m | 0.6632 | INS 先验 + 深度×1.0448 尺度校正 |
| 1.2 | 冻结基线 `d8_full75`（三跑中位） | 23.45 cm | 21.70 | 0.816 m | 0.6669 | 无 INS 初始化 |
| 5.1 | `d8_51_insinit`（本方法） | **3.38 cm** | 16.45 | 2.046 m | 0.2907 | INS 先验；深度尺度实测无偏（相机高 3.5117 vs 官方 3.5098 m） |
| 5.1 | `d8_51_l4`（无 INS） | 2280.88 cm | 13.42 | 4.189 m | 0.1676 | 基线崩溃 → INS 初始化价值的直接证据 |
| 1.2 | KISS-ICP（对照，相机中心口径） | 7.08 cm | — | — | — | 仅轨迹；无渲染/语义能力（PSNR/mIoU 无定义） |
| 5.1 | KISS-ICP（对照，相机中心口径） | 7.88 cm | — | — | — | 同上 |

- 深度管线质量：UniDepth + LiDAR 锚定 holdout absrel 中位 **0.0293（1.2）/ 0.0623（5.1）**；
  5.1 深度尺度无偏（地面像素反推相机高比值 1.0005）。
- 1.2 两序列真值近完美直线 ⇒ SE(3) 旋转自由度几乎不可观测，端点最易漂；逐段误差形态见
  `tools/benchmarks/error_profile` 流程与 [`docs/baseline_freeze.md`](docs/baseline_freeze.md)。
- 复跑须知：PSNR 噪声底 0.11 dB ⇒ 结论须 ≥3 跑中位；评测前 `tools/eval/regression_test_ate.py` 必须 PASS。

## 5. 仓库结构

```
railgauss-osdar23/
├── scripts/                  # SLAM 核心（slam.py 含 30 个 V4_* 开关；fork 自 SGS-SLAM）
├── utils/                    # 评测/高斯/重建辅助（fork）
├── datasets/gradslam_datasets/  # 数据集加载器（railway.py = OSDaR23 铁路加载器）
├── viz_scripts/              # 在线/离线重建可视化
├── configs/
│   ├── railway/              # d 系列（冻结研究线）：d8_full75 基线、d8_gk1045_insinit=L7、消融 d8_bl_* 等
│   └── data/                 # railway_*.yaml / osdar23_*.yaml（内参与分辨率）
├── tools/
│   ├── eval/                 # eval_ate.py（指标唯一出口）+ regression_test_ate.py（每次必跑）
│   ├── data_prep/            # OSDaR23 官方数据 → 训练集 全流程（下载/投影/深度/语义 L1~L4）
│   └── benchmarks/           # KISS-ICP 对照、真值导出、LiDAR 投影标定、帧对比图
└── docs/                     # 冻结口径文档（方法定义/ATE/投影/语义掩码/远场/基线/数据构建）
```

## 6. 环境

```bash
conda env create -f environment.yml     # python 3.10, torch 1.12.1, cudatoolkit 11.6
conda activate sgs-slam
pip install -r requirements.txt         # 含 depth-aware diff-gaussian-rasterization（JonathonLuiten fork）
```

- 关键依赖：`diff_gaussian_rasterization`（带深度版本，源码编译，需 CUDA）。
- 深度预处理另需独立环境运行 [UniDepthV2](https://github.com/lpiccinelli-eth/UniDepth)
  （权重 `lpiccinelli/unidepth-v2-vitl14`，推理时须 cwd 仓库根）。
- 显存：`mapping_window_size=24` 满载约 22.6 GB（RTX 3090 24G 级别）。

## 7. 快速开始

```bash
# ① 数据准备（官方 zip → 训练集布局）：见 tools/data_prep/README.md（下载→投影→深度→语义 L1~L4）

# ② 训练（1.2 冻结基线）
python scripts/slam.py configs/railway/d8_full75.py

# ③ 训练（本方法 = L7：LiDAR 尺度校正 + INS 初始化）
python scripts/slam.py configs/railway/d8_gk1045_insinit.py

# ④ 评测（渲染 + 指标）
python scripts/eval_novel_view.py configs/railway/d8_full75.py

# ⑤ ATE（口径唯一出口；先过回归自检）
python tools/eval/regression_test_ate.py
python tools/eval/eval_ate.py --gt gt.tum --est est.tum --repr w2c_t --method ours --dataset OSDaR23 --sequence 1.2
```

主要 `V4_*` 开关（天空门控示例）：`V4_SKY_GATE=1 V4_SKY_ID=<sky类id>`；其余开关语义见 `scripts/slam.py` 源内注释。

## 8. 文档索引（docs/）

| 文件 | 内容 |
|------|------|
| [`method_definition.md`](docs/method_definition.md) | ★ 「本方法」九项定义 + ATE 基线约束 + 指标台账规则（对外口径唯一权威） |
| [`protocol_ATE_Sim3.md`](docs/protocol_ATE_Sim3.md) | ATE 口径（SE(3)/Sim(3)） |
| [`protocol_projection.md`](docs/protocol_projection.md) | 投影口径（唯一权威，`tools/data_prep/proj_canon.py`） |
| [`protocol_semantic_mask.md`](docs/protocol_semantic_mask.md) | 语义掩码方案 |
| [`protocol_farfield_d_series.md`](docs/protocol_farfield_d_series.md) | 远场阈值定义 |
| [`protocol_d_series.md`](docs/protocol_d_series.md) | d 系列实验口径 |
| [`baseline_freeze.md`](docs/baseline_freeze.md) | 冻结基线 d8_full75 口径 |
| [`data_prep_5.1.md`](docs/data_prep_5.1.md) | 5.1 深度/语义构建全程记录（口径、坑、核验三腿） |
| [`upstream_README_SGS-SLAM.md`](docs/upstream_README_SGS-SLAM.md) | 上游 SGS-SLAM 原版 README（存档） |

## 9. 引用

如使用本仓库，请同时引用上游 SGS-SLAM 与 OSDaR23 数据集：

```bibtex
@inproceedings{li2024sgs,
  title     = {SGS-SLAM: Semantic Gaussian Splatting For Neural Dense SLAM},
  author    = {Li, Mingrui and Liu, Shuhong and Zhou, Heng and Zhu, Guohao and Cheng, Na and Deng, Tianchen and Wang, Hongyu},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2024}
}
@inproceedings{tagiew2023osdar23,
  title     = {{OSDaR23}: Open Sensor Data for Rail 2023},
  author    = {Tagiew, Rustam and Klasek, Pavel and Tilly, Roman and Schwalbe, Karsten and others},
  booktitle = {2023 8th International Conference on Robotics and Automation Engineering (ICRAE)},
  pages     = {270--276},
  year      = {2023},
  note      = {arXiv:2305.03001}
}
```

其他依赖方法：UniDepth（Piccinelli et al., CVPR 2024）、SegFormer（Xie et al., NeurIPS 2021）、
SAM 2（Ravi et al., 2024）、KISS-ICP（Vizzo et al., RA-L 2023）。

## 10. 许可证

- 代码：BSD-3-Clause（继承上游 SGS-SLAM，见 [`LICENSE`](LICENSE)）；改动部分同样以 BSD-3-Clause 发布。
- 上游与第三方署名见 [`NOTICE.md`](NOTICE.md)。
- OSDaR23 数据集本身**不随本仓库分发**，使用须遵循 DZSF/德铁官方许可条款。

---

*本仓库由 RailGauss-SLAM 项目组维护（大创项目·铁路语义高斯 SLAM）。私有阶段文档与口径以 docs/ 为准。*
