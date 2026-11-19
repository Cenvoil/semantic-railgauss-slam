# NOTICE — 归属与第三方组件

## 上游项目（本仓库的修改基础）

本仓库是 **[SGS-SLAM](https://github.com/ShuhongLL/SGS-SLAM)**（ECCV 2024）的衍生作品：

> Mingrui Li*, Shuhong Liu*, Heng Zhou, Guohao Zhu, Na Cheng, Tianchen Deng, Hongyu Wang.
> "SGS-SLAM: Semantic Gaussian Splatting For Neural Dense SLAM." ECCV 2024. arXiv:2402.03246

- 上游采用 **BSD 3-Clause License**（见 [`LICENSE`](LICENSE)）。
- 本仓库保留上游版权声明；对上游的修改（场景适配开关层、深度管线接口、评测口径工具等）
  由本项目按同一许可证发布。
- 上游原版 README 存档于 [`docs/upstream_README_SGS-SLAM.md`](docs/upstream_README_SGS-SLAM.md)。
- 主干未改的证据：`loss_weights`（tracking im0.5 / depth1.0 / seg0.05）与上游 Replica 配置逐位一致。

## 第三方组件

| 组件 | 用途 | 许可/来源 |
|------|------|-----------|
| [UniDepthV2](https://github.com/lpiccinelli-eth/UniDepth) | 单目稠密深度估计（权重 `lpiccinelli/unidepth-v2-vitl14`） | 其官方仓库许可；权重遵循其发布条款 |
| [SegFormer](https://github.com/NVlabs/SegFormer) | 语义填充推理 | MIT（NVlabs） |
| [SAM 2](https://github.com/facebookresearch/sam2) | 实例掩膜精修 | Apache-2.0（Meta） |
| [KISS-ICP](https://github.com/PRBonn/kiss-icp) | LiDAR 里程计基线对照 | MIT（PRBonn） |
| diff-gaussian-rasterization-w-depth | 高斯光栅化（带深度） | [JonathonLuiten fork](https://github.com/JonathonLuiten/diff-gaussian-rasterization-w-depth)（见 requirements.txt 固定 commit） |
| [gradslam](https://github.com/gradslam/gradslam) 数据加载器骨架 | `datasets/gradslam_datasets/` 基类 | MIT（上游随 SGS-SLAM 分发） |

## 数据集（不随本仓库分发）

- **OSDaR23 — Open Sensor Data for Rail 2023**（DZSF / Deutsches Zentrum für Schienenverkehrsforschung，
  [data.fid-move.de](https://data.fid-move.de)，arXiv:2305.03001）。使用须遵循官方发布条款。
- 本仓库代码在数据集之外的部分不受数据许可约束。
