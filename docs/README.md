# docs — 冻结口径文档索引

> 本目录文档为项目**冻结口径库**的存档快照（2026-10-11 复制自项目内部口径库，
> 已去除服务器基础设施信息；内容除清洗外**逐字保留**）。口径冲突时以本目录为对外权威；
> 项目内部若口径演进，以内部冻结文件为准并同步更新本目录。

| 文件 | 冻结日期 | 内容 |
|------|---------|------|
| [method_definition.md](method_definition.md) | 2026-10-09 | ★ 「本方法」九项定义（A1~A9）、ATE 基线约束（KITTI 10cm 级 + 自查清单）、真值泄漏红线、指标台账规则 |
| [protocol_ATE_Sim3.md](protocol_ATE_Sim3.md) | 2026-09-21 | ATE 口径：SE(3)/Sim(3) 对齐、均值 vs RMSE、口径地基（eval_ate + regression_test_ate） |
| [protocol_projection.md](protocol_projection.md) | 2026-09-25 | 投影口径唯一权威（`tools/data_prep/proj_canon.py`），官方两套外参冲突的裁定 |
| [protocol_semantic_mask.md](protocol_semantic_mask.md) | 2026-09-18 | 语义掩码方案（L1 官方几何 / L2 SAM2 / L3 stuff / L4 仲裁） |
| [protocol_farfield_d_series.md](protocol_farfield_d_series.md) | 2026-09-21 | 远场阈值定义（d 系列） |
| [protocol_d_series.md](protocol_d_series.md) | 2026-09-18 | d 系列实验口径 |
| [baseline_freeze.md](baseline_freeze.md) | 2026-09-20 | 冻结基线 d8_full75 口径（改动即解冻） |
| [data_prep_5.1.md](data_prep_5.1.md) | 2026-10-04 | 5.1 深度/语义图构建全程：口径裁定、投影三判据、深度三段式与三腿核验、L1~L4、坑索引 |
| [upstream_README_SGS-SLAM.md](upstream_README_SGS-SLAM.md) | — | 上游 SGS-SLAM 原版 README（存档，含官方环境/引用信息） |

## 引用指标的纪律（摘要）

1. 数字只来自实际执行；外来数字必须注明出处；
2. ATE 引用必带口径标签（w2c-t / camcenter，SE(3)/Sim(3)，是否 INS 初始化）；
3. 跨数据集（OSDaR23 vs KITTI）数字不可直接相减（21 m 铁路直线 vs 千米级城市道路）；
4. 「本方法」= 九项全开；缺项只能称「基线 / 消融 / 官方 SGS-SLAM」（A7 缺失须标 `INS: 无`）。
