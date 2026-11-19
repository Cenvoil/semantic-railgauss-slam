# 投影口径（唯一权威）—— 2026-09-25 v2 定稿

> 状态：**冻结**。改动即解冻，须重跑验证判据并经 Gary 批准。
> 证据链与实验记录：`文档/分析与方案/自身问题审计_2026-09-24.md`「9-25 凌晨续审」F/G 节。
> 实现：服务器 `~/haichuan/tmp/d1/proj_canon.py`（md5 4b8fafe2b1b0fc27bdd65a031feb9896）。

## 1. 一句话

**官方 labels.json 的 `rgb_highres_center` 外参（离轨顶 3.51m + 下俯 6.49°）本来就与
图像内容自洽——v2 口径 = labels.json `coordinate_systems` 原值，零拟合参数。**
此前全部 r 曲线伪影的根源是：测量脚本错用了 **calibration.txt CAMERA 节的另一套外参**
（2.078m + 无俯角），那套与图像不符。官方数据没有错，是我们读错了文件。

## 2. 定稿常量与公式

```
内参（d 系列）：FX=FY=7267.95450880415  CX=2056.049238502414  CY=1232.862908875167
              （原生 1486.863 − 254 裁切；原生 4112×3008 → 发布 4112×2504）
畸变：k1,k2,k3 = -0.0764891, -0.188057, 1.78279（p1=p2=0，官方原值）
外参（labels.json coordinate_systems['rgb_highres_center'].pose_wrt_parent 原值，xyzw）：
    t = (0.0801578, -0.332862, 3.50982)
    q = (-0.00313306, 0.0562995, 0.00482918, 0.998397)   # 旋转 6.49° 绕 Y = 下俯

Pc = R^T (P - t) ； u0 = CX + FX·(-Pc_y)/Pc_x ； v0 = CY + FY·(-Pc_z)/Pc_x
k = 1 + k1·r2 + k2·r2² + k3·r2³ ； u = CX+(u0-CX)k ； v = CY+(v0-CY)k
轨顶平面内容定律（本模型自算）：v ≈ 414.8 + 25334/z
（VH=414.8 与 k_vis 近场切线消失点 414 精确重合）
```

## 3. 两套官方外参的对照（根因定位）

| 来源 | 平移 z（离轨顶） | 旋转 | 与图像内容 | 用途判定 |
|---|---|---|---|---|
| **labels.json coordinate_systems** | **3.510 m** | **6.49° 下俯** | **自洽**（poly3d↔poly2d 失配 5.3px） | **唯一权威** |
| calibration.txt CAMERA 节 | 2.078 m | 无俯角（Rc≈I） | 不符（同口径对账失配 277px） | **与图像对齐禁用** |

注：d 系列训练管线（yaml/数据集构建）从一开始就用的是 labels.json 外参 →
训练、位姿、已交指标全部本来就自洽，无需任何补救；错读 calibration.txt 的只有
9-21 起的各测量脚本（含 9-24 审计的「硬对账」——它对的是 calibration.txt 的转录，
没对图像内容）。

**差异归属（9-25 官方文档调研定稿，agent 全文核查 arXiv:2305.03001 / DZSF 报告 44 /
Labeling Guide / 官方 readme / GitHub raillabel issues，并逐字比对官方 1_calibration_1.1.zip）**：
1. **发布分辨率就是 4112×2504**（论文/ETR/readme/LabelingGuide/json/PNG 六处一致）；
   它是 Teledyne GenieNano C4040 传感器原生 4112×3008 的**官方中央裁切**
   （calibration.txt cy=1486.86 ↔ 3008 全幅，json cy=1232.86 ↔ 2504 发布图，恰差 254 铁证）。
   「图被额外裁切」的假设不成立。
2. **两文件的 RGB 外参整组互换**：calibration.txt 给 12MP 组（rgb_highres）配了
   2.078m/无俯角、给 5MP 组（rgb_center/left/right）配了 3.51m/5.8-6° 俯仰；json 恰好
   相反。ir 组两文件几乎一致（差 5mm）→ 参照系相同，1.43m 差异**不是口径差异，是数据矛盾**。
   判定为 calibration.txt 的 RGB 行**错位/陈旧**（官方未承认的缺陷）。
3. **第三方佐证**：GitHub raillabel issue #26 有用户用 json 外参（3.5098m/6.47° 俯仰）
   成功投影 rgb_highres_center（含截图）；issue #14 记录官方勘误先例（3D 框导出错误、
   lidar 伪内参，2023-10-23 同 DOI 修复）——但 RGB 外参矛盾无官方条目（未被发现）。
4. **官方投影三步法**（zip 内 readme）：lidar→非光学相机系（json 外参，前/左/上）→
   乘四元数 (0.5,−0.5,0.5,−0.5) → 光学系（右/下/前）→ 内参投影。本口径的轴序映射
   矩阵与其逆等价，已验证一致。
5. 官方证据文件已存本地：`文档/官方资料_osdar23/`（calibration_cal11.txt、
   readme_cal11.md、labels_cal11.json、dzsf_report44.txt、dzsf_labelguide.txt、
   etr_osdar23.txt、raillabel issues JSON 等）。

## 4. 独立确认证据（内容定律，非拟合来源）

| # | 方法 | 关键数字 | 依赖 G？ |
|---|---|---|---|
| 1 | poly2d 轨线 span 定律 | span(v)=0.438·(v−406)；反推高度 3.44m/俯角 6.49°（=json 外参 2% 内） | 是 |
| 2 | k_vis 近场切线消失点 | vp=414 ↔ 本口径轨面消失点 414.8 | **否** |
| 3 | 轨对宽度守恒 | calibration.txt 模型推 W=−21.9→2.24→1.51m（不可能）；json 口径恒 1.508m | 是 |
| 4 | 官方 cuboid 底面 vs bbox 底边 | 226-250m 缓冲挡 dv 22-31px（calib 口径 757px） | **否** |
| 5 | 深度图自身行结构 | 隐含相机高 3.46-3.54m（v41 生成器 HGT=3.369+俯角建模吻合） | **否** |

## 5. 最终对照表（取代全部旧 r 曲线）

`out/r_canon/r_table.json`（depth_vs_lidar_canon.py，20 帧 0,5,…,95，r=mean(z_lidar/z_input)，
深度=训练输入 d1_l4，rail=sem==13，proj_canon v2）：

| 段 | 10-15 | 15-20 | 20-25 | 30-35 | 45-60 | 60-90 | 90-120 |
|---|---|---|---|---|---|---|---|
| rail | n<20 | 1.027 | 1.023 | 1.026 | 1.026 | 1.004 | 0.791 |
| nonrail | 0.974 | 1.003 | 0.999 | 0.996 | 0.964 | 0.929 | 0.923 |

判读：**10-90m 深度与 LiDAR 差 ≤3%（rail）/≤4%（nonrail），无系统性压缩**；
90-120m 段 rail 0.79（深度偏远 ~27%，n=537）为遗留二阶偏差，候选原因（远轨线宽
1-2px 与深度图平滑、UniDepth 远场形状、s_f 锚区不含远场），留待归因，不改变结论。

## 6. 作废与替代清单

| 旧 | 状态 | 替代 |
|---|---|---|
| compress_curve.py / render_lidar_cmp.py / lidar_rail_split.py / lidar_r_final.py | 封存 `_sealed_r曲线_9-21to24/` | depth_vs_lidar_canon.py + §5 表 |
| 全部旧 r 曲线数字（0.64→3.43、1.44→4.86、0.96→4.31 等） | 作废（投影伪影） | §5 表 |
| 「输入深度 15m 起系统性压缩至 4.9×」结论 | 作废 | 深度源无偏（10-90m ≤4%） |
| 9-24 审计「定稿唯一权威投影公式」（calibration.txt，无俯角） | 作废 | 本文档 §2（labels.json 原值） |
| 9-25 凌晨内容定律拟合版 proj_canon v1（TZ=3.443/θ=6.49°） | 被 v2 取代（v1 失配 40.9px vs v2 5.3px） | 本文档 §2 |
| overlay_check.py 的「对齐成功」目检结论 | 作废（目检容差过宽） | verify1/2/5 定量对账 |

不受影响（明确）：k_vis 全部结论（纯图像几何，且消失点 414 与本口径精确互证）、
v41 深度图（不重做）、SGS-SLAM 训练与全部已交指标（训练外参本就是 labels.json，
完全自洽，旋转≡INS）、gk1045/gk1043 后校正、语义管线与交付。

## 7. 使用规则

1. 任何 LiDAR→图像、poly3d/cuboid→图像、深度 vs LiDAR 脚本：`from proj_canon import project`。
2. 新测量脚本另起名，不复活封存目录里的任何实现。
3. 引用 §5 数字时注明口径（proj_canon v2 + 20 帧 + r 定义）。
4. 引用官方外参时必须写明来源文件：**labels.json（用于图像对齐）≠ calibration.txt（禁用）**。
5. 大创叙事五条证据链（深度数据无辜）：§4 表全部 + §5 跨帧对照表。
