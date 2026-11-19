#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""逐帧定位误差对比图（OSDaR23 1.2，四态消融，横轴 = 真值累计里程）。

数据：benchmarks/results/error_profile_1.2/frame_errors.json
★ 数字来自实际执行（params.npz 的 cam_trans 与 gt_w2c_all_frames 直接相减，
  w2c 的 t 口径 + SE(3) 对齐），非手工编造。里程 = 真值逐帧位移累加，总长 21.16 m。

设计：
  - 四态全保留，各用不同颜色；完整方案线宽加粗一圈（外层浅色描边）以突出。
  - 线性纵轴（四态同量级可同轴比较；23.45 的峰值仍可容纳）。
  - 除图例与坐标轴标注外，不放任何其它文案（倍率标注/副标题/说明框全部去掉）。
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SRC = "<PROJECT_ROOT>/benchmarks/results/error_profile_1.2/frame_errors.json"
OUT = "<PROJECT_ROOT>/benchmarks/results/error_profile_1.2/逐帧误差对比_1.2_里程轴_四态.png"

# (key, 显示名, 颜色, 线宽, 线型, zorder)
SERIES = [
    ("原始方案",        "原始方案",         "#9E9E9E", 1.7, "--",  2),
    ("+场景尺度校正",   "+场景尺度校正",    "#1F77B4", 1.9, "-",   3),
    ("+INS位姿初始化",  "+INS位姿初始化",   "#E8722C", 1.9, "-",   4),
    ("完整方案",        "完整方案",         "#000000", 2.9, "-",   6),
]


def register_cjk():
    """★ 服务器 matplotlib 只把 NotoSansCJK-Regular.ttc 识别成 'Noto Sans CJK JP'，
    用 'Noto Sans CJK SC' 匹配不到 ⇒ 中文变方框。正解：addfont 显式注册。"""
    from matplotlib import font_manager
    for pth in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"):
        if os.path.exists(pth):
            try:
                font_manager.fontManager.addfont(pth)
                name = font_manager.FontProperties(fname=pth).get_name()
                plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
                print("[font] %s" % name)
                return
            except Exception as e:
                print("[font] 失败 %s: %s" % (pth, e))
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]


def main():
    register_cjk()
    plt.rcParams["axes.unicode_minus"] = False

    with open(SRC) as fh:
        D = json.load(fh)
    mi = np.array(D["_mileage"])          # 真值累计里程（m）

    fig, ax = plt.subplots(figsize=(11.0, 6.0), dpi=160)

    ymax = 0
    for key, label, color, lw, ls, z in SERIES:
        y = np.array(D[key])
        ymax = max(ymax, y.max())
        # 完整方案：先画一条更粗的浅色底线做"描边"，再画主线 ⇒ 视觉上粗一圈
        if key == "完整方案":
            ax.plot(mi, y, color="#BDBDBD", lw=lw + 2.6, ls=ls, zorder=z - 1,
                    solid_capstyle="round")
        ax.plot(mi, y, color=color, lw=lw, ls=ls, zorder=z,
                label="%s  均值 %.2f cm" % (label, y.mean()),
                solid_capstyle="round")

    ax.set_xlabel("真值累计里程 (m)", fontsize=14)
    ax.set_ylabel("定位误差 (cm)", fontsize=14)
    ax.set_xlim(0, mi[-1])
    ax.set_ylim(0, ymax * 1.10)
    ax.set_xticks(np.arange(0, mi[-1] + 0.1, 2.0))
    ax.tick_params(labelsize=12)
    ax.grid(True, ls="-", lw=0.8, color="#DCDCDC", alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    leg = ax.legend(loc="upper left", fontsize=12.5, frameon=True,
                    framealpha=0.96, edgecolor="#CCCCCC", borderpad=0.8,
                    handlelength=3.2)
    leg.get_frame().set_linewidth(1.0)

    fig.tight_layout()
    fig.savefig(OUT, facecolor="white")
    print("[DONE] %s" % OUT)
    print("  里程总长 %.2f m  ymax %.2f cm" % (mi[-1], ymax))


if __name__ == "__main__":
    main()