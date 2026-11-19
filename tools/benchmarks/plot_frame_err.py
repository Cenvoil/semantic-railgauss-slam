#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""逐帧定位误差对比图（OSDaR23 1.2，四态消融）。

数据：benchmarks/results/error_profile_1.2/frame_errors.json
★ 数字来自实际执行（从 params.npz 的 cam_trans 与 gt_w2c_all_frames 直接相减，
  w2c 的 t 口径 + SE(3) 对齐），非手工编造。

设计要点（针对"让轨迹对比更清晰"）：
  1. **仅两条曲线**：原始方案（基线）vs 完整方案 —— 中间两个过渡配置删掉，
     因为它们都挤在 2~4 cm 带里，四条线互相遮挡反而看不清对比。
  2. **双对数纵轴**：两条线量级差 10 倍（23.45 vs 2.22 cm），线性轴下完整方案
     会被压成贴底的直线。主轴放完整方案（放大细节），右侧副轴放基线（保留量级对比）。
  3. **标注均值线**：两条均值各配一条水平虚线 + 右侧数值标签，一眼读出倍数关系。
  4. **突出倍率**：在图上直接标注 "10.6×"，比让人去比数字更直观。
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

SRC = "<PROJECT_ROOT>/benchmarks/results/error_profile_1.2/frame_errors.json"
OUT = "<PROJECT_ROOT>/benchmarks/results/error_profile_1.2/逐帧误差对比_1.2_双线.png"

# 配色（沿用原图风格：灰=基线，黑=完整方案）
C_BASE = "#8C8C8C"       # 原始方案：中性灰（弱化，作为"改造前"参照）
C_FULL = "#111111"       # 完整方案：黑（强调，作为本文方法）
C_GRID = "#D9D9D9"


def main():
    with open(SRC) as fh:
        D = json.load(fh)

    base = np.array(D["原始方案"])            # 23.45 cm
    full = np.array(D["完整方案"])            # 2.22 cm
    x = np.arange(len(base))

    # ---- 字体：★ 必须用 font_manager 直接注册字体文件，不能只靠 font.sans-serif名 ----
    #  坑：系统装了 Noto Sans CJK（.ttc），但 matplotlib 只把它识别成
    #  "Noto Sans CJK JP"（.ttc 的第一个 face），用 "Noto Sans CJK SC" 这个名字
    #  匹配不到 ⇒ 回退 DejaVu Sans ⇒ 中文全是方框。
    #  正解：addfont() 显式注册 ttc 文件，之后用注册出的名字。
    from matplotlib import font_manager
    CJK = None
    for pth in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
                "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"):
        if os.path.exists(pth):
            try:
                font_manager.fontManager.addfont(pth)
                CJK = font_manager.FontProperties(fname=pth).get_name()
                print("[font] %s -> %s" % (os.path.basename(pth), CJK))
                break
            except Exception as e:
                print("[font] %s 注册失败: %s" % (pth, e))
    plt.rcParams["font.sans-serif"] = ([CJK] if CJK else []) + ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(11.0, 6.2), dpi=160)

    # ---- 右副轴：基线（23 cm 量级）----
    ax2 = ax.twinx()
    ax2.plot(x, base, color=C_BASE, lw=2.6, ls="--", zorder=2,
             label="原始方案（基线）  均值 %.2f cm" % base.mean())
    m = base.mean()
    ax2.axhline(m, color=C_BASE, lw=1.0, ls=":", alpha=0.75, zorder=1)
    ax2.text(1.5, m + 1.4, "均值 %.2f cm" % m, color=C_BASE,
             fontsize=12.5, fontweight="bold", ha="left", va="bottom")
    ax2.set_ylim(0, max(base) * 1.12)
    ax2.set_ylabel("原始方案  定位误差 (cm)", fontsize=13, color=C_BASE)
    ax2.tick_params(axis="y", colors=C_BASE, labelsize=11)
    ax2.spines["right"].set_color(C_BASE)
    ax2.spines["right"].set_alpha(0.45)
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: "%g" % v))

    # ---- 主轴：完整方案（2 cm 量级，放大细节）----
    ax.plot(x, full, color=C_FULL, lw=2.8, zorder=4,
            label="完整方案（本文方法）  均值 %.2f cm" % full.mean())
    mf = full.mean()
    ax.axhline(mf, color=C_FULL, lw=1.0, ls=":", alpha=0.6, zorder=1)
    ax.text(1.5, mf + 0.16, "均值 %.2f cm" % mf, color=C_FULL,
            fontsize=12.5, fontweight="bold", ha="left", va="bottom")
    ax.set_ylim(0, max(full) * 1.45)
    ax.set_ylabel("完整方案  定位误差 (cm)", fontsize=13, color="#111111")
    ax.tick_params(axis="y", labelsize=11)

    # ---- 倍率标注（放在两线之间空白处）----
    ratio = base.mean() / full.mean()
    ax.text(0.40, 0.895,
            "平均定位误差降低 %.1f×" % ratio,
            transform=ax.transAxes, fontsize=14, fontweight="bold",
            color="#C00000", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.45", fc="#FFF5F5", ec="#C00000", lw=1.4,
                      alpha=0.95))

    # ---- 坐标轴与网格 ----
    ax.set_xlabel("帧序号", fontsize=13)
    ax.set_xlim(-2, len(base) + 1)
    ax.set_xticks([0, 20, 40, 60, 80, 99])
    ax.grid(True, ls="-", lw=0.8, color=C_GRID, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top",):
        ax.spines[s].set_visible(False)
    ax2.spines["top"].set_visible(False)

    # ---- 图例（两条线，用代理句柄统一到一个 legend 里）----
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    leg = ax.legend(h1 + h2, l1 + l2, loc="lower left", fontsize=12.5,
                    frameon=True, framealpha=0.96, edgecolor="#CCCCCC",
                    borderpad=0.8, handlelength=3.4)
    leg.get_frame().set_linewidth(1.0)

    # ---- 副标题：口径声明（必写，避免误读）----
    fig.text(0.5, 0.015,
             "OSDaR23 · 1_calibration_1.2 序列 · 100 帧 · 定位误差 = SE(3) 对齐后逐帧 "
             "w2c 平移误差（均值）　|　中间两个过渡配置（3.41 / 2.57 cm）因与完整方案同量级、"
             "互相遮挡而省略",
             ha="center", fontsize=9.5, color="#666666")

    fig.tight_layout(rect=[0, 0.035, 1, 1])
    fig.savefig(OUT, facecolor="white")
    print("[DONE] %s" % OUT)
    print("  原始方案 mean=%.2f  完整方案 mean=%.2f  倍率=%.2fx"
          % (base.mean(), full.mean(), ratio))


if __name__ == "__main__":
    main()