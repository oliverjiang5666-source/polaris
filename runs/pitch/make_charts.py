"""生成山东业绩报告用图表 (PNG)"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.font_manager as fm

OUT = Path(__file__).parent / "assets"
OUT.mkdir(exist_ok=True, parents=True)

# 注册系统中文字体
CN_FONT_CANDIDATES = [
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/System/Library/AssetsV2/com_apple_MobileAsset_Font8/f14049099a04e570b893c01d9a4cd71f87c9e8d8.asset/AssetData/Lantinghei.ttc",
]
for fp in CN_FONT_CANDIDATES:
    try:
        fm.fontManager.addfont(fp)
    except Exception:
        pass

# 全局样式
rcParams["font.family"] = ["STHeiti", "Lantinghei SC", "Heiti TC", "Arial Unicode MS", "sans-serif"]
rcParams["font.size"] = 11
rcParams["axes.spines.top"] = False
rcParams["axes.spines.right"] = False
rcParams["axes.edgecolor"] = "#cbd5e1"
rcParams["axes.labelcolor"] = "#475569"
rcParams["xtick.color"] = "#64748b"
rcParams["ytick.color"] = "#64748b"
rcParams["grid.color"] = "#e2e8f0"
rcParams["grid.linewidth"] = 0.8

NAVY   = "#1e3a8a"
TEAL   = "#0891b2"
GREEN  = "#059669"
AMBER  = "#d97706"
GRAY   = "#94a3b8"
LIGHT  = "#e0e7ff"

# ============================================================
# Figure 1: 算法演进 bar chart
# ============================================================

algs = ["LightGBM\n+ LP", "MILP\n(工业标准)", "MILP\n最佳配置", "Regime V3\n(自研)", "Tensor DP V2\n(MIT 2025+自研)"]
revenues = [44.16, 41.80, 46.64, 53.81, 56.81]  # 单位：百万元
captures = [53.1, 50.3, 56.1, 64.8, 68.4]
colors = [GRAY, GRAY, GRAY, TEAL, NAVY]

fig, ax = plt.subplots(figsize=(11, 5.2), dpi=150)
fig.patch.set_facecolor("white")

bars = ax.bar(algs, revenues, color=colors, width=0.62, edgecolor="white", linewidth=1.5)

# Oracle 线
ax.axhline(83.08, color=AMBER, linewidth=1.4, linestyle=(0, (5, 3)), alpha=0.6)
ax.text(4.4, 83.08, "Oracle 上限 ¥8,308万", ha="right", va="bottom",
        color=AMBER, fontsize=9.5, fontweight="600")

# 柱上标签
for bar, rev, cap in zip(bars, revenues, captures):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 1.2,
            f"¥{rev:.0f}万", ha="center", va="bottom",
            fontsize=11, fontweight="700", color="#1f2937")
    ax.text(bar.get_x() + bar.get_width()/2, h/2,
            f"{cap:.1f}%", ha="center", va="center",
            fontsize=11, fontweight="600", color="white")

ax.set_ylim(0, 92)
ax.set_ylabel("山东 365 天年化收入（百万元）", fontweight="600")
ax.set_title("算法代际演进：从工业标准到 SOTA", loc="left", pad=16,
             fontsize=14, fontweight="700", color="#1f2937")
ax.grid(axis="y", alpha=0.5)
ax.set_axisbelow(True)

# 对最后一列加高亮框
last_bar = bars[-1]
from matplotlib.patches import FancyBboxPatch
rect = FancyBboxPatch(
    (last_bar.get_x() - 0.02, 0),
    last_bar.get_width() + 0.04, last_bar.get_height(),
    boxstyle="round,pad=0.01",
    linewidth=2.2, edgecolor=NAVY, facecolor="none", zorder=3,
)
ax.add_patch(rect)

plt.tight_layout()
plt.savefig(OUT / "fig1_evolution.png", dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"✓ {OUT / 'fig1_evolution.png'}")


# ============================================================
# Figure 2: 贡献归因（瀑布图）
# ============================================================

fig, ax = plt.subplots(figsize=(10, 4.6), dpi=150)
fig.patch.set_facecolor("white")

# 从 MILP 到 Tensor DP V2 的演进
stages = ["MILP\n工业标准", "+ dev=0%\n(LP 退化)", "+ 日类型\n分类器", "+ stochastic DP\n求解算法", "Tensor DP V2\n当前 SOTA"]
values = [50.3, 56.1, 64.8, 63.75, 68.4]
deltas = [0, 5.8, 8.7, -1.05, 4.65]

colors_w = [GRAY, GREEN, GREEN, "#dc2626", GREEN]

# 瀑布图（相对变化）
cum = 0
xs = range(len(stages))
for i, (stage, delta, color) in enumerate(zip(stages, deltas, colors_w)):
    if i == 0:
        ax.bar(i, values[i], color=GRAY, width=0.55, edgecolor="white", linewidth=1.5)
        ax.text(i, values[i] + 0.5, f"{values[i]:.1f}%", ha="center", va="bottom",
                fontsize=11, fontweight="700", color="#1f2937")
        cum = values[i]
    elif i == len(stages) - 1:
        # 最后一根：从 0 到最终值
        ax.bar(i, values[i], color=NAVY, width=0.55, edgecolor="white", linewidth=1.5)
        ax.text(i, values[i] + 0.5, f"{values[i]:.1f}%", ha="center", va="bottom",
                fontsize=12, fontweight="800", color=NAVY)
        cum = values[i]
    else:
        # 中间柱：显示 delta
        if delta > 0:
            ax.bar(i, delta, bottom=cum, color=color, width=0.55, edgecolor="white", linewidth=1.5)
            ax.text(i, cum + delta + 0.5, f"+{delta:.1f}pp", ha="center", va="bottom",
                    fontsize=10.5, fontweight="700", color=color)
            cum += delta
        else:
            ax.bar(i, -delta, bottom=cum + delta, color=color, width=0.55, edgecolor="white", linewidth=1.5)
            ax.text(i, cum + 0.5, f"{delta:.1f}pp", ha="center", va="bottom",
                    fontsize=10.5, fontweight="700", color=color)
            cum += delta

    # 连线
    if 0 < i < len(stages) - 1:
        prev_top = cum - delta
        ax.plot([i - 1 + 0.28, i - 0.28], [prev_top, prev_top],
                color="#94a3b8", linestyle=":", linewidth=1)

ax.set_xticks(xs)
ax.set_xticklabels(stages, fontsize=10)
ax.set_ylim(0, 78)
ax.set_ylabel("Capture vs Oracle (%)", fontweight="600")
ax.set_title("性能提升归因：场景质量 > 求解算法", loc="left", pad=16,
             fontsize=14, fontweight="700", color="#1f2937")
ax.grid(axis="y", alpha=0.5)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(OUT / "fig2_attribution.png", dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"✓ {OUT / 'fig2_attribution.png'}")


# ============================================================
# Figure 3: per-day capture 分布
# ============================================================

# 加载真实数据
df_v2 = pd.read_csv("/Users/jjj/Desktop/工作/电力交易/energy-storage-rl/runs/vfa_dp/shandong_vfa_dp_full365d_regime_conditioned.csv")
df_v1 = pd.read_csv("/Users/jjj/Desktop/工作/电力交易/energy-storage-rl/runs/vfa_dp/shandong_vfa_dp_full365d.csv")

fig, ax = plt.subplots(figsize=(10, 4.6), dpi=150)
fig.patch.set_facecolor("white")

# 双 histogram
bins = np.linspace(-100, 100, 41)
ax.hist(df_v1["capture_pct"], bins=bins, color=GRAY, alpha=0.65,
        label=f"Tensor DP V1 (capture 63.8%)", edgecolor="white", linewidth=0.5)
ax.hist(df_v2["capture_pct"], bins=bins, color=NAVY, alpha=0.80,
        label=f"Tensor DP V2 (capture 68.4%)", edgecolor="white", linewidth=0.5)

# 中位数线
v2_med = df_v2["capture_pct"].median()
ax.axvline(v2_med, color="#06b6d4", linewidth=1.8, linestyle="--", alpha=0.9)
ax.text(v2_med + 1, ax.get_ylim()[1] * 0.92, f"V2 中位数 {v2_med:.1f}%",
        color="#06b6d4", fontweight="700", fontsize=10)

ax.set_xlabel("单日捕获率 Capture (%)", fontweight="600")
ax.set_ylabel("天数", fontweight="600")
ax.set_title("365 天单日捕获率分布", loc="left", pad=16,
             fontsize=14, fontweight="700", color="#1f2937")
ax.legend(loc="upper left", frameon=False, fontsize=10)
ax.grid(axis="y", alpha=0.4)
ax.set_axisbelow(True)
ax.set_xlim(-90, 100)

plt.tight_layout()
plt.savefig(OUT / "fig3_distribution.png", dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"✓ {OUT / 'fig3_distribution.png'}")


# ============================================================
# Figure 4: 季度表现曲线
# ============================================================

# 每月（30天 window）capture
df_v2["month_bin"] = ((df_v2["day_idx"] - df_v2["day_idx"].min()) // 30).astype(int)

fig, ax = plt.subplots(figsize=(10, 4.2), dpi=150)
fig.patch.set_facecolor("white")

# 按 30 天窗口 rolling capture
df_v2 = df_v2.sort_values("day_idx").reset_index(drop=True)
rolling_window = 30
rolling_capture = df_v2["vfa_dp_revenue"].rolling(rolling_window).sum() / \
                  df_v2["oracle_revenue"].rolling(rolling_window).sum() * 100
days = df_v2["day_idx"].values
# 相对天数（0..365）
relative_days = np.arange(len(df_v2))

ax.plot(relative_days, rolling_capture, color=NAVY, linewidth=2.2, label="30 天滚动 capture")
ax.fill_between(relative_days, rolling_capture, alpha=0.15, color=NAVY)

# 季度分割线
for q_end in [91, 182, 273]:
    ax.axvline(q_end, color="#cbd5e1", linewidth=1, linestyle=":")

# 总平均线
ax.axhline(68.4, color=TEAL, linewidth=1.5, linestyle="--", alpha=0.7)
ax.text(len(df_v2) * 0.98, 68.4 + 1.5, "全年 68.4%", color=TEAL,
        ha="right", va="bottom", fontweight="700", fontsize=10)

# Regime V3 参考线
ax.axhline(64.8, color=GRAY, linewidth=1.2, linestyle=":", alpha=0.7)
ax.text(len(df_v2) * 0.02, 64.8 - 1.8, "Regime V3 基准 64.8%", color=GRAY,
        ha="left", va="top", fontweight="600", fontsize=9)

# 季度标签
for qi, q_label in enumerate(["Q1", "Q2", "Q3", "Q4"]):
    ax.text(45 + qi * 91, 100, q_label, ha="center", fontweight="700", fontsize=11,
            color="#64748b")

ax.set_ylim(30, 105)
ax.set_xlim(0, len(df_v2))
ax.set_xlabel("测试期天数（2024 walk-forward）", fontweight="600")
ax.set_ylabel("滚动 Capture (%)", fontweight="600")
ax.set_title("全年 4 季度捕获率演化", loc="left", pad=16,
             fontsize=14, fontweight="700", color="#1f2937")
ax.grid(axis="y", alpha=0.4)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(OUT / "fig4_rolling.png", dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"✓ {OUT / 'fig4_rolling.png'}")

print("\nAll charts generated in", OUT)
