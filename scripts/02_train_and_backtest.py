"""
完整Pipeline：特征 → 环境 → Baseline → GRPO训练 → Walk-Forward回测

一个脚本跑完全部。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch
from rich.console import Console
from rich.table import Table
from loguru import logger
from copy import deepcopy

from data.features import build_features, FEATURE_COLS
from env.battery_env import BatteryEnv, BatteryParams, ACTION_NAMES, ACTION_POWER_RATIOS, N_ACTIONS
from agent.baselines import tou_strategy, threshold_strategy, intraday_strategy, do_nothing
from agent.grpo import train_grpo, pretrain_from_baseline, pretrain_from_oracle, train_grpo_trajectory
from agent.baselines import hindsight_oracle

console = Console()


def run_baseline(env: BatteryEnv, df: pd.DataFrame, strategy: str) -> dict:
    """跑一个baseline策略，返回总收益等指标"""
    obs, info = env.reset()
    env._start = 0
    env._step = 0
    env._soc = 0.5
    env._revenue = 0.0
    env._cycles = 0.0

    n_steps = min(len(df) - 1, env._n - env._start - 1)
    actions_taken = []

    for step in range(n_steps):
        idx = env._start + step
        if idx >= len(df):
            break

        row = df.iloc[idx]
        price = row["price"]
        hour = row.get("hour", 0)
        soc = env._soc

        if strategy == "tou":
            action = tou_strategy(obs, hour)
        elif strategy == "threshold":
            ma_96 = row.get("price_ma_96", price)
            action = threshold_strategy(obs, price, ma_96)
        elif strategy == "intraday":
            # 用过去24小时的价格（非后视）
            lookback = min(step, 96)
            start_lb = max(0, idx - lookback)
            prices_so_far = df["price"].values[start_lb:idx+1]
            action = intraday_strategy(prices_so_far, price, soc)
        elif strategy == "oracle":
            action = hindsight_oracle(idx, df["price"].values, soc)
        else:  # do_nothing
            action = 0

        obs, reward, term, trunc, info = env.step(action)
        actions_taken.append(action)

        if term or trunc:
            break

    return {
        "revenue": info["revenue"],
        "cycles": info["cycles"],
        "steps": len(actions_taken),
        "actions": np.array(actions_taken),
    }


def run_rl(env: BatteryEnv, policy, n_steps: int) -> dict:
    """跑RL策略"""
    obs, info = env.reset()
    env._start = 0
    env._step = 0
    env._soc = 0.5
    env._revenue = 0.0
    env._cycles = 0.0

    actions_taken = []
    for _ in range(n_steps):
        action = policy.act(torch.FloatTensor(obs), deterministic=True)
        obs, reward, term, trunc, info = env.step(action)
        actions_taken.append(action)
        if term or trunc:
            break

    return {
        "revenue": info["revenue"],
        "cycles": info["cycles"],
        "steps": len(actions_taken),
        "actions": np.array(actions_taken),
    }


def main():
    console.print("\n[bold]===== Energy Storage RL — ERCOT Pipeline =====[/bold]\n")

    # Step 1: 构建特征
    console.print("[bold cyan]Step 1: 构建特征[/bold cyan]")
    df = build_features("HB_WEST")
    console.print(f"  数据量: {len(df):,} rows")
    console.print(f"  时间范围: {df['timestamp'].min()} ~ {df['timestamp'].max()}")
    console.print(f"  均价: ${df['price'].mean():.2f}/MWh")
    console.print(f"  负电价: {(df['price'] < 0).mean():.1%}")
    console.print(f"  >$100: {(df['price'] > 100).mean():.1%}")

    # Step 2: Walk-Forward回测
    console.print(f"\n[bold cyan]Step 2: Walk-Forward 回测[/bold cyan]")

    # 按年份分割（每年只有Q1数据，所以按年=按数据块）
    df["year"] = df["timestamp"].dt.year
    years = sorted(df["year"].unique())
    console.print(f"  数据年份: {years}")
    console.print(f"  训练前3年，从第4年开始测试\n")

    min_train_years = 3
    results = []
    battery = BatteryParams()

    for i in range(min_train_years, len(years)):
        test_year = years[i]
        train_years = years[:i]

        df_train = df[df["year"].isin(train_years)].reset_index(drop=True)
        df_test = df[df["year"] == test_year].reset_index(drop=True)

        if len(df_test) < 96:
            continue

        console.print(f"  [cyan]测试 {test_year}[/cyan] (训练{len(train_years)}年, "
                       f"训练{len(df_train):,}行, 测试{len(df_test):,}行)")

        # --- Grid Search最优Threshold参数 ---
        train_env = BatteryEnv(df_train, battery, episode_length=96*7, randomize_start=True)
        obs_dim = train_env.N_OBS

        best_cr, best_dr, best_rev = 0.7, 1.3, -1e18
        prices_arr = df_train["price"].values
        ma_96_arr = df_train["price_ma_96"].values if "price_ma_96" in df_train.columns else prices_arr
        for cr in np.arange(0.50, 0.85, 0.05):
            for dr in np.arange(1.15, 1.55, 0.05):
                gs_env = BatteryEnv(df_train, battery, episode_length=len(df_train)-1, randomize_start=False)
                obs, _ = gs_env.reset()
                for s in range(len(df_train) - 1):
                    if ma_96_arr[s] <= 0:
                        a = 0
                    else:
                        r = prices_arr[s] / ma_96_arr[s]
                        sc = cr + 0.15
                        sd = dr - 0.15
                        if r < cr: a = 2
                        elif r < sc: a = 1
                        elif r > dr: a = 4
                        elif r > sd: a = 3
                        else: a = 0
                    obs, _, term, trunc, info = gs_env.step(a)
                    if term or trunc:
                        break
                rev = info["revenue"]
                if rev > best_rev:
                    best_rev = rev
                    best_cr, best_dr = cr, dr
        console.print(f"    最优阈值: charge={best_cr:.2f} discharge={best_dr:.2f} (训练收入${best_rev:+,.0f})")

        # --- BC训练（用最优参数生成演示数据）---
        console.print(f"    BC训练 (20 epochs, 参数[{best_cr:.2f},{best_dr:.2f}])...")
        policy = pretrain_from_baseline(train_env, df_train, obs_dim, n_epochs=20,
                                        charge_ratio=best_cr, discharge_ratio=best_dr)

        # --- 测试 ---
        test_env = BatteryEnv(df_test, battery, episode_length=len(df_test) - 1, randomize_start=False)
        n_test_steps = len(df_test) - 1

        # RL
        rl_result = run_rl(test_env, policy, n_test_steps)

        # RL动作分布
        rl_acts = rl_result["actions"]
        act_pct = np.bincount(rl_acts, minlength=5) / max(len(rl_acts), 1)
        act_str = " ".join(f"{ACTION_NAMES[i][:3]}={act_pct[i]:.0%}" for i in range(5))
        console.print(f"    RL动作: [{act_str}]")

        # Baselines
        baseline_results = {}
        for strat in ["tou", "threshold", "intraday", "oracle"]:
            benv = BatteryEnv(df_test, battery, episode_length=n_test_steps, randomize_start=False)
            baseline_results[strat] = run_baseline(benv, df_test, strat)

        # 记录
        row = {
            "quarter": str(test_year),
            "rl_revenue": rl_result["revenue"],
            "rl_cycles": rl_result["cycles"],
        }
        for strat, br in baseline_results.items():
            row[f"{strat}_revenue"] = br["revenue"]
            row[f"{strat}_cycles"] = br["cycles"]
        results.append(row)

        # 打印本季度结果
        fair_baselines = {k: v for k, v in baseline_results.items() if k != "oracle"}
        best_base = max(fair_baselines.items(), key=lambda x: x[1]["revenue"])
        console.print(
            f"    RL=${rl_result['revenue']:+,.0f} ({rl_result['cycles']:.1f}cyc) | "
            f"Thresh=${baseline_results['threshold']['revenue']:+,.0f} | "
            f"Intra=${baseline_results['intraday']['revenue']:+,.0f} | "
            f"Oracle=${baseline_results['oracle']['revenue']:+,.0f} | "
            f"Best(fair)={best_base[0]}"
        )

    # === 汇总 ===
    if not results:
        console.print("[red]无有效测试季度[/red]")
        return

    rdf = pd.DataFrame(results)

    console.print(f"\n[bold cyan]===== 汇总 =====[/bold cyan]\n")

    table = Table(title="Walk-Forward Results (200MW/400MWh Battery)")
    table.add_column("Quarter", style="cyan")
    table.add_column("RL", style="green")
    table.add_column("TOU", style="yellow")
    table.add_column("Threshold", style="blue")
    table.add_column("Intraday", style="magenta")
    table.add_column("Oracle*", style="dim")
    table.add_column("RL Win?", style="bold")

    for _, row in rdf.iterrows():
        best_base = max(row["tou_revenue"], row["threshold_revenue"], row["intraday_revenue"])
        win = "✅" if row["rl_revenue"] > best_base else "❌"
        table.add_row(
            row["quarter"],
            f"${row['rl_revenue']:+,.0f}",
            f"${row['tou_revenue']:+,.0f}",
            f"${row['threshold_revenue']:+,.0f}",
            f"${row['intraday_revenue']:+,.0f}",
            f"${row['oracle_revenue']:+,.0f}",
            win,
        )

    # 总计
    rl_total = rdf["rl_revenue"].sum()
    tou_total = rdf["tou_revenue"].sum()
    thresh_total = rdf["threshold_revenue"].sum()
    intra_total = rdf["intraday_revenue"].sum()
    oracle_total = rdf["oracle_revenue"].sum()
    best_total = max(tou_total, thresh_total, intra_total)

    table.add_row(
        "[bold]TOTAL[/bold]",
        f"[bold]${rl_total:+,.0f}[/bold]",
        f"[bold]${tou_total:+,.0f}[/bold]",
        f"[bold]${thresh_total:+,.0f}[/bold]",
        f"[bold]${intra_total:+,.0f}[/bold]",
        f"${oracle_total:+,.0f}",
        f"[bold]{'✅' if rl_total > best_total else '❌'}[/bold]",
    )
    console.print(table)

    # 年化
    n_q = len(rdf)
    years = n_q / 4
    console.print(f"\n  测试期: {n_q}个季度 ({years:.1f}年)")
    console.print(f"  RL年收入:       ${rl_total / years:+,.0f}")
    console.print(f"  TOU年收入:      ${tou_total / years:+,.0f}")
    console.print(f"  Threshold年收入: ${thresh_total / years:+,.0f}")
    console.print(f"  Intraday年收入:  ${intra_total / years:+,.0f}")
    console.print(f"  Oracle*年收入:   ${oracle_total / years:+,.0f}  ← 有未来信息，非公平对比")

    win_rate = sum(1 for _, r in rdf.iterrows()
                   if r["rl_revenue"] > max(r["tou_revenue"], r["threshold_revenue"], r["intraday_revenue"]))
    vs_thresh = rl_total / max(thresh_total, 1) - 1
    console.print(f"  RL vs Threshold: {vs_thresh:+.1%}")
    console.print(f"  RL胜率: {win_rate}/{n_q} ({win_rate/n_q:.0%})")

    if rl_total > best_total:
        delta = rl_total - best_total
        console.print(f"\n  [bold green]✅ RL比最佳baseline每年多赚 ${delta/years:+,.0f}[/bold green]")
    else:
        console.print(f"\n  [yellow]RL未超过最佳baseline[/yellow]")


if __name__ == "__main__":
    main()
