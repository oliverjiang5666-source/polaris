"""
Polaris 山东 · Script 06 — L2 A/B: VAE vs Regime-conditioned scenarios
=======================================================================

训练 Conditional VAE (完整 80 epoch on 2020-11 → 2024-12 historical) 然后
用 VAE 采样场景 vs 原 regime-conditioned bootstrap 作 Tensor DP 输入.

测试集: 2025 全年代表 60 天 (每 6 天取 1 天) 做 A/B 回测, 看平均 revenue / capture.

Usage:
    PYTHONPATH=. python3 products/polaris_shandong/scripts/06_vae_ab_vs_regime.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from config import BatteryConfig
from oracle.lp_oracle import solve_day
from optimization.milp.data_loader import load_province
from optimization.milp.scenario_generator import RegimeClassifier
from optimization.vfa_dp.tensor_dp import TensorDP, DPConfig

from products.polaris_shandong.forecast_v3_vae import VAEScenarioGenerator, VAEConfig


OUTPUT = ROOT / "runs" / "polaris_shandong" / "l2_ab"
OUTPUT.mkdir(parents=True, exist_ok=True)


def build_regime_scenarios(classifier, data, target_day, R_cap=500):
    probs, _ = classifier.predict_regime_probs(data, target_day)
    probs = probs / probs.sum()
    train_labels = classifier.train_labels
    n_train = len(train_labels)
    prices_all = data.rt_prices[:n_train]
    n_reg = classifier.n_regimes
    counts = np.array([(train_labels == c).sum() for c in range(n_reg)])
    w = np.zeros(n_train)
    for d in range(n_train):
        if counts[train_labels[d]] > 0:
            w[d] = probs[train_labels[d]] / counts[train_labels[d]]
    w = w / w.sum()
    idx = np.argsort(-w)[:R_cap]
    idx = np.sort(idx)
    return prices_all[idx].T, np.tile((w[idx] / w[idx].sum())[None, :], (96, 1))


def run_ab(
    capacity_mw: float = 100.0,
    capacity_mwh: float = 200.0,
    delta_soc: float = 0.005,
    R_cap: int = 500,
    vae_epochs: int = 80,
    n_test_days: int = 60,
):
    battery = BatteryConfig(capacity_mw=capacity_mw, capacity_mwh=capacity_mwh)
    data = load_province("shandong")
    df = data.df

    # === Train VAE on history (before 2025) ===
    logger.info("Training VAE...")
    df_train = df.loc[:"2024-12-31"]
    vae = VAEScenarioGenerator(VAEConfig(epochs=vae_epochs, lr=1e-3))
    t0 = time.time()
    vae.fit(df_train, price_col="rt_price")
    logger.info(f"VAE trained in {time.time()-t0:.1f}s\n")

    # === Train regime classifier ===
    day_starts = df.index[::96].normalize()
    y25 = int((day_starts >= pd.Timestamp("2025-01-01")).argmax())
    clf = RegimeClassifier(n_regimes=12)
    clf.fit(data, train_day_end=y25)

    # === Test days: 每 6 天采 1 天 (~60 天) ===
    all_2025 = list(range(y25, len(day_starts)))
    step = max(1, len(all_2025) // n_test_days)
    test_days = all_2025[::step][:n_test_days]

    dp = TensorDP(battery, DPConfig(delta_soc=delta_soc))
    logger.info(f"Test days: {len(test_days)} ({df.index[test_days[0]*96].date()} ~ {df.index[test_days[-1]*96].date()})")

    records = []
    for i, d in enumerate(test_days):
        actual_rt = data.rt_prices[d]
        oracle = solve_day(actual_rt, battery, init_soc=0.5)
        oracle_rev = oracle["revenue"]

        # Regime-conditioned
        scen_r, probs_r = build_regime_scenarios(clf, data, d, R_cap)
        V_r = dp.backward_induction(scen_r, probs_r)
        sim_r = dp.forward_simulate(V_r, actual_rt, init_soc=0.5)
        rev_r = float((sim_r["powers"] * actual_rt * 0.25).sum() - 2 * np.abs(sim_r["powers"]).sum() * 0.25)

        # VAE
        scen_v, probs_v = vae.sample_scenarios(d, df, n_scenarios=R_cap)
        V_v = dp.backward_induction(scen_v, probs_v)
        sim_v = dp.forward_simulate(V_v, actual_rt, init_soc=0.5)
        rev_v = float((sim_v["powers"] * actual_rt * 0.25).sum() - 2 * np.abs(sim_v["powers"]).sum() * 0.25)

        records.append({
            "day_idx": d,
            "date": str(df.index[d*96].date()),
            "regime_rev": rev_r,
            "regime_cap": rev_r / oracle_rev * 100 if oracle_rev > 0 else 0,
            "vae_rev": rev_v,
            "vae_cap": rev_v / oracle_rev * 100 if oracle_rev > 0 else 0,
            "oracle_rev": oracle_rev,
            "vae_vs_regime": rev_v - rev_r,
        })

        if (i + 1) % 10 == 0 or i == 0:
            logger.info(
                f"  {records[-1]['date']} ({i+1}/{len(test_days)}): "
                f"regime=¥{rev_r:>7,.0f} ({records[-1]['regime_cap']:.1f}%)  "
                f"VAE=¥{rev_v:>7,.0f} ({records[-1]['vae_cap']:.1f}%)  "
                f"Δ=¥{rev_v-rev_r:+,.0f}"
            )

    df_res = pd.DataFrame(records)
    total_regime = df_res["regime_rev"].sum()
    total_vae = df_res["vae_rev"].sum()
    total_oracle = df_res["oracle_rev"].sum()

    logger.info(f"\n{'='*70}")
    logger.info(f"  L2 A/B 汇总 ({len(test_days)} 代表天, 100MW/200MWh)")
    logger.info(f"{'='*70}")
    logger.info(f"  Regime-conditioned: ¥{total_regime:>12,.0f}  capture {total_regime/total_oracle*100:>5.2f}%")
    logger.info(f"  VAE generator:      ¥{total_vae:>12,.0f}  capture {total_vae/total_oracle*100:>5.2f}%")
    gain = total_vae - total_regime
    logger.info(f"  VAE vs Regime:      ¥{gain:+,.0f} ({gain/total_regime*100:+.2f}%)")
    logger.info(f"  Capture improvement: {(total_vae/total_oracle - total_regime/total_oracle)*100:+.2f} pp")
    logger.info(f"\n  VAE 胜天数: {(df_res['vae_vs_regime'] > 0).sum()}/{len(df_res)}")
    logger.info(f"  Regime 胜天数: {(df_res['vae_vs_regime'] < 0).sum()}/{len(df_res)}")

    df_res.to_csv(OUTPUT / "vae_vs_regime_ab.csv", index=False)
    return df_res


if __name__ == "__main__":
    run_ab(vae_epochs=80, n_test_days=60)
