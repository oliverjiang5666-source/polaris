"""
Step 1: 拉取ERCOT历史电价数据

用gridstatus拉取DAM(日前,小时级)和RTM(实时,15分钟级)结算点价格。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from data.fetch_ercot import fetch_and_save
from data.storage import load
from config import MarketConfig

console = Console()

def main():
    cfg = MarketConfig()
    console.print(f"\n[bold]===== ERCOT Data Fetch =====[/bold]\n")
    console.print(f"Years: {cfg.years}")
    console.print(f"Hubs: {cfg.hubs}\n")

    dam, rtm = fetch_and_save(cfg.years)

    # 验证
    console.print(f"\n[bold cyan]Data Summary[/bold cyan]")
    if not dam.empty:
        console.print(f"  DAM: {len(dam):,} rows, {dam['timestamp'].min()} ~ {dam['timestamp'].max()}")
        console.print(f"  DAM columns: {list(dam.columns)}")
    if not rtm.empty:
        console.print(f"  RTM: {len(rtm):,} rows, {rtm['timestamp'].min()} ~ {rtm['timestamp'].max()}")
        console.print(f"  RTM columns: {list(rtm.columns)}")

    # 打印价格统计
    if not rtm.empty:
        price_cols = [c for c in rtm.columns if "price" in c or "spp" in c]
        if price_cols:
            pc = price_cols[0]
            console.print(f"\n[bold cyan]RTM Price Stats ({pc})[/bold cyan]")
            console.print(f"  Mean:   ${rtm[pc].mean():.2f}/MWh")
            console.print(f"  Median: ${rtm[pc].median():.2f}/MWh")
            console.print(f"  Min:    ${rtm[pc].min():.2f}/MWh")
            console.print(f"  Max:    ${rtm[pc].max():.2f}/MWh")
            console.print(f"  Std:    ${rtm[pc].std():.2f}/MWh")

            # 峰谷比
            hourly = rtm.copy()
            hourly["hour"] = hourly["timestamp"].dt.hour
            peak = hourly[hourly["hour"].between(16, 20)][pc].mean()
            valley = hourly[hourly["hour"].between(1, 5)][pc].mean()
            if valley > 0:
                console.print(f"  Peak/Valley: {peak/valley:.1f}x")


if __name__ == "__main__":
    main()
