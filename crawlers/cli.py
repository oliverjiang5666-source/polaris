"""CLI命令行工具"""

from __future__ import annotations
import asyncio
import click
from datetime import date, datetime, timedelta
from loguru import logger

from crawlers.config.provinces import PROVINCES, get_province, list_provinces


@click.group()
def cli():
    """中国22省电力现货数据爬虫"""
    pass


@cli.command()
@click.option("--source", default="dianchacha", type=click.Choice(["dianchacha", "sgcc"]))
@click.option("--province", default=None, help="省份英文名（sgcc模式必需）")
@click.option("--headless/--no-headless", default=False, help="是否使用无头浏览器")
def explore(source, province, headless):
    """探测数据源API结构（需要浏览器交互）"""
    from crawlers.config.settings import settings
    settings.headless = headless

    async def _run():
        from crawlers.browser.manager import browser_manager
        try:
            if source == "dianchacha":
                from crawlers.sources.dianchacha import DianChaChaAdapter
                adapter = DianChaChaAdapter()
                await adapter.explore_api()
            elif source == "sgcc":
                if not province:
                    click.echo("SGCC模式需要指定省份: --province shandong")
                    return
                from crawlers.sources.sgcc_pmos import SGCCPmosAdapter
                spec = get_province(province)
                adapter = SGCCPmosAdapter()
                await adapter.explore_api(spec)
        finally:
            await browser_manager.stop()

    asyncio.run(_run())


@cli.command()
@click.option("--province", required=True, help="省份英文名 (shandong/shanxi/...)")
@click.option("--date", "target_date", required=True, help="日期 YYYY-MM-DD")
@click.option("--source", default="dianchacha")
def fetch(province, target_date, source):
    """抓取指定省份指定日期的数据"""
    spec = get_province(province)
    d = datetime.strptime(target_date, "%Y-%m-%d").date()

    async def _run():
        from crawlers.sources.dianchacha import DianChaChaAdapter
        from crawlers.storage.db import db
        from crawlers.pipeline.quality import QualityChecker

        adapter = DianChaChaAdapter()
        try:
            records = await adapter.fetch_day(spec, d)
            click.echo(f"获取 {len(records)} 条记录")

            if records:
                checker = QualityChecker()
                report = checker.check(records, spec)
                click.echo(report.summary())

                db.upsert(records)
                click.echo(f"已写入DuckDB (总计 {db.total_records()} 条)")
        finally:
            await adapter.close()

    asyncio.run(_run())


@cli.command()
@click.option("--province", required=True)
@click.option("--start", required=True, help="起始日期 YYYY-MM-DD")
@click.option("--end", required=True, help="结束日期 YYYY-MM-DD")
@click.option("--source", default="dianchacha")
def backfill(province, start, end, source):
    """回补历史数据"""
    spec = get_province(province)
    start_d = datetime.strptime(start, "%Y-%m-%d").date()
    end_d = datetime.strptime(end, "%Y-%m-%d").date()
    days = (end_d - start_d).days + 1
    click.echo(f"回补 {spec.name_cn} {start} ~ {end} ({days}天)")

    async def _run():
        from crawlers.sources.dianchacha import DianChaChaAdapter
        from crawlers.storage.db import db

        adapter = DianChaChaAdapter()
        try:
            records = await adapter.fetch_range(
                spec, start_d, end_d,
                on_day_done=lambda d, n: click.echo(f"  {d}: {n}条"),
            )
            if records:
                db.upsert(records)
                click.echo(f"回补完成: {len(records)}条 → DuckDB")
        finally:
            await adapter.close()

    asyncio.run(_run())


@cli.command("export")
@click.option("--province", default=None, help="指定省份，不填则导出全部")
@click.option("--format", "fmt", default="parquet", type=click.Choice(["parquet"]))
def export_cmd(province, fmt):
    """导出数据为parquet"""
    from crawlers.pipeline.export import export_all_parquet, export_province_parquet

    if province:
        path = export_province_parquet(province)
        click.echo(f"导出: {path}")
    else:
        export_all_parquet()
        click.echo("全部省份已导出")


@cli.command()
def stats():
    """查看数据库统计"""
    from crawlers.storage.db import db
    df = db.get_stats()
    if df.empty:
        click.echo("数据库为空")
    else:
        click.echo(f"\n总记录数: {db.total_records():,}")
        click.echo(df.to_string(index=False))


@cli.command("list")
def list_cmd():
    """列出所有支持的省份"""
    click.echo(f"\n已注册 {len(PROVINCES)} 个省份:\n")
    click.echo(f"{'英文名':<15} {'中文名':<8} {'电网':<8} {'出清间隔':>8} {'数据起始':<12}")
    click.echo("-" * 60)
    for name in sorted(PROVINCES):
        spec = PROVINCES[name]
        click.echo(
            f"{spec.name:<15} {spec.name_cn:<8} {spec.grid:<8} "
            f"{spec.settlement_interval:>5}min  {spec.price_start:<12}"
        )


@cli.command()
def check():
    """检查各省数据完整性"""
    from crawlers.storage.db import db
    from crawlers.config.provinces import PROVINCES

    stats_df = db.get_stats()
    if stats_df.empty:
        click.echo("数据库为空，请先抓取数据")
        return

    click.echo(f"\n数据完整性检查 ({len(PROVINCES)} 省):\n")
    for name, spec in sorted(PROVINCES.items()):
        row = stats_df[stats_df["province"] == spec.name_cn]
        if row.empty:
            click.echo(f"  {spec.name_cn:<6} ❌ 无数据")
        else:
            r = row.iloc[0]
            click.echo(
                f"  {spec.name_cn:<6} ✓ {r['record_count']:>8,}条  "
                f"{r['indicator_count']}个指标  "
                f"{r['earliest']} ~ {r['latest']}"
            )
