"""
国网PMOS统一适配器 — 备选数据源

覆盖国网体系所有省份：pmos.{code}.sgcc.com.cn
各省共享相同的前端框架和API结构，通过province_code参数化。
"""

from __future__ import annotations
import asyncio
import json
import httpx
from datetime import date, datetime, timedelta
from loguru import logger

from crawlers.sources.base import SourceAdapter, RawRecord
from crawlers.config.provinces import ProvinceSpec
from crawlers.config.settings import settings
from crawlers.browser.manager import browser_manager
from crawlers.browser.interceptor import XHRInterceptor


class SGCCPmosAdapter(SourceAdapter):
    """国网PMOS统一适配器"""

    name = "sgcc_pmos"

    def __init__(self):
        self._api_configs: dict[str, dict] = {}  # province_code → config
        self._load_configs()

    def _load_configs(self):
        config_path = settings.data_dir / "sgcc_api_configs.json"
        if config_path.exists():
            self._api_configs = json.loads(config_path.read_text())

    def _save_configs(self):
        config_path = settings.data_dir / "sgcc_api_configs.json"
        config_path.write_text(json.dumps(self._api_configs, ensure_ascii=False, indent=2))

    def _get_base_url(self, spec: ProvinceSpec) -> str:
        return f"https://pmos.{spec.province_code}.sgcc.com.cn"

    async def explore_api(self, province: ProvinceSpec):
        """探测指定省份的PMOS API结构"""
        interceptor = XHRInterceptor()
        base_url = self._get_base_url(province)

        await browser_manager.load_cookies(f"sgcc_{province.province_code}")
        page = await browser_manager.new_page()
        await interceptor.attach(page)

        logger.info("Opening {} PMOS: {}", province.name_cn, base_url)
        await page.goto(base_url, wait_until="networkidle", timeout=60000)

        logger.info("请在浏览器中登录并导航到现货价格数据页面，完成后按Enter...")
        await asyncio.get_event_loop().run_in_executor(None, input, "")

        await browser_manager.save_cookies(f"sgcc_{province.province_code}")

        # 分析
        report = interceptor.save_report(f"sgcc_{province.province_code}_api_discovery.json")
        price_apis = interceptor.get_price_apis()
        logger.info("发现 {} 个价格API", len(price_apis))

        if price_apis:
            self._api_configs[province.province_code] = {
                "base_url": base_url,
                "endpoints": [
                    {
                        "url": api.url,
                        "method": api.method,
                        "request_body": api.request_body,
                        "auth_headers": {
                            k: v for k, v in api.request_headers.items()
                            if k.lower() in ["authorization", "token", "x-token", "cookie"]
                        },
                    }
                    for api in price_apis
                ],
                "discovered_at": datetime.now().isoformat(),
            }
            self._save_configs()

        await page.close()
        return price_apis

    async def fetch_day(
        self, province: ProvinceSpec, target_date: date
    ) -> list[RawRecord]:
        config = self._api_configs.get(province.province_code)
        if not config:
            raise RuntimeError(
                f"SGCC API配置未就绪 ({province.name_cn})。"
                f"请先运行: python -m crawlers explore sgcc --province {province.name}"
            )
        # 类似dianchacha的API调用逻辑
        # 具体实现需要在explore后根据API格式填充
        logger.warning("SGCC fetch_day 需要在API探测后实现具体逻辑")
        return []

    async def check_availability(self, province: ProvinceSpec) -> bool:
        return province.province_code in self._api_configs
