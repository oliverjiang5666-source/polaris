"""
电查查适配器 — 首选数据源（AES解密已破解）

数据流：
  1. Playwright访问页面建立session → 手机号验证码登录 → 保存cookie
  2. 带cookie请求API → 获取加密JSON → AES-128-CBC解密 → RawRecord
  3. 后续请求复用cookie，过期则重新登录

API清单（已探测确认）：
  - /electric/api/v2/visitor/trade/datatype?regionId=  → 数据类型树（明文）
  - /electric/api/v2/trade/common/power                → 省份列表（明文）
  - /electric/api/v2/visitor/trade/basic?regionId=&typeId=&timeDimension=MINUTE → 现货数据（加密，需登录）
"""

from __future__ import annotations
import asyncio
import json
import random
from datetime import date, datetime, timedelta
from loguru import logger
from playwright.async_api import Page

from crawlers.sources.base import SourceAdapter, RawRecord
from crawlers.sources.crypto import decrypt_response
from crawlers.config.provinces import ProvinceSpec
from crawlers.config.settings import settings
from crawlers.browser.manager import browser_manager

# 省份 → regionId（从 /trade/common/power 获取，2026-04-09确认）
REGION_IDS = {
    "山东": 370000, "山西": 140000, "广东": 440000, "甘肃": 620000,
    "浙江": 330000, "四川": 510000, "福建": 350000, "内蒙古": 150000,
    "上海": 310000, "江苏": 320000, "安徽": 340000, "辽宁": 210000,
    "河南": 410000, "湖北": 420000, "重庆": 500000, "湖南": 430000,
    "宁夏": 640000, "吉林": 220000, "黑龙江": 230000,
    "新疆": 650000, "青海": 630000, "江西": 360000,
    "蒙西": 150000,
}

# 现货指标typeId（从 /visitor/trade/datatype 获取）
SPOT_TYPE_IDS = {
    "日前价格": {"id": 21, "unit": "元/兆瓦时", "dim": "MINUTE"},
    "实时价格": {"id": 22, "unit": "元/兆瓦时", "dim": "MINUTE"},
    "负荷":     {"id": 23, "unit": "MW",       "dim": "MINUTE"},
    "联络线":   {"id": 24, "unit": "MW",       "dim": "MINUTE"},
    "风电出力": {"id": 28, "unit": "MW",       "dim": "MINUTE"},
    "光伏出力": {"id": 29, "unit": "MW",       "dim": "MINUTE"},
    "新能源总出力": {"id": 30, "unit": "MW",   "dim": "MINUTE"},
    "充放价格": {"id": 55354, "unit": "元/兆瓦时", "dim": "DAY"},
}

_BASE = "https://www.dianchacha.cn"
_API = "/electric/api/v2"


class DianChaChaAdapter(SourceAdapter):
    name = "dianchacha"

    def __init__(self):
        self._page: Page = None
        self._logged_in = False
        self._key_refreshed = False

    async def _ensure_page(self) -> Page:
        if self._page and not self._page.is_closed():
            return self._page

        # 首次启动：自动刷新AES密钥
        if not self._key_refreshed:
            from crawlers.sources.crypto import refresh_key_from_js
            await refresh_key_from_js()
            self._key_refreshed = True

        await browser_manager.load_cookies("dianchacha")
        self._page = await browser_manager.new_page()
        await self._page.goto(_BASE, wait_until="networkidle", timeout=30000)
        return self._page

    async def _api_get(self, path: str):
        """通过浏览器fetch请求API，自动带cookie和session"""
        page = await self._ensure_page()
        result = await page.evaluate(f"""
        async () => {{
            try {{
                let r = await fetch('{path}');
                return await r.json();
            }} catch(e) {{
                return {{error: e.message}};
            }}
        }}
        """)
        return result

    async def login(self):
        """手机号验证码登录（需要用户输入）"""
        page = await self._ensure_page()
        await page.goto(_BASE, wait_until="networkidle", timeout=30000)

        # 点击登录按钮
        try:
            login_btn = page.get_by_text("登录").first
            await login_btn.click()
            await page.wait_for_timeout(2000)
        except Exception:
            pass

        logger.info("="*60)
        logger.info("电查查登录")
        logger.info("="*60)
        logger.info("浏览器已打开登录弹窗。")
        logger.info("请输入你的手机号和验证码完成登录。")
        logger.info("登录成功后按Enter继续...")
        logger.info("="*60)

        await asyncio.get_event_loop().run_in_executor(None, input, "")

        await browser_manager.save_cookies("dianchacha")
        self._logged_in = True
        logger.info("登录成功，cookie已保存")

    async def _check_login_status(self) -> bool:
        r = await self._api_get(f"{_API}/visitor/trade/basic?regionId=370000&typeId=21&timeDimension=MINUTE")
        if r and r.get("data") and isinstance(r["data"], str) and len(r["data"]) > 10:
            return True
        return False

    # ============================================================
    # 核心抓取
    # ============================================================

    async def fetch_day(self, province: ProvinceSpec, target_date: date) -> list[RawRecord]:
        region_id = REGION_IDS.get(province.name_cn)
        if not region_id:
            logger.warning("未知regionId: {}", province.name_cn)
            return []

        records = []
        province_label = province.name_cn
        if not province_label.endswith(("省","市","区")):
            province_label += "省"

        for indicator_name, spec in SPOT_TYPE_IDS.items():
            if indicator_name not in province.indicator_map:
                continue

            type_id = spec["id"]
            unit = spec["unit"]
            dim = spec["dim"]

            path = (
                f"{_API}/visitor/trade/basic"
                f"?regionId={region_id}&typeId={type_id}&timeDimension={dim}"
            )

            resp = await self._api_get(path)
            if not resp:
                continue

            data = resp.get("data")
            if not data or not isinstance(data, str) or len(data) < 10:
                continue

            try:
                decrypted = decrypt_response(data)
            except Exception as e:
                logger.error("解密失败 {} {}: {}", province.name_cn, indicator_name, e)
                continue

            # 解析解密后的数据
            values = []
            if isinstance(decrypted, dict):
                values = decrypted.get("values", [])
            elif isinstance(decrypted, list):
                values = decrypted

            for v in values:
                try:
                    ts_str = v.get("dataTime") or v.get("dateTimeStr")
                    val = v.get("dataNumValue") or v.get("dataValue")
                    if ts_str is None or val is None:
                        continue

                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    # 过滤目标日期
                    if ts.date() != target_date:
                        continue

                    records.append(RawRecord(
                        indicator=indicator_name,
                        province=province_label,
                        timestamp=ts,
                        value=float(val),
                        unit=v.get("dataUnit", unit),
                        source="dianchacha",
                    ))
                except Exception:
                    continue

            if records:
                logger.debug("  {} {} → {}条", province.name_cn, indicator_name, len([r for r in records if r.indicator == indicator_name]))

            await asyncio.sleep(random.uniform(0.5, 1.5))

        return records

    async def fetch_all_available(self, province: ProvinceSpec) -> list[RawRecord]:
        """获取该省所有可用的现货数据（不限日期）"""
        region_id = REGION_IDS.get(province.name_cn)
        if not region_id:
            return []

        records = []
        province_label = province.name_cn
        if not province_label.endswith(("省","市","区")):
            province_label += "省"

        for indicator_name, spec in SPOT_TYPE_IDS.items():
            if indicator_name not in province.indicator_map:
                continue

            path = (
                f"{_API}/visitor/trade/basic"
                f"?regionId={region_id}&typeId={spec['id']}&timeDimension={spec['dim']}"
            )

            resp = await self._api_get(path)
            data = resp.get("data") if resp else None
            if not data or not isinstance(data, str) or len(data) < 10:
                continue

            try:
                decrypted = decrypt_response(data)
            except Exception:
                continue

            values = decrypted.get("values", []) if isinstance(decrypted, dict) else (decrypted if isinstance(decrypted, list) else [])

            for v in values:
                try:
                    ts_str = v.get("dataTime") or v.get("dateTimeStr")
                    val = v.get("dataNumValue") or v.get("dataValue")
                    if ts_str is None or val is None:
                        continue
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    records.append(RawRecord(
                        indicator=indicator_name,
                        province=province_label,
                        timestamp=ts,
                        value=float(val),
                        unit=v.get("dataUnit", spec["unit"]),
                        source="dianchacha",
                    ))
                except Exception:
                    continue

            logger.info("  {} {}: {}条", province.name_cn, indicator_name, len([r for r in records if r.indicator == indicator_name]))
            await asyncio.sleep(random.uniform(1.0, 2.0))

        return records

    async def check_availability(self, province: ProvinceSpec) -> bool:
        return REGION_IDS.get(province.name_cn) is not None

    async def close(self):
        if self._page and not self._page.is_closed():
            await self._page.close()
            self._page = None

    # ============================================================
    # API探测
    # ============================================================

    async def explore_api(self):
        """探测模式：登录并验证数据访问"""
        from crawlers.browser.interceptor import XHRInterceptor
        interceptor = XHRInterceptor()

        page = await self._ensure_page()
        await interceptor.attach(page)

        if not await self._check_login_status():
            logger.info("未登录，启动登录流程...")
            await self.login()

        if await self._check_login_status():
            logger.success("登录验证成功！可以获取现货MINUTE数据")

            # 测试几个省
            from crawlers.config.provinces import get_province
            for prov_name in ["shandong", "shanxi", "guangdong"]:
                spec = get_province(prov_name)
                records = await self.fetch_day(spec, date.today() - timedelta(days=1))
                logger.info("{}: {}条记录", spec.name_cn, len(records))
        else:
            logger.error("登录失败或数据访问受限")

        report = interceptor.save_report("dianchacha_explore.json")
        logger.info("探测报告: {}", report)
