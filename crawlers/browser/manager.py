"""Playwright 浏览器生命周期管理 + 反检测"""

from __future__ import annotations
import asyncio
import json
from pathlib import Path
from playwright.async_api import async_playwright, Browser, BrowserContext, Page
from playwright_stealth import Stealth
from loguru import logger

from crawlers.config.settings import settings


class BrowserManager:
    """单例浏览器管理器，复用browser实例"""

    def __init__(self):
        self._playwright = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None

    async def start(self) -> BrowserContext:
        if self._context:
            return self._context

        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=settings.headless,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
            ],
        )
        self._context = await self._browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            locale="zh-CN",
            timezone_id="Asia/Shanghai",
        )
        logger.info("Browser started (headless={})", settings.headless)
        return self._context

    async def new_page(self) -> Page:
        ctx = await self.start()
        page = await ctx.new_page()
        # stealth v2: apply via context init scripts
        try:
            stealth = Stealth()
            for script in stealth._evasion_scripts():
                await page.add_init_script(script)
        except Exception:
            pass  # stealth is best-effort
        return page

    async def load_cookies(self, source: str) -> bool:
        cookie_file = settings.cookie_dir / f"{source}.json"
        if not cookie_file.exists():
            return False
        ctx = await self.start()
        cookies = json.loads(cookie_file.read_text())
        await ctx.add_cookies(cookies)
        logger.info("Loaded {} cookies for {}", len(cookies), source)
        return True

    async def save_cookies(self, source: str):
        ctx = await self.start()
        cookies = await ctx.cookies()
        cookie_file = settings.cookie_dir / f"{source}.json"
        cookie_file.write_text(json.dumps(cookies, ensure_ascii=False, indent=2))
        logger.info("Saved {} cookies for {}", len(cookies), source)

    async def stop(self):
        if self._browser:
            await self._browser.close()
            self._browser = None
            self._context = None
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None
        logger.info("Browser stopped")


# 全局单例
browser_manager = BrowserManager()
