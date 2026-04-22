"""登录态管理 — cookie持久化 + 自动续期"""

from __future__ import annotations
import asyncio
from playwright.async_api import Page
from loguru import logger

from crawlers.browser.manager import browser_manager
from crawlers.config.settings import settings


async def ensure_login(source: str, login_url: str, check_url: str) -> Page:
    """
    确保登录态有效。
    1. 尝试加载已保存的cookie
    2. 访问check_url验证cookie有效性
    3. 无效则打开login_url让用户手动登录
    4. 保存新cookie

    Args:
        source: 数据源标识 (dianchacha/sgcc等)
        login_url: 登录页URL
        check_url: 用于验证登录态的页面URL
    """
    page = await browser_manager.new_page()

    # 尝试已保存的cookie
    loaded = await browser_manager.load_cookies(source)
    if loaded:
        await page.goto(check_url, wait_until="networkidle", timeout=settings.browser_timeout)
        if await _is_logged_in(page):
            logger.info("{}: cookie有效，跳过登录", source)
            return page
        logger.warning("{}: cookie过期，需要重新登录", source)

    # 需要手动登录
    await page.goto(login_url, wait_until="networkidle", timeout=settings.browser_timeout)

    logger.info("="*60)
    logger.info("请在弹出的浏览器中手动登录 {}", source)
    logger.info("登录完成后，按Enter继续...")
    logger.info("="*60)

    # 切换到非headless让用户操作
    await asyncio.get_event_loop().run_in_executor(None, input, "")

    # 保存cookie
    await browser_manager.save_cookies(source)
    logger.info("{}: 登录完成，cookie已保存", source)

    return page


async def _is_logged_in(page: Page) -> bool:
    """检查页面是否处于登录态（通用启发式）"""
    url = page.url
    # 如果被重定向到登录页，说明未登录
    login_keywords = ["login", "signin", "登录", "auth"]
    if any(kw in url.lower() for kw in login_keywords):
        return False
    # 检查页面内容是否有登录相关元素
    content = await page.content()
    if any(kw in content for kw in ["请登录", "登录/注册", "login-form"]):
        return False
    return True
