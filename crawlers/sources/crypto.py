"""
电查查 AES-128-CBC 解密模块

密钥来源：电查查前端JS，每次发版可能变化。
支持自动从JS提取最新密钥。
"""

from __future__ import annotations
import json
import base64
import re
import httpx
from pathlib import Path
from loguru import logger
from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad

from crawlers.config.settings import settings

# 默认密钥（2026-04-09提取）
_DEFAULT_KEY = b"w*qYPqDydSIEdc#*"
_DEFAULT_IV = b"mLPeKAH6psn9dnS2"

# 缓存路径
_KEY_CACHE = settings.data_dir / "dcc_aes_key.json"

# 运行时密钥
_current_key: bytes = _DEFAULT_KEY
_current_iv: bytes = _DEFAULT_IV


def _load_cached_key():
    global _current_key, _current_iv
    if _KEY_CACHE.exists():
        try:
            data = json.loads(_KEY_CACHE.read_text())
            _current_key = data["key"].encode()
            _current_iv = data["iv"].encode()
            logger.debug("Loaded cached AES key")
        except Exception:
            pass


def _save_key(key: str, iv: str):
    _KEY_CACHE.write_text(json.dumps({"key": key, "iv": iv}))


async def refresh_key_from_js():
    """
    从电查查前端JS自动提取最新AES密钥。
    使用Playwright加载页面，拦截JS文件并提取密钥。
    每次爬虫启动时调用，确保密钥最新。
    """
    global _current_key, _current_iv

    from playwright.async_api import async_playwright

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            ctx = await browser.new_context(
                viewport={"width": 1920, "height": 1080},
                user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            )
            page = await ctx.new_page()

            js_bodies = []

            async def on_resp(response):
                url = response.url
                if url.endswith(".js") and "dianchacha" in url:
                    try:
                        body = await response.text()
                        if "enc.Utf8.parse" in body:
                            js_bodies.append(body)
                    except Exception:
                        pass

            page.on("response", on_resp)
            await page.goto("https://www.dianchacha.cn/transaction", wait_until="networkidle", timeout=30000)
            await browser.close()

            for body in js_bodies:
                matches = re.findall(r'enc\.Utf8\.parse\(["\']([^"\']{16})["\']\)', body)
                if len(matches) >= 2:
                    new_key = matches[0]
                    new_iv = matches[1]
                    if new_key != _current_key.decode() or new_iv != _current_iv.decode():
                        logger.info("发现新密钥: key={}...{}, iv={}...{}",
                                    new_key[:4], new_key[-4:], new_iv[:4], new_iv[-4:])
                    _current_key = new_key.encode()
                    _current_iv = new_iv.encode()
                    _save_key(new_key, new_iv)
                    logger.info("AES密钥已更新并缓存")
                    return True

    except Exception as e:
        logger.debug("密钥刷新失败: {}", e)

    logger.warning("未能从JS提取新密钥，使用缓存/默认值")
    return False


def decrypt_response(encrypted_b64: str):
    """
    解密电查查API响应的data字段。
    {"status": 200, "data": "<base64加密>"} → 原始JSON
    """
    if not isinstance(encrypted_b64, str) or len(encrypted_b64) < 10:
        return encrypted_b64

    raw = base64.b64decode(encrypted_b64)
    cipher = AES.new(_current_key, AES.MODE_CBC, _current_iv)
    decrypted = unpad(cipher.decrypt(raw), AES.block_size)
    return json.loads(decrypted.decode("utf-8"))


# 启动时加载缓存密钥
_load_cached_key()
