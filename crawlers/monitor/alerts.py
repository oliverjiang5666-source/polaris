"""告警推送 — Bark / 钉钉"""

from __future__ import annotations
import httpx
from loguru import logger

from crawlers.config.settings import settings


async def send_alert(title: str, message: str):
    """发送告警通知"""
    if settings.bark_url:
        await _send_bark(title, message)
    if settings.dingtalk_webhook:
        await _send_dingtalk(title, message)
    if not settings.bark_url and not settings.dingtalk_webhook:
        logger.warning("ALERT (no push configured): {} - {}", title, message)


async def _send_bark(title: str, message: str):
    try:
        url = f"{settings.bark_url}/{title}/{message}"
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, timeout=10)
            if resp.status_code == 200:
                logger.info("Bark alert sent: {}", title)
            else:
                logger.warning("Bark alert failed: {}", resp.status_code)
    except Exception as e:
        logger.error("Bark error: {}", e)


async def _send_dingtalk(title: str, message: str):
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                settings.dingtalk_webhook,
                json={
                    "msgtype": "text",
                    "text": {"content": f"[电力爬虫] {title}\n{message}"},
                },
                timeout=10,
            )
            if resp.status_code == 200:
                logger.info("DingTalk alert sent: {}", title)
    except Exception as e:
        logger.error("DingTalk error: {}", e)
