"""
XHR拦截器 — 核心工具

用Playwright拦截SPA网站的XHR/Fetch请求，
自动发现并记录API endpoint、参数格式、鉴权方式。
一旦摸清API，后续可直接用httpx调用，脱离浏览器。
"""

from __future__ import annotations
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from playwright.async_api import Page, Response
from loguru import logger

from crawlers.config.settings import settings


@dataclass
class CapturedAPI:
    url: str
    method: str
    request_headers: dict
    request_body: str | None
    status: int
    response_body: str | None
    content_type: str = ""
    timestamp: str = ""

    def is_json(self) -> bool:
        return "json" in self.content_type.lower()

    def contains_price_data(self) -> bool:
        keywords = ["price", "价格", "出清", "spot", "clear", "settle",
                     "load", "负荷", "wind", "solar", "光伏", "风电"]
        text = (self.response_body or "") + (self.url or "")
        return any(kw in text.lower() for kw in keywords)


class XHRInterceptor:
    """拦截页面的所有XHR/Fetch请求"""

    def __init__(self):
        self.captured: list[CapturedAPI] = []
        self._listening = False

    async def attach(self, page: Page):
        if self._listening:
            return
        page.on("response", self._on_response)
        self._listening = True
        logger.info("XHR interceptor attached")

    async def _on_response(self, response: Response):
        url = response.url
        # 只捕获API调用，跳过静态资源
        if any(ext in url for ext in [".js", ".css", ".png", ".jpg", ".svg", ".woff", ".ico"]):
            return

        content_type = response.headers.get("content-type", "")
        if "json" not in content_type and "text" not in content_type:
            return

        try:
            body = await response.text()
        except Exception:
            body = None

        request = response.request
        try:
            req_body = request.post_data
        except Exception:
            req_body = None

        captured = CapturedAPI(
            url=url,
            method=request.method,
            request_headers=dict(request.headers),
            request_body=req_body,
            status=response.status,
            response_body=body[:5000] if body else None,  # 截断大响应
            content_type=content_type,
            timestamp=datetime.now().isoformat(),
        )
        self.captured.append(captured)

        # 高亮可能的价格数据API
        if captured.is_json() and captured.contains_price_data():
            logger.success("PRICE API FOUND: {} {} ({})", request.method, url, response.status)

    def get_price_apis(self) -> list[CapturedAPI]:
        return [c for c in self.captured if c.is_json() and c.contains_price_data()]

    def get_all_json_apis(self) -> list[CapturedAPI]:
        return [c for c in self.captured if c.is_json()]

    def save_report(self, filename: str = "api_discovery.json"):
        out_path = settings.data_dir / filename
        data = []
        for c in self.captured:
            if c.is_json():
                data.append({
                    "url": c.url,
                    "method": c.method,
                    "status": c.status,
                    "content_type": c.content_type,
                    "has_price_data": c.contains_price_data(),
                    "request_body": c.request_body,
                    "response_preview": (c.response_body or "")[:500],
                    "timestamp": c.timestamp,
                })
        out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        logger.info("Saved {} API calls to {}", len(data), out_path)
        return out_path

    def clear(self):
        self.captured.clear()
