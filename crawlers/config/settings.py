"""全局配置"""

from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


_PROJECT_ROOT = Path(__file__).parent.parent.parent


class CrawlerSettings(BaseSettings):
    model_config = {"env_prefix": "CRAWLER_"}

    # 路径
    project_root: Path = _PROJECT_ROOT
    data_dir: Path = _PROJECT_ROOT / "data" / "china" / "crawled"
    db_path: Path = _PROJECT_ROOT / "data" / "china" / "crawled" / "crawler.duckdb"
    cookie_dir: Path = _PROJECT_ROOT / "data" / "china" / "crawled" / "cookies"

    # 浏览器
    headless: bool = True
    browser_timeout: int = 30000  # ms

    # 频率控制
    min_request_interval: float = 3.0
    max_request_interval: float = 8.0
    burst_limit: int = 20
    burst_cooldown: float = 60.0
    daily_limit: int = 500

    # 重试
    max_retries: int = 3
    retry_delay: float = 10.0

    # 告警
    bark_url: str = ""
    dingtalk_webhook: str = ""

    def model_post_init(self, __context):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cookie_dir.mkdir(parents=True, exist_ok=True)


settings = CrawlerSettings()
