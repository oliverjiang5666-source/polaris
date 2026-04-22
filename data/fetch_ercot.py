"""
ERCOT Public API 数据拉取

直接调用 ERCOT ESR API，用subscription key认证。
数据产品：
  NP4-190-CD = DAM Settlement Point Prices（日前，小时级）
  NP6-905-CD = RTM Settlement Point Prices（实时，15分钟级）
"""
import os
import io
import zipfile
import requests
import pandas as pd
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://api.ercot.com/api/public-data"
API_KEY = os.getenv("ERCOT_API_KEY", "")

PRODUCTS = {
    "dam_spp": "NP4-190-CD",   # Day-Ahead Market SPP
    "rtm_spp": "NP6-905-CD",   # Real-Time Market SPP
}


def _headers():
    return {"Ocp-Apim-Subscription-Key": API_KEY}


def list_products():
    """列出所有可用数据产品"""
    r = requests.get(BASE_URL, headers=_headers(), timeout=30)
    r.raise_for_status()
    return r.json()


def get_product_info(product_id: str):
    """获取产品详细信息"""
    r = requests.get(f"{BASE_URL}/{product_id}", headers=_headers(), timeout=30)
    r.raise_for_status()
    return r.json()


def list_archives(product_id: str, page: int = 1, size: int = 50):
    """列出产品的历史存档文件"""
    r = requests.get(
        f"{BASE_URL}/archive/{product_id}",
        headers=_headers(),
        params={"page": page, "size": size},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def download_archive(product_id: str, doc_ids: list[int], save_dir: Path) -> list[Path]:
    """下载存档zip文件并解压"""
    save_dir.mkdir(parents=True, exist_ok=True)

    r = requests.post(
        f"{BASE_URL}/archive/{product_id}/download",
        headers={**_headers(), "Content-Type": "application/json"},
        json={"docIds": doc_ids},
        timeout=120,
    )
    r.raise_for_status()

    extracted = []
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        for name in z.namelist():
            z.extract(name, save_dir)
            extracted.append(save_dir / name)
            logger.info(f"  Extracted: {name}")

    return extracted


def fetch_dam_spp() -> pd.DataFrame:
    """拉取DAM Settlement Point Prices"""
    product_id = PRODUCTS["dam_spp"]
    logger.info(f"Fetching DAM SPP archives for {product_id}...")

    archives = list_archives(product_id)
    logger.info(f"Found {len(archives.get('archives', []))} archive files")

    return archives


def fetch_rtm_spp() -> pd.DataFrame:
    """拉取RTM Settlement Point Prices"""
    product_id = PRODUCTS["rtm_spp"]
    logger.info(f"Fetching RTM SPP archives for {product_id}...")

    archives = list_archives(product_id)
    logger.info(f"Found {len(archives.get('archives', []))} archive files")

    return archives
