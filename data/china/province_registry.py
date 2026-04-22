"""
省份元数据注册表

每个省的数据特征、指标名称、特殊处理规则集中在这里。
避免在代码各处散落if/else分支。
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class ProvinceSpec:
    name: str               # 英文标识
    name_cn: str            # 中文名
    # 指标名称 → 标准列名映射
    indicator_map: dict = field(default_factory=dict)
    # 数据起止（价格数据）
    price_start: str = ""
    price_end: str = ""
    # 能力标记
    has_wind: bool = True
    has_solar: bool = True
    has_renewable_total: bool = True
    # 特殊处理
    clip_solar_negative: bool = False
    has_regional_prices: bool = False
    sentinel_value: float = -9999.0


PROVINCES = {
    "shandong": ProvinceSpec(
        name="shandong",
        name_cn="山东",
        indicator_map={
            "实时价格": "rt_price",
            "日前价格": "da_price",
            "负荷": "load_mw",
            "风电出力": "wind_mw",
            "光伏出力": "solar_mw",
            "新能源总出力": "renewable_mw",
            "联络线": "tie_line_mw",
        },
        price_start="2020-11-01",
        price_end="2026-04-09",
        clip_solar_negative=True,  # 37.8%光伏负值需clip
    ),

    "shanxi": ProvinceSpec(
        name="shanxi",
        name_cn="山西",
        indicator_map={
            "实时价格": "rt_price",
            "日前价格": "da_price",
            "负荷": "load_mw",
            "风电出力": "wind_mw",
            "光伏出力": "solar_mw",
            "新能源总出力": "renewable_mw",
            "联络线": "tie_line_mw",
            "实时市场出清电量": "clearing_volume_mw",
        },
        price_start="2021-04-01",
        price_end="2026-04-09",
    ),

    "guangdong": ProvinceSpec(
        name="guangdong",
        name_cn="广东",
        indicator_map={
            "实时价格": "rt_price",
            "日前价格": "da_price",
            "负荷": "load_mw",
            "省内A类电源": "gen_class_a_mw",     # 核电+大火电
            "省内B类电源": "gen_class_b_mw",     # 燃气+新能源 → renewable proxy
            "地方电源出力": "local_gen_mw",
            "粤港联络线": "hk_tie_line_mw",
            "西电东送电力": "west_east_mw",       # → tie_line_mw proxy
        },
        price_start="2021-11-01",
        price_end="2026-04-03",
        has_wind=False,
        has_solar=False,
        has_renewable_total=False,
    ),

    "gansu": ProvinceSpec(
        name="gansu",
        name_cn="甘肃",
        indicator_map={
            "实时价格": "rt_price",
            "日前价格": "da_price",
            "负荷": "load_mw",
            "风电出力": "wind_mw",
            "光伏出力": "solar_mw",
            "新能源总出力": "renewable_mw",
            "联络线": "tie_line_mw",
            "河东实时价格": "rt_price_hedong",
            "河东日前价格": "da_price_hedong",
            "河西实时价格": "rt_price_hexi",
            "河西日前价格": "da_price_hexi",
            "水电总出力": "hydro_mw",
            "发电总出力": "total_gen_mw",
        },
        price_start="2021-04-02",
        price_end="2026-04-03",
        has_regional_prices=True,
        sentinel_value=-9999.0,
    ),
}


def get_province(name: str) -> ProvinceSpec:
    if name not in PROVINCES:
        raise ValueError(f"Unknown province: {name}. Available: {list(PROVINCES.keys())}")
    return PROVINCES[name]


def list_provinces() -> list[str]:
    return list(PROVINCES.keys())
