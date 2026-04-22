"""
22省电力现货市场注册表

扩展自 data/china/province_registry.py，增加爬虫所需字段。
所有省份差异封装在ProvinceSpec中，爬虫代码无if/else省份分支。
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class ProvinceSpec:
    name: str                # 英文标识
    name_cn: str             # 中文名
    grid: str = "sgcc"       # "sgcc" | "csg" | "mengxi"
    province_code: str = ""  # pmos URL省份代码
    # 指标名称 → 标准列名
    indicator_map: dict = field(default_factory=dict)
    # 数据源优先级
    source_priority: list = field(default_factory=lambda: ["dianchacha"])
    # 数据特征
    settlement_interval: int = 15   # 分钟 (山西为5)
    price_start: str = ""
    has_wind: bool = True
    has_solar: bool = True
    has_renewable_total: bool = True
    has_regional_prices: bool = False
    clip_solar_negative: bool = False
    sentinel_value: float = -9999.0


# 通用指标映射（大多数省份共享）
_COMMON_INDICATORS = {
    "实时价格": "rt_price",
    "日前价格": "da_price",
    "负荷": "load_mw",
    "风电出力": "wind_mw",
    "光伏出力": "solar_mw",
    "新能源总出力": "renewable_mw",
    "联络线": "tie_line_mw",
}


PROVINCES: dict[str, ProvinceSpec] = {
    # ============================================================
    # 第一批试点 (2017)
    # ============================================================
    "shandong": ProvinceSpec(
        name="shandong", name_cn="山东",
        grid="sgcc", province_code="sd",
        indicator_map={
            **_COMMON_INDICATORS,
        },
        price_start="2020-11-01",
        clip_solar_negative=True,
    ),

    "shanxi": ProvinceSpec(
        name="shanxi", name_cn="山西",
        grid="sgcc", province_code="sx",
        indicator_map={
            **_COMMON_INDICATORS,
            "实时市场出清电量": "clearing_volume_mw",
        },
        settlement_interval=5,  # 山西5分钟出清
        price_start="2021-04-01",
    ),

    "guangdong": ProvinceSpec(
        name="guangdong", name_cn="广东",
        grid="csg", province_code="gd",
        indicator_map={
            "实时价格": "rt_price",
            "日前价格": "da_price",
            "负荷": "load_mw",
            "省内A类电源": "gen_class_a_mw",
            "省内B类电源": "gen_class_b_mw",
            "地方电源出力": "local_gen_mw",
            "粤港联络线": "hk_tie_line_mw",
            "西电东送电力": "west_east_mw",
        },
        price_start="2021-11-01",
        has_wind=False, has_solar=False, has_renewable_total=False,
    ),

    "gansu": ProvinceSpec(
        name="gansu", name_cn="甘肃",
        grid="sgcc", province_code="gs",
        indicator_map={
            **_COMMON_INDICATORS,
            "河东实时价格": "rt_price_hedong",
            "河东日前价格": "da_price_hedong",
            "河西实时价格": "rt_price_hexi",
            "河西日前价格": "da_price_hexi",
            "水电总出力": "hydro_mw",
            "发电总出力": "total_gen_mw",
        },
        price_start="2021-04-02",
        has_regional_prices=True,
        sentinel_value=-9999.0,
    ),

    "zhejiang": ProvinceSpec(
        name="zhejiang", name_cn="浙江",
        grid="sgcc", province_code="zj",
        indicator_map={**_COMMON_INDICATORS},
        price_start="2022-01-01",
    ),

    "sichuan": ProvinceSpec(
        name="sichuan", name_cn="四川",
        grid="sgcc", province_code="sc",
        indicator_map={
            **_COMMON_INDICATORS,
            "水电总出力": "hydro_mw",
        },
        price_start="2022-01-01",
    ),

    "fujian": ProvinceSpec(
        name="fujian", name_cn="福建",
        grid="sgcc", province_code="fj",
        indicator_map={
            **_COMMON_INDICATORS,
            "核电出力": "nuclear_mw",
        },
        price_start="2022-06-01",
    ),

    "mengxi": ProvinceSpec(
        name="mengxi", name_cn="蒙西",
        grid="mengxi", province_code="nmg",
        indicator_map={**_COMMON_INDICATORS},
        source_priority=["dianchacha", "mengxi"],
        price_start="2025-02-01",
    ),

    # ============================================================
    # 第二批试点 (2020)
    # ============================================================
    "shanghai": ProvinceSpec(
        name="shanghai", name_cn="上海",
        grid="sgcc", province_code="sh",
        indicator_map={**_COMMON_INDICATORS},
        price_start="2023-01-01",
    ),

    "jiangsu": ProvinceSpec(
        name="jiangsu", name_cn="江苏",
        grid="sgcc", province_code="js",
        indicator_map={**_COMMON_INDICATORS},
        price_start="2023-06-01",
    ),

    "anhui": ProvinceSpec(
        name="anhui", name_cn="安徽",
        grid="sgcc", province_code="ah",
        indicator_map={**_COMMON_INDICATORS},
        price_start="2023-06-01",
    ),

    "liaoning": ProvinceSpec(
        name="liaoning", name_cn="辽宁",
        grid="sgcc", province_code="ln",
        indicator_map={**_COMMON_INDICATORS},
        price_start="2023-06-01",
    ),

    "henan": ProvinceSpec(
        name="henan", name_cn="河南",
        grid="sgcc", province_code="ha",
        indicator_map={**_COMMON_INDICATORS},
        price_start="2023-06-01",
    ),

    "hubei": ProvinceSpec(
        name="hubei", name_cn="湖北",
        grid="sgcc", province_code="hb",
        indicator_map={
            **_COMMON_INDICATORS,
            "水电总出力": "hydro_mw",
        },
        price_start="2023-06-01",
    ),

    # ============================================================
    # 后续加入 (2024-2025)
    # ============================================================
    "chongqing": ProvinceSpec(
        name="chongqing", name_cn="重庆",
        grid="sgcc", province_code="cq",
        indicator_map={**_COMMON_INDICATORS},
        price_start="2024-06-01",
    ),

    "hunan": ProvinceSpec(
        name="hunan", name_cn="湖南",
        grid="sgcc", province_code="hn",
        indicator_map={**_COMMON_INDICATORS},
        price_start="2024-06-01",
    ),

    "ningxia": ProvinceSpec(
        name="ningxia", name_cn="宁夏",
        grid="sgcc", province_code="nx",
        indicator_map={**_COMMON_INDICATORS},
        price_start="2024-06-01",
    ),

    "jilin": ProvinceSpec(
        name="jilin", name_cn="吉林",
        grid="sgcc", province_code="jl",
        indicator_map={**_COMMON_INDICATORS},
        price_start="2024-12-01",
    ),

    "heilongjiang": ProvinceSpec(
        name="heilongjiang", name_cn="黑龙江",
        grid="sgcc", province_code="hlj",
        indicator_map={**_COMMON_INDICATORS},
        price_start="2024-12-01",
    ),

    "xinjiang": ProvinceSpec(
        name="xinjiang", name_cn="新疆",
        grid="sgcc", province_code="xj",
        indicator_map={**_COMMON_INDICATORS},
        price_start="2025-01-01",
    ),

    "qinghai": ProvinceSpec(
        name="qinghai", name_cn="青海",
        grid="sgcc", province_code="qh",
        indicator_map={
            **_COMMON_INDICATORS,
            "水电总出力": "hydro_mw",
        },
        price_start="2025-01-01",
    ),

    "jiangxi": ProvinceSpec(
        name="jiangxi", name_cn="江西",
        grid="sgcc", province_code="jx",
        indicator_map={**_COMMON_INDICATORS},
        price_start="2025-01-01",
    ),
}


def get_province(name: str) -> ProvinceSpec:
    if name not in PROVINCES:
        available = ", ".join(sorted(PROVINCES.keys()))
        raise ValueError(f"Unknown province: {name}. Available: {available}")
    return PROVINCES[name]


def list_provinces() -> list[str]:
    return sorted(PROVINCES.keys())


def list_by_grid(grid: str) -> list[str]:
    return [k for k, v in PROVINCES.items() if v.grid == grid]
