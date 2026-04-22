#!/usr/bin/env python3
"""
电查查VIP全量爬取脚本
运行: python3 crawl_all_provinces.py

1. 打开浏览器让你登录（如果需要）
2. 抓取token
3. 22省×8指标全量爬取（用AES解密）
4. 每省存一个CSV到 data/china/crawled/{province}/
"""
import asyncio, json, os, csv, sys, time
from datetime import datetime, date, timedelta
from pathlib import Path
from playwright.async_api import async_playwright

# 加载解密模块
sys.path.insert(0, str(Path(__file__).parent))
from crawlers.sources.crypto import decrypt_response, refresh_key_from_js

BASE_URL = "https://www.dianchacha.cn"
API = "/electric/api/v2/trade/power/data/spot"
OUT_DIR = Path("data/china/crawled")
COOKIE_PATH = OUT_DIR / "cookies" / "dianchacha.json"

# 22省regionId
REGIONS = {
    "shandong": (370000, "山东"), "shanxi": (140000, "山西"),
    "guangdong": (440000, "广东"), "gansu": (620000, "甘肃"),
    "zhejiang": (330000, "浙江"), "sichuan": (510000, "四川"),
    "fujian": (350000, "福建"), "mengxi": (150000, "蒙西"),
    "shanghai": (310000, "上海"), "jiangsu": (320000, "江苏"),
    "anhui": (340000, "安徽"), "liaoning": (210000, "辽宁"),
    "henan": (410000, "河南"), "hubei": (420000, "湖北"),
    "chongqing": (500000, "重庆"), "hunan": (430000, "湖南"),
    "ningxia": (640000, "宁夏"), "jilin": (220000, "吉林"),
    "heilongjiang": (230000, "黑龙江"), "xinjiang": (650000, "新疆"),
    "qinghai": (630000, "青海"), "jiangxi": (360000, "江西"),
}

# 指标typeId
INDICATORS = {
    "日前价格": 21, "实时价格": 22, "负荷": 23, "联络线": 24,
    "风电出力": 28, "光伏出力": 29, "新能源总出力": 30, "充放价格": 55354,
}


async def fetch_encrypted(page, token, region_id, type_id, start_str, end_str):
    """在浏览器内fetch API，返回加密数据"""
    url = f"{API}?&regionId={region_id}&typeId={type_id}&timeInterval=15&start={start_str}&end={end_str}&watershed={end_str}"
    result = await page.evaluate("""
    async (args) => {
        let [token, url] = args;
        try {
            let r = await fetch(url, {
                headers: {'JWTHeaderName': token, 'Authorization': token, 'Accept': 'application/json'}
            });
            let d = await r.json();
            return {status: d.status, data: d.data, message: d.message};
        } catch(e) {
            return {error: e.message};
        }
    }
    """, [token, url])
    return result


async def main():
    # 刷新AES密钥
    print("刷新AES密钥...")
    await refresh_key_from_js()

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        ctx = await browser.new_context(
            viewport={"width": 1280, "height": 900},
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            locale="zh-CN",
        )
        page = await ctx.new_page()

        # 加载cookie
        if COOKIE_PATH.exists():
            cookies = json.loads(COOKIE_PATH.read_text())
            await ctx.add_cookies(cookies)

        await page.goto(f"{BASE_URL}/transaction", wait_until="networkidle", timeout=30000)

        # 检查登录状态
        text = await page.evaluate("() => document.body?.innerText?.slice(0,200) || ''")
        if "登录" in text and "VIP" not in text:
            print("\n请在浏览器中登录电查查，登录后按Enter...")
            input()

        # 保存cookie
        cookies = await ctx.cookies()
        COOKIE_PATH.parent.mkdir(parents=True, exist_ok=True)
        COOKIE_PATH.write_text(json.dumps(cookies, ensure_ascii=False, indent=2))

        # 获取token
        token = await page.evaluate("""
        () => { let t = JSON.parse(localStorage.getItem('token')); return t ? t.value : ''; }
        """)
        if not token:
            print("ERROR: 无法获取token，请确认已登录")
            await browser.close()
            return
        print(f"Token OK (len={len(token)})")

        # 验证
        print("\n验证VIP数据访问...")
        test = await fetch_encrypted(page, token, 370000, 21, "20260408000000", "20260408235959")
        if test.get("status") == 200 and test.get("data"):
            dec = decrypt_response(test["data"])
            vals = dec.get("values", []) if isinstance(dec, dict) else []
            print(f"✓ 验证通过: {len(vals)}条数据")
        else:
            print(f"✗ 验证失败: {test}")
            await browser.close()
            return

        # ============================================================
        # 全量爬取22省
        # ============================================================
        print(f"\n{'='*60}")
        print(f"  开始22省全量爬取")
        print(f"  VIP到期: 2026-04-16，抓紧时间！")
        print(f"{'='*60}\n")

        for prov_name, (region_id, cn_name) in REGIONS.items():
            prov_dir = OUT_DIR / prov_name
            prov_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n--- {cn_name} ({prov_name}, regionId={region_id}) ---")

            for indicator, type_id in INDICATORS.items():
                # 请求最近30天数据（电查查默认范围）
                end_dt = datetime.now()
                start_dt = end_dt - timedelta(days=30)
                start_str = start_dt.strftime("%Y%m%d000000")
                end_str = end_dt.strftime("%Y%m%d235959")

                result = await fetch_encrypted(page, token, region_id, type_id, start_str, end_str)

                if result.get("status") != 200 or not result.get("data"):
                    print(f"  ✗ {indicator}: 无数据")
                    continue

                try:
                    dec = decrypt_response(result["data"])
                    vals = dec.get("values", []) if isinstance(dec, dict) else []
                    if not vals:
                        print(f"  ? {indicator}: 解密OK但values空")
                        continue

                    # 保存
                    csv_path = prov_dir / f"{prov_name}_{indicator.replace('/','_')}.csv"
                    with open(csv_path, "w", newline="", encoding="utf-8") as f:
                        if vals:
                            writer = csv.DictWriter(f, fieldnames=vals[0].keys())
                            writer.writeheader()
                            writer.writerows(vals)

                    print(f"  ✓ {indicator}: {len(vals)}条 → {csv_path.name}")

                except Exception as e:
                    print(f"  ✗ {indicator}: 解密失败 {e}")

                await asyncio.sleep(1.5)  # 低调

            # 省间休息
            await asyncio.sleep(2)

        # 生成总结
        print(f"\n{'='*60}")
        print("  爬取完成！")
        print(f"{'='*60}")

        for prov_name in REGIONS:
            prov_dir = OUT_DIR / prov_name
            csvs = list(prov_dir.glob("*.csv"))
            if csvs:
                total = sum(sum(1 for _ in open(c)) - 1 for c in csvs)
                print(f"  {prov_name}: {len(csvs)}个文件, {total}条")
            else:
                print(f"  {prov_name}: 无数据")

        await browser.close()
        print("\n完成！浏览器已关闭。")

asyncio.run(main())
