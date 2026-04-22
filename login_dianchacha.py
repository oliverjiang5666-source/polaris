"""
电查查登录脚本 — 终端运行: python3 login_dianchacha.py
"""
import asyncio
import json
from playwright.async_api import async_playwright

COOKIE_PATH = "data/china/crawled/cookies/dianchacha.json"
TOKEN_PATH = "data/china/crawled/cookies/dianchacha_token.json"

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        ctx = await browser.new_context(
            viewport={"width": 1280, "height": 900},
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            locale="zh-CN",
        )
        page = await ctx.new_page()
        await page.goto("https://www.dianchacha.cn/", wait_until="networkidle", timeout=30000)

        print()
        print("=" * 60)
        print("  浏览器已打开电查查")
        print("  1. 点右上角「登录」")
        print("  2. 输入手机号 → 获取验证码 → 登录")
        print("  3. 看到页面变成已登录状态后")
        print("  4. 回到这里按 Enter")
        print("=" * 60)
        input("\n按 Enter 继续...")

        # 1. 保存cookie
        cookies = await ctx.cookies()
        with open(COOKIE_PATH, "w") as f:
            json.dump(cookies, f, ensure_ascii=False, indent=2)
        print(f"Cookie: {len(cookies)} 个")

        # 2. 关键：提取localStorage中的token
        storage = await page.evaluate("""
        () => {
            let result = {};
            for (let i = 0; i < localStorage.length; i++) {
                let key = localStorage.key(i);
                result[key] = localStorage.getItem(key);
            }
            return result;
        }
        """)
        with open(TOKEN_PATH, "w") as f:
            json.dump(storage, f, ensure_ascii=False, indent=2)
        print(f"localStorage: {len(storage)} 项")
        for k, v in storage.items():
            preview = v[:80] if len(v) < 80 else v[:77] + "..."
            print(f"  {k}: {preview}")

        # 3. 找token，检查API请求头
        print("\n=== 拦截API请求查看认证头 ===")
        captured_headers = []
        async def on_req(request):
            if "electric" in request.url and "api" in request.url:
                captured_headers.append({
                    "url": request.url,
                    "headers": dict(request.headers),
                })
        page.on("request", on_req)

        # 导航到交易数据页触发API
        await page.goto("https://www.dianchacha.cn/transaction", wait_until="networkidle", timeout=20000)
        await page.wait_for_timeout(2000)

        print(f"捕获 {len(captured_headers)} 个API请求")
        for req in captured_headers[:3]:
            print(f"\n  URL: {req['url'][:100]}")
            for k, v in req["headers"].items():
                if k.lower() in ["authorization", "token", "x-token", "x-access-token",
                                  "bearer", "cookie", "x-auth", "access-token"]:
                    print(f"  AUTH: {k} = {v[:80]}")
                if "token" in k.lower() or "auth" in k.lower():
                    print(f"  {k} = {v[:80]}")

        # 4. 用发现的认证方式请求数据
        print("\n=== 验证数据访问 ===")
        from crawlers.sources.crypto import decrypt_response

        tests = [
            (370000, 21, "山东日前价格"),
            (370000, 22, "山东实时价格"),
            (140000, 21, "山西日前价格"),
            (440000, 21, "广东日前价格"),
            (620000, 21, "甘肃日前价格"),
        ]
        for rid, tid, name in tests:
            result = await page.evaluate(
                """async (url) => {
                    try { let r = await fetch(url); return await r.json(); }
                    catch(e) { return {error: e.message}; }
                }""",
                f"/electric/api/v2/visitor/trade/basic?regionId={rid}&typeId={tid}&timeDimension=MINUTE",
            )
            data = result.get("data") if result else None
            if data and isinstance(data, str) and len(data) > 50:
                dec = decrypt_response(data)
                vals = dec.get("values", []) if isinstance(dec, dict) else []
                if vals:
                    print(f"  ✓ {name}: {len(vals)}条  最新={vals[0].get('dataTime','')}")
                else:
                    # 检查是否有msg
                    msg = dec.get("msg", "") if isinstance(dec, dict) else ""
                    print(f"  ? {name}: values空 msg={msg}")
            else:
                status = result.get("status") if result else None
                msg = result.get("message", "") if result else ""
                print(f"  ✗ {name}: status={status} {msg}")

        await browser.close()
        print("\n完成！")

asyncio.run(main())
