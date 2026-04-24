"""
开VPN后运行此脚本。
自动测试ERCOT API → 列出数据产品 → 下载DAM和RTM电价数据。

用法：
  cd ~/Desktop/energy-storage-rl
  .venv/bin/python scripts/00_test_and_download.py
"""
import sys, os, io, json, zipfile, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
import pandas as pd

API_KEY = os.environ.get("ERCOT_API_KEY", "")
if not API_KEY:
    raise RuntimeError("请设置 ERCOT_API_KEY 环境变量（见 .env.example）")
# 两个API都试
BASE_PUBLIC = "https://api.ercot.com/api/public-reports"
BASE_ESR = "https://api.ercot.com/api/public-data"
BASE = BASE_PUBLIC  # 默认用public-reports
HEADERS = {"Ocp-Apim-Subscription-Key": API_KEY}
RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def test_connection():
    print("=" * 60)
    print("Step 1: 测试API连接")
    print("=" * 60)
    try:
        r = requests.get(BASE, headers=HEADERS, timeout=30)
        print(f"  Status: {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, list):
                print(f"  ✅ 连接成功！共 {len(data)} 个数据产品")
                # 找到我们需要的产品
                for p in data:
                    pid = p.get("productId", "")
                    name = p.get("name", "")
                    if "NP4-190" in pid or "NP6-905" in pid or "SPP" in name.upper():
                        print(f"  📌 {pid}: {name}")
            else:
                print(f"  Response: {str(data)[:300]}")
            return True
        elif r.status_code == 403:
            print(f"  ❌ 403 Forbidden — VPN没开或API key无效")
            print(f"  Response: {r.text[:300]}")
            return False
        else:
            print(f"  ❌ {r.status_code}: {r.text[:300]}")
            return False
    except Exception as e:
        print(f"  ❌ 连接失败: {e}")
        return False


def list_and_save_archives(product_id: str, name: str):
    print(f"\n{'=' * 60}")
    print(f"Step 2: 列出 {name} ({product_id}) 的存档")
    print("=" * 60)
    try:
        r = requests.get(
            f"{BASE}/archive/{product_id}",
            headers=HEADERS,
            params={"page": 1, "size": 100},
            timeout=30,
        )
        print(f"  Status: {r.status_code}")
        if r.status_code != 200:
            print(f"  Error: {r.text[:300]}")
            return []

        data = r.json()
        archives = data.get("archives", [])
        print(f"  找到 {len(archives)} 个存档文件")

        # 保存存档列表
        list_path = RAW_DIR / f"{product_id}_archives.json"
        with open(list_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  存档列表已保存: {list_path}")

        for a in archives[:10]:
            print(f"    docId={a.get('docId')}, {a.get('friendlyName', '')}, {a.get('postDatetime', '')}")
        if len(archives) > 10:
            print(f"    ... 还有 {len(archives) - 10} 个")

        return archives

    except Exception as e:
        print(f"  ❌ 失败: {e}")
        return []


def download_archives(product_id: str, archives: list, max_files: int = 5):
    """下载前max_files个存档（最新的）"""
    print(f"\n{'=' * 60}")
    print(f"Step 3: 下载 {product_id} 存档 (最多{max_files}个)")
    print("=" * 60)

    save_dir = RAW_DIR / product_id
    save_dir.mkdir(parents=True, exist_ok=True)

    # 取最新的几个
    to_download = archives[:max_files]

    for i, archive in enumerate(to_download):
        doc_id = archive.get("docId")
        name = archive.get("friendlyName", f"doc_{doc_id}")
        print(f"  [{i+1}/{len(to_download)}] 下载 {name} (docId={doc_id})...")

        try:
            r = requests.post(
                f"{BASE}/archive/{product_id}/download",
                headers={**HEADERS, "Content-Type": "application/json"},
                json={"docIds": [doc_id]},
                timeout=120,
            )
            if r.status_code == 200:
                # 判断是zip还是csv
                content_type = r.headers.get("content-type", "")
                if "zip" in content_type or r.content[:4] == b"PK\x03\x04":
                    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                        for zname in z.namelist():
                            z.extract(zname, save_dir)
                            print(f"    ✅ 解压: {zname}")
                else:
                    # 直接保存
                    fpath = save_dir / f"{name}.csv"
                    fpath.write_bytes(r.content)
                    print(f"    ✅ 保存: {fpath.name}")
            else:
                print(f"    ❌ {r.status_code}: {r.text[:200]}")

            time.sleep(1)  # 限速

        except Exception as e:
            print(f"    ❌ 失败: {e}")

    # 尝试读取下载的文件
    print(f"\n  检查下载的文件:")
    csv_files = list(save_dir.glob("*.csv"))
    for f in csv_files[:3]:
        try:
            df = pd.read_csv(f)
            print(f"    {f.name}: {len(df)} rows, columns={list(df.columns)[:5]}")
        except:
            print(f"    {f.name}: 无法解析")


def try_direct_query(product_id: str):
    """尝试直接查询数据（不下载文件）"""
    print(f"\n{'=' * 60}")
    print(f"Step 4: 尝试直接查询 {product_id}")
    print("=" * 60)

    # 有些产品支持直接查询
    try:
        r = requests.get(
            f"{BASE}/{product_id}",
            headers=HEADERS,
            timeout=30,
        )
        print(f"  Status: {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            # 查看artifacts（子报告）
            artifacts = data.get("artifacts", [])
            print(f"  Artifacts: {len(artifacts)}")
            for a in artifacts:
                print(f"    {a.get('displayName', 'unknown')}")
                links = a.get("links", [])
                for link in links:
                    print(f"      {link.get('rel', '')}: {link.get('href', '')}")
    except Exception as e:
        print(f"  ❌ 失败: {e}")


def try_bundle_download(product_id: str):
    """尝试下载bundle（年度打包数据）"""
    print(f"\n{'=' * 60}")
    print(f"Step 5: 查找bundle（年度打包）{product_id}")
    print("=" * 60)
    try:
        r = requests.get(
            f"{BASE}/bundle/{product_id}",
            headers=HEADERS,
            params={"page": 1, "size": 20},
            timeout=30,
        )
        print(f"  Status: {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            bundles = data.get("bundles", data.get("archives", []))
            print(f"  找到 {len(bundles)} 个bundle")
            for b in bundles[:5]:
                print(f"    docId={b.get('docId')}, {b.get('friendlyName', '')}")

            # 下载第一个bundle试试
            if bundles:
                doc_id = bundles[0].get("docId")
                fname = bundles[0].get("friendlyName", "bundle")
                print(f"\n  下载bundle: {fname} (docId={doc_id})...")

                save_dir = RAW_DIR / f"{product_id}_bundles"
                save_dir.mkdir(parents=True, exist_ok=True)

                r2 = requests.post(
                    f"{BASE}/bundle/{product_id}/download",
                    headers={**HEADERS, "Content-Type": "application/json"},
                    json={"docIds": [doc_id]},
                    timeout=300,
                )
                if r2.status_code == 200:
                    if r2.content[:4] == b"PK\x03\x04":
                        with zipfile.ZipFile(io.BytesIO(r2.content)) as z:
                            for zname in z.namelist():
                                z.extract(zname, save_dir)
                                fsize = (save_dir / zname).stat().st_size
                                print(f"    ✅ 解压: {zname} ({fsize/1024:.0f}KB)")
                    else:
                        fpath = save_dir / f"{fname}"
                        fpath.write_bytes(r2.content)
                        print(f"    ✅ 保存: {fpath} ({len(r2.content)/1024:.0f}KB)")
                else:
                    print(f"    ❌ {r2.status_code}: {r2.text[:200]}")
        else:
            print(f"  {r.text[:300]}")
    except Exception as e:
        print(f"  ❌ 失败: {e}")


def main():
    print("\n🔌 ERCOT Data Fetch Script")
    print("确保已开启VPN（美国节点）\n")

    # Step 0: 尝试两个API base URL
    global BASE
    print("尝试 public-reports API...")
    if not test_connection():
        print("\npublic-reports失败，尝试 public-data API...")
        BASE = BASE_ESR
        HEADERS["Ocp-Apim-Subscription-Key"] = API_KEY
        if not test_connection():
            print("\n⚠️  两个API都连接失败。请检查：")
            print("  1. VPN是否已开启并连接到美国节点")
            print("  2. 登录 https://data.ercot.com → Profile → Subscriptions → 复制Primary Key")
            print("  3. 把key粘贴到 .env 文件的 ERCOT_API_KEY= 后面")
            print(f"  当前使用的key: {API_KEY[:8]}...")
            return

    # Step 2+3: DAM电价
    dam_id = "NP4-190-CD"
    try_direct_query(dam_id)
    dam_archives = list_and_save_archives(dam_id, "DAM SPP")
    if dam_archives:
        download_archives(dam_id, dam_archives, max_files=3)
    try_bundle_download(dam_id)

    # Step 2+3: RTM电价
    rtm_id = "NP6-905-CD"
    try_direct_query(rtm_id)
    rtm_archives = list_and_save_archives(rtm_id, "RTM SPP")
    if rtm_archives:
        download_archives(rtm_id, rtm_archives, max_files=3)
    try_bundle_download(rtm_id)

    print(f"\n{'=' * 60}")
    print("完成！检查 data/raw/ 目录下的文件")
    print("=" * 60)
    for f in sorted(RAW_DIR.rglob("*")):
        if f.is_file() and not f.name.startswith("."):
            print(f"  {f.relative_to(RAW_DIR)} ({f.stat().st_size / 1024:.0f}KB)")


if __name__ == "__main__":
    main()
