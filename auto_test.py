#!/usr/bin/env python3
"""
全自动开炮脚本：向生图 API 发送 POST，打印状态码与响应/报错，用于自愈闭环。
"""
import json
import sys

try:
    import urllib.request
    req = urllib.request.Request(
        "http://127.0.0.1:8000/api/v1/generate_direct",
        data=json.dumps({"prompt": "cat"}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        body = resp.read().decode("utf-8")
        code = resp.getcode()
        print(f"HTTP {code}")
        print(body)
        try:
            data = json.loads(body)
            if data.get("status") == "success" and data.get("image_url"):
                print("\n[OK] 生图成功:", data.get("image_url"))
                sys.exit(0)
        except Exception:
            pass
        sys.exit(0 if 200 <= code < 300 else 1)
except urllib.error.HTTPError as e:
    print(f"HTTP {e.code}")
    print(e.read().decode("utf-8"))
    sys.exit(1)
except Exception as e:
    print("Request failed:", type(e).__name__, str(e))
    sys.exit(1)
