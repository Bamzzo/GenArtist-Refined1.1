#!/usr/bin/env python3
"""
阶段一重构验收脚本：环境与变量测试 + 硬编码残留扫描。
不加载任何大模型，仅做拦截与检查。
从项目根运行: python tools/sanity_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# 确保项目根在 path 中以便导入 config
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ---------------------------------------------------------------------------
# 环境与变量测试 (Env Check)
# ---------------------------------------------------------------------------

def mask_api_key(key: str) -> str:
    if not key:
        return "(未设置)"
    if len(key) <= 7:
        return "****"
    return f"{key[:3]}****{key[-4:]}"


def run_env_check() -> None:
    from config import OPENAI_API_KEY, MODEL_ZOO_DIR, WORK_DIR

    print("[Env Check] 加载 config 成功")

    # OPENAI_API_KEY 脱敏
    masked = mask_api_key(OPENAI_API_KEY)
    print(f"  OPENAI_API_KEY: {masked}")

    # MODEL_ZOO_DIR / WORK_DIR 解析与存在性
    for name, p in [("MODEL_ZOO_DIR", MODEL_ZOO_DIR), ("WORK_DIR", WORK_DIR)]:
        resolved = Path(p).resolve()
        if not resolved.is_absolute():
            raise AssertionError(f"{name} 未解析为绝对路径: {resolved}")
        if not resolved.exists():
            raise AssertionError(f"{name} 目录不存在: {resolved}")
        if not resolved.is_dir():
            raise AssertionError(f"{name} 不是目录: {resolved}")
        print(f"  {name}: {resolved} [存在]")


# ---------------------------------------------------------------------------
# 硬编码残留扫描 (Hardcode Scanner)
# ---------------------------------------------------------------------------

# 禁止出现的字面片段（出现即视为漏改）
FORBIDDEN_LITERALS = [
    "###################################",
    "./diffusers-generation-text-box",
    "./Inpaint-Anything/pretrained_models/big-lama",
    "./instruct-pix2pix/checkpoints/MagicBrush-epoch-000168.ckpt",
]

# 若行内包含此片段且不包含 MODEL_ZOO_DIR，则视为漏改
CONDITIONAL_FORBIDDEN = "GroundingDINO/weights/groundingdino_swint_ogc.pth"

TARGET_FILES = [
    "demo_t2i.py",
    "agent_tool_generate.py",
    "agent_tool_aux.py",
    "agent_tool_edit.py",
]


def run_hardcode_scan() -> list[tuple[str, int, str]]:
    errors: list[tuple[str, int, str]] = []

    for rel_path in TARGET_FILES:
        path = _ROOT / rel_path
        if not path.exists():
            errors.append((rel_path, 0, f"文件不存在: {path}"))
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for line_no, line in enumerate(text.splitlines(), start=1):
            # 禁止字面
            for bad in FORBIDDEN_LITERALS:
                if bad in line:
                    errors.append((rel_path, line_no, line.strip()))
                    break
            # 条件禁止：GroundingDINO 权重路径未与 MODEL_ZOO_DIR 一起出现
            if CONDITIONAL_FORBIDDEN in line and "MODEL_ZOO_DIR" not in line:
                errors.append((rel_path, line_no, line.strip()))

    return errors


# ---------------------------------------------------------------------------
# 输出与主流程
# ---------------------------------------------------------------------------

RED = "\033[91m"
GREEN = "\033[92m"
RESET = "\033[0m"


def main() -> None:
    # 保证 Windows 控制台能正确显示颜色与 Unicode（如 ✅）
    if sys.platform == "win32":
        import io
        if hasattr(sys.stdout, "buffer"):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    failed = False

    # 1) Env Check
    try:
        run_env_check()
    except Exception as e:
        print(f"{RED}[Env Check] 失败: {e}{RESET}")
        failed = True

    # 2) Hardcode Scanner
    errors = run_hardcode_scan()
    if errors:
        failed = True
        print(f"\n{RED}[Hardcode Scanner] 发现硬编码残留，请修复：{RESET}")
        for path, line_no, content in errors:
            print(f"  {RED}  {path}:{line_no}  ->  {content[:80]}{'...' if len(content) > 80 else ''}{RESET}")
        print(f"\n{RED}请移除上述硬编码路径/占位符，改为使用 config.MODEL_ZOO_DIR / config 等配置。{RESET}")

    if failed:
        sys.exit(1)

    print(f"\n{GREEN}\u2705 阶段一重构：资产劫持与沙盒化验收彻底通过！{RESET}")


if __name__ == "__main__":
    main()
