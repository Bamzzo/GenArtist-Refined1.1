#!/usr/bin/env python3
"""
阶段二重构验收脚本：进程绞杀与显存安全检查。

检查内容：
1. 子进程 & input.json 磁盘 I/O 根除；
2. agent_tool_* API 函数签名存在性；
3. AST 层面的 Module 级高危调用（.to("cuda") / .cuda() / .from_pretrained() / load_model() / load_sd()）拦截。

从项目根运行：python tools/sanity_check_p2.py
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import List, Tuple


ROOT = Path(__file__).resolve().parents[1]

DEMO_FILE = ROOT / "demo_t2i.py"
AGENT_FILES = [
    ROOT / "agent_tool_generate.py",
    ROOT / "agent_tool_aux.py",
    ROOT / "agent_tool_edit.py",
]


RED = "\033[91m"
GREEN = "\033[92m"
RESET = "\033[0m"


def _print_utf8_setup() -> None:
    """保证 Windows 控制台能输出 Unicode 和 ANSI 颜色。"""
    if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
        import io

        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


# -----------------------------------------------------------------------------
# 1. 子进程与 input.json 磁盘 I/O 检查
# -----------------------------------------------------------------------------

def check_no_subprocess_and_input_json() -> List[Tuple[Path, int, str]]:
    errors: List[Tuple[Path, int, str]] = []

    # 1) demo_t2i.py 不能有 os.system / subprocess.*
    demo_text = DEMO_FILE.read_text(encoding="utf-8", errors="replace")
    for lineno, line in enumerate(demo_text.splitlines(), start=1):
        if "os.system(" in line or "subprocess." in line:
            errors.append((DEMO_FILE, lineno, line.strip()))

    # 2) 主流程脚本不能对 input.json 做读写
    #    仅拦截真正 I/O 语句（包含 open/json.* 等），允许注释/文档中提到 input.json
    forbidden_snippet = "input.json"
    for path in [DEMO_FILE] + AGENT_FILES:
        text = path.read_text(encoding="utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if forbidden_snippet in line and ("open(" in line or "json." in line):
                errors.append((path, lineno, line.strip()))

    return errors


# -----------------------------------------------------------------------------
# 2. 接口连通性扫描 (API Signature Check)
# -----------------------------------------------------------------------------

def check_api_signatures() -> List[str]:
    """使用 AST 静态检查 run_*_tool 是否存在，避免真正导入模块。"""
    missing: List[str] = []

    expected = [
        (ROOT / "agent_tool_generate.py", "run_generate_tool"),
        (ROOT / "agent_tool_aux.py", "run_aux_tool"),
        (ROOT / "agent_tool_edit.py", "run_edit_tool"),
    ]

    for path, func_name in expected:
        src = path.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(src, filename=str(path))
        except SyntaxError as e:
            missing.append(f"{path}: 语法错误，无法找到 {func_name}: {e}")
            continue

        has_func = any(
            isinstance(node, ast.FunctionDef) and node.name == func_name
            for node in tree.body
        )
        if not has_func:
            missing.append(f"{path}: 未找到函数 {func_name}")

    return missing


# -----------------------------------------------------------------------------
# 3. 致命显存地雷扫描 (VRAM Bomb Check)
# -----------------------------------------------------------------------------

HIGH_RISK_ATTRS = {"to", "cuda", "from_pretrained", "load_model", "load_sd"}
HIGH_RISK_NAMES = {"load_model", "load_sd"}


def _iter_module_level_calls(tree: ast.Module):
    """遍历模块级（不进入函数/类内部）的 ast.Call 节点。"""

    def walk(node):
        # 不进入函数或类定义体（它们内部的调用是允许的）
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.Call):
                yield child
            # 继续深入，但仍然避开函数/类
            if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                yield from walk(child)

    for top in tree.body:
        # 只检查模块顶层语句
        if isinstance(top, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        yield from walk(top)


def _func_name_from_call(call: ast.Call) -> str:
    """获取调用的函数名（支持简单的 attr 链），用于日志。"""
    f = call.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        parts = []
        cur = f
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
        parts.reverse()
        return ".".join(parts)
    return ast.dump(f)


def _is_high_risk_call(call: ast.Call) -> bool:
    """判断一个 module-level Call 是否属于显存高危调用。"""
    func = call.func

    # 属性调用，如 obj.to(...), model.cuda(), Pipeline.from_pretrained(...)
    if isinstance(func, ast.Attribute):
        attr = func.attr
        if attr in HIGH_RISK_ATTRS:
            # 对 .to / .cuda 再看一下是否明显指向 cuda 设备
            if attr in {"to", "cuda"}:
                for arg in call.args:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        if "cuda" in arg.value.lower():
                            return True
                # .cuda() 无参时也视为高危
                if attr == "cuda":
                    return True
            else:
                # from_pretrained / load_model / load_sd 一律视为高危
                return True

    # 直接函数名调用，如 load_model(...), load_sd(...)
    if isinstance(func, ast.Name) and func.id in HIGH_RISK_NAMES:
        return True

    return False


def check_vram_bombs() -> List[Tuple[Path, int, str]]:
    errors: List[Tuple[Path, int, str]] = []

    for path in AGENT_FILES:
        src = path.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(src, filename=str(path))
        except SyntaxError as e:
            errors.append((path, e.lineno or 0, f"语法错误: {e}"))
            continue

        for call in _iter_module_level_calls(tree):
            if _is_high_risk_call(call):
                lineno = call.lineno
                fname = _func_name_from_call(call)
                errors.append((path, lineno, f"模块级高危调用: {fname}"))

    return errors


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------


def main() -> None:
    _print_utf8_setup()

    any_error = False

    # 1) 子进程 & input.json
    subproc_errors = check_no_subprocess_and_input_json()
    if subproc_errors:
        any_error = True
        print(f"{RED}[Subprocess & Disk I/O Check] 发现违规调用：{RESET}")
        for path, lineno, line in subproc_errors:
            print(f"  {RED}{path.name}:{lineno} -> {line[:100]}{RESET}")

    # 2) API 签名
    api_errors = check_api_signatures()
    if api_errors:
        any_error = True
        print(f"\n{RED}[API Signature Check] 发现缺失或异常：{RESET}")
        for msg in api_errors:
            print(f"  {RED}{msg}{RESET}")

    # 3) VRAM Bomb
    vram_errors = check_vram_bombs()
    if vram_errors:
        any_error = True
        print(f"\n{RED}[VRAM Bomb Check] 发现模块级高危调用：{RESET}")
        for path, lineno, desc in vram_errors:
            print(f"  {RED}{path.name}:{lineno} -> {desc}{RESET}")

    if any_error:
        print(f"\n{RED}阶段二验收未通过，请先修复上述问题后再重试。{RESET}")
        sys.exit(1)

    print(f"\n{GREEN}\u2705 阶段二重构：进程绞杀与显存安全验收彻底通过！{RESET}")


if __name__ == "__main__":
    main()

