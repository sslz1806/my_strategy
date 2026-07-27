"""
Claude Agent SDK 通用封装。

提供 inject_config() 和 run_agent() 两个通用函数，可在项目中任意位置 import 使用。

Usage:
    # 方式一：把 agent/ 加入 sys.path（推荐，任何位置都能用）
    import sys
    sys.path.insert(0, r"D:\Admin\Desktop\project\stats_quant_fund\agent")
    from cc_sdk import inject_config, run_agent

    # 方式二：从 agent 包导入（需要在 agent/__init__.py 存在时）
    from agent.cc_sdk import inject_config, run_agent

    inject_config()
    result = run_agent("分析代码", ...)

路径约定:
    - 项目根目录由本文件的 __file__ 推导（agent/ 的父目录），不受 import 位置影响
    - cwd 默认指向 <项目根>/.sdk_sessions/（SDK session 隔离目录）
    - add_dirs 默认包含项目根目录（只读，Write/Edit 天然无法写入）
"""

import os
import json
import asyncio
from pathlib import Path

from claude_agent_sdk import (
    query,
    ClaudeAgentOptions,
    AssistantMessage,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
)

# ============================================================
# 路径常量（基于本文件位置，import 路径不影响）
# ============================================================
_THIS_FILE = Path(__file__).resolve()
# cc_sdk.py 在 agent/ 子目录下，项目根在父目录
_PROJECT_ROOT = _THIS_FILE.parent.parent
_SDK_SESSIONS_DIR = _PROJECT_ROOT / ".sdk_sessions"


# ============================================================
# 1. inject_config — API 配置注入
# ============================================================
def inject_config() -> None:
    """
    从 ~/.claude/settings.json 读取 API 配置并注入 os.environ。

    读取 settings.json 中 env 段落的 ANTHROPIC_* / CLAUDE_CODE_* 变量，
    缺失的变量用内置默认值补充。

    可通过设置环境变量覆盖默认值:
        ANTHROPIC_BASE_URL, ANTHROPIC_AUTH_TOKEN, ANTHROPIC_MODEL
    """
    settings_path = Path.home() / ".claude" / "settings.json"
    if settings_path.exists():
        with open(settings_path, encoding="utf-8") as f:
            env_vars = json.load(f).get("env", {})
        for key, value in env_vars.items():
            if key.startswith("ANTHROPIC") or key.startswith("CLAUDE_CODE"):
                os.environ[key] = os.path.expandvars(value)

    # 内置默认值（不会被已有环境变量覆盖）
    defaults = {
        "ANTHROPIC_BASE_URL": "http://192.168.1.67:9443/anthropic",
        "ANTHROPIC_AUTH_TOKEN": "c5930172761e6415e2d2e1ddc5f74108",
        "ANTHROPIC_MODEL": "deepseek-v4-pro",
    }
    for k, v in defaults.items():
        if k not in os.environ:
            os.environ[k] = v


# ============================================================
# 2. run_agent — Agent SDK 查询封装
# ============================================================
def run_agent(
    prompt: str,
    *,
    model: str = "deepseek-v4-pro",
    system_prompt: str = "",
    max_turns: int | None = None,
    allowed_tools: list[str] | None = None,
    disallowed_tools: list[str] | None = None,
    permission_mode: str = "bypassPermissions",
    tools: list[str] | None = None,
    cwd: str | None = None,
    add_dirs: list[str] | None = None,
    mcp_servers: dict | None = None,
    effort: str = "max",
    skills: list[str] | str | None = None,
    plugins: list[dict] | None = None,
    thinking: dict | None = None,
) -> dict:
    """
    在独立事件循环中运行 Agent SDK 查询。

    Agent SDK 内部使用 anyio.open_process 启动 CLI 子进程。
    Jupyter 的嵌套事件循环与 anyio 在 Windows 上不兼容，
    因此在独立线程中用 asyncio.run() 创建全新事件循环避开。

    默认行为:
        - cwd = <项目根>/.sdk_sessions/（与手动 Claude Code session 隔离）
        - add_dirs = [项目根目录]（只读，Write/Edit 受限于 cwd）
        - disallowed_tools = []（全部放行）
        - effort = "max"（最高推理力度）

    Args:
        prompt: 用户提问。
        model: 模型名称，默认 "deepseek-v4-pro"。
        system_prompt: 系统提示，默认空。
        max_turns: 最大对话轮次，None 表示不限制。
        allowed_tools: 额外自动许可的工具列表。
        disallowed_tools: 禁用的工具列表，默认 []（全部放行）。
            Write/Edit 天然受限于 cwd，add_dirs 只读。
        permission_mode: 权限模式，默认 "bypassPermissions"。
        tools: 可用工具集，None 表示全部内置工具。
        cwd: 工作目录，默认 .sdk_sessions/。
        add_dirs: 额外可访问的目录（只读），默认 [项目根目录]。
        mcp_servers: MCP 工具服务器配置，如 {"futures": mcp_config}。
        effort: 推理力度，默认 "max"。
        skills: 启用的 skill，"all" 表示全部加载。
        plugins: 本地插件列表，默认 []。
        thinking: 思考模式配置，默认 None。

    Returns:
        {
            text: str          — 模型完整回复文本,
            stop_reason: str   — 停止原因 ("end_turn" / "max_turns" / ...),
            duration_ms: int   — 耗时(毫秒),
            num_turns: int     — 对话轮次,
            cost: float        — USD 费用,
            is_error: bool     — 是否出错,
            tool_calls: list   — 所有工具调用 [{name, input}, ...],
        }
    """
    # 默认值：基于本文件位置的项目路径（不受 import 位置影响）
    if cwd is None:
        cwd = str(_SDK_SESSIONS_DIR)
        os.makedirs(cwd, exist_ok=True)
    if add_dirs is None:
        add_dirs = [str(_PROJECT_ROOT)]
    if disallowed_tools is None:
        disallowed_tools = []

    async def _run():
        options = ClaudeAgentOptions(
            model=model,
            permission_mode=permission_mode,
            max_turns=max_turns,
            allowed_tools=allowed_tools or [],
            disallowed_tools=disallowed_tools,
            tools=tools,
            cwd=cwd,
            add_dirs=add_dirs,
            system_prompt=system_prompt or None,
            mcp_servers=mcp_servers or {},
            effort=effort,
            skills=skills,
            plugins=plugins or [],
            thinking=thinking,
        )

        text_parts: list[str] = []
        tool_calls: list[dict] = []  # 收集所有工具调用
        stats: dict = {}

        async for msg in query(prompt=prompt, options=options):
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        try:
                            print(block.text, end="", flush=True)
                        except UnicodeEncodeError:
                            print(
                                block.text.encode("ascii", "replace").decode(),
                                end="",
                            )
                        text_parts.append(block.text)
                    elif isinstance(block, ToolUseBlock):
                        print(f"\n[调用工具: {block.name}]", end="")
                        if block.input:
                            tool_calls.append(
                                {"name": block.name, "input": block.input}
                            )
            elif isinstance(msg, ResultMessage):
                stats = {
                    "stop_reason": msg.stop_reason or "unknown",
                    "duration_ms": msg.duration_ms,
                    "num_turns": msg.num_turns,
                    "cost": msg.total_cost_usd,
                    "is_error": msg.is_error,
                }

        print(
            f"\n--- [{stats.get('stop_reason', '?')}] "
            f"{stats.get('duration_ms', 0)}ms | "
            f"{stats.get('num_turns', 0)} turns ---"
        )
        return {"text": "".join(text_parts), **stats, "tool_calls": tool_calls}

    return asyncio.run(_run())


# ============================================================
# 自测：验证路径解析 + 简单问答
# ============================================================
if __name__ == "__main__":
    inject_config()
    print("=" * 50)
    print("cc_sdk 自测")
    print(f"项目根目录 : {_PROJECT_ROOT}")
    print(f"SDK Sessions: {_SDK_SESSIONS_DIR}")
    for v in ["ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_MODEL"]:
        print(f"  [OK] {v}")
    print("=" * 50)

    result = run_agent(
        prompt="用一句话介绍你自己，然后列出你能使用的工具名称。",
        system_prompt="你是量化基金数据分析助手。简短回答。",
        max_turns=3,
    )

    print("\n" + "=" * 50)
    print("自测结果")
    print(f"  停止原因: {result.get('stop_reason')}")
    print(f"  耗时: {result.get('duration_ms', 0)}ms")
    print(f"  对话轮次: {result.get('num_turns', 0)}")
    print(f"  费用: ${result.get('cost', 'N/A')}")
    print(f"  出错: {result.get('is_error', 'N/A')}")
    print("=" * 50)
