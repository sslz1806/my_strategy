"""
测试 Claude Agent SDK 解析 Excel 期货持仓信息

让 Agent SDK 自主读取 基金资产统计.xlsx 的"期货持仓"Sheet，
提取并按产品汇总期货合约持仓数据。

用法:
    python agent/test_futures_extraction.py

前置条件:
    - claude-agent-sdk 已安装 (当前版本 0.2.103)
    - ~/.claude/settings.json 中有 ANTHROPIC_BASE_URL 等配置
    - 基金资产统计.xlsx 存在于 stats_stock_asset/report/ 目录
"""
import sys
from pathlib import Path

# 同目录下的 cc_sdk.py（Python 脚本目录自动在 sys.path 首位）
from cc_sdk import inject_config, run_agent

# 项目根目录（本脚本在 agent/ 子目录下）
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)

import claude_agent_sdk as sdk
from claude_agent_sdk import create_sdk_mcp_server


# ============================================================
# 测试数据：Excel 文件路径
# ============================================================
FUTURES_EXCEL = (
    r"D:\Admin\Desktop\project\stats_quant_fund\agent\浙商证券_陆生生量化中性进取2号私募证券投资基金_收益互换估值报告_20260722.xlsx"
)


# ============================================================
# 结构化返回工具：submit_result
# ============================================================
@sdk.tool(
    "submit_result",
    "提交期货持仓提取的结构化结果。分析完成后必须调用此工具提交数据。",
    {
        "type": "object",
        "properties": {
            "product": {"type": "string", "description": "产品名称"},
            "valuation_date": {"type": "string", "description": "估值日期"},
            "futures": {
                "type": "array",
                "description": "期货持仓列表",
                "items": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "合约代码"},
                        "name": {"type": "string", "description": "合约名称"},
                        "qty": {"type": "number", "description": "持仓数量，负数为空头"},
                        "notional": {"type": "number", "description": "名义本金"},
                        "market_value": {"type": "number", "description": "市值"},
                        "pnl": {"type": "number", "description": "浮动盈亏"},
                    },
                },
            },
            "summary": {"type": "string", "description": "一句话总结期货持仓情况"},
        },
        "required": ["product", "futures", "summary"],
    },
)
async def submit_result(_args: dict) -> dict:
    """
    数据快递员工具：函数体不需要做事。
    消息循环已经在 ToolUseBlock.input 中拿到结构化数据。
    """
    return {"content": [{"type": "text", "text": "数据已接收"}]}


MCP_CONFIG = create_sdk_mcp_server(
    name="futures_extractor",
    tools=[submit_result],
)


# ============================================================
# 测试任务：提取期货持仓
# ============================================================
def test_extract_futures():
    """
    让 Agent SDK 解析 Excel 中的期货持仓信息，并以结构化 JSON 返回。

    Agent 分析完数据后调用 submit_result 工具提交结果，
    消息循环捕获 ToolUseBlock.input 作为 Python dict 直接可用。
    """
    excel_path = FUTURES_EXCEL

    prompt = (
        f"请提取这份 Excel 中的期货持仓信息，按产品汇总合约和数量。\n\n"
        f"文件路径: {excel_path}\n\n"
        f"分析完成后调用 submit_result 工具提交结构化结果。"
    )

    system = "你是量化基金数据分析助手，擅长用 pandas 处理 Excel。用中文回答，简洁专业。分析完毕后必须调用 submit_result 提交数据。"

    print("=" * 60)
    print("Claude Agent SDK — 期货持仓解析测试")
    print(f"Excel 文件: {excel_path}")
    print(f"项目根目录: {_PROJECT_ROOT}")
    print("=" * 60)

    result = run_agent(
        prompt=prompt,
        system_prompt=system,
        skills="xlsx",
        mcp_servers={"futures": MCP_CONFIG},
    )

    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    print(f"  停止原因: {result.get('stop_reason')}")
    print(f"  耗时: {result.get('duration_ms', 0)}ms")
    print(f"  对话轮次: {result.get('num_turns', 0)}")
    print(f"  费用: ${result.get('cost', 'N/A')}")
    print(f"  出错: {result.get('is_error', 'N/A')}")

    # 从通用 tool_calls 中筛选 submit_result 的数据
    structured = None
    for tc in result.get("tool_calls", []):
        if tc["name"] == "submit_result":
            structured = tc["input"]
            break

    if structured:
        print("\n--- 结构化返回 (Python dict) ---")
        print(f"  产品: {structured.get('product')}")
        print(f"  估值日期: {structured.get('valuation_date')}")
        print(f"  总结: {structured.get('summary')}")
        print("  期货持仓:")
        for f in structured.get("futures", []):
            print(
                f"    {f.get('code', '?')} x{f.get('qty', 0)}  "
                f"市值={f.get('market_value', 0):,.0f}  "
                f"盈亏={f.get('pnl', 0):,.0f}"
            )
    else:
        print("\n[!] 未收到结构化数据，Agent 可能未调用 submit_result 工具")
        print("--- 模型文本回复 ---")
        text = result.get("text", "(无输出)")
        try:
            print(text)
        except UnicodeEncodeError:
            print(text.encode("ascii", "replace").decode())

    return result


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    inject_config()
    print("配置注入完成")
    for v in ["ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_MODEL"]:
        print(f"  [OK] {v}")

    test_extract_futures()
