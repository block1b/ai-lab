"""MCP Server：用 FastMCP 把财报工具封装成标准 MCP 服务（stdio 传输）。

MCP 是工具层的「USB-C 接口」——工具写成 MCP Server 后，任何支持 MCP 的
Agent（WorkBuddy / Qoder / Claude / 本项目的客户端）都能即插即用。

运行：python mcp/server.py
（stdio 模式下会有 JSON-RPC 输出，不要手动乱敲；由 MCP Client 启动它）
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# mcp 2.x 把 FastMCP 改名为 MCPServer；两种版本都兼容
try:
    from mcp.server.fastmcp import FastMCP as MCPServer  # mcp < 2
except ImportError:
    from mcp.server.mcpserver import MCPServer  # mcp >= 2

from tools.finance_tools import calculate, get_metric, growth_rate, rag_search

mcp = MCPServer("finance-report-tools")


@mcp.tool()
def calculate_tool(expression: str) -> str:
    """安全算术计算器。支持 + - * / ** %，用于算增长率、利润率。"""
    return calculate.invoke({"expression": expression})


@mcp.tool()
def growth_rate_tool(current: float, previous: float) -> str:
    """计算增长率百分比。"""
    return growth_rate.invoke({"current": current, "previous": previous})


@mcp.tool()
def get_metric_tool(year: int, metric: str) -> str:
    """查询结构化财务数据。metric: revenue/net_profit/gross_margin/rd_spend/overseas_ratio。"""
    return get_metric.invoke({"year": year, "metric": metric})


@mcp.tool()
def rag_search_tool(query: str) -> str:
    """检索财报文本段落。"""
    return rag_search.invoke({"query": query})


if __name__ == "__main__":
    mcp.run(transport="stdio")
