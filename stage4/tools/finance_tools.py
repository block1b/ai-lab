"""阶段 4 工具层：Function Calling 的 4 个工具。

设计要点（对应打卡表「工具参数填错 → Pydantic 校验 + 失败重试」）：
- 每个工具用 @tool + Pydantic 类型注解，参数错误会被框架拦截并反馈给模型
- 工具内部只做确定性计算/检索，不调用 LLM，保证可测
- rag_search 是「把阶段 3 的 RAG 封装成 Agent 工具」的实现
"""

import ast
import math
import os
import sys
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.tools import tool
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from corpus import FINANCIALS, load_documents  # noqa: E402


class CalcInput(BaseModel):
    expression: str = Field(description="合法的算术表达式，如 (12.8-10.0)/10.0")


class GrowthInput(BaseModel):
    current: float = Field(description="本期数值")
    previous: float = Field(description="上期数值")


class SearchInput(BaseModel):
    query: str = Field(description="要检索的财报问题，如 2024年营收")


class MetricInput(BaseModel):
    year: int = Field(description="年份，如 2024")
    metric: str = Field(description="指标名：revenue/net_profit/gross_margin/rd_spend/overseas_ratio")


_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod)
_ALLOWED_UNARYOPS = (ast.UAdd, ast.USub)


def _safe_eval(expr: str) -> float:
    """只允许算术 AST 节点，杜绝 eval 的代码执行风险。"""
    node = ast.parse(expr, mode="eval").body

    def visit(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, ast.BinOp) and isinstance(n.op, _ALLOWED_BINOPS):
            left, right = visit(n.left), visit(n.right)
            if isinstance(n.op, ast.Add): return left + right
            if isinstance(n.op, ast.Sub): return left - right
            if isinstance(n.op, ast.Mult): return left * right
            if isinstance(n.op, ast.Div):
                if right == 0: raise ValueError("除数不能为 0")
                return left / right
            if isinstance(n.op, ast.Pow):
                if abs(left) > 1e6 or abs(right) > 100: raise ValueError("幂运算超出范围")
                return left ** right
            if isinstance(n.op, ast.Mod):
                if right == 0: raise ValueError("取模除数不能为 0")
                return left % right
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, _ALLOWED_UNARYOPS):
            v = visit(n.operand)
            return v if isinstance(n.op, ast.UAdd) else -v
        raise ValueError(f"不支持的表达式节点: {type(n).__name__}")

    result = visit(node)
    if not math.isfinite(result):
        raise ValueError("结果不是有限数")
    return result


@tool("calculate", args_schema=CalcInput)
def calculate(expression: str) -> str:
    """安全算术计算器。支持 + - * / ** % 和括号，用于算增长率、利润率等。"""
    try:
        value = _safe_eval(expression)
        return f"{expression} = {round(value, 6)}"
    except (ValueError, SyntaxError, TypeError) as exc:
        return f"计算失败: {exc}"


@tool("growth_rate", args_schema=GrowthInput)
def growth_rate(current: float, previous: float) -> str:
    """计算增长率（百分比）。current 本期，previous 上期。"""
    if previous == 0:
        return "计算失败: 上期为 0，无法计算增长率"
    rate = (current - previous) / previous * 100
    return f"增长率 = ({current} - {previous}) / {previous} × 100% = {round(rate, 2)}%"


@tool("get_metric", args_schema=MetricInput)
def get_metric(year: int, metric: str) -> str:
    """查询结构化财务数据。year 年份，metric 指标名（见函数文档）。"""
    data = FINANCIALS.get(year)
    if data is None:
        return f"未找到 {year} 年数据，可用年份: {list(FINANCIALS)}"
    if metric not in data:
        return f"未知指标 {metric}，可用指标: {list(data)}"
    return f"{year} 年 {metric} = {data[metric]}"


# ---- 把阶段 3 的 RAG 封装成工具 ----
_DOCS = load_documents()
_KEYWORDS: dict[str, list[str]] = {}


def _terms(text: str) -> set[str]:
    """中文按 2-gram 切分 + 数字按词切分（与 stage3 BM25 一致）。"""
    import re
    cjk = re.findall(r"[一-龥]+", text)
    nums = re.findall(r"\d+(?:\.\d+)?", text)
    grams = set(nums)
    for seg in cjk:
        grams.update(seg[i:i + 2] for i in range(len(seg) - 1))
    return grams


def _init_keywords():
    """纯 Python 轻量关键词索引（不依赖向量库，避免工具层强依赖 embedding）。"""
    for i, d in enumerate(_DOCS):
        _KEYWORDS[i] = _terms(d.page_content)


_init_keywords()


@tool("rag_search", args_schema=SearchInput)
def rag_search(query: str) -> str:
    """从公司财报文档中检索相关段落（阶段3 RAG 的检索能力封装）。
    适合查"风险提示""战略展望""海外收入占比"等文本事实。
    """
    qterms = _terms(query)
    scored = []
    for i, d in enumerate(_DOCS):
        overlap = len(qterms & _KEYWORDS[i])
        if overlap:
            scored.append((overlap, d))
    scored.sort(key=lambda x: x[0], reverse=True)
    if not scored:
        return "财报中未检索到相关内容"
    top = scored[:3]
    return "\n\n".join(f"[{d.metadata['source']}] {d.page_content}" for _, d in top)


ALL_TOOLS = [calculate, growth_rate, get_metric, rag_search]
TOOL_MAP = {t.name: t for t in ALL_TOOLS}


if __name__ == "__main__":
    for t in ALL_TOOLS:
        print(f"{t.name}: {t.description[:50]}...")
    print("\n自测:")
    print(calculate.invoke({"expression": "(12.8-10.0)/10.0*100"}))
    print(growth_rate.invoke({"current": 12.8, "previous": 10.0}))
    print(get_metric.invoke({"year": 2024, "metric": "revenue"}))
    print(rag_search.invoke({"query": "风险提示 客户集中度"}))
