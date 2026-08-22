"""Alpha191 批量因子回测。

本脚本只负责四件事：读取本地数据（行业分类缺失时由适配器调用米筐兜底）、
计算 Alpha191 因子、调用 ``因子回测.alpha.analyze_factor``、组合完整 HTML 报告。
IC、RankIC、分组收益、净值与所有绩效指标均由 ``analyze_factor`` 返回，本文件
不会再自行实现任何回测或指标计算逻辑。
"""

from __future__ import annotations

import argparse
import base64
import html
import inspect
import io
import os
import re
import sys
import textwrap
import time
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

import matplotlib

# 必须在导入 pyplot 前选择无界面后端，否则批量运行仍可能尝试创建图形窗口。
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import polars as pl

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from 因子回测.alpha import analyze_factor
from 因子回测.alpha_191.adapter import load_factor_data_with_industry
from 因子回测.alpha_191.alpha_formulas import Alpha191Formulas
from my_utils.email_fun import send_email


DEFAULT_START_DATE = "2021-01-01"
DEFAULT_END_DATE = "2026-07-01"
DEFAULT_RETURN_PERIOD = 5
DEFAULT_GROUP_NUM = 5
DEFAULT_MIN_RECORDS = 500
DEFAULT_MAX_STOCKS = 300
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_REPORT_FILE = os.path.join(OUTPUT_DIR, "alpha191_backtest_report.html")


def build_factor_panel(
    factor_wide: pd.DataFrame,
    daily_returns: pd.DataFrame,
) -> pl.DataFrame:
    """将 Alpha 宽表和适配器日收益宽表转换为 ``analyze_factor`` 的长表输入。

    ``analyze_factor`` 约定 ``daily_ret[t]`` 表示从 t-1 到 t 的单日收益，并在内部
    生成 t+1 起的未来收益。因此这里只做日期、股票代码和字段对齐，不计算未来收益或
    任何回测指标，避免和统一框架重复实现。
    """
    if not isinstance(factor_wide, pd.DataFrame):
        raise TypeError("因子计算结果必须为 pandas DataFrame（index=日期，columns=股票）")
    if not isinstance(daily_returns, pd.DataFrame):
        raise TypeError("日收益数据必须为 pandas DataFrame（index=日期，columns=股票）")

    common_dates = factor_wide.index.intersection(daily_returns.index)
    common_codes = factor_wide.columns.intersection(daily_returns.columns)
    if len(common_dates) == 0 or len(common_codes) == 0:
        return pl.DataFrame(
            schema={
                "trading_date": pl.Date,
                "code": pl.String,
                "factor": pl.Float64,
                "daily_ret": pl.Float64,
            }
        )

    factor = factor_wide.loc[common_dates, common_codes].copy()
    returns = daily_returns.loc[common_dates, common_codes].copy()
    factor.index.name = "trading_date"
    factor.columns.name = "code"
    returns.index.name = "trading_date"
    returns.columns.name = "code"

    # 保留因子或收益缺失的行交给框架统一过滤，不能在这里改变其截面样本口径。
    # DataFrame.melt 不依赖 pandas 即将废弃的 stack 旧实现，并保留空值行。
    factor_long = factor.reset_index().melt(
        id_vars="trading_date", var_name="code", value_name="factor"
    )
    return_long = returns.reset_index().melt(
        id_vars="trading_date", var_name="code", value_name="daily_ret"
    )
    panel = factor_long.merge(return_long, on=["trading_date", "code"], how="inner")
    panel["trading_date"] = pd.to_datetime(panel["trading_date"]).dt.date
    return pl.from_pandas(panel).select("trading_date", "code", "factor", "daily_ret")


def _method_implementation(method_src: str) -> str:
    """提取 ``alpha_NNN_df`` 方法体（去掉 def 行与公共缩进），作为代码兜底。"""
    body = [
        line for line in method_src.strip().splitlines()
        if not line.strip().startswith("def ")
    ]
    if not body:
        return ""
    impl = textwrap.dedent("\n".join(body)).strip()
    # 单行 return 语句去掉 return 前缀，让公式更简洁
    if impl.startswith("return ") and "\n" not in impl:
        impl = impl[len("return "):].strip()
    return impl


def _is_empty_impl(impl: str) -> bool:
    """判断实现是否为「未实现」的空表/空 Series 返回。"""
    compact = impl.replace(" ", "").replace("\n", "")
    return (
        "pd.DataFrame(index=self.close.index,columns=self.close.columns,dtype=float)" in compact
        or "pd.Series(dtype=float)" in compact
    )


def load_alpha_formulas() -> Dict[int, str]:
    """提取 191 个因子的公式文本，供 HTML 报告展示。

    优先取 ``alpha_formulas.py`` 里的公式注释原文（``# Alpha#NNN\\t公式``，
    覆盖 Alpha 1~101 中已实现的 88 个）；没有注释的因子（Alpha 102~191 及
    未实现的 13 个）回退到 ``alpha_NNN_df`` 的实现代码作为公式说明。
    """
    from 因子回测.alpha_191 import alpha_formulas as af

    source = inspect.getsource(af)

    # 1. 公式注释原文：# Alpha#NNN\t<公式>
    formulas: Dict[int, str] = {}
    for match in re.finditer(r"# Alpha#(\d+)\s+(.+)", source):
        num = int(match.group(1))
        formula = match.group(2).strip()
        if formula and not formula.startswith("~"):
            formulas[num] = formula

    # 2. 无注释因子回退到 alpha_NNN_df 实现代码
    cls = af.Alpha191Formulas
    for num in range(1, 192):
        if num in formulas:
            continue
        method = getattr(cls, f"alpha_{num:03d}_df", None)
        if method is None:
            formulas[num] = "（无实现）"
            continue
        try:
            method_src = inspect.getsource(method)
        except (OSError, TypeError):
            formulas[num] = "（无法读取实现）"
            continue
        impl = _method_implementation(method_src)
        formulas[num] = "（未实现）" if _is_empty_impl(impl) else impl

    return formulas


def _framework_stat_row(analysis: Mapping[str, Any], return_period: int) -> Dict[str, Any]:
    """读取指定窗口的框架统计行；此处不重新计算任何指标。"""
    stats = analysis.get("ic_stats")
    if not isinstance(stats, pl.DataFrame):
        raise RuntimeError("analyze_factor 未返回 ic_stats，无法取得 RankIC 排序字段")

    matched = stats.filter(pl.col("window") == return_period)
    if matched.is_empty():
        raise RuntimeError(f"analyze_factor 未返回 {return_period} 日窗口的 IC/RankIC 统计")
    return matched.row(0, named=True)


def run_single_alpha(
    alpha_num: int,
    factor_wide: pd.DataFrame,
    daily_returns: pd.DataFrame,
    analyze: Callable[..., Dict[str, Any]] = analyze_factor,
    return_period: int = DEFAULT_RETURN_PERIOD,
    group_num: int = DEFAULT_GROUP_NUM,
) -> Dict[str, Any]:
    """使用统一因子框架回测单个 Alpha，并原样保存框架结果供报告渲染。"""
    started = time.perf_counter()
    record: Dict[str, Any] = {
        "alpha": alpha_num,
        "status": "unavailable",
        "reason": None,
        "elapsed_seconds": None,
        "analysis": None,
        "rank_ic_mean": None,
    }
    try:
        if factor_wide.empty or factor_wide.notna().sum().sum() == 0:
            record["reason"] = "因子计算结果为空，无法传入 analyze_factor"
            return record

        panel = build_factor_panel(factor_wide, daily_returns)
        if panel.is_empty():
            record["reason"] = "因子与日收益没有可对齐的日期或股票代码"
            return record

        # 所有 IC、RankIC、分组净值和图表均在此唯一入口中生成。
        analysis = analyze(
            data=panel,
            factor_col="factor",
            ret_col="daily_ret",
            ret_windows=(return_period,),
            ic_windows=(return_period,),
            group_num=group_num,
            plot=True,
            save_result=False,
        )
        stats = _framework_stat_row(analysis, return_period)
        record.update(
            {
                "status": "ok",
                "analysis": analysis,
                "ic_mean": stats.get("ic_mean"),
                "ic_std": stats.get("ic_std"),
                "ic_ir": stats.get("ic_ir"),
                "ic_positive_ratio": stats.get("ic_positive_ratio"),
                "rank_ic_mean": stats.get("rank_ic_mean"),
                "rank_ic_std": stats.get("rank_ic_std"),
                "rank_ic_ir": stats.get("rank_ic_ir"),
                "rank_ic_positive_ratio": stats.get("rank_ic_positive_ratio"),
            }
        )
    except Exception as exc:  # 单个公式或框架异常不得阻断其他 190 个因子的报告。
        record["status"] = "error"
        record["reason"] = f"{type(exc).__name__}: {str(exc)[:500]}"
    finally:
        record["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    return record


def run_batch_backtest(
    data: Mapping[str, Any],
    alpha_numbers: Iterable[int] = range(1, 192),
    return_period: int = DEFAULT_RETURN_PERIOD,
    group_num: int = DEFAULT_GROUP_NUM,
    analyze: Callable[..., Dict[str, Any]] = analyze_factor,
) -> List[Dict[str, Any]]:
    """计算指定 Alpha 并全部委托 ``analyze_factor`` 回测。

    不能计算或缺字段的公式同样会生成一条结果记录，供 HTML 明确标记“未能复现”；
    不再将慢因子静默跳过，也不使用旧批量回测检查点中的自算指标。
    """
    daily_returns = data.get("returns")
    if not isinstance(daily_returns, pd.DataFrame):
        reason = "数据适配器未返回 returns 日收益宽表，无法调用 analyze_factor"
        return [
            {
                "alpha": alpha_num,
                "status": "unavailable",
                "reason": reason,
                "elapsed_seconds": 0.0,
                "analysis": None,
                "rank_ic_mean": None,
            }
            for alpha_num in alpha_numbers
        ]

    # 行业中性化因子使用适配器的本地优先、米筐兜底分类；无论是否成功都会如实写入结果。
    formulas = Alpha191Formulas(data, industry_map=data.get("industry"))
    records: List[Dict[str, Any]] = []
    for alpha_num in alpha_numbers:
        method_name = f"alpha_{alpha_num:03d}_df"
        try:
            factor_wide = getattr(formulas, method_name)()
        except AttributeError:
            records.append(
                {
                    "alpha": alpha_num,
                    "status": "unavailable",
                    "reason": f"公式库未提供 {method_name}，无法复现该因子",
                    "elapsed_seconds": 0.0,
                    "analysis": None,
                    "rank_ic_mean": None,
                }
            )
            continue
        except Exception as exc:
            reason = f"因子计算失败：{type(exc).__name__}: {str(exc)[:500]}"
            # 缺行业分类是数据无法复现，不是公式逻辑故障；单独标为 unavailable，
            # 使最终 HTML 明确说明本地和米筐数据未能满足公式所需输入。
            status = "unavailable" if "缺少行业分类数据" in str(exc) else "error"
            records.append(
                {
                    "alpha": alpha_num,
                    "status": status,
                    "reason": reason,
                    "elapsed_seconds": 0.0,
                    "analysis": None,
                    "rank_ic_mean": None,
                }
            )
            continue

        records.append(
            run_single_alpha(
                alpha_num=alpha_num,
                factor_wide=factor_wide,
                daily_returns=daily_returns,
                analyze=analyze,
                return_period=return_period,
                group_num=group_num,
            )
        )
    return records


def _number_or_none(value: Any) -> Optional[float]:
    """将框架返回的标量规范为报告可排序数值，NaN/inf 均表示不可用。"""
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result and abs(result) != float("inf") else None


def _table_to_html(table: Any) -> str:
    """直接展示 analyze_factor 的 Polars 统计表，不在报告层派生新指标。"""
    if not isinstance(table, pl.DataFrame) or table.is_empty():
        return '<p class="empty">框架未返回可展示的统计结果。</p>'
    return table.to_pandas().to_html(
        index=False,
        border=0,
        classes="framework-table",
        float_format=lambda value: f"{value:.6f}",
        na_rep="—",
        escape=True,
    )


def _figure_to_data_uri(figure: Any) -> Optional[str]:
    """把框架生成的 Matplotlib 图嵌入单文件 HTML，避免报告依赖外部 PNG。"""
    if figure is None:
        return None
    buffer = io.BytesIO()
    try:
        figure.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"
    finally:
        buffer.close()
        plt.close(figure)


def _prepare_report_artifacts(record: Dict[str, Any]) -> None:
    """尽早序列化图表并释放 Figure，防止 191 个因子的图对象占满内存。"""
    if record.get("status") != "ok" or record.get("framework_tables") is not None:
        return
    analysis = record.get("analysis")
    if not isinstance(analysis, Mapping):
        return

    record["framework_tables"] = {
        "ic_stats": _table_to_html(analysis.get("ic_stats")),
        "group_stats": _table_to_html(analysis.get("group_stats")),
    }
    figure_labels = {
        "nav": "分组净值曲线",
        "ic_series": "IC / RankIC 时序",
        "cumulative_ic": "累计 IC / RankIC",
    }
    record["figure_data_uris"] = [
        (figure_labels.get(name, name), uri)
        for name, figure in analysis.get("figures", {}).items()
        for uri in [_figure_to_data_uri(figure)]
        if uri is not None
    ]
    # 图像、指标已经被保存在 HTML 片段中；删除原对象防止完整批量运行占用过多内存。
    record["analysis"] = None


def _ordered_records(records: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    """成功因子按框架给出的 RankIC 均值降序，无法复现的记录统一排在后面。"""
    def sort_key(item: Mapping[str, Any]) -> tuple:
        rank_ic = _number_or_none(item.get("rank_ic_mean"))
        # None 排在成功记录末尾；0 是有效 RankIC，不能被误判为缺失。
        return (
            item.get("status") != "ok",
            -(rank_ic if rank_ic is not None else float("-inf")),
            int(item.get("alpha", 0)),
        )

    return sorted(
        records,
        key=sort_key,
    )


def _format_metric(value: Any, ratio: bool = False) -> str:
    number = _number_or_none(value)
    if number is None:
        return "—"
    return f"{number:.2%}" if ratio else f"{number:.6f}"


def _summary_row(record: Mapping[str, Any], rank: Optional[int]) -> str:
    alpha_num = int(record["alpha"])
    status = record.get("status", "error")
    metric_cells = "".join(
        f"<td>{_format_metric(record.get(name), ratio=name.endswith('ratio'))}</td>"
        for name in ("rank_ic_mean", "rank_ic_ir", "ic_mean", "ic_ir")
    )
    reason = html.escape(str(record.get("reason") or ""))
    return (
        f'<tr class="status-{html.escape(status)}">'
        f"<td>{rank if rank is not None else '—'}</td>"
        f'<td><a href="#alpha-{alpha_num:03d}">Alpha{alpha_num:03d}</a></td>'
        f"<td>{html.escape(status)}</td>{metric_cells}"
        f"<td>{_format_metric(record.get('elapsed_seconds'))} 秒</td>"
        f"<td>{reason or '—'}</td></tr>"
    )


def _detail_section(record: Mapping[str, Any], rank: Optional[int], formula: Optional[str] = None) -> str:
    alpha_num = int(record["alpha"])
    status = record.get("status", "error")
    status_text = {"ok": "框架回测完成", "unavailable": "未能复现", "error": "计算或框架报错"}.get(status, status)
    heading = f"Alpha{alpha_num:03d}" + (f"（RankIC 排名 #{rank}）" if rank is not None else "")
    formula_html = (
        f'<p class="alpha-formula"><strong>公式：</strong><code>{html.escape(formula)}</code></p>'
        if formula
        else ""
    )
    if status != "ok":
        reason = html.escape(str(record.get("reason") or "未记录具体原因"))
        return (
            f'<section id="alpha-{alpha_num:03d}" class="factor-card unavailable">'
            f"<h2>{heading}</h2>{formula_html}<p><strong>{status_text}</strong>：{reason}</p>"
            "<p>该因子没有可用的 analyze_factor 输出，因此不伪造图表或指标。</p></section>"
        )

    mutable_record = dict(record)
    _prepare_report_artifacts(mutable_record)
    artifacts = mutable_record.get("framework_tables", {})
    figures = "".join(
        f'<figure><figcaption>{html.escape(label)}</figcaption><img src="{uri}" alt="{html.escape(label)}"></figure>'
        for label, uri in mutable_record.get("figure_data_uris", [])
    ) or '<p class="empty">框架未返回图像。</p>'
    return f'''<section id="alpha-{alpha_num:03d}" class="factor-card">
<h2>{heading}</h2>
{formula_html}
<p>状态：<strong>{status_text}</strong>；耗时：{_format_metric(record.get("elapsed_seconds"))} 秒。</p>
<h3>IC / RankIC 统计</h3>{artifacts.get("ic_stats", '<p class="empty">无</p>')}
<h3>分组回测统计</h3>{artifacts.get("group_stats", '<p class="empty">无</p>')}
<div class="figures">{figures}</div>
</section>'''


def render_html_report(records: Sequence[Dict[str, Any]], metadata: Mapping[str, Any]) -> str:
    """输出完整单文件报告：191 个因子逐一展示，成功记录按 RankIC 排序。"""
    for record in records:
        _prepare_report_artifacts(record)
    ordered = _ordered_records(records)
    formulas = load_alpha_formulas()
    rank_by_alpha = {
        int(record["alpha"]): index
        for index, record in enumerate((item for item in ordered if item.get("status") == "ok"), start=1)
    }
    summary_rows = "".join(
        _summary_row(record, rank_by_alpha.get(int(record["alpha"]))) for record in ordered
    )
    detail_sections = "\n".join(
        _detail_section(record, rank_by_alpha.get(int(record["alpha"])), formula=formulas.get(int(record["alpha"])))
        for record in ordered
    )
    successful = sum(record.get("status") == "ok" for record in records)
    unavailable = sum(record.get("status") == "unavailable" for record in records)
    errors = sum(record.get("status") == "error" for record in records)
    source_note = html.escape(str(metadata.get("data_source", "本地日线数据；行业分类由本地优先、米筐兜底的适配器提供")))

    return f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>Alpha191 analyze_factor 回测报告</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,"Microsoft YaHei",sans-serif;max-width:1440px;margin:auto;padding:24px;background:#f6f8fa;color:#1f2937;line-height:1.55}}
h1{{border-bottom:3px solid #2563eb;padding-bottom:12px}} h2{{margin-top:0;color:#1d4ed8}} h3{{margin:20px 0 8px}}
.notice,.summary,.factor-card{{background:#fff;border-radius:10px;padding:16px 20px;margin:16px 0;box-shadow:0 1px 3px #0000001a}}
.notice{{border-left:5px solid #2563eb}} .counts{{display:flex;gap:12px;flex-wrap:wrap}} .count{{padding:8px 14px;background:#eff6ff;border-radius:7px}}
table{{width:100%;border-collapse:collapse;margin:8px 0 16px;background:#fff}} th{{background:#1d4ed8;color:#fff}} th,td{{padding:8px;border-bottom:1px solid #dbe3ee;text-align:left;vertical-align:top;font-size:13px}}
.framework-table{{width:auto;min-width:60%}} .status-unavailable,.status-error{{background:#fff7ed}} .factor-card.unavailable{{border-left:5px solid #f59e0b}}
.alpha-formula{{margin:6px 0 10px}} .alpha-formula code{{white-space:pre-wrap;word-break:break-all;background:#f1f5f9;padding:4px 8px;border-radius:4px;font-size:12px;font-family:Consolas,Menlo,monospace;color:#334155}}
.figures{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:16px}} figure{{margin:0;background:#f8fafc;padding:10px;border-radius:8px}} figcaption{{font-weight:600;margin:2px 0 8px}} img{{width:100%;height:auto;background:#fff}}
a{{color:#1d4ed8}} .empty{{color:#64748b}} footer{{font-size:12px;color:#64748b;padding:16px 0}}
</style></head><body>
<h1>Alpha191 批量因子回测报告</h1>
<div class="notice"><p><strong>回测口径：</strong>每个可用因子均直接调用本地 <code>analyze_factor</code>；本报告只呈现其 IC/RankIC、分组回测统计和图表，不自行计算任何回测指标。</p>
<p><strong>数据来源：</strong>{source_note}</p><p>日期：{html.escape(str(metadata.get("start_date", "—")))} ～ {html.escape(str(metadata.get("end_date", "—")))}；股票数：{html.escape(str(metadata.get("stock_count", "—")))}</p></div>
<div class="counts"><div class="count">总因子：{len(records)}</div><div class="count">框架回测成功：{successful}</div><div class="count">未能复现：{unavailable}</div><div class="count">运行错误：{errors}</div></div>
<section class="summary"><h2>汇总（按 RankIC 均值降序）</h2><table><thead><tr><th>排名</th><th>因子</th><th>状态</th><th>RankIC</th><th>RankIC IR</th><th>IC</th><th>IC IR</th><th>耗时</th><th>无法复现/错误原因</th></tr></thead><tbody>{summary_rows}</tbody></table></section>
<main>{detail_sections}</main>
<footer>生成时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}；本报告为单文件 HTML，图表已内嵌。</footer>
</body></html>'''


def _select_most_complete_stocks(data: Mapping[str, Any], max_stocks: int) -> Dict[str, Any]:
    """沿用原脚本的样本控制：按本地收盘价记录数选取最完整的股票。"""
    close = data.get("close")
    if not isinstance(close, pd.DataFrame):
        raise ValueError("数据适配器未返回 close 宽表")
    if max_stocks <= 0 or max_stocks >= len(close.columns):
        selected = close.columns.tolist()
    else:
        selected = close.notna().sum(axis=0).nlargest(max_stocks).index.tolist()
    return {
        key: value.loc[:, value.columns.intersection(selected)].copy()
        if isinstance(value, pd.DataFrame)
        else value
        for key, value in data.items()
    }


def _parse_alpha_numbers(raw_value: str) -> List[int]:
    """支持 ``1-191``、``1,5,10`` 形式，便于调试单个因子而不改变默认全量行为。"""
    alpha_numbers: List[int] = []
    for item in raw_value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start, end = (int(part.strip()) for part in item.split("-", maxsplit=1))
            alpha_numbers.extend(range(start, end + 1))
        else:
            alpha_numbers.append(int(item))
    result = sorted(set(alpha_numbers))
    if not result or any(number < 1 or number > 191 for number in result):
        raise ValueError("因子编号必须在 1 到 191 之间")
    return result


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="使用本地 analyze_factor 运行 Alpha191 批量回测")
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument("--return-period", type=int, default=DEFAULT_RETURN_PERIOD)
    parser.add_argument("--group-num", type=int, default=DEFAULT_GROUP_NUM)
    parser.add_argument("--min-records", type=int, default=DEFAULT_MIN_RECORDS)
    parser.add_argument("--max-stocks", type=int, default=DEFAULT_MAX_STOCKS)
    parser.add_argument("--alphas", default="1-191", help="例如 1-191 或 1,5,10，仅用于调试")
    parser.add_argument("--report-file", default=DEFAULT_REPORT_FILE)
    parser.add_argument("--send-email", action="store_true", help="显式开启后才发送邮件")
    parser.add_argument("--receiver", action="append", default=[], help="邮件接收人，可重复传入")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argument_parser().parse_args(argv)
    if args.return_period <= 0 or args.group_num < 2:
        raise ValueError("return_period 必须大于 0，group_num 必须至少为 2")
    if args.send_email and not args.receiver:
        raise ValueError("启用 --send-email 时必须至少传入一个 --receiver")

    alpha_numbers = _parse_alpha_numbers(args.alphas)
    started = time.perf_counter()
    metadata: Dict[str, Any] = {
        "start_date": args.start_date,
        "end_date": args.end_date,
        "stock_count": 0,
        "data_source": "本地日线数据；行业分类由本地优先、米筐 API 兜底的适配器提供",
    }
    try:
        # 适配器负责读取本地数据，并在行业分类层面调用米筐作为兜底来源。
        raw_data = load_factor_data_with_industry(
            args.start_date,
            args.end_date,
            min_records=args.min_records,
        )
        data = _select_most_complete_stocks(raw_data, args.max_stocks)
        metadata["stock_count"] = len(data["close"].columns)
        records = run_batch_backtest(
            data=data,
            alpha_numbers=alpha_numbers,
            return_period=args.return_period,
            group_num=args.group_num,
        )
    except Exception as exc:
        # 数据完全不可用时仍输出完整 191（或指定范围）因子列表，满足可追溯要求。
        reason = f"数据读取或适配失败：{type(exc).__name__}: {str(exc)[:500]}"
        records = [
            {
                "alpha": alpha_num,
                "status": "unavailable",
                "reason": reason,
                "elapsed_seconds": 0.0,
                "analysis": None,
                "rank_ic_mean": None,
            }
            for alpha_num in alpha_numbers
        ]

    metadata["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    report = render_html_report(records, metadata)
    report_path = os.path.abspath(args.report_file)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as report_file:
        report_file.write(report)
    print(f"HTML 报告已生成：{report_path}")

    if args.send_email:
        send_email(
            subject=f"Alpha191 analyze_factor 回测报告（{args.start_date}~{args.end_date}）",
            body=report,
            body_type="html",
            receiver_emails=args.receiver,
        )
        print("邮件已发送")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
