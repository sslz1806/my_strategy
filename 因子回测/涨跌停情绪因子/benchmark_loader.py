"""
benchmark_loader.py — 统一加载各宽基指数的日频收益序列。

支持的基准名称:
- "all_a_value_weight": 全 A 市值加权（需传入 prepared_daily + calendar）
- "zz500":  中证 500  (000905.SH)
- "zz1000": 中证 1000 (000852.SH)
- "zz2000": 中证 2000 (932000.CSI)
"""

from datetime import date
from pathlib import Path
from typing import List, Optional

import logging
import pandas as pd
import polars as pl

# 中证 500 本地缓存路径
_ZZ500_LOCAL_PATH = Path("E:/working/stock_data/barra_cne5/benchmark_zz500.parquet")

# 指数代码映射：基准名 → (gm_code, ts_code)
_BENCHMARK_CODE_MAP = {
    "zz500":  ("SHSE.000905", "000905.SH"),
    "zz1000": ("SHSE.000852", "000852.SH"),
    "zz2000": ("CSI.932000",  "932000.CSI"),
}


def load_benchmark(
    name: str,
    start_date: date,
    end_date: date,
    prepared_daily: Optional[pl.DataFrame] = None,
    calendar: Optional[pl.DataFrame] = None,
    source: str = "auto",
) -> pd.DataFrame:
    """
    加载指定宽基指数的日频收益序列。

    返回的 DataFrame 包含 trading_date 和 market_daily_ret 两列，
    与 timing_engine.build_value_weighted_benchmark 输出格式一致。
    收益为小数形式（0.01 = 1%）。
    """
    if name == "all_a_value_weight":
        return _load_all_a_value_weight(prepared_daily, calendar, start_date, end_date)
    elif name in _BENCHMARK_CODE_MAP:
        return _load_index_benchmark(name, start_date, end_date, source)
    else:
        raise ValueError(f"未知基准名称: {name}，可选: all_a_value_weight, zz500, zz1000, zz2000")


def _load_all_a_value_weight(prepared_daily, calendar, start_date, end_date):
    """从 prepared_daily 计算全 A 市值加权收益。"""
    if prepared_daily is None or calendar is None:
        raise ValueError("all_a_value_weight 基准需要 prepared_daily 和 calendar 参数")
    # 直接延时导入避免循环依赖
    from 因子回测.涨跌停情绪因子.timing_engine import build_value_weighted_benchmark
    result = build_value_weighted_benchmark(prepared_daily, calendar)
    # 按日期范围过滤
    result = result[
        (result["trading_date"] >= pd.Timestamp(start_date))
        & (result["trading_date"] <= pd.Timestamp(end_date))
    ].reset_index(drop=True)
    return result


def _load_index_benchmark(name, start_date, end_date, source="auto"):
    """从本地或 API 加载指数收益。

    数据来源优先级:
    1. local:  仅 zz500 有本地 parquet 缓存
    2. gm:     掘金 API (stock_api.gm_get_index_day_data)
    3. ts:     Tushare API (stock_api.ts.index_daily)
    """
    # 尝试本地（仅 zz500）
    if source in ("auto", "local") and name == "zz500" and _ZZ500_LOCAL_PATH.exists():
        local = pl.read_parquet(_ZZ500_LOCAL_PATH)
        result = local.select(
            pl.col("trading_date"),
            pl.col("ret_1d").alias("market_daily_ret"),
        ).filter(
            pl.col("trading_date").is_between(
                pl.lit(start_date).cast(pl.Date),
                pl.lit(end_date).cast(pl.Date),
            )
        ).sort("trading_date").to_pandas()
        result["trading_date"] = pd.to_datetime(result["trading_date"])
        return result.reset_index(drop=True)

    # 回退到掘金 API
    if source in ("auto", "gm"):
        try:
            gm_code = _BENCHMARK_CODE_MAP[name][0]
            from my_utils.stock_api import stock_api
            api = stock_api()
            df = api.gm_get_index_day_data(gm_code, start_date, end_date)
            if df is not None and len(df) > 0:
                result = df[["trading_date", "pct"]].copy()
                result["trading_date"] = pd.to_datetime(result["trading_date"])
                result["market_daily_ret"] = result["pct"] / 100.0  # 百分比 → 小数
                return result.sort_values("trading_date")[["trading_date", "market_daily_ret"]].reset_index(drop=True)
        except Exception:
            logging.getLogger(__name__).warning(
                "掘金 API 加载基准 %s 失败，回退到 Tushare", name, exc_info=True)

    # 最后回退到 Tushare
    if source in ("auto", "ts"):
        try:
            ts_code = _BENCHMARK_CODE_MAP[name][1]
            from my_utils.stock_api import stock_api
            api = stock_api()
            ts = api.ts
            df = ts.index_daily(ts_code=ts_code,
                                start_date=start_date.strftime("%Y%m%d"),
                                end_date=end_date.strftime("%Y%m%d"))
            if df is not None and len(df) > 0:
                result = df.rename(columns={"trade_date": "trading_date", "pct_chg": "market_daily_ret"})
                result["trading_date"] = pd.to_datetime(result["trading_date"])
                result["market_daily_ret"] = result["market_daily_ret"] / 100.0
                return result.sort_values("trading_date")[["trading_date", "market_daily_ret"]].reset_index(drop=True)
        except Exception:
            logging.getLogger(__name__).warning(
                "Tushare API 加载基准 %s 失败", name, exc_info=True)

    raise RuntimeError(f"无法加载基准 {name}：所有数据源均失败")


def list_available_benchmarks() -> List[str]:
    """返回当前可用的基准名称列表（用于 notebook 中遍历）。"""
    return ["all_a_value_weight", "zz500", "zz1000", "zz2000"]
