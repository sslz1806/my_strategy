"""
测试 add_future_return 在 Polars panel 场景下的正确性。

核心关注：is_panel=True 时，Polars 版用 .over(code) 包裹 shift 表达式，
能否正确做到"只在本股票内 shift，不跨股票污染"。
与 Pandas 版（groupby+transform）做交叉验证。
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from 因子回测.alpha import add_future_return


# ============================================================
# 基础 panel 场景：多股票、日期对齐
# ============================================================

def test_polars_panel_matches_pandas_basic():
    """Polars panel 与 Pandas panel 结果一致（基础场景）"""
    dates = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4), date(2024, 1, 5)]
    codes = ["A", "B"]

    records = []
    for c in codes:
        for i, d in enumerate(dates):
            records.append({"code": c, "trading_date": d, "pct": 0.01 * (i + 1) * (1 if c == "A" else -1)})

    df_pd = pd.DataFrame(records)
    df_pl = pl.DataFrame(records)

    # Polars 有 code 列 → is_panel=True
    result_pl = add_future_return(df_pl, horizons=(1, 2, 3))
    # Pandas 有 code 列 → is_panel=True
    result_pd = add_future_return(df_pd, horizons=(1, 2, 3))

    for h in [1, 2, 3]:
        col = f"future_pct_{h}d"
        pl_vals = result_pl.sort(["code", "trading_date"])[col].to_list()
        pd_vals = result_pd.sort_values(["code", "trading_date"])[col].tolist()
        # 让两边 None/nan 对齐
        pd_vals_fixed = [v if not (isinstance(v, float) and np.isnan(v)) else None for v in pd_vals]
        assert pl_vals == pd_vals_fixed, (
            f"不匹配: horizon={h}\n  pl={pl_vals}\n  pd={pd_vals_fixed}"
        )


# ============================================================
# 核心验证：跨股票污染检测
# ============================================================

def test_polars_panel_no_cross_stock_contamination():
    """
    如果 .over(code) 没起作用，股票A最后一行的 shift(-1) 会取到股票B第一行。
    本测试故意让两股票数据不同，检测边界是否出界。
    """
    df = pl.DataFrame({
        "code": ["A", "A", "A", "B", "B", "B"],
        "trading_date": [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)] * 2,
        "pct": [0.10, 0.20, -0.10, 0.50, 0.60, -0.30],  # B的收益率明显不同于A
    })

    result = add_future_return(df, horizons=(1, 2))

    # A 最后一行 (2024-01-04) 的 future_pct_1d 应为 None（A没有更多数据）
    a_last = result.filter(
        (pl.col("code") == "A") & (pl.col("trading_date") == date(2024, 1, 4))
    ).row(0, named=True)
    assert a_last["future_pct_1d"] is None, (
        f"A最后一行future_pct_1d应为None，实际为{a_last['future_pct_1d']}"
    )
    assert a_last["future_pct_2d"] is None, (
        f"A最后一行future_pct_2d应为None，实际为{a_last['future_pct_2d']}"
    )

    # A 倒数第二行 (2024-01-03) 的 future_pct_1d 应为 -0.10，不能取到B的0.50
    a_second = result.filter(
        (pl.col("code") == "A") & (pl.col("trading_date") == date(2024, 1, 3))
    ).row(0, named=True)
    assert np.isclose(a_second["future_pct_1d"], -0.10), (
        f"A第二行future_pct_1d应为-0.10，实际为{a_second['future_pct_1d']}"
    )
    # A 第一行 (2024-01-02) 的 future_pct_2d = (1+0.20)*(1-0.10)-1 = 0.08
    a_first = result.filter(
        (pl.col("code") == "A") & (pl.col("trading_date") == date(2024, 1, 2))
    ).row(0, named=True)
    expected = (1.20 * 0.90) - 1.0
    assert np.isclose(a_first["future_pct_2d"], expected), (
        f"A第一行future_pct_2d应为{expected}，实际为{a_first['future_pct_2d']}"
    )

    # 同样的验证对B：B第一行的 future_pct_1d 应为 0.60（不是A的0.20）
    b_first = result.filter(
        (pl.col("code") == "B") & (pl.col("trading_date") == date(2024, 1, 2))
    ).row(0, named=True)
    assert np.isclose(b_first["future_pct_1d"], 0.60), (
        f"B第一行future_pct_1d应为0.60，实际为{b_first['future_pct_1d']}"
    )


# ============================================================
# 场景：各股票记录数不同（不等长 panel）
# ============================================================

def test_polars_panel_uneven_lengths():
    """
    股票 A 有 5 个交易日，股票 B 只有 3 个。
    shift 不能把 B 的 None 边界传染给 A。
    """
    df = pl.DataFrame({
        "code": ["A"] * 5 + ["B"] * 3,
        "trading_date": [
            date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4),
            date(2024, 1, 5), date(2024, 1, 8),  # A 比 B 多两天
            date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4),  # B 只有 3 天
        ],
        "pct": [0.01, 0.02, 0.03, 0.04, 0.05, 0.10, 0.20, 0.30],
    })

    result = add_future_return(df, horizons=(1, 3))

    # A 最后两天（2024-01-05, 2024-01-08）的 future_pct_3d 应为 None
    a_last_two = result.filter(
        (pl.col("code") == "A") &
        (pl.col("trading_date").is_in([date(2024, 1, 5), date(2024, 1, 8)]))
    ).select("future_pct_3d").to_series().to_list()
    assert all(v is None for v in a_last_two), (
        f"A 最后两天 future_pct_3d 应全为 None，实际为 {a_last_two}"
    )

    # A 第一行 (2024-01-02) 的 future_pct_3d 应正确累乘
    a_first = result.filter(
        (pl.col("code") == "A") & (pl.col("trading_date") == date(2024, 1, 2))
    ).row(0, named=True)
    # future_pct_3d = (1+r[1])*(1+r[2])*(1+r[3])-1 = 1.02*1.03*1.04-1 = 0.092624
    expected_a = (1.02 * 1.03 * 1.04) - 1.0
    assert np.isclose(a_first["future_pct_3d"], expected_a), (
        f"A第一行future_pct_3d应为{expected_a}，实际为{a_first['future_pct_3d']}"
    )


# ============================================================
# 场景：股票内日期不连续（有间隔）
# ============================================================

def test_polars_panel_non_contiguous_dates():
    """
    股票内日期不是连续日（中间跳了几天），
    shift(-1) 应只取下一行（下一个交易日），而不是按自然日跳。
    """
    df = pl.DataFrame({
        "code": ["A", "A", "A"],
        "trading_date": [date(2024, 1, 2), date(2024, 1, 5), date(2024, 1, 8)],  # 中间有间隔
        "pct": [0.10, 0.20, -0.10],
    })

    result = add_future_return(df, horizons=(1,))

    # future_pct_1d 应取排序后的下一行，即 0.20
    a_first = result.filter(
        (pl.col("code") == "A") & (pl.col("trading_date") == date(2024, 1, 2))
    ).row(0, named=True)
    assert np.isclose(a_first["future_pct_1d"], 0.20), (
        f"A第一行future_pct_1d应为0.20，实际为{a_first['future_pct_1d']}"
    )


# ============================================================
# 场景：Polars panel vs Pandas panel 大规模随机交叉验证
# ============================================================

def test_polars_panel_vs_pandas_random():
    """随机生成 panel 数据，Polars vs Pandas 逐值对比"""
    rng = np.random.default_rng(42)
    all_codes = [f"S{i:04d}" for i in range(50)]  # 50 只股票
    all_dates = [date(2024, 1, 2) + timedelta(days=int(i)) for i in range(60)]  # 60 个交易日

    records = []
    for code in all_codes:
        # 每只股票记录数略有不同 — 模拟真实场景
        n_dates = rng.integers(30, 61)
        for d in all_dates[:n_dates]:
            records.append({
                "code": code,
                "trading_date": d,
                "pct": round(float(rng.normal(0, 0.02)), 6),
            })

    df_pd = pd.DataFrame(records).sort_values(["code", "trading_date"]).reset_index(drop=True)
    df_pl = pl.DataFrame(records).sort(["code", "trading_date"])

    result_pl = add_future_return(df_pl, horizons=(1, 5, 10))
    result_pd = add_future_return(df_pd, horizons=(1, 5, 10))

    # 统一 trading_date 的 dtype，避免 merge 报错
    # Pandas (object) vs Polars→Pandas (datetime64[ms])
    pl_pd = result_pl.to_pandas()
    pl_pd["trading_date"] = pl_pd["trading_date"].dt.date
    result_pd["trading_date"] = result_pd["trading_date"].apply(
        lambda x: x.date() if isinstance(x, pd.Timestamp) else x
    )

    merged = result_pd.merge(
        pl_pd,
        on=["code", "trading_date"],
        suffixes=("_pd", "_pl"),
    )

    for h in [1, 5, 10]:
        col_pd = f"future_pct_{h}d_pd"
        col_pl = f"future_pct_{h}d_pl"
        mask_both_notna = merged[col_pd].notna() & merged[col_pl].notna()
        diff = (merged.loc[mask_both_notna, col_pd].values -
                merged.loc[mask_both_notna, col_pl].values)
        max_diff = np.max(np.abs(diff))
        assert max_diff < 1e-10, (
            f"horizon={h}: 最大差异={max_diff}, "
            f"Pd/Pl 不一致的样本数={(diff != 0).sum()}"
        )
    # 检查 None/NaN 对齐（双方应为同位置的缺失）
    for h in [1, 5, 10]:
        col_pd = f"future_pct_{h}d_pd"
        col_pl = f"future_pct_{h}d_pl"
        pd_nan = merged[col_pd].isna()
        pl_nan = merged[col_pl].isna()
        assert (pd_nan == pl_nan).all(), (
            f"horizon={h}: Pandas/Polars 的 NaN 位置不一致"
        )


# ============================================================
# 场景：无 code 列的 Polars 时序（is_panel=False）
# ============================================================

def test_polars_time_series_non_panel():
    """无 code 列时，Polars 按全局时序计算，.over() 不影响"""
    df = pl.DataFrame({
        "trading_date": [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4), date(2024, 1, 5)],
        "pct": [0.10, 0.20, -0.10, 0.25],
    })
    result = add_future_return(df, horizons=(1, 3))

    assert np.isclose(result["future_pct_1d"][0], 0.20)
    assert np.isclose(result["future_pct_3d"][0], (1.20 * 0.90 * 1.25) - 1.0)
    assert result["future_pct_1d"][3] is None
