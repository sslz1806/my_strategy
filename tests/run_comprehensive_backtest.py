"""
全面时序因子分组回测检验脚本。

在新进程中完整跑：读取数据 -> 构建因子 -> 获取多指数收益 ->
合并各指数 -> backtest_timeseries_factor -> 检查 G1 收益等关键统计。

重点关注:
  1. limit_down_next_ret 的 G1 累计收益是否非零
  2. 所有 benchmark 和 horizon 组合是否都成功
  3. 是否有其他列 (G2-G5) 也出现不合理值
"""
import importlib
import sys
import traceback
import warnings
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ========== 项目根目录 ==========
for candidate in (Path.cwd(), *Path.cwd().parents):
    if (candidate / "my_utils").is_dir():
        PROJECT_ROOT = candidate
        break
else:
    raise RuntimeError("未找到项目根目录")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from my_utils.fun import read_day_data
from my_utils.rqdata import RqData

# 按 reload 方式导入 timing_engine (确保用最新版)
TE_PATH = (PROJECT_ROOT / "因子回测" / "涨跌停情绪因子" / "timing_engine.py")
TE_SPEC = importlib.util.spec_from_file_location("timing_engine", str(TE_PATH))
timing_engine = importlib.util.module_from_spec(TE_SPEC)
TE_SPEC.loader.exec_module(timing_engine)

prepare_stock_daily = timing_engine.prepare_stock_daily
build_daily_sentiment_factors = timing_engine.build_daily_sentiment_factors

from 因子回测.alpha import backtest_timeseries_factor

# ========== 参数 ==========
START_DATE = date(2018, 1, 2)
END_DATE = date(2026, 7, 27)
DATA_SOURCE = "rq_stock_all_data"
WINDOW = 5
HORIZONS = (1, 3, 5, 10)
MIN_HISTORY = 252

FACTOR_COLUMNS = [
    "limit_up_ratio",
    "limit_down_ratio",
    "limit_up_next_ret",
    "limit_down_next_ret",
]
FACTOR_LABELS = {
    "limit_up_ratio": "涨停占比",
    "limit_down_ratio": "跌停占比",
    "limit_up_next_ret": "涨停次日收益",
    "limit_down_next_ret": "跌停次日收益",
}

BENCHMARK_CODES = {
    "hs300": "000300.XSHG",
    "zz500": "000905.XSHG",
    "zz1000": "000852.XSHG",
    "zz2000": "000906.XSHG",
    "cyb": "399006.XSHE",
    "kc50": "000688.XSHG",
}
BENCHMARK_LABELS = {
    "hs300": "沪深300", "zz500": "中证500", "zz1000": "中证1000",
    "zz2000": "中证2000", "cyb": "创业板指", "kc50": "科创50",
}
BENCHMARK_ORDER = ["hs300", "zz500", "zz1000", "zz2000", "cyb", "kc50"]

# ================================================================
# 1. 构建情绪因子
# ================================================================
print("=" * 70)
print("阶段 1/3: 读取原始日线并构建情绪因子")
print("=" * 70)

daily_fields = [
    "code", "trading_date", "close", "pre_close", "limit_up", "limit_down",
    "is_st", "is_suspended", "total_mv",
]
daily_raw = read_day_data(
    START_DATE, END_DATE, fields=daily_fields, file_path=DATA_SOURCE,
).sort(["code", "trading_date"])

print(f"  原始日线行数: {len(daily_raw)}")
prepared_daily, trading_calendar = prepare_stock_daily(daily_raw)
print(f"  合格股票日线行数: {len(prepared_daily)}")
print(f"  交易日历行数: {len(trading_calendar)}")

factor_daily = build_daily_sentiment_factors(prepared_daily, trading_calendar, window=WINDOW)

factor_data = (
    factor_daily.to_pandas()
    .assign(trading_date=lambda d: pd.to_datetime(d["trading_date"]))
    .sort_values("trading_date")
    .reset_index(drop=True)
)

# 裁剪到有效范围
factor_data = factor_data.dropna(subset=FACTOR_COLUMNS, how="all").reset_index(drop=True)
print(f"  因子数据范围: {factor_data['trading_date'].min()} ~ {factor_data['trading_date'].max()}")
print(f"  有效交易日: {len(factor_data)}")

# 对各因子做基本统计
print("\n  --- 因子基本统计 ---")
for col in FACTOR_COLUMNS:
    valid = factor_data[col].dropna()
    print(f"  {col:25s}: 有效值={len(valid):5d}, 均值={valid.mean():.6f}, "
          f"std={valid.std():.6f}, min={valid.min():.6f}, max={valid.max():.6f}")

# ================================================================
# 2. 获取多指数收益
# ================================================================
print("\n" + "=" * 70)
print("阶段 2/3: 获取多指数收益 (RQData)")
print("=" * 70)

rq_client = RqData()
rq_returns = rq_client.get_return(
    list(BENCHMARK_CODES.values()), START_DATE, END_DATE,
)

normalized = (rq_returns.reset_index()[["order_book_id", "date", "return"]].copy())
normalized["date"] = pd.to_datetime(normalized["date"], errors="raise")
normalized["return"] = pd.to_numeric(normalized["return"], errors="raise").astype(float)

# 过滤 NaN/inf
bad_mask = ~np.isfinite(normalized["return"].to_numpy())
if bad_mask.any():
    bad_rows = normalized.loc[bad_mask]
    print(f"  ⚠ 跳过 {len(bad_rows)} 行 NaN/inf 指数收益")
    normalized = normalized.loc[~bad_mask].copy()

benchmark_returns = {}
for label, code in BENCHMARK_CODES.items():
    single = (
        normalized[normalized["order_book_id"] == code]
        [["date", "return"]]
        .rename(columns={"date": "trading_date", "return": "benchmark_ret"})
        .sort_values("trading_date")
        .reset_index(drop=True)
    )
    if single.empty:
        raise ValueError(f"指数 {label}({code}) 无有效收益数据")
    benchmark_returns[label] = single

common_start = max(d["trading_date"].min() for d in benchmark_returns.values())
common_end = min(d["trading_date"].max() for d in benchmark_returns.values())
for label in benchmark_returns:
    benchmark_returns[label] = (
        benchmark_returns[label]
        .query("trading_date >= @common_start and trading_date <= @common_end")
        .reset_index(drop=True)
    )
print(f"  指数共同区间: {common_start.date()} ~ {common_end.date()}")
print(f"  交易日数: {len(benchmark_returns['hs300'])}")

for label in BENCHMARK_ORDER:
    br = benchmark_returns[label]
    print(f"  {label:8s} ({BENCHMARK_LABELS[label]:6s}): {br['trading_date'].min().date()} ~ "
          f"{br['trading_date'].max().date()}, n={len(br)}")

# ================================================================
# 3. 批量回测
# ================================================================
print("\n" + "=" * 70)
print("阶段 3/3: 批量回测 (backtest_timeseries_factor)")
print("=" * 70)

failed_combos = []
all_results = []

total = len(FACTOR_COLUMNS) * len(BENCHMARK_ORDER) * len(HORIZONS)
count = 0

for factor in FACTOR_COLUMNS:
    for benchmark in BENCHMARK_ORDER:
        # 合并因子与指数收益
        analysis_data = pd.merge(
            factor_data[["trading_date", factor]],
            benchmark_returns[benchmark],
            on="trading_date",
            how="inner",
        ).set_index("trading_date")

        analysis_data["ret_pct"] = analysis_data["benchmark_ret"] * 100

        for horizon in HORIZONS:
            count += 1
            try:
                result = backtest_timeseries_factor(
                    analysis_data,
                    factor_col=factor,
                    index_ret_col="ret_pct",
                    q=5,
                    hold_period=horizon,
                    plot=False,
                    verbose=False,
                )

                # 提取关键指标
                perf = result["group_performance"]
                group_stats = result["group_stats"]
                nav = result["group_nav"]

                groups = [g for g in perf.index if g.startswith("G")]
                if len(groups) < 2:
                    failed_combos.append({
                        "factor": factor, "benchmark": benchmark, "horizon": horizon,
                        "reason": f"实际分组数={len(groups)}"
                    })
                    continue

                g1 = perf.loc["G1"]
                g_last = perf.loc[groups[-1]]
                bm = perf.loc["买入持有基准"]
                g1_return = g1["累计收益"]

                # 单调性检查
                cum_vals = [perf.loc[g, "累计收益"] for g in groups]
                monotonic_up = all(
                    cum_vals[i] <= cum_vals[i + 1] for i in range(len(cum_vals) - 1)
                )

                # G1~G5 全部累计收益
                group_returns_dict = {g: round(perf.loc[g, "累计收益"], 4) for g in groups}

                all_results.append({
                    "factor": factor,
                    "benchmark": benchmark,
                    "horizon": horizon,
                    "n_groups": len(groups),
                    "G1_累计收益": round(g1_return, 4),
                    "Gq_累计收益": round(g_last["累计收益"], 4),
                    "Gq-G1": round(g_last["累计收益"] - g1_return, 4),
                    "G1_夏普": round(g1["夏普比率"], 4),
                    "基准_累计收益": round(bm["累计收益"], 4),
                    "单调递增": monotonic_up,
                    "G_all": str(group_returns_dict),
                    "n_obs": int(group_stats["样本数"].sum()),
                })

                if count % 10 == 0:
                    print(f"  进度: {count}/{total}")

            except Exception as e:
                failed_combos.append({
                    "factor": factor, "benchmark": benchmark, "horizon": horizon,
                    "reason": f"{type(e).__name__}: {str(e)[:200]}"
                })
                print(f"  X 失败 [{count}/{total}]: {factor}, {benchmark}, {horizon}d -> {type(e).__name__}: {e}")

print(f"\n  总组合: {total}, 成功: {len(all_results)}, 失败: {len(failed_combos)}")

# ================================================================
# 4. 汇总与检查
# ================================================================
print("\n" + "=" * 70)
print("检查结果汇总")
print("=" * 70)

results_df = pd.DataFrame(all_results)
failed_df = pd.DataFrame(failed_combos) if failed_combos else pd.DataFrame()

# --- 4a. 失败组合 ---
if len(failed_df) > 0:
    print(f"\n{'='*70}")
    print(f"【检查 1】失败组合 ({len(failed_combos)} / {total})")
    print(f"{'='*70}")
    print(failed_df.to_string(index=False))
else:
    print(f"\n{'='*70}")
    print("【检查 1】所有 {total} 个组合全部成功！")
    print(f"{'='*70}")

# --- 4b. limit_down_next_ret 的 G1 收益 ---
print(f"\n{'='*70}")
print("【检查 2】limit_down_next_ret 的 G1 累计收益")
print(f"{'='*70}")

ldnr_results = results_df[results_df["factor"] == "limit_down_next_ret"]
if len(ldnr_results) > 0:
    ldnr_g1 = ldnr_results["G1_累计收益"]
    print(f"  G1 累计收益: min={ldnr_g1.min():.4f}%, max={ldnr_g1.max():.4f}%, "
          f"mean={ldnr_g1.mean():.4f}%, median={ldnr_g1.median():.4f}%")
    zero_g1 = (ldnr_g1.abs() < 0.001).sum()
    print(f"  接近零(=0)的组合数: {zero_g1} / {len(ldnr_g1)}")
    if zero_g1 > 0:
        print("  ⚠ 以下组合 G1 累计收益接近零：")
        near_zero = ldnr_results[ldnr_results["G1_累计收益"].abs() < 0.001]
        for _, r in near_zero.iterrows():
            print(f"    {r['benchmark']:8s}, hold={r['horizon']}d, "
                  f"G1={r['G1_累计收益']:.4f}%, Gq={r['Gq_累计收益']:.4f}%, 基准={r['基准_累计收益']:.4f}%")
    else:
        print("  ✓ 所有组合 G1 累计收益均非零，正常运行")
else:
    print("  ⚠ 无 limit_down_next_ret 的回测结果")

# --- 4c. 各因子 G1 收益分布 ---
print(f"\n{'='*70}")
print("【检查 3】各因子 G1 累计收益跨指数分布")
print(f"{'='*70}")

for factor in FACTOR_COLUMNS:
    sub = results_df[results_df["factor"] == factor]
    if len(sub) == 0:
        continue
    g1_vals = sub["G1_累计收益"]
    gq_vals = sub["Gq_累计收益"]
    spread = sub["Gq-G1"]
    print(f"\n  {factor} ({FACTOR_LABELS[factor]}):")
    print(f"    G1 累计收益: min={g1_vals.min():7.2f}%, max={g1_vals.max():7.2f}%, "
          f"mean={g1_vals.mean():7.2f}%")
    print(f"    Gq 累计收益: min={gq_vals.min():7.2f}%, max={gq_vals.max():7.2f}%, "
          f"mean={gq_vals.mean():7.2f}%")
    print(f"    Gq-G1 多空差: min={spread.min():7.2f}%, max={spread.max():7.2f}%, "
          f"mean={spread.mean():7.2f}%")

# --- 4d. 查看哪些组合 G1 收益为正（不合理：G1 应是最差的） ---
print(f"\n{'='*70}")
print("【检查 4】G1 累计收益为正的组合（可能不合理：G1 应是最差的分组）")
print(f"{'='*70}")
positive_g1 = results_df[results_df["G1_累计收益"] > 0]
if len(positive_g1) > 0:
    print(f"  共 {len(positive_g1)} 个组合 G1 收益为正:")
    for _, r in positive_g1.iterrows():
        print(f"    {r['factor']:25s} | {r['benchmark']:8s} | hold={r['horizon']}d | "
              f"G1={r['G1_累计收益']:7.2f}% | Gq={r['Gq_累计收益']:7.2f}% | 基准={r['基准_累计收益']:7.2f}%")
else:
    print("  ✓ 所有组合 G1 收益均为负，符合预期（G1 应是最差分组）")

# --- 4e. 查看 Gq-G1 为负或接近零的组合 ---
print(f"\n{'='*70}")
print("【检查 5】Gq-G1 多空差 ≤ 0 的组合（因子无区分力）")
print(f"{'='*70}")
bad_spread = results_df[results_df["Gq-G1"] <= 0]
if len(bad_spread) > 0:
    print(f"  共 {len(bad_spread)} 个组合:")
    for _, r in bad_spread.iterrows():
        print(f"    {r['factor']:25s} | {r['benchmark']:8s} | hold={r['horizon']}d | "
              f"Gq-G1={r['Gq-G1']:7.2f}%")
else:
    print("  ✓ 所有组合 Gq-G1 > 0，因子有正区分力")

# --- 4f. 跨因子跨指数摘要 ---
print(f"\n{'='*70}")
print("跨指数平均 Gq-G1 多空差（按因子 x 持仓期）")
print(f"{'='*70}")
pivot = results_df.pivot_table(
    index=["factor", "horizon"], values="Gq-G1", aggfunc="mean"
).round(2)
print(pivot.to_string())

print(f"\n{'='*70}")
print("跨指数平均 G1 累计收益（按因子 x 持仓期）")
print(f"{'='*70}")
pivot_g1 = results_df.pivot_table(
    index=["factor", "horizon"], values="G1_累计收益", aggfunc="mean"
).round(2)
print(pivot_g1.to_string())

# --- 4g. 打印所有 G_all 列检查 G2-G5 ---
print(f"\n{'='*70}")
print("【检查 6】各组累计收益完整明细（检查 G2-G5 是否异常）")
print(f"{'='*70}")
for factor in FACTOR_COLUMNS:
    print(f"\n  --- {factor} ({FACTOR_LABELS[factor]}) ---")
    sub = results_df[results_df["factor"] == factor]
    for _, r in sub.iterrows():
        print(f"    {r['benchmark']:8s} hold={r['horizon']}d: {r['G_all']}")

print("\n" + "=" * 70)
print("检查完成。")
print("=" * 70)
