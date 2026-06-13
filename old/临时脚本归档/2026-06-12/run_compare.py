"""
回测对比脚本：
版本A（当前）：last_limit_desc != "1天1板"
版本B（候选）：last_limit_desc == "1天1板"

对两版分别跑2026年回测并对比
"""
import datetime as dt
import warnings
warnings.filterwarnings('ignore')

from my_utils.fun import *
from my_utils.trade_fun import *

# 日志
logging = get_logger(log_file='log/回测对比.log', inherit=False)

# ===== 1. 数据准备 =====
start_date = dt.date(2024, 1, 1)
end_date = dt.datetime.today()

stock_data = read_day_data(start_date=start_date, end_date=end_date, file_path='gm_stock_all_data')
stock_data = stock_data.with_columns([
    (pl.col('mv_A_free_float') / 1e8).alias('mv_A_free_float'),
    (pl.col('total_mv') / 1e8).alias('total_mv')
])
logging.info(f"数据加载完成, 共{len(stock_data)}行")

# ===== 2. 特征计算 =====
# 涨停标记
stock_data = mark_limit_status(stock_data)
stock_data = mark_limit_desc(stock_data)
stock_data = mark_last_limit_desc(stock_data)
stock_data = cal_limit_avg_turnover(stock_data, window=5, turnover_col='turnover_rate')

# 均线
stock_data = add_sma(stock_data, window=5)
stock_data = add_sma(stock_data, window=7)

# 其他特征
stock_data = stock_data.with_columns(
    ((pl.col("open") - pl.col("pre_close")) / pl.col("pre_close") * 100).alias("open_pct"),
    ((pl.col("close") - pl.col("sma_7")) / pl.col("sma_7") * 100).alias("close_sma7_pct"),
    (pl.col("amount") * 100 / pl.col("volume")).alias("vwap"),
    ((pl.col("low") <= pl.col("limit_down") * 1.01)).alias("touch_limit_down"),
    (pl.col("close") / pl.col("pre_close") - 1).alias("pct"),
)
stock_data = cal_n_lowest(stock_data)

# 排序与移位
stock_data = stock_data.sort(["code", "trading_date"])
stock_data = stock_data.with_columns([
    pl.col("limit_status").shift(1).over("code").alias("prev_limit_status"),
    pl.col("sma_7").shift(1).over("code").alias("prev_sma_7"),
    pl.col("pct").shift(1).over("code").alias("pre_pct"),
    pl.col("vwap").shift(1).over("code").alias("pre_vwap"),
    pl.col("close_sma7_pct").shift(1).over("code").alias("pre_close_sma7_pct"),
])

# ===== 3. 公共筛选条件 =====
params_dict = {
    'low': -5,
    'high': -2.5,
    'mv_min': 35,
    'mv_max': 1000,
    'prev_limit_status': ['断板', '炸板'],
    'avg_limit_turnover_5_min': -1
}

def make_signal(stock_data, version='A'):
    """生成信号，version='A'表示!=, version='B'表示=="""
    if version == 'A':
        last_limit_cond = (pl.col("last_limit_desc") != "1天1板")
        tag = "排除1天1板"
    else:
        last_limit_cond = (pl.col("last_limit_desc") == "1天1板")
        tag = "仅1天1板"

    stock_data = stock_data.with_columns(
        signal=pl.when(
            ~(pl.col("is_st")) &
            ~(pl.col("code").str.split(".").list[1].str.starts_with("30") |
              pl.col("code").str.split(".").list[1].str.starts_with("688") |
              pl.col("code").str.split(".").list[1].str.starts_with("90") |
              pl.col("code").str.split(".").list[1].str.starts_with("20")
              ) &
            (pl.col("prev_limit_status").is_in(params_dict['prev_limit_status'])) &
            (pl.col("open_pct") >= params_dict["low"]) &
            (pl.col("open_pct") <= params_dict["high"]) &
            (pl.col("pre_close") >= pl.col("prev_sma_7")) &
            last_limit_cond &
            (pl.col("last_limit_desc").is_not_null()) &
            (pl.col("mv_A_free_float") >= params_dict["mv_min"]) &
            (pl.col("mv_A_free_float") <= params_dict["mv_max"]) &
            ((pl.col("open") / pl.col("lowest_30")) <= 3)
        ).then(1).otherwise(0)
    )
    signal_df = stock_data.filter(pl.col("signal") == 1)
    return signal_df, tag


# ===== 4. 生成两种版本的信号 =====
signal_A, tag_A = make_signal(stock_data, 'A')
signal_B, tag_B = make_signal(stock_data, 'B')

logging.info(f"版本A({tag_A}): 共{len(signal_A)}条信号")
logging.info(f"版本B({tag_B}): 共{len(signal_B)}条信号")

# ===== 5. 分别跑回测 =====
def run_backtest(signal_df, tag, version_label):
    """运行完整的回测流程，返回结果DataFrame"""
    start_date_str = "2026-01-01"
    end_date_str = dt.datetime.today().strftime("%Y-%m-%d")

    logging.info(f"\n{'='*60}")
    logging.info(f"回测版本: {version_label} ({tag})")
    logging.info(f"回测区间: {start_date_str} 至 {end_date_str}")
    logging.info(f"{'='*60}")

    result_df, merged_df = cal_trade_info(
        信号文件=signal_df,
        trade_fun=trade,
        start_date=start_date_str,
        end_date=end_date_str
    )

    # 收益率加权处理（沿用原代码0.4权重）
    merged_df = merged_df.with_columns(
        (pl.col("profit") * 0.4).alias("weight_profit")
    )

    # 打印回测报告 - 用plot=True让logging输出指标，然后捕获
    back_metrics = report_backtest_full(
        merged_df.to_pandas(),
        start_date=start_date_str,
        end_date=end_date_str,
        profit_col='weight_profit',
        plot=False
    )
    # back_metrics 是 metrics_df，手动打印关键指标
    logging.info(f"\n=== {version_label} ({tag}) 完整回测指标 ===")
    for _, row in back_metrics.iterrows():
        logging.info(f"{row['指标名称']}: {row['指标值']}")
    logging.info("=" * 40)

    return merged_df, back_metrics


# 跑版本A
merged_A, metrics_A_df = run_backtest(signal_A, tag_A, "版本A(当前)")
# 跑版本B
merged_B, metrics_B_df = run_backtest(signal_B, tag_B, "版本B(候选)")

# ===== 6. 汇总对比 =====
logging.info(f"\n{'='*60}")
logging.info(f"汇总对比")
logging.info(f"{'='*60}")

# 从report_backtest_full提取关键指标对比
def extract_backtest_key_metrics(metrics_df, tag):
    """从report_backtest_full返回的DataFrame提取关键指标"""
    d = {'版本': tag}
    for _, row in metrics_df.iterrows():
        d[row['指标名称']] = row['指标值']
    return d

key_metrics_A = extract_backtest_key_metrics(metrics_A_df, tag_A)
key_metrics_B = extract_backtest_key_metrics(metrics_B_df, tag_B)
compare_keys = ['策略总收益率', '策略年化收益率', '策略胜率', '策略盈亏比', '夏普比率', '最大回撤', '最终净值', '平均持仓天数']
compare_rows = []
for m in [key_metrics_A, key_metrics_B]:
    row = {'版本': m['版本']}
    for k in compare_keys:
        row[k] = m.get(k, '')
    compare_rows.append(row)

compare_df = pd.DataFrame(compare_rows)
logging.info("\n完整回测指标对比：\n" + compare_df.to_string(index=False))
print("\n========== 完整回测指标对比 ==========")
print(compare_df.to_string(index=False))
print("=======================================")

# 原有的逐笔统计指标
def extract_metrics(merged_df, tag):
    """从merged_df提取关键指标"""
    pdf = merged_df.to_pandas()
    n_trades = len(pdf)
    win_rate = (pdf['weight_profit'] > 0).mean() * 100
    avg_profit = pdf['weight_profit'].mean()
    profit_std = pdf['weight_profit'].std()
    total_profit = pdf['weight_profit'].sum()
    median_profit = pdf['weight_profit'].median()

    # 盈亏比
    wins = pdf[pdf['weight_profit'] > 0]['weight_profit']
    losses = pdf[pdf['weight_profit'] < 0]['weight_profit']
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = abs(losses.mean()) if len(losses) > 0 else 0
    profit_loss_ratio = avg_win / avg_loss if avg_loss != 0 else float('inf')

    return {
        '版本': tag,
        '交易次数': n_trades,
        '胜率(%)': round(win_rate, 2),
        '平均收益(%)': round(avg_profit, 2),
        '总收益(%)': round(total_profit, 2),
        '收益中位数(%)': round(median_profit, 2),
        '盈亏比': round(profit_loss_ratio, 2),
        '收益标准差': round(profit_std, 2),
    }

metrics_A = extract_metrics(merged_A, tag_A)
metrics_B = extract_metrics(merged_B, tag_B)

# 打印对比表
import pandas as pd
compare_df2 = pd.DataFrame([metrics_A, metrics_B])
logging.info("\n逐笔交易统计对比：\n" + compare_df2.to_string(index=False))
print("\n--- 逐笔交易统计对比 ---")
print(compare_df2.to_string(index=False))
