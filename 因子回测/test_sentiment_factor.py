# -*- coding: utf-8 -*-
"""
情绪因子测试脚本
因子定义：情绪因子 = ts_rank(涨停家数 - 跌停家数, 252) + ts_rank(昨日涨停今日溢价, 252)
"""

import sys
import os
sys.path.append("c:/Users/20561/Desktop/策略")

from my_utils.fun import *
from my_utils.mapping import *
import polars as pl
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 输出目录
output_dir = "c:/Users/20561/Desktop/策略/因子回测/因子分析结果"
os.makedirs(output_dir, exist_ok=True)

print("="*80)
print("情绪因子测试开始")
print("="*80)

# ==================== 步骤1：读取数据 ====================
print("\n[步骤1] 读取日线数据...")
start_date = dt.date(2021, 1, 1)
end_date = dt.date(2025, 1, 1)

stock_data = read_day_data(start_date=start_date, end_date=end_date)
stock_data = stock_data.drop_nulls(subset=['open', 'close', 'pre_close', 'limit_up', 'limit_down'])
print(f"  数据加载完成: {stock_data.shape}")
print(f"  股票数量: {stock_data['code'].n_unique()}")
print(f"  交易日数量: {stock_data['trading_date'].n_unique()}")

# 标记涨停状态
print("\n[步骤2] 标记涨停状态...")
stock_data = mark_limit_status(stock_data)
print("  涨停状态标记完成")

# ==================== 步骤2：计算市场情绪指标 ====================
print("\n[步骤3] 计算每日涨跌停家数...")
daily_stats = stock_data.group_by('trading_date').agg([
    pl.col('is_limit_up').sum().alias('limit_up_count'),
    ((pl.col('close') <= pl.col('limit_down') * 1.001)).sum().alias('limit_down_count')
]).sort('trading_date')

daily_stats = daily_stats.with_columns([
    (pl.col('limit_up_count') - pl.col('limit_down_count')).alias('limit_spread')
])

print("  涨跌停统计前5行:")
print(daily_stats.head())

# 计算昨日涨停股票今日的溢价
print("\n[步骤4] 计算昨日涨停今日溢价...")
stock_data = stock_data.sort(['code', 'trading_date'])
stock_data = stock_data.with_columns([
    pl.col('is_limit_up').shift(1).over('code').alias('yesterday_limit_up')
])

stock_data = stock_data.with_columns([
    ((pl.col('close') - pl.col('pre_close')) / pl.col('pre_close') * 100).alias('today_return')
])

yesterday_limit_stocks = stock_data.filter(pl.col('yesterday_limit_up') == True)

daily_premium = yesterday_limit_stocks.group_by('trading_date').agg([
    pl.col('today_return').mean().alias('limit_up_premium'),
    pl.col('code').count().alias('limit_up_stock_count')
]).sort('trading_date')

print("  溢价统计前5行:")
print(daily_premium.head())

# 合并两个指标
market_sentiment = daily_stats.join(daily_premium, on='trading_date', how='left').sort('trading_date')
market_sentiment = market_sentiment.with_columns([
    pl.col('limit_up_premium').fill_null(0)
])
print(f"\n  市场情绪数据 shape: {market_sentiment.shape}")

# ==================== 步骤3：计算252日滚动百分位排名 ====================
print("\n[步骤5] 计算252日滚动百分位排名...")

def rolling_percent_rank(series, window=252):
    """计算滚动百分位排名（0-1）"""
    ranks = []
    for i in range(len(series)):
        if i < window - 1:
            ranks.append(np.nan)
        else:
            window_data = series.iloc[i-window+1:i+1]
            current_value = series.iloc[i]
            rank = (window_data <= current_value).sum() / len(window_data)
            ranks.append(rank)
    return pd.Series(ranks, index=series.index)

market_sentiment_pd = market_sentiment.to_pandas().set_index('trading_date')

market_sentiment_pd['limit_spread_rank'] = rolling_percent_rank(market_sentiment_pd['limit_spread'], 252)
market_sentiment_pd['limit_up_premium_rank'] = rolling_percent_rank(market_sentiment_pd['limit_up_premium'], 252)
market_sentiment_pd['sentiment_factor'] = (
    market_sentiment_pd['limit_spread_rank'] +
    market_sentiment_pd['limit_up_premium_rank']
)

print("  情绪因子计算完成，后10行:")
print(market_sentiment_pd[['limit_spread_rank', 'limit_up_premium_rank', 'sentiment_factor']].tail(10))

# ==================== 画出情绪因子时序图 ====================
print("\n[步骤6] 生成情绪因子时序图...")
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

axes[0].plot(market_sentiment_pd.index, market_sentiment_pd['limit_spread'], label='涨跌停家数差', color='blue', linewidth=1)
axes[0].set_title('涨跌停家数差', fontsize=14)
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(market_sentiment_pd.index, market_sentiment_pd['limit_up_premium'], label='昨日涨停今日溢价(%)', color='orange', linewidth=1)
axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
axes[1].set_title('昨日涨停今日溢价', fontsize=14)
axes[1].legend()
axes[1].grid(alpha=0.3)

axes[2].plot(market_sentiment_pd.index, market_sentiment_pd['sentiment_factor'], label='情绪因子', color='green', linewidth=2)
axes[2].set_title('情绪因子（涨跌停差排名 + 溢价排名）', fontsize=14)
axes[2].legend()
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{output_dir}/sentiment_factor_timeline.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  时序图已保存: {output_dir}/sentiment_factor_timeline.png")

# ==================== 步骤4：计算等权全市场指数 ====================
print("\n[步骤7] 计算等权全市场指数...")
stock_data = stock_data.sort(['code', 'trading_date'])
stock_data = stock_data.with_columns([
    ((pl.col('close') / pl.col('pre_close') - 1) * 100).alias('daily_return')
])

equal_weight_index = stock_data.group_by('trading_date').agg([
    pl.col('daily_return').mean().alias('equal_weight_return')
]).sort('trading_date').to_pandas().set_index('trading_date')

equal_weight_index['nav'] = (1 + equal_weight_index['equal_weight_return'] / 100).cumprod()

print("  等权全市场指数计算完成，前10行:")
print(equal_weight_index.head(10))

# 合并情绪因子和指数数据
analysis_data = pd.merge(
    market_sentiment_pd[['sentiment_factor', 'limit_spread_rank', 'limit_up_premium_rank']],
    equal_weight_index[['equal_weight_return', 'nav']],
    left_index=True, right_index=True, how='inner'
)

for period in [1, 5, 10, 20]:
    analysis_data[f'future_return_{period}d'] = (
        analysis_data['nav'].shift(-period) / analysis_data['nav'] - 1
    ) * 100

print(f"\n  分析数据 shape: {analysis_data.shape}")

# ==================== 步骤5：相关性分析 ====================
print("\n" + "="*80)
print("情绪因子与未来收益的相关性分析")
print("="*80)

corr_results = []
for period in [1, 5, 10, 20]:
    corr = analysis_data['sentiment_factor'].corr(analysis_data[f'future_return_{period}d'])
    corr_results.append({
        '持仓周期': f'{period}天',
        '相关系数': corr
    })

corr_df = pd.DataFrame(corr_results)
print("\n情绪因子与未来收益的相关性:")
print(corr_df.to_string(index=False))

print("\n涨跌停家数差排名与未来收益的相关性:")
for period in [1, 5, 10, 20]:
    corr = analysis_data['limit_spread_rank'].corr(analysis_data[f'future_return_{period}d'])
    print(f"  {period}天: {corr:.4f}")

print("\n溢价排名与未来收益的相关性:")
for period in [1, 5, 10, 20]:
    corr = analysis_data['limit_up_premium_rank'].corr(analysis_data[f'future_return_{period}d'])
    print(f"  {period}天: {corr:.4f}")

# 散点图
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
periods = [1, 5, 10, 20]

for i, period in enumerate(periods):
    ax = axes[i//2, i%2]
    valid_data = analysis_data.dropna(subset=['sentiment_factor', f'future_return_{period}d'])
    ax.scatter(valid_data['sentiment_factor'], valid_data[f'future_return_{period}d'],
               alpha=0.5, s=20)

    if len(valid_data) > 0:
        z = np.polyfit(valid_data['sentiment_factor'], valid_data[f'future_return_{period}d'], 1)
        p = np.poly1d(z)
        x_range = np.linspace(valid_data['sentiment_factor'].min(),
                               valid_data['sentiment_factor'].max(), 100)
        ax.plot(x_range, p(x_range), "r--", alpha=0.8, linewidth=2)

    corr = analysis_data['sentiment_factor'].corr(analysis_data[f'future_return_{period}d'])
    ax.set_title(f'情绪因子 vs 未来{period}日收益 (corr={corr:.4f})', fontsize=12)
    ax.set_xlabel('情绪因子')
    ax.set_ylabel(f'未来{period}日收益(%)')
    ax.grid(alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig(f"{output_dir}/sentiment_factor_scatter.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  散点图已保存: {output_dir}/sentiment_factor_scatter.png")

# ==================== 步骤6：分组测试 ====================
print("\n" + "="*80)
print("分组测试")
print("="*80)

period = 5
analysis_data_clean = analysis_data.dropna(subset=['sentiment_factor']).copy()
analysis_data_clean['factor_group'] = pd.qcut(analysis_data_clean['sentiment_factor'],
                                               q=5, labels=['G1', 'G2', 'G3', 'G4', 'G5'])

group_returns = analysis_data_clean.groupby('factor_group')[f'future_return_{period}d'].agg([
    'mean', 'std', 'count'
])
group_returns.columns = ['平均收益(%)', '收益标准差', '样本数']

print(f"\n情绪因子分组未来{period}日收益统计:")
print(group_returns.round(4))

# 画柱状图
plt.figure(figsize=(10, 6))
colors = ['#e74c3c', '#e67e22', '#f1c40f', '#27ae60', '#2980b9']
bars = plt.bar(group_returns.index, group_returns['平均收益(%)'], color=colors, alpha=0.7)
plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height>0 else -0.05),
             f'{height:.4f}%', ha='center', va='bottom' if height>0 else 'top')

plt.title(f'情绪因子分组未来{period}日平均收益', fontsize=14)
plt.xlabel('因子分组（G1最低，G5最高）', fontsize=12)
plt.ylabel('平均收益(%)', fontsize=12)
plt.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f"{output_dir}/sentiment_factor_group_return.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  分组收益图已保存: {output_dir}/sentiment_factor_group_return.png")

# ==================== 步骤7：择时策略回测 ====================
print("\n" + "="*80)
print("择时策略回测")
print("="*80)

def backtest_timing_strategy(data, factor_col, signal_type='high', hold_period=5, threshold=0.8):
    """择时策略回测"""
    data = data.copy().dropna(subset=[factor_col])

    if signal_type == 'high':
        threshold_value = data[factor_col].quantile(threshold)
        data['signal'] = (data[factor_col] >= threshold_value).astype(int)
    else:
        threshold_value = data[factor_col].quantile(1 - threshold)
        data['signal'] = (data[factor_col] <= threshold_value).astype(int)

    data['position'] = 0
    in_position = False
    hold_days = 0

    for i in range(len(data)):
        if not in_position and data['signal'].iloc[i] == 1:
            in_position = True
            hold_days = 0

        if in_position:
            data['position'].iloc[i] = 1
            hold_days += 1
            if hold_days >= hold_period:
                in_position = False

    data['strategy_return'] = data['position'] * data['equal_weight_return'] / 100
    data['benchmark_return'] = data['equal_weight_return'] / 100
    data['strategy_nav'] = (1 + data['strategy_return']).cumprod()
    data['benchmark_nav'] = (1 + data['benchmark_return']).cumprod()

    return data

hold_period = 5
threshold = 0.8

strategy_high = backtest_timing_strategy(
    analysis_data, 'sentiment_factor', signal_type='high',
    hold_period=hold_period, threshold=threshold
)

strategy_low = backtest_timing_strategy(
    analysis_data, 'sentiment_factor', signal_type='low',
    hold_period=hold_period, threshold=threshold
)

def calculate_performance_metrics(data, return_col, nav_col):
    """计算策略表现指标"""
    returns = data[return_col].dropna()
    nav = data[nav_col].dropna()

    if len(returns) == 0:
        return None

    total_return = nav.iloc[-1] - 1
    years = len(returns) / 252
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    daily_rf = 0.03 / 252
    excess_return = returns - daily_rf
    sharpe = np.sqrt(252) * excess_return.mean() / returns.std() if returns.std() > 0 else 0

    peak = nav.expanding().max()
    drawdown = (nav - peak) / peak
    max_drawdown = drawdown.min()

    win_rate = (returns > 0).mean()
    position_days = (data['position'] == 1).sum()
    position_ratio = position_days / len(data)

    return {
        '累计收益': f'{total_return*100:.2f}%',
        '年化收益': f'{annual_return*100:.2f}%',
        '夏普比率': f'{sharpe:.2f}',
        '最大回撤': f'{max_drawdown*100:.2f}%',
        '胜率': f'{win_rate*100:.2f}%',
        '持仓占比': f'{position_ratio*100:.2f}%'
    }

print("\n" + "="*60)
print("策略1：情绪因子高时做多")
print("="*60)
metrics_high = calculate_performance_metrics(strategy_high, 'strategy_return', 'strategy_nav')
for k, v in metrics_high.items():
    print(f"  {k}: {v}")

print("\n" + "="*60)
print("策略2：情绪因子低时做多（逆向策略）")
print("="*60)
metrics_low = calculate_performance_metrics(strategy_low, 'strategy_return', 'strategy_nav')
for k, v in metrics_low.items():
    print(f"  {k}: {v}")

print("\n" + "="*60)
print("基准：买入持有")
print("="*60)
benchmark_data = strategy_high.copy()
benchmark_data['position'] = 1
benchmark_data['strategy_return'] = benchmark_data['benchmark_return']
benchmark_data['strategy_nav'] = benchmark_data['benchmark_nav']
metrics_benchmark = calculate_performance_metrics(benchmark_data, 'strategy_return', 'strategy_nav')
for k, v in metrics_benchmark.items():
    print(f"  {k}: {v}")

# 画出净值曲线
plt.figure(figsize=(14, 7))

plt.plot(strategy_high.index, strategy_high['benchmark_nav'],
         label='买入持有（基准）', color='gray', linewidth=2, linestyle='--')
plt.plot(strategy_high.index, strategy_high['strategy_nav'],
         label='情绪高时做多', color='red', linewidth=2)
plt.plot(strategy_low.index, strategy_low['strategy_nav'],
         label='情绪低时做多（逆向）', color='blue', linewidth=2)

plt.title('情绪因子择时策略净值对比', fontsize=16)
plt.xlabel('日期', fontsize=12)
plt.ylabel('净值（初始=1）', fontsize=12)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{output_dir}/sentiment_factor_nav.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  净值曲线图已保存: {output_dir}/sentiment_factor_nav.png")

# 保存结果数据
analysis_data.to_csv(f"{output_dir}/sentiment_factor_data.csv", encoding='utf-8-sig')
print(f"  分析数据已保存: {output_dir}/sentiment_factor_data.csv")

# 保存策略表现汇总
summary_df = pd.DataFrame({
    '策略': ['情绪高时做多', '情绪低时做多', '买入持有'],
    '累计收益': [metrics_high['累计收益'], metrics_low['累计收益'], metrics_benchmark['累计收益']],
    '年化收益': [metrics_high['年化收益'], metrics_low['年化收益'], metrics_benchmark['年化收益']],
    '夏普比率': [metrics_high['夏普比率'], metrics_low['夏普比率'], metrics_benchmark['夏普比率']],
    '最大回撤': [metrics_high['最大回撤'], metrics_low['最大回撤'], metrics_benchmark['最大回撤']],
    '胜率': [metrics_high['胜率'], metrics_low['胜率'], metrics_benchmark['胜率']],
    '持仓占比': [metrics_high['持仓占比'], metrics_low['持仓占比'], metrics_benchmark['持仓占比']]
})
summary_df.to_csv(f"{output_dir}/sentiment_factor_summary.csv", index=False, encoding='utf-8-sig')
print(f"  策略汇总已保存: {output_dir}/sentiment_factor_summary.csv")

print("\n" + "="*80)
print("情绪因子测试完成！")
print("="*80)
print(f"\n结果文件保存在: {output_dir}")
print("  - sentiment_factor_timeline.png    情绪因子时序图")
print("  - sentiment_factor_scatter.png     散点图")
print("  - sentiment_factor_group_return.png  分组收益图")
print("  - sentiment_factor_nav.png         净值曲线图")
print("  - sentiment_factor_data.csv        分析数据")
print("  - sentiment_factor_summary.csv     策略汇总")
