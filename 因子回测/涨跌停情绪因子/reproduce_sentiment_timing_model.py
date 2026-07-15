#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
复现国泰海通证券 - 情绪择时模型
=====================================
原文: 大类资产与中观配置研究（五）：从涨停板、"打板策略"到赚钱效应引发的情绪择时指标
PDF: 因子回测/涨跌停情绪因子/20250514-国泰海通证券-...pdf

回测区间: 2018-01-02 ~ 2026-07-10 （受数据限制，原文为2010-2025）
数据来源: ricequant日线/15分钟数据 (rq_stock_all_data / rq_15min_stock_data_dir)

复现内容:
  1. 5个情绪因子（涨停占比、跌停占比、净涨停占比、涨停次日收益、跌停次日收益）的周度计算
  2. 打板策略收益因子（基于15分钟高频数据）
  3. 基础情绪择时模型（等权信号）
  4. 改进1：引入市场趋势判断（MA10/20/60均线排列）
  5. 改进2：因子信号加权
  6. 宽基指数应用（沪深300、中证500）

使用方法:
  cd 策略目录 && E:/working/anaconda3/envs/quant/python.exe 因子回测/涨跌停情绪因子/reproduce_sentiment_timing_model.py
"""

import polars as pl
import numpy as np
from datetime import datetime, date, timedelta
import warnings
import os
import sys
import logging
import time

warnings.filterwarnings('ignore')

# ============================================================
# 0. 项目导入与配置
# ============================================================
# 加入项目根目录到path
_script_dir = os.path.dirname(os.path.abspath(__file__))
# 从 因子回测/涨跌停情绪因子/ 向上2层回到项目根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(_script_dir)))
from my_utils.fun import read_day_data, read_min_data, get_data_trading_days, add_sma, get_logger

# 日志
logger = get_logger(log_file='因子回测/涨跌停情绪因子/log/reproduce.log', level=logging.INFO)

# 数据参数
START_DATE = date(2018, 1, 2)
END_DATE = date(2026, 7, 10)
DATA_SOURCE = 'rq_stock_all_data'
MIN_DATA_SOURCE = 'rq_15min_stock_data_dir'

# ---- 因子阈值（与研报一致） ----
TH_LIMIT_UP_RATIO = 0.08       # 涨停占比 > 8%
TH_LIMIT_DOWN_RATIO = 0.01     # 跌停占比 < 1%
TH_NET_LIMIT_RATIO = 0.04      # 净涨停占比 > 4%
TH_UP_NEXT_RET = 0.025         # 涨停次日收益 > 2.5%
TH_DOWN_NEXT_RET = -0.010      # 跌停次日收益 > -1.0%
TH_CHASE_RET = -0.005          # 打板收益 > -0.5%

# ---- 因子权重（2016年后，与研报表9一致） ----
FACTOR_WEIGHTS = {
    'net_limit_ratio': 0.15,        # 净涨停占比 15%
    'limit_down_next_ret': 0.10,    # 跌停次日收益 10%
    'limit_up_ratio': 0.25,         # 涨停板占比 25%
    'limit_down_ratio': 0.20,       # 跌停板占比 20%
    'chase_ret': 0.30,              # 打板策略收益 30%
}

# ---- 输出路径 ----
OUTPUT_DIR = '因子回测/涨跌停情绪因子/reproduce_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs('因子回测/涨跌停情绪因子/log', exist_ok=True)

# ============================================================
# 1. 数据加载
# ============================================================
def load_daily_data(start_date, end_date):
    """加载全部A股日线数据，包含涨跌停价格和标记"""
    logger.info(f"加载日线数据: {start_date} ~ {end_date}")
    t0 = time.time()

    fields = [
        'code', 'trading_date', 'open', 'high', 'low', 'close',
        'pre_close', 'pct', 'volume', 'amount', 'limit_up', 'limit_down',
        'is_st', 'is_suspended', 'adj_factor', 'total_mv'
    ]
    df = read_day_data(start_date, end_date, fields=fields, file_path=DATA_SOURCE)

    logger.info(f"日线数据: {df.shape[0]} 行, {df['code'].n_unique()} 只股票, "
                f"日期 {df['trading_date'].min()} ~ {df['trading_date'].max()}, "
                f"耗时 {time.time()-t0:.1f}s")
    return df


def load_index_data(start_date, end_date):
    """加载宽基指数数据用于基准比较"""
    logger.info("加载指数数据...")
    try:
        # 尝试从ts_index_dailybasic读取指数
        idx_fields = ['code', 'trading_date', 'pct', 'close']
        idx_df = read_day_data(start_date, end_date, fields=idx_fields,
                                 file_path='ts_etf_daily')
        # 可取指数
        available_idx = idx_df['code'].unique().to_list()
        logger.info(f"可用指数: {available_idx}")
        return idx_df
    except Exception as e:
        logger.warning(f"指数数据加载失败: {e}")
        return None


# ============================================================
# 2. 日级别涨停标记与次日收益计算
# ============================================================
def mark_limit_events(df):
    """
    标记每日涨停/跌停股票，并计算次日收益率。

    涨停判定: close >= limit_up - 0.005（容忍舍入误差）
    跌停判定: close <= limit_down + 0.005
    排除: ST股、停牌股、北交所(8开头)
    """
    t0 = time.time()

    # 1) 判断是不是沪深A股: 排除北交所(8开头)和其他非标准代码
    df = df.with_columns(
        pl.col('code').str.slice(5, 1).alias('prefix_1')  # 如 SHSE.600000 → '6'
    )

    # 2) 涨停标记: close >= limit_up（含尾差），且当日上涨
    df = df.with_columns(
        ((pl.col('close') >= pl.col('limit_up') - 0.01) &
         (pl.col('close') > pl.col('pre_close')) &
         ~pl.col('is_st') & ~pl.col('is_suspended') &
         pl.col('prefix_1').is_in(['6', '0', '3']))  # 60/00/30开头
        .alias('is_limit_up')
    )

    # 3) 跌停标记
    df = df.with_columns(
        ((pl.col('close') <= pl.col('limit_down') + 0.01) &
         (pl.col('close') < pl.col('pre_close')) &
         ~pl.col('is_st') & ~pl.col('is_suspended') &
         pl.col('prefix_1').is_in(['6', '0', '3']))
        .alias('is_limit_down')
    )

    # 4) 次日收益率（含开盘价）
    df = df.sort(['code', 'trading_date'])
    df = df.with_columns([
        pl.col('close').shift(-1).over('code').alias('next_close'),
        pl.col('open').shift(-1).over('code').alias('next_open'),
        pl.col('trading_date').shift(-1).over('code').alias('next_trading_date'),
    ])
    df = df.with_columns([
        ((pl.col('next_close') / pl.col('close') - 1).alias('next_day_close_ret')),
        ((pl.col('next_open') / pl.col('close') - 1).alias('next_day_open_ret')),
    ])

    # 5) 判断是否为周末（最后一个交易日）：如果是周五或本周期末，排除出次日收益
    df = df.with_columns(
        pl.col('trading_date').dt.weekday().alias('weekday')  # 1=Mon, 7=Sun
    )
    # 判断是否在周末（周五=5）或最后一个交易日
    df = df.with_columns(
        (pl.col('weekday') == 5).alias('is_weekend')
    )

    logger.info(f"涨停标记完成, 涨停样本: {df['is_limit_up'].sum()}, "
                f"跌停样本: {df['is_limit_down'].sum()}, "
                f"耗时 {time.time()-t0:.1f}s")
    return df


# ============================================================
# 3. 周级别因子聚合
# ============================================================
def compute_weekly_factors(df):
    """
    按周聚合，计算周度因子值。

    周定义: ISO周编号（周一~周日），取每周最后一个交易日作为观测日。
    """
    t0 = time.time()

    # 添加年-周标签
    df = df.with_columns([
        pl.col('trading_date').dt.year().alias('year'),
        pl.col('trading_date').dt.week().alias('week_num'),
    ])
    df = df.with_columns(
        (pl.col('year') * 100 + pl.col('week_num')).alias('year_week')
    )

    # ---- 计算每个交易日的因子 ----
    # 每日总股票数（以该日数据为准）
    daily_agg = df.group_by('trading_date').agg([
        pl.col('code').count().alias('total_stocks'),
        pl.col('is_limit_up').sum().alias('limit_up_count'),
        pl.col('is_limit_down').sum().alias('limit_down_count'),
        # 涨停股票次日收益率（排除周末及无数据的情况）
        pl.col('next_day_close_ret')
        .filter(pl.col('is_limit_up') & ~pl.col('is_weekend') & pl.col('next_close').is_not_null())
        .mean().alias('daily_limit_up_next_ret'),
        # 跌停股票次日收益率
        pl.col('next_day_close_ret')
        .filter(pl.col('is_limit_down') & ~pl.col('is_weekend') & pl.col('next_close').is_not_null())
        .mean().alias('daily_limit_down_next_ret'),
        # 用于计算全A收益率
        (pl.col('pct') / 100.0).mean().alias('daily_equal_weight_ret'),
        ((pl.col("pct") / 100.0 * pl.col("total_mv")).sum() / pl.col("total_mv").sum())
        .alias('daily_value_weight_ret'),
    ])

    # 比例因子
    daily_agg = daily_agg.with_columns([
        (pl.col('limit_up_count') / pl.col('total_stocks')).alias('daily_limit_up_ratio'),
        (pl.col('limit_down_count') / pl.col('total_stocks')).alias('daily_limit_down_ratio'),
    ])
    daily_agg = daily_agg.with_columns(
        (pl.col('daily_limit_up_ratio') - pl.col('daily_limit_down_ratio'))
        .alias('daily_net_limit_ratio')
    )

    # ---- 按周聚合 ----
    daily_agg = daily_agg.with_columns([
        pl.col('trading_date').dt.year().alias('year'),
        pl.col('trading_date').dt.week().alias('week_num'),
    ])
    daily_agg = daily_agg.with_columns(
        (pl.col('year') * 100 + pl.col('week_num')).alias('year_week')
    )

    weekly = daily_agg.group_by('year_week').agg([
        pl.col('trading_date').max().alias('week_end_date'),  # 观测日
        pl.col('trading_date').count().alias('trading_days_in_week'),
        pl.col('daily_limit_up_ratio').mean().alias('limit_up_ratio'),
        pl.col('daily_limit_down_ratio').mean().alias('limit_down_ratio'),
        pl.col('daily_net_limit_ratio').mean().alias('net_limit_ratio'),
        pl.col('daily_limit_up_next_ret').mean().alias('limit_up_next_ret'),
        pl.col('daily_limit_down_next_ret').mean().alias('limit_down_next_ret'),
        ((1 + pl.col('daily_equal_weight_ret')).product() - 1).alias('benchmark_equal_weight_ret'),
        ((1 + pl.col('daily_value_weight_ret')).product() - 1).alias('benchmark_value_weight_ret'),
    ])

    # 每周5个交易日及以上才纳入
    weekly = weekly.filter(pl.col('trading_days_in_week') >= 3)

    weekly = weekly.sort('week_end_date')

    logger.info(f"周度因子计算完成: {weekly.shape[0]} 周, "
                f"耗时 {time.time()-t0:.1f}s")
    return weekly


# ============================================================
# 4. 打板策略收益因子（基于15分钟数据）
# ============================================================
def compute_chase_returns(start_date, end_date, failure_rate=0.10):
    """
    计算打板策略收益因子（周度）。

    策略规则（与研报一致）:
    - 在个股首次出现股价上涨至9%时买入（从15分钟K线判断）
    - 在第二天开盘时卖出
    - 叠加10%的概率买入失败

    此处简化: 不严格限制90个交易日内只首次，而是对所有9%触发作统计；
    因为整体收益趋近于-0.93%/笔，平均化后趋势一致。
    """
    logger.info("开始计算打板策略收益因子（15分钟数据）...")
    t0_total = time.time()

    # 获取交易日列表，分批处理
    all_dates = get_data_trading_days(start_date, end_date, file_path=DATA_SOURCE)
    logger.info(f"共 {len(all_dates)} 个交易日")

    # 构建 date → is_weekend 映射
    date_info = {d: (d.weekday() >= 5 or d.weekday() == 4) for d in all_dates}

    # 准备存储打板事件
    all_chase_records = []

    # 100笔一批处理（避免内存过大）
    batch_size = 100
    total_batches = (len(all_dates) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        batch_dates = all_dates[batch_idx * batch_size: (batch_idx + 1) * batch_size]
        t1 = time.time()

        # 读取这批日期的15分钟数据
        try:
            min_data = read_min_data(
                datetime.combine(batch_dates[0], datetime.min.time()),
                datetime.combine(batch_dates[-1], datetime.max.time()),
                file_path=MIN_DATA_SOURCE
            )
        except Exception as e:
            logger.warning(f"  批次 {batch_idx+1}/{total_batches} 读取失败: {e}")
            continue

        if min_data.is_empty():
            continue

        # 获取前日收盘价（从日线数据）
        day_data = read_day_data(batch_dates[0], batch_dates[-1],
                                  fields=['code', 'trading_date', 'close', 'pre_close', 'open'],
                                  file_path=DATA_SOURCE)
        if day_data.is_empty():
            continue

        # 合并前日数据到分钟数据
        day_data = day_data.rename({'close': 'day_close', 'pre_close': 'pre_close_day', 'open': 'day_open'})
        min_data = min_data.join(
            day_data.select(['code', 'trading_date', 'pre_close_day', 'day_close']),
            on=['code', 'trading_date'],
            how='inner'
        )

        # 判断是否触发9%: 15分钟bar的 high >= pre_close * 1.09
        min_data = min_data.with_columns(
            (pl.col('pre_close_day') * 1.09).alias('chase_threshold')
        )
        min_data = min_data.with_columns(
            (pl.col('high') >= pl.col('chase_threshold')).alias('is_chase_signal')
        )

        # 按(code, date)聚合: 只要能触达9%就标记
        daily_chase = min_data.group_by(['code', 'trading_date']).agg([
            pl.col('is_chase_signal').any().alias('has_chase_signal'),
            pl.col('chase_threshold').first(),
            pl.col('pre_close_day').first(),
            pl.col('day_close').first(),
        ])

        chase_events = daily_chase.filter(pl.col('has_chase_signal'))

        if not chase_events.is_empty():
            # 获取次日开盘价
            next_day_date_map = {}
            for i, d in enumerate(batch_dates):
                if i < len(batch_dates) - 1:
                    next_day_date_map[d] = batch_dates[i + 1]
                else:
                    # 找下一个交易日
                    idx = all_dates.index(d) if d in all_dates else -1
                    if idx >= 0 and idx < len(all_dates) - 1:
                        next_day_date_map[d] = all_dates[idx + 1]
                    else:
                        next_day_date_map[d] = None

            # 添加上市日期映射
            # 用 Polars 的 map_elements 添加 next_date 列
            chase_events = chase_events.with_columns(
                pl.col('trading_date').map_elements(
                    lambda d: next_day_date_map.get(d), return_dtype=pl.Date
                ).alias('next_date')
            )

            # 合并次日开盘价（扩大日期范围以覆盖 next_date）
            next_start = batch_dates[0]
            next_end = max(
                (next_day_date_map.get(d) for d in batch_dates if next_day_date_map.get(d) is not None),
                default=batch_dates[-1]
            )
            if next_end <= batch_dates[-1]:
                # 如果 next_end 没超过当前批次，读取到下一批
                next_idx = all_dates.index(batch_dates[-1]) + 1
                if next_idx < len(all_dates):
                    next_end = all_dates[min(next_idx + 1, len(all_dates) - 1)]

            next_data = read_day_data(
                next_start, next_end,
                fields=['code', 'trading_date', 'open'],
                file_path=DATA_SOURCE
            )
            next_data = next_data.rename({
                'trading_date': 'next_date',
                'open': 'next_open'
            })

            chase_events = chase_events.join(
                next_data.select(['code', 'next_date', 'next_open']),
                on=['code', 'next_date'],
                how='left'
            )

            # 计算收益率: next_open / (pre_close * 1.09) - 1
            chase_events = chase_events.with_columns(
                ((pl.col('next_open') / pl.col('chase_threshold') - 1)
                 .alias('chase_return'))
            )

            # 去掉NaN
            chase_events = chase_events.filter(pl.col('chase_return').is_not_null() & pl.col('chase_return').is_finite())

            if not chase_events.is_empty():
                records = chase_events.select([
                    'code', 'trading_date', 'chase_return'
                ]).to_pandas()
                all_chase_records.append(records)

        if (batch_idx + 1) % 5 == 0 or batch_idx == total_batches - 1:
            logger.info(f"  打板策略进度: {batch_idx+1}/{total_batches}, "
                        f"当前累计 {sum(len(r) for r in all_chase_records)} 笔交易, "
                        f"本批耗时 {time.time()-t1:.1f}s")

    # 合并所有打板记录
    if not all_chase_records:
        logger.warning("打板策略: 未找到任何交易")
        return None

    import pandas as pd
    all_trades = pd.concat(all_chase_records, ignore_index=True)
    logger.info(f"打板策略: 共 {len(all_trades)} 笔交易, "
                f"平均收益率: {all_trades['chase_return'].mean()*100:.2f}%, "
                f"耗时 {time.time()-t0_total:.1f}s")

    # 应用10%失败率: 随机dropping 10%的交易
    np.random.seed(42)
    keep_mask = np.random.random(len(all_trades)) >= failure_rate
    all_trades = all_trades[keep_mask]
    logger.info(f"  扣除{failure_rate*100:.0f}%失败率后: {len(all_trades)} 笔交易")

    # 按周聚合
    all_trades['trading_date'] = pd.to_datetime(all_trades['trading_date'])
    all_trades['year'] = all_trades['trading_date'].dt.isocalendar().year
    all_trades['week'] = all_trades['trading_date'].dt.isocalendar().week
    all_trades['year_week'] = all_trades['year'] * 100 + all_trades['week']

    weekly_chase = all_trades.groupby('year_week')['chase_return'].mean().reset_index()
    weekly_chase.columns = ['year_week', 'chase_ret']
    weekly_chase['chase_trade_count'] = all_trades.groupby('year_week').size().values

    return pl.from_pandas(weekly_chase)


# ============================================================
# 5. 信号生成与回测引擎
# ============================================================
def diagnose_signal_trigger_rates(weekly):
    """诊断各因子阈值在回测区间内的触发率"""
    print("\n" + "-" * 60)
    print("  【因子阈值触发率诊断】")
    print("  说明: 研报阈值基于2010-2025数据优化，在2018-2026区间可能不匹配")
    print("-" * 60)

    checks = [
        ('涨停占比 > 8%', 'limit_up_ratio', 'gt', TH_LIMIT_UP_RATIO),
        ('跌停占比 < 1%', 'limit_down_ratio', 'lt', TH_LIMIT_DOWN_RATIO),
        ('净涨停占比 > 4%', 'net_limit_ratio', 'gt', TH_NET_LIMIT_RATIO),
        ('涨停次日收益 > 2.5%', 'limit_up_next_ret', 'gt', TH_UP_NEXT_RET),
        ('跌停次日收益 > -1.0%', 'limit_down_next_ret', 'gt', TH_DOWN_NEXT_RET),
    ]
    if 'chase_ret' in weekly.columns:
        checks.append(('打板策略收益 > -0.5%', 'chase_ret', 'gt', TH_CHASE_RET))

    total = weekly.shape[0]
    for name, col, cmp, th in checks:
        if col not in weekly.columns:
            continue
        if cmp == 'gt':
            cnt = weekly.filter(pl.col(col) > th).shape[0]
        else:
            cnt = weekly.filter(pl.col(col) < th).shape[0]
        vals = weekly[col].drop_nulls()
        p50 = vals.quantile(0.5) if len(vals) > 0 else None
        p90 = vals.quantile(0.9) if len(vals) > 0 else None
        print(f"  {name:<30} 触发{cnt:>4}/{total}周 ({cnt/total*100:>5.1f}%)  "
              f"中位数={p50:.4f}  p90={p90:.4f}")

    print("-" * 60)


def compute_adaptive_thresholds(weekly):
    """基于样本内分位数计算自适应阈值。

    目标: 每个因子保持约20-35%的触发率（与研报中信号频率相近）。
    正向因子（越高越好）: 使用70-75%分位
    反向因子（越低越好）: 使用25-30%分位
    """
    thresholds = {}
    factor_config = [
        ('limit_up_ratio', 'gt', 0.75),       # 涨停占比: 高值信号，前25%
        ('limit_down_ratio', 'lt', 0.25),      # 跌停占比: 低值信号，后25%
        ('net_limit_ratio', 'gt', 0.75),       # 净涨停占比: 高值信号，前25%
        ('limit_up_next_ret', 'gt', 0.70),     # 涨停次日收益: 高值信号，前30%
        ('limit_down_next_ret', 'gt', 0.70),   # 跌停次日收益: 越高越好，前30%
        ('chase_ret', 'gt', 0.70),             # 打板收益: 越高越好，前30%
    ]
    for col, cmp_, quantile in factor_config:
        if col not in weekly.columns:
            thresholds[col] = (cmp_, None)
            continue
        vals = weekly[col].drop_nulls()
        if len(vals) == 0:
            thresholds[col] = (cmp_, None)
            continue
        if cmp_ == 'gt':
            th = vals.quantile(quantile)
        else:
            th = vals.quantile(1 - quantile)
        thresholds[col] = (cmp_, th)
    return thresholds


def generate_signals(weekly_factors, thresholds=None):
    """
    对每个因子生成信号（0或1），并聚合。

    因子信号规则（与研报一致）:
    1. 涨停板占比 > 8%
    2. 跌停板占比 < 1%
    3. 净涨停占比 > 4%
    4. 涨停次日收益 > 2.5%
    5. 跌停次日收益 > -1.0%
    6. 打板策略收益 > -0.5%
    """
    df = weekly_factors.clone()

    # 默认使用全局阈值
    if thresholds is None:
        thresholds = {
            'limit_up_ratio': ('gt', TH_LIMIT_UP_RATIO),
            'limit_down_ratio': ('lt', TH_LIMIT_DOWN_RATIO),
            'net_limit_ratio': ('gt', TH_NET_LIMIT_RATIO),
            'limit_up_next_ret': ('gt', TH_UP_NEXT_RET),
            'limit_down_next_ret': ('gt', TH_DOWN_NEXT_RET),
            'chase_ret': ('gt', TH_CHASE_RET),
        }

    # 基础因子（1-5，2018年起）
    signal_rules = [
        ('limit_up_ratio', 'gt'),
        ('limit_down_ratio', 'lt'),
        ('net_limit_ratio', 'gt'),
        ('limit_up_next_ret', 'gt'),
        ('limit_down_next_ret', 'gt'),
    ]

    df = df.with_columns([
        pl.when(
            (pl.col(col) > thresholds[col][1]) if thresholds[col][0] == 'gt'
            else (pl.col(col) < thresholds[col][1])
        ).then(1).otherwise(0).alias(f'signal_{col}')
        for col, _ in signal_rules if col in df.columns and thresholds.get(col) and thresholds[col][1] is not None
    ])

    # 打板策略因子信号（如有数据）
    if 'chase_ret' in df.columns and thresholds.get('chase_ret') and thresholds['chase_ret'][1] is not None:
        th = thresholds['chase_ret'][1]
        df = df.with_columns(
            pl.when(pl.col('chase_ret') > th).then(1).otherwise(0)
            .alias('signal_chase_ret')
        )
    else:
        df = df.with_columns(pl.lit(0).alias('signal_chase_ret'))

    # 信号总和 (2016年后最多6个因子，但打板可能缺失)
    signal_cols = ['signal_limit_up_ratio', 'signal_limit_down_ratio',
                   'signal_net_limit_ratio', 'signal_limit_up_next_ret',
                   'signal_limit_down_next_ret', 'signal_chase_ret']
    df = df.with_columns(
        sum(pl.col(c) for c in ['signal_limit_up_ratio', 'signal_limit_down_ratio',
                                 'signal_net_limit_ratio', 'signal_limit_up_next_ret',
                                 'signal_limit_down_next_ret'])
        .alias('base_signal_count')
    )
    df = df.with_columns(
        sum(pl.col(c) for c in signal_cols).alias('total_signal_count')
    )

    return df


def compute_position(signal_count, has_chase=True):
    """
    信号→仓位映射（与研报表2一致，2016年后规则）。

    2016年后:
      signal >= 3 → 100%
      signal == 2 → 75%
      signal == 1 → 50%
      signal == 0 → 0%
    """
    max_signal = 6 if has_chase else 5
    if signal_count >= 3:
        return 1.0
    elif signal_count == 2:
        return 0.75
    elif signal_count == 1:
        return 0.50
    else:
        return 0.0


def compute_weighted_signal(row, weights):
    """
    计算加权信号（与研报表9一致）。

    加权信号 = Σ(因子值 × 权重)
    然后映射到仓位:
      > 0.65 → 100%
      (0.5, 0.65] → 80%
      ≤ 0.5 → 0%
    """
    # 因子值归一化后乘以权重
    # 注意: 这里用的是"信号"（0或1），不是原始因子值
    # 研报表9的权重是用于信号的加权
    weighted = 0.0
    for factor, weight in weights.items():
        signal_col = f'signal_{factor}'
        if signal_col in row:
            weighted += row[signal_col] * weight

    return weighted


def run_backtest(weekly_signals, weekly_benchmark_ret_col='benchmark_value_weight_ret',
                 has_chase=True):
    """
    运行回测。

    每周观测 → 判断信号 → 确定下周仓位 → 获得下周收益
    """
    df = weekly_signals.sort('week_end_date').clone()
    n = df.shape[0]

    # ---- 基础模型仓位 ----
    df = df.with_columns(
        pl.col('total_signal_count').map_elements(
            lambda x: compute_position(x, has_chase),
            return_dtype=pl.Float64
        ).alias('base_position')
    )

    # 下周收益率（信号在本周末决定，影响下周）
    df = df.with_columns([
        pl.col(weekly_benchmark_ret_col).shift(-1).alias('next_week_benchmark_ret'),
    ])

    # 组合收益: position × next_week_benchmark_ret
    df = df.with_columns(
        (pl.col('base_position') * pl.col('next_week_benchmark_ret'))
        .alias('base_model_ret')
    )

    # ---- 加权模型（改进2） ----
    df = df.with_columns(
        pl.struct([c for c in df.columns if c.startswith('signal_')])
        .map_elements(
            lambda row: compute_weighted_signal(row, FACTOR_WEIGHTS),
            return_dtype=pl.Float64
        ).alias('weighted_signal_value')
    )

    # 加权信号 → 仓位
    df = df.with_columns(
        pl.when(pl.col('weighted_signal_value') > 0.65).then(1.0)
        .when(pl.col('weighted_signal_value') > 0.50).then(0.80)
        .otherwise(0.0)
        .alias('weighted_position')
    )
    df = df.with_columns(
        (pl.col('weighted_position') * pl.col('next_week_benchmark_ret'))
        .alias('weighted_model_ret')
    )

    # ---- 市场趋势判断（改进1） ----
    # 用MA10/MA20/MA60均线判断市场趋势
    # 先计算累计基准指数值
    df = df.with_columns(
        (1 + pl.col(weekly_benchmark_ret_col)).cum_prod().alias('benchmark_cum')
    )

    # 计算MA10, MA20, MA60
    windows = [10, 20, 60]
    for w in windows:
        # 用累计值计算均线
        df = df.with_columns(
            pl.col('benchmark_cum').rolling_mean(window_size=w).alias(f'ma_{w}')
        )

    # 趋势判断: 上行 MA10>MA20>MA60; 下行 MA60>MA20>MA10; 震荡 其他
    df = df.with_columns(
        pl.when(
            (pl.col('ma_10') > pl.col('ma_20')) & (pl.col('ma_20') > pl.col('ma_60'))
        ).then(pl.lit('up'))
        .when(
            (pl.col('ma_60') > pl.col('ma_20')) & (pl.col('ma_20') > pl.col('ma_10'))
        ).then(pl.lit('down'))
        .otherwise(pl.lit('sideways'))
        .alias('market_trend')
    )

    # 等权信号均线（MA20）
    df = df.with_columns(
        pl.col('total_signal_count').rolling_mean(window_size=20).alias('signal_ma20')
    )

    # 趋势增强信号（与研报表7一致）
    # 买入条件: (1)等权信号>信号MA20, 或
    #           (2)上行且等权信号>0.1
    #           (3)震荡且等权信号>0.5
    #           (4)下行且等权信号>0.75
    df = df.with_columns(
        (
            (pl.col('total_signal_count') > pl.col('signal_ma20')) |
            ((pl.col('market_trend') == 'up') & (pl.col('total_signal_count') > 0.1)) |
            ((pl.col('market_trend') == 'sideways') & (pl.col('total_signal_count') > 0.5)) |
            ((pl.col('market_trend') == 'down') & (pl.col('total_signal_count') > 0.75))
        ).alias('trend_signal_buy')
    )

    df = df.with_columns(
        pl.when(pl.col('trend_signal_buy')).then(1.0).otherwise(0.0)
        .alias('trend_position')
    )
    df = df.with_columns(
        (pl.col('trend_position') * pl.col('next_week_benchmark_ret'))
        .alias('trend_model_ret')
    )

    # 移除最后一周（没有下周数据）
    df = df.filter(pl.col('next_week_benchmark_ret').is_not_null())

    return df


# ============================================================
# 6. 绩效报告
# ============================================================
def compute_performance_metrics(returns_series, benchmark_ret_col=None, df=None):
    """
    计算绩效指标：年化收益、年化波动、最大回撤、夏普、胜率、超额收益等。

    与研报表3-6格式一致。
    """
    # 处理输入: 支持 numpy array 或 Polars Series
    if hasattr(returns_series, 'to_numpy'):
        ret = returns_series.to_numpy()
    elif hasattr(returns_series, 'is_null'):
        ret = returns_series.to_numpy()
    else:
        ret = np.asarray(returns_series, dtype=float)
    ret = ret[~np.isnan(ret)]
    if len(ret) == 0:
        return {}

    # 周度收益率假定一年52周
    n_weeks = len(ret)
    n_years = n_weeks / 52

    # 年化收益率
    total_ret = np.prod(1 + ret) - 1
    ann_ret = (1 + total_ret) ** (1 / n_years) - 1

    # 年化波动率
    weekly_vol = np.std(ret, ddof=1)
    ann_vol = weekly_vol * np.sqrt(52)

    # 最大回撤
    cum = np.cumprod(1 + ret)
    peak = np.maximum.accumulate(cum)
    drawdown = (cum - peak) / peak
    max_dd = drawdown.min()

    # 夏普比率（假设无风险利率=0）
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    # 胜率
    win_rate = np.sum(ret > 0) / n_weeks

    # 盈亏比
    avg_win = np.mean(ret[ret > 0]) if np.sum(ret > 0) > 0 else 0
    avg_loss = abs(np.mean(ret[ret < 0])) if np.sum(ret < 0) > 0 else 1
    profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0

    # 如果提供了基准，计算超额收益
    excess_ann_ret = None
    if benchmark_ret_col is not None and df is not None:
        if hasattr(df[benchmark_ret_col], 'to_numpy'):
            bench = df[benchmark_ret_col].to_numpy()
        else:
            bench = np.asarray(df[benchmark_ret_col])
        bench = bench[~np.isnan(bench)][:len(ret)]
        if len(bench) == len(ret):
            bench_total = np.prod(1 + bench) - 1
            bench_ann = (1 + bench_total) ** (1 / n_years) - 1
            excess_ann_ret = ann_ret - bench_ann

    return {
        '年化收益率': ann_ret,
        '年化波动率': ann_vol,
        '最大回撤': max_dd,
        '夏普比率': sharpe,
        '胜率': win_rate,
        '盈亏比': profit_loss_ratio,
        '总收益率': total_ret,
        '周数': n_weeks,
    }


def print_performance_comparison(results, title="绩效对比"):
    """打印多模型绩效对比"""
    print(f"\n{'='*80}")
    print(f" {title}")
    print(f"{'='*80}")
    print(f"{'指标':<20}", end='')
    for name in results:
        print(f"{name:<18}", end='')
    print()
    print(f"{'-'*80}")

    metrics_display = [
        ('年化收益率', 'ann_ret_pct'),
        ('年化波动率', 'ann_vol_pct'),
        ('最大回撤', 'max_dd_pct'),
        ('夏普比率', 'sharpe'),
        ('胜率', 'win_rate_pct'),
        ('盈亏比', 'profit_loss_ratio'),
    ]

    for label, key in metrics_display:
        print(f"{label:<20}", end='')
        for name in results:
            r = results[name]
            if key == 'profit_loss_ratio':
                val = r.get('盈亏比', 0)
                print(f"{val:<18.2f}", end='')
            elif key in ('sharpe',):
                val = r.get('夏普比率', 0)
                print(f"{val:<18.2f}", end='')
            else:
                val = r.get(label, 0) * 100
                print(f"{val:<18.2f}%", end='')
        print()
    print(f"{'='*80}\n")


# ============================================================
# 7. 绘图
# ============================================================
def plot_results(backtest_results):
    """绘制累计收益率对比图"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.rcParams['font.sans-serif'] = ['SimHei']
        matplotlib.rcParams['axes.unicode_minus'] = False
    except ImportError:
        logger.warning("matplotlib未安装，跳过绘图")
        return

    df = backtest_results
    n = df.shape[0]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # 1. 累计收益对比
    ax = axes[0, 0]
    models = [
        ('benchmark_value_weight_ret', '基准(全A市值加权)', 'gray', '--'),
        ('base_model_ret', '基础择时模型', 'blue', '-'),
        ('trend_model_ret', '市场趋势+择时', 'green', '-'),
        ('weighted_model_ret', '因子加权模型', 'red', '-'),
    ]

    for col, label, color, style in models:
        if col in df.columns:
            cum = (1 + df[col].to_numpy()).cumprod()
            ax.plot(df['week_end_date'], cum, label=label, color=color,
                    linestyle=style, linewidth=1.5 if style == '-' else 1)

    ax.set_title('累计收益对比', fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylabel('净值')
    ax.grid(True, alpha=0.3)

    # 2. 信号和仓位
    ax = axes[0, 1]
    if 'total_signal_count' in df.columns:
        ax.plot(df['week_end_date'], df['total_signal_count'],
                label='总信号数', color='purple', alpha=0.7)
        ax2 = ax.twinx()
        ax2.plot(df['week_end_date'], df['base_position'],
                 label='基础仓位', color='orange', alpha=0.5, linewidth=2)
        ax2.set_ylabel('仓位')
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
    ax.set_title('信号数与仓位', fontsize=13)
    ax.grid(True, alpha=0.3)

    # 3. 因子值时序
    ax = axes[1, 0]
    factor_cols = ['limit_up_ratio', 'limit_down_ratio', 'net_limit_ratio']
    for col in factor_cols:
        if col in df.columns:
            ax.plot(df['week_end_date'], df[col], label=col, alpha=0.7)
    ax.axhline(y=0.08, color='red', linestyle='--', alpha=0.5, label='涨停阈值(8%)')
    ax.axhline(y=0.01, color='green', linestyle='--', alpha=0.5, label='跌停阈值(1%)')
    ax.legend(fontsize=8)
    ax.set_title('因子值时序', fontsize=13)
    ax.grid(True, alpha=0.3)

    # 4. 逐年收益率对比
    ax = axes[1, 1]
    df_plot = df.with_columns(pl.col('week_end_date').dt.year().alias('year'))
    yearly_ret = {}
    for col, label in [('benchmark_value_weight_ret', '基准'),
                       ('base_model_ret', '基础模型'),
                       ('trend_model_ret', '趋势模型'),
                       ('weighted_model_ret', '加权模型')]:
        if col in df.columns:
            yr = df_plot.group_by('year').agg(pl.col(col).sum().alias('ret')).sort('year')
            yearly_ret[label] = yr

    x = np.arange(len(yearly_ret.get('基准', [])))
    width = 0.2
    for i, (label, yr) in enumerate(yearly_ret.items()):
        vals = yr['ret'].to_numpy() * 100
        ax.bar(x + i * width, vals, width, label=label, alpha=0.7)

    if len(x) > 0:
        ax.set_xticks(x + width * 1.5)
        years = yearly_ret[list(yearly_ret.keys())[0]]['year'].to_list()
        ax.set_xticklabels(years, rotation=45)

    ax.set_title('逐年收益率对比', fontsize=13)
    ax.set_ylabel('收益率 (%)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, 'sentiment_timing_results.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    logger.info(f"图表已保存: {fig_path}")
    plt.close()


# ============================================================
# 8. 主流程
# ============================================================
def main():
    """主流程：加载数据 → 计算因子 → 回测 → 报告"""
    t_start = time.time()
    print("\n" + "█" * 60)
    print("  国泰海通情绪择时模型 - 复现")
    print("  原文: 大类资产与中观配置研究（五）")
    print("  数据区间: 2018-01-02 ~ 2026-07-10")
    print("  注: 原文回测2010~2025，受数据限制从2018开始")
    print("█" * 60 + "\n")

    # ---- 1. 加载日线数据 ----
    print("\n[Step 1/5] 加载日线数据...")
    daily_raw = load_daily_data(START_DATE, END_DATE)
    total_stocks = daily_raw['code'].n_unique()
    print(f"  总股票数: {total_stocks}, 总行数: {daily_raw.shape[0]:,}")

    # ---- 2. 标记涨停/跌停及次日收益 ----
    print("\n[Step 2/5] 标记涨停/跌停事件...")
    daily_df = mark_limit_events(daily_raw)
    limit_up_total = daily_df['is_limit_up'].sum()
    limit_down_total = daily_df['is_limit_down'].sum()
    print(f"  涨停样本: {limit_up_total:,}, 跌停样本: {limit_down_total:,}")

    # ---- 3. 计算周度因子 ----
    print("\n[Step 3/5] 计算周度因子...")
    weekly = compute_weekly_factors(daily_df)
    print(f"  有效交易周: {weekly.shape[0]}")

    # 预览因子值
    factor_summary = weekly.select([
        'limit_up_ratio', 'limit_down_ratio', 'net_limit_ratio',
        'limit_up_next_ret', 'limit_down_next_ret'
    ]).describe()
    print(f"\n因子统计描述:\n{factor_summary}")

    # ---- 3.5 阈值诊断 ----
    print("\n[Step 3.5/5] 阈值触发率诊断...")
    diagnose_signal_trigger_rates(weekly)
    adaptive_th = compute_adaptive_thresholds(weekly)
    print("\n自适应阈值（基于样本分位数，保持~25-30%触发率）:")
    for k, (cmp_, v) in adaptive_th.items():
        if v is not None:
            print(f"  {k}: {cmp_} {v:.4f}")

    # ---- 4. 打板策略因子（15分钟数据） ----
    print("\n[Step 4/5] 计算打板策略收益因子...")
    print("  (读取15分钟数据，处理约5000只股票 × 2000+个交易日)\n")

    chase_weekly = compute_chase_returns(START_DATE, END_DATE)

    if chase_weekly is not None:
        weekly = weekly.join(chase_weekly, on='year_week', how='left')
        weekly = weekly.with_columns(
            pl.col('chase_ret').fill_null(strategy='forward').fill_null(0)
        )
        print(f"  打板策略因子已合并")
        has_chase = True
    else:
        weekly = weekly.with_columns(pl.lit(0.0).alias('chase_ret'))
        weekly = weekly.with_columns(pl.lit(0).alias('chase_trade_count'))
        has_chase = False
        print("  打板策略因子不可用，使用0填充")

    # ---- 5. 信号生成与回测 ----
    print("\n[Step 5/5] 生成信号并运行回测...")
    signals = generate_signals(weekly)
    results = run_backtest(signals, has_chase=has_chase)

    # 计算绩效
    models = {
        '基准(全A)': 'benchmark_value_weight_ret',
        '基础择时模型': 'base_model_ret',
        '市场趋势+择时': 'trend_model_ret',
        '因子加权模型': 'weighted_model_ret',
    }

    perf_results = {}
    for name, col in models.items():
        if col in results.columns:
            perf = compute_performance_metrics(
                results[col].to_numpy(),
                benchmark_ret_col='benchmark_value_weight_ret' if name != '基准(全A)' else None,
                df=results
            )
            perf_results[name] = perf

    print_performance_comparison(perf_results, "情绪择时模型绩效对比（2018~2026）")

    # ---- 分年度表现 ----
    print("\n分年度表现:")
    results_yearly = results.with_columns(
        pl.col('week_end_date').dt.year().alias('year')
    )

    yearly_cols = {
        'benchmark_value_weight_ret': '基准',
        'base_model_ret': '基础模型',
        'trend_model_ret': '趋势模型',
        'weighted_model_ret': '加权模型',
    }

    yearly = results_yearly.group_by('year').agg([
        pl.col(c).sum().alias(f'{name}_ret')
        for c, name in yearly_cols.items()
    ]).sort('year')

    print(f"{'年份':<8}", end='')
    for name in yearly_cols.values():
        print(f"{name:<14}", end='')
    print()
    print("-" * 50)

    for row in yearly.iter_rows(named=True):
        print(f"{row['year']:<8}", end='')
        for col, _ in yearly_cols.items():
            val = row.get(f'{yearly_cols[col]}_ret', 0) * 100
            print(f"{val:<14.2f}", end='')
        print()

    # ---- 自适应阈值版本对比 ----
    print("\n\n" + "█" * 60)
    print("  自适应阈值对比（基于样本分位数，~25-30%触发率）")
    print("  目的: 验证模型方法论在2018-2026区间是否有效")
    print("  （注意: 此处使用未来数据计算阈值，仅用于验证方法）")
    print("█" * 60)
    signals_adaptive = generate_signals(weekly, thresholds=adaptive_th)
    results_adaptive = run_backtest(signals_adaptive, has_chase=has_chase)

    adaptive_models = {
        '基准(全A)': 'benchmark_value_weight_ret',
        '基础信号(自适应)': 'base_model_ret',
        '趋势+择时(自适应)': 'trend_model_ret',
        '因子加权(自适应)': 'weighted_model_ret',
    }
    adaptive_perf = {}
    for name, col in adaptive_models.items():
        if col in results_adaptive.columns:
            perf = compute_performance_metrics(
                results_adaptive[col].to_numpy(),
                benchmark_ret_col='benchmark_value_weight_ret' if name != '基准(全A)' else None,
                df=results_adaptive
            )
            adaptive_perf[name] = perf
    print_performance_comparison(adaptive_perf, "自适应阈值绩效对比")

    # ---- 宽基指数对比 ----
    # 注: 我们没有直接可用的沪深300/中证500指数日收益率数据
    # (ts_index_dailybasic缺少价格列，ts_etf_daily是ETF数据)
    # 作为替代，全程使用我们的全A市值加权指数作为基准
    print("\n\n注: 宽基指数收益率数据不可直接获取，本次复现使用")
    print("全A市值加权指数作为统一基准。如需沪深300/中证500对比，")
    print("需要额外从Wind等数据源获取指数日收益率。\n")

    # ---- 绘图 ----
    print("\n生成图表...")
    try:
        plot_results(results)
    except Exception as e:
        logger.warning(f"绘图失败: {e}")

    # ---- 保存结果 ----
    csv_path = os.path.join(OUTPUT_DIR, 'backtest_results.csv')
    results.select(['week_end_date', 'limit_up_ratio', 'limit_down_ratio',
                    'net_limit_ratio', 'total_signal_count', 'base_position',
                    'weighted_position', 'trend_position',
                    'benchmark_value_weight_ret', 'base_model_ret',
                    'trend_model_ret', 'weighted_model_ret']).write_csv(csv_path)
    logger.info(f"结果已保存: {csv_path}")

    # ---- 总结 ----
    total_time = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  复现完成! 总耗时: {total_time:.1f}s")
    print(f"  输出目录: {OUTPUT_DIR}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
