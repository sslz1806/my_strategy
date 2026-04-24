"""
小市值策略优化版 - 测试脚本
"""
import sys
sys.path.append('C://Users/20561/Desktop/策略')

from my_utils.fun import read_day_data, get_data_trading_days, get_logger
import pandas as pd
import polars as pl
import numpy as np
import datetime as dt

logging = get_logger(log_file='因子回测/log/小市值策略优化版.log', inherit=False)

# ========== 策略参数 ==========
START_DATE = '2024-01-01'
END_DATE = '2026-04-20'
INITIAL_CASH = 100_000.0
COMMISSION = 0.0003
SLIPPAGE = 0.000
STOCK_NUM = 3
REFRESH_RATE = 5
MV_MIN = 15
MV_MAX = 50
TURNOVER_MIN = 1.0
STOP_LOSS = -0.08
INDUSTRY_DIVERSIFY = True

# ========== 读取数据 ==========
logging.info(f'读取数据: {START_DATE} 至 {END_DATE}')
stock_data = read_day_data(start_date=START_DATE, end_date=END_DATE)

stock_data = stock_data.with_columns([
    (pl.col('total_mv') / 1e8).alias('market_cap'),
    (pl.col('turnover_rate') * 100).alias('turnover_pct'),
    (pl.col('close') / pl.col('pre_close') - 1).alias('daily_return'),
])

logging.info(f'数据行数: {len(stock_data)}')

# ========== 选股函数 ==========
def select_stocks_optimized(daily_data, mv_min, mv_max, top_n, turnover_min=1.0, industry_diversify=True):
    filtered = daily_data.filter(
        (pl.col('market_cap') >= mv_min) &
        (pl.col('market_cap') <= mv_max) &
        (~pl.col('is_st')) &
        (~pl.col('is_suspended'))
    )

    if turnover_min > 0:
        filtered = filtered.filter(pl.col('turnover_pct') >= turnover_min)

    if len(filtered) == 0:
        return []

    filtered = filtered.sort('market_cap', descending=False)

    if industry_diversify:
        selected = []
        industry_selected = set()

        for row in filtered.to_dicts():
            if len(selected) >= top_n:
                break

            code = row['code']
            if code.startswith('SZSE.000') or code.startswith('SHSE.600'):
                industry = 'main_board'
            elif code.startswith('SZSE.002'):
                industry = 'sme'
            elif code.startswith('SZSE.300'):
                industry = 'chinext'
            elif code.startswith('SHSE.688'):
                industry = 'star'
            else:
                industry = 'other'

            if industry not in industry_selected:
                industry_selected.add(industry)
                vol = row.get('volatility_20d')
                vol = vol if (vol is not None and vol == vol) else 0.02
                selected.append((row['code'], row['name'], industry, row['market_cap'], vol))

        return selected
    else:
        return [(row['code'], row['name'], 'unknown',
                row['market_cap'], row.get('volatility_20d', 0.02))
               for row in filtered.head(top_n).to_dicts()]

# ========== 生成交易信号 ==========
logging.info('生成交易信号...')
trading_days = get_data_trading_days(START_DATE, END_DATE)
logging.info(f'交易日数量: {len(trading_days)}')

orders_list = []
day_count = 0
current_holdings = {}
cash = INITIAL_CASH

for day in trading_days:
    day_ts = pd.Timestamp(day)
    day_count += 1

    daily_data = stock_data.filter(pl.col('trading_date') == day)

    # 止损检查
    stocks_to_stop = []
    for code, holding in list(current_holdings.items()):
        day_row = daily_data.filter(pl.col('code') == code)
        if len(day_row) == 0:
            continue
        current_price = day_row['close'].to_list()[0]
        if current_price < holding['buy_price'] * (1 + STOP_LOSS):
            stocks_to_stop.append(code)

    # 每5个交易日调仓
    need_rebalance = (day_count % REFRESH_RATE == 1)

    if need_rebalance or len(stocks_to_stop) > 0:
        # 卖出
        all_sell_codes = set(stocks_to_stop)
        if need_rebalance:
            all_sell_codes.update(current_holdings.keys())

        for code in all_sell_codes:
            holding = current_holdings.get(code)
            if not holding:
                continue
            day_row = daily_data.filter(pl.col('code') == code)
            if len(day_row) == 0:
                continue
            sell_price = day_row['open'].to_list()[0]
            volume = holding['volume']

            orders_list.append({
                'datetime': day_ts,
                'code': code,
                'direction': -1,
                'price': sell_price,
                'volume': volume,
                'buy_time': holding['buy_time'],
                'sell_time': day_ts,
                'cash_ratio': 0,
            })
            cash += sell_price * volume * (1 - COMMISSION - 0.001)

        for code in all_sell_codes:
            current_holdings.pop(code, None)

        # 买入
        if need_rebalance:
            new_stocks = select_stocks_optimized(
                daily_data, MV_MIN, MV_MAX, STOCK_NUM,
                turnover_min=TURNOVER_MIN,
                industry_diversify=INDUSTRY_DIVERSIFY
            )

            if len(new_stocks) > 0:
                vols = [v for _, _, _, _, v in new_stocks]
                inv_vols = [1/max(v, 0.01) for v in vols]
                total_inv_vol = sum(inv_vols)

                for code, name, industry, market_cap, volatility in new_stocks:
                    day_row = daily_data.filter(pl.col('code') == code)
                    if len(day_row) == 0:
                        continue
                    buy_price = day_row['open'].to_list()[0]

                    weight = (1/max(volatility, 0.01)) / total_inv_vol
                    alloc = cash * weight
                    unit_cost = buy_price * (1 + SLIPPAGE)
                    volume = int(alloc // unit_cost)
                    volume_real = (volume // 100) * 100

                    if volume_real < 100:
                        continue

                    orders_list.append({
                        'datetime': day_ts,
                        'code': code,
                        'direction': 1,
                        'price': buy_price,
                        'volume': volume_real,
                        'buy_time': day_ts,
                        'sell_time': None,
                        'cash_ratio': weight,
                    })
                    cash -= buy_price * volume_real * (1 + SLIPPAGE) * (1 + COMMISSION)
                    current_holdings[code] = {
                        'volume': volume_real,
                        'buy_price': buy_price,
                        'buy_time': day_ts,
                    }

        if day_count % 20 == 0:
            logging.info(f'{day} 调仓，现金: {cash:.2f}')

orders_df = pd.DataFrame(orders_list)
orders_df['datetime'] = pd.to_datetime(orders_df['datetime'])
logging.info(f'信号生成完成: 买入{len(orders_df[orders_df["direction"]==1])}笔, 卖出{len(orders_df[orders_df["direction"]==-1])}笔')

# ========== 回测 ==========
from my_backtester.my_backtester import Backtester

backtester = Backtester(
    orders=orders_df,
    initial_cash=INITIAL_CASH,
    commission=COMMISSION,
    slippage=SLIPPAGE
)

logging.info('开始回测...')
result = backtester.run(start_time=START_DATE, end_time=END_DATE)
logging.info('回测完成！')

# ========== 报告 ==========
metrics, fig = backtester.report(
    start_date=pd.Timestamp(START_DATE),
    end_date=pd.Timestamp(END_DATE)
)
fig.show()
logging.info('\n' + metrics.to_string(index=False))