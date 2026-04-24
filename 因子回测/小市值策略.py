"""
小市值策略回测 - 修正版
严格按照聚宽JoinQuant原策略逻辑实现

核心逻辑：
1. 筛选市值20-30亿股票，取最小3只
2. 每5个交易日调仓，开盘买入
3. 动态资金分配（当前现金/持仓数）
"""
import sys
sys.path.append("C://Users/20561/Desktop/策略")

import pandas as pd
import polars as pl
from my_utils.fun import read_day_data, get_data_trading_days, get_logger

logging = get_logger(log_file='因子回测/log/小市值策略.log', inherit=False)

# ========== 策略参数 ==========
START_DATE = '2024-01-04'  # 数据起始日
END_DATE = '2026-04-20'
INITIAL_CASH = 100_000.0
COMMISSION = 0.0003
SLIPPAGE = 0.000  # 聚宽默认滑点为0
STOCK_NUM = 3
REFRESH_RATE = 5
MV_MIN = 20
MV_MAX = 30

# ========== 读取数据 ==========
logging.info(f"读取数据: {START_DATE} 至 {END_DATE}")
stock_data = read_day_data(start_date=START_DATE, end_date=END_DATE)
stock_data = stock_data.with_columns([(pl.col('total_mv') / 1e8).alias('market_cap')])
logging.info(f"数据行数: {len(stock_data)}")

# ========== 选股函数 ==========
def select_stocks(daily_data, mv_min, mv_max, top_n):
    """选出市值最小的top_n只股票"""
    filtered = daily_data.filter(
        (pl.col('market_cap') >= mv_min) &
        (pl.col('market_cap') <= mv_max) &
        (~pl.col('is_st')) &
        (~pl.col('is_suspended'))
    )
    if len(filtered) == 0:
        return []
    return filtered.sort('market_cap', descending=False).head(top_n)['code'].to_list()

# ========== 生成交易信号 ==========
logging.info("生成交易信号...")
trading_days = get_data_trading_days(START_DATE, END_DATE)
logging.info(f"交易日数量: {len(trading_days)}")

orders_list = []
day_count = 0
current_holdings = {}  # code -> volume (持仓记录)

# 模拟账户现金（用于动态分配）
cash = INITIAL_CASH

for day in trading_days:
    day_ts = pd.Timestamp(day)
    day_count += 1

    daily_data = stock_data.filter(pl.col('trading_date') == day)

    # 每5个交易日调仓
    if day_count % REFRESH_RATE == 1:
        # === 卖出旧持仓 ===
        stocks_to_sell = list(current_holdings.keys())
        for code in stocks_to_sell:
            day_row = daily_data.filter(pl.col('code') == code)
            if len(day_row) == 0:
                logging.warning(f"{day} 卖出找不到 {code}")
                continue
            sell_price = day_row['open'].to_list()[0]
            volume = current_holdings[code]

            orders_list.append({
                'datetime': day_ts,
                'code': code,
                'direction': -1,
                'price': sell_price,
                'volume': volume,
                'buy_time': None,  # 卖出时不需指定buy_time
                'sell_time': day_ts,
                'cash_ratio': 0,
            })

            # 更新现金（卖出得到的钱）
            proceeds = sell_price * volume * (1 - COMMISSION - 0.001)  # 扣除佣金和印花税
            cash += proceeds

        current_holdings = {}

        # === 买入新股票 ===
        new_stocks = select_stocks(daily_data, MV_MIN, MV_MAX, STOCK_NUM)

        if len(new_stocks) > 0:
            # 聚宽原策略的资金分配方式：剩余现金 / 持仓数量
            alloc_per_stock = cash / min(len(new_stocks), STOCK_NUM)

            for code in new_stocks:
                day_row = daily_data.filter(pl.col('code') == code)
                if len(day_row) == 0:
                    continue
                buy_price = day_row['open'].to_list()[0]

                # 计算买入数量
                unit_cost = buy_price * (1 + SLIPPAGE)
                volume = int(alloc_per_stock // unit_cost)
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
                    'cash_ratio': 1.0 / STOCK_NUM,
                })

                # 扣除买入资金
                cost = buy_price * volume_real * (1 + SLIPPAGE) * (1 + COMMISSION)
                cash -= cost

                current_holdings[code] = volume_real

        logging.info(f"{day} 调仓，现金剩余: {cash:.2f}")

# 转换为DataFrame
orders_df = pd.DataFrame(orders_list)
orders_df['datetime'] = pd.to_datetime(orders_df['datetime'])

logging.info(f"信号生成完成: 买入{len(orders_df[orders_df['direction']==1])}笔, 卖出{len(orders_df[orders_df['direction']==-1])}笔")

# ========== 回测 ==========
from my_backtester.my_backtester import Backtester

backtester = Backtester(
    orders=orders_df,
    initial_cash=INITIAL_CASH,
    commission=COMMISSION,
    slippage=SLIPPAGE
)

logging.info("开始回测...")
result = backtester.run(start_time=START_DATE, end_time=END_DATE)

# ========== 输出结果 ==========
logging.info("回测完成！")
try:
    metrics_df, fig = backtester.report(
        start_date=pd.Timestamp(START_DATE),
        end_date=pd.Timestamp(END_DATE),
        benchmark_code='SHSE.000300',
        return_method='compound',
        plot=True
    )
    logging.info("\n" + "="*50)
    logging.info("回测指标")
    logging.info("="*50)
    logging.info("\n" + metrics_df.to_string(index=False))
    if fig:
        fig.show()
except Exception as e:
    logging.error(f"报告生成失败: {e}")
    import traceback
    traceback.print_exc()

# 保存结果
pd.DataFrame(backtester.trade_log).to_csv('因子回测/小市值策略交易记录.csv', index=False, encoding='utf-8-sig')
result.to_csv('因子回测/小市值策略资金变化.csv', index=False, encoding='utf-8-sig')
logging.info("结果已保存")