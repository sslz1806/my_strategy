#%% 初始化数据
import sys
sys.path.append(r'C:\Users\20561\Desktop\策略')
from my_utils.fun import *
import polars as pl
import pandas as pd
import datetime as dt
import time 

logging = get_logger(log_file='log/实盘.log',inherit=False)

start_date = dt.date(2025,12,1)
end_date = dt.datetime.today()
#end_date = dt.date(2026,2,12)

# 获取指定日期的日线数据
stock_data = read_day_data(start_date=start_date,end_date=end_date,file_path='gm_stock_all_data')
# 将市值列除以1e8进行缩放
stock_data = stock_data.with_columns([
   (pl.col('mv_A_free_float') / 1e8).alias('mv_A_free_float'),
   (pl.col('total_mv') / 1e8).alias('total_mv')
])
#%% 添加最新一天数据
from my_utils.mapping import *
from my_utils.stock_api import *
api = stock_api()

def gm_add_auction(stock_data):
    """"
    利用掘金接口增加早盘数据current(symbols=stock_list,include_call_auction=True),主要是获取open即可
    分成pl和pd分别处理
    """
    # 将stock_data最后一天的股票代码
    if isinstance(stock_data, pl.DataFrame):
        stock_data = stock_data.sort(['trading_date', 'code'])
        last_date = stock_data.select(pl.col('trading_date').max()).item()
        stock_list = stock_data.filter(pl.col('trading_date') == last_date).select(pl.col('code')).to_series().to_list()

        new_data = current(symbols=stock_list,include_call_auction=True)
        new_data = pd.DataFrame(new_data)
        new_data['trading_date'] = new_data['created_at']
        # 清洗数据
        new_data = clean_stocks_data(new_data)

        # 1. 将ts_data转为Polars
        new_data_pl = pl.from_pandas(new_data)
        
        # new_data_pl = new_data_pl.with_columns(
        #     pl.col('trading_date').str.strptime(pl.Date, "%Y-%m-%d").alias('trading_date')
        # )
        
        # 2. 统一所有列的数据类型（核心修复）
        # 先获取stock_data的完整schema
        target_schema = stock_data.schema
        
        # 逐个处理列：存在的列强制转换类型，不存在的列添加并设置类型
        for col, dtype in target_schema.items():
            if col in new_data_pl.columns:
                # 强制转换已有列的类型为stock_data的类型
                new_data_pl = new_data_pl.with_columns(
                    pl.col(col).cast(dtype).alias(col)
                )
            else:
                # 添加缺失列并设置类型
                new_data_pl = new_data_pl.with_columns(
                    pl.lit(None, dtype=dtype).alias(col)
                )
        
        
        # 4. 严格按照stock_data的列顺序排序
        new_data_pl = new_data_pl.select(stock_data.columns)

        
        
        # 5. 合并并重新排序（关键：确保时间顺序正确）
        concat_data = stock_data.vstack(new_data_pl, in_place=False)
        concat_data = concat_data.sort(by=['code', 'trading_date'])  # 按股票+日期排序

        # 6. 用前一交易日的close填充pre_close（核心修正）
        if 'pre_close' in concat_data.columns and 'close' in concat_data.columns:
            concat_data = concat_data.with_columns(
                pl.when(pl.col('pre_close').is_null())
                .then(pl.col('close').shift(1).over('code'))  # 取同一股票前一天的close
                .otherwise(pl.col('pre_close'))
                .alias('pre_close')
            )
        
        # 7. 补充pct有缺失的数据为(close/pre_close-1)*100
        if 'pct' in concat_data.columns and 'close' in concat_data.columns and 'pre_close' in concat_data.columns:
            concat_data = concat_data.with_columns(
                pl.when(pl.col('pct').is_null())
                .then((pl.col('close') / pl.col('pre_close') - 1) * 100)
                .otherwise(pl.col('pct'))
                .alias('pct')
            )

        # 7.填充缺失值,[free_float_mv,name,type_name,type,industry]这些列如果有缺失,则用前一交易日的填充（核心修正）
        need_cols = ['free_float_mv', 'name', 'type_name', 'type', 'industry', 'mv_A_free_float', 'total_mv', 'is_st']
        for col in need_cols:
            if col in concat_data.columns:
                concat_data = concat_data.with_columns(
                    pl.col(col).fill_null(pl.col(col).shift(1).over('code')).alias(col)
                )

        
        
    elif isinstance(stock_data, pd.DataFrame):
        # 1. 原始数据排序
        stock_data_sorted = stock_data.sort_values(by=['code', 'trading_date']).reset_index(drop=True)
        # 2. 取最后一个交易日
        last_date = stock_data_sorted['trading_date'].unique().max()
        # 3. 提取最后交易日的所有股票代码列表
        stock_list = stock_data_sorted[stock_data_sorted['trading_date'] == last_date]['code'].tolist()

        new_data = current(symbols=stock_list,include_call_auction=True)
        new_data = pd.DataFrame(new_data)
        new_data['trading_date'] = new_data['created_at']
        # 清洗数据
        new_data = clean_stocks_data(new_data)

        # 4. 获取需要给 ts_data 补充的列（stock_data 有而 ts_data 没有的列）
        # 使用 reindex 自动补齐并保留列顺序（pandas 会用 NaN/NaT 填充）
        new_data = new_data.reindex(columns=stock_data.columns)
        concat_data = pd.concat([stock_data, new_data], ignore_index=True)
        concat_data = concat_data.sort_values(by=['code', 'trading_date'])  # 按股票+日期排序
        
        # 5. 用前一交易日的close填充pre_close（核心修正）
        if 'pre_close' in concat_data.columns and 'close' in concat_data.columns:
            concat_data['pre_close'] = concat_data.groupby('code').apply(
                lambda group: group['pre_close'].fillna(group['close'].shift(1))
            ).reset_index(level=0, drop=True)  # 取同一股票前一天的close
    return concat_data


today = dt.date.today()
#today = dt.date(2026,2,13)
today_str = today.strftime("%Y-%m-%d")
# 使用示例：
if stock_data is not None:
    if today in stock_data['trading_date'].unique():
        logging.info(f"验证成功：数据中已包含 {today_str} 的行情")
        stocks_data = stock_data
    else:
        #stocks_data = ts_add_auction(stocks_data,m_ts)
        stocks_data = gm_add_auction(stock_data)
        # 检查是否成功添加了今天的数据
        if today in stocks_data['trading_date'].unique():
            if isinstance(stocks_data, pl.DataFrame):
                logging.info(f"新增数据行数: {stocks_data.filter(pl.col('trading_date') == today).height}")
            else:
                logging.info(f"新增数据行数: {len(stocks_data[stocks_data['trading_date'] == today])}")
        else:
            logging.info(f"添加失败未发现 {today_str} 的新增数据")
else:
    logging.info("没有历史数据可添加最新行情")


# 增加指标
stocks_data = stocks_data.with_columns(
    (
        (pl.col("open") - pl.col("pre_close")) 
        / pl.col("pre_close") 
        * 100
    ).alias("open_pct"),  # 开盘涨幅百分比
    #((pl.col("close") - pl.col("sma_7")) / pl.col("sma_7") * 100).alias("close_sma7_pct"), #乖离率
    #(pl.col("amount")*100 / pl.col("volume")).alias("vwap"),
    ((pl.col("low") <= pl.col("limit_down")*1.01)).alias("touch_limit_down"), # 是否触及跌停
    (pl.col("close")/pl.col("pre_close")-1).alias("pct")
)

#%% qmt下单卖出
if __name__ == "__main__":
    from my_utils.my_qmt import *
    from my_utils.mapping import convert_code_format
    import time 
    positions = positions_df()
    code_list = positions['证券代码'].tolist()

    #logging.info('-'*18 + '【%s】' % asset.account_id + '-'*18)
    if not positions.empty:logging.info(f"持仓股票: {code_list}\n")
    
    # 批量异步下单：5笔订单，瞬间提交
    seq_list = []  # 存所有请求序号
    stop_loss = 0.08

    # 遍历positions
    for i, row in enumerate(positions.itertuples()):
        code = convert_code_format(row.证券代码)
        volume = row.可用数量
        cost = row.开仓价格

        if volume <= 0:
            continue
        # 从stocks_data查询股票当前pct,如果pct>=7，则不卖出
        code_data = stocks_data.filter(pl.col('code') == code).filter(pl.col('trading_date') == today)
        if code_data.height == 0:
            logging.info(f"{code} 没有找到当天数据，无法判断是否卖出，跳过")
            continue
        current_price = code_data.select(pl.col('close')).item()
        current_pct = code_data.select(pl.col('pct')).item()*100
        open_price = code_data.select(pl.col('open')).item()
        open_pct = (open_price / code_data.select(pl.col('pre_close')).item() - 1) * 100
        
        # 1. 开盘跌幅超过6%,卖出
        if open_pct <= -6:
            logging.info(f"{code} 开盘跌幅 {open_pct:.2f}% 超过6% 卖出")
            # 异步下单：这行代码0.001秒就完成，只返回请求序号，不等待券商处理
            seq = xt_trader.order_stock_async(
                account=ID,
                stock_code=convert_code_format(code,format='suffix'),  # 转换成带后缀的格式
                order_type=xtconstant.STOCK_SELL,
                order_volume=volume,
                price_type=xtconstant.MARKET_MINE_PRICE_FIRST, # MARKET_MINE_PRICE_FIRST本方最优,LATEST_PRICE最新价
                price=10.5 + i,
                strategy_name="strategy1",
                order_remark=f"async_order_{i+1}"
            )
            seq_list.append(seq)
            continue  # 如果开盘跌幅超过6%，直接卖出，不再执行后续条件判断

        # 2. 止损卖出
        if current_price <= cost * (1 - stop_loss):
            logging.info(f"{code} 当前价格 {current_price:.2f} 跌破止损线 卖出")
            seq = xt_trader.order_stock_async(
                account=ID,
                stock_code=convert_code_format(code,format='suffix'),  # 转换成带后缀的格式
                order_type=xtconstant.STOCK_SELL,
                order_volume=volume,
                price_type=xtconstant.MARKET_MINE_PRICE_FIRST, # MARKET_MINE_PRICE_FIRST本方最优,LATEST_PRICE最新价
                price=10.5 + i,
                strategy_name="strategy1",
                order_remark=f"async_order_{i+1}"
            )
            seq_list.append(seq)
            continue
        
        # 3. 中午和尾盘时间,涨幅小于7%，卖出
        if (11 <= dt.datetime.now().hour < 13 or dt.datetime.now().hour >= 14) and current_pct <= 7:
            logging.info(f"{code} 当前涨幅 {current_pct:.2f}% 小于7% 卖出")
            seq = xt_trader.order_stock_async(
                account=ID,
                stock_code=convert_code_format(code,format='suffix'),  # 转换成带后缀的格式
                order_type=xtconstant.STOCK_SELL,
                order_volume=volume,
                price_type=xtconstant.MARKET_MINE_PRICE_FIRST, # MARKET_MINE_PRICE_FIRST本方最优,LATEST_PRICE最新价
                price=10.5 + i,
                strategy_name="strategy1",
                order_remark=f"async_order_{i+1}"
            )
            seq_list.append(seq)
            logging.info(f"第{i+1}单提交成功，请求序号：{seq}")
            continue

    # 后续：券商后台同时处理这5单，1秒后通过回调返回所有订单号
    # 这里模拟等待回调结果（实际是自动触发on_order_stock_async_response）
    time.sleep(1)
    logging.info("全部处理完成，回调收到所有订单号")