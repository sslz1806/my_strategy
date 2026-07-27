#%% 初始化数据
import sys
sys.path.append(r'C:\Users\20561\Desktop\策略')
from my_utils.fun import *
import polars as pl
import pandas as pd
import datetime as dt
import time 
import os
import json
import uuid
from pathlib import Path
from urllib import request, parse

logging = get_logger(log_file='log/实盘.log',inherit=False)

start_date = dt.date(2025,12,1) 
end_date = dt.datetime.today()
#end_date = dt.date(2026,2,12)

# 获取指定日期的日线数据
stock_data = read_day_data(start_date=start_date,end_date=end_date)
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
        need_cols = ['free_float_mv', 'name', 'type_name', 'type', 'industry','mv_A_free_float','total_mv','is_st']
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

#%% 计算特征
# 1.涨停标记
# 标记涨停状态：limit_status
stocks_data = mark_limit_status(stocks_data)
# 标记涨停描述：limit_desc
stocks_data = mark_limit_desc(stocks_data)
# 记录最近的一次涨停描述：last_limit_desc
stocks_data = mark_last_limit_desc(stocks_data)


# 2.均线特征
# 计算均线:sma_{window}
stocks_data = add_sma(stocks_data, window=5)
stocks_data = add_sma(stocks_data, window=7)

# 3.计算开盘涨幅:open_chg,乖离率,成交均价vwap
stocks_data = stocks_data.with_columns(
    (
        (pl.col("open") - pl.col("pre_close")) 
        / pl.col("pre_close") 
        * 100
    ).alias("open_pct"),  # 开盘涨幅百分比
    ((pl.col("close") - pl.col("sma_7")) / pl.col("sma_7") * 100).alias("close_sma7_pct"), #乖离率
    (pl.col("amount")*100 / pl.col("volume")).alias("vwap"),
    ((pl.col("low") <= pl.col("limit_down")*1.01)).alias("touch_limit_down"), # 是否触及跌停
    (pl.col("close")/pl.col("pre_close")-1).alias("pct")
)
stocks_data = cal_n_lowest(stocks_data)

# 筛选stock_data行 为买入信号 1. 昨日涨停or断板or炸板   2.今日低开-3%至-4%  3.昨日收盘在昨日日五线上   4. 最近一次涨停描述 !=一天一板 或者 非空
# 1. 先确保数据按股票和日期排序
stocks_data = stocks_data.sort(["code", "trading_date"])

# 2. 在每个股票组内计算移位数据（关键步骤）
stocks_data = stocks_data.with_columns([
    # 同一股票内的前一天涨停状态
    pl.col("limit_status").shift(1).over("code").alias("prev_limit_status"),
    # 同一股票内的前一天5日均线
    pl.col("sma_7").shift(1).over("code").alias("prev_sma_7"),
    #pl.col("ema_7").shift(1).over("code").alias("prev_ema_7"),
    #pl.col("volume_ratio_5").shift(1).over("code").alias("pre_volume_ratio_5"),
    pl.col("pct").shift(1).over("code").alias("pre_pct"),
    pl.col("vwap").shift(1).over("code").alias("pre_vwap"),
    #pl.col("ewma_volatility_8").shift(1).over("code").alias("pre_ewma_volatility_8"),
    pl.col("close_sma7_pct").shift(1).over("code").alias("pre_close_sma7_pct"),
    pl.col("amount").shift(1).over("code").alias("pre_amount"),
])

# 计算开盘相对昨日成交均价的折价（9:30可用，无未来信息）
stocks_data = stocks_data.with_columns([
    ((pl.col("open") / (pl.col("pre_vwap") / 100) - 1) * 100).alias("open_pre_vwap_pct"),
])

# 3. 基于组内移位后的数据筛选买入信号
# 条件参数字典
params_dict={
    'low':-5,
    'high':-2.5,
    'mv_min':35,
    'mv_max':1000,
    'prev_limit_status':['断板','炸板'],
    'avg_limit_turnover_5_min':-1,
    'pos30_max':1.8,
    'open_pre_vwap_pct_max':-4.6,
}

# 筛选数据并添加signal列
stocks_data = stocks_data.with_columns(
    # 构建筛选条件表达式
    signal = pl.when(
        # 非st,创业,科创
        ~(pl.col("is_st")) &
        ~(pl.col("code").str.split(".").list[1].str.starts_with("30") | 
          pl.col("code").str.split(".").list[1].str.starts_with("688") |
            pl.col("code").str.split(".").list[1].str.starts_with("90") |
            pl.col("code").str.split(".").list[1].str.starts_with("20")   
        ) &
        # 1. 昨日or断板or炸板（使用组内移位后的数据）
        (pl.col("prev_limit_status").is_in(params_dict['prev_limit_status'])) &

        # 2. 今日低开-3%至-4%
        (pl.col("open_pct") >= params_dict["low"]) & 
        (pl.col("open_pct") <= params_dict["high"]) &

        # 3. 昨日收盘在昨日5日均线上（注意你代码里写的是prev_sma_7，确认是否是笔误）
        (pl.col("pre_close") >= pl.col("prev_sma_7")) &

        # 4. 最近一次涨停描述 != 一天一板 且 非空
        (pl.col("last_limit_desc") != "1天1板") &
        (pl.col("last_limit_desc").is_not_null()) &

        # 5. 自由流通值区间
        (pl.col("mv_A_free_float") >= params_dict["mv_min"]) & 
        (pl.col("mv_A_free_float") <= params_dict["mv_max"]) &

        # 6. 最近的涨停平均换手率
        #(pl.col("avg_limit_turnover_5") >= params_dict["avg_limit_turnover_5_min"]) &

        # 6. 绝对位置不能太高，不能触发严重异动（参数字典控制阈值）
        ((pl.col("open")/pl.col("lowest_30")) <= params_dict["pos30_max"]) &

        # 7. 均衡版落地过滤：开盘相对昨日成交均价的折价幅度
        (pl.col("open_pre_vwap_pct") <= params_dict["open_pre_vwap_pct_max"]) &

        # 8. 昨日成交额不低于1亿（过滤流动性不足的票）
        (pl.col("pre_amount") >= 1e8)
        
    ).then(1).otherwise(0)
)


# 如果需要只筛选出signal=1的行（可选）
信号文件 = stocks_data.filter(pl.col("signal") == 1)

logging.info(f"回测信号参数: {params_dict}")

#%% 添加回测,判断风控
from my_utils.trade_fun import *
start_date_str = start_date.strftime("%Y-%m-%d")
end_date_str = end_date.strftime("%Y-%m-%d")
#end_date_str = '2025-11-17'

logging.info(f"回测时间区间: {start_date_str} 至 {end_date_str}")
result_df,merged_df = cal_trade_info(信号文件,trade_fun=trade,start_date=start_date_str,end_date=end_date_str)


from my_utils.trade_fun import report_backtest_full
from my_utils.fun import *

# 直接使用固定权重 0.45
merged_df = merged_df.with_columns((pl.col("profit") * 0.45).alias("weight_profit"))
result_df['weight_profit'] = result_df['profit'] * 0.45


# 回测结果汇报
logging.info("回测结果:")
#back_result = report_backtest_full(merged_df.to_pandas(), start_date='2025-01-01', end_date=end_date_str, profit_col='weight_profit')
# 将回测结果保存到日志中
#logging.info("\n" + back_result.to_string(index=False))
#%% qmt实盘买入
# 从merged_df筛选今天的交易股票
today_trades = merged_df.filter(pl.col("trading_date") == today)
today_trades = today_trades.sort(pl.col("open"), descending=True) # 按照开盘价排序，优先买入开盘价较高的股票
code_list = today_trades['code'].to_list()
name_list = today_trades['name'].to_list()
open_list = today_trades['open'].to_list()
仓位 = 0.45  # 固定仓位权重
logging.info('日期：{}'.format(today))
logging.info(f"今天的所有信号的股票:")
for i, (code, name) in enumerate(zip(code_list, name_list)):
    logging.info(f"  {i+1}. {code} - {name} - 仓位: {仓位:.4f}")

#%% qmt下单买入
if __name__ == "__main__":
    from my_utils.my_qmt import *
    from my_utils.mapping import convert_code_format
    import time 
    asset = xt_trader.query_stock_asset(ID)
    # 先判空再访问属性
    if not asset:
        logging.error("获取资产信息失败，终止执行")
        sys.exit(1)
    logging.info('-'*18 + '【%s】' % asset.account_id + '-'*18)
    logging.info(f"资产总额: {asset.total_asset}\n"
                    f"持仓市值：{asset.market_value}\n"
                    f"可用资金：{asset.cash}\n")
    # 给today_trades增加一列分配资金,分配资金=min(可用资金,总资产*仓位)
    available_cash = min(asset.cash, asset.total_asset * 仓位)
    #available_cash = asset.cash
    #available_cash = asset.total_asset * 仓位  # 直接使用可用资金进行分配
    allocated_cash = available_cash / len(code_list) if len(code_list) > 0 else 0
    logging.info(f"分配资金{available_cash:.2f}, 每单分配资金: {allocated_cash:.2f} ")
    # 记录开始时间
    start_time = time.time()

    # 批量异步下单：5笔订单，瞬间提交
    seq_list = []  # 存所有请求序号
    if len(code_list) == 0 or available_cash <= 0 or 仓位 <= 0:
        logging.info("没有符合条件的交易信号或资金不足，今日不执行任何订单。")
        sys.exit(0)
    logging.info(f"今日信号股票: {code_list}")
    add_rate = 0.006  # 买点上移0.6%，可以根据需要调整
    # input 超时自动同意：09:29:30 前可手动输入，到点未输入则自动 y
    import msvcrt
    cutoff_time = dt.time(9, 29, 00)
    now_time = dt.datetime.now().time()
    if now_time >= cutoff_time:
        # 已过截止时间，直接自动同意
        user_input = 'y'
        logging.info(f"当前时间 {now_time.strftime('%H:%M:%S')} ≥ 09:29:30，自动确认增加开盘价加0.6%作为下单价格")
    else:
        # 未到截止时间，显示提示并等待输入（超时自动 y）
        logging.info(f"当前时间 {now_time.strftime('%H:%M:%S')}，是否增加开盘价加0.6%作为下单价格？(y/n) 在 09:29:30 前未输入将自动同意")
        user_input = ''
        while dt.datetime.now().time() < cutoff_time:
            if msvcrt.kbhit():
                ch = msvcrt.getch().decode().lower()
                if ch == 'y' or ch == 'n':
                    user_input = ch
                    break
            time.sleep(0.1)
        if not user_input:
            user_input = 'y'
            logging.info("输入超时（已过 09:29:30），自动同意")
    if user_input == 'y':
        for i, (code, open_price,name) in enumerate(zip(code_list, open_list,name_list)):
            # 计算每单股票数量
            if allocated_cash > 0 and open_price > 0:
                order_vol = allocated_cash / open_price  # 计算理论上的订单数量
                if order_vol < 70 : # 80股以下不下单
                    logging.info(f"订单数量不足，跳过下单 {code}，理论数量: {order_vol:.2f}")
                    continue  # 跳过这单，资金留存在账户中
                elif order_vol>=80 and order_vol<100:
                    order_vol = 100  # 最小下单数量为100股
                else:
                    order_vol = round(allocated_cash / open_price / 100) * 100  # 四舍五入取整，按100股倍数下单

            else:
                logging.info(f"无法下单 {code}，分配资金: {allocated_cash:.2f}, 开盘价: {open_price:.2f}")
                continue  # 跳过这单
            
            logging.info(f"第{i+1}单提交成功,代码:{code},名称：{name},订单数量：{order_vol},分配资金：{allocated_cash:.2f}")
            # 异步下单：这行代码0.001秒就完成，只返回请求序号，不等待券商处理
            order_price = round(open_price * (1+add_rate), 2)
            seq = xt_trader.order_stock_async(
                account=ID,
                stock_code=convert_code_format(code,format='suffix'),  # 转换成带后缀的格式
                order_type=xtconstant.STOCK_BUY,
                order_volume=order_vol,
                price_type=xtconstant.FIX_PRICE,  #MARKET_PEER_PRICE_FIRST对手方最优,LATEST_PRICE最新价
                price=order_price,
                strategy_name="strategy1",
                order_remark=f"async_order_{i+1}"
            )
            seq_list.append(seq)
            logging.info(f"第{i+1}单提交成功，请求序号：{seq}")
        

    # 计算提交5单的耗时
    submit_time = time.time() - start_time
    logging.info(f"异步批量提交5单总耗时：{submit_time:.3f}秒")

    # 后续：券商后台同时处理这5单，1秒后通过回调返回所有订单号
    # 这里模拟等待回调结果（实际是自动触发on_order_stock_async_response）
    time.sleep(1)
    logging.info("全部处理完成，回调收到所有订单号")