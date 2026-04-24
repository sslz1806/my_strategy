import sys
sys.path.append("C://Users/20561/Desktop/策略") #C:\Users\20561\Desktop\策略\任务
import pandas as pd
from my_utils.stock_api import *
import numpy as np
import datetime as dt
from datetime import datetime,timedelta,time
import polars as pl
from my_utils.mapping import *
from concurrent.futures import ThreadPoolExecutor, as_completed

# 统计涨停情况(分主板，创业板，科创板进行统计)
today = dt.date.today()
start_date = (today-timedelta(days=50)).strftime("%Y-%m-%d")
end_date = (today-timedelta(days=1)).strftime("%Y-%m-%d")
"""
计算涨停情况
:param df: DataFrame, 包含股票数据
:return: DataFrame, 添加了涨停情况和连板次数的DataFrame
"""
api = stock_api()


stock_list = get_all_stocks(date=end_date) #排除st,创业,科创,以及排除流通市值过大或过小
# 剔除创业板和科创板股票
stock_list = [stock for stock in stock_list if not (stock.split('.')[1].startswith('30') or stock.split('.')[1].startswith('688'))]
#stocks_data =get_history_symbols(symbols=stock_list, start_date=start_date, end_date=end_date)
print(f"股票池个数:{len(stock_list)}")
stocks_data = api.batch_get_history_symbols(stock_list,start_date=start_date, end_date=end_date)

# 统一列名
stocks_data['pct'] = (stocks_data['close']-stocks_data['pre_close'])/stocks_data['pre_close'] *100
stocks_data['vwap']= stocks_data['amount']/stocks_data['volume']


#stock_list = stock_list[0:1000]  # 只取前10只股票进行测试
print(f"数据日期:{stocks_data['trading_date'].unique().min()} - {stocks_data['trading_date'].unique().max()}")

#%%
from my_utils.mapping import *

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
        # 清洗数据
        new_data = clean_stocks_data(new_data)

        # 1. 将ts_data转为Polars
        new_data_pl = pl.from_pandas(new_data)
        new_data_pl = new_data_pl.with_columns(
            pl.col('trading_date').str.strptime(pl.Date, "%Y-%m-%d").alias('trading_date')
        )
        
        # 2. 统一所有列的数据类型（核心修复）
        # 先获取stock_data的完整schema
        target_schema = stock_data.schema
        
        # 逐个处理列：存在的列强制转换类型，不存在的列添加并设置类型
        for col, dtype in target_schema.items():
            if col in ts_data_pl.columns:
                # 强制转换已有列的类型为stock_data的类型
                ts_data_pl = ts_data_pl.with_columns(
                    pl.col(col).cast(dtype).alias(col)
                )
            else:
                # 添加缺失列并设置类型
                ts_data_pl = ts_data_pl.with_columns(
                    pl.lit(None, dtype=dtype).alias(col)
                )
        
        
        # 4. 严格按照stock_data的列顺序排序
        ts_data_pl = ts_data_pl.select(stock_data.columns)
        
        # 5. 合并
        concat_data = stock_data.vstack(ts_data_pl, in_place=False)

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


today_str = dt.date.today().strftime("%Y-%m-%d")
# 使用示例：
if stocks_data is not None:
    if today in stocks_data['trading_date'].unique():
        print(f"验证成功：数据中已包含 {today_str} 的行情")
    else:
        #stocks_data = ts_add_auction(stocks_data,m_ts)
        stocks_data = gm_add_auction(stocks_data)
        # 检查是否成功添加了今天的数据
        if today in stocks_data['trading_date'].unique():
            print(f"新增数据行数: {len(stocks_data[stocks_data['trading_date'] == today])}")
        else:
            print(f"添加失败未发现 {today_str} 的新增数据")
else:
    print("没有历史数据可添加最新行情")


#%%
from my_utils.pd_fun import *
# 1. 先确保数据按股票代码和日期排序
stocks_data = stocks_data.sort_values(['code', 'trading_date']).reset_index(drop=True)
# 标记涨停状态：limit_status
stocks_data = mark_limit_status(stocks_data)
# 标记涨停描述：limit_desc
stocks_data = mark_limit_desc(stocks_data)
# 记录最近的一次涨停描述：last_limit_desc
stocks_data = mark_last_limit_desc(stocks_data)
# 计算均线:sma_{window}
stocks_data['sma_7'] = sma(stocks_data['close'],window=7)
stocks_data['sma_10'] = sma(stocks_data['close'],window=10)
stocks_data = cal_n_lowest(stocks_data)
stocks_data['close_sma7_pct'] = (stocks_data['close']-stocks_data['sma_7'])/stocks_data['sma_7']*100
stocks_data['open_pct']=(stocks_data['open']-stocks_data['pre_close'])/stocks_data['pre_close']*100


# 2. 在每个股票组内计算移位数据（关键步骤）
# 按股票分组后进行移位操作
stocks_data['prev_limit_status'] = stocks_data.groupby('code')['limit_status'].shift(1)
stocks_data['prev_sma_7'] = stocks_data.groupby('code')['sma_7'].shift(1)
stocks_data['prev_sma_10'] = stocks_data.groupby('code')['sma_10'].shift(1)
stocks_data['pre_pct'] = stocks_data.groupby('code')['pct'].shift(1)
stocks_data['pre_vwap'] = stocks_data.groupby('code')['vwap'].shift(1)
stocks_data['pre_close_sma7_pct'] = stocks_data.groupby('code')['close_sma7_pct'].shift(1)

# 计算收盘价与7日均线的百分比差值
stocks_data['close_sma7_pct'] = (stocks_data['close'] - stocks_data['sma_7']) / stocks_data['sma_7'] * 100

# 3. 基于组内移位后的数据筛选买入信号
low = -5
high = -2.5
信号文件 = stocks_data[
    # 1. 昨日是断板或炸板
    (stocks_data["prev_limit_status"].isin(["断板", "炸板"])) &
    
    # 2. 今日低开幅度在-3%至-4%之间（根据您的描述调整了高低值）
    (stocks_data["open_pct"] >= low) & (stocks_data["open_pct"] <= high) &
    
    # 3. 昨日收盘在昨日5日均线上
    (stocks_data["pre_close"] >= stocks_data["prev_sma_7"]) &
    
    # 4. 最近一次涨停描述不是一天一板且不为空
    (
        #(stocks_data["last_limit_desc"] != "1天1板") & 
        (stocks_data["last_limit_desc"].notnull()) 
    ) 

    # 5. 自由流通值在30亿到1000亿之间（经过stock_list已经筛选）
    # &(
    #     (stocks_data["mv_A_free_float"] >= 30 * 1e8) &
    #     (stocks_data["mv_A_free_float"] <= 1000 * 1e8)
    # ) 
    
    # 6. 绝对位置不能太高，30日最低点至今涨幅不超过3倍
    &(stocks_data["open"] / stocks_data["lowest_30"] <= 3)
]
#%% 
# 打印每日的股票代码
today_str = (dt.date.today()).strftime("%Y-%m-%d")
today_stocks = 信号文件[信号文件['trading_date']==today]
today_stocks.sort_values('close_sma7_pct')
today_stocks['name'] = api.get_stock_name(today_stocks['code'])
today_stocks['code'] = today_stocks['code'].apply(lambda x: x.split('.')[1])
filter_stocks = today_stocks[today_stocks['last_limit_desc']!='1天1板']
code_list = filter_stocks['code'].to_list()
# 提取每个code的后缀数字并打印
print('日期：{}'.format(today))
print('今日排除首板信号股票代码:')
for code in code_list:
    # 分割字符串，取小数点后的部分
    suffix_number = code
    print(suffix_number)

code_list = today_stocks['code'].to_list()
print('今日所有信号股票代码:')
for code in code_list:
    # 分割字符串，取小数点后的部分
    suffix_number = code
    print(suffix_number)


# 发送信号的邮件
from my_utils.email_fun import sendStringEmail,send_email
sender = '2056123357@qq.com'
user_list = ['2056123357@qq.com','1712167056@qq.com','1162690293@qq.com']
subject = f"{today_str} 买入信号股票代码"
content = f"今日排除首板信号股票代码:\n{filter_stocks['code'].to_list()}\n名称:{filter_stocks['name'].to_list()}\n\n今日所有信号股票代码:\n{today_stocks['code'].to_list()}\n名称:{today_stocks['name'].to_list()}"
html_content = f"""<body style="line-height:1.2;">
今日排除首板信号股票代码:<br>
{"".join([f"• {code}<br>" for code in filter_stocks['code'].to_list()])}
名称:<br>
{"".join([f"• {name}<br>" for name in filter_stocks['name'].to_list()])}
今日所有信号股票代码:<br>
{"".join([f"• {code}<br>" for code in today_stocks['code'].to_list()])}
名称:<br>
{"".join([f"• {name}<br>" for name in today_stocks['name'].to_list()])}
</body>
"""
send_email(
    subject=subject,
    body=html_content,
    sender_email=sender,
    receiver_emails=user_list,
    body_type="html"
)