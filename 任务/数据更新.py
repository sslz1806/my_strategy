import sys
DATA_ROOT_DIR = r'E:\working\stock_data'
# 替换为fun.py实际所在的绝对路径（比如："D:/projects/common_code"） D:\桌面\策略\fun.py
sys.path.append("C://Users/20561/Desktop/策略") #C:\Users\20561\Desktop\策略\任务
# 对stock_data_partitioned进行维护(股票基础数据,是否st,流通市值,复权因子)
import polars as pl
import datetime
import tinyshare as tns
import mapping
import pandas as pd
from fun import *
import datetime as dt
ts_token = 'YzAEH11Yc7jZCHjeJa63fnbpSt3k9Je3GvWn0390oiBKO95bVJjP7u5L34e2ff6b'
ts =tns.pro_api(ts_token)
mins_token = 'fbdsJ45z9Nodp7FbUgDEsm1Oi8boH7Wuiqn7cQJnRAvs5bSwuB4e0iOBbe16ef40'
m_ts =tns.pro_api(mins_token)

# stock_data = pl.read_parquet("stock_data_partitioned")
# print(stock_data)  

mins = read_min_data(start_time=dt.datetime(2025,11,10),end_time=dt.datetime(2025,11,11))
day = read_day_data(start_date=dt.datetime(2025,1,1),end_date=dt.datetime.today(),file_path='ts_stock_all_data')
adj = read_day_data(start_date=dt.datetime(2025,8,1),end_date=dt.datetime(2025,10,1),file_path='ts_adj')
mkt = read_day_data(start_date=dt.datetime(2025,1,1),end_date=dt.datetime.today(),file_path='ts_daily_basic')
start_date = day.select(pl.col("trading_date").max()).item()
end_date = datetime.date.today()
# 打印列信息
# print(f"日数据列信息:\n {day.schema}\n个数:{len(day.columns)}")
# print(f"分钟数据列信息:\n {mins.schema}\n个数:{len(mins.columns)}")

#%% 更新日线基础数据
from stock_api import *
from fun import *
api = stock_api()
#api.ts_download_date_data(filename='ts_stock_all_data(pyarrow)',start_date='2025-10-01',end_date='2025-11-11',max_workers=8)
def update_day_data(day_data,save_dir='ts_stock_all_data',mode='insert'):
    save_dir = os.path.join(DATA_ROOT_DIR, save_dir)
    # 没有目录则创建
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    existing_dates = []
    # 1.遍历目录中所有以"trading_date="开头的分区目录获取已有日期
    for item in os.listdir(save_dir):
        if item.startswith("trading_date="):
            # 提取日期部分 (去掉"trading_date="前缀)
            date_str = item.split("=")[1]
            existing_dates.append(date_str)
    
    if mode=='insert':
        # 筛选出day_data中大于start_date的数据
        start_date = max(existing_dates)
        start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d').date()
        new_data = day_data.filter(pl.col("trading_date")>start_date)
    else:
        new_data = day_data
    # 2.获取已有数据的schema并转换
    existing_schema = get_parquet_dir_schema(save_dir)
    # 强制转换新数据的列类型以匹配已有schema
    if existing_schema:
        # 构建转换表达式
        convert_exprs = []
        for col, dtype in existing_schema.items():
            if col in new_data.columns:
                # 将列转换为已有schema中的类型
                convert_exprs.append(pl.col(col).cast(dtype).alias(col))
        # 执行转换
        new_data = new_data.select(convert_exprs)
        
        # 确保所有schema中的列都存在于新数据中
        missing_cols = [col for col in existing_schema.keys() if col not in new_data.columns]
        if missing_cols:
            print(f"警告: 新数据缺少以下列，已自动添加空值列: {missing_cols}")
            for col in missing_cols:
                new_data = new_data.with_column(pl.lit(None).cast(existing_schema[col]).alias(col))
    else:
        print("目录中没有数据,直接添加新数据")

    # 3.更新日线数据
    new_data = new_data.sort(['trading_date','code'])
    print(f"准备更新日线数据,共{new_data.height}条记录")
    new_data.write_parquet(save_dir,partition_by=['trading_date'])

today = datetime.date.today()
day = api.ts_get_stocks_data(start_date='2025-12-10',end_date=today.strftime('%Y-%m-%d'))
day = pl.from_pandas(day)
update_day_data(day,save_dir='ts_stock_all_data')

#%% 更新复权因子数据
def update_adj_factor_data(start_date='2021-01-01',end_date=None,save_dir='ts_adj',mode='insert'):
    """
    更新复权因子数据
    start_date:起始日期字符串,格式'YYYY-MM-DD'
    end_date:结束日期字符串,格式'YYYY-MM-DD',默认今天
    save_dir:保存目录
    mode:更新模式,'insert'表示增量更新,'update'表示全量更新
    """
    import os
    from stock_api import stock_api
    from mapping import convert_date_format
    import datetime
    import polars as pl
    from fun import get_parquet_dir_schema
    api = stock_api()
    start_date = convert_date_format(start_date,to_format='%Y-%m-%d')
    end_date = convert_date_format(end_date,to_format='%Y-%m-%d')
    save_dir = os.path.join(DATA_ROOT_DIR, save_dir)
    # 没有目录则创建
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if end_date is None:
        end_date = datetime.date.today().strftime('%Y-%m-%d')
    
    existing_dates = []
    # 1.遍历目录中所有以"trade_date="开头的分区目录获取已有日期
    for item in os.listdir(save_dir):
        if item.startswith("trading_date="):
            # 提取日期部分 (去掉"trade_date="前缀)
            date_str = item.split("=")[1]
            existing_dates.append(date_str)

    if existing_dates:
        exsisting_schema = get_parquet_dir_schema(save_dir)
    else:
        print("复权因子数据目录中没有数据,将直接添加新数据")
        exsisting_schema = {}
    
    if mode=='insert':
        # 筛选出需要更新的日期范围
        start_date_existing = max(existing_dates) if existing_dates else start_date
        start_date_dt = datetime.datetime.strptime(start_date_existing, '%Y-%m-%d').date() + datetime.timedelta(days=1)
        start_date = start_date_dt.strftime('%Y-%m-%d')
    
    # 2.获取复权因子数据
    adj_factor_df = api.ts_get_adj_factor(start_date=start_date,end_date=end_date)
    if adj_factor_df is None or adj_factor_df.empty:
        print("没有需要更新的数据,更新结束")
        return
    adj_factor_pl = pl.from_pandas(adj_factor_df)
    # 强制转换新数据的列类型以匹配已有schema
    if exsisting_schema:
        # 构建转换表达式
        convert_exprs = []
        for col, dtype in exsisting_schema.items():
            if col in adj_factor_pl.columns:
                # 将列转换为已有schema中的类型
                convert_exprs.append(pl.col(col).cast(dtype).alias(col))
        # 执行转换
        adj_factor_pl = adj_factor_pl.select(convert_exprs)
        
        # 确保所有schema中的列都存在于新数据中
        missing_cols = [col for col in exsisting_schema.keys() if col not in adj_factor_pl.columns]
        if missing_cols:
            print(f"警告: 新数据缺少以下列，已自动添加空值列: {missing_cols}")
            for col in missing_cols:
                adj_factor_pl = adj_factor_pl.with_column(pl.lit(None).cast(exsisting_schema[col]).alias(col))
    else:
        print("目录中没有数据,直接添加新数据")
    # 3.保存复权因子数据
    adj_factor_pl = adj_factor_pl.sort(['trading_date','code'])
    print(f"准备更新复权因子数据,共{adj_factor_pl.height}条记录")
    adj_factor_pl.write_parquet(save_dir,partition_by=['trading_date'])


update_adj_factor_data(start_date='2025-11-01',end_date=None,save_dir='ts_adj',mode='insert')

#%% 更新每日指标数据ts.daily_basic
def update_daily_basic_data(start_date='2021-01-01',end_date=None,save_dir='ts_daily_basic',mode='insert'):
    """
    更新每日指标数据ts.daily_basic
    start_date:起始日期字符串,格式'YYYY-MM-DD'
    end_date:结束日期字符串,格式'YYYY-MM-DD',默认今天
    save_dir:保存目录
    mode:更新模式,'insert'表示增量更新,'update'表示全量更新
    """
    import os
    from stock_api import stock_api
    from mapping import convert_date_format
    import datetime
    import polars as pl
    from fun import get_parquet_dir_schema
    api = stock_api()
    start_date = convert_date_format(start_date,to_format='%Y-%m-%d')
    end_date = convert_date_format(end_date,to_format='%Y-%m-%d')
    save_dir = os.path.join(DATA_ROOT_DIR, save_dir)
    # 没有目录则创建
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if end_date is None:
        end_date = datetime.date.today().strftime('%Y-%m-%d')
    
    existing_dates = []
    # 1.遍历目录中所有以"trade_date="开头的分区目录获取已有日期
    for item in os.listdir(save_dir):
        if item.startswith("trading_date="):
            # 提取日期部分 (去掉"trade_date="前缀)
            date_str = item.split("=")[1]
            existing_dates.append(date_str)

    if existing_dates:
        exsisting_schema = get_parquet_dir_schema(save_dir)
    else:
        print("每日指标数据目录中没有数据,将直接添加新数据")
        exsisting_schema = {}
    
    if mode=='insert':
        # 筛选出需要更新的日期范围
        start_date_existing = max(existing_dates) if existing_dates else start_date
        start_date_dt = datetime.datetime.strptime(start_date_existing, '%Y-%m-%d').date() + datetime.timedelta(days=1)
        start_date = start_date_dt.strftime('%Y-%m-%d')
    
    # 2.获取每日指标数据
    daily_basic_df = api.ts_get_daily_basic(start_date=start_date,end_date=end_date)
    if daily_basic_df is None or daily_basic_df.empty:
        print("没有需要更新的数据,更新结束")
        return
    daily_basic_pl = pl.from_pandas(daily_basic_df)
    # 强制转换新数据的列类型以匹配已有schema
    if exsisting_schema:
        # 构建转换表达式
        convert_exprs = []
        for col, dtype in exsisting_schema.items():
            if col in daily_basic_pl.columns:
                # 将列转换为已有schema中的类型
                convert_exprs.append(pl.col(col).cast(dtype).alias(col))
        # 执行转换
        daily_basic_pl = daily_basic_pl.select(convert_exprs)
        
        # 确保所有schema中的列都存在于新数据中
        missing_cols = [col for col in exsisting_schema.keys() if col not in daily_basic_pl.columns]
        if missing_cols:
            print(f"警告: 新数据缺少以下列，已自动添加空值列: {missing_cols}")
            for col in missing_cols:
                daily_basic_pl = daily_basic_pl.with_column(pl.lit(None).cast(exsisting_schema[col]).alias(col))
    else:
        print("目录中没有数据,直接添加新数据")
    # 3.保存每日指标数据
    daily_basic_pl = daily_basic_pl.sort(['trading_date','code'])
    print(f"准备更新每日指标数据,共{daily_basic_pl.height}条记录")
    daily_basic_pl.write_parquet(save_dir,partition_by=['trading_date'])
update_daily_basic_data(start_date='2021-01-01',end_date=None,save_dir='ts_daily_basic',mode='insert')

#%% 利用基础行情的数据更新分钟数据
#api.gm_batch_get_minute_data(symbols,start_time='2025-10-01',end_time='2025-11-11',frequency='900s',n=15,batch_size=50, max_workers=5)
def update_min_data_by_day_data(day_data,min_data_dir='15min_stock_data_dir',n=15):
    """
    day_data:polars DataFrame,包含交易日和股票代码等信息
    min_data_dir:分钟数据文件存储目录,parquet格式
    从day_data中获取交易日,然后对每个交易日中的股票,更新对应的分钟数据文件
    """
    import os
    from stock_api import stock_api
    api = stock_api()
    # 没有目录则创建
    min_data_dir = os.path.join(DATA_ROOT_DIR, min_data_dir)   
    if not os.path.exists(min_data_dir):
        os.makedirs(min_data_dir)
    # 1.从目录中获取已存在的分钟数据日期trading_date=%Y-%m-%d文件列表
    existing_dates = []
    for item in os.listdir(min_data_dir):
        if item.startswith("trading_date="):
            # 提取日期部分 (去掉"trading_date="前缀)
            date_str = item.split("=")[1]
            date_formal = datetime.datetime.strptime(date_str, '%Y-%m-%d').date()
            existing_dates.append(date_formal)

    # 读取第一个已有分区的schema作为基准（若存在）
    if existing_dates:
        base_schema = get_parquet_dir_schema(min_data_dir)
    else:
        print("分钟数据目录中没有数据,将直接添加新数据")
        base_schema = {}

    # 2.获取需要更新的交易日列表
    trading_dates = day_data.select(pl.col("trading_date").unique()).to_series().to_list()
    dates_to_update = [date for date in trading_dates if date not in existing_dates]
    print(f"需要更新的交易日有{len(dates_to_update)}个: {dates_to_update}")

    from tqdm import tqdm
    # 3.利用gm数据源获取分钟数据并更新
    with tqdm(total=len(dates_to_update), desc="整体进度", unit="交易日") as date_pbar:
        for date in dates_to_update:
            # 获取该交易日的所有股票代码
            codes = day_data.filter(pl.col("trading_date") == date).select(pl.col("code")).to_series().to_list()
            date_pbar.set_postfix({"当前交易日": str(date), "待处理股票数": len(codes)})
            print(f"正在更新{date}的{len(codes)}只股票分钟数据...")
            # 批量获取分钟数据,获取一天所有股票的分钟数据
            min_data = api.gm_batch_get_minute_data(symbols=codes,start_time=date,end_time=date+datetime.timedelta(days=1),frequency=f'{n*60}s',n=n,max_workers=16)
            
            if min_data is not None and not min_data.empty:
                min_data['trading_date'] = date
                min_data_pl = pl.from_pandas(min_data)
                min_data_pl = min_data_pl.sort(['trading_date','code','datetime'])
                # 强制转换列类型以匹配已有schema
                convert_exprs = []
                for col, dtype in base_schema.items():
                    if col in min_data_pl.columns:
                        # 将列转换为已有schema中的类型
                        convert_exprs.append(pl.col(col).cast(dtype).alias(col))
                # 执行转换
                min_data_pl = min_data_pl.select(convert_exprs)
                missing_cols = [col for col in base_schema.keys() if col not in min_data_pl.columns]
                if missing_cols:
                    print(f"警告: 新数据缺少以下列，已自动添加空值列: {missing_cols}")
                    for col in missing_cols:
                        min_data_pl = min_data_pl.with_column(pl.lit(None).cast(base_schema[col]).alias(col))

                # 保存为parquet分区文件
                min_data_pl.write_parquet(min_data_dir, partition_by=['trading_date'])
                success_count = len(min_data['code'].unique())
                print(f"{date}的分钟数据更新完成！成功获取{success_count}只股票数据，保存到{min_data_dir}目录。")
            else:
                print(f"{date}没有获取到分钟数据，跳过保存。")

            # 更新外层交易日进度条
            date_pbar.update(1)
    print(f"\n所有交易日处理完毕！共更新{len(dates_to_update)}个交易日的分钟数据")

# 增量更新分钟数据
update_min_data_by_day_data(day, min_data_dir='15min_stock_data_dir', n=15)
