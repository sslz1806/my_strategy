"""
掘金数据源数据更新脚本 v2

功能:
1. 更新日线基础数据 (gm_stock_all_data)
2. 更新15分钟数据 (15min_stock_data_dir 或 gm_15min_stock_data_dir)
"""
import sys
DATA_ROOT_DIR = r'E:\working\stock_data'
sys.path.append("C://Users/20561/Desktop/策略")

import polars as pl
import datetime
import pandas as pd
import os
from my_utils.fun import *
from my_utils.stock_api import *
from my_utils.mapping import *

# 初始化日志
logging = get_logger(log_file='log/数据更新v2.log', inherit=False)

# 初始化API
api = stock_api()

print("=" * 70)
print("掘金数据源数据更新脚本 v2")
print("=" * 70)

start_date = datetime.date(2025, 1, 1)  # 设置为你需要的起始日期
end_date = datetime.date.today()  # 设置为今天
#%% 更新日线基础数据
def update_day_data_gm(day_data, save_dir='gm_stock_all_data', mode='insert'):
    """
    更新日线基础数据到Parquet分区文件

    参数:
        day_data: polars DataFrame，包含日线数据
        save_dir: 保存目录名称
        mode: 更新模式，'insert'表示增量更新，'update'表示全量更新
    """
    save_dir = os.path.join(DATA_ROOT_DIR, save_dir)

    # 创建目录（如果不存在）
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"创建目录: {save_dir}")

    # 获取已有日期列表
    existing_dates = []
    for item in os.listdir(save_dir):
        if item.startswith("trading_date="):
            date_str = item.split("=")[1]
            existing_dates.append(date_str)

    if mode == 'insert' and existing_dates:
        # 增量更新：只保留大于最新日期的数据
        start_date_str = max(existing_dates)
        start_date = datetime.datetime.strptime(start_date_str, '%Y-%m-%d').date()
        new_data = day_data.filter(pl.col("trading_date") > start_date)
        print(f"增量更新模式: 保留 > {start_date} 的数据")
    else:
        new_data = day_data
        print(f"全量更新模式: 更新全部数据")

    if new_data.is_empty():
        print("没有新数据需要更新")
        return

    # 获取已有数据的schema并转换
    existing_schema = get_parquet_dir_schema(save_dir)

    if existing_schema:
        # 强制转换新数据的列类型以匹配已有schema
        convert_exprs = []
        for col, dtype in existing_schema.items():
            if col in new_data.columns:
                convert_exprs.append(pl.col(col).cast(dtype).alias(col))

        if convert_exprs:
            new_data = new_data.select(convert_exprs)

        # 确保所有schema中的列都存在于新数据中
        missing_cols = [col for col in existing_schema.keys() if col not in new_data.columns]
        if missing_cols:
            print(f"警告: 新数据缺少以下列，已自动添加空值列: {missing_cols}")
            for col in missing_cols:
                new_data = new_data.with_column(pl.lit(None).cast(existing_schema[col]).alias(col))
    else:
        print("目录中没有数据，直接添加新数据")

    # 排序并保存
    new_data = new_data.sort(['trading_date', 'code'])
    print(f"准备更新日线数据，共 {new_data.height} 条记录")
    new_data.write_parquet(save_dir, partition_by=['trading_date'])
    print(f"✓ 数据已保存到: {save_dir}")


#%% 利用基础行情的数据更新分钟数据
def update_min_data_by_day_data_gm(day_data, min_data_dir='15min_stock_data_dir', n=15):
    """
    day_data:polars DataFrame,包含交易日和股票代码等信息
    min_data_dir:分钟数据文件存储目录,parquet格式
    从day_data中获取交易日,然后对每个交易日中的股票,更新对应的分钟数据文件
    """
    import os
    from my_utils.stock_api import stock_api
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
            min_data = api.gm_batch_get_minute_data(
                symbols=codes,
                start_time=date,
                end_time=date + datetime.timedelta(days=1),
                frequency=f'{n*60}s',
                n=n,
                max_workers=16
            )

            if min_data is not None and not min_data.empty:
                min_data['trading_date'] = date
                min_data_pl = pl.from_pandas(min_data)
                min_data_pl = min_data_pl.sort(['trading_date', 'code', 'datetime'])

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


#%% 主程序执行
print("\n" + "=" * 70)
print("步骤1: 读取现有数据，获取最新日期")
print("=" * 70)

# 尝试读取现有掘金日线数据
gm_data_dir = os.path.join(DATA_ROOT_DIR, 'gm_stock_all_data')

if os.path.exists(gm_data_dir):
    try:
        exsist_data = read_day_data(
            start_date=start_date,
            end_date=end_date,
            file_path='gm_stock_all_data'
        )
        latest_date = exsist_data.select(pl.col("trading_date").max()).item()
        print(f"✓ 现有数据最新日期: {latest_date}")
        print(f"  总记录数: {exsist_data.height}")
        print(f"  字段数: {len(exsist_data.columns)}")
        print(f"\n字段列表:")
        for i, col in enumerate(exsist_data.columns, 1):
            print(f"  {i}. {col}")
    except Exception as e:
        print(f"⚠ 读取现有数据失败: {e}")
        exsist_data = None
        latest_date = datetime.date(2020, 1, 1)
else:
    print(f"ℹ 数据目录不存在，将创建新数据: {gm_data_dir}")
    exsist_data = None
    latest_date = datetime.date(2020, 1, 1)
    os.makedirs(gm_data_dir, exist_ok=True)


print("\n" + "=" * 70)
print("步骤2: 获取并更新日线数据")
print("=" * 70)

# 从最新日期的下一天开始
if exsist_data is not None:
    start_date = latest_date + datetime.timedelta(days=1)
else:
    # 首次运行，设置起始日期（可根据需要调整）
    start_date = datetime.date(2020, 1, 1)

print(f"更新日期范围: {start_date} ~ {end_date}")

# 检查是否需要更新
if start_date > end_date:
    print("数据已是最新，无需更新")
else:
    print(f"需要获取 { (end_date - start_date).days + 1 } 天的数据")

    # 调用掘金API批量获取数据
    print("\n正在获取掘金数据，请稍候...")
    day_data_df = api.gm_get_daily_data_multi_dates(
        start_date=str(start_date),
        end_date=str(end_date)
    )

    if day_data_df is not None and not day_data_df.empty:
        print(f"\n✓ 数据获取成功！共 {len(day_data_df)} 条记录")
        print(f"  交易日数量: {day_data_df['trading_date'].nunique()}")
        print(f"  股票数量: {day_data_df['code'].nunique()}")

        # 转换为Polars DataFrame
        day_data_pl = pl.from_pandas(day_data_df)

        # 确保trading_date是date类型
        if day_data_pl['trading_date'].dtype != pl.Date:
            day_data_pl = day_data_pl.with_columns(
                pl.col('trading_date').cast(pl.Date).alias('trading_date')
            )

        # 执行更新
        update_day_data_gm(day_data_pl, save_dir='gm_stock_all_data', mode='insert')
    else:
        print("✗ 数据获取失败或返回空数据")
        day_data_pl = None


print("\n" + "=" * 70)
print("步骤3: 更新分钟数据")
print("=" * 70)

exsist_data = read_day_data(
    start_date=datetime.date(2026, 1, 1),
    end_date=end_date,
    file_path='gm_stock_all_data'
)

if exsist_data is not None and not exsist_data.is_empty():
    # 增量更新分钟数据
    update_min_data_by_day_data_gm(
        exsist_data,
        min_data_dir='15min_stock_data_dir',
        n=15
    )
else:
    print("没有新的日线数据，跳过分钟数据更新")


print("\n" + "=" * 70)
print("验证更新结果")
print("=" * 70)

# 读取更新后的数据验证
try:
    updated_data = read_day_data(
        start_date=start_date,
        end_date=end_date,
        file_path='gm_stock_all_data'
    )

    print("✓ 数据读取成功")
    print(f"\n更新后统计:")
    print(f"  总记录数: {updated_data.height}")
    print(f"  最新日期: {updated_data.select(pl.col('trading_date').max()).item()}")
    print(f"  最早日期: {updated_data.select(pl.col('trading_date').min()).item()}")
    print(f"  交易日数量: {updated_data.select(pl.col('trading_date').n_unique()).item()}")
    print(f"  股票数量: {updated_data.select(pl.col('code').n_unique()).item()}")

    # 显示最近几天的数据统计
    print(f"\n最近5个交易日数据量:")
    recent_stats = (
        updated_data
        .group_by('trading_date')
        .agg(pl.count().alias('count'))
        .sort('trading_date', descending=True)
        .head(5)
    )
    print(recent_stats.to_pandas().to_string(index=False))

except Exception as e:
    print(f"✗ 读取数据失败: {e}")


print("\n" + "=" * 70)
print("数据更新v2完成!")
print("=" * 70)
