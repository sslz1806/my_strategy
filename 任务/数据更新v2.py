"""
掘金数据源数据更新脚本 v2

功能:
1. 更新日线基础数据 (gm_stock_all_data)
2. 更新15分钟数据 (15min_stock_data_dir 或 gm_15min_stock_data_dir)
"""
import argparse
import sys
from typing import Optional
DATA_ROOT_DIR = r'E:\working\stock_data'
sys.path.append("C://Users/20561/Desktop/策略")

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

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

GM_MINUTE_HISTORY_WINDOW_DAYS = 180


def parse_args(argv=None):
    """解析命令行参数；默认值保持 run_update_data.bat 的日常增量行为。"""
    parser = argparse.ArgumentParser(description="掘金数据源数据更新脚本 v2")
    parser.add_argument("--start-date", default="2025-01-01", help="起始日期 YYYY-MM-DD")
    parser.add_argument(
        "--end-date",
        default=datetime.date.today().strftime("%Y-%m-%d"),
        help="结束日期 YYYY-MM-DD",
    )
    parser.add_argument(
        "--mode",
        choices=["insert", "update"],
        default="insert",
        help="insert=增量; update=按指定范围覆盖",
    )
    parser.add_argument("--skip-day", action="store_true", help="跳过日线更新")
    parser.add_argument("--skip-min", action="store_true", help="跳过 15 分钟更新")
    parser.add_argument(
        "--allow-old-min",
        action="store_true",
        help="跳过 GM 分钟线最近 180 天权限保护；仅在账号已开通更长历史分钟权限时使用",
    )
    parser.add_argument(
        "--min-align",
        choices=["left", "right", "both"],
        default="both",
        help="分钟线对齐方式",
    )
    return parser.parse_args(argv)


def parse_cli_date(value: str) -> datetime.date:
    """将 CLI 日期字符串转换为 date，错误格式交给 argparse 风格异常提示。"""
    return datetime.datetime.strptime(value, "%Y-%m-%d").date()


def gm_minute_history_floor(today: Optional[datetime.date] = None) -> datetime.date:
    """计算 GM 普通分钟 Bar 权限当前允许查询的最早日期。"""
    if today is None:
        today = datetime.date.today()
    return today - datetime.timedelta(days=GM_MINUTE_HISTORY_WINDOW_DAYS)


def check_gm_minute_history_window(
    start_date: datetime.date,
    today: Optional[datetime.date] = None,
) -> tuple[bool, datetime.date]:
    """
    判断分钟线起始日期是否落在 GM 普通权限窗口内。

    GM 接口在未开通更长历史权限时，只允许拉取最近 180 个自然日的分钟 Bar。
    历史补数如果先按 update 模式清理旧分区、再发现接口无权限，会造成无意义
    的失败循环；因此在进入分钟线写入逻辑前先做保护。
    """
    min_allowed_date = gm_minute_history_floor(today)
    return start_date >= min_allowed_date, min_allowed_date


def remove_date_partitions(save_dir: str, dates: list[datetime.date]) -> None:
    """
    只删除指定交易日分区。

    历史 update 模式需要覆盖旧分区，但不能影响范围外数据；因此按
    `trading_date=YYYY-MM-DD` 精确删除目标日期。
    """
    import shutil

    target_dir = os.path.join(DATA_ROOT_DIR, save_dir)
    for date in dates:
        partition_dir = os.path.join(target_dir, f"trading_date={date:%Y-%m-%d}")
        if os.path.exists(partition_dir):
            shutil.rmtree(partition_dir)
            print(f"已清理旧分区: {partition_dir}")


#%% 更新日线基础数据
def update_day_data_gm(day_data, save_dir='gm_stock_all_data', mode='insert'):
    """
    更新日线基础数据到Parquet分区文件

    参数:
        day_data: polars DataFrame，包含日线数据
        save_dir: 保存目录名称
        mode: 更新模式，'insert'表示增量更新，'update'表示全量更新
    """
    save_dir_name = save_dir
    save_dir = os.path.join(DATA_ROOT_DIR, save_dir_name)

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
        if mode == 'update':
            dates_to_update = (
                new_data.select(pl.col("trading_date").unique().sort()).to_series().to_list()
            )
            remove_date_partitions(save_dir_name, dates_to_update)
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
def update_min_data_by_day_data_gm(
    day_data,
    min_data_dir='15min_stock_data_dir',
    n=15,
    align='left',
    mode='insert',
):
    """
    day_data:polars DataFrame,包含交易日和股票代码等信息
    min_data_dir:分钟数据文件存储目录,parquet格式
    align:bar时间戳对齐方式,'left'左对齐(默认,datetime=bar开始时间,11:30/15:00补close快照)
          /'right'右对齐(datetime=bar结束时间,09:30/13:00补open快照),均为18根bar/天,
          不同对齐方式必须存到不同目录
    从day_data中获取交易日,然后对每个交易日中的股票,更新对应的分钟数据文件
    """
    import os
    from my_utils.stock_api import stock_api
    api = stock_api()

    # 没有目录则创建
    min_data_dir_name = min_data_dir
    min_data_dir = os.path.join(DATA_ROOT_DIR, min_data_dir_name)
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
    if mode == 'update':
        dates_to_update = trading_dates
        remove_date_partitions(min_data_dir_name, dates_to_update)
    else:
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
                max_workers=16,
                align=align
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


def main(argv=None) -> int:
    """脚本入口：支持日常增量和指定历史区间覆盖更新。"""
    args = parse_args(argv)
    requested_start_date = parse_cli_date(args.start_date)
    end_date = parse_cli_date(args.end_date)

    print("=" * 70)
    print("掘金数据源数据更新脚本 v2")
    print("=" * 70)

    if requested_start_date > end_date:
        print(f"起始日期大于结束日期，无需更新: {requested_start_date} > {end_date}")
        return 0

    print("\n" + "=" * 70)
    print("步骤1: 读取现有数据，获取最新日期")
    print("=" * 70)

    gm_data_dir = os.path.join(DATA_ROOT_DIR, 'gm_stock_all_data')
    exsist_data = None
    latest_date = None

    if os.path.exists(gm_data_dir):
        try:
            exsist_data = read_day_data(
                start_date=requested_start_date,
                end_date=end_date,
                file_path='gm_stock_all_data'
            )
            if exsist_data is not None and not exsist_data.is_empty():
                latest_date = exsist_data.select(pl.col("trading_date").max()).item()
                print(f"✓ 指定范围内现有数据最新日期: {latest_date}")
                print(f"  总记录数: {exsist_data.height}")
                print(f"  字段数: {len(exsist_data.columns)}")
                print(f"\n字段列表:")
                for i, col in enumerate(exsist_data.columns, 1):
                    print(f"  {i}. {col}")
            else:
                print("指定范围内暂无现有日线数据")
        except Exception as e:
            print(f"⚠ 读取现有数据失败: {e}")
            exsist_data = None
    else:
        print(f"ℹ 数据目录不存在，将创建新数据: {gm_data_dir}")
        os.makedirs(gm_data_dir, exist_ok=True)

    update_start_date = requested_start_date
    if args.mode == "insert" and latest_date is not None:
        # 日常增量：从指定范围内已有最新日期的下一天开始；历史 update 不覆盖用户起点。
        update_start_date = latest_date + datetime.timedelta(days=1)

    print("\n" + "=" * 70)
    print("步骤2: 获取并更新日线数据")
    print("=" * 70)

    if args.skip_day:
        print("已指定 --skip-day，跳过日线更新")
    else:
        print(f"更新日期范围: {update_start_date} ~ {end_date}")
        if update_start_date > end_date:
            print("数据已是最新，无需更新")
        else:
            print(f"需要获取 {(end_date - update_start_date).days + 1} 天的数据")
            print("\n正在获取掘金数据，请稍候...")
            day_data_df = api.gm_get_daily_data_multi_dates(
                start_date=str(update_start_date),
                end_date=str(end_date)
            )

            if day_data_df is not None and not day_data_df.empty:
                print(f"\n✓ 数据获取成功！共 {len(day_data_df)} 条记录")
                print(f"  交易日数量: {day_data_df['trading_date'].nunique()}")
                print(f"  股票数量: {day_data_df['code'].nunique()}")

                day_data_pl = pl.from_pandas(day_data_df)
                if day_data_pl['trading_date'].dtype != pl.Date:
                    day_data_pl = day_data_pl.with_columns(
                        pl.col('trading_date').cast(pl.Date).alias('trading_date')
                    )

                update_day_data_gm(day_data_pl, save_dir='gm_stock_all_data', mode=args.mode)
            else:
                print("✗ 数据获取失败或返回空数据")

    print("\n" + "=" * 70)
    print("步骤3: 更新分钟数据")
    print("=" * 70)

    if args.skip_min:
        print("已指定 --skip-min，跳过分钟数据更新")
    else:
        min_start_date = update_start_date if not args.skip_day else requested_start_date
        min_window_ok, min_allowed_date = check_gm_minute_history_window(min_start_date)
        if not min_window_ok and not args.allow_old_min:
            print(
                f"掘金分钟线接口当前普通权限只允许最近 {GM_MINUTE_HISTORY_WINDOW_DAYS} 个自然日 Bar，"
                f"最早可查日期约为 {min_allowed_date}；本次分钟起始日期为 {min_start_date}。"
            )
            print(
                "已在清理分钟分区前停止。若确认账号已开通更长历史分钟权限，"
                "可加 --allow-old-min 强制执行。"
            )
            return 2
        try:
            minute_source_data = read_day_data(
                start_date=min_start_date,
                end_date=end_date,
                file_path='gm_stock_all_data'
            )
        except Exception as e:
            print(f"读取分钟更新所需日线数据失败，跳过分钟更新: {e}")
            minute_source_data = None

        if minute_source_data is not None and not minute_source_data.is_empty():
            if args.min_align in ("left", "both"):
                # 左对齐目录(兼容现有回测/实盘代码): datetime=bar开始时间。
                update_min_data_by_day_data_gm(
                    minute_source_data,
                    min_data_dir='15min_stock_data_dir',
                    n=15,
                    align='left',
                    mode=args.mode,
                )
            if args.min_align in ("right", "both"):
                # 右对齐目录: datetime=bar结束时间，与左对齐快照逻辑镜像。
                update_min_data_by_day_data_gm(
                    minute_source_data,
                    min_data_dir='15min_stock_data_right_dir',
                    n=15,
                    align='right',
                    mode=args.mode,
                )
        else:
            print("没有可用于分钟更新的日线数据，跳过分钟数据更新")

    print("\n" + "=" * 70)
    print("验证更新结果")
    print("=" * 70)

    try:
        verify_start = requested_start_date if args.mode == "update" else update_start_date
        if verify_start <= end_date:
            updated_data = read_day_data(
                start_date=verify_start,
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

            print(f"\n最近5个交易日数据量:")
            recent_stats = (
                updated_data
                .group_by('trading_date')
                .agg(pl.count().alias('count'))
                .sort('trading_date', descending=True)
                .head(5)
            )
            print(recent_stats.to_pandas().to_string(index=False))
        else:
            print("本次没有新增日期需要验证")

    except Exception as e:
        print(f"✗ 读取数据失败: {e}")

    print("\n" + "=" * 70)
    print("数据更新v2完成!")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
