import sys
DATA_ROOT_DIR = r'E:\working\stock_data'
sys.path.append("C://Users/20561/Desktop/策略")

import polars as pl
import pandas as pd
import datetime
import os
import time
from fun import *
from mapping import *
from stock_api import stock_api
from functools import wraps, partial

logging = get_logger(log_file='log/指数ETF板块数据更新.log', inherit=False)

# 初始化API
api = stock_api()
ts = api.ts

# 重试装饰器
import threading
class TimeoutException(Exception):
    pass

from contextlib import contextmanager
@contextmanager
def timeout_context(seconds):
    """
    超时上下文管理器，用于Windows系统
    """
    timer = threading.Event()

    def handler():
        timer.set()

    t = threading.Timer(seconds, handler)
    t.start()

    try:
        yield timer
    finally:
        t.cancel()

def retry_with_timeout(max_retries=3, retry_interval=2, timeout=120):
    """
    函数重试装饰器
    Args:
        max_retries: 最大重试次数
        retry_interval: 重试间隔(秒)
        timeout: 超时时间(秒)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    # 调用被装饰的函数
                    with timeout_context(timeout):
                        return func(*args, **kwargs)
                except Exception as e:
                    # 记录错误信息
                    logging.warning(f"操作失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}")

                    # 如果不是最后一次尝试，等待后重试
                    if attempt < max_retries - 1:
                        logging.info(f"将在 {retry_interval} 秒后重试...")
                        time.sleep(retry_interval)

            # 所有重试都失败时抛出异常或返回失败
            logging.error(f"达到最大重试次数 {max_retries}，操作最终失败")
            raise e

        return wrapper
    return decorator


def update_generic_data(
    fetch_func,
    save_dir,
    mode='insert',
    date_col='trade_date',
    code_col='ts_code',
    extra_kwargs=None
):
    """
    通用数据更新函数

    参数:
        fetch_func: 获取数据的函数，接收start_date, end_date等参数
        save_dir: 保存目录
        mode: 'insert'增量更新，'update'全量更新
        date_col: 日期列名
        code_col: 代码列名
        extra_kwargs: 传递给fetch_func的额外参数字典
    """
    import datetime as dt
    from fun import get_parquet_dir_schema

    save_dir = os.path.join(DATA_ROOT_DIR, save_dir)

    # 创建目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 获取已有日期
    existing_dates = []
    if os.path.exists(save_dir):
        for item in os.listdir(save_dir):
            if item.startswith("trading_date="):
                date_str = item.split("=")[1]
                existing_dates.append(date_str)

    # 确定日期范围
    today = dt.date.today().strftime('%Y%m%d')

    if mode == 'insert' and existing_dates:
        start_date = max(existing_dates).replace('-', '')
        # 从已有日期的下一天开始
        start_date_dt = dt.datetime.strptime(start_date, '%Y%m%d').date()
        start_date_dt = start_date_dt + dt.timedelta(days=1)
        start_date = start_date_dt.strftime('%Y%m%d')
    else:
        # 全量更新，从2020年开始
        start_date = '20200101'

    end_date = today

    if start_date > end_date:
        logging.info(f"无需更新，已有最新数据至 {max(existing_dates) if existing_dates else '无'}")
        return None

    logging.info(f"开始获取数据: {start_date} 至 {end_date}")

    # 获取数据
    kwargs = extra_kwargs or {}
    df = fetch_func(start_date=start_date, end_date=end_date, **kwargs)

    if df is None or df.empty:
        logging.info("没有获取到数据")
        return None

    # 数据清洗
    df = clean_stocks_data(df)

    # 转换为Polars
    df_pl = pl.from_pandas(df)

    # 获取已有schema并转换
    existing_schema = get_parquet_dir_schema(save_dir)
    if existing_schema:
        convert_exprs = []
        for col, dtype in existing_schema.items():
            if col in df_pl.columns:
                convert_exprs.append(pl.col(col).cast(dtype).alias(col))
        if convert_exprs:
            df_pl = df_pl.select(convert_exprs)

        # 补齐缺失列
        missing_cols = [col for col in existing_schema.keys() if col not in df_pl.columns]
        if missing_cols:
            logging.info(f"警告: 新数据缺少以下列，已自动添加空值列: {missing_cols}")
            for col in missing_cols:
                df_pl = df_pl.with_columns(pl.lit(None).cast(existing_schema[col]).alias(col))

    # 保存数据
    df_pl = df_pl.sort(['trading_date', 'code'])
    logging.info(f"准备保存数据，共 {df_pl.height} 条记录")
    df_pl.write_parquet(save_dir, partition_by=['trading_date'])

    return df_pl


def update_index_daily(start_date=None, end_date=None, save_dir='ts_index_daily', mode='insert'):
    """
    更新指数日线数据 (index_daily)
    """
    logging.info("=" * 50)
    logging.info("开始更新指数日线数据")

    @retry_with_timeout(max_retries=5, retry_interval=3, timeout=180)
    def get_index_list(market):
        """获取指数列表（带重试）"""
        return ts.index_basic(market=market)

    @retry_with_timeout(max_retries=5, retry_interval=3, timeout=180)
    def get_index_daily_batch(codes_str, start_date, end_date):
        """获取指数日线数据（带重试）"""
        return ts.index_daily(ts_code=codes_str, start_date=start_date, end_date=end_date)

    def fetch_data(start_date, end_date):
        # 先获取指数列表
        try:
            index_list = get_index_list(market='SSE')
        except Exception as e:
            logging.error(f"获取上交所指数列表失败: {e}")
            index_list = None

        if index_list is not None and not index_list.empty:
            try:
                index_list2 = get_index_list(market='SZSE')
                if index_list2 is not None and not index_list2.empty:
                    index_list = pd.concat([index_list, index_list2], ignore_index=True)
            except Exception as e:
                logging.warning(f"获取深交所指数列表失败，仅使用上交所数据: {e}")

        if index_list is None or index_list.empty:
            logging.info("未获取到指数列表")
            return None

        time.sleep(1)

        # 获取指数代码列表
        if 'ts_code' in index_list.columns:
            ts_codes = index_list['ts_code'].tolist()
        elif 'code' in index_list.columns:
            ts_codes = index_list['code'].tolist()
        else:
            logging.info("指数列表中未找到代码列")
            return None

        # 按指数分批获取数据
        all_data = []
        batch_size = 50

        for i in range(0, len(ts_codes), batch_size):
            batch_codes = ts_codes[i:i + batch_size]
            codes_str = ','.join(batch_codes)

            try:
                df = get_index_daily_batch(codes_str, start_date, end_date)
                if df is not None and not df.empty:
                    all_data.append(df)
                    logging.info(f"已获取指数批次 {i // batch_size + 1}/{(len(ts_codes) + batch_size - 1) // batch_size} 数据")
            except Exception as e:
                logging.error(f"获取指数批次 {i // batch_size + 1} 失败: {e}")
                # 继续处理下一批，不跳过整个获取

            time.sleep(1)  # 请求间隔

        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return None

    return update_generic_data(
        fetch_func=fetch_data,
        save_dir=save_dir,
        mode=mode
    )


def update_index_dailybasic(start_date=None, end_date=None, save_dir='ts_index_dailybasic', mode='insert'):
    """
    更新指数每日基本指标 (index_dailybasic)
    """
    logging.info("=" * 50)
    logging.info("开始更新指数每日基本指标")

    @retry_with_timeout(max_retries=5, retry_interval=3, timeout=180)
    def get_index_dailybasic(date_str):
        """获取指数基本指标（带重试）"""
        return ts.index_dailybasic(trade_date=date_str)

    def fetch_data(start_date, end_date):
        all_data = []
        from datetime import datetime, timedelta

        start_dt = datetime.strptime(start_date, '%Y%m%d')
        end_dt = datetime.strptime(end_date, '%Y%m%d')

        current_dt = start_dt
        while current_dt <= end_dt:
            date_str = current_dt.strftime('%Y%m%d')
            try:
                df = get_index_dailybasic(date_str)
                if df is not None and not df.empty:
                    all_data.append(df)
                    logging.info(f"已获取 {date_str} 指数基本指标")
            except Exception as e:
                logging.error(f"获取 {date_str} 数据失败: {e}")
                # 继续处理下一天，不跳过整个获取
            current_dt += timedelta(days=1)
            time.sleep(0.5)  # 请求间隔

        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return None

    return update_generic_data(
        fetch_func=fetch_data,
        save_dir=save_dir,
        mode=mode
    )


def update_etf_list(save_dir='ts_etf_list'):
    """
    更新ETF基础信息列表 (fund_etf)
    """
    logging.info("=" * 50)
    logging.info("开始更新ETF基础信息列表")

    @retry_with_timeout(max_retries=5, retry_interval=3, timeout=180)
    def get_fund_etf(market):
        """获取ETF列表（带重试）"""
        return ts.fund_etf(market=market)

    save_dir = os.path.join(DATA_ROOT_DIR, save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    try:
        df = None
        try:
            df = get_fund_etf(market='SSE')
        except Exception as e:
            logging.warning(f"获取上交所ETF列表失败: {e}")

        df2 = None
        try:
            df2 = get_fund_etf(market='SZSE')
        except Exception as e:
            logging.warning(f"获取深交所ETF列表失败: {e}")

        if df is None and df2 is None:
            logging.info("未获取到ETF列表")
            return None

        all_df = []
        if df is not None and not df.empty:
            all_df.append(df)
        if df2 is not None and not df2.empty:
            all_df.append(df2)

        if not all_df:
            return None

        time.sleep(1)

        df = pd.concat(all_df, ignore_index=True)
        df = clean_stocks_data(df)

        # 保存为单个文件（列表数据不分区）
        save_path = os.path.join(save_dir, 'etf_list.parquet')
        df.to_parquet(save_path, index=False)
        logging.info(f"ETF列表保存完成，共 {len(df)} 条记录")

        return df
    except Exception as e:
        logging.error(f"获取ETF列表失败: {e}")
        return None


def update_etf_daily(start_date=None, end_date=None, save_dir='ts_etf_daily', mode='insert'):
    """
    更新ETF日线数据 (fund_daily)
    """
    logging.info("=" * 50)
    logging.info("开始更新ETF日线数据")

    @retry_with_timeout(max_retries=5, retry_interval=3, timeout=180)
    def get_fund_daily(date_str):
        """获取基金日线数据（带重试）"""
        return ts.fund_daily(trade_date=date_str)

    def fetch_data(start_date, end_date):
        all_data = []
        from datetime import datetime, timedelta

        start_dt = datetime.strptime(start_date, '%Y%m%d')
        end_dt = datetime.strptime(end_date, '%Y%m%d')

        current_dt = start_dt
        while current_dt <= end_dt:
            date_str = current_dt.strftime('%Y%m%d')
            try:
                df = get_fund_daily(date_str)
                if df is not None and not df.empty:
                    # 只保留ETF（过滤掉其他基金）
                    # ETF代码通常以51开头（SH）或15开头（SZ）
                    if 'ts_code' in df.columns:
                        df = df[df['ts_code'].str.startswith(('51', '15', '56', '58', '159'))]
                    all_data.append(df)
                    logging.info(f"已获取 {date_str} ETF数据")
            except Exception as e:
                logging.error(f"获取 {date_str} 数据失败: {e}")
                # 继续处理下一天，不跳过整个获取
            current_dt += timedelta(days=1)
            time.sleep(0.5)  # 请求间隔

        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return None

    return update_generic_data(
        fetch_func=fetch_data,
        save_dir=save_dir,
        mode=mode
    )


def update_swl_index_list(save_dir='ts_swl_index_list'):
    """
    更新申万行业指数列表 (index_member, market='SW')
    """
    logging.info("=" * 50)
    logging.info("开始更新申万行业指数列表")

    @retry_with_timeout(max_retries=5, retry_interval=3, timeout=180)
    def get_swl_index_list():
        """获取申万指数列表（带重试）"""
        return ts.index_basic(market='SW')

    save_dir = os.path.join(DATA_ROOT_DIR, save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    try:
        # 获取申万一级行业指数
        df = get_swl_index_list()

        if df is None or df.empty:
            logging.info("未获取到申万行业指数列表")
            return None

        time.sleep(1)
        df = clean_stocks_data(df)

        # 保存为单个文件
        save_path = os.path.join(save_dir, 'swl_index_list.parquet')
        df.to_parquet(save_path, index=False)
        logging.info(f"申万行业指数列表保存完成，共 {len(df)} 条记录")

        return df
    except Exception as e:
        logging.error(f"获取申万行业指数列表失败: {e}")
        return None


def update_swl_daily(start_date=None, end_date=None, save_dir='ts_swl_daily', mode='insert'):
    """
    更新申万行业日线数据 (sw_daily)
    """
    logging.info("=" * 50)
    logging.info("开始更新申万行业日线数据")

    @retry_with_timeout(max_retries=5, retry_interval=3, timeout=180)
    def get_sw_daily(date_str):
        """获取申万行业日线数据（带重试）"""
        return ts.sw_daily(trade_date=date_str)

    def fetch_data(start_date, end_date):
        all_data = []
        from datetime import datetime, timedelta

        start_dt = datetime.strptime(start_date, '%Y%m%d')
        end_dt = datetime.strptime(end_date, '%Y%m%d')

        current_dt = start_dt
        while current_dt <= end_dt:
            date_str = current_dt.strftime('%Y%m%d')
            try:
                df = get_sw_daily(date_str)
                if df is not None and not df.empty:
                    all_data.append(df)
                    logging.info(f"已获取 {date_str} 申万行业数据")
            except Exception as e:
                logging.error(f"获取 {date_str} 申万行业数据失败: {e}")
                # 继续处理下一天，不跳过整个获取
            current_dt += timedelta(days=1)
            time.sleep(0.5)  # 请求间隔

        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return None

    return update_generic_data(
        fetch_func=fetch_data,
        save_dir=save_dir,
        mode=mode
    )


if __name__ == "__main__":
    logging.info("开始更新指数、ETF、板块数据")

    # 更新指数数据
    update_index_daily(mode='insert')
    update_index_dailybasic(mode='insert')

    # 更新ETF数据
    update_etf_list()
    update_etf_daily(mode='insert')

    # 更新申万行业数据
    update_swl_index_list()
    update_swl_daily(mode='insert')

    logging.info("所有数据更新完成！")
