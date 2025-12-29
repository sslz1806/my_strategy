'''
股票api,功能:
init: 相关的token
'''
import numpy as np
from scipy.stats import rankdata
import pandas as pd
import akshare as ak
from gm.api import *
import datetime
import os
import logging
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed,TimeoutError
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import functools
import tinyshare as tns
from functools import wraps,partial
import threading
from threading import Timer
from mapping import *
import sys
import polars as pl

#%% 配置装饰器
def timer(func):
    """计时装饰器：测量函数的执行时间"""
    @functools.wraps(func)  # 保留原函数的元数据
    def wrapper(*args, **kwargs):
        start_time = time.time()  # 记录开始时间
        result = func(*args, **kwargs)  # 调用原函数
        end_time = time.time()  # 记录结束时间
        elapsed_time = end_time - start_time  # 计算耗时
        print(f"函数 {func.__name__} 执行耗时: {elapsed_time:.2f} 秒")
        return result
    return wrapper
"""
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler()
    ],
    encoding='utf-8'
)
"""
now_str = datetime.datetime.now().strftime('%Y-%m-%d')

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
# 重试装饰器,date参数为函数的第二个
def retry_with_timeout(max_retries=3, timeout=120):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            date = args[0] if args else kwargs.get('trade_date')
            
            for attempt in range(max_retries):
                if attempt !=0:
                    logging.info(f"开始第 {attempt + 1} 次尝试调取 {date} 的数据")
                
                try:
                    with timeout_context(timeout) as timer:
                        return func(*args, **kwargs)    
                            
                except TimeoutException:
                    logging.warning(f"{date}: 第 {attempt + 1} 次尝试超时")
                    if attempt == max_retries - 1:
                        error_msg = f"达到最大重试次数 ({max_retries})"
                        logging.error(f"{date}: {error_msg}")
                        raise TimeoutError(error_msg)
                    continue
                
                    
                except Exception as e:
                    error_msg = f"执行出错: {str(e)}"
                    logging.error(f"{date}: {error_msg}")
                    raise
                    
            return None  # 不应该到达这里
            
        return wrapper
    return decorator

import logging
from io import StringIO
from contextlib import contextmanager
#上下文管理器：屏蔽正常输出和低级别日志，保留错误信息
@contextmanager
def suppress_non_error_output():
    """
    上下文管理器：屏蔽正常输出和低级别日志，保留错误信息
    """
    # 1. 保存原始stdout和日志配置
    original_stdout = sys.stdout
    original_log_level = logging.getLogger().getEffectiveLevel()
    
    try:
        # 2. 重定向stdout到空缓冲区（屏蔽正常print输出）
        sys.stdout = StringIO()
        
        # 3. 调整日志级别：只保留ERROR及以上（屏蔽INFO/DEBUG等）
        logging.getLogger().setLevel(logging.ERROR)
        
        yield  # 执行被包裹的代码块
        
    finally:
        # 4. 恢复原始配置
        sys.stdout = original_stdout
        logging.getLogger().setLevel(original_log_level)


#%% 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('function_output.log', encoding='utf-8'),  # 日志写入文件
        logging.StreamHandler(sys.stdout)           # 同时输出到控制台
    ]
)
# 日志转换装饰器
def print_to_log(func):
    """将函数中的print输出重定向到日志的装饰器（支持多线程）"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 使用线程本地存储
        thread_local = threading.local()
        
        # 保存原始的stdout
        original_stdout = sys.stdout
        
        # 创建一个StringIO对象来捕获print输出
        captured_output = StringIO()
        sys.stdout = captured_output
        
        try:
            # 执行函数
            result = func(*args, **kwargs)
            
            # 获取捕获的输出
            output = captured_output.getvalue().strip()
            
            # 如果有输出，记录到日志
            if output:
                # 按行分割输出，每行单独记录
                for line in output.split('\n'):
                    if line:  # 跳过空行
                        logging.info(f"[{func.__name__}] {line}")
                    
            return result
        except Exception as e:
            logging.error(f"[{func.__name__}] Error: {str(e)}")
            raise
        finally:
            # 恢复原始的stdout
            sys.stdout = original_stdout
            captured_output.close()
    
    return wrapper

import builtins

def enable_print_logging(level=logging.INFO):
    original_print = builtins.print
    # 使用RLock允许同一线程重入
    lock = threading.RLock()

    def new_print(*args, **kwargs):
        # 先执行print（无锁，允许并行输出到控制台）
        original_print(*args, **kwargs)
        
        # 仅对日志写入加锁（减少锁范围）
        try:
            msg = " ".join(str(a) for a in args)
        except Exception:
            msg = " <print对象无法序列化> "
        with lock:
            logging.log(level, msg)
    
    builtins.print = new_print
    return original_print

# 启用（放在 logging.basicConfig 后）
#enable_print_logging()
#%% 因子函数
def calculate_pct_chg(df, n_days=1, price_col='close', prev_col='pre_close'):
    """
    计算 N 日涨幅（默认 1 日）
    
    参数:
    - df: 输入 DataFrame或Series（已按 symbol 分组，除非n_days=1可以不用）
    - n_days: 间隔天数，默认为 1
    - price_col: 当前价格列名
    - prev_col: 前一日价格列名（若 n_days=1，直接使用此列；否则通过 shift 计算）
    """
    if n_days == 1:
        # 1 日涨幅直接使用 pre_close 列
        return (df[price_col] - df[prev_col]) / df[prev_col] * 100
    else:
        if isinstance(df, pd.Series):
            # 如果是 Series，直接使用 shift(n_days)
            prev_price = df.shift(n_days)
            return (df - prev_price) / prev_price * 100
        # 如果是 DataFrame，使用指定的列进行计算
        else:
            # N 日涨幅通过 shift(n_days) 获取 N 天前的价格
            prev_price = df[price_col].shift(n_days) 
            return (df[price_col] - prev_price) / prev_price * 100
        
def ts_sum(df, window=10):
    """
    包装函数，用于估计滚动求和。
    :param df: 一个pandas DataFrame。
    :param window: 滚动窗口大小。
    :return: 一个pandas DataFrame，包含过去 'window' 天的时间序列求和结果。
    """
    return df.rolling(window).sum()

def sma(df, window=10): #窗口10的移动平均线
    """
    包装函数，用于估计简单移动平均线（SMA）。
    :param df: 一个pandas DataFrame。
    :param window: 滚动窗口大小。
    :return: 一个pandas DataFrame，包含过去 'window' 天的时间序列平均值。
    """
    return df.rolling(window).mean()

def stddev(df, window=10):
    ''' df序列数据中滚动10天的标准差'''
    return df.rolling(window).std()

def correlation(x, y, window=10):
    ''' x和y 对应同名列的 滚动10天序列相关性,输出一个df,列名为x和y的列名,值为滚动的相关系数'''
        # 重设索引确保对齐
    x = x.reset_index(drop=True)
    y = y.reset_index(drop=True)
    return x.rolling(window).corr(y).fillna(0).replace([np.inf, -np.inf], 0)

def covariance(x, y, window=10): 
    ''' x和y 对应列的 滚动10天协方差'''
    return x.rolling(window).cov(y)

def rolling_rank(na):
    """
    计算一个序列（如DataFrame或Series的一列）中最后一个值的排名（等级）。
    参数说明：
    ----------
    na : array-like
        输入的数值序列，可以是list、numpy数组、pandas Series等。
        通常用于rolling窗口内的序列。
    返回值：
    -------
    rank : int
        最后一个值在整个序列中的排名（从1开始，数值越大排名越高，具体取决于method参数）。
    详细说明：
    ----------
    本函数常用于滑动窗口（rolling window）操作中，计算窗口内最后一个元素在当前窗口内的排名。
    例如，若na为[3, 1, 4, 2]，则最后一个值2在该序列中的排名为2（按从小到大排序，1为第1，2为第2，3为第3，4为第4）。

    rankdata函数用法说明：
    ---------------------
    rankdata(na, method='min')
        - na: 输入的序列
        - method: 排名方法
            'min' 表示遇到相同数值时，分配最小的排名。例如[1, 2, 2, 3]的排名为[1, 2, 2, 4]
            其他常用method还有'average', 'max', 'dense', 'ordinal'等，具体可查scipy文档。
        - 返回值: 一个与输入序列等长的排名数组，排名从1开始。

    例子：
    -----
    >>> rolling_rank([3, 1, 4, 2])
    2
    >>> rolling_rank([1, 2, 2, 3])
    4

    注意事项：
    ---------
    - 如果序列中有重复值，method='min'会给重复值分配相同的最小排名。
    - 返回的是最后一个元素的排名。
    """
    return rankdata(na, method='min')[-1]

def ts_rank(df, window=10): 
    ''' 最新一天在滚动10天内的排名，返回滚动的rank值(值越高排名越高)'''
    return df.rolling(window).apply(rolling_rank)

def rolling_prod(na): 
    ''' 所有数乘积'''
    return np.prod(na)

def product(df, window=10): 
    ''' 滚动窗口10天的乘积'''
    return df.rolling(window).apply(rolling_prod)

def ts_min(df, window=10): 
    ''' 滚动窗口10天的最小值'''
    return df.rolling(window).min()

def ts_max(df, window=10):
    return df.rolling(window).max()

def delta(df, period=1): 
    ''' df序列中差分函数，后-前'''
    return df.diff(period)

def row_rank(df): 
    ''' 不同产品按照值来进行百分比排名，小的排名高,即 1  2 3 4-> 1/4 2/4 3/4 4/4(数值信息转化成等级信息)'''
    return df.rank(axis=1, method='min', pct=True)
    # return df.rank(pct=True)

def scale(df, k=1): 
    ''' 对df缩放并归一'''
    return df.mul(k).div(np.abs(df).sum()) # 1.对df每个值乘k 2.对df每个值除以(sum(df))

def ts_argmax(df, window=10): 
    """找出 DataFrame 中每个滚动窗口内最大值所在的时间点的索引,如1 2 3 4,越近越大"""
    return df.rolling(window).apply(np.argmax) + 1 

def ts_argmin(df, window=10): 
    ''' 找出 DataFrame 中每个滚动窗口内最小值所在的时间点，value为那个时间点'''
    return df.rolling(window).apply(np.argmin) + 1

def decay_linear(df, period=10): 
    ''' 将df中的序列按照从小到大线性权重加权平均'''
    weights = np.array(range(1, period + 1))
    sum_weights = np.sum(weights)
    return df.rolling(period).apply(lambda x: np.sum(weights * x) / sum_weights)
#%% stock_api类
# endregion
def get_all_stocks(date):
    """
    剔除ST股、科创板/创业板股票，上市不足100日、退市股、B股
    """
        # 日期类型自动转换
    if isinstance(date, str):
        current_date = datetime.datetime.strptime(date, '%Y-%m-%d').date()
    elif isinstance(date, datetime):
        current_date = date.date()
    elif isinstance(date, pd.Timestamp):
        current_date = date.date()
    else:
        current_date = date
    previous_trading_date = get_previous_n_trading_dates(exchange='SHSE', date=current_date.strftime('%Y-%m-%d'), n=1)[0]  # 获取前一交易日
    trade_date=get_trading_dates_by_year(exchange='SHSE',start_year=int(current_date.year),end_year=int(current_date.year))
    if current_date.strftime('%Y-%m-%d') not in trade_date['trade_date'].values:
        current_date=datetime.datetime.strptime(previous_trading_date, '%Y-%m-%d').date()

    # ---------------------- 1. 获取基础股票池（A股，排除科创板/创业板） ----------------------
    symbols = get_symbol_infos(
        sec_type1=1010,         # 证券大类：股票
        sec_type2=101001,       # 证券细类：A股
        exchanges=['SHSE', 'SZSE'],  # 交易所：沪深主板
        df=True
    )
    
    # 排除科创板（代码以SHSE.688开头）和创业板（代码以SZSE.300开头）
    valid_symbols = symbols[symbols['board']==10100101]['symbol'].tolist() #主板
    
    # ---------------------- 2. 剔除ST股、退市股、上市不足100日 ----------------------
    # 获取标的详细信息（含ST状态、上市日期、退市日期）
    symbol_details = get_symbols(
        sec_type1=1010,
        symbols=valid_symbols,
        trade_date=current_date.strftime('%Y-%m-%d'),
        df=True
    )
      # 计算上市天数
    symbol_details['listed_days'] = pd.to_timedelta((current_date - symbol_details['listed_date'].dt.date)).dt.days
    # 将delisted_date转换为date类型进行比较
    symbol_details['delisted_date'] = symbol_details['delisted_date'].dt.date
    # 过滤条件：非ST、未退市（delisted_date>当前日期）、上市>=100日
    filtered_symbols = symbol_details[
        (~symbol_details['is_st']) & 
        (symbol_details['delisted_date'] > current_date) & 
        (symbol_details['listed_days'] >= 100)
    ]['symbol'].tolist()

    # ---------------------- 3. 流通市值过大或者国小 ----------------------
    mktvalue_data = stk_get_daily_mktvalue_pt(
        symbols=filtered_symbols,
        fields='a_mv',
        trade_date=current_date.strftime('%Y-%m-%d'),
        df=True
    )
        # 过滤条件：市值25亿<=tot_mv<=1000亿
    final_symbols =  mktvalue_data [
        (mktvalue_data ['a_mv'] >= 20e8) & 
        (mktvalue_data ['a_mv'] <= 1000e8)
    ]['symbol'].tolist()
    return final_symbols

class stock_api:
    def __init__(self,config = 
        {'token':'f60e5c28159d9dc4e3d51de7dd16d5e132f70841', #f60e5c28159d9dc4e3d51de7dd16d5e132f70841
        'mins_token':'fbdsJ45z9Nodp7FbUgDEsm1Oi8boH7Wuiqn7cQJnRAvs5bSwuB4e0iOBbe16ef40',
        'ts_token':'YzAEH11Yc7jZCHjeJa63fnbpSt3k9Je3GvWn0390oiBKO95bVJjP7u5L34e2ff6b'
        }
        ):
        set_token(config['token'])
        self.config = config
        self.m_ts =tns.pro_api(config["mins_token"]) if "mins_token" in config else None
        self.ts =tns.pro_api(config["ts_token"]) if "ts_token" in config else None

    def convert_stock_code(self, code, to_format='gm'):
        """
        转换股票代码格式
        
        参数:
        - code: 股票代码，支持纯6位数代码、SZSE/SHSE前缀格式、后缀.SH/.SZ格式
        - to_format: 目标格式，可选值:
        - 'gm': 掘金格式 (SZSE.000001, SHSE.600001)
        - 'suffix': 后缀格式 (000001.SZ, 600001.SH)
        - 'pure': 纯数字格式 (000001, 600001)
        
        返回:
        - 转换后的股票代码字符串
        """
        # 如果是列表，递归处理每个元素
        if isinstance(code, list) or isinstance(code, pd.Series):
            return [self.convert_stock_code(c, to_format) for c in code]
        
        # 处理空值
        if not code:
            return None
        

        # 统一转为字符串
        code = str(code).upper().strip()
        
        # 提取纯数字代码部分
        pure_code = ''
        
        # 情况1: SZSE.000001 或 SHSE.600001 格式
        if 'SZSE.' in code or 'SHSE.' in code:
            pure_code = code.split('.')[-1]
            market = 'SZ' if 'SZSE.' in code else 'SH'
        
        # 情况2: 000001.SZ 或 600001.SH 格式
        elif '.SZ' in code or '.SH' in code:
            pure_code = code.split('.')[0]
            market = code.split('.')[-1]

        elif len(code)==8:
           # 深市: 0开头或3开头的6位数
            if code.startswith(('SZ')):
                market = 'SZ'
                pure_code=code[2:]
            # 沪市: 6或9开头的6位数
            elif code.startswith(('SH')):
                market = 'SH'
                pure_code=code[2:]
            else:
                return None  # 无法识别的代码
        
        # 情况3: 纯数字代码，需要判断市场
        elif code.isdigit() and len(code) == 6:
            pure_code = code
            # 深市: 0开头或3开头的6位数
            if code.startswith(('0', '3')):
                market = 'SZ'
            # 沪市: 6或9开头的6位数
            elif code.startswith(('6', '9')):
                market = 'SH'
            else:
                return None  # 无法识别的代码
        
        # 无法识别的格式
        else:
            return None
        
        # 转换为目标格式
        if to_format.lower() == 'gm':
            # 掘金格式
            return f"{'SZSE' if market == 'SZ' else 'SHSE'}.{pure_code}"
        
        elif to_format.lower() == 'suffix':
            # 后缀格式
            return f"{pure_code}.{market}"
        
        elif to_format.lower() == 'pure':
            # 纯数字格式
            return pure_code
        
        else:
            raise ValueError(f"不支持的目标格式: {to_format}，支持的格式有 'gm', 'suffix', 'pure'")
        
    def get_stock_name(self, symbol, default_name="未知"):
        """
        根据股票代码获取股票名称
        
        参数:
        - symbol: 股票代码 或者 股票代码列表,series
        - default_name: 未找到时返回的默认名称
        
        返回:
        - 股票名称
        """
        try:
            if isinstance(symbol, list) or isinstance(symbol, pd.Series):
                # 如果是列表或Series，递归处理每个元素
                return [self.get_stock_name(s, default_name) for s in symbol]
            # 确保代码格式统一 (如果需要)
            if hasattr(self, 'convert_stock_code'):
                symbol = self.convert_stock_code(symbol, 'gm')
            
            # 使用掘金API的get_history_symbol获取股票名称
            symbol_info = get_instruments(symbols=[symbol], df=True)
            
            if not symbol_info.empty and 'sec_name' in symbol_info.columns:
                return symbol_info['sec_name'].values[0]
            else:
                # 如果上面方法失败，尝试使用get_symbols获取
                symbol_info = get_symbols(
                    symbols=[symbol],
                    df=True
                )
                if not symbol_info.empty and 'sec_name' in symbol_info.columns:
                    return symbol_info['sec_name'].values[0]
                    
            return default_name
            
        except Exception as e:
            logging.error(f"获取股票名称出错: {e}")
            return default_name
    
    def get_all_date_data_with_future(self, start_time, end_time, list_assets, data_path='data_bfq',fields = ['date','symbol','open','high','low','close','volume','amount','pre_close','pct_chg','pct_chg_3d','pct_chg_5d','pct_chg_10d','pct_chg_20d']):
        """从本地获取对应的股票数据并加上未来涨幅，按照股票代码拼接起来，并打印读取成功和失败的信息"""
        list_all = []
        failed_list = []
        for c in list_assets:
            # 读取个股数据
            file_path = f'{data_path}/{c}.csv'
            if not os.path.exists(file_path):
                logging.info(f"[读取失败] 文件不存在: {file_path}")
                failed_list.append(c)
                continue
            try:
                # 读取股票数据
                df = pd.read_csv(file_path)
                # 加上未来涨幅
                for n in [1,3,5,10]:
                    if n ==1:
                        df[f'future_{n}d_chg'] = df[f'pct_chg'].shift(-n)
                    else:
                        df[f'future_{n}d_chg'] = df[f'pct_chg_{n}d'].shift(-n)
                    
                filtered_df = df[(df['date'] >= start_time) & (df['date'] <= end_time)]
                if filtered_df.empty:
                    logging.info(f"[读取失败] {c} 在指定日期区间无数据")
                    failed_list.append(c)
                    continue
                list_all.append(filtered_df)
            except Exception as e:
                logging.info(f"[读取失败] {c}，错误信息: {e}")
                failed_list.append(c)
                continue

        if not list_all:
            logging.info("没有任何股票数据被成功读取！")
            return pd.DataFrame()  # 返回空DataFrame

        df_all = pd.concat(list_all)
        df_all = df_all.reset_index()
        
        logging.info(f"全部读取完成，成功: {len(list_all)} 只，失败: {len(failed_list)} 只")
        if failed_list:
            logging.info("读取失败的股票代码：", failed_list)
        # 加上未来涨幅
        for n in [1,3,5,10]:
            fields += [f'future_{n}d_chg']
        df_all=df_all[fields]
        return df_all

    def display_stock_info(self):
        stock_data = self.get_stock_data()
        print(stock_data)

    def get_top_n_pct_data(self,start_time=None,end_time='2025-08-01',pre_n=5,stocks_data = None, list_assets = None, top_n=5, data_path = 'data_qfq',need_info=True):
        """
        获取时间区间内涨幅前n的股票数据
        可以传入股票数据,也可传入股票列表,如果都没有传入,则从本地读取所有股票数据
        参数:
        - start_time: 开始时间,如果为None则获取pre_n天前的交易日
        - end_time: 结束时间,默认'2025-08-01'
        - pre_n: n天前的交易日,默认5,如果传入start_time则忽略此参数
        - stocks_data: 股票数据,如果为None则利用list_assets从本地读取
        - list_assets: 股票代码列表,如果还是为None则从本地读取所有股票数据
        - top_n: 获取涨幅前n的股票,默认5
        - data_path: 数据文件夹路径,默认'data_qfq'
        - need_info: 是否需要返回股票名称等信息,默认True(不仅返回数据,还返回股票名称等信息,返回两个df结果)
        返回: 涨幅前n的股票数据,以及涨幅前n的股票代码和总涨幅 stocks_data,stocks_pct_chg
        """
        if start_time is None:
            # 获取n天前交易日
            trade_day = get_previous_n_trading_dates(exchange='SHSE', date=end_time, n=pre_n)
            start_time = trade_day[0]  # 获取n天前的交易日作为开始时间
        # 如果没有传入股票数据，则从本地读取
        if stocks_data is None:
            if list_assets is None:
                list_assets = self.get_stocks_list(data_path)
            stocks_data = self.get_all_date_data(start_time, end_time, list_assets, data_path)
        # 计算每只股票在时间区间内的总涨幅
        pct_chg_summary = stocks_data.groupby('symbol')['pct_chg'].sum().reset_index()
        # 按照总涨幅排序，取前n名
        top_n_stocks = pct_chg_summary.nlargest(top_n, 'pct_chg').reset_index()
        top_n_data = stocks_data[stocks_data['symbol'].isin(top_n_stocks['symbol'])]
        if need_info:
            top_n_stocks['name'] = self.get_stock_name(top_n_stocks['symbol'])
            logging.info(f"在{start_time}到{end_time}区间内,涨幅前{top_n}的股票：")
            print(top_n_stocks)
            # 获取这些股票的详细数据
            return top_n_data, top_n_stocks
        else:
            return top_n_data

            # 定义单个文件读取函数

    def get_top_n_data_with_field(self, start_time=None, end_time='2025-08-01', pre_n=5, stocks_data=None, list_assets=None, top_n=5, data_path='data_qfq', field='pct_chg', need_info=True):
        """
        获取时间区间内field字段前n的股票数据，并返回指定字段
        参数:
        - start_time: 开始时间,如果为None则获取pre_n天前的交易日
        - end_time: 结束时间,默认'2025-08-01'
        - pre_n: n天前的交易日,默认5,如果传入start_time则忽略此参数
        - stocks_data: 股票数据,如果为None则利用list_assets从本地读取
        - list_assets: 股票代码列表,如果还是为None则从本地读取所有股票数据
        - top_n: 获取涨幅前n的股票,默认5
        - data_path: 数据文件夹路径,默认'data_qfq'
        - fields: 需要返回的字段列表,如果为None则返回全部字段
        返回: 涨幅前n的股票数据,以及涨幅前n的股票代码和总涨幅 stocks_data,stocks_pct_chg
        """
        if start_time is None:
            # 获取n天前交易日
            trade_day = get_previous_n_trading_dates(exchange='SHSE', date=end_time, n=pre_n)
            start_time = trade_day[0]
        # 如果没有传入股票数据，则从本地读取
        if stocks_data is None:
            if list_assets is None:
                list_assets = self.get_stocks_list(data_path)
            stocks_data = self.get_all_date_data(start_time, end_time, list_assets, data_path, fields=[field])
        # 计算每天指定字段前n的股票数据
        if field not in stocks_data.columns:
            raise ValueError(f"指定的字段 '{field}' 不在股票数据中，请检查字段名是否正确。")
        top_n_data = stocks_data.groupby('date').apply(lambda x: x.nlargest(top_n, field)).reset_index(drop=True)
        top_n_data['name'] = top_n_data['symbol'].apply(self.get_stock_name)
        logging.info(f"在{start_time}到{end_time}区间内, 每日{field}前{top_n}的股票：")
        print(top_n_data[['date', 'symbol', 'name', field]])
        return top_n_data

    def read_stock_file(self,symbol,data_path='data_bfq', start_time=None, end_time=None, fields=None):
        file_path = f'{data_path}/{symbol}.csv'
        if not os.path.exists(file_path):
            return {'success': False, 'symbol': symbol, 'error': '文件不存在'}
        
        try:
            # 确定要读取的列
            if fields is not None:
                # 先读取文件头获取可用列
                available_fields = pd.read_csv(file_path, nrows=0).columns.tolist()
                # 筛选存在的列
                valid_fields = [f for f in fields if f in available_fields]
                if not valid_fields:
                    return {'success': False, 'symbol': symbol, 'error': '没有有效字段'}
                
                # 读取指定列并过滤日期
                df = pd.read_csv(file_path, encoding='utf-8', usecols=valid_fields)
            else:
                # 读取全部列
                df = pd.read_csv(file_path, encoding='utf-8')
            
            # 过滤日期 (如果date已经转为datetime，可以直接比较)
            filtered_df = df[(df['date'] >= start_time) & (df['date'] <= end_time)].reset_index(drop=True)
            
            if filtered_df.empty:
                return {'success': False, 'symbol': symbol, 'error': '指定日期区间无数据'}
            
            # 确保有symbol列
            if 'symbol' not in filtered_df.columns:
                filtered_df['symbol'] = symbol
                
            return {'success': True, 'data': filtered_df}
        
        except Exception as e:
            return {'success': False, 'symbol': symbol, 'error': str(e)}
    
    #%% 使用掘金api
    # 获取多个股票的历史基础数据（行情+基本信息）
    def get_history_symbols(self,symbols, 
                                start_date: str, 
                                end_date: str,
                                max_retries: int = 3):
        """
        用掘金api批量获取多个股票的历史数据并拼接成一个DataFrame
        
        参数:
            symbols: 股票代码列表
            start_date: 开始日期，格式如'2020-01-01'
            end_date: 结束日期，格式如'2023-01-01'
            max_retries: 获取失败时的最大重试次数
        
        返回:
            拼接后的DataFrame，包含所有股票的历史数据，增加'symbol'列标识股票代码
            如果所有获取都失败则返回None
        """
        all_data = []
        for symbol in symbols:
            retry_count = 0
            success = False
            while retry_count < max_retries and not success:
                try:
                    # 调用你的get_history_symbol函数获取单个股票数据
                    df1 = get_history_symbol(symbol=symbol,start_date=start_date, end_date=end_date,df=True)
                    df2 = history(symbol,frequency='1d',start_time=start_date, end_time=end_date,df=True)
                    df = pd.merge(df1,df2,left_on='trade_date',right_on='bob',suffixes=('', '_right'))
                    # 删除右侧DataFrame中与左侧重复的列
                    cols_to_drop = [col for col in df.columns if col.endswith('_right')]
                    df = df.drop(columns=cols_to_drop)
                    # 检查返回的数据是否有效
                    if df is not None and not df.empty:
                        # 添加股票代码列，方便后续区分不同股票
                        df['symbol'] = symbol
                        all_data.append(df)
                        success = True
                        #logging.info(f"成功获取 {symbol} 的数据")
                    else:
                        logging.info(f"未获取到 {symbol} 的数据")
                        success = True  # 没有数据也算处理完成，不再重试
                    
                except Exception as e:
                    retry_count += 1
                    logging.info(f"获取 {symbol} 数据失败（第{retry_count}次重试）: {str(e)}")
                    if retry_count >= max_retries:
                        logging.info(f"{symbol} 达到最大重试次数，跳过该股票")
        
        # 检查是否获取到数据
        if not all_data:
            logging.info("未获取到任何股票数据")
            return None
        
        # 拼接所有股票数据
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # 按日期和股票代码排序
        combined_df = combined_df.sort_values(by=['trade_date', 'symbol'])
        combined_df=combined_df.dropna(axis=1, how='all')
        # 将 'trade_date' 和 'symbol' 移到最前面，保持其他列的相对顺序不变
        cols = combined_df.columns.tolist()
        cols = ['trade_date', 'symbol'] + [col for col in cols if col not in ['trade_date', 'symbol']]
        combined_df = combined_df[cols]
        return combined_df.drop(columns=['position','exercise_price','conversion_price','sec_type1','sec_type2','board','exchange','sec_abbr','trade_n','option_margin_ratio1','option_margin_ratio2'])

    # 按股票分批获取基础数据
    def batch_get_history_symbols(self,stock_list,start_date, end_date, batch_size=200,max_retries=3, max_workers=5):
        """
        分批并多线程获取股票历史数据
        
        参数:
            stock_list: 股票代码列表
            batch_size: 每批处理的股票数量
            start_date: 开始日期，格式如'2020-01-01'
            end_date: 结束日期，格式如'2023-01-01'
            max_retries: 获取失败时的最大重试次数
            max_workers: 最大线程数，默认使用系统推荐值
        
        返回:
            所有股票合并后的历史数据DataFrame，如果全部失败则返回None
        """
        # 将股票列表分成多个批次
        batches = [stock_list[i:i+batch_size] for i in range(0, len(stock_list), batch_size)]
        logging.info(f"将{len(stock_list)}只股票分成{len(batches)}多线程批处理")
        
        all_results = []
        
        # 使用线程池执行任务
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有批次任务
            futures = {
                executor.submit(
                    self.get_history_symbols, 
                    batch, 
                    start_date, 
                    end_date, 
                    max_retries
                ): batch for batch in batches
            }
            
            # 处理完成的任务
            for future in as_completed(futures):
                batch = futures[future]
                try:
                    result = future.result()
                    if result is not None and not result.empty:
                        all_results.append(result)
                        logging.info(f"完成批次处理，股票代码: {batch[:3]}...(共{len(batch)}只)")
                    else:
                        logging.info(f"批次处理未返回有效数据，股票代码: {batch[:3]}...(共{len(batch)}只)")
                except Exception as e:
                    logging.info(f"批次处理出错，股票代码: {batch[:3]}...(共{len(batch)}只)，错误: {str(e)}")
        
        # 合并所有结果
        if not all_results:
            logging.info("所有批次处理均未返回有效数据")
            return None
        
        final_df = pd.concat(all_results, ignore_index=True)
        logging.info(f"所有批次处理完成，共合并{len(final_df)}条记录")
        
        # 可以根据需要再次排序
        if 'trade_date' in final_df.columns and 'symbol' in final_df.columns:
            final_df = final_df.sort_values(by=['trade_date', 'symbol'])
        final_df = clean_stocks_data(final_df)
        return final_df
    
    # 利用掘金api获取特定股票的1min分钟数据,并合成需要的n分钟数据
    def gm_get_minute_data(self,symbols,start_time='2025-10-01',end_time='2025-11-11',frequency='900s',n=15):
        """
        symbols: 股票代码列表
        start_time: 开始时间 '2023-01-01 09:30:00'
        end_time: 结束时间 '2023-01-31 15:00:00'
        frequency: 原始分钟数据频率，默认'60s'
        n: 目标分钟数据频率，如5表示5分钟
        """
        all_data = []
        for symbol in symbols:
            try:
                # 不包括end_time的date的数据
                df = history(symbol, frequency=frequency, start_time=start_time, end_time=end_time, df=True)
                if df is None or df.empty:
                    logging.info(f"{symbol} 无分钟数据，跳过")
                    continue
                # 处理需要复制的特殊时段（eob为11:30或15:00）
                df['symbol'] = symbol
                # 移除时区信息（转换为naive datetime）
                df['bob'] = df['bob'].dt.tz_localize(None)
                df['eob'] = df['eob'].dt.tz_localize(None)
                new_rows = []
                for _, row in df.iterrows():
                    # 保留原始行
                    new_rows.append(row)
                    
                    # 检查结束时间是否为11:30或15:00（忽略日期部分）
                    eob_time = row['eob'].time()
                    if eob_time in (pd.Timestamp('11:30:00').time(), pd.Timestamp('15:00:00').time()):
                        # 复制一行作为新K线
                        new_row = row.copy()
                        # 新K线的开始时间设为原结束时间（11:30:00或15:00:00）
                        new_row['bob'] = row['eob']
                        # 新K线的结束时间 = 开始时间（因为是1分钟周期逻辑）
                        new_row['eob'] = new_row['bob']
                        # 高开低收均设为原K线的收盘价
                        new_row[['open', 'high', 'low', 'close']] = row['close']
                        new_rows.append(new_row)
                # 转换为DataFrame并按时间排序
                processed_df = pd.DataFrame(new_rows).sort_values('bob')
                # 去重处理（防止极端情况的重复）
                processed_df = processed_df.drop_duplicates(subset=['bob', 'symbol'])
                # 移除临时的eob列
                processed_df = processed_df.drop(columns=['eob'])
                all_data.append(processed_df)

                #logging.info(f"已获取并合成 {symbol} 的{n}分钟数据")
            except Exception as e:
                logging.info(f"获取 {symbol} 分钟数据失败: {str(e)}")
        
        if not all_data:
            logging.info("未获取到任何股票的分钟数据")
            return None
        
        combined_df = pd.concat(all_data, ignore_index=True)

        # 列重命名
        combined_df.rename(columns={'bob': 'datetime','symbol':'code'}, inplace=True)
        combined_df['datetime'] = convert_date_format(combined_df['datetime'],to_format='datetime')
        return combined_df.drop(columns=['frequency','position'])
    
    # 批量多线程获取多个股票的n分钟数据,不包括end_time的date的数据,end_time的date需要加一天
    def gm_batch_get_minute_data(self,symbols,start_time='2025-10-01',end_time='2025-11-11',frequency='900s',n=15,batch_size=100, max_workers=15):
        """
        批量多线程获取多个股票的n分钟数据
        
        参数:
            symbols: 股票代码列表
            start_time: 开始时间 '2023-01-01 09:30:00'
            end_time: 结束时间 '2023-01-31 15:00:00'
            frequency: 原始分钟数据频率，默认'60s'
            n: 目标分钟数据频率，如5表示5分钟
            batch_size: 每批处理的股票数量
            max_workers: 最大线程数，默认使用系统推荐值
        
        返回:
            所有股票合并后的n分钟数据DataFrame，如果全部失败则返回None
        """
        # 将股票列表分成多个批次
        batches = [symbols[i:i+batch_size] for i in range(0, len(symbols), batch_size)]
        logging.info(f"将{len(symbols)}只股票分成{len(batches)}多线程批处理")
        
        all_results = []
        
        from tqdm import tqdm
        # 使用线程池执行任务，利用tqmd显示进度条
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有批次任务
            futures = {
                executor.submit(
                    self.gm_get_minute_data, 
                    batch, 
                    start_time, 
                    end_time, 
                    frequency,
                    n
                ): batch for batch in batches
            }
            
            # 处理完成的任务
            for future in as_completed(futures):
                batch = futures[future]
                try:
                    result = future.result()
                    if result is not None and not result.empty:
                        all_results.append(result)
                        #logging.info(f"完成批次处理，股票代码: {batch[:3]}...(共{len(batch)}只)")
                    else:
                        logging.info(f"批次处理未返回有效数据，股票代码: {batch[:3]}...(共{len(batch)}只)")
                except Exception as e:
                    logging.info(f"批次处理出错，股票代码: {batch[:3]}...(共{len(batch)}只)，错误: {str(e)}")
        
        # 合并所有结果
        if not all_results:
            logging.info("所有批次处理均未返回有效数据")
            return None
        
        final_df = pd.concat(all_results, ignore_index=True)
        logging.info(f"所有批次处理完成，共合并{len(final_df)}条记录")
        return final_df
              
    #%% 使用tushare_api
    # 获取所有股票的所有所有日期数据
    def ts_get_stocks_data(self, stock_list=None, start_date='2025-01-01', end_date='2025-10-31', max_workers=5):
        """
        stock_list: 列表，为None时为全市场股票
        max_workers: 最大并发线程数（根据接口限流调整，建议5-10）
        """
        # 转化成api需要的格式
        start_date = convert_date_format(start_date,to_format="%Y%m%d")
        end_date = convert_date_format(end_date,to_format="%Y%m%d")
        # 1. 处理交易日历（原有逻辑不变）
        trading_date = self.ts.trade_cal(start_date=start_date, end_date=end_date)
        trading_date = trading_date.sort_values('cal_date')
        trading_dates = trading_date[trading_date['is_open'] == 1]['cal_date'].tolist()
        if not trading_dates:
            logging.info("指定日期范围内没有交易日")
            return None

        # 2. 转换股票代码（全局执行一次，避免线程内重复转换）
        stock_list = self.convert_stock_code(stock_list, to_format='suffix')

        # 3. 定义单交易日处理函数（供线程调用）
        @retry_with_timeout(max_retries=3, timeout=30)
        def process_single_date(trade_date,stock_list=None):
            """单交易日数据获取+合并逻辑"""
            #检查trade_date格式
            # 数字格式
            if isinstance(trade_date, int):
                trade_date = str(trade_date)

            # 3.1 获取日线数据（主表）
            # with suppress_non_error_output():
            #     daily_data = self.ts.daily(ts_code=stock_list, trade_date=trade_date)
            # if daily_data.empty:
            #     logging.info(f"{trade_date} 无日线数据，跳过")
            #     raise ValueError(f"{trade_date} 无日线数据")
            # daily_data['trade_date'] = daily_data['trade_date'].astype(str)
            with suppress_non_error_output():
                daily_data = self.ts.bak_daily(trade_date=trade_date)
            if daily_data is None or (not isinstance(daily_data, type(None)) and daily_data.empty):
                logging.info(f"{trade_date} 无备份日线数据，跳过")
                raise ValueError(f"{trade_date} 无备份日线数据")
            daily_data['trade_date'] = daily_data['trade_date'].astype(str)
            daily_data = daily_data.drop(columns=['interval_3','interval_6'])

            # 3.2 获取涨停数据
            get_limit = partial(self.ts.stk_limit, trade_date=trade_date)
            with suppress_non_error_output():
                limit_data = retry_with_timeout(max_retries=3, timeout=30)(get_limit)()
            #limit_data = ts.stk_limit(ts_code=stock_list, trade_date=trade_date)
            if limit_data.empty and 'ts_code' not in limit_data.columns:
                raise ValueError(f"{trade_date} 无涨停数据")
            limit_data['trade_date'] = limit_data['trade_date'].astype(str)

            # 3.3 获取ST信息（带重试超时）
            get_st = partial(self.ts.stock_st, trade_date=trade_date)
            with suppress_non_error_output():
                st_data = retry_with_timeout(max_retries=3, timeout=30)(get_st)()
            #st_data =ts.stock_st(ts_code=stock_list, trade_date=trade_date)
            # 如果当日无ST数据，尝试获取前几天的数据（最多尝试10次）
            count=0
            while st_data.empty and count<=10:
                st_date = datetime.datetime.strptime(trade_date, '%Y%m%d') - datetime.timedelta(days=1)
                st_date_str = st_date.strftime('%Y%m%d')
                with suppress_non_error_output():
                    st_data = retry_with_timeout(max_retries=3, timeout=30)(partial(self.ts.stock_st, trade_date=st_date_str))()
                #st_data = daily_data[['ts_code','trade_date']]
                count += 1
                if not st_data.empty:
                    logging.info(f"{trade_date} 无ST数据，使用{st_date_str}的ST数据")
                    break
                time.sleep(1)  # 等待1秒再试
            if st_data.empty:
                raise ValueError(f"{trade_date} 无ST数据")
            st_data['trade_date'] = trade_date
            st_data['trade_date'] = st_data['trade_date'].astype(str)

            # 3.4 数据合并（原有逻辑不变）
            merged1 = pd.merge(daily_data, limit_data, on=['ts_code'], how='left', suffixes=('', '_right'))
            merged2 = pd.merge(merged1, st_data, on=['ts_code'], how='left', suffixes=('', '_st'))

            # 筛选最终列
            daily_cols = daily_data.columns.tolist()
            final_cols = [col for col in merged2.columns if col in daily_cols or '_right' not in col and '_st' not in col]
            merged_data = merged2[final_cols].copy()

            logging.info(f"已获取 {trade_date} 的股票数据")
            time.sleep(2)  # 避免请求过快
            return merged_data


        # 4. 多线程执行任务
        all_stock_data = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有交易日任务（返回Future对象列表）
            future_to_date = {
                executor.submit(process_single_date, trade_date, stock_list): trade_date
                for trade_date in trading_dates
            }

            # 遍历任务结果，收集有效数据
            for future in as_completed(future_to_date):
                trade_date = future_to_date[future]
                try:
                    result = future.result()  # 获取线程执行结果
                    if result is not None:
                        all_stock_data.append(result)
                except Exception as e:
                    logging.info(f"获取 {trade_date} 数据时出错: {str(e)}")
                    continue

        # 5. 合并所有数据（原有逻辑不变）
        if not all_stock_data:
            logging.info("未获取到任何股票数据")
            return None

        # 合并并清洗数据
        result_df = pd.concat(all_stock_data, ignore_index=True)
        result_df = clean_stocks_data(result_df)
        #result_df = result_df.sort_values(by=['trading_date', 'code']).reset_index(drop=True)

        return result_df

    #@print_to_log
    def ts_download_date_data(self, filename='ts_stcok_all_data', start_date='2025-01-01', end_date='2025-10-31', mode='append',df_type = 'polars',max_workers=5):
        """
        按日期分割的Parquet文件下载与更新函数（日期格式已通过clean_stocks_data统一为%Y-%m-%d）
        
        参数:
            filename: 存储数据的Parquet文件路径
            start_date: 开始日期（格式'YYYY-MM-DD'）
            end_date: 结束日期（格式'YYYY-MM-DD'）
            mode: 操作模式，'append'为追加模式，'update'为更新模式
            max_workers: 最大并发线程数
            df_type:pandas,polars(读取和存取parquet的格式),默认pandas。polars:按照polars读取exisiting_data和按照polars存取new_data
        """
        # 检查文件是否存在及已有日期范围
        from datetime import datetime, timedelta, date
        import polars as pl
        file_exists = os.path.exists(filename)
        existing_dates = []
        # 转化成date格式
        start_date = convert_date_format(start_date)
        end_date = convert_date_format(end_date)
        
        def add_one_day(date_value):
            if not isinstance(date_value, date):
                date_value = convert_date_format(date_value)
            return date_value + timedelta(days=1)
            # 辅助函数：日期减一天
        def subtract_one_day(date_value):
            if not isinstance(date_value, date):
                date_value = convert_date_format(date_value)
            return date_value- timedelta(days=1)
        
        if file_exists:
            try:
                # 读取已有数据并提取交易日（已为%Y-%m-%d格式）
                df = pd.read_parquet(filename)
                if 'trading_date' in df.columns:
                    existing_dates = sorted(df['trading_date'].unique().tolist())
            except Exception as e:
                logging.info(f"读取已有文件失败: {str(e)}")
                return None
        
        # 追加模式逻辑
        if mode == 'append':
            # 1.筛选日期范围
            if file_exists and existing_dates:
                last_existing_date = existing_dates[-1]  # 最后一个已有日期（已排序）
                first_existing_date = existing_dates[0]  # 第一个已有日期
                
                # 情况1：需要追加到已有数据之后（start <= 最后日期 且 end > 最后日期）
                if start_date <= last_existing_date and end_date > last_existing_date:
                    append_start = add_one_day(last_existing_date)
                    append_end = end_date
                    logging.info(f"检测到需追加历史数据之后的新日期: {append_start} 至 {append_end}")
                
                # 情况2：需要追加到已有数据之前（start < 第一个日期 且 end >= 第一个日期）
                elif start_date < first_existing_date and end_date >= first_existing_date:
                    append_start = start_date
                    append_end = subtract_one_day(first_existing_date)
                    logging.info(f"检测到需追加历史数据之前的日期: {append_start} 至 {append_end}")
                
                # 其他异常情况
                else:
                    # 日期完全在已有范围之前
                    if end_date < first_existing_date:
                        logging.info(f"错误: 日期范围({start_date}至{end_date})完全早于已有数据({first_existing_date}至{last_existing_date})，请检查是否需要更新模式")
                    # 日期完全在已有范围之后且不连续
                    elif start_date > last_existing_date:
                        logging.info(f"错误: 日期范围({start_date}至{end_date})与已有数据({first_existing_date}至{last_existing_date})存在断层，请先补充中间日期")
                    # 日期完全在已有范围内（重复）
                    elif start_date >= first_existing_date and end_date <= last_existing_date:
                        logging.info(f"错误: 日期范围({start_date}至{end_date})已存在于数据中，无需重复追加")
                    # 日期部分重叠（交叉）
                    else:
                        logging.info(f"错误: 日期范围({start_date}至{end_date})与已有数据({first_existing_date}至{last_existing_date})存在交叉，请使用更新模式")
                    return None
            else:
                # 文件不存在，直接使用输入的日期范围
                append_start = start_date
                append_end = end_date
                logging.info(f"文件不存在，首次获取数据: {append_start} 至 {append_end}")
            
            # 2.调用数据获取函数（全量获取，stock_list=None）
            logging.info(f"开始追加数据: {append_start} 至 {append_end}")
            new_data = self.ts_get_stocks_data(
                stock_list=None,
                start_date=append_start,
                end_date=append_end,
                max_workers=max_workers
            )
            
            if new_data.empty:
                logging.info("没有获取到新数据，无需追加")
                return None
        
            # 3.保存数据（追加模式）
            if file_exists:
                if df_type=='pandas':
                    existing_df = pd.read_parquet(filename)
                    existing_df['trading_date'] = existing_df['trading_date']
                    # 去重（保留最新数据，基于股票代码和交易日）
                    combined_df = pd.concat([existing_df, new_data], ignore_index=True)\
                                .drop_duplicates(subset=['code', 'trading_date'], keep='last')
                    combined_df.to_parquet(
                        filename,  # 保存目录
                        partition_by=['trading_date'],  # 按交易日分区
                        index=False,
                        engine='pyarrow'  # 或 'pyarrow'
                    )
                else: #polars
                    # 1.取出已有数据
                    existing_df = (pl.scan_parquet(
                        filename,
                        hive_partitioning=True,
                    ).collect().sort(['code', 'trading_date'])
                    )
                    existing_df = existing_df.with_columns(
                        pl.col('trading_date').cast(pl.Date).alias('trading_date')
                    )
                    # 以new_data的列为准，补齐existing_df缺失的列
                    new_data_pl = pl.from_pandas(new_data)
                    new_data_pl = new_data_pl.with_columns(
                        pl.col('trading_date').cast(pl.Date).alias('trading_date')
                    )
                    combined_df = merge_polars_dfs(new_data_pl,existing_df)
                    combined_df.write_parquet(
                        filename,  # 保存目录
                        partition_by=['trading_date'],  # 按交易日分区
                        use_pyarrow=True,
                        compression='zstd' # 可选压缩,压缩方式: 'snappy', 'gzip', 'zstd', 'brotli'
                    )
            else: # 追加模式文件不存在
                if df_type=='pandas':
                    new_data.to_parquet(
                        filename,  # 保存目录
                        partition_cols=['trading_date'],  # 按交易日分区
                        index=False,
                        engine='pyarrow'  # 或 'pyarrow'
                    )
                else: #polars
                    import polars as pl
                    new_data_pl = pl.from_pandas(new_data)
                    new_data_pl.write_parquet(
                        filename,  # 保存目录
                        partition_by=['trading_date'],  # 按交易日分区
                        use_pyarrow=True,
                        compression='zstd' # 可选压缩,压缩方式: 'snappy', 'gzip', 'zstd', 'brotli'
                    )
            
            logging.info(f"追加完成，新增 {len(new_data)} 条记录")
        
        # 更新模式逻辑
        elif mode == 'update':
            logging.info(f"开始更新数据: {start_date} 至 {end_date}")
            # 1.重新获取指定范围数据
            update_data = self.ts_get_stocks_data(
                stock_list=None,
                start_date=start_date,
                end_date=end_date,
                max_workers=max_workers
            )
            
            if update_data.empty:
                logging.info("没有获取到更新数据")
                return None
            
            # 2.保存数据（更新模式）
            if file_exists:
                if df_type=='pandas':
                    # 读取已有数据，替换更新范围内的记录
                    existing_df = pd.read_parquet(filename)
                    existing_df['trading_date'] = existing_df['trading_date']
                    # 筛选出不在更新范围内的旧数据
                    mask = (existing_df['trading_date'] < start_date) | (existing_df['trading_date'] > end_date)
                    remaining_df = existing_df[mask]
                    # 合并剩余数据和新数据
                    combined_df = pd.concat([remaining_df, update_data], ignore_index=True)\
                                .drop_duplicates(subset=['code', 'trading_date'], keep='last')
                    combined_df.to_parquet(
                        filename,  # 保存目录
                        partition_by=['trading_date'],  # 按交易日分区
                        index=False,
                        engine='pyarrow'  # 或 'pyarrow'
                    )
                else: #polars
                    # 读取已有数据，替换更新范围内的记录
                    existing_df = (pl.scan_parquet(
                        filename,
                        hive_partitioning=True,
                    ).collect().sort(['code', 'trading_date'])
                    )
                    existing_df = existing_df.with_columns(
                        pl.col('trading_date').cast(pl.Date).alias('trading_date')
                    )
                    new_data_pl = pl.from_pandas(update_data)
                    # 确保新数据的日期类型与现有数据一致
                    new_data_pl = new_data_pl.with_columns(
                        pl.col('trading_date').cast(pl.Date).alias('trading_date')
                    )
                    # 筛选出不在更新范围内的旧数据
                    mask = (existing_df['trading_date'] < start_date) | (existing_df['trading_date'] > end_date)
                    remaining_df = existing_df.filter(mask)
                    
                    # 合并剩余数据和新数据
                    combined_df = merge_polars_dfs(new_data_pl,remaining_df)
                    combined_df.write_parquet(
                        filename,  # 保存目录
                        partition_by=['trading_date'],  # 按交易日分区
                        use_pyarrow=True,
                        compression='zstd' # 可选压缩,压缩方式: 'snappy', 'gzip', 'zstd', 'brotli'
                    )
            else:
                # 文件不存在，直接保存新数据
                if df_type=='pandas':
                    update_data.to_parquet(
                        filename,  # 保存目录
                        partition_cols=['trading_date'],  # 按交易日分区
                        index=False,
                        engine='pyarrow'  # 或 'pyarrow'
                    )
                else: #polars
                    import polars as pl
                    new_data_pl = pl.from_pandas(update_data)
                    new_data_pl.write_parquet(
                        filename,  # 保存目录
                        partition_by=['trading_date'],  # 按交易日分区
                        use_pyarrow=True,
                        compression='zstd' # 可选压缩,压缩方式: 'snappy', 'gzip', 'zstd', 'brotli'
                    )
            
            logging.info(f"更新完成，处理 {len(update_data)} 条记录")
        
        else:
            logging.info(f"错误: 不支持的模式 {mode}，请使用'append'或'update'")

    # 从下载好的ts本地数据中获取数据
    @timer
    def ts_get_all_date_data(self,stock_list=None,start_date='2021-01-01',end_date='2025-09-01',trade_date=None,data_path='ts_stock_all_data',df_type='polars'):
        """
        从本地获取对应的股票数据，使用多进程加速读取并在读取时筛选所需列
        
        参数:
        - stock_list: 股票代码列表,如果为None则获取全部股票
        - start_date: 开始时间
        - end_date: 结束时间  
        - trade_date: 指定交易日,如果不为None则只获取该交易日的数据
        - data_path: 数据文件夹路径
        - df_type:pandas,polars(读取和存取parquet的格式),默认pandas。polars:按照polars读取exisiting_data和按照polars存取new_data
        - max_workers: 最大进程数

        返回:
        - DataFrame: 包含所有股票数据的合并结果
        """
        import polars as pl
        start_date = convert_date_format(start_date)
        end_date = convert_date_format(end_date)

        if df_type=='pandas':
            df = pd.read_parquet(data_path)
            df['trading_date'] = df['trading_date'].astype(str)
            # 时间过滤
            if trade_date:
                df = df[df['trading_date'] == trade_date]
                logging.info(f"获取指定交易日 {trade_date} 的数据")
            else:
                df = df[(df['trading_date'] >= start_date) & (df['trading_date'] <= end_date)]
                logging.info(f"获取日期范围 {start_date} 至 {end_date} 的数据")
            # 股票代码过滤
            if stock_list:
                df = df[df['code'].isin(stock_list)]
                logging.info(f"过滤股票代码，共 {len(stock_list)} 只")
            return df
        else:  # polars
            # 创建延迟计算的LazyFrame
            lazy_df = pl.scan_parquet(
                data_path,
                hive_partitioning=True,
            )
            
            # 转换日期列为字符串（仍在延迟计算阶段）
            lazy_df = lazy_df.with_columns(
                pl.col('trading_date').cast(pl.Date).alias('trading_date')
            )
            
            # 时间过滤（延迟计算）
            if trade_date:
                lazy_df = lazy_df.filter(pl.col('trading_date') == trade_date)
                logging.info(f"获取指定交易日 {trade_date} 的数据")
            else:
                lazy_df = lazy_df.filter(
                    (pl.col('trading_date') >= start_date) & (pl.col('trading_date') <= end_date)
                )
                logging.info(f"获取日期范围 {start_date} 至 {end_date} 的数据")
            
            # 股票代码过滤（延迟计算）
            if stock_list:
                lazy_df = lazy_df.filter(pl.col('code').is_in(stock_list))
                logging.info(f"过滤股票代码，共 {len(stock_list)} 只")
            
            # 最后才收集结果，只加载需要的数据到内存
            df = lazy_df.collect()
            return df

    # 每日批量获取分钟数据 m_ts.stk_mins_ts(ts_code="600000.SH",freq="5min",start_date="2025-08-25 09:00:00",end_date="2025-08-25 19:00:00")
    def ts_get_date_mins(self,trade_date,stock_list=None,freq='5min',batch_size=100,max_workers=5,df_type='polars',save_path=None):
        """
        获取指定交易日的分钟数据，支持多线程批量获取.tushare的api限制8000条数据
        
        参数:
            trade_date: 交易日期，格式'YYYY-MM-DD'
            stock_list: 股票代码列表，如果为None则获取全部股票
            freq: 分钟数据频率，如'1min', '5min', '15min', '30min', '60min'
            batch_size: 每批处理的股票数量
            max_workers: 最大线程数，默认使用系统推荐值
        
        返回:
            包含所有股票分钟数据的DataFrame
        """
        if stock_list is None:
            stock_list = self.ts_get_all_date_data(trade_date=trade_date, data_path='ts_stock_all_data')['code'].unique().to_list()
        stock_list = self.convert_stock_code(stock_list, to_format='suffix') #转化成后缀模式调用ts的api

        # 将股票列表分成多个批次
        batches = [stock_list[i:i+batch_size] for i in range(0, len(stock_list), batch_size)]
        logging.info(f"将{len(stock_list)}只股票分成{len(batches)}多线程批处理")
        
        all_results = []
        start_time = f"{trade_date} 09:00:00"
        end_time = f"{trade_date} 15:00:00"

        @retry_with_timeout(max_retries=3, timeout=30)
        def process_batch(batch, freq, start_time, end_time, m_ts):
            with suppress_non_error_output():
                m_ts.stk_mins_ts(
                    ts_code=",".join(batch),
                    freq=freq,
                    start_time=start_time,
                    end_time=end_time
                )
        # 使用线程池执行任务
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有批次任务
            futures = {
                executor.submit(
                    process_batch, 
                    batch, 
                    freq, 
                    start_time, 
                    end_time, 
                    self.m_ts
                ): batch for batch in batches
            }
            
            # 处理完成的任务
            for future in as_completed(futures):
                batch = futures[future]
                try:
                    result = future.result()
                    if result is not None and not result.empty:
                        all_results.append(result)
                        logging.info(f"完成批次处理，股票代码: {batch[:3]}...(共{len(batch)}只)")
                    else:
                        logging.info(f"批次处理未返回有效数据，股票代码: {batch[:3]}...(共{len(batch)}只)")
                except Exception as e:
                    logging.info(f"批次处理出错，股票代码: {batch[:3]}...(共{len(batch)}只)，错误: {str(e)}")
        
        # 合并所有结果
        if not all_results:
            logging.info("所有批次处理均未返回有效数据")
            return None
        
        final_df = pd.concat(all_results, ignore_index=True)
        
        logging.info(f"所有批次处理完成，共合并{len(final_df)}条记录")
        final_df = clean_stocks_data(final_df)
        final_df['trading_date'] = trade_date
        if df_type=='polars':
            final_df = pl.from_pandas(final_df)
        if save_path:
            if df_type=='pandas':
                final_df.to_parquet(
                    save_path,  # 保存目录
                    partition_cols=['datetime'],  # 按交易日分区
                    index=False,
                    engine='pyarrow'  # 或 'pyarrow'
                )
            else: #polars
                final_df_pl = final_df if isinstance(final_df, pl.DataFrame) else pl.from_pandas(final_df)
                final_df_pl.write_parquet(
                    save_path,  # 保存目录
                    partition_by=['trading_date'],  # 按交易日分区
                    use_pyarrow=True,
                    compression='zstd' # 可选压缩,压缩方式: 'snappy', 'gzip', 'zstd', 'brotli'
                )
            logging.info(f"分钟数据已保存到 {save_path}")
        time.sleep(1)  # 避免请求过快
        return final_df

    # 下载所有天的分钟数据
    def ts_download_all_mins(self,start_date='2025-08-01',end_date='2025-08-31',stock_list=None,freq='5min',batch_size=50,max_workers=1,df_type='polars',data_path='ts_mins_data'):
        """
        下载指定日期范围内的分钟数据，按天保存为Parquet文件
        
        参数:
            start_date: 开始日期（格式'YYYY-MM-DD'）
            end_date: 结束日期（格式'YYYY-MM-DD'）
            stock_list: 股票代码列表，如果为None则获取全部股票
            freq: 分钟数据频率，如'1min', '5min', '15min', '30min', '60min'
            batch_size: 每批处理的股票数量
            max_workers: 最大线程数，默认使用系统推荐值
            df_type:pandas,polars(读取和存取parquet的格式),默认pandas。polars:按照polars读取exisiting_data和按照polars存取new_data
            data_path: 保存分钟数据的文件夹路径
        """
        # 获取交易日
        trading_date = self.ts.trade_cal(start_date=start_date.replace('-', ''), end_date=end_date.replace('-', ''))
        trading_date = trading_date.sort_values('cal_date')
        trading_dates = trading_date[trading_date['is_open'] == 1]['cal_date'].tolist()
        if not trading_dates:
            logging.info("指定日期范围内没有交易日")
            return None
        for date in trading_dates:
            date_str = datetime.datetime.strptime(str(date), '%Y%m%d').strftime('%Y-%m-%d')
            logging.info(f"开始获取 {date_str} 的分钟数据...")
            self.ts_get_date_mins(
                trade_date=date_str,
                stock_list=stock_list,
                freq=freq,
                batch_size=batch_size,
                max_workers=max_workers,
                df_type=df_type,
                save_path=data_path
            )
            time.sleep(2)  # 避免请求过快
            logging.info(f"{date_str} 的分钟数据获取完成并保存到 {data_path}")

    # 按照交易日多线程获取复权因子数据（ts.adj_factor(ts_code='', trade_date='20251013')）
    def ts_get_adj_factor(self,start_date='2025-01-01',end_date='2025-10-01', max_workers=10):
        """
        获取指定日期范围内的复权因子数据，支持多线程批量获取
        
        参数:
            start_date: 开始日期，格式'YYYY-MM-DD'
            end_date: 结束日期，格式'YYYY-MM-DD'
            max_workers: 最大线程数，默认使用系统推荐值
        
        返回:
            包含所有股票复权因子数据的DataFrame
        """
        # 转化成api需要的格式
        start_date = convert_date_format(start_date,to_format="%Y%m%d")
        end_date = convert_date_format(end_date,to_format="%Y%m%d")
        # 获取交易日
        trading_date = self.ts.trade_cal(start_date=start_date, end_date=end_date)
        if trading_date is None or trading_date.empty:
            logging.info("未能获取交易日历数据")
            return None
        trading_date = trading_date.sort_values('cal_date')
        trading_dates = trading_date[trading_date['is_open'] == 1]['cal_date'].tolist()
        if not trading_dates:
            logging.info("指定日期范围内没有交易日")
            return None

        all_results = []

        @retry_with_timeout(max_retries=3, timeout=30)
        def process_single_date(trade_date):
            with suppress_non_error_output():
                df = self.ts.adj_factor(trade_date=str(trade_date))
            if df is None or df.empty:
                logging.info(f"{trade_date} 无复权因子数据，跳过")
                raise ValueError(f"{trade_date} 无复权因子数据")
            logging.info(f"已获取 {trade_date} 的复权因子数据")
            time.sleep(1)  # 避免请求过快
            return df

        # 使用线程池执行任务
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有交易日任务
            futures = {
                executor.submit(process_single_date, trade_date): trade_date
                for trade_date in trading_dates
            }
            
            # 处理完成的任务
            for future in as_completed(futures):
                trade_date = futures[future]
                try:
                    result = future.result()
                    if result is not None and not result.empty:
                        all_results.append(result)
                except Exception as e:
                    logging.info(f"获取 {trade_date} 复权因子数据时出错: {str(e)}")
        
        # 合并所有结果
        if not all_results:
            logging.info("所有交易日均未返回有效复权因子数据")
            return None
        
        final_df = pd.concat(all_results, ignore_index=True)
        logging.info(f"所有交易日复权因子数据获取完成，共合并{len(final_df)}条记录")
        final_df = clean_stocks_data(final_df)
        return final_df

    # 获取基础指标数据ts.daily_basic(ts_code='',trade_date='20251013')
    def ts_get_daily_basic(self,start_date,end_date=None,stock_list=None,max_workers=3):
        """
        获取指定日期范围内的每日基础指标数据，支持多线程批量获取
        
        参数:
            start_date: 开始日期，格式'YYYY-MM-DD'
            end_date: 结束日期，格式'YYYY-MM-DD'
            stock_list: 股票代码列表，如果为None则获取全部股票
            max_workers: 最大线程数，默认使用系统推荐值
        
        返回:
            包含所有股票每日基础指标数据的DataFrame
        """
        # 转化成api需要的格式
        start_date = convert_date_format(start_date,to_format="%Y%m%d")
        if end_date:
            end_date = convert_date_format(end_date,to_format="%Y%m%d")
        else:
            end_date = convert_date_format(datetime.datetime.now().strftime('%Y-%m-%d'),to_format="%Y%m%d")
        # 获取交易日
        trading_date = self.ts.trade_cal(start_date=start_date, end_date=end_date)
        if trading_date is None or trading_date.empty:
            logging.info("未能获取交易日历数据")
            return None
        trading_date = trading_date.sort_values('cal_date')
        trading_dates = trading_date[trading_date['is_open'] == 1]['cal_date'].tolist()
        if not trading_dates:
            logging.info("指定日期范围内没有交易日")
            return None

        all_results = []

        @retry_with_timeout(max_retries=3, timeout=30)
        def process_single_date(trade_date):
            with suppress_non_error_output():
                df = self.ts.daily_basic(trade_date=str(trade_date))
            if df is None or df.empty:
                logging.info(f"{trade_date} 无每日基础指标数据，跳过")
                raise ValueError(f"{trade_date} 无每日基础指标数据")
            logging.info(f"已获取 {trade_date} 的每日基础指标数据")
            time.sleep(1)  # 避免请求过快
            return df

        # 使用线程池执行任务
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有交易日任务
            futures = {
                executor.submit(process_single_date, trade_date): trade_date
                for trade_date in trading_dates
            }
            
            # 处理完成的任务
            for future in as_completed(futures):
                trade_date = futures[future]
                try:
                    result = future.result()
                    if result is not None and not result.empty:
                        all_results.append(result)
                except Exception as e:
                    logging.info(f"获取 {trade_date} 每日基础指标数据时出错: {str(e)}")

        # 合并所有结果
        if not all_results:
            logging.info("所有交易日均未返回有效每日基础指标数据")
            return None
        final_df = pd.concat(all_results, ignore_index=True)
        logging.info(f"所有交易日每日基础指标数据获取完成，共合并{len(final_df)}条记录")
        final_df = clean_stocks_data(final_df)
        return final_df

    #%% 本地api
    @timer
    def get_all_date_data(self, start_time, end_time, list_assets, data_path='data_qfq', fields=None, max_workers=os.cpu_count()*4):
        """从本地获取对应的股票数据，使用多进程加速读取并在读取时筛选所需列
        
        参数:
        - start_time: 开始时间
        - end_time: 结束时间  
        - list_assets: 股票代码列表
        - data_path: 数据文件夹路径
        - fields: 需要的字段列表，如果为None则读取全部字段
        - max_workers: 最大进程数

        返回:
        - DataFrame: 包含所有股票数据的合并结果
        """
        list_all = []
        failed_list = []
        
        # 使用进程池并行读取文件
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有读取任务
            future_to_symbol = {executor.submit(self.read_stock_file, symbol,data_path,start_time,end_time,fields): symbol for symbol in list_assets}
            
            # 处理任务结果
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result()
                    
                    if result['success']:
                        list_all.append(result['data'])
                    else:
                        failed_list.append(symbol)
                        logging.info(f"[读取失败] {symbol}，原因: {result['error']}")
                except Exception as e:
                    failed_list.append(symbol)
                    logging.info(f"[处理失败] {symbol}，原因: {str(e)}")
        
        # 检查是否有成功读取的数据
        if not list_all:
            logging.info("没有任何股票数据被成功读取！")
            return pd.DataFrame()  # 返回空DataFrame
        
        # 合并所有数据
        #logging.info(f"合并 {len(list_all)} 个股票数据...")
        df_all = pd.concat(list_all, ignore_index=False)
        
        logging.info(f"全部读取完成，成功: {len(list_all)} 只，失败: {len(failed_list)} 只")
        if failed_list:
            logging.info(f"读取失败的股票代码：{failed_list}")
        
        return df_all

    def get_stocks_list(self, data_path='data_qfq'):
        """
        查找本地指定路径下所有股票代码（假设每只股票一个csv文件，文件名为股票代码）
        :param data_path: 存放股票数据的文件夹路径
        :return: 股票代码列表
        """
        import os
        stock_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
        # 去掉.csv后缀，得到股票代码
        stock_codes = [os.path.splitext(f)[0] for f in stock_files]
        return stock_codes

    def get_industry_list(self,industry_path='data_industry_em'):
        """获取本地保存的行业列表
        
        Returns:
            list: 包含行业代码和名称的元组列表 [(code1, name1), (code2, name2),...]
        """
        if not os.path.exists(industry_path):
            logging.error("行业数据文件夹不存在")
            return []
            
        # 读取所有CSV文件名
        industry_files = [f for f in os.listdir(industry_path) if f.endswith('.csv')]
        
        # 从文件名中提取行业代码和名称
        industry_list = []
        for filename in industry_files:
            # 文件名格式为: code_name.csv
            code_name = filename.replace('.csv', '')
            code, name = code_name.split('_', 1)
            industry_list.append((code, name))
            
        return industry_list

    def get_industry_data(self,industry_code=None, industry_name=None, start_date=None, end_date=None,    industry_path = 'data_industry_em'):
        """获取指定行业的历史数据
        
        Args:
            industry_code: 行业代码
            industry_name: 行业名称
            start_date: 开始日期，格式'YYYY-MM-DD'
            end_date: 结束日期，格式'YYYY-MM-DD'
        
        Returns:
            pd.DataFrame: 行业历史数据
        """

        
        # 获取所有行业列表
        industry_list = self.get_industry_list()
        
        # 根据代码或名称查找对应文件
        target_file = None
        if industry_code:
            target_file = next((f"{code}_{name}.csv" for code, name in industry_list 
                            if code == industry_code), None)
        elif industry_name:
            target_file = next((f"{code}_{name}.csv" for code, name in industry_list 
                            if name == industry_name), None)
        else:
            raise ValueError("必须指定行业代码或名称之一")
            
        if not target_file:
            raise ValueError("未找到指定的行业数据")
            
        # 读取数据
        file_path = os.path.join(industry_path, target_file)
        df = pd.read_csv(file_path)
        
        # 时间过滤
        if start_date:
            df = df[df['date'] >= start_date]
        if end_date:
            df = df[df['date'] <= end_date]
            
        return df


    def get_industry_info(self,industry_name):
        """
        获取行业成分股信息
        
        Args:
            industry_name: 行业名称，如"小金属"、"通信设备"等
            
        Returns:
            pandas.DataFrame: 包含以下字段的DataFrame:
                - 序号：成分股序号
                - 代码：股票代码
                - 名称：股票名称
                - 最新价：最新股价
                - 涨跌幅：当日涨跌幅
                - 涨跌：股价涨跌额
                - 涨速：涨跌速度
                - 换手：换手率
                - 量比：成交量比
                - 振幅：股价振幅
                - 成交额：成交金额
                - 流通股：流通股本
                - 流通市值：流通市值
                - 市盈率：市盈率
        """
        try:
            # 调用akshare接口获取行业成分股数据
            df = ak.stock_board_industry_cons_em(symbol=industry_name)
            
            # 标准化处理
            df = df.rename(columns={
                "代码": "symbol",
                "名称": "name",
                "最新价": "price",
                "涨跌幅": "pct_chg",
                "换手率": "turnover",
                "振幅": "amplitude",
                "成交额": "amount",
                "流通股": "float_shares",
                "流通市值": "float_market_value",
                "市盈率": "pe",
                '昨收': 'pre_close',
                '今开': 'open',
                '最高': 'high',
                '最低': 'low',
                '成交量': 'volume',
            })
            
            
            logging.info(f"获取行业 {industry_name} 成分股信息成功，共 {len(df)} 只股票")
            return df
            
        except Exception as e:
            logging.error(f"获取行业 {industry_name} 成分股信息失败: {str(e)}")
            raise

    def add_future_pct_chg(self,df, n_list=[1, 3, 5, 10, 20]):
        """
        为df中每只股票计算未来n日涨幅
        :param df: 包含所有股票日线数据的DataFrame，需含['symbol', 'date', 'close']
        :param n_list: 需要计算的未来天数列表
        :return: 增加了未来涨幅字段的DataFrame
        """
        df = df.sort_values(['symbol', 'date']) #按照股票顺序排序
        for n in n_list:
            if n==1:
                df['future_1d_chg'] = df.groupby('symbol')[f'pct_chg'].shift(-1) # df已经按照symbol排序
                continue
            df[f'future_{n}d_chg'] = df.groupby('symbol')[f'pct_chg_{n}d'].shift(-n)
        return df

    def calculate_k_features(self,df):
        """
        计算K线特征
        1. k线形体特征(k线实体、影线、波动率、阴阳线等)
        2. 均线特征
        """
        # 复制数据避免修改原始DataFrame
        df.sort_values(by=['symbol', 'date'], inplace=True)  # 确保数据按股票和日期排序
        result = df.copy()
        
        # 1.计算K线形态特征
        result['k_center'] = (result['high'] + result['low'] + result['open'] + result['close']) / 4
        result['body_size'] = abs(result['close'] - result['open'])
        result['shadow_size'] = result['high'] - result['low']
        result['body_ratio'] = result['body_size'] / (result['shadow_size'] + 1e-8)
        result['upper_shadow'] = result['high'] - result[['open', 'close']].max(axis=1)
        result['lower_shadow'] = result[['open', 'close']].min(axis=1) - result['low']
        result['volatility'] = (result['high'] - result['low']) / result['low'] * 100
        
        # 判断阴阳线
        result['is_positive'] = (result['close'] > result['open']).astype(int)
        result['is_negative'] = (result['close'] < result['open']).astype(int)
        
        # 连续阴阳线特征
        result['continuous_positive'] = 0
        result['continuous_negative'] = 0
        
        # 计算连续阴阳线数量
        for i in range(1, len(result)):
            if result['is_positive'].iloc[i] == 1:
                result.loc[result.index[i], 'continuous_positive'] = result['continuous_positive'].iloc[i-1] + 1
                result.loc[result.index[i], 'continuous_negative'] = 0
            elif result['is_negative'].iloc[i] == 1:
                result.loc[result.index[i], 'continuous_negative'] = result['continuous_negative'].iloc[i-1] + 1
                result.loc[result.index[i], 'continuous_positive'] = 0
        
        # 计算量价特征
        result['vol_price_ratio'] = result['volume'] / (result['close'] + 1e-8)
        
        # 实体比例
        result['entity_ratio'] = result['body_size'] / (result['shadow_size'] + 1e-8)
        
        # K线形态分类 (大阳线、大阴线、十字星等)
        result['doji'] = (result['body_size'] < 0.1 * result['shadow_size']).astype(int)  # 十字星
        result['large_positive'] = ((result['is_positive'] == 1) & 
                                (result['body_size'] > 0.6 * result['shadow_size'])).astype(int)  # 大阳线
        result['large_negative'] = ((result['is_negative'] == 1) & 
                                (result['body_size'] > 0.6 * result['shadow_size'])).astype(int)  # 大阴线
        
        #2. 计算均线特征
        # 计算不同周期的均线
        for period in [5, 10, 20, 30, 60]:
            result[f'sma_{period}'] = result.groupby('symbol')['close'].transform(lambda x: x.rolling(window=period, min_periods=1).mean())


        return result
    
