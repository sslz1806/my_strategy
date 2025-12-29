# 将指定目录添加到系统路径
import sys
DATA_ROOT_DIR = r'E:\working\stock_data'

# 取数,标记特征
# 需要的特征:涨停状态（未涨停,炸板,断板-三天之内有涨停），涨停描述(几天几板)，sma(均线)
import polars as pl
import pandas as pd
import numpy as np
from mapping import *
import os
import logging
from typing import Optional,Dict


def get_logger(
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    file_format: Dict[str, str] = {'format': '%(asctime)s- %(levelname)s- %(message)s', 'datefmt': '%Y%m%d %H:%M:%S'},
    console_format: Dict[str, str] = {'format': '%(message)s'}
) -> logging.Logger:
    """
    极简版：字典参数控制日志格式，有外部配置则复用，无则新建
    :param log_file: 可选，日志文件路径（None则仅控制台输出）
    :param level: 日志级别，默认INFO
    :param file_format: 文件格式配置，默认带时间/级别前缀
    :param console_format: 控制台格式配置，默认仅日志内容
    :return: 根logger对象
    """
    root_logger = logging.getLogger()
    
    # 核心判断：根logger无有效处理器时才配置
    if not any(
        isinstance(h, (logging.StreamHandler, logging.FileHandler)) and not h.stream.closed
        for h in root_logger.handlers
    ):
        # 创建格式器（简化变量名）
        f_formatter = logging.Formatter(fmt=file_format['format'], datefmt=file_format.get('datefmt'))
        c_formatter = logging.Formatter(fmt=console_format['format'], datefmt=console_format.get('datefmt'))
        
        # 创建控制台处理器
        c_handler = logging.StreamHandler()
        c_handler.setFormatter(c_formatter)
        
        # 构建处理器列表
        handlers = [c_handler]
        if log_file:
            dir_path = os.path.dirname(log_file.strip())
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            f_handler = logging.FileHandler(log_file, encoding='utf-8')
            f_handler.setFormatter(f_formatter)
            handlers.append(f_handler)
        
        # 最终配置
        logging.basicConfig(level=level, handlers=handlers)
    
    return root_logger

def get_parquet_dir_schema(data_dir: str):
    """
    获取指定目录中Parquet文件的统一schema信息
    
    参数:
        data_dir: Parquet文件所在的根目录（支持分区存储）
    
    返回:
        包含列名和对应类型的schema字典，如果目录中无Parquet文件则返回None
    """
    import os
    data_dir = os.path.join(DATA_ROOT_DIR, data_dir)
    # 递归查找目录中第一个Parquet文件
    def find_first_parquet(root_dir: str):
        for dirpath, _, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename.endswith(".parquet"):
                    return os.path.join(dirpath, filename)
        return None
    
    # 查找第一个Parquet文件
    first_parquet = find_first_parquet(data_dir)
    
    if not first_parquet:
        print(f"警告：在目录 {data_dir} 中未找到任何Parquet文件")
        return None
    
    # 仅读取schema（n_rows=0表示不加载数据）
    try:
        schema = pl.read_parquet(first_parquet, n_rows=0).schema
        print(f"成功从 {first_parquet} 读取schema")
        return schema
    except Exception as e:
        print(f"读取schema失败：{str(e)}")
        return None
def read_min_data(start_time,end_time,stock_list=None,file_path='15min_stock_data_dir') -> pl.DataFrame:
    """
    读取分钟数据，并根据时间范围和股票列表进行过滤
    start_time,end_time:datetime.datetime对象
    stock_list:股票代码列表,gm格式
    file_path:parquet文件路径
    例如:stock_list=['SZSE.000001','SHSE.600000']
    """
    import datetime as dt
    # 拼接文件路径（基于 fun.py 目录）
    file_path = os.path.join(DATA_ROOT_DIR, file_path)
    if isinstance(start_time, pd._libs.tslibs.timestamps.Timestamp):
        start_time = pd.to_datetime(start_time)
    if isinstance(end_time, pd._libs.tslibs.timestamps.Timestamp):
        end_time = pd.to_datetime(end_time)

    # 2. 新增：处理 datetime.date 类型（转为 datetime.datetime 对象）
    if isinstance(start_time, dt.date) and not isinstance(start_time, dt.datetime):
        # date转datetime（默认00:00:00）
        start_time = dt.datetime.combine(start_time, dt.time.min)
    if isinstance(end_time, dt.date) and not isinstance(end_time, dt.datetime):
        # date转datetime（默认23:59:59，避免漏数据）
        end_time = dt.datetime.combine(end_time, dt.time.max)
    start_date = start_time.date()
    end_date = end_time.date()
    schema = {
        "code": pl.String,
        "datetime": pl.Datetime,
        "open": pl.Float64,
        "high": pl.Float64,
        "low": pl.Float64,
        "close": pl.Float64,
        "volume": pl.Float64,  # 强制指定为Float64，覆盖文件元数据
        "trading_date": pl.Date
    }

    # 扫描时应用schema，避免类型推断冲突
    df = pl.scan_parquet(
        file_path,
        schema=schema,  # 显式传入schema
        cast_options=pl.ScanCastOptions(datetime_cast='nanosecond-downcast'),
    )
    df = df.with_columns(pl.col("volume").cast(pl.Float64))
    df = df.filter(pl.col("trading_date").is_between(start_date, end_date))
    if stock_list is not None:
        df = df.filter(pl.col("code").is_in(stock_list))

    return df.collect()

def read_day_data(start_date,end_date,stock_list=None,fields=None,file_path='ts_stock_all_data') -> pl.DataFrame:
    """
    读取日线数据，并根据时间范围,股票列表,需要的列进行过滤
    start_date,end_date:datetime.date对象
    stock_list:股票代码列表,gm格式
    fields:需要的列列表,None表示全部列
    file_path:parquet文件路径
    例如:fields=['code','trading_date','open','close']
    """

    file_path = os.path.join(DATA_ROOT_DIR, file_path)
    start_date = convert_date_format(start_date,to_format='date')
    end_date = convert_date_format(end_date,to_format='date')
    df =pl.scan_parquet(file_path).filter(pl.col("trading_date").is_between(start_date,end_date))
    if stock_list is not None:
        df = df.filter(pl.col("code").is_in(stock_list))
    if fields is not None:
        df = df.select(fields)
    return df.collect()

#%% 计算特征函数
def mark_limit_status(stock_data: pl.DataFrame,db_days=2) -> pl.DataFrame:
    """
    标记涨停状态（未涨停, 炸板, 断板, 正常涨停）
    db_days:涨停后标记断板的天数
    """
    stock_data = stock_data.sort(["code", "trading_date"])
    stock_data = stock_data.with_columns([
        ((pl.col("close") >= pl.col("limit_up") * 0.999)).alias("is_limit_up"),
        ((pl.col("high") >= pl.col("limit_up") * 0.999) & (pl.col("close") < pl.col("limit_up") * 0.999)).alias("is_broken_limit"),
    ])
    # 断板标记：最近3天（不含当天）有涨停，且当天未涨停也未炸板
    def mark_db(group: pl.DataFrame) -> pl.DataFrame:
        is_limit_up = group["is_limit_up"].to_list()
        status = []
        for i in range(len(is_limit_up)):
            # 最近3或2天（不含当天）有涨停
            recent = is_limit_up[max(0, i-db_days):i]
            if not is_limit_up[i] and not group["is_broken_limit"][i] and any(recent):
                status.append("断板")
            # 如果昨天炸板过，今天也算断板
            # elif not is_limit_up[i] and not group["is_broken_limit"][i] and i>0 and group["is_broken_limit"][i-1]:
            #     status.append("断板")
            else:
                status.append(None)
        return group.with_columns([
            pl.Series("is_db", [s == "断板" for s in status]),
            pl.Series("limit_status_ext", status, dtype=pl.String)  # 明确指定为字符串类型
        ])
    stock_data = stock_data.group_by("code").map_groups(mark_db)
    # 综合状态
    stock_data = stock_data.with_columns([
        pl.when(pl.col("is_limit_up")).then(pl.lit("涨停"))  # 用pl.lit()明确表示这是字符串字面量
        .when(pl.col("is_broken_limit")).then(pl.lit("炸板"))
        .when(pl.col("is_db")).then(pl.lit("断板"))
        .otherwise(pl.lit("未涨停")).alias("limit_status")
    ])
    return stock_data

def mark_limit_desc(stock_data: pl.DataFrame) -> pl.DataFrame:
    """
    标记涨停描述（几天几板），允许中间有断板，直到再次未涨停为止
    逻辑：按日计算，每天根据从最近一个未涨停日到当前的区间：
         - 区间总天数为n，其中涨停的天数为m，则标记为"n天m板"
         - 未涨停日标记为"无"，并作为新的计算起点
    """
    def calc_desc(group):
        # 确保按交易日排序
        group = group.sort("trading_date")
        is_limit_up = group["is_limit_up"].to_list()
        limit_status = group["limit_status"].to_list()
        desc_list = []
        period_start = 0  # 记录当前周期的起始索引（最近一个未涨停日的下一天）
        
        for i in range(len(is_limit_up)):
            # 计算从周期开始到当前的总天数
            total_days = i - period_start + 1
            
            # 计算从周期开始到当前的涨停天数
            up_days = sum(is_limit_up[period_start:i+1])
            
            if limit_status[i]=='涨停' or limit_status[i]=='炸板':
                # 涨停日：计算到今天为止的"n天m板"
                desc_list.append(f"{total_days}天{up_days}板")
            elif limit_status[i]=='未涨停':
                # 未涨停日：标记为"无"，并重置周期起点
                desc_list.append("未涨停")
                period_start = i + 1  # 下一个周期从明天开始
            elif limit_status[i]=='炸板':
                desc_list.append('炸板')
            elif limit_status[i]=='断板':
                desc_list.append('断板')
        
        return group.with_columns([
            pl.Series("limit_desc", desc_list)
        ])
    
    # 按股票代码分组计算
    return stock_data.group_by("code").map_groups(calc_desc)

def mark_last_limit_desc(stock_data: pl.DataFrame) -> pl.DataFrame:
    """
    向前记录最后一次涨停状态的函数
    逻辑：当遇到涨停、炸板或断板状态时，从当前周期开始到前一天（不包括今日）的范围内，
         记录最后一次出现的涨停描述limit_desc；未涨停时不记录
    """
    def calc_last_desc(group):
        # 确保按交易日排序
        group = group.sort("trading_date")
        limit_status = group["limit_status"].to_list()
        limit_desc = group["limit_desc"].to_list()  # 假设已存在涨停描述列
        last_limit_desc_list = []
        period_start = 0  # 记录当前周期的起始索引（最近一个未涨停日的下一天）
        last_valid_desc = None  # 记录最后一次有效的涨停描述
        
        for i in range(len(limit_status)): 
            if i ==0:
                last_limit_desc_list.append(None)
                continue
            
            pre_status = limit_status[i-1] #昨天的涨停状态
            # 对于涨停、炸板、断板状态，需要查找之前的最后一次涨停描述
            if pre_status in ['涨停', '炸板', '断板']:
                # 查找从period_start到i-1（不包括当前）的最后一次涨停描述
                # 从i-1向前搜索，找到最近的一个有效描述，直到
                for j in range(i-1, period_start-1, -1):# i-1到period_start
                    if limit_status[j] in ['涨停']:
                        last_valid_desc = limit_desc[j]
                        break
                
                last_limit_desc_list.append(last_valid_desc)
            else:  # 昨日未涨停状态
                last_limit_desc_list.append(None)
                period_start = i - 1  # 重置为昨天
                last_valid_desc = None  # 重置最后有效描述
        
        return group.with_columns([
            pl.Series("last_limit_desc", last_limit_desc_list,dtype=pl.String)
        ])
    
    # 按股票代码分组计算
    return stock_data.group_by("code").map_groups(calc_last_desc)

def add_sma(stock_data: pl.DataFrame, window: int = 5,column:str="close") -> pl.DataFrame:
    """
    计算close的均线
    """
    return stock_data.with_columns([
        pl.col(column).rolling_mean(window,min_samples=1).over("code").alias(f"sma_{window}")
    ])

# 计算字段与前N日均值的比(如量比这个指标=今日volumn/前N日volumn均值)
def add_pre_n_ratio(stock_data: pl.DataFrame, field: str='volume', n: int = 5) -> pl.DataFrame:
    """
    计算指定字段与前N日均值的比值（如量比=今日成交量/前N日成交量均值）
    
    参数:
        stock_data: Polars DataFrame，需包含"code"（股票代码）、"trading_date"（交易日）及指定字段
        field: 要计算的字段名称（如"volume"）
        n: 前N日的窗口大小，默认5天
    
    返回:
        添加了{field}_ratio_{n}列的DataFrame，值为当前字段值除以前N日均值
    """
    # 确保数据按股票代码和交易日排序
    stock_data = stock_data.sort(["code", "trading_date"])
    
    # 计算前N日均值（不含当日）：
    # 1. 先计算包含当日的N天滚动均值
    # 2. 用shift(1)将结果后移1天，得到前N天（不含当日）的均值
    prev_n_mean = pl.col(field).rolling_mean(window_size=n, min_samples=1).over("code").shift(1)
    
    # 计算比值（当前值 / 前N日均值），避免除零
    ratio_col = (pl.col(field) / prev_n_mean).fill_nan(0.0).fill_null(0.0)
    
    # 添加结果列
    return stock_data.with_columns(
        ratio_col.alias(f"{field}_ratio_{n}")
    )


def cal_industry_concentration(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """
    遍历信号文件中的每个信号，计算窗口内出现该信号股票的行业的比例
    (即: window天内industry=该股票industry的股票数量/窗口天数内所有股票数量)
    
    参数:
        df: pandas DataFrame，必须包含列：trading_date(交易日期)、code(股票代码)、industry(行业)
        window: 计算窗口天数，默认3天
    
    返回:
        新增industry_concentration列的DataFrame
    """
    # 复制原数据避免修改输入
    df = df.copy()
    # 按交易日期和股票代码排序
    df = df.sort_values(['trading_date', 'code']).reset_index(drop=True)
    
    # 获取所有唯一的交易日期并排序
    trading_dates = sorted(df['trading_date'].unique())
    n_dates = len(trading_dates)
    
    # 存储每个信号的行业集中度结果
    industry_concentration = []
    
    for i, current_date in enumerate(trading_dates):
        # 前window-1天没有足够的窗口数据，填充0.0
        if i < window:
            # 获取当前日期的信号数量，每个信号都填充0.0
            current_day_count = len(df[df['trading_date'] == current_date])
            industry_concentration.extend([0.0] * current_day_count)
            continue
        
        # 计算窗口起始日期
        window_start_date = trading_dates[i - window]
        
        # 筛选窗口内的所有数据
        window_mask = (df['trading_date'] >= window_start_date) & (df['trading_date'] <= current_date)
        window_df = df[window_mask]
        
        # 窗口内总股票数量（注意：如果同一股票同一天多条记录会被多次计数）
        total_stocks = len(window_df)
        
        # 获取当前日期的所有信号
        current_day_signals = df[df['trading_date'] == current_date]
        
        # 计算当前日期每个信号的行业集中度
        for _, row in current_day_signals.iterrows():
            industry = row['industry']
            # 窗口内该行业的股票数量
            industry_count = len(window_df[window_df['industry'] == industry])
            # 计算比例（避免除零）
            concentration = industry_count / total_stocks if total_stocks > 0 else 0.0
            industry_concentration.append(concentration)
    
    # 添加行业集中度列
    df['industry_concentration'] = industry_concentration
    
    return df


def cal_n_lowest(stock_data: pl.DataFrame, window: int = 30, include_today: bool = False) -> pl.DataFrame:
    """
    计算股票n日内的最低股价（兼容旧版本Polars，仅使用基础API）
    
    参数:
        stock_data: 包含股票数据的DataFrame，需包含"code"（股票代码）、"trading_date"（交易日）、"low"（当日最低价）列
        window: 计算最低股价的窗口大小，默认30天（即“n日”中的n）
        include_today: 是否包含当天价格，默认为False（计算“前n天”最低值，不含当天；True则计算“包含当天在内的n天”最低值）
    
    返回:
        添加了n日内最低股价列的DataFrame，新增列名为 `n_lowest_{window}`（如window=30时为`n_lowest_30`）
    """
    # 1. 先确保数据按“股票代码+交易日”排序（滚动计算必须基于时间顺序）
    stock_data = stock_data.sort(["code", "trading_date"])
    
    # 2. 按股票分组计算滚动最低价（核心逻辑，兼容所有Polars版本）
    if include_today:
        # 场景1：包含当天 → 直接计算“当前日及之前共window天”的最低价
        # 窗口范围：[当前日 - window + 1 天, 当前日]，共window天
        rolling_min = pl.col("low").rolling_min(window_size=window,min_samples=1).over("code")
    else:
        # 场景2：不包含当天 → 先算“前window天（含前1天）”的最低价，再整体后移1天（排除当天）
        # 步骤1：计算“当前日及之前共window天”的最低价（此时包含当天）
        # 步骤2：用shift(1)将结果后移1天 → 当天结果变为“前window天（不含当天）”的最低价
        rolling_min = pl.col("low").rolling_min(window_size=window,min_samples=1).over("code").shift(1)
    
    # 3. 添加结果列并返回
    return stock_data.with_columns(
        rolling_min.alias(f"lowest_{window}")
    )


def add_ema(stock_data: pl.DataFrame, window: int = 5) -> pl.DataFrame:
    """
    计算EMA（指数加权移动平均线），模仿SMA写法
    参数:
        stock_data: Polars DataFrame，需包含'close'和'code'列
        window: EMA窗口长度
    返回:
        添加了ema_{window}列的DataFrame
    """
    # 先转为pandas分组计算EMA
    df = stock_data.select(['code', 'trading_date', 'close']).to_pandas()
    df = df.sort_values(['code', 'trading_date'])
    df[f'ema_{window}'] = df.groupby('code')['close'].transform(lambda x: x.ewm(span=window, adjust=False).mean())
    # 按原顺序合并回polars
    stock_data = stock_data.with_columns(
        pl.Series(f'ema_{window}', df[f'ema_{window}'])
    )
    return stock_data

def add_ewma_volatility(stock_data: pl.DataFrame, window: int = 10, alpha: float = 0.94) -> pl.DataFrame:
    """
    计算EWMA波动率（指数加权移动平均波动率）
    参数:
        stock_data: 需包含 "code"、"trading_date"、"pct" 列（pct为涨跌幅，如3表示3%）
        window: 初始方差的计算窗口（递归起点）
        alpha: 衰减系数（0<α≤1）
    返回:
        添加 ewma_volatility_{window} 列的DataFrame（单位：%，与pct列一致）
    """
    import numpy as np

    stock_data = stock_data.sort(["code", "trading_date"])

    # 步骤1：将 pct 转为小数形式（如3%→0.03），再计算平方收益率
    stock_data = stock_data.with_columns(
        (pl.col("pct") / 100).pow(2).alias("return_squared")
    )

    # 步骤2：计算初始方差（前window天的等权重方差，作为递归起点）
    stock_data = stock_data.with_columns(
        pl.col("return_squared")
        .rolling_var(window_size=window, min_periods=1, ddof=1)
        .over("code")
        .alias("initial_variance")
    )

    # 步骤3：分组递归计算EWMA方差和波动率
    def calc_ewma_vol(group: pl.DataFrame) -> pl.DataFrame:
        returns_sq = group["return_squared"].to_list()
        initial_var = group["initial_variance"].to_list()
        ewma_var = []
        for i in range(len(returns_sq)):
            if i == 0:
                v = initial_var[i]
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    v = 0.0
                ewma_var.append(v)
            else:
                prev = ewma_var[i-1]
                if prev is None or (isinstance(prev, float) and np.isnan(prev)):
                    prev = 0.0
                ewma_var.append(alpha * returns_sq[i] + (1 - alpha) * prev)
        ewma_vol = [float(v ** 0.5) * 100 if v >= 0 else 0.0 for v in ewma_var]
        return group.with_columns([
            pl.Series(f"ewma_volatility_{window}", ewma_vol)
        ])

    stock_data = stock_data.group_by("code").map_groups(calc_ewma_vol)
    stock_data = stock_data.drop(["return_squared", "initial_variance"])
    return stock_data