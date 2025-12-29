import pandas as pd
import numpy as np
column_mapping = {
        '代码': 'code',
        'ts_code':'code',
        'symbol':'code',
        '名称': 'name',
        '最新价': 'close',
        '涨跌幅': 'pct',
        'pct_chg':'pct',
        'trade_date':'trading_date',
        '昨收': 'pre_close',
        '今开': 'open',
        '最高': 'high',
        '最低': 'low',
        '成交量': 'volume',
        'vol':'volume',
        '成交额': 'amount',
        '时间戳': 'datetime1',
        '流通市值':'mv_A_free_float',
        'upper_limit':'limit_up',
        'lower_limit':'limit_down',
        'up_limit':'limit_up',
        'down_limit':'limit_down',
        'price':'close',
        'trade_time':'datetime',
        'pct_change':'pct',
    }


def convert_code_format(code,format='gm'):
    """
    转换股票代码格式，支持多种输入格式(过滤北交所):
    - 000001.SZ -> SZSE.000001
    - sz000001 -> SZSE.000001
    - 600000.SH -> SHSE.600000
    - sh600000 -> SHSE.600000
    会过滤掉北交所代码。
    
    Args:
        code: str或list，股票代码或股票代码列表
        format:gm,suffix,pure
        
    Returns:
        str或list：转换后的代码格式，北交所代码返回None
    """
    def _convert_single_code(code,format='gm'):
        # 过滤北交所代码
        if 'bj' in code.lower():
            return None

        # 1.处理输入代码  
        # 处理带点的格式 (000001.SZ,SHSE.000001)
        if '.' in code:
            if '.SZ' in code or '.SH' in code:
                code_num, market = code.split('.')
                market = market.upper()
            elif 'SZSE' in code or 'SHSE' in code:
                market,code_num = code.split('.')
                market = market.upper()
            else:
                raise ValueError(f"Unknown market code: {code}")
        # 处理不带点的格式 (sz000001,SZ00001的8位数)
        else:
            # 提取市场代码和数字
            if len(code) > 6:
                market = code[:2].upper()
                code_num = code[2:]
            elif len(code) == 6:
                if code.startswith('6'):
                    market = 'SH'  
                elif code.startswith('0'):
                    market = 'SZ'
                else:
                    raise ValueError(f"Unknown market code: {code}")
                code_num = code
            else:
                raise ValueError(f"Unknown market code: {code}")
            
        # 2.统一处理市场代码
        if format=='gm': #掘金格式
            if market in ['SH', 'SHSE']:
                return f'SHSE.{code_num}'
            elif market in ['SZ', 'SZSE']:
                return f'SZSE.{code_num}'
            else:
                raise ValueError(f"Unknown market code: {market}")
        elif format=='suffix':
            if market in ['SH', 'SHSE']:
                return f'{code_num}.SH'
            elif market in ['SZ', 'SZSE']:
                return f'{code_num}.SZ'
            else:
                raise ValueError(f"Unknown market code: {market}")
        elif format=='pure':
            return code_num
        else:
            raise ValueError(f"Unknown format: {format}")
            

    if isinstance(code, str):
        return _convert_single_code(code,format=format)
    elif isinstance(code, list) or isinstance(code, pd.Series):  # 这里使用直接的 list 类型检查
        # 转换并过滤None值
        converted = [_convert_single_code(c, format=format) for c in code]
        return converted
    elif isinstance(code, __builtins__.list):
        # 转换并过滤None值
        converted = [_convert_single_code(c,format=format) for c in code]
        return converted
    # 数字格式
    elif isinstance(code, int):
        code_str = str(code).zfill(6)
        return _convert_single_code(code_str,format=format)
    else:
        raise TypeError("code should be a string or a list of strings")

def clean_stocks_data(df, code_format='gm',column_mapping=column_mapping):
    """
    1.根据提供的列名映射关系修改DataFrame的列名,2.修改code格式并剔除北交所,3.修改日期格式
    
    参数:
        df (pd.DataFrame): 需要修改列名的DataFrame
        column_mapping (dict): 列名映射关系，键为原始列名，值为目标列名
        
    返回:
        pd.DataFrame: 列名已修改的DataFrame
    """
    # 创建一个新的列名映射字典，只包含DataFrame中实际存在的列
    existing_mapping = {original: target for original, target in column_mapping.items() 
                       if original in df.columns}
    
    # 重命名列
    renamed_df = df.rename(columns=existing_mapping)
    renamed_df['code'] = convert_code_format(renamed_df['code'],format=code_format) #北交所为空
    renamed_df = renamed_df.dropna(subset=['code']) # 删除北交所
    
    # 处理日期列trading_date
    # 3. 标准化日期列trading_date（如果存在）
    if 'trading_date' in renamed_df.columns:
        # 处理可能的缺失值
        renamed_df['trading_date'] = renamed_df['trading_date'].replace(['', 'nan', np.nan], None)
        # 应用日期转换
        #renamed_df['trading_date'] = renamed_df['trading_date'].apply(convert_to_standard_date)
        # 如果日期是int类型，先转换为字符串
        if renamed_df['trading_date'].dtype == 'int64' or renamed_df['trading_date'].dtype == 'float64':
            renamed_df['trading_date'] = renamed_df['trading_date'].astype(str)
        renamed_df['trading_date'] = convert_date_format(renamed_df['trading_date'])
    return renamed_df

# 定义日期转换函数
def convert_to_standard_date(date_value):

    if date_value is None:
        return None
        
    # 处理datetime类型（带时区或不带时区）
    if isinstance(date_value, pd.Timestamp):
        # 移除时区信息并转换为字符串
        return date_value.tz_localize(None).strftime('%Y-%m-%d')
        
    # 处理字符串类型
    if isinstance(date_value, str):
        # 尝试常见格式
        for fmt in ['%Y-%m-%d', '%Y%m%d', '%Y/%m/%d', '%d-%m-%Y', '%d/%m/%Y']:
            try:
                return pd.to_datetime(date_value, format=fmt).strftime('%Y-%m-%d')
            except ValueError:
                continue
        
        # 如果以上格式都不匹配，尝试自动解析
        try:
            return pd.to_datetime(date_value).tz_localize(None).strftime('%Y-%m-%d')
        except ValueError:
            return None  # 无法解析的日期返回None
    
    # 处理其他类型（如整数表示的日期）
    try:
        return pd.to_datetime(date_value).tz_localize(None).strftime('%Y-%m-%d')
    except (ValueError, TypeError):
        return None
    
    
# 定义通用日期格式转换函数
def convert_date_format(date_value,to_format='date'):
    import datetime
    """
    将日期转换为指定格式的字符串。支持类型包括:字符串、datetime对象、pd.Timestamp、pd.Series和列表。
    
    参数:
        date_value(str,datetime,pd.Timestamp,series,list都有可能): 日期值，可以是字符串、datetime对象或pd.Timestamp
        to_format: 目标日期格式，默认为'%Y-%m-%d',还有可能是'%Y%m%d','%Y/%m/%d','datetime','date
        
    返回:
        str或datetime: 转换后的日期字符串或datetime对象
    """
    if date_value is None:
        return None
    
    if isinstance(date_value, pd.Series):
        # 处理带时区的pd.Series
        date_series = pd.to_datetime(date_value).dt.tz_localize(None)
        if to_format == 'datetime': #datetime64类型
            return date_series
        elif to_format == 'date': #datetime.date类型的object
            return date_series.dt.date
        else:
            return date_series.dt.strftime(to_format) #srting类型
        
    def _convert_single_date(date_value, to_format='%Y-%m-%d'):
        try:
            # 统一转换为pandas Timestamp（自动处理各种输入类型）
            # tz_localize(None) 移除时区信息（如需保留可删除此参数）
            dt = pd.to_datetime(date_value).tz_localize(None)
            
            # 根据目标格式返回对应类型
            if to_format == 'datetime':
                return dt  # 返回pd.Timestamp类型
            elif to_format == 'date':
                return dt.date()  # 返回datetime.date类型
            else:
                return dt.strftime(to_format)  # 返回指定格式的字符串
                
        except (ValueError, TypeError):
            # 无法解析的情况返回None
            return None

    if isinstance(date_value, str) or isinstance(date_value, datetime.datetime) or isinstance(date_value, pd.Timestamp) or isinstance(date_value, datetime.date):
        return _convert_single_date(date_value,to_format=to_format)
    elif isinstance(date_value, list):
        converted = [_convert_single_date(d,to_format=to_format) for d in date_value]
        return converted
    else:
        raise TypeError("date_value should be a string, datetime, pd.Timestamp, list or pd.Series")
    


def merge_polars_dfs(
    df1,
    df2,
    keep_schema: str = "left"  # "left" 保留df1的列结构，"right" 保留df2的列结构
    ):
    import polars as pl
    from polars import Schema, DataFrame
    """
    合并两个 Polars DataFrame,给existing新增new_df的列名,并统一成existing的列类型
    
    参数:
        df1: 第一个 DataFrame,
        df2: 第二个 DataFrame,
        keep_schema: 保留哪个DataFrame的列结构，"left" 或 "right"
    
    返回:
        合并后的 DataFrame（去重并按索引排序）
    """
    # 1. 确定基准列（按 keep_schema 保留的列）
    if keep_schema == "right":
        new_df = df2
        existing_df = df1
    else:  # keep_schema == "left"
        new_df = df1
        existing_df = df2
    
    new_columns = new_df.columns
    new_schema = new_df.schema

    
    # 2. 给另一个DataFrame补充缺失列并填充None
    existing_df = existing_df.with_columns(
        [pl.lit(None).alias(col) for col in new_columns if col not in existing_df.columns]
    ) 
    # 对齐列顺序
    exsisting_columns = existing_df.columns
    existing_schema = existing_df.schema
    new_df = new_df.select(existing_df.columns)
    
    # 3. 统一同名列的数据类型（以基准表类型为准）
    mismatched_cols = [
        col for col in new_columns 
        if existing_schema[col] != new_schema[col]
    ]
    print(f"类型不匹配的列: {mismatched_cols}")
    print(f"基准表列类型: {[ (col, existing_schema[col]) for col in mismatched_cols ]}")
    print(f"另一个表列类型: {[ (col, new_schema[col]) for col in mismatched_cols ]}")
    # new_df转换不匹配的列类型
    if mismatched_cols:
        for col in mismatched_cols:
            target_type = existing_schema[col]
            # 尝试安全转换（处理可能的类型不兼容）
            new_df = new_df.with_columns(
                pl.col(col).cast(target_type, strict=False).alias(col)
            )
    
    # 4. 拼接并去重（按全部列去重，也可自定义subset）
    combined_df = pl.concat([
        existing_df,# 确保基准表列顺序
        new_df
    ]).unique(
        subset=['code', 'trading_date'],  # 按所有列去重，如需指定列可改为 ['code', 'trading_date']
        keep='last'   # 保留后出现的记录（其他表的数据会覆盖基准表重复项）
    )
    
    return combined_df