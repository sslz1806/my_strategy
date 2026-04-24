# 需要的特征:涨停状态（未涨停,炸板,断板-三天之内有涨停），涨停描述(几天几板)，sma(均线)
import polars as pl
import pandas as pd
import numpy as np

def mark_limit_status(stock_data: pd.DataFrame,db_days=2) -> pd.DataFrame:
    """
    标记涨停状态（未涨停, 炸板, 断板, 正常涨停）
    """
    # 复制数据避免修改原数据
    stock_data = stock_data.copy()
    # 按股票代码和交易日排序
    stock_data = stock_data.sort_values(["code", "trading_date"])
    
    # 计算涨停和炸板标记
    stock_data["is_limit_up"] = (stock_data["close"] >= stock_data["limit_up"] * 0.999)
    stock_data["is_broken_limit"] = (
        (stock_data["high"] >= stock_data["limit_up"] * 0.999) & 
        (stock_data["close"] < stock_data["limit_up"] * 0.999)
    )
    
    # 断板标记：最近3天（不含当天）有涨停，且当天未涨停也未炸板
    def mark_db(group: pd.DataFrame) -> pd.DataFrame:
        is_limit_up = group["is_limit_up"].tolist()
        is_broken = group["is_broken_limit"].tolist()
        is_db = []
        limit_status_ext = []
        
        for i in range(len(is_limit_up)):
            # 取最近db_days天（不含当天）的数据
            recent_start = max(0, i - db_days)
            recent = is_limit_up[recent_start:i]
            
            # 判断是否为断板
            if not is_limit_up[i] and not is_broken[i] and any(recent):
                is_db.append(True)
                limit_status_ext.append("断板")
            else:
                is_db.append(False)
                limit_status_ext.append(None)
        
        group["is_db"] = is_db
        group["limit_status_ext"] = limit_status_ext
        return group
    
    # 按股票代码分组处理
    stock_data = stock_data.groupby("code", group_keys=False).apply(mark_db)
    
    # 综合状态判断
    stock_data["limit_status"] = np.select(
        [
            stock_data["is_limit_up"],
            stock_data["is_broken_limit"],
            stock_data["is_db"]
        ],
        [
            "涨停",
            "炸板",
            "断板"
        ],
        default="未涨停"
    )
    
    return stock_data

def mark_limit_desc(stock_data: pd.DataFrame) -> pd.DataFrame:
    """
    标记涨停描述（几天几板），允许中间有断板，直到再次未涨停为止
    """
    stock_data = stock_data.copy()
    stock_data = stock_data.sort_values(["code", "trading_date"])
    
    def calc_desc(group: pd.DataFrame) -> pd.DataFrame:
        group = group.sort_values("trading_date").reset_index(drop=True)
        is_limit_up = group["is_limit_up"].tolist()
        limit_status = group["limit_status"].tolist()
        desc_list = []
        period_start = 0  # 当前周期起始索引
        
        for i in range(len(is_limit_up)):
            total_days = i - period_start + 1
            up_days = sum(is_limit_up[period_start:i+1])
            
            if limit_status[i] in ['涨停', '炸板']:
                desc_list.append(f"{total_days}天{up_days}板")
            elif limit_status[i] == '未涨停':
                desc_list.append("未涨停")
                period_start = i + 1  # 重置周期起点
            elif limit_status[i] == '断板':
                desc_list.append('断板')
            else:
                desc_list.append(None)
        
        group["limit_desc"] = desc_list
        return group
    
    return stock_data.groupby("code", group_keys=False).apply(calc_desc)
def mark_last_limit_desc(stock_data: pd.DataFrame) -> pd.DataFrame:
    """
    向前记录最后一次涨停状态的函数
    """
    stock_data = stock_data.copy()
    stock_data = stock_data.sort_values(["code", "trading_date"])
    
    def calc_last_desc(group: pd.DataFrame) -> pd.DataFrame:
        group = group.sort_values("trading_date").reset_index(drop=True)
        limit_status = group["limit_status"].tolist()
        limit_desc = group["limit_desc"].tolist()
        last_limit_desc_list = []
        period_start = 0
        last_valid_desc = None
        
        for i in range(len(limit_status)):
            if i == 0:
                last_limit_desc_list.append(None)
                continue
            
            pre_status = limit_status[i-1]
            
            if pre_status in ['涨停', '炸板', '断板']:
                # 向前搜索最近的涨停描述
                found = False
                for j in range(i-1, period_start-1, -1):
                    if limit_status[j] == '涨停':
                        last_valid_desc = limit_desc[j]
                        found = True
                        break
                last_limit_desc_list.append(last_valid_desc if found else None)
            else:
                last_limit_desc_list.append(None)
                period_start = i -1
                last_valid_desc = None
        
        group["last_limit_desc"] = last_limit_desc_list
        return group
    
    return stock_data.groupby("code", group_keys=False).apply(calc_last_desc)

def cal_n_lowest(stock_data: pd.DataFrame, window: int = 30, include_today: bool = False) -> pd.DataFrame:
    """
    计算股票n日内的最低股价
    
    参数:
        stock_data: 包含股票数据的DataFrame，需包含"code"（股票代码）、"trading_date"（交易日）、"low"（当日最低价）列
        window: 计算最低股价的窗口大小，默认30天（即“n日”中的n）
        include_today: 是否包含当天价格，默认为False（计算“前n天”最低值，不含当天；True则计算“包含当天在内的n天”最低值）
    
    返回:
        添加了n日内最低股价列的DataFrame，新增列名为 `lowest_{window}`（如window=30时为`lowest_30`）
    """
    # 1. 确保数据按“股票代码+交易日”排序
    stock_data = stock_data.sort_values(["code", "trading_date"]).copy()
    
    # 2. 按股票分组计算滚动最低价
    if include_today:
        # 包含当天：计算当前日及之前共window天的最低价
        rolling_min = stock_data.groupby("code")["low"].rolling(window=window, min_periods=1).min()
    else:
        # 不包含当天：先计算包含当天的窗口最小值，再整体后移1天
        rolling_min = stock_data.groupby("code")["low"].rolling(window=window, min_periods=1).min().shift(1)
    
    # 3. 添加结果列并返回
    col_name = f"lowest_{window}"
    stock_data[col_name] = rolling_min.reset_index(level=0, drop=True)
    
    return stock_data

def add_sma(stock_data: pd.DataFrame, window: int =5,column: str = "close") -> pd.DataFrame:
    """
    添加简单移动平均线（SMA）列到股票数据中。
    
    参数:
        stock_data: 包含股票数据的DataFrame，需包含"code"（股票代码）、"trading_date"（交易日）、"close"（收盘价）列
        window: 移动平均线的窗口大小，默认5天（即“SMA5”）
    """
    # 1. 确保数据按“股票代码+交易日”排序
    stock_data = stock_data.sort_values(["code", "trading_date"]).copy()
    
    # 2. 按股票分组计算SMA
    sma = stock_data.groupby("code")[column].rolling(window=window, min_periods=1).mean()
    
    # 3. 添加结果列并返回
    col_name = f"sma_{window}"
    stock_data[col_name] = sma.reset_index(level=0, drop=True)
    
    return stock_data