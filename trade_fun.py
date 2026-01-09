import polars as pl
import plotly
import math
from scipy import stats
import datetime as dt
import numpy as np
import pandas as pd
import fun 
from fun import *
from tqdm import tqdm  # 导入tqdm
from multiprocessing import Pool  # 导入线程池（而非进程池）
from multiprocessing import cpu_count  # 仍可用于计算线程数
from concurrent.futures import ProcessPoolExecutor,as_completed  # 导入线程池执行器
from concurrent.futures import ThreadPoolExecutor

def calculate_time_ratio(current_time):
    """
    计算当前时间在交易日中的比例（转换为天数）
    交易日时间：9:30-11:30（2小时）和13:00-15:00（2小时），共4小时
    返回值：当天持有时间占比（0-1之间）
    """
    import datetime
    from datetime import time
    if not current_time:
        return 0.0
        
    # 上午交易时段
    if time(9, 30) <= current_time <= time(11, 30):
        minutes = (current_time.hour - 9) * 60 + (current_time.minute - 30)
    # 下午交易时段
    elif time(13, 0) <= current_time <= time(15, 0):
        # 加上上午的120分钟，再计算下午的分钟数
        minutes = 120 + (current_time.hour - 13) * 60 + current_time.minute
    # 非交易时间
    else:
        return 0.0
        
    # 转换为天数比例（4小时 = 240分钟 = 0.5天）
    return minutes / 240

# 保持原有的trade函数不变，但修改参数以适应并行处理
def trade(code_list,trade_date:dt.date,fee_rate = 0.005,need_adj=True,stop_loss_pct=0.09):
    """
    并行处理的单个交易任务
    params包含: code_list, trade_date, interval参数等
    """
    # 获取股票数据（10天的分钟线和日线数据）
    import time as Time
    start_process_time = Time.time()
    from datetime import datetime, timedelta, time
    start_date = trade_date
    end_date = start_date + timedelta(days=15)

    # 1. 获取数据
    try:
        # 获取分钟线数据
        stock_data = read_day_data(start_date,end_date,code_list,file_path='ts_stock_all_data')
        
        # 获取日线数据（用于补充pre_close, limit_up等字段）
        mins_data = read_min_data(start_date,end_date,code_list)
        
        # 获取复权因子数据
        if need_adj:
            adj_data = read_day_data(start_date,end_date,code_list,file_path='ts_adj')
            stock_data = stock_data.join(
                adj_data[['trading_date', 'code', 'adj_factor']],
                on=['trading_date', 'code'],
                how='left',
            )
            # 改名为adj
            stock_data = stock_data.rename({'adj_factor': 'adj'})

        # 合并日线字段到分钟线数据
        fields_to_merge = ['pre_close', 'limit_up', 'limit_down']
        if need_adj:
            fields_to_merge+=['adj'] 
        mins_data = mins_data.join(
            stock_data[['trading_date', 'code'] + fields_to_merge],
            on=['trading_date', 'code'],
            how='left',
        )
        mins_data = mins_data.drop_nulls(subset=['open', 'close', 'pre_close', 'limit_up', 'limit_down'])
        
    except Exception as e:
        logging.info(f"获取{code}数据失败: {str(e)}")
        return None
    
    # 2. 回测，买入和卖出逻辑
    result = []
    # 遍历code列表,code_data
    for code in code_list: 
        code_mins_data = mins_data.filter(pl.col("code") == code)
        # 初始化交易信息
        trade_info = {
            'code': code, 
            'buy_time': None, 
            'buy_price': None, 
            'sell_time': None, 
            'sell_price': None, 
            'profit': None,
            'holding_days': None,
            'sell_reason': None # 卖出原因
        }
        
        # 如果没有数据，直接返回
        if code_mins_data.height == 0:
            logging.info(f"获取{code}在{start_date} 到 {end_date}股票数据失败")
            return None
        
        # 获取交易日期列表,遍历code_data中的交易日期
        trading_date_list = sorted(code_mins_data['trading_date'].unique().to_list())
        if not trading_date_list:
            return None
        
        buy_date = trading_date_list[0]
        
        # 2.1 买入逻辑（9:30的开盘价）
        # buy_data = code_mins_data.filter(
        #     (pl.col("trading_date") == buy_date) &
        #     (pl.col("datetime").dt.hour() == 9) &
        #     (pl.col("datetime").dt.minute() == 30)
        # )
        # buy_price = buy_data['open'].to_list()[0] if buy_data.height > 0 else None
        # buy_adj = buy_data['adj'].to_list()[0] if 'adj' in buy_data.columns else 1
        # trade_info['buy_time'] = datetime.combine(buy_date, time(9, 30))
        # trade_info['buy_price'] = buy_price
        buy_data = code_mins_data.filter(
            (pl.col("trading_date") == buy_date) &
            (pl.col("datetime").dt.hour() == 9) &
            (pl.col("datetime").dt.minute() == 30)
        )
        if buy_data.height == 0:
            logging.info(f"{code}在{buy_date}日买入时间段无交易数据，跳过该股票")
            continue
        open_price = buy_data['open'].to_list()[0]
        pre_close = buy_data['pre_close'].to_list()[0]
        # 如果与昨日相比跌破-7%则不买入
        # if (open_price/pre_close -1)*100 <= -8:
        #     continue
        buy_price = open_price
        buy_adj = buy_data['adj'].to_list()[0] if 'adj' in buy_data.columns else 1
        trade_info['buy_time'] = datetime.combine(buy_date, time(9, 30))
        trade_info['buy_price'] = buy_price

        buy_date_index = trading_date_list.index(buy_date) # 记录买入日期索引，计算持有天数用
        
        # 2.2 卖出逻辑(止损卖出,未涨停卖出)
        target_time = [time(9, 30), time(11, 30), time(15, 00)]
        end_date = trading_date_list[-1]
        sell_triggered = False

        ## 取后一天T+1,遍历信号票的每一天特定时间检测卖出
        for i,single_date in enumerate(trading_date_list[1:]): 
            #single_date_str = str(single_date)
            full_days = (i+1)
            code_daily_data = code_mins_data.filter(
                (pl.col("trading_date") == single_date) &
                (pl.col("datetime").dt.time().is_in(target_time))
            )
            
            if code_daily_data.height == 0:
                continue
                
            # 获取当日的涨停价和前收盘价
            limit_up = code_daily_data['limit_up'].to_list()[0]
            pre_close = code_daily_data['pre_close'].to_list()[0]
            limit_down  = code_daily_data['limit_down'].to_list()[0]
            
            # 检查每个目标时间点
            for row in code_daily_data.iter_rows(named=True):
                current_price = row['close']
                current_time = row['datetime'].time()
                open_pct = (row['open']/row['pre_close']-1) *100
                # 计算当天持有时间比例并得到总持有天数
                same_day_ratio = calculate_time_ratio(current_time)
                total_holding_days = round(full_days + same_day_ratio, 2)
                adj = row['adj'] if 'adj' in row.keys() else 1

                # 1. 9:30 第二天开盘大幅低开
                if open_pct <=-7 and current_time==target_time[0]:
                    # 如果跌停并且最高价也跌停则无法卖出
                    if current_price<=limit_down: 
                        if row['high']<=limit_down:
                            continue
                    # 否则可以卖出
                    buy_price_fee = trade_info['buy_price']* buy_adj * (1 + fee_rate)
                    sell_price_fee = current_price* adj * (1 - fee_rate)
                    profit = (sell_price_fee - buy_price_fee) / buy_price_fee * 100
                    trade_info.update({
                        'sell_time': datetime.combine(row['trading_date'],current_time),
                        'sell_price': current_price,
                        'profit': profit,
                        'holding_days':total_holding_days,
                        'sell_reason': '大幅低开卖出'
                    })
                    sell_triggered = True
                    break

                # 跌停无法卖出
                if current_price<=limit_down: 
                    continue
                
                # 2. 止损条件：价格低于买入价的90%
                if trade_info['buy_price'] and current_price <= trade_info['buy_price'] * (1-stop_loss_pct):
                    buy_price_fee = trade_info['buy_price']* buy_adj * (1 + fee_rate)
                    sell_price_fee = current_price* adj * (1 - fee_rate)
                    profit = (sell_price_fee - buy_price_fee) / buy_price_fee * 100
                    trade_info.update({
                        'sell_time': datetime.combine(row['trading_date'],current_time),
                        'sell_price': current_price,
                        'profit': profit,
                        'holding_days':total_holding_days,
                        'sell_reason': '止损卖出'
                    })
                    sell_triggered = True
                    break
                    
                # 3. 止盈/正常卖出条件：未涨停且在目标时间点
                if current_price < limit_up * 0.97 and current_time in target_time[1:]:
                    buy_price_fee = trade_info['buy_price']* buy_adj * (1 + fee_rate)
                    sell_price_fee = current_price* adj * (1 - fee_rate)
                    profit = (sell_price_fee - buy_price_fee) / buy_price_fee * 100
                    trade_info.update({
                        'sell_time': datetime.combine(row['trading_date'],current_time),
                        'sell_price': current_price,
                        'profit': profit,
                        'holding_days':total_holding_days,
                        'sell_reason': '未涨停卖出'
                    })
                    sell_triggered = True
                    break


            if sell_triggered: # 已经完成卖出逻辑,无需循环
                break

        # 如果未触发卖出条件，在最后一天收盘价卖出
        if not trade_info['sell_time']:
            final_day_data = code_mins_data.filter(pl.col("trading_date") == end_date)
            adj = final_day_data['adj'][0] if 'adj' in final_day_data.columns else 1
            if final_day_data.height > 0:
                sell_price = final_day_data['close'].to_list()[-1]
                final_time = final_day_data['datetime'].to_list()[-1].time()

                # 计算持有天数
                full_days = trading_date_list.index(end_date) - buy_date_index
                same_day_ratio = calculate_time_ratio(final_time)
                total_holding_days = round(full_days + same_day_ratio, 2)
                if trade_info['buy_price']:
                    buy_price_fee = trade_info['buy_price']* buy_adj * (1 + fee_rate)
                    sell_price_fee = sell_price * adj * (1 - fee_rate)
                    profit = (sell_price_fee - buy_price_fee) / buy_price_fee * 100
                    trade_info['profit'] = profit
                trade_info.update({
                    'sell_time':datetime.combine(end_date,final_time),
                    'sell_price': sell_price,
                    'holding_days': total_holding_days,
                    'sell_reason': '最后卖出'
                })

        result.append(trade_info)
    
    duration = Time.time() - start_process_time
    #logging.info(f"完成回测日期: {trade_date}, 股票数量: {len(code_list)}, 用时: {duration:.2f}秒")
    return result


def cal_trade_info(信号文件:pd, trade_fun=trade,start_date: str = None, end_date: str = None):
    """
    批量处理信号文件，多进程并行回测
    注意：不再传trade_fun参数，直接调用修正后的trade函数（避免参数传递冗余）

    返回:
    result_df:回测的结果文件 - pd
    merged_df:回测信号+结果的详细文件 - pd | pl
    """
    import datetime as dt
    logging = get_logger()
    if isinstance(start_date, str):
        start_date = dt.datetime.strptime(start_date, "%Y-%m-%d").date()
    if isinstance(end_date, str):
        end_date = dt.datetime.strptime(end_date, "%Y-%m-%d").date()
    # 1. 筛选回测日期范围（确保信号文件的trade_date是字符串）
    # 首先将列转换为日期类型
        # 1. 判断数据类型并统一处理日期列
    is_polars = isinstance(信号文件, pl.DataFrame)
    date_column = 'trading_date' if 'trading_date' in 信号文件.columns else 'trade_date'

    if is_polars:
        # Polars处理逻辑
        # 转换日期为标准字符串格式
        信号文件 = 信号文件.with_columns(
            pl.col(date_column)
            .cast(pl.Date)
            .alias("trading_date")
        )
        # 获取唯一日期列表
        trade_date_list = 信号文件.select(pl.col(date_column).unique()).to_series().to_list()
        # 确定股票代码列名
        stock_code_col = 'symbol' if 'symbol' in 信号文件.columns else 'code'
        # 排序
        信号文件 = 信号文件.sort([date_column, stock_code_col])
    else:
        # Pandas处理逻辑
        信号文件[date_column] = pd.to_datetime(信号文件[date_column]).dt.strftime('%Y-%m-%d')
        trade_date_list = 信号文件[date_column].unique().tolist()
        stock_code_col = 'symbol' if 'symbol' in 信号文件.columns else 'code'
        信号文件 = 信号文件.sort_values([date_column, stock_code_col])
    
    # 2. 筛选日期范围
    if start_date:
        trade_date_list = [d for d in trade_date_list if d >= start_date]
    if end_date:
        trade_date_list = [d for d in trade_date_list if d <= end_date]
    if not trade_date_list:
        logging.info("没有符合条件的回测日期")
        return (pd.DataFrame() if not is_polars else pl.DataFrame(), 
                pd.DataFrame() if not is_polars else pl.DataFrame())
    trade_date_list.sort()
    
    # 3. 准备并行任务
    """
    tasks = []
    results=[]
    for trade_date in trade_date_list:
        if is_polars:
            当日信号 = 信号文件.filter(pl.col(date_column) == trade_date)
            股票列表 = 当日信号.select(pl.col(stock_code_col)).to_series().to_list()
        else:
            当日信号 = 信号文件[信号文件[date_column] == trade_date]
            股票列表 = 当日信号[stock_code_col].tolist()
        
        if 股票列表:
            tasks.append((股票列表, trade_date))
            logging.info(f"准备回测日期: {trade_date}, 股票数量: {len(股票列表)}")
            # task.append((股票列表,trade_date))
            result = trade_fun(股票列表,trade_date)
            if result:
                results.append(result)  # 逐个执行任务并收集结果
    """

    # 4. 准备每日任务的信号数据（提前按日期拆分，避免多进程中重复筛选）
    # 构建日期到当日信号的映射（键：日期，值：当日信号的股票列表）
    date_to_stocks = {}
    results=[]
    for trade_date in trade_date_list:
        if is_polars:
            当日信号 = 信号文件.filter(pl.col("trading_date") == trade_date)
            股票列表 = 当日信号.select(pl.col(stock_code_col)).to_series().to_list()
        else:
            当日信号 = 信号文件[信号文件[date_column] == trade_date]
            股票列表 = 当日信号[stock_code_col].tolist()
        if 股票列表:
            date_to_stocks[trade_date] = 股票列表
            #logging.info(f"准备回测日期: {trade_date}, 股票数量: {len(股票列表)}")
            # import time
            # start_time = time.time()
            # result = trade_fun(股票列表,trade_date) #单线程
            # if result:
            #     results.append(result)  # 逐个执行任务并收集结果 
            # duration = time.time() - start_time
            # logging.info(f"完成回测日期: {trade_date}, 股票数量: {len(股票列表)}, 用时: {duration:.2f}秒")

    tasks = [(stocks, date) for date, stocks in date_to_stocks.items()]
    # # 计算最大进程数（避免占用全部CPU）
    max_workers = max(1, int(cpu_count()*2))
    logging.info(f"使用{max_workers}个进程并行处理，共{len(tasks)}个日期任务")

    # # 多进程执行（用partial固定trade_fun的参数结构，确保和任务格式匹配）
    # # 注意：Windows必须在if __name__ == "__main__"中执行，否则会报错
    # with Pool(processes=max_workers) as pool:
    #     # 每个任务是(股票列表, 日期)，trade_fun需要接收这两个参数
    #     results = pool.starmap(trade_fun, tasks)  # starmap用于传递多参数元组
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_date = {executor.submit(trade_fun, stocks, date): date for date, stocks in date_to_stocks.items()}
        for future in tqdm(
            as_completed(future_to_date),
            total=int(len(tasks)),
            desc="回测任务进度",
            unit="个日期",
            ncols=80  # 进度条宽度，可根据需要调整
            ):
            date = future_to_date[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
                #logging.info(f"完成回测日期: {date},股票数量: {len(date_to_stocks[date])}")
            except Exception as e:
                tqdm.write(f"❌ 回测日期 {date} 时发生错误: {str(e)} ⚠️")

    logging.info(f"所有回测任务完成，共处理{len(results)}个日期的结果")

    # 4. 展平结果列表，转换为对应类型的DataFrame
    valid_results = []
    for date_result in results:
        if date_result:
            valid_results.extend(date_result)  # 展平：[[a,b],[c,d]] → [a,b,c,d]
    
    if is_polars:
        result_df = pl.DataFrame(valid_results)
        # 处理日期和列名
        if 'buy_time' in result_df.columns:
            result_df = result_df.with_columns(
                pl.col("buy_time")
                .dt.date()  # 截取前10个字符，即"YYYY-MM-DD"部分
                .alias("trading_date")
            )
        if 'code' in result_df.columns and stock_code_col != 'code':
            result_df = result_df.rename({"code": stock_code_col})
        # 合并
        merged_df = 信号文件.join(result_df, on=[stock_code_col, 'trading_date'], how='outer')
    else:
        result_df = pd.DataFrame(valid_results)
        # 处理日期和列名
        if 'buy_time' in result_df.columns:
            result_df['trading_date'] = pd.to_datetime(result_df['buy_time']).dt.date()
        if 'code' in result_df.columns and stock_code_col != 'code':
            result_df = result_df.rename(columns={'code': stock_code_col})
        # 合并
        merged_df = pd.merge(信号文件, result_df, on=[stock_code_col, 'trading_date'], how='outer')

    return  pd.DataFrame(valid_results), merged_df

# 取sell_date为前n个窗口的股票表现与平均水平进行对比。高于平均水平并且昨日的股票触及跌停达到一定比率，则仓位调整为min_weight
def adjust_weight_by_near_n(回测结果, max_weight=0.4, min_weight=0, window=20, down_limit_ratio=0.35,win_rate_threshold=0.43, profit_loss_ratio_threshold=1.3):
    """
    adjust_weight_by_near_n 调整仓位逻辑:
    
    :param 回测结果: 回测结果pl.DataFrame或pd.DataFrame，包含交易记录
    :param max_weight: 每日的最大仓位
    :param min_weight: 每日的最小仓位
    :param window: 近期窗口
    :param down_limit_ratio: 跌停比率阈值
    :param win_rate_threshold: 胜率阈值
    :param profit_loss_ratio_threshold: 盈亏比阈值
    """
    import polars as pl
    if isinstance(回测结果, pd.DataFrame):
        回测结果 = pl.from_pandas(回测结果)
    
    # 按trading_date排序
    回测结果 = 回测结果.sort('trading_date')
    # 获取唯一的交易日期列表
    trading_dates = 回测结果['trading_date'].unique().to_list()
    # 利用sell_time获取出sell_date.（sell_time为datetime）
    回测结果 = 回测结果.with_columns(
        pl.col('sell_time').dt.date().alias('sell_date')
    )
    
    # 生成「日期-权重」映射表（每个日期对应一个权重）
    date_weight = []
    # 1.按照sell_date筛选出前window天的数据计算胜率和盈亏比 2.计算昨日信号股票的跌停比率 3.根据两个条件判断调整仓位
    for i, current_date in enumerate(trading_dates):
        # 前window天数据不足时，默认最大仓位
        if i < window:
            date_weight.append((current_date, max_weight))
            continue
        
        # 取当前日期之前的window天作为统计窗口
        window_dates = trading_dates[i - window : i]
        window_data = 回测结果.filter(pl.col('sell_date').is_in(window_dates))
        
        # ---------------------- 1. 计算前window天的胜率和盈亏比 ----------------------
        # 处理窗口内无数据的情况
        total_in_window = window_data.height
        if total_in_window == 0:
            win_rate = 0.0
            profit_loss_ratio = 0.0
        else:
            # 拆分盈利/亏损交易
            win_trades = window_data.filter(pl.col('profit') > 0)
            loss_trades = window_data.filter(pl.col('profit') < 0)
            
            # 计算胜率（盈利交易数/总交易数）
            win_count = win_trades.height
            win_rate = win_count / total_in_window
            
            # 计算盈亏比（平均盈利 / 平均亏损的绝对值）
            avg_win = win_trades['profit'].mean() if win_count > 0 else 0.0
            avg_loss_abs = abs(loss_trades['profit'].mean()) if loss_trades.height > 0 else 0.0
            
            # 处理极端情况：无亏损交易时盈亏比设为极大值（视为达标）；无盈利交易时设为0
            if avg_loss_abs == 0:
                profit_loss_ratio = float('inf')
            elif avg_win == 0:
                profit_loss_ratio = 0.0
            else:
                profit_loss_ratio = avg_win / avg_loss_abs
        
        # 2. 计算昨日的跌停比率
        previous_date = trading_dates[i - 1]
        previous_data = 回测结果.filter(pl.col('trading_date') == previous_date)
        
        # 处理昨日无数据的情况
        total_previous = previous_data.height
        if total_previous == 0:
            down_ratio = 0  # 无数据时视为跌停比率为0
        else:
            down_count = previous_data.filter(pl.col('touch_limit_down') == True).height
            down_ratio = down_count / total_previous
        
        # ---------------------- 3. 双条件判断：调整仓位 ----------------------
        # 条件1：前window天胜率≥阈值 且 盈亏比≥阈值（视为"高于平均水平"）
        # 条件2：昨日跌停比率≥阈值
        if (win_rate >= win_rate_threshold) and (profit_loss_ratio >= profit_loss_ratio_threshold) and (down_ratio >= down_limit_ratio):
            date_weight.append((current_date, min_weight))
        # elif (win_rate<=0.4 and profit_loss_ratio<=1.4 and down_ratio>=down_limit_ratio):
        #     date_weight.append((current_date, 0))
        else:
            date_weight.append((current_date, max_weight))
    
    # 合并权重数据并清理临时列
    weight_df = pl.DataFrame(date_weight, schema=['trading_date', 'weight'])
    回测结果 = 回测结果.join(weight_df, on='trading_date', how='left')
    
    # 4.汇报调整正确率以及调整绩效(调整后减亏比例=(max_weight-min_weight)*(-profit).mean() )
    total_adjusted = weight_df.filter(pl.col('weight') == min_weight).height # 调整天数
    merged_with_weight = 回测结果.join(weight_df, on='trading_date', how='left')
    if total_adjusted == 0:
        adjust_success_rate = 0.0
        total_profit_loss_improvement = 0.0
    else:
        adjusted_trades = merged_with_weight.filter(pl.col('weight') == min_weight)
        losing_trades = adjusted_trades.filter(pl.col('profit') < 0)
        total_losing_trades = losing_trades.height
        if total_losing_trades == 0:
            adjust_success_rate = 1.0  # 全部调整成功
            total_profit_loss_improvement = 0.0
        else:
            # 亏损交易调成min_weight即为正确调整
            adjust_success_rate = total_losing_trades / adjusted_trades.height*100
            # 计算总的改善比例
            total_profit_loss_improvement = (max_weight - min_weight) * (-losing_trades['profit']).mean() 
    logging.info(f"总调整天数: {total_adjusted}, 调整正确率: {adjust_success_rate:.2f}%, 预计平均亏损改善: {total_profit_loss_improvement:.4f}%")
    return 回测结果.drop('sell_date')

# 动态调整仓位函数
def mark_weight(回测结果, max_weight=0.4, min_weight=0.3):
    """
    遍历回测结果,根据昨日的持仓情况动态调整当日的持仓
    max_weight: 每日购买的最大仓位比例
    min_weight: 每日购买的最小仓位比例 
    调整仓位思路:
    遍历回测结果,查看昨日策略股票的涨跌状况。如果触及跌停数目超过昨日策略总票数的50%,则今日仓位降低为min_weight,否则为max_weight
    """
    import polars as pl
    if isinstance(回测结果, pd.DataFrame):
        回测结果 = pl.from_pandas(回测结果)
    
    # 按trading_date排序
    回测结果 = 回测结果.sort('trading_date')
    # 获取唯一的交易日期列表
    trading_dates = 回测结果['trading_date'].unique().to_list()
    
    # 生成「日期-权重」映射表（每个日期对应一个权重）
    date_weight = []
    for i, current_date in enumerate(trading_dates):
        if i == 0:
            # 第一天，默认最大仓位
            date_weight.append((current_date, max_weight))
            continue
        
        previous_date = trading_dates[i - 1]
        previous_day_data = 回测结果.filter(pl.col('trading_date') == previous_date)
        
        total_stocks = previous_day_data.height
        if total_stocks == 0:
            date_weight.append((current_date, max_weight))
            continue
        
        # 计算昨日触及跌停的股票数量
        跌停_stocks = previous_day_data.filter(pl.col('touch_limit_down') == True).height
        跌停_ratio = 跌停_stocks / total_stocks
        
        # 根据跌停比例调整今日仓位
        if 跌停_ratio > 0.4:
            date_weight.append((current_date, min_weight))
        else:
            date_weight.append((current_date, max_weight))
    
    # 将日期-权重映射转为DataFrame
    weight_df = pl.DataFrame(date_weight, schema=['trading_date', 'weight'])
    
    # 通过trading_date关联，给每天的所有股票赋当天的权重
    回测结果 = 回测结果.join(weight_df, on='trading_date', how='left')
    
    return 回测结果

#%% 回测结果汇报函数
def report_backtest_full(
    result_df: pd.DataFrame,
    start_date,
    end_date,
    profit_col: str = 'profit',
    buy_date_col:str = 'buy_time',
    sell_date_col: str = 'buy_time',
    holding_days_col:str = 'holding_days',
    benchmark_code: str = "399300.SZ",
    risk_free_rate: float = 0.02,
    return_method = 'compound',
    plot = True,
    second_y = True
):
    """
    result_df 交割单:包括策略的 1.卖出信息 2.利润信息 的交割单
    回测结果汇报函数（含净值曲线、最大回撤、夏普比率、超额收益等）
    """
    # 忽略

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import tinyshare as tns
    from mapping import convert_code_format,clean_stocks_data
    from stock_api import stock_api
    from fun import get_logger
    logging = get_logger(log_file='回测.log',inherit=False)
    ts_token = 'YzAEH11Yc7jZCHjeJa63fnbpSt3k9Je3GvWn0390oiBKO95bVJjP7u5L34e2ff6b'
    ts =tns.pro_api(ts_token)

    # 1. 策略净值曲线计算
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    result_df[sell_date_col] = pd.to_datetime(result_df[sell_date_col])
    result_df = result_df[
        (result_df[sell_date_col] >= start_date) & 
        (result_df[sell_date_col] <= end_date)
    ]
    result_df = result_df.sort_values(sell_date_col).reset_index(drop=True)
    result_df['sell_date_date'] = pd.to_datetime(result_df[sell_date_col]).dt.date
    result_df['buy_date_date'] = pd.to_datetime(result_df[buy_date_col]).dt.date # 提取买入日期（仅日期部分）
    daily_returns = result_df.groupby('sell_date_date')[profit_col].mean() / 100  # 转为小数
    net_values = (1 + daily_returns).cumprod()
    if return_method!='compound':
        net_values = 1 + daily_returns.cumsum()  # 单利：初始净值1 + 每日收益累计和
    strategy_curve = net_values.copy()
    #strategy_curve.index = pd.to_datetime(strategy_curve.index).normalize()

    # 2. 获取指数净值曲线
    api = stock_api()
    index_data = api.gm_get_index_day_data(index_code='SHSE.000001',start_date=start_date.strftime('%Y-%m-%d'),end_date=end_date.strftime('%Y-%m-%d'))
    # index_data = ts.index_daily(ts_code=convert_code_format(benchmark_code,format='suffix'), start_date=start_date.strftime('%Y%m%d'), end_date=end_date.strftime('%Y%m%d'))
    # index_data = clean_stocks_data(index_data)
    index_df = index_data
    index_df['trading_date_date'] = pd.to_datetime(index_df['trading_date']).dt.date
    index_df = index_df.sort_values('trading_date_date').reset_index(drop=True)
    if not index_df.empty:
        # if 'pct' not in index_df.columns:
        #     index_df['pct'] = index_df['close'].pct_change()
        # index_df['net_value'] = (1 + index_df['pct']/100).cumprod()
        # # 将第一个交易日的净值设为1
        # index_df.loc[index_df.index[0], 'net_value'] = 1

        index_df['net_value'] = index_df['close'] / index_df['close'].iloc[0]
        index_curve = index_df.set_index('trading_date_date')['net_value']
    else:
        raise ValueError("未获取到有效的指数数据")

    # 3. 对齐日期
    strategy_curve = strategy_curve.sort_index()
    index_curve = index_curve.sort_index()
    strategy_curve = strategy_curve.reindex(index_curve.index, method='ffill').fillna(1)

    # 4. 核心指标
    total_return = strategy_curve.iloc[-1] - 1 if len(strategy_curve) > 0 else 0
    if len(strategy_curve) >= 2:
        first_date = strategy_curve.index[0]
        last_date = strategy_curve.index[-1]
        total_days = (last_date - first_date).days
        years = total_days / 365
    else:
        years = 0
    annualized_return = (strategy_curve.iloc[-1]) ** (1 / years) - 1 if years > 0 and strategy_curve.iloc[-1] > 0 else 0

    # 最大回撤
    roll_max = strategy_curve.cummax()
    drawdown = (strategy_curve - roll_max) / roll_max
    max_drawdown = drawdown.min()
    # 找到最大回撤的开始和结束时间
    max_drawdown_end = drawdown.idxmin()
    # 找到最大回撤开始时间（即之前的最高点）
    max_drawdown_start = roll_max.loc[:max_drawdown_end].idxmax()

    # 夏普比率
    daily_ret = strategy_curve.pct_change().dropna()
    daily_drawdown = daily_ret.where(daily_ret < 0, 0)
    rf_daily = risk_free_rate / 252
    excess_daily = daily_ret - rf_daily
    sharpe_ratio = (excess_daily.mean() / excess_daily.std()) * np.sqrt(252) if excess_daily.std() > 0 else 0

    # 超额收益
    bench_years = (index_curve.index[-1] - index_curve.index[0]).days / 365 if len(index_curve) > 1 else 0
    bench_annual = (index_curve.iloc[-1]) ** (1 / bench_years) - 1 if bench_years > 0 else 0
    excess_return = annualized_return - bench_annual

    # 胜率和盈亏比
    # 筛选出结果不等于0的记录（排除盈亏平衡的情况）
    non_zero_df = result_df[result_df[profit_col] != 0]

    # 计算赢率：大于0的笔数 / 不等于0的笔数
    win_rate = (non_zero_df[profit_col] > 0).mean()
    avg_win = result_df[result_df[profit_col] > 0][profit_col].mean()
    avg_loss = abs(result_df[result_df[profit_col] < 0][profit_col].mean())
    profit_loss_ratio = avg_win / avg_loss if avg_loss != 0 else float('inf')

    # 平均开仓个数
    daily_buy_count = result_df.groupby('buy_date_date')['code'].nunique()
    # 计算平均值（排除无买入的日期，仅统计有交易的日期）
    avg_daily_buy_count = daily_buy_count.mean() if not daily_buy_count.empty else 0

    # 平均持仓天数
    valid_holding_days = result_df[
        result_df[holding_days_col].notna() &  # 排除空值
        (result_df[holding_days_col] > 0)      # 排除0或负数（异常数据）
    ][holding_days_col]
    avg_holding_days = valid_holding_days.mean() if not valid_holding_days.empty else 0

    # 8. 将结果整理成DataFrame并返回
    metrics_df = pd.DataFrame({
        '指标名称': [
            '回测开始日期', '回测结束日期', '策略胜率', '策略盈亏比',
            '每单位风险期望收益', '策略总收益率', '策略年化收益率',
            '最大回撤', '最大回撤开始日期', '最大回撤结束日期',
            '夏普比率', '策略超额年化收益率',
            '最终净值','每日平均买入股票个数', '平均持仓天数',
        ],
        '指标值': [
            first_date, last_date, f"{win_rate:.2%}",
            f"{profit_loss_ratio:.2f}", f"{win_rate*(profit_loss_ratio+1) - 1 :.4f}",
            f"{total_return:.2%}", f"{annualized_return:.2%}", f"{max_drawdown:.2%}",
            max_drawdown_start, max_drawdown_end,
            f"{sharpe_ratio:.2f}", f"{excess_return:.2%}", f"{strategy_curve.iloc[-1]:.4f}",
            f"{avg_daily_buy_count:.2f}", f"{avg_holding_days:.2f} 天"
        ]
    })

    if not plot:
        return metrics_df
    # 5. 输出结果
    logging.info(
        f"\n回测时间:{strategy_curve.index[0]} - {strategy_curve.index[-1]}\n"
        f"策略胜率: {win_rate:.2%}\n"
        f"策略盈亏比: {profit_loss_ratio:.2f}\n"
        f"每日平均开仓个数: {avg_daily_buy_count:.2f}\n"
        f"平均持仓天数: {avg_holding_days:.2f} 天\n"
        f"每单位风险期望收益:{win_rate*(profit_loss_ratio+1) -1 :.4f}\n"
        f"策略总收益率: {total_return:.2%}\n"
        f"策略年化收益率: {annualized_return:.2%}\n"
        f"最大回撤: {max_drawdown:.2%}\n"
        f"最大回撤阶段: {max_drawdown_start} 至 {max_drawdown_end}\n"
        f"夏普比率: {sharpe_ratio:.2f}\n"
        f"策略超额年化收益率: {excess_return:.2%}\n"
        f"最终净值: {strategy_curve.iloc[-1]:.4f}"
    )

    
    # 6. 绘制净值曲线
    plt.figure(figsize=(14, 7))
    plt.plot(strategy_curve.index, strategy_curve.values, label='Strategy Net Value')
    plt.plot(index_curve.index, index_curve.values, label='Index Net Value')
    plt.xlabel('Date')
    plt.ylabel('Net Value')
    plt.title('Strategy Net Value vs Index Net Value')
    plt.legend()
    plt.grid()
    plt.show()
    
    # 7. 使用plotly绘制可交互净值曲线（新增回撤直方图）
    # 创建图形 - 修改为3个子图：净值曲线、收益直方图、回撤直方图
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=('净值曲线对比', '策略每日收益率', '策略每日回撤'),
        specs=[
            [{"secondary_y": True}],  # 净值曲线（双Y轴）
            [{"secondary_y": False}], # 收益直方图
            [{"secondary_y": False}]  # 回撤直方图
        ],
        row_heights=[0.5, 0.25, 0.13]  # 调整子图高度比例
    )

    # 7.1 添加策略净值曲线（左Y轴）
    fig.add_trace(
        go.Scatter(
            x=strategy_curve.index, 
            y=strategy_curve.values, 
            name='策略净值',
            line=dict(color='#1f77b4', width=2),
            hovertemplate='日期: %{x}<br>策略净值: %{y:.4f}<extra></extra>'
        ),
        row=1, col=1,
        secondary_y=False
    )

    # 7.2 添加指数净值曲线（右Y轴）
    fig.add_trace(
        go.Scatter(
            x=index_curve.index, 
            y=index_curve.values, 
            name='指数净值',
            line=dict(color='#ff7f0e', width=2),
            hovertemplate='日期: %{x}<br>指数净值: %{y:.4f}<extra></extra>'
        ),
        row=1, col=1,
        secondary_y=second_y
    )

    # 7.3 添加每日收益率直方图
    # 准备收益率数据和颜色
    ret_colors = ['#d62728' if x > 0 else '#2ca02c' for x in daily_ret.values]
    
    fig.add_trace(
        go.Bar(
            x=daily_ret.index, 
            y=daily_ret.values, 
            name='策略每日收益率',
            marker_color=ret_colors,
            hovertemplate='日期: %{x}<br>收益率: %{y:.2%}<extra></extra>'
        ),
        row=2, col=1
    )

    # 7.4 添加每日回撤直方图
    fig.add_trace(
        go.Bar(
            x=daily_drawdown.index, 
            y=daily_drawdown.values, 
            name='策略每日回撤',
            marker_color='#2ca02c',  # 绿色
            hovertemplate='日期: %{x}<br>回撤率: %{y:.2%}<extra></extra>'
        ),
        row=3, col=1
    )

    # 7.5 更新布局
    fig.update_layout(
        height=800,  # 增加高度以容纳3个子图
        title_text="回测结果可视化",
        title_font=dict(size=16, weight='bold'),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=12)
        ),
        plot_bgcolor='rgba(248,248,248,1)',  # 浅灰色背景
        paper_bgcolor='white'
    )

    # 7.6 更新轴标签和样式
    # 净值曲线轴标签
    fig.update_yaxes(title_text="策略净值", secondary_y=False, row=1, col=1, 
                     title_font=dict(size=12), tickfont=dict(size=10))
    fig.update_yaxes(title_text="指数净值", secondary_y=True, row=1, col=1,
                     title_font=dict(size=12), tickfont=dict(size=10))
    
    # 收益率轴标签
    fig.update_yaxes(title_text="收益率", row=2, col=1,
                     title_font=dict(size=12), tickfont=dict(size=10),
                     tickformat='.2%')  # 百分比格式
    
    # 回撤轴标签
    fig.update_yaxes(title_text="回撤率", row=3, col=1,
                     title_font=dict(size=12), tickfont=dict(size=10),
                     tickformat='.2%')  # 百分比格式
    
    # X轴样式
    fig.update_xaxes(
        title_text="日期", row=3, col=1,
        tickfont=dict(size=10, family='Arial', color='gray'),
        #tickangle=-45,
        title_font=dict(size=12)
    )

    # 7.7 添加网格线
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200,200,200,0.2)', row=1)
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200,200,200,0.2)', row=2)
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200,200,200,0.2)', row=3)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200,200,200,0.2)', row=1)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200,200,200,0.2)', row=2)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200,200,200,0.2)', row=3)

    # 显示图形
    fig.show()
    
    return metrics_df
