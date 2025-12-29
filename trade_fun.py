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
def trade(code_list,trade_date:dt.date,fee_rate = 0.002,need_adj=True,stop_loss_pct=0.09):
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

                # 跌停无法卖出
                if current_price<=limit_down:
                    continue
                
                # 1. 止损条件：价格低于买入价的90%
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
                    
                # 2. 止盈/正常卖出条件：未涨停且在目标时间点
                if current_price < limit_up * 0.975 and current_time in target_time[1:]:
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

                # 3. 第二天开盘大幅低开
                if open_pct <=-7 and current_time==target_time[0]:
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
def adjust_weight_by_near_n(回测结果, max_weight=0.4, min_weight=0, window=20, down_limit_ratio=0.35,win_rate_threshold=0.43, profit_loss_ratio_threshold=1.4):
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