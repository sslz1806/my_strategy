# 定义因子类(因子分析,通用计算函数),以及一些功能函数
import polars as pl
import statsmodels.api as sm
import sys
sys.path.append('D://桌面/策略')
from stock_api import *
import pandas as pd
import numpy as np
api = stock_api()

def cal_next_return(stock_data: pl.DataFrame, days=5) -> pl.DataFrame:
    stock_data = stock_data.sort(['code','trading_date'])
    stock_data = stock_data.with_columns([
        ((pl.col('close').shift(-days) - pl.col('close')) / pl.col('close')*100).over('code').alias(f'return_{days}d')
    ])
    return stock_data


def ols_neutralize(group: pl.DataFrame, y_column: str, x_columns: list) -> pl.DataFrame:
    """
    简化版OLS中性化（仅处理None/inf缺失值，x列问题直接报错）
    逻辑：过滤None/inf→回归（x列重复等问题直接报错）→残差对齐
    """
    # 1. 过滤：y/x列无None/inf的有效样本
    valid_cols = [y_column] + x_columns
    valid_group = group.drop_nulls(valid_cols).filter(
        # 列表推导式生成所有条件，再用 & 合并
        *[pl.col(col).is_finite() for col in valid_cols]
    )
        
    min_samples = len(x_columns) + 1
    if len(valid_group) < min_samples:
        # 有效样本不足，直接返回null残差列
        return group.with_columns(
            pl.lit(None, pl.Float64).alias(f'{y_column}_neutralized')
        )
    
    # 2. 有效样本回归（x列重复/矩阵奇异等问题直接抛错，不静默返回null）
    try:
        X = sm.add_constant(valid_group[x_columns].to_numpy(), has_constant='add')
        residuals = sm.OLS(valid_group[y_column].to_numpy(), X).fit(disp=0).resid
    except Exception as e:
        # 直接抛出错误（而非返回null），便于定位问题（如x列重复、多重共线性）
        raise RuntimeError(f"OLS回归失败：{str(e)[:100]}") from e
    
    # 3. 残差对齐原始分组（有效样本填残差，无效填null）
    # 改用code唯一匹配（比全列匹配更可靠，避免列过多导致的匹配问题）
    valid_group_with_resid = valid_group.with_columns(
        pl.Series(residuals).alias(f'{y_column}_neutralized')
    )
    
    return group.join(
        valid_group_with_resid[['code', f'{y_column}_neutralized']],
        on='code',
        how='left'
    )[group.columns + [f'{y_column}_neutralized']]  # 保持原始列顺序


def analyze_ic(factor_data, stock_data, start_date, end_date, adjust_freq=1,return_periods=[1, 5, 10, 20],save_results=False):
    """
    分析因子与股票收益率的相关性（IC）
    :param factor_data: DataFrame，包含因子数据(宽数据格式)
    :param stock_data: DataFrame，包含股票日线数据(长数据格式)
    :param start_date: str，分析开始日期
    :param end_date: str，分析结束日期
    :param adjust_freq: int，调仓频率，单位为天。默认每日调仓
    :factor_group_num: int, 因子分组数量
    :param return_periods: list，分析的未来收益周期，例如[1,5,10,20]表示分析1日、5日、10日和20日收益

    功能:
    1. 合并因子数据(宽格式)和股票数据(长格式)
    2. 根据return_periods计算下期收益(future_return_{}d)
    3. 按照调仓日期日期计算IC和RankIC,汇报各个下期收益的统计指标(均值,标准差,IR,正相关比例等)
    4. 可视化每个下期收益的ic,
    """
    print(f"开始IC分析: 从{start_date}到{end_date}，调仓频率={adjust_freq}天")
    
    # 1. 数据预处理
    # 确保日期格式一致
    factor_data['trading_date'] = pd.to_datetime(factor_data['trading_date'])
    stock_data['trading_date'] = pd.to_datetime(stock_data['trading_date'])
    factor_data = factor_data.melt(id_vars=['trading_date'], var_name='code', value_name='factor')
    factor_data['code'] = api.convert_stock_code(factor_data['code'])
    
    # 筛选时间范围内的数据
    factor_data = factor_data[(factor_data['trading_date'] >= start_date) & (factor_data['trading_date'] <= end_date)]
    stock_data = stock_data[(stock_data['trading_date'] >= start_date) & (stock_data['trading_date'] <= end_date)]
    
    # 获取所有交易日
    all_dates = sorted(factor_data['trading_date'].unique())
    
    # 2. 获取调仓日期
    rebalance_dates = all_dates[::adjust_freq]
    print(f"分析期间共有{len(rebalance_dates)}个调仓日")
    
    # 3. 计算未来收益率
    # 对stock_data按股票分组，计算未来N日收益率
    stock_data_grouped = stock_data.sort_values(['code', 'trading_date']).groupby('code')
    
    # 初始化收益率列
    for period in return_periods:
        stock_data[f'future_return_{period}d'] = np.nan
    
    # 计算每只股票的未来收益率
    for code, group in stock_data_grouped:
        for period in return_periods: # 计算每个下期收益
            # 计算未来N日收益率: (future_price / current_price) - 1
            stock_data.loc[group.index, f'future_return_{period}d'] = group['close'].shift(-period) / group['close'] - 1
    
    # 4. 计算每个调仓日的IC
    ic_results = [] # 包含每个调仓日因子的所有下期收益的IC
    merged_all = pd.merge(
        factor_data, 
        stock_data, 
        on=['trading_date', 'code'], 
        how='inner'
    )
    merged_all = merged_all.dropna(subset=['factor'])
    
    # 按照调仓日期计算所有下期收益的IC
    for rebalance_date in rebalance_dates:
        # 筛选当天数据
        day_data = merged_all[merged_all['trading_date'] == rebalance_date]
        
        if len(day_data) < 30:
            print(f"警告: {rebalance_date.strftime('%Y-%m-%d')} 样本数量不足，跳过IC计算")
            continue
        
        # 计算每个收益期的IC
        for period in return_periods:
            return_col = f'future_return_{period}d'
            
            # 去除收益率为空的样本
            valid_data = day_data.dropna(subset=[return_col])
            if len(valid_data) < 30:
                continue
                
            # 计算IC值
            ic = valid_data['factor'].corr(valid_data[return_col], method='pearson')
            rank_ic = valid_data['factor'].corr(valid_data[return_col], method='spearman')
            
            ic_results.append({
                'trading_date': rebalance_date,
                'period': period,
                'ic': ic,
                'rank_ic': rank_ic,
                'sample_size': len(valid_data)
            })
            
    
    # 5. 转换结果为DataFrame
    ic_df = pd.DataFrame(ic_results) # 包含每个调仓日的所有下期收益的IC值

    # 6. IC分析和统计
    print("\n==== 因子IC分析结果 ====")
    
    # 按下期收益期分组进行统计
    for period in return_periods:
        # 每个持仓周期ic
        period_ic = ic_df[ic_df['period'] == period] 
        
        if not period_ic.empty:
            ic_mean = period_ic['ic'].mean()
            ic_std = period_ic['ic'].std()
            ic_t_stat = ic_mean / (ic_std / np.sqrt(len(period_ic)))
            ic_ir = ic_mean / ic_std if ic_std != 0 else float('inf')
            ic_positive_ratio = (period_ic['ic'] > 0).mean()
            
            print(f"\n{period}日收益IC统计:")
            print(f"IC均值: {ic_mean:.4f}")
            print(f"IC标准差: {ic_std:.4f}")
            print(f"IC t-统计量: {ic_t_stat:.4f}")
            print(f"IR比率(IC均值/IC标准差): {ic_ir:.4f}")
            print(f"IC>0占比: {ic_positive_ratio:.2%}")
            print(f"样本数量: {len(period_ic)}")
            
            # RankIC统计
            rank_ic_mean = period_ic['rank_ic'].mean()
            rank_ic_std = period_ic['rank_ic'].std()
            rank_ic_ir = rank_ic_mean / rank_ic_std if rank_ic_std != 0 else float('inf')
            
            print(f"\nRankIC均值: {rank_ic_mean:.4f}")
            print(f"RankIC IR比率: {rank_ic_ir:.4f}")
    
    # 7. 可视化
    # 创建保存结果的目录
    output_dir = '因子分析结果'
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    # 保存IC数据
    ic_df.to_csv(f'{output_dir}/ic_analysis.csv', index=False, encoding='utf-8-sig')
    
    summary_df = pd.DataFrame({
    'period': return_periods,
    'ic_mean': [ic_df[ic_df['period']==p]['ic'].mean() for p in return_periods],
    'rank_ic_mean': [ic_df[ic_df['period']==p]['rank_ic'].mean() for p in return_periods],
    'ic_ir': [ic_df[ic_df['period']==p]['ic'].mean()/ic_df[ic_df['period']==p]['ic'].std() 
             for p in return_periods],
    'rank_ic_ir': [ic_df[ic_df['period']==p]['rank_ic'].mean()/ic_df[ic_df['period']==p]['rank_ic'].std() 
                  for p in return_periods]
    })
    print("\n==== 因子IC汇总 ====")
    print(summary_df)
    if save_results:
        summary_df.to_csv(f'{output_dir}/ic_summary.csv', index=False, encoding='utf-8-sig')
    
    # 绘制IC时间序列图
    import matplotlib.pyplot as plt
    
    plt.rcParams["font.family"] = ["SimHei"]  # 中文显示
    plt.rcParams["axes.unicode_minus"] = False  # 负号显示
    
    # 为不同收益期绘制IC时间序列图
    for period in return_periods:
        period_ic = ic_df[ic_df['period'] == period]
        if period_ic.empty:
            continue
            
        plt.figure(figsize=(12, 6))
        plt.plot(period_ic['date'], period_ic['ic'], marker='o', markersize=4, linewidth=1)
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.axhline(y=period_ic['ic'].mean(), color='g', linestyle='--', label=f'均值: {period_ic["ic"].mean():.4f}')
        plt.title(f'{period}日收益IC时间序列')
        plt.xlabel('日期')
        plt.ylabel('IC值')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{output_dir}/ic_{period}d_timeseries.png') if save_results else None
        
        # IC分布直方图
        plt.figure(figsize=(10, 6))
        plt.hist(period_ic['ic'], bins=20, alpha=0.7, color='skyblue')
        plt.axvline(period_ic['ic'].mean(), color='r', linestyle='dashed', linewidth=1)
        plt.text(period_ic['ic'].mean(), plt.ylim()[1]*0.9, f'均值: {period_ic["ic"].mean():.4f}', 
                color='r', ha='center')
        plt.title(f'{period}日收益IC分布直方图')
        plt.xlabel('IC值')
        plt.ylabel('频率')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/ic_{period}d_histogram.png') if save_results else None

    
    # IC衰减曲线
    ic_decay = []
    rank_ic_decay = []
    
    for period in return_periods:
        period_ic = ic_df[ic_df['period'] == period]
        
        if not period_ic.empty:
            ic_decay.append({
                'period': period,
                'ic_mean': period_ic['ic'].mean(),
                'ic_std': period_ic['ic'].std(),
                'rank_ic_mean': period_ic['rank_ic'].mean(),
                'rank_ic_std': period_ic['rank_ic'].std()
            })
    
    ic_decay_df = pd.DataFrame(ic_decay)
    
    if not ic_decay_df.empty:
        plt.figure(figsize=(12, 6))
        plt.errorbar(ic_decay_df['period'], ic_decay_df['ic_mean'], 
                    yerr=ic_decay_df['ic_std'], fmt='o-', capsize=5, label='IC')
        plt.errorbar(ic_decay_df['period'], ic_decay_df['rank_ic_mean'], 
                    yerr=ic_decay_df['rank_ic_std'], fmt='s-', capsize=5, label='RankIC')
        plt.title('IC衰减曲线')
        plt.xlabel('收益期限(天)')
        plt.ylabel('IC均值')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{output_dir}/ic_decay_curve.png') if save_results else None
    
    # 8. 生成摘要报告
    if save_results:
        with open(f'{output_dir}/ic_analysis_summary.txt', 'w', encoding='utf-8') as f:
            f.write("==================== 因子IC分析摘要 ====================\n")
            f.write(f"分析期间: {start_date} 至 {end_date}\n")
            f.write(f"调仓频率: {adjust_freq}天\n\n")
            
            for period in return_periods:
                period_ic = ic_df[ic_df['period'] == period]
                period_rank_ic = ic_df[ic_df['period'] == period]
                
                if not period_ic.empty:
                    ic_mean = period_ic['ic'].mean()
                    ic_std = period_ic['ic'].std()
                    ic_t_stat = ic_mean / (ic_std / np.sqrt(len(period_ic)))
                    ic_ir = ic_mean / ic_std if ic_std != 0 else float('inf')
                    ic_positive_ratio = (period_ic['ic'] > 0).mean()
                    
                    f.write(f"{period}日收益IC统计:\n")
                    f.write(f"IC均值: {ic_mean:.4f}\n")
                    f.write(f"IC标准差: {ic_std:.4f}\n")
                    f.write(f"IC t-统计量: {ic_t_stat:.4f}\n")
                    f.write(f"IR比率(IC均值/IC标准差): {ic_ir:.4f}\n")
                    f.write(f"IC>0占比: {ic_positive_ratio:.2%}\n")
                    f.write(f"样本数量: {len(period_ic)}\n\n")
                    
                    # RankIC统计
                    rank_ic_mean = period_rank_ic['rank_ic'].mean()
                    rank_ic_std = period_rank_ic['rank_ic'].std()
                    rank_ic_ir = rank_ic_mean / rank_ic_std if rank_ic_std != 0 else float('inf')
                    
                    f.write(f"RankIC均值: {rank_ic_mean:.4f}\n")
                    f.write(f"RankIC IR比率: {rank_ic_ir:.4f}\n\n")
                    
            f.write("==================================================\n")
        
    return {
        'ic_df': ic_df,
        'ic_decay_df': ic_decay_df
    }


import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


def analyze_factor(factor_data, stock_data, start_date, end_date, adjust_freq=1,
                   return_period=5, group_num=5, save_result=False):
    """
    简化版因子分析函数：仅分析指定持仓周期的因子表现
    核心功能：
    1. 每日IC/IR分析（指定持仓周期）
    2. 每日因子分组收益统计 + 调仓日净值曲线分析
    :param factor_data: DataFrame，因子数据(宽数据格式)
    :param stock_data: DataFrame，股票日线数据(长数据格式)
    :param start_date: str，分析开始时间
    :param end_date: str，分析结束时间
    :param adjust_freq: int，调仓频率（仅作用于净值曲线），默认每日调仓
    :param return_period: int，分析的未来收益周期(持仓周期)，默认5天
    :param group_num: int, 因子分组数量，默认5组
    :param save_result: bool，是否保存分析结果和图表
    :return: dict，包含IC分析、分组收益、净值数据
    """
    # 基础配置
    print(f"开始因子综合分析: {start_date} 至 {end_date} | 持仓周期{return_period}天 | 调仓频率{adjust_freq}天")
    output_dir = '因子分析结果'
    os.makedirs(output_dir, exist_ok=True) if save_result else None
    
    # ===================== 1. 数据预处理 =====================
    # 日期格式统一
    for df in [factor_data, stock_data]:
        df['trading_date'] = pd.to_datetime(df['trading_date'])
    start_date, end_date = pd.to_datetime(start_date), pd.to_datetime(end_date)
    
    # 因子宽转长 + 时间筛选
    factor_data = factor_data.melt(id_vars=['trading_date'], var_name='code', value_name='factor')
    factor_data = factor_data[(factor_data['trading_date'] >= start_date) & (factor_data['trading_date'] <= end_date)]
    stock_data = stock_data[(stock_data['trading_date'] >= start_date) & (stock_data['trading_date'] <= end_date)]
    
    # 交易日配置
    all_dates = sorted(factor_data['trading_date'].unique())
    rebalance_dates = all_dates[::adjust_freq]
    print(f"有效交易日: {len(all_dates)} | 调仓日数量: {len(rebalance_dates)}")
    
    # 计算单周期未来收益率
    return_col = f'future_return_{return_period}d'
    stock_data = stock_data.sort_values(['code', 'trading_date'])
    stock_data[return_col] = stock_data.groupby('code')['close'].shift(-return_period) / stock_data['close'] - 1
    
    # 合并数据
    merged_all = pd.merge(factor_data, stock_data, on=['trading_date', 'code'], how='inner')
    merged_all = merged_all.dropna(subset=['factor'])
    
    # ===================== 2. 每日IC/IR分析 =====================
    print("\n==== 一、每日IC/IR分析 ====")
    ic_results = []
    
    # 批量计算每日IC（简化循环逻辑）
    for trade_date, day_data in merged_all.groupby('trading_date'):
        if len(day_data) < 30:
            continue
        
        valid_data = day_data.dropna(subset=[return_col])
        if len(valid_data) < 30:
            continue
        
        # 计算IC和RankIC
        ic = valid_data['factor'].corr(valid_data[return_col], method='pearson')
        rank_ic = valid_data['factor'].corr(valid_data[return_col], method='spearman')
        
        ic_results.append({
            'trading_date': trade_date,
            'ic': ic,
            'rank_ic': rank_ic,
            'sample_size': len(valid_data)
        })
    
    # IC统计汇总（简化版）
    ic_df = pd.DataFrame(ic_results)
    if not ic_df.empty:
        ic_stats = {
            'ic_mean': ic_df['ic'].mean(),
            'ic_std': ic_df['ic'].std(),
            'ic_ir': ic_df['ic'].mean() / ic_df['ic'].std() if ic_df['ic'].std() != 0 else np.nan,
            'ic_positive_ratio': (ic_df['ic'] > 0).mean(),
            'rank_ic_mean': ic_df['rank_ic'].mean(),
            'rank_ic_ir': ic_df['rank_ic'].mean() / ic_df['rank_ic'].std() if ic_df['rank_ic'].std() != 0 else np.nan
        }
        # 打印IC结果
        print(f"IC均值: {ic_stats['ic_mean']:.4f} | IC_IR: {ic_stats['ic_ir']:.4f} | IC>0占比: {ic_stats['ic_positive_ratio']:.2%}")
        print(f"RankIC均值: {ic_stats['rank_ic_mean']:.4f} | RankIC_IR: {ic_stats['rank_ic_ir']:.4f}")
    
    # ===================== 3. 因子分组（每日标记 + 收益统计） =====================
    print("\n==== 二、因子分层收益分析 ====")
    # 核心优化：用groupby + apply批量标记每日分组（简化循环）    
    def assign_group(day_data):
        """单日因子分组函数（简化版）"""
        if len(day_data) < group_num * 10:
            day_data['factor_group'] = np.nan
            return day_data
        
        try:
            day_data['factor_group'] = pd.qcut(
                day_data['factor'], 
                q=group_num, 
                labels=[f'G{i+1}' for i in range(group_num)],
                duplicates='drop'
            )
        except:
            day_data['factor_group'] = np.nan
        return day_data
    
    # 批量标记所有交易日分组
    merged_all = merged_all.groupby('trading_date',group_keys=False).apply(assign_group)
    merged_all_grouped = merged_all.dropna(subset=['factor_group', return_col])
    
    # 每日分组收益统计（简化版）
    daily_group_returns = merged_all_grouped.groupby(['trading_date', 'factor_group'])[return_col].agg([
        ('mean', 'mean'), ('count', 'count')
    ]).reset_index()
    
    # 调仓日净值计算
    rebalance_returns = daily_group_returns[daily_group_returns['trading_date'].isin(rebalance_dates)]
    if not rebalance_returns.empty:
        # 净值计算（仅调仓日）
        nav_df = rebalance_returns.pivot_table(
            index='trading_date', 
            columns='factor_group', 
            values='mean'
        ).fillna(0)
        nav_df = (1 + nav_df).cumprod()
    else:
        print(f"警告：无有效调仓日收益数据")
        nav_df = pd.DataFrame()
    
    # 分组收益统计（简化版）
    group_stats = []
    for group in [f'G{i+1}' for i in range(group_num)]:
        if group not in daily_group_returns['factor_group'].unique():
            continue
        
        group_data = daily_group_returns[daily_group_returns['factor_group'] == group]['mean']
        mean_return = group_data.mean()
        std_return = group_data.std()
        
        # 年化收益/夏普（简化版）
        annual_return = mean_return * 252 / return_period
        sharpe = (annual_return / std_return) * np.sqrt(252 / return_period) if std_return != 0 else np.nan
        
        # 最大回撤（仅调仓日净值）
        max_dd = np.nan
        if group in nav_df.columns:
            peak = nav_df[group].expanding().max()
            max_dd = ((nav_df[group] - peak) / peak).min()
        
        group_stats.append({
            'factor_group': group,
            'mean_daily_return': mean_return,
            'std_return': std_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'positive_ratio': (group_data > 0).mean(),
            'sample_count': len(group_data)
        })
    
    # 打印分组统计
    group_stats_df = pd.DataFrame(group_stats)
    print("\n分组收益统计（基于所有交易日）:")
    print(group_stats_df.round(4))
    
    # ===================== 4. 可视化（简化版） =====================
    import matplotlib.pyplot as plt
    
    plt.rcParams["font.family"] = ["SimHei"]  # 中文显示
    plt.rcParams["axes.unicode_minus"] = False  # 负号显示
    # IC时间序列图
    if not ic_df.empty:
        plt.figure(figsize=(12, 6))
        plt.plot(ic_df['trading_date'], ic_df['ic'], marker='o', markersize=3, linewidth=1, color='#1f77b4')
        plt.axhline(y=ic_df['ic'].mean(), color='g', linestyle='--', label=f'IC均值: {ic_df["ic"].mean():.4f}')
        plt.title(f'{return_period}日收益IC时间序列（每日计算）')
        plt.xlabel('日期'), plt.ylabel('IC值'), plt.grid(alpha=0.3), plt.legend()
        plt.tight_layout() 
        plt.savefig(f'{output_dir}/ic_timeseries.png', dpi=300) if save_result else None
        plt.show()
        plt.close()
    
    # 调仓日净值曲线（移除多空曲线）
    if not nav_df.empty:
        plt.figure(figsize=(12, 6))
        for col in nav_df.columns:
            plt.plot(nav_df.index, nav_df[col], label=col, linewidth=1.5)
        plt.title(f'{return_period}日持仓周期分组净值曲线（调仓频率{adjust_freq}天）')
        plt.xlabel('日期'), plt.ylabel('累计净值(初始=1)'), plt.grid(alpha=0.3), plt.legend()
        plt.tight_layout()
        plt.savefig(f'{output_dir}/group_nav.png', dpi=300) if save_result else None
        plt.show()
        plt.close()
    
    # 分组平均收益柱状图
    group_mean = daily_group_returns.groupby('factor_group')['mean'].mean().reset_index()
    plt.figure(figsize=(10, 6))
    bars = plt.bar(group_mean['factor_group'], group_mean['mean'], 
                    color=['#2ecc71', '#f39c12', '#e74c3c', '#9b59b6', '#3498db'][:group_num])
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # 数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + (0.0001 if height>0 else -0.0001),
                f'{height:.6f}', ha='center', va='bottom' if height>0 else 'top')
    
    plt.title(f'{return_period}日持仓周期分组平均收益（所有交易日）')
    plt.xlabel('因子分组'), plt.ylabel('平均日收益'), plt.grid(alpha=0.3, axis='y')
    plt.tight_layout() 
    plt.savefig(f'{output_dir}/group_return.png', dpi=300) if save_result else None
    plt.show()
    plt.close()
    
    # ===================== 5. 结果保存（简化版） =====================
    if save_result:
        # 保存核心结果
        ic_df.to_csv(f'{output_dir}/ic_daily.csv', index=False, encoding='utf-8-sig')
        group_stats_df.to_csv(f'{output_dir}/group_stats.csv', index=False, encoding='utf-8-sig')
        daily_group_returns.to_csv(f'{output_dir}/daily_group_returns.csv', index=False, encoding='utf-8-sig')
        if not nav_df.empty:
            nav_df.to_csv(f'{output_dir}/nav_rebalance.csv', encoding='utf-8-sig')
        
        # 简易报告
        with open(f'{output_dir}/analysis_report.txt', 'w', encoding='utf-8') as f:
            f.write(f"因子分析报告（持仓周期{return_period}天）\n")
            f.write(f"分析区间: {start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')}\n")
            f.write(f"调仓频率: {adjust_freq}天 | 分组数量: {group_num}\n\n")
            f.write("IC统计:\n" + pd.Series(ic_stats).round(4).to_string() + "\n\n")
            f.write("分组收益统计:\n" + group_stats_df.round(4).to_string(index=False))
    
    # ===================== 6. 返回结果（简化版） =====================
    return {
        'ic_df': ic_df,                  # 每日IC数据
        'ic_stats': ic_stats if 'ic_stats' in locals() else {},  # IC统计
        'daily_group_returns': daily_group_returns,  # 每日分组收益
        'nav_df': nav_df,                # 调仓日净值
        'group_stats': group_stats_df    # 分组收益统计
    }


def analyze_factor(
    factor_data: pd.DataFrame,
    close_data: pd.DataFrame,
    start_date: str,
    end_date: str,
    adjust_freq: int = 1,
    return_period: int = 5,
    group_num: int = 5,
    save_result: bool = False
) -> dict:
    """
    超极简版因子分析：纯宽表向量化计算（无长格式转换）
    核心优化：用corrwith直接计算IC，RankIC通过rank+corrwith实现
    """
    # ===================== 1. 数据预处理（纯宽表） =====================
    print(f"开始因子分析: {start_date} ~ {end_date} | 持仓{return_period}天 | 调仓{adjust_freq}天")
    output_dir = '因子分析结果'
    os.makedirs(output_dir, exist_ok=True) if save_result else None

    # 日期格式化 + 切片
    start_date, end_date = pd.to_datetime(start_date), pd.to_datetime(end_date)
    # 用 trading_date 列筛选日期范围（包含起始和结束日期）
    factor_wide = factor_data[(factor_data.index>= start_date) & 
                            (factor_data.index <= end_date)].sort_index().copy()
    close_wide = close_data[(close_data.index >= start_date) & 
                            (close_data.index <= end_date)].sort_index().copy()
    # 计算未来N天收益（宽表）
    ret_wide = close_wide.shift(-return_period) / close_wide - 1

    # 步骤1：先对齐股票代码（避免列不一致导致的空值）
    common_stocks = factor_wide.columns.intersection(ret_wide.columns)
    factor_wide = factor_wide[common_stocks].copy()
    ret_wide = ret_wide[common_stocks].copy()

    # 步骤3：过滤全空行（单日期因子/收益全为空）→ 同步过滤factor和ret
    # 过滤因子全空行
    factor_valid_mask = factor_wide.notna().sum(axis=1) >= group_num * 5
    factor_wide = factor_wide.loc[factor_valid_mask].copy()
    # 同步过滤收益数据（保证日期完全一致）
    ret_wide = ret_wide.loc[factor_wide.index].copy()

    # 步骤4：最后过滤收益全空行（兜底，避免后续计算报错）
    ret_valid_mask = ret_wide.notna().sum(axis=1) >= group_num * 5
    ret_wide = ret_wide.loc[ret_valid_mask].copy()
    factor_wide = factor_wide.loc[ret_wide.index].copy()

    # 最终验证：日期和股票完全一致
    assert factor_wide.index.equals(ret_wide.index), "因子和收益日期未对齐！"
    assert factor_wide.columns.equals(ret_wide.columns), "因子和收益股票代码未对齐！"

    all_dates = factor_wide.index.sort_values()
    rebalance_dates = all_dates[::adjust_freq]
    print(f"有效交易日: {len(all_dates)} | 调仓日: {len(rebalance_dates)}")

    # ===================== 2. 宽表直接计算IC/RankIC（核心优化） =====================
    print("\n==== 一、IC/IR分析（纯宽表计算） ====")
    # 2.1 计算Pearson IC（你的思路：corrwith直接按行计算）
    ic_series = factor_wide.corrwith(ret_wide, axis=1)  # axis=1：每行（日期）计算因子与收益的相关系数
    ic_series.name = 'ic'

    # 2.2 计算RankIC（因子排名与收益排名的相关系数）
    factor_rank = factor_wide.rank(axis=1, method='dense')  # 每行（日期）内因子排名
    ret_rank = ret_wide.rank(axis=1, method='dense')        # 每行（日期）内收益排名
    rank_ic_series = factor_rank.corrwith(ret_rank, axis=1)
    rank_ic_series.name = 'rank_ic'

    # 合并IC/RankIC结果
    ic_df = pd.concat([ic_series, rank_ic_series], axis=1).dropna()

    # IC统计（极简版）
    ic_stats = {
        'ic_mean': ic_df['ic'].mean(),
        'ic_ir': ic_df['ic'].mean() / ic_df['ic'].std() if ic_df['ic'].std() != 0 else np.nan,
        'rank_ic_mean': ic_df['rank_ic'].mean(),
        'rank_ic_ir': ic_df['rank_ic'].mean() / ic_df['rank_ic'].std() if ic_df['rank_ic'].std() != 0 else np.nan,
        'ic_pos_ratio': (ic_df['ic'] > 0).mean(),
        'rank_ic_pos_ratio': (ic_df['rank_ic'] > 0).mean()
    }
    print(f"IC均值: {ic_stats['ic_mean']:.4f} | IC_IR: {ic_stats['ic_ir']:.4f} | IC>0占比: {ic_stats['ic_pos_ratio']:.2%}")
    print(f"RankIC均值: {ic_stats['rank_ic_mean']:.4f} | RankIC_IR: {ic_stats['rank_ic_ir']:.4f} | RankIC>0占比: {ic_stats['rank_ic_pos_ratio']:.2%}")

    # ===================== 3. 宽表分组收益（纯向量化） =====================
    print("\n==== 二、分组收益分析 ====")
    # 3.1 每日分组（宽表直接生成分组矩阵）
    def daily_group(factor_row: pd.Series) -> pd.Series:
        """单日期因子分组（返回分组标签）"""
        valid_mask = factor_row.notna()
        if valid_mask.sum() < group_num * 5:
            return pd.Series(np.nan, index=factor_row.index)
        # 分位数分组
        return pd.qcut(
            factor_row[valid_mask].rank(method='dense'),
            q=group_num,
            labels=[f'G{i+1}' for i in range(group_num)],
            duplicates='drop'
        ).reindex(factor_row.index)

    # 生成分组矩阵（index=日期，columns=代码，values=分组标签）
    group_matrix = factor_wide.apply(daily_group, axis=1)

    # 3.2 计算每日分组收益（宽表掩码+向量化）
    group_returns = {}
    for group in [f'G{i+1}' for i in range(group_num)]:
        # 生成分组掩码（True=该股票属于该分组）
        group_mask = group_matrix == group
        # 分组收益 = (收益 * 掩码).sum() / 掩码数量（避免除以0）
        group_ret = (ret_wide * group_mask).sum(axis=1) / group_mask.sum(axis=1).replace(0, np.nan)
        group_returns[group] = group_ret

    group_returns_wide = pd.DataFrame(group_returns, index=ret_wide.index).dropna()
    # 调仓日净值
    nav_wide = (1 + group_returns_wide.loc[rebalance_dates]).cumprod()

    # 3.3 分组统计
    group_stats = []
    for group in nav_wide.columns:
        daily_ret = group_returns_wide[group].dropna()
        annual_ret = daily_ret.mean() * 252 / return_period
        sharpe = (annual_ret / daily_ret.std()) * np.sqrt(252 / return_period) if daily_ret.std() != 0 else np.nan
        peak = nav_wide[group].expanding().max()
        max_dd = ((nav_wide[group] - peak) / peak).min()

        group_stats.append({
            'group': group,
            'mean_daily_ret': daily_ret.mean(),
            'annual_ret': annual_ret,
            'sharpe': sharpe,
            'max_dd': max_dd,
            'pos_ratio': (daily_ret > 0).mean()
        })
    group_stats_df = pd.DataFrame(group_stats)
    print("分组收益统计:")
    print(group_stats_df.round(4))

    # ===================== 4. 可视化 + 保存 =====================
    plt.rcParams["font.family"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    # 根据group_stats画出每组平均收益的柱状+折线图
    plt.figure(figsize=(10, 6))
    # 1. 绘制柱状图（保留原有逻辑）
    bars = plt.bar(
        group_stats_df['group'], 
        group_stats_df['mean_daily_ret'],
        color=['#2ecc71', '#f39c12', '#e74c3c', '#9b59b6', '#3498db'][:group_num],
        alpha=0.7  # 柱子加一点透明度，避免和折线重叠
    )  
    # 2. 绘制折线图（新增核心代码）
    plt.plot(
        group_stats_df['group'],  # x轴和柱状图一致（分组）
        group_stats_df['mean_daily_ret'],  # y轴和柱状图一致（平均日收益）
        color='red',  # 折线颜色
        marker='o',  # 每个点加圆点标记
        linewidth=2,  # 线宽
        markersize=6  # 标记大小
    )
    # 保留原有辅助元素
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2., 
            height + (0.0001 if height>0 else -0.0001),
            f'{height:.6f}', 
            ha='center', 
            va='bottom' if height>0 else 'top'
        )
    plt.title(f'{return_period}日持仓周期分组平均收益（宽表计算）')
    plt.xlabel('因子分组'), plt.ylabel('平均日收益'), plt.grid(alpha=0.3, axis='y')
    # plt.savefig(f'{output_dir}/group_return_ultra.png', dpi=300) if save_result else None
    plt.show(), plt.close()

    # IC曲线
    if not ic_df.empty:
        plt.figure(figsize=(12, 6))
        plt.plot(ic_df.index, ic_df['ic'], 'o-', markersize=3, linewidth=1, label='IC')
        plt.plot(ic_df.index, ic_df['rank_ic'], 'x-', markersize=3, linewidth=1, label='RankIC')
        plt.axhline(ic_stats['ic_mean'], color='r', linestyle='--', label=f'IC均值: {ic_stats["ic_mean"]:.4f}')
        plt.axhline(ic_stats['rank_ic_mean'], color='g', linestyle='--', label=f'RankIC均值: {ic_stats["rank_ic_mean"]:.4f}')
        plt.title(f'{return_period}日收益IC/RankIC曲线（宽表直接计算）'), plt.grid(alpha=0.3), plt.legend()
        plt.savefig(f'{output_dir}/ic_curve_ultra.png', dpi=300) if save_result else None
        plt.show(), plt.close()

    # 净值曲线
    if not nav_wide.empty:
        plt.figure(figsize=(12, 6))
        nav_wide.plot(ax=plt.gca(), linewidth=1.5)
        plt.title(f'分组净值曲线（调仓{adjust_freq}天）'), plt.ylabel('累计净值'), plt.grid(alpha=0.3)
        plt.savefig(f'{output_dir}/nav_curve_ultra.png', dpi=300) if save_result else None
        plt.show(), plt.close()

    # 保存结果
    if save_result:
        ic_df.to_csv(f'{output_dir}/ic_results_ultra.csv', encoding='utf-8-sig')
        group_stats_df.to_csv(f'{output_dir}/group_stats_ultra.csv', encoding='utf-8-sig')
        nav_wide.to_csv(f'{output_dir}/nav_results_ultra.csv', encoding='utf-8-sig')

    return {
        'ic_df': ic_df, 'ic_stats': ic_stats,
        'group_returns': group_returns_wide, 'nav_df': nav_wide, 'group_stats': group_stats_df
    }