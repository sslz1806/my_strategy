# 定义因子类(因子分析,通用计算函数),以及一些功能函数
import polars as pl
import statsmodels.api as sm
import sys
sys.path.append('D://桌面/策略')
from my_utils.stock_api import *
import pandas as pd
import numpy as np
from typing import Optional, Sequence
api = stock_api()

def add_future_return(
    df,
    ret_col: str = "pct",
    price_col: str = "close",
    pre_close_col: str = "pre_close",
    horizons: Sequence[int] = (1, 5, 10, 20),
    date_col: str = "trading_date",
    code_col: str = "code",
):
    """
    为输入 DataFrame 添加未来 n 日累计收益列。

    支持 Polars 和 Pandas，有 code 列则按股票分组计算，否则按单一时序。
    ret_col 列存在直接用，不存在则用 price_col/pre_close_col-1。
    输出列名格式: future_{ret_col}_{h}d
    """
    is_polars = isinstance(df, pl.DataFrame)
    is_panel = (code_col in df.columns)

    # 排序
    if is_polars:
        sorted_df = df.sort([code_col, date_col] if is_panel else date_col)
    else:
        sorted_df = df.sort_values([code_col, date_col] if is_panel else date_col).copy()

    # 计算未来收益
    if is_polars:
        gross_expr = (pl.col(ret_col) + 1.0) if ret_col in sorted_df.columns else (pl.col(price_col) / pl.col(pre_close_col))
        exprs = []
        for h in horizons:
            # 手动展开 shift 乘法，避免 rolling_product 在低版本 Polars 上不存在
            future_gross = gross_expr.shift(-1)
            for i in range(2, h + 1):
                future_gross = future_gross * gross_expr.shift(-i)
            future_ret = future_gross - 1
            if is_panel:
                future_ret = future_ret.over(code_col)
            exprs.append(future_ret.alias(f"future_{ret_col}_{h}d"))
        sorted_df = sorted_df.with_columns(exprs)
    else:
        if ret_col in sorted_df.columns:
            gross = 1.0 + sorted_df[ret_col].astype(float)
        else:
            gross = sorted_df[price_col].astype(float) / sorted_df[pre_close_col].astype(float)
        sorted_df["__gross__"] = gross
        for h in horizons:
            if is_panel:
                sorted_df[f"future_{ret_col}_{h}d"] = (
                    sorted_df.groupby(code_col)["__gross__"]
                    .transform(lambda s: s.rolling(h, min_periods=h).apply(np.prod, raw=True).shift(-h))
                    - 1.0
                )
            else:
                sorted_df[f"future_{ret_col}_{h}d"] = (
                    sorted_df["__gross__"]
                    .rolling(h, min_periods=h).apply(np.prod, raw=True).shift(-h)
                    - 1.0
                )
        sorted_df = sorted_df.drop(columns=["__gross__"])

    return sorted_df


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


def analyze_ic(factor_data, stock_data, start_date, end_date, adjust_freq=1,
               return_periods=[1, 5, 10, 20], ret_col=None, save_results=False):
    """
    分析因子与股票收益率的相关性（IC）

    参数
    ----
    factor_data : DataFrame
        因子宽表，包含 trading_date 列和股票代码列（宽格式）
    stock_data : DataFrame
        股票日线数据（长格式，含 trading_date, code, 及日收益率或 close 列）
    start_date, end_date : str
        分析起止日期
    adjust_freq : int
        调仓频率（天），默认每日
    return_periods : list
        未来收益周期列表，如 [1, 5, 10, 20]
    ret_col : str, optional
        stock_data 中的日收益率列名（如 'pct', 'returns'），
        传入后使用 add_future_return 从该列复利计算未来收益，
        由调用方保证该列考虑了复权。
        为 None 时向后兼容：优先用已有的 future_return_{}d 列，
        都没有则从 close 列计算（发出 FutureWarning）。
    save_results : bool
        是否保存结果到文件
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
    if ret_col is not None:
        # 用 add_future_return 从可靠的日收益率列复利计算
        # 调用方负责 ret_col 列的准确性（如用前复权价算的 pct_chg）
        stock_data = add_future_return(
            stock_data, ret_col=ret_col, horizons=tuple(return_periods),
            date_col='trading_date', code_col='code',
        )
        # 重命名列以匹配内部约定 future_return_{period}d
        rename_map = {f'future_{ret_col}_{p}d': f'future_return_{p}d' for p in return_periods}
        stock_data.rename(columns=rename_map, inplace=True)
    elif all(f'future_return_{p}d' in stock_data.columns for p in return_periods):
        print("检测到 stock_data 已有未来收益列，跳过计算")
    else:
        # 向后兼容兜底：从 close 列计算（未考虑复权，不推荐）
        import warnings
        warnings.warn(
            "analyze_ic 使用 raw close 计算未来收益（未考虑复权除息），"
            "建议传入 ret_col 参数使用可靠的日收益率列。",
            FutureWarning,
        )
        stock_data_grouped = stock_data.sort_values(['code', 'trading_date']).groupby('code')
        for period in return_periods:
            stock_data.loc[:, f'future_return_{period}d'] = np.nan
        for _, group in stock_data_grouped:
            for period in return_periods:
                stock_data.loc[group.index, f'future_return_{period}d'] = (
                    group['close'].shift(-period) / group['close'] - 1
                )
    
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


def analyze_factor_bak(
    factor_data: pd.DataFrame,
    ret_data: pd.DataFrame,
    start_date: str,
    end_date: str,
    adjust_freq: int = 1,
    return_period: int = 5,
    group_num: int = 5,
    save_result: bool = False
) -> dict:
    """
    超极简版因子分析：纯宽表向量化计算（无长格式转换）

    参数
    ----
    factor_data : pd.DataFrame
        因子值宽表，index=交易日，columns=股票代码
    ret_data : pd.DataFrame
        未来N日累计收益宽表，index=交易日，columns=股票代码
        （与 factor_data 的 index 和 columns 对齐）
        调用方负责收益计算的准确性（如前复权、pct累乘等），
        本函数只做齐对、对齐和滤波后直接计算 IC/分组收益。
    start_date, end_date, adjust_freq, return_period, group_num, save_result
        参见原函数文档。

    说明
    ----
    - IC/RankIC 仍按每个交易日截面计算。
    - 分组收益与净值曲线只在 rebalance_dates（由 adjust_freq 决定）上计算，
      不在非调仓日重新分组。
    - 当 adjust_freq != return_period 时，相邻调仓收益会重叠，函数会发出
      UserWarning，此时净值/年化/夏普仅作参考。
    """
    # ===================== 1. 数据预处理（纯宽表） =====================
    print(f"开始因子分析: {start_date} ~ {end_date} | 持仓{return_period}天 | 调仓{adjust_freq}天")
    output_dir = '因子分析结果'
    os.makedirs(output_dir, exist_ok=True) if save_result else None

    # 日期格式化 + 切片
    start_date, end_date = pd.to_datetime(start_date), pd.to_datetime(end_date)
    factor_wide = factor_data[(factor_data.index>= start_date) &
                            (factor_data.index <= end_date)].sort_index().copy()
    ret_wide = ret_data[(ret_data.index >= start_date) &
                        (ret_data.index <= end_date)].sort_index().copy()

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
    # 3.1 调仓日分组（宽表直接生成分组矩阵，仅在 rebalance_dates 上生效）
    def daily_group(factor_row: pd.Series) -> pd.Series:
        """单日期因子分组（返回分组标签）"""
        valid_mask = factor_row.notna()
        if valid_mask.sum() < group_num * 5:
            return pd.Series(np.nan, index=factor_row.index)
        # 百分位分组法（比pd.qcut更稳健，不受大量重复值影响）
        # 将因子值按排名等分为group_num组
        ranks = factor_row[valid_mask].rank(method='dense')
        pct = ranks / ranks.max()  # 归一化到 [0, 1]
        group_idx = (pct * group_num).clip(0, group_num - 1).astype(int)
        group_labels = pd.Series(
            [f'G{i+1}' for i in range(group_num)],
            index=range(group_num)
        )
        result = group_idx.map(group_labels)
        return result.reindex(factor_row.index)

    # 生成分组矩阵（index=日期，columns=代码，values=分组标签）
    group_matrix = factor_wide.apply(daily_group, axis=1)

    # 修正：只在调仓日进行分组，其余日期置为 NaN，避免每日分组与净值口径不一致
    group_matrix = group_matrix.where(group_matrix.index.isin(rebalance_dates), np.nan)

    # 收益周期与调仓频率不一致时提示：相邻调仓收益会重叠，净值/年化/夏普可能失真
    if adjust_freq != return_period:
        import warnings
        warnings.warn(
            f"adjust_freq({adjust_freq}) != return_period({return_period})，"
            "相邻调仓收益会重叠，净值/年化/夏普可能失真，建议两者相等。",
            UserWarning
        )

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
    # 每年调仓次数由调仓频率 adjust_freq 决定，而不是持仓周期 return_period
    periods_per_year = 252 / adjust_freq
    for group in nav_wide.columns:
        # 只取调仓日的收益用于统计（非调仓日已被置为 NaN）
        daily_ret = group_returns_wide[group].dropna()
        annual_ret = daily_ret.mean() * periods_per_year
        sharpe = (annual_ret / daily_ret.std()) * np.sqrt(periods_per_year) if daily_ret.std() != 0 else np.nan
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


def _group_backtest(
    data: pl.DataFrame,
    ret_col: str,
    ret_windows: Sequence[int],
    group_num: int,
) -> pl.DataFrame:
    """按 FactorAna 口径延续调仓日分组，并计算每日等权收益。"""
    group_wide = (
        data.select("trading_date", "code", "__group")
        .pivot(on="code", index="trading_date", values="__group")
        .sort("trading_date")
    )
    ret_wide = (
        data.select("trading_date", "code", ret_col)
        .pivot(on="code", index="trading_date", values=ret_col)
        .sort("trading_date")
    )
    stock_cols = group_wide.columns[1:]
    dates = group_wide["trading_date"].to_list()
    group_matrix = group_wide.select(stock_cols).to_numpy()
    ret_matrix = ret_wide.select(stock_cols).to_numpy()
    time_index = np.arange(len(dates))

    results = []
    for window in ret_windows:
        # 收益日 t 使用最近一个调仓日（严格早于 t）的分组；例如 w=3 时，
        # 第 0 日分组依次作用于第 1、2、3 日收益，第 3 日收盘再调仓。
        anchor = ((time_index[1:] - 1) // window) * window
        held_groups = group_matrix[anchor]
        for group in range(1, group_num + 1):
            daily_return = np.nanmean(
                np.where(held_groups == group, ret_matrix[1:], np.nan),
                axis=1,
            )
            results.append(
                pl.DataFrame(
                    {
                        "trading_date": dates[1:],
                        "window": window,
                        "group": f"G{group}",
                        "return": daily_return,
                    }
                )
            )

    return (
        pl.concat(results)
        .filter(pl.col("return").is_finite())
        .with_columns(
            pl.col("group").str.slice(1).cast(pl.Int16).alias("__group_order")
        )
        .sort("window", "__group_order", "trading_date")
        .with_columns(
            (1 + pl.col("return"))
            .cum_prod()
            .over("window", "group")
            .alias("nav")
        )
        .with_columns(
            (
                pl.col("nav")
                / pl.max_horizontal(
                    pl.lit(1.0),
                    pl.col("nav").cum_max().over("window", "group"),
                )
                - 1
            )
            .alias("drawdown")
        )
        .drop("__group_order")
    )


def _plot_factor_analysis(
    ic: pl.DataFrame,
    group_returns: pl.DataFrame,
    benchmark: Optional[pl.DataFrame],
    ret_windows: Sequence[int],
    ic_windows: Sequence[int],
    ic_rolling_window: int,
) -> dict:
    """将净值、IC 和累计 IC 各画在一个多窗口 Figure 中。"""
    plt.rcParams["font.family"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    def make_axes(windows):
        figure, axes = plt.subplots(
            len(windows), 1, figsize=(12, 4 * len(windows)), squeeze=False
        )
        return figure, axes[:, 0]

    figures = {}
    figure, axes = make_axes(ret_windows)
    for axis, window in zip(axes, ret_windows):
        frame = group_returns.filter(pl.col("window") == window)
        for group in sorted(
            frame["group"].unique().to_list(), key=lambda value: int(value[1:])
        ):
            group_frame = frame.filter(pl.col("group") == group).sort("trading_date")
            axis.plot(group_frame["trading_date"], group_frame["nav"], label=group)
        if benchmark is not None:
            axis.plot(
                benchmark["trading_date"], benchmark["nav"],
                label="Benchmark", color="#777777", linestyle="--",
            )
        axis.set_title(f"{window}期调仓：分组净值")
        axis.grid(alpha=0.3)
        axis.legend()
    figure.tight_layout()
    figures["nav"] = figure

    figure, axes = make_axes(ic_windows)
    for axis, window in zip(axes, ic_windows):
        frame = ic.filter(pl.col("window") == window).sort("trading_date")
        # 因子预热期过长、截面样本不足或因子为常量时，可能没有任何有效 IC。
        # 此时仍返回框架图对象给调用方展示“无可用 IC”，而不是格式化 None 后中断整份报告。
        if frame.is_empty():
            axis.text(
                0.5,
                0.5,
                "该窗口没有可用 IC / RankIC 样本",
                horizontalalignment="center",
                verticalalignment="center",
                transform=axis.transAxes,
            )
            axis.axhline(0, color="black", linewidth=0.8)
            axis.set_title(f"未来{window}期收益：IC / RankIC")
            axis.grid(alpha=0.3)
            continue
        ic_line = axis.plot(
            frame["trading_date"],
            frame["ic"],
            label="IC",
            linewidth=0.7,
            alpha=0.3,
        )[0]
        rank_ic_line = axis.plot(
            frame["trading_date"],
            frame["rank_ic"],
            label="RankIC",
            linewidth=0.7,
            alpha=0.3,
        )[0]
        # 原始日频 IC 波动较大；滚动均值只用于平滑展示，不参与统计结果计算。
        axis.plot(
            frame["trading_date"],
            frame["ic"].rolling_mean(ic_rolling_window, min_samples=1),
            color=ic_line.get_color(),
            linewidth=2,
            alpha=0.8,
            label=f"IC {ic_rolling_window}日滚动",
        )
        axis.plot(
            frame["trading_date"],
            frame["rank_ic"].rolling_mean(ic_rolling_window, min_samples=1),
            color=rank_ic_line.get_color(),
            linewidth=2,
            alpha=0.8,
            label=f"RankIC {ic_rolling_window}日滚动",
        )
        ic_mean = frame["ic"].mean()
        rank_ic_mean = frame["rank_ic"].mean()
        axis.axhline(
            ic_mean,
            color=ic_line.get_color(),
            linestyle="--",
            label=f"IC均值: {ic_mean:.4f}",
        )
        axis.axhline(
            rank_ic_mean,
            color=rank_ic_line.get_color(),
            linestyle="--",
            label=f"RankIC均值: {rank_ic_mean:.4f}",
        )
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_title(f"未来{window}期收益：IC / RankIC")
        axis.grid(alpha=0.3)
        axis.legend()
    figure.tight_layout()
    figures["ic_series"] = figure

    figure, axes = make_axes(ic_windows)
    for axis, window in zip(axes, ic_windows):
        frame = ic.filter(pl.col("window") == window).sort("trading_date")
        if frame.is_empty():
            axis.text(
                0.5,
                0.5,
                "该窗口没有可用累计 IC / RankIC 样本",
                horizontalalignment="center",
                verticalalignment="center",
                transform=axis.transAxes,
            )
            axis.axhline(0, color="black", linewidth=0.8)
            axis.set_title(f"未来{window}期收益：累计IC")
            axis.grid(alpha=0.3)
            continue
        axis.plot(frame["trading_date"], frame["cum_ic"], label="累计IC")
        axis.plot(frame["trading_date"], frame["cum_rank_ic"], label="累计RankIC")
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_title(f"未来{window}期收益：累计IC")
        axis.grid(alpha=0.3)
        axis.legend()
    figure.tight_layout()
    figures["cumulative_ic"] = figure
    return figures


def analyze_factor(
    data: pl.DataFrame,
    factor_col: str,
    ret_col: str = "daily_ret",
    ret_windows: Sequence[int] = (1, 3, 5),
    ic_windows: Sequence[int] = (1, 3, 5),
    ic_rolling_window: int = 30,
    group_num: int = 5,
    plot: bool = True,
    save_result: bool = False,
) -> dict:
    """
    分析单张 Polars 长表中的截面因子。

    输入固定包含 ``trading_date``、``code``、因子列和单期收益列；可选的
    ``benchmark_ret`` 是同口径基准单期收益。``daily_ret[t]`` 表示 t-1 到 t
    的收益，因此 t 日因子从 t+1 日收益开始生效。

    ``ret_windows`` 是分组策略调仓间隔。调仓间隔内沿用同一组股票，并用
    每日收益连续画净值，所以没有重叠收益或稀疏净值。``ic_windows`` 是 IC
    对应的未来累计收益窗口。``ic_rolling_window`` 只控制 IC 时序图的平滑窗口。
    """
    # ===================== 1. 参数标准化与数据排序 =====================
    # 窗口参数去重排序；data 按日期+代码排序，保证时序正确
    ret_windows = tuple(sorted(set(ret_windows)))
    ic_windows = tuple(sorted(set(ic_windows)))
    data = data.sort("trading_date", "code")

    # ===================== 2. 截面分组（每天独立排序等分） =====================
    # 2.1 计算每个交易日因子值的截面排名（ordinal：1~N，不处理重复值）
    rank = pl.col(factor_col).rank(method="ordinal").over("trading_date")
    # 2.2 计算每个交易日的有效样本数，用于等比例分组
    count = pl.col(factor_col).count().over("trading_date")
    # 2.3 将排名映射到 1~group_num 组：G1 因子值最小，G{group_num} 最大
    data = data.with_columns(
        (((rank - 1) * group_num / count).floor() + 1)
        .cast(pl.Int16)
        .alias("__group")
    )

    # ===================== 3. 因子一阶自相关 =====================
    # 对每只股票取上一交易日的因子值，再在每个交易日横截面计算 Spearman
    # 相关系数；首日没有历史因子值，会在汇总均值时自然被过滤。
    data = data.with_columns(
        pl.col(factor_col).shift(1).over("code").alias("__previous_factor")
    )
    factor_autocorr = (
        data.group_by("trading_date")
        .agg(
            pl.corr(
                pl.col(factor_col),
                pl.col("__previous_factor"),
                method="spearman",
            ).alias("factor_autocorr")
        )
        .filter(pl.col("factor_autocorr").is_finite())
        .select(pl.col("factor_autocorr").mean())
        .item()
    )
    data = data.drop("__previous_factor")

    # ===================== 4. 计算 IC 所需的未来累计收益 =====================
    # IC 需要看因子值与未来多期收益的相关性，一次性生成 ic_windows 对应的未来收益列
    # 分组回测本身只使用单期收益 ret_col，未来收益由 _group_backtest 内部按窗口复利
    data = add_future_return(
        data,
        ret_col=ret_col,
        horizons=ic_windows,
        date_col="trading_date",
        code_col="code",
    )

    # ===================== 5. 截面 IC / RankIC 计算 =====================
    # 4.1 对每个 ic_window，在每一天截面上计算因子与未来收益的 Pearson/Spearman 相关系数
    ic_frames = []
    for window in ic_windows:
        future_col = f"future_{ret_col}_{window}d"
        # 只保留因子值和未来收益都有限的样本，避免 NaN/inf 污染相关系数
        valid = pl.col(factor_col).is_finite() & pl.col(future_col).is_finite()
        paired_factor = pl.when(valid).then(pl.col(factor_col))
        paired_return = pl.when(valid).then(pl.col(future_col))
        ic_frames.append(
            data.group_by("trading_date")
            .agg(
                pl.corr(paired_factor, paired_return).alias("ic"),
                pl.corr(
                    paired_factor, paired_return, method="spearman"
                ).alias("rank_ic"),
            )
            .with_columns(pl.lit(window).alias("window"))
        )
    # 4.2 合并所有窗口结果，并过滤掉无效 IC
    ic = pl.concat(ic_frames).filter(
        pl.col("ic").is_finite() & pl.col("rank_ic").is_finite()
    )
    # 4.3 按窗口和日期排序后，计算累计 IC 曲线（用于观察因子持续性）
    ic = (
        ic.sort("window", "trading_date")
        .with_columns(
            pl.col("ic").cum_sum().over("window").alias("cum_ic"),
            pl.col("rank_ic").cum_sum().over("window").alias("cum_rank_ic"),
        )
        .select("trading_date", "window", "ic", "rank_ic", "cum_ic", "cum_rank_ic")
    )
    # 4.4 IC 统计汇总：均值、标准差、IR、正占比（按窗口聚合）
    ic_stats = (
        ic.group_by("window")
        .agg(
            pl.col("ic").mean().alias("ic_mean"),
            pl.col("ic").std().alias("ic_std"),
            (pl.col("ic") > 0).mean().alias("ic_positive_ratio"),
            pl.col("rank_ic").mean().alias("rank_ic_mean"),
            pl.col("rank_ic").std().alias("rank_ic_std"),
            (pl.col("rank_ic") > 0).mean().alias("rank_ic_positive_ratio"),
        )
        .with_columns(
            pl.when(pl.col("ic_std") > 0)
            .then(pl.col("ic_mean") / pl.col("ic_std"))
            .otherwise(None)
            .alias("ic_ir"),
            pl.when(pl.col("rank_ic_std") > 0)
            .then(pl.col("rank_ic_mean") / pl.col("rank_ic_std"))
            .otherwise(None)
            .alias("rank_ic_ir"),
        )
        .sort("window")
    )

    # ===================== 6. 分组回测收益计算 =====================
    # 5.1 基于步骤 2 的分组标签，按 ret_windows 调仓，计算每组每日等权收益和净值
    group_returns = _group_backtest(data, ret_col, ret_windows, group_num)
    # 5.2 分组绩效统计：日均收益、年化收益、夏普、最大回撤、胜率
    group_stats = (
        group_returns.group_by("window", "group")
        .agg(
            pl.col("return").mean().alias("mean_return"),
            (pl.col("nav").last().pow(252.0 / pl.len()) - 1).alias("annual_return"),
            pl.when(pl.col("return").std() > 0)
            .then(pl.col("return").mean() / pl.col("return").std() * np.sqrt(252))
            .otherwise(None)
            .alias("sharpe"),
            pl.col("drawdown").min().alias("max_drawdown"),
            (pl.col("return") > 0).mean().alias("positive_ratio"),
        )
        .with_columns(
            pl.col("group").str.slice(1).cast(pl.Int16).alias("__group_order")
        )
        .sort("window", "__group_order")
        .drop("__group_order")
    )

    # ===================== 7. 基准净值计算（可选） =====================
    # 如果输入包含 benchmark_ret 列且存在有效值，则合成基准累计净值
    benchmark = None
    benchmark_values = (
        data["benchmark_ret"].drop_nulls()
        if "benchmark_ret" in data.columns
        else None
    )
    if benchmark_values is not None and benchmark_values.is_finite().any():
        benchmark = (
            data.group_by("trading_date")
            .agg(
                pl.col("benchmark_ret")
                .filter(pl.col("benchmark_ret").is_finite())
                .first()
                .alias("return")
            )
            .sort("trading_date")
            .slice(1)  # 首日没有前一期收益，收益从第二日开始
            .with_columns((1 + pl.col("return")).cum_prod().alias("nav"))
        )

    # ===================== 8. 可视化 =====================
    # 需要展示或保存时，调用统一绘图函数生成四张图：分组收益、净值、IC序列、累计IC
    figures = {}
    if plot or save_result:
        figures = _plot_factor_analysis(
            ic,
            group_returns,
            benchmark,
            ret_windows,
            ic_windows,
            ic_rolling_window,
        )
        if plot and "agg" not in plt.get_backend().lower():
            plt.show()

    # ===================== 9. 结果保存 =====================
    if save_result:
        output_dir = "因子分析结果"
        os.makedirs(output_dir, exist_ok=True)
        ic.write_csv(f"{output_dir}/ic.csv")
        ic_stats.write_csv(f"{output_dir}/ic_stats.csv")
        group_returns.write_csv(f"{output_dir}/group_returns.csv")
        group_stats.write_csv(f"{output_dir}/group_stats.csv")
        if benchmark is not None:
            benchmark.write_csv(f"{output_dir}/benchmark.csv")
        for name, figure in figures.items():
            figure.savefig(f"{output_dir}/{name}.png", dpi=200, bbox_inches="tight")

    # ===================== 10. 返回结果 =====================
    return {
        "factor_autocorr": factor_autocorr,
        "ic": ic,
        "ic_stats": ic_stats,
        "group_returns": group_returns,
        "group_stats": group_stats,
        "nav": group_returns.select("trading_date", "window", "group", "nav"),
        "benchmark": benchmark,
        "figures": figures,
    }


# ============================================================
# 时序因子分组回测
# ============================================================

def backtest_timeseries_factor(
    analysis_data: pd.DataFrame,
    factor_col: str,
    index_ret_col: str,
    q: int = 5,
    hold_period: int = 5,
    plot: bool = True,
    verbose: bool = True,
    window: int = 252,
) -> dict:
    """
    时序因子分组回测：分组统计 + 每组独立策略回测 + 净值对比

    将单时序因子（如市场情绪指标、大盘择时信号）按分位数分组，
    对每组分别做策略回测并对比绩效。

    参数
    ----
    analysis_data : pd.DataFrame
        时序因子数据。index 为时间轴（交易日/15分钟时间戳/任意有序
        时间戳均可，函数不假定其频率粒度）。
        必须含 factor_col 和 index_ret_col 两列。
    factor_col : str
        因子值列名。
    index_ret_col : str
        单期收益率列名（%单位，如 0.5 表示 0.5%）。NaN 在内部填充为 0。
    q : int
        等分位组数，默认 5。
    hold_period : int
        持仓期数（与 index 粒度一致：日线传天数，分钟线传分钟期数），
        默认 5。
    plot : bool
        是否自动显示图表。Figure 始终通过返回值返回以供二次加工。
    verbose : bool
        是否打印分组统计和绩效汇总。批量调用（如跨指数/跨因子循环）时
        设为 False 以保持输出干净。
    window : int
        滚动分位数窗口长度，默认 252。t 日只使用最近 window 期（包含 t 日）
        的有效因子值计算分位数边界；窗口不足时不分组。

    返回
    ----
    dict : {
        'group_stats': pd.DataFrame,
            分组未来收益统计。index=factor_group (G1~Gq)，
            columns=['平均收益(%)', '收益标准差', '样本数']
        'group_performance': pd.DataFrame,
            分组策略绩效。index=group_name (G1~Gq + '买入持有基准')，
            columns=['累计收益(%)', '年化收益(%)', '夏普比率',
                     '最大回撤(%)', '胜率(%)', '持仓占比(%)']
        'group_nav': pd.DataFrame,
            分组策略与买入持有基准的统一净值宽表，columns=G1~Gq + 买入持有基准，
            index 从首个有效分组日的下一期开始。
        'future_return_col': str,
            内部生成的未来收益列名，格式 f'future_return_{hold_period}d'
        'fig_bar': plt.Figure | None,
            分组平均收益柱状图 Figure 对象
        'fig_nav': plt.Figure | None,
            分组策略净值对比图 Figure 对象
    }

    注意
    ----
    - 内部保留 backtest_group_strategy 和 calculate_group_performance_metrics
      作为嵌套函数，与 notebook 原版实现一致。
    - 分组边界使用截至 t 日（包含 t 日）的最近 window 期因子，不使用未来数据。
    - 未来收益从 index_ret_col 复利合成，区间为 t+1 至 t+h。
    - t 日收盘信号在 t+1 起持有 hold_period 个收益期。
    - 有效样本数小于 q 时抛 ValueError。
    """
    # ---------- 嵌套函数：滚动时序分组 ----------
    def assign_groups(factor, q, window):
        """按截至当日的滚动分位数边界，将因子划分到 G1~Gq。"""
        if isinstance(q, bool) or not isinstance(q, (int, np.integer)) or q < 1:
            raise ValueError("q 必须是正整数")
        if (
            isinstance(window, bool)
            or not isinstance(window, (int, np.integer))
            or window < 1
        ):
            raise ValueError("window 必须是正整数")

        factor = pd.to_numeric(factor, errors="coerce").replace(
            [np.inf, -np.inf], np.nan,
        )
        rolling = factor.rolling(window=window, min_periods=window)

        # 每个分位数边界都由最近 window 期计算，rolling 默认包含当前 t 日。
        # 当前值每超过一个边界，组号增加 1；等于边界时留在较低组。
        group_number = pd.Series(1, index=factor.index, dtype="Int64")
        for quantile in np.arange(1, q) / q:
            boundary = rolling.quantile(quantile)
            group_number += (factor > boundary).fillna(False).astype("Int64")

        # 必须凑满一个完整有效窗口；当日因子缺失时也不能误标成 G1。
        valid = factor.notna() & rolling.count().eq(window)
        groups = pd.Series(pd.NA, index=factor.index, dtype="object")
        groups.loc[valid] = "G" + group_number.loc[valid].astype(str)
        return groups

    # ---------- 嵌套函数：单分组策略回测 ----------
    def backtest_group_strategy(data, group_col, target_group, hold_period):
        """单个分组策略回测"""
        data = data.copy()
        # 预热结束后的因子缺失日仍保留在收益时间轴中，但不产生新信号。
        data['signal'] = (data[group_col] == target_group).fillna(False).astype(int)

        signal_arr = data['signal'].to_numpy()
        # t 日收盘按 close[t] 成交后，首个可获得的日收益是 ret[t+1]。
        # position[i] 对应 data.iloc[i + 1] 的收益期，避免把已走完的 ret[t] 计入。
        position = np.zeros(max(len(data) - 1, 0), dtype=np.int8)
        remaining_days = 0

        for i, s in enumerate(signal_arr[:-1]):
            if s == 1:
                remaining_days = hold_period
            if remaining_days > 0:
                position[i] = 1
                remaining_days -= 1

        strategy_data = data.iloc[1:].copy()
        strategy_data['position'] = position
        strategy_data['strategy_return'] = strategy_data['position'] * strategy_data[index_ret_col] / 100
        strategy_data['strategy_nav'] = (1 + strategy_data['strategy_return']).cumprod()
        return strategy_data

    # ---------- 嵌套函数：分组绩效指标 ----------
    def calculate_group_performance_metrics(data, group_name):
        """计算分组策略表现指标"""
        returns = data['strategy_return'].dropna()
        nav = data['strategy_nav'].dropna()

        if len(returns) == 0:
            return None

        total_return = nav.iloc[-1] - 1
        years = len(returns) / 252
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        daily_rf = 0.03 / 252
        excess_return = returns - daily_rf
        sharpe = np.sqrt(252) * excess_return.mean() / returns.std() if returns.std() > 0 else 0

        peak = nav.expanding().max()
        drawdown = (nav - peak) / peak
        max_drawdown = drawdown.min()

        win_rate = (returns > 0).mean()
        position_days = (data['position'] == 1).sum()
        position_ratio = position_days / len(data)

        return {
            '分组': group_name,
            '累计收益': total_return * 100,
            '年化收益': annual_return * 100,
            '夏普比率': sharpe,
            '最大回撤': max_drawdown * 100,
            '胜率': win_rate * 100,
            '持仓占比': position_ratio * 100,
        }

    # ==================== 主流程 ====================
    # 设置中文字体（避免图标题/标签出现方框）
    plt.rcParams["font.family"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    analysis_data = analysis_data.copy()
    analysis_data = analysis_data.sort_index()

    # 1. 未来收益合成（从 index_ret_col 复利）
    future_return_col = f'future_return_{hold_period}d'
    ret_series = analysis_data[index_ret_col].fillna(0)
    gross = 1 + ret_series / 100
    future_gross = gross.rolling(hold_period).apply(np.prod, raw=True).shift(-hold_period)
    analysis_data[future_return_col] = (future_gross - 1) * 100

    # 2. 分组 —— 最近 window 期滚动分位数包含 t 日，不使用 t+1 之后的数据
    analysis_data['factor_group'] = assign_groups(
        analysis_data[factor_col], q=q, window=window,
    )
    valid_group = analysis_data['factor_group'].notna()
    valid_sample_count = int(valid_group.sum())
    if valid_sample_count < q:
        raise ValueError(
            f"滚动窗口完成后的有效样本数 ({valid_sample_count}) "
            f"小于分组数 ({q})，无法分组；window={window}"
        )

    # 分组统计只使用有标签的日期；策略回测则保留预热结束后的完整时间轴，
    # 防止中间缺失因子把不相邻交易日压缩成相邻收益期。
    analysis_data_grouped = analysis_data.loc[valid_group].copy()
    first_valid_position = int(np.flatnonzero(valid_group.to_numpy())[0])
    backtest_data = analysis_data.iloc[first_valid_position:].copy()

    groups = sorted(
        analysis_data_grouped['factor_group'].unique(),
        key=lambda group: int(group[1:]),
    )
    if len(groups) < q and verbose:
        print(f"⚠ 因子重复值过多，部分 G 组为空（有效组数 {len(groups)} < {q}）")

    # 3. 分组未来收益统计
    group_returns = analysis_data_grouped.groupby('factor_group', observed=True)[future_return_col].agg(
        ['mean', 'std', 'count']
    )
    group_returns.columns = ['平均收益(%)', '收益标准差', '样本数']
    if verbose:
        print(f"\n{factor_col} 分组未来{hold_period}期收益统计:")
        print(group_returns.round(4))

    # 4. 柱状图（仅 plot=True 时创建 Figure）
    fig_bar = None
    colors = [
        '#e74c3c', '#e67e22', '#f1c40f', '#27ae60', '#2980b9',
        '#3498db', '#9b59b6', '#1abc9c', '#e84393', '#00b894',
    ][:q]
    if plot:
        fig_bar, ax_bar = plt.subplots(figsize=(10, 6))
        bars = ax_bar.bar(
            group_returns.index, group_returns['平均收益(%)'],
            color=colors, alpha=0.7,
        )
        ax_bar.axhline(y=0, color='black', linestyle='-', alpha=0.5)

        for bar in bars:
            height = bar.get_height()
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2.,
                height + (0.01 if height > 0 else -0.05),
                f'{height:.4f}%',
                ha='center',
                va='bottom' if height > 0 else 'top',
            )

        ax_bar.set_title(f'{factor_col} 分组未来{hold_period}期平均收益', fontsize=14)
        ax_bar.set_xlabel('因子分组（G1最低，Gq最高）', fontsize=12)
        ax_bar.set_ylabel('平均收益(%)', fontsize=12)
        ax_bar.grid(alpha=0.3, axis='y')
        fig_bar.tight_layout()

    # 5. 分组策略回测
    group_results = {}
    for group in groups:
        group_strategy = backtest_group_strategy(
            backtest_data, 'factor_group',
            target_group=group, hold_period=hold_period,
        )
        group_results[group] = group_strategy

    # 基准：买入持有
    # 与分组策略相同：首个有效分组日只用于 t 日收盘决策，基准从 t+1 开始。
    benchmark_data = backtest_data.iloc[1:].copy()
    benchmark_data['position'] = 1
    benchmark_data['strategy_return'] = benchmark_data[index_ret_col] / 100
    benchmark_data['strategy_nav'] = (1 + benchmark_data['strategy_return']).cumprod()

    # 绩效指标
    all_metrics = []
    for group in groups:
        metrics = calculate_group_performance_metrics(group_results[group], group)
        all_metrics.append(metrics)

    benchmark_metrics = calculate_group_performance_metrics(benchmark_data, '买入持有基准')
    all_metrics.append(benchmark_metrics)

    performance_df = pd.DataFrame(all_metrics).set_index('分组')
    # 所有净值先显式校验索引，再按位置写入同一宽表，避免 pandas 静默重排或补 NaN。
    nav_index = benchmark_data.index
    group_nav_df = pd.DataFrame(index=nav_index)
    for group in groups:
        group_nav = group_results[group]['strategy_nav']
        if not group_nav.index.equals(nav_index):
            raise RuntimeError(f"{group} 净值日期与基准日期不一致")
        group_nav_df[group] = group_nav.to_numpy()
    group_nav_df['买入持有基准'] = benchmark_data['strategy_nav'].to_numpy()

    display_df = performance_df.copy()
    for col in ['累计收益', '年化收益', '最大回撤', '胜率', '持仓占比']:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f'{x:.2f}%')
    if '夏普比率' in display_df.columns:
        display_df['夏普比率'] = display_df['夏普比率'].apply(lambda x: f'{x:.2f}')
    if verbose:
        print("\n" + "=" * 90)
        print("所有分组策略表现汇总")
        print("=" * 90)
        print(display_df)

    # 6. 净值对比图（仅 plot=True 时创建 Figure）
    fig_nav = None
    if plot:
        fig_nav, ax_nav = plt.subplots(figsize=(14, 8))
        nav_colors = colors[:len(groups)] + ['#888888']

        for i, group in enumerate(groups):
            ax_nav.plot(
                group_nav_df.index, group_nav_df[group],
                label=f'{group}', color=nav_colors[i], linewidth=2,
            )

        ax_nav.plot(
            group_nav_df.index, group_nav_df['买入持有基准'],
            label='买入持有基准', color='#888888', linewidth=2, linestyle='--',
        )

        ax_nav.set_title(f'{factor_col} 各分组策略净值对比（持有{hold_period}期）', fontsize=16)
        ax_nav.set_xlabel('时间', fontsize=12)
        ax_nav.set_ylabel('净值（初始=1）', fontsize=12)
        ax_nav.legend(fontsize=11, loc='best')
        ax_nav.grid(alpha=0.3)
        fig_nav.tight_layout()
        fig_nav.show()

    return {
        'group_stats': group_returns,
        'group_performance': performance_df,
        'group_nav': group_nav_df,
        'future_return_col': future_return_col,
        'fig_bar': fig_bar,
        'fig_nav': fig_nav,
    }
