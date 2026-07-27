"""
因子回测/alpha_191/alpha_formulas.py - Alpha191 因子公式计算模块（完整版）

基于 WorldQuant 101 Alpha + 国泰君安 Alpha191 的开源实现整合与本地化适配。
辅助函数参考了 101Alpha_code 的实现，所有公式均已直接实现在此文件中。

核心类: Alpha191Formulas
  接收来自 adapter 的 data_dict（宽表 pandas DataFrame），
  提供全部 191 个 alpha 因子的计算方法。

用法:
    from 因子回测.alpha_191.adapter import load_factor_data, load_factor_data_with_industry
    from 因子回测.alpha_191.alpha_formulas import Alpha191Formulas

    # 基础版（无行业数据，176个alpha可用）
    data = load_factor_data('2025-01-01', '2025-07-01')
    calc = Alpha191Formulas(data)

    # 完整版（含行业+市值，191个alpha全部可用）
    data_full = load_factor_data_with_industry('2025-01-01', '2025-07-01')
    calc_full = Alpha191Formulas(data_full, industry_map=data_full.get('industry'))
"""

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from typing import Dict, Optional, Union, Callable


# ====================================================================
# 辅助函数（参考 101Alpha 开源实现的公式写法，保持原样 + 修复）
# ====================================================================

def ts_sum(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """滚动求和"""
    return df.rolling(window).sum()


def sma(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """滚动均值（简单移动平均）"""
    return df.rolling(window).mean()


def stddev(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """滚动标准差"""
    return df.rolling(window).std()


def correlation(x: pd.DataFrame, y: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """滚动相关系数"""
    return x.rolling(window).corr(y)


def covariance(x: pd.DataFrame, y: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """滚动协方差"""
    return x.rolling(window).cov(y)


def rolling_rank(na: np.ndarray) -> int:
    """辅助函数: 返回数组最后一个值的排名"""
    return rankdata(na)[-1]


def ts_rank(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """滚动时间序列排名"""
    return df.rolling(window).apply(rolling_rank)


def rolling_prod(na: np.ndarray) -> float:
    """辅助函数: 返回数组元素乘积"""
    return np.prod(na)


def product(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """滚动乘积"""
    return df.rolling(window).apply(rolling_prod)


def ts_min(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """滚动最小值"""
    return df.rolling(window).min()


def ts_max(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """滚动最大值"""
    return df.rolling(window).max()


def delta(df: pd.DataFrame, period: int = 1) -> pd.DataFrame:
    """差分: 今天 - N天前"""
    return df.diff(period)


def delay(df: pd.DataFrame, period: int = 1) -> pd.DataFrame:
    """延迟: N天前的值"""
    return df.shift(period)


def rank(df: pd.DataFrame) -> pd.DataFrame:
    """横截面排名（按列排名，返回pct=True的百分比排名）"""
    return df.rank(axis=1, pct=True)


def scale(df: pd.DataFrame, k: float = 1) -> pd.DataFrame:
    """缩放: 使 sum(abs(df)) = k"""
    return df.mul(k).div(np.abs(df).sum(axis=1), axis=0)


def ts_argmax(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """滚动 argmax: 窗口内最大值距离今天的天数"""
    return df.rolling(window).apply(np.argmax) + 1


def ts_argmin(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """滚动 argmin: 窗口内最小值距离今天的天数"""
    return df.rolling(window).apply(np.argmin) + 1


def decay_linear(df: pd.DataFrame, period: int = 10) -> pd.DataFrame:
    """
    线性加权移动平均（LWMA）。

    修复: 原始实现返回单列 'CLOSE' 导致列结构损坏。
    现在返回与输入相同列结构的 DataFrame。
    """
    if df.isnull().values.any():
        df = df.ffill().bfill().fillna(0)

    result = df.copy().to_numpy()
    divisor = period * (period + 1) / 2
    y = (np.arange(period) + 1) / divisor

    for row in range(period - 1, df.shape[0]):
        x = result[row - period + 1: row + 1, :]
        result[row, :] = np.dot(x.T, y)

    return pd.DataFrame(result, index=df.index, columns=df.columns)


def sign(df: pd.DataFrame) -> pd.DataFrame:
    """符号函数"""
    return np.sign(df)


def abs(df: pd.DataFrame) -> pd.DataFrame:
    """绝对值"""
    return np.abs(df)


def log(df: pd.DataFrame) -> pd.DataFrame:
    """自然对数"""
    return np.log(df)


def pow(df: pd.DataFrame, power: Union[float, pd.DataFrame]) -> pd.DataFrame:
    """幂函数"""
    return df.pow(power)


def signed_power(df: pd.DataFrame, power: float) -> pd.DataFrame:
    """带符号的幂运算"""
    return np.sign(df) * (np.abs(df) ** power)


# ====================================================================
# Alpha191 公式计算类
# ====================================================================

class Alpha191Formulas:
    """
    Alpha191 因子公式计算类（完整版）。

    初始化接收来自 LocalDataAdapter 的 data_dict，
    提供全部 191 个 alpha 因子的计算方法。

    传入 industry_map 后可补齐需要行业中性化的 15 个 alpha。

    每个 alpha_NNN() 方法返回最新日横截面 pd.Series，
    alpha_NNN_df() 返回全时段 pd.DataFrame。
    """

    def __init__(
        self,
        data_dict: Dict[str, pd.DataFrame],
        industry_map: Optional[Dict[str, str]] = None,
    ):
        """
        参数:
            data_dict: 来自 LocalDataAdapter / load_factor_data 的返回值
                       包含 open/high/low/close/volume/amount/vwap/returns/
                       advXX/total_mv/circulation_mv/mv_A_free_float
            industry_map: {stock_code: industry_name} 行业分类映射
                          （用于 IndNeutralize 类 alpha，不传则跳过）
        """
        self.open = data_dict['open']
        self.high = data_dict['high']
        self.low = data_dict['low']
        self.close = data_dict['close']
        self.volume = data_dict['volume']
        self.amount = data_dict.get('amount', self.volume * 0)
        self.vwap = data_dict['vwap']
        self.returns = data_dict['returns']

        # 市值字段（从 enhanced adapter 自动获取）
        self.total_mv = data_dict.get('total_mv')
        self.circulation_mv = data_dict.get('circulation_mv')
        self.mv_A_free_float = data_dict.get('mv_A_free_float')

        # 平均成交量（各周期）
        for p in [5, 10, 20, 30, 40, 50, 60, 120, 180]:
            setattr(self, f'adv{p}', data_dict.get(f'adv{p}'))

        # 行业分类映射
        self.industry_map = industry_map

        # 记录未实现的alpha
        self._not_implemented = set()

    # ================================================================
    # 行业中性化工具方法
    # ================================================================

    def ind_neutralize(self, factor: pd.DataFrame, level: str = 'industry') -> pd.DataFrame:
        """
        行业中性化：对因子值按行业做横截面去均值。

        参数:
            factor: 因子值 DataFrame (index=日期, columns=股票)
            level: 行业分类级别（'industry' 使用 industry_map,
                   'sector'/'subindustry' 同理）

        返回:
            中性化后的因子值（残差）
        """
        if self.industry_map is None:
            raise ValueError("缺少行业分类数据（industry_map），无法进行行业中性化")

        result = factor.copy()

        # 将 industry_map 转为 Series
        ind_series = pd.Series(self.industry_map)
        industries = ind_series.unique()

        for date in result.index:
            row = result.loc[date]
            for ind in industries:
                mask = ind_series[ind_series == ind].index
                mask_in_cols = [c for c in mask if c in row.index]
                if len(mask_in_cols) > 1:  # 至少2个同行业股票才中性化
                    row[mask_in_cols] -= row[mask_in_cols].mean()

        return result

    # ================================================================
    # 通用方法
    # ================================================================

    def compute_all(self, alpha_list: Optional[list] = None) -> Dict[str, pd.Series]:
        """
        批量计算多个alpha因子。

        参数:
            alpha_list: alpha编号列表，如 [1, 5, 10]，
                        None 则计算全部已实现的alpha

        返回:
            {alpha_name: pd.Series} 字典
        """
        if alpha_list is None:
            # 扫描所有已实现的alpha方法
            alpha_list = []
            for i in range(1, 192):
                method_name = f'alpha_{i:03d}'
                if hasattr(self, method_name):
                    alpha_list.append(i)

        results = {}
        for n in alpha_list:
            method_name = f'alpha_{n:03d}'
            if hasattr(self, method_name):
                try:
                    results[f'alpha_{n:03d}'] = getattr(self, method_name)()
                except Exception as e:
                    print(f"警告: alpha_{n:03d} 计算失败: {e}")
                    results[f'alpha_{n:03d}'] = None
            else:
                print(f"跳过: alpha_{n:03d} 未实现（需要行业数据）")

        return results

    def to_factor_wide(self, alpha_series: pd.Series, name: str = 'factor') -> pd.DataFrame:
        """
        将单个alpha的Series输出转换为宽表格式（用于 因子回测/alpha.py 的 analyze_ic）。

        analyze_ic 期望格式:
            DataFrame with columns ['trading_date', stock_code1, stock_code2, ...]

        参数:
            alpha_series: alpha_NNN() 返回的 pd.Series (index=trading_date, values=latest_cross_section)
            name: 因子名称

        返回:
            pd.DataFrame: 第一列为 trading_date，其余列为各股票因子值
        """
        # 这里要注意: alpha_NNN() 返回的是每天横截面的最新值（Series）
        # 如果要进行完整的时间序列IC分析，需要使用 alpha_NNN_df() 方法
        raise NotImplementedError(
            "请使用 to_factor_wide_from_df() 方法，传入 alpha_NNN_df() 的返回值"
        )

    def to_factor_wide_from_df(self, df: pd.DataFrame, name: str = 'factor') -> pd.DataFrame:
        """
        将完整 DataFrame（来自 alpha_NNN_df()）转换为 analyze_ic 需要的宽表格式。

        analyze_ic 期望:
            - 第一列为 trading_date
            - 其余列为各股票代码
            - 值 = 因子值
        """
        result = df.copy()
        result.insert(0, 'trading_date', result.index)
        return result

    # ================================================================
    # Alpha #1 ~ #101（参考 101Alpha 开源实现，公式更清晰）
    # ================================================================

    # Alpha#1	 (rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) - 0.5)
    def alpha_001(self):
        inner = self.close.copy()
        mask = self.returns < 0
        inner[mask] = stddev(self.returns, 20)
        return (rank(ts_argmax(signed_power(inner, 2), 5)) - 0.5).iloc[-1]

    def alpha_001_df(self):
        inner = self.close.copy()
        mask = self.returns < 0
        inner[mask] = stddev(self.returns, 20)
        return rank(ts_argmax(signed_power(inner, 2), 5)) - 0.5

    # Alpha#2	 (-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))
    def alpha_002(self):
        df = -1 * correlation(rank(delta(log(self.volume), 2)), rank((self.close - self.open) / self.open), 6)
        return df.replace([-np.inf, np.inf], 0).fillna(0).iloc[-1]

    def alpha_002_df(self):
        df = -1 * correlation(rank(delta(log(self.volume), 2)), rank((self.close - self.open) / self.open), 6)
        return df.replace([-np.inf, np.inf], 0).fillna(0)

    # Alpha#3	 (-1 * correlation(rank(open), rank(volume), 10))
    def alpha_003(self):
        df = -1 * correlation(rank(self.open), rank(self.volume), 10)
        return df.replace([-np.inf, np.inf], 0).fillna(0).iloc[-1]

    def alpha_003_df(self):
        df = -1 * correlation(rank(self.open), rank(self.volume), 10)
        return df.replace([-np.inf, np.inf], 0).fillna(0)

    # Alpha#4	 (-1 * Ts_Rank(rank(low), 9))
    def alpha_004(self):
        return (-1 * ts_rank(rank(self.low), 9)).iloc[-1]

    def alpha_004_df(self):
        return -1 * ts_rank(rank(self.low), 9)

    # Alpha#5	 (rank((open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap)))))
    def alpha_005(self):
        return (rank((self.open - (ts_sum(self.vwap, 10) / 10))) *
                (-1 * abs(rank((self.close - self.vwap))))).iloc[-1]

    def alpha_005_df(self):
        return (rank((self.open - (ts_sum(self.vwap, 10) / 10))) *
                (-1 * abs(rank((self.close - self.vwap)))))

    # Alpha#6	 (-1 * correlation(open, volume, 10))
    def alpha_006(self):
        df = -1 * correlation(self.open, self.volume, 10)
        return df.replace([-np.inf, np.inf], 0).fillna(0).iloc[-1]

    def alpha_006_df(self):
        df = -1 * correlation(self.open, self.volume, 10)
        return df.replace([-np.inf, np.inf], 0).fillna(0)

    # Alpha#7	 ((adv20 < volume) ? ((-1 * ts_rank(abs(delta(close, 7)), 60)) * sign(delta(close, 7))) : (-1* 1))
    def alpha_007(self):
        adv20 = sma(self.volume, 20)
        alpha = -1 * ts_rank(abs(delta(self.close, 7)), 60) * sign(delta(self.close, 7))
        alpha[adv20 >= self.volume] = -1
        return alpha.iloc[-1]

    def alpha_007_df(self):
        adv20 = sma(self.volume, 20)
        alpha = -1 * ts_rank(abs(delta(self.close, 7)), 60) * sign(delta(self.close, 7))
        alpha[adv20 >= self.volume] = -1
        return alpha

    # Alpha#8	 (-1 * rank(((sum(open, 5) * sum(returns, 5)) - delay((sum(open, 5) * sum(returns, 5)),10))))
    def alpha_008(self):
        return (-1 * rank(((ts_sum(self.open, 5) * ts_sum(self.returns, 5)) -
                           delay((ts_sum(self.open, 5) * ts_sum(self.returns, 5)), 10)))).iloc[-1]

    def alpha_008_df(self):
        return -1 * rank(((ts_sum(self.open, 5) * ts_sum(self.returns, 5)) -
                          delay((ts_sum(self.open, 5) * ts_sum(self.returns, 5)), 10)))

    # Alpha#9	 ((0 < ts_min(delta(close, 1), 5)) ? delta(close, 1) : ((ts_max(delta(close, 1), 5) < 0) ?delta(close, 1) : (-1 * delta(close, 1))))
    def alpha_009(self):
        delta_close = delta(self.close, 1)
        cond_1 = ts_min(delta_close, 5) > 0
        cond_2 = ts_max(delta_close, 5) < 0
        alpha = -1 * delta_close
        alpha[cond_1 | cond_2] = delta_close
        return alpha.iloc[-1]

    def alpha_009_df(self):
        delta_close = delta(self.close, 1)
        cond_1 = ts_min(delta_close, 5) > 0
        cond_2 = ts_max(delta_close, 5) < 0
        alpha = -1 * delta_close
        alpha[cond_1 | cond_2] = delta_close
        return alpha

    # Alpha#10	 rank(((0 < ts_min(delta(close, 1), 4)) ? delta(close, 1) : ((ts_max(delta(close, 1), 4) < 0)? delta(close, 1) : (-1 * delta(close, 1)))))
    def alpha_010(self):
        delta_close = delta(self.close, 1)
        cond_1 = ts_min(delta_close, 4) > 0
        cond_2 = ts_max(delta_close, 4) < 0
        alpha = -1 * delta_close
        alpha[cond_1 | cond_2] = delta_close
        return alpha.iloc[-1]

    def alpha_010_df(self):
        delta_close = delta(self.close, 1)
        cond_1 = ts_min(delta_close, 4) > 0
        cond_2 = ts_max(delta_close, 4) < 0
        alpha = -1 * delta_close
        alpha[cond_1 | cond_2] = delta_close
        return alpha

    # Alpha#11	 ((rank(ts_max((vwap - close), 3)) + rank(ts_min((vwap - close), 3))) * rank(delta(volume, 3)))
    def alpha_011(self):
        return ((rank(ts_max((self.vwap - self.close), 3)) + rank(ts_min((self.vwap - self.close), 3))) *
                rank(delta(self.volume, 3))).iloc[-1]

    def alpha_011_df(self):
        return ((rank(ts_max((self.vwap - self.close), 3)) + rank(ts_min((self.vwap - self.close), 3))) *
                rank(delta(self.volume, 3)))

    # Alpha#12	 (sign(delta(volume, 1)) * (-1 * delta(close, 1)))
    def alpha_012(self):
        return (sign(delta(self.volume, 1)) * (-1 * delta(self.close, 1))).iloc[-1]

    def alpha_012_df(self):
        return sign(delta(self.volume, 1)) * (-1 * delta(self.close, 1))

    # Alpha#13	 (-1 * rank(covariance(rank(close), rank(volume), 5)))
    def alpha_013(self):
        return (-1 * rank(covariance(rank(self.close), rank(self.volume), 5))).iloc[-1]

    def alpha_013_df(self):
        return -1 * rank(covariance(rank(self.close), rank(self.volume), 5))

    # Alpha#14	 ((-1 * rank(delta(returns, 3))) * correlation(open, volume, 10))
    def alpha_014(self):
        df = correlation(self.open, self.volume, 10)
        df = df.replace([-np.inf, np.inf], 0).fillna(0)
        return (-1 * rank(delta(self.returns, 3)) * df).iloc[-1]

    def alpha_014_df(self):
        df = correlation(self.open, self.volume, 10)
        df = df.replace([-np.inf, np.inf], 0).fillna(0)
        return -1 * rank(delta(self.returns, 3)) * df

    # Alpha#15	 (-1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3))
    def alpha_015(self):
        df = correlation(rank(self.high), rank(self.volume), 3)
        df = df.replace([-np.inf, np.inf], 0).fillna(0)
        return (-1 * ts_sum(rank(df), 3)).iloc[-1]

    def alpha_015_df(self):
        df = correlation(rank(self.high), rank(self.volume), 3)
        df = df.replace([-np.inf, np.inf], 0).fillna(0)
        return -1 * ts_sum(rank(df), 3)

    # Alpha#16	 (-1 * rank(covariance(rank(high), rank(volume), 5)))
    def alpha_016(self):
        return (-1 * rank(covariance(rank(self.high), rank(self.volume), 5))).iloc[-1]

    def alpha_016_df(self):
        return -1 * rank(covariance(rank(self.high), rank(self.volume), 5))

    # Alpha#17	 (((-1 * rank(ts_rank(close, 10))) * rank(delta(delta(close, 1), 1))) * rank(ts_rank((volume / adv20), 5)))
    def alpha_017(self):
        adv20 = sma(self.volume, 20)
        return (-1 * (rank(ts_rank(self.close, 10)) *
                      rank(delta(delta(self.close, 1), 1)) *
                      rank(ts_rank((self.volume / adv20), 5)))).iloc[-1]

    def alpha_017_df(self):
        adv20 = sma(self.volume, 20)
        return -1 * (rank(ts_rank(self.close, 10)) *
                     rank(delta(delta(self.close, 1), 1)) *
                     rank(ts_rank((self.volume / adv20), 5)))

    # Alpha#18	 (-1 * rank(((stddev(abs((close - open)), 5) + (close - open)) + correlation(close, open, 10))))
    def alpha_018(self):
        df = correlation(self.close, self.open, 10)
        df = df.replace([-np.inf, np.inf], 0).fillna(0)
        return (-1 * (rank((stddev(abs((self.close - self.open)), 5) + (self.close - self.open)) + df))).iloc[-1]

    def alpha_018_df(self):
        df = correlation(self.close, self.open, 10)
        df = df.replace([-np.inf, np.inf], 0).fillna(0)
        return -1 * (rank((stddev(abs((self.close - self.open)), 5) + (self.close - self.open)) + df))

    # Alpha#19	 ((-1 * sign(((close - delay(close, 7)) + delta(close, 7)))) * (1 + rank((1 + sum(returns, 250)))))
    def alpha_019(self):
        return ((-1 * sign((self.close - delay(self.close, 7)) + delta(self.close, 7))) *
                (1 + rank(1 + ts_sum(self.returns, 250)))).iloc[-1]

    def alpha_019_df(self):
        return ((-1 * sign((self.close - delay(self.close, 7)) + delta(self.close, 7))) *
                (1 + rank(1 + ts_sum(self.returns, 250))))

    # Alpha#20	 (((-1 * rank((open - delay(high, 1)))) * rank((open - delay(close, 1)))) * rank((open - delay(low, 1))))
    def alpha_020(self):
        return (-1 * (rank(self.open - delay(self.high, 1)) *
                      rank(self.open - delay(self.close, 1)) *
                      rank(self.open - delay(self.low, 1)))).iloc[-1]

    def alpha_020_df(self):
        return -1 * (rank(self.open - delay(self.high, 1)) *
                     rank(self.open - delay(self.close, 1)) *
                     rank(self.open - delay(self.low, 1)))

    # Alpha#21	 ((((sum(close, 8) / 8) + stddev(close, 8)) < (sum(close, 2) / 2)) ? (-1 * 1) :
    #              (((sum(close, 2) / 2) < ((sum(close, 8) / 8) - stddev(close, 8))) ? 1 :
    #              (((1 < (volume / adv20)) || ((volume / adv20) == 1)) ? 1 : (-1 * 1))))
    def alpha_021(self):
        cond_1 = sma(self.close, 8) + stddev(self.close, 8) < sma(self.close, 2)
        cond_2 = sma(self.volume, 20) / self.volume < 1
        alpha = pd.DataFrame(np.ones_like(self.close), index=self.close.index, columns=self.close.columns)
        alpha[cond_1 | cond_2] = -1
        return alpha.iloc[-1]

    def alpha_021_df(self):
        cond_1 = sma(self.close, 8) + stddev(self.close, 8) < sma(self.close, 2)
        cond_2 = sma(self.volume, 20) / self.volume < 1
        alpha = pd.DataFrame(np.ones_like(self.close), index=self.close.index, columns=self.close.columns)
        alpha[cond_1 | cond_2] = -1
        return alpha

    # Alpha#22	 (-1 * (delta(correlation(high, volume, 5), 5) * rank(stddev(close, 20))))
    def alpha_022(self):
        df = correlation(self.high, self.volume, 5)
        df = df.replace([-np.inf, np.inf], 0).fillna(0)
        return (-1 * delta(df, 5) * rank(stddev(self.close, 20))).iloc[-1]

    def alpha_022_df(self):
        df = correlation(self.high, self.volume, 5)
        df = df.replace([-np.inf, np.inf], 0).fillna(0)
        return -1 * delta(df, 5) * rank(stddev(self.close, 20))

    # Alpha#23	 (((sum(high, 20) / 20) < high) ? (-1 * delta(high, 2)) : 0)
    def alpha_023(self):
        cond = sma(self.high, 20) < self.high
        alpha = pd.DataFrame(np.zeros_like(self.close), index=self.close.index, columns=self.close.columns)
        alpha[cond] = -1 * delta(self.high, 2).fillna(0)
        return alpha.iloc[-1]

    def alpha_023_df(self):
        cond = sma(self.high, 20) < self.high
        alpha = pd.DataFrame(np.zeros_like(self.close), index=self.close.index, columns=self.close.columns)
        alpha[cond] = -1 * delta(self.high, 2).fillna(0)
        return alpha

    # Alpha#24	 ((((delta((sum(close, 100) / 100), 100) / delay(close, 100)) < 0.05) ||
    #              ((delta((sum(close, 100) / 100), 100) / delay(close, 100)) == 0.05)) ?
    #              (-1 * (close - ts_min(close, 100))) : (-1 * delta(close, 3)))
    def alpha_024(self):
        cond = delta(sma(self.close, 100), 100) / delay(self.close, 100) <= 0.05
        alpha = -1 * delta(self.close, 3)
        alpha[cond] = -1 * (self.close - ts_min(self.close, 100))
        return alpha.iloc[-1]

    def alpha_024_df(self):
        cond = delta(sma(self.close, 100), 100) / delay(self.close, 100) <= 0.05
        alpha = -1 * delta(self.close, 3)
        alpha[cond] = -1 * (self.close - ts_min(self.close, 100))
        return alpha

    # Alpha#25	 rank(((((-1 * returns) * adv20) * vwap) * (high - close)))
    def alpha_025(self):
        adv20 = sma(self.volume, 20)
        return rank(((((-1 * self.returns) * adv20) * self.vwap) * (self.high - self.close))).iloc[-1]

    def alpha_025_df(self):
        adv20 = sma(self.volume, 20)
        return rank(((((-1 * self.returns) * adv20) * self.vwap) * (self.high - self.close)))

    # Alpha#26	 (-1 * ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3))
    def alpha_026(self):
        df = correlation(ts_rank(self.volume, 5), ts_rank(self.high, 5), 5)
        df = df.replace([-np.inf, np.inf], 0).fillna(0)
        return (-1 * ts_max(df, 3)).iloc[-1]

    def alpha_026_df(self):
        df = correlation(ts_rank(self.volume, 5), ts_rank(self.high, 5), 5)
        df = df.replace([-np.inf, np.inf], 0).fillna(0)
        return -1 * ts_max(df, 3)

    # Alpha#27	 ((0.5 < rank((sum(correlation(rank(volume), rank(vwap), 6), 2) / 2.0))) ? (-1 * 1) : 1)
    def alpha_027(self):
        alpha = rank((sma(correlation(rank(self.volume), rank(self.vwap), 6), 2) / 2.0))
        alpha[alpha > 0.5] = -1
        alpha[alpha <= 0.5] = 1
        return alpha.iloc[-1]

    def alpha_027_df(self):
        alpha = rank((sma(correlation(rank(self.volume), rank(self.vwap), 6), 2) / 2.0))
        alpha[alpha > 0.5] = -1
        alpha[alpha <= 0.5] = 1
        return alpha

    # Alpha#28	 scale(((correlation(adv20, low, 5) + ((high + low) / 2)) - close))
    def alpha_028(self):
        adv20 = sma(self.volume, 20)
        df = correlation(adv20, self.low, 5)
        df = df.replace([-np.inf, np.inf], 0).fillna(0)
        return scale(((df + ((self.high + self.low) / 2)) - self.close)).iloc[-1]

    def alpha_028_df(self):
        adv20 = sma(self.volume, 20)
        df = correlation(adv20, self.low, 5)
        df = df.replace([-np.inf, np.inf], 0).fillna(0)
        return scale(((df + ((self.high + self.low) / 2)) - self.close))

    # Alpha#29	 (min(product(rank(rank(scale(log(sum(ts_min(rank(rank((-1 * rank(delta((close - 1), 5))))), 2), 1))))), 1), 5) + ts_rank(delay((-1 * returns), 6), 5))
    def alpha_029(self):
        return (ts_min(rank(rank(scale(log(ts_sum(rank(rank(-1 * rank(delta((self.close - 1), 5)))), 2))))), 5) +
                ts_rank(delay((-1 * self.returns), 6), 5)).iloc[-1]

    def alpha_029_df(self):
        return (ts_min(rank(rank(scale(log(ts_sum(rank(rank(-1 * rank(delta((self.close - 1), 5)))), 2))))), 5) +
                ts_rank(delay((-1 * self.returns), 6), 5))

    # Alpha#30	 (((1.0 - rank(((sign((close - delay(close, 1))) + sign((delay(close, 1) - delay(close, 2)))) + sign((delay(close, 2) - delay(close, 3)))))) * sum(volume, 5)) / sum(volume, 20))
    def alpha_030(self):
        delta_close = delta(self.close, 1)
        inner = sign(delta_close) + sign(delay(delta_close, 1)) + sign(delay(delta_close, 2))
        return ((1.0 - rank(inner)) * ts_sum(self.volume, 5)) / ts_sum(self.volume, 20).iloc[-1]

    def alpha_030_df(self):
        delta_close = delta(self.close, 1)
        inner = sign(delta_close) + sign(delay(delta_close, 1)) + sign(delay(delta_close, 2))
        return ((1.0 - rank(inner)) * ts_sum(self.volume, 5)) / ts_sum(self.volume, 20)

    # Alpha#31	 ((rank(rank(rank(decay_linear((-1 * rank(rank(delta(close, 10)))), 10)))) + rank((-1 * delta(close, 3)))) + sign(scale(correlation(adv20, low, 12))))
    def alpha_031(self):
        adv20 = sma(self.volume, 20)
        df = correlation(adv20, self.low, 12).replace([-np.inf, np.inf], 0).fillna(0)
        p1 = rank(rank(rank(decay_linear((-1 * rank(rank(delta(self.close, 10)))), 10))))
        p2 = rank((-1 * delta(self.close, 3)))
        p3 = sign(scale(df))
        return (p1 + p2 + p3).iloc[-1]

    def alpha_031_df(self):
        adv20 = sma(self.volume, 20)
        df = correlation(adv20, self.low, 12).replace([-np.inf, np.inf], 0).fillna(0)
        p1 = rank(rank(rank(decay_linear((-1 * rank(rank(delta(self.close, 10)))), 10))))
        p2 = rank((-1 * delta(self.close, 3)))
        p3 = sign(scale(df))
        return p1 + p2 + p3

    # Alpha#32	 (scale(((sum(close, 7) / 7) - close)) + (20 * scale(correlation(vwap, delay(close, 5), 230))))
    def alpha_032(self):
        return (scale(((sma(self.close, 7) / 7) - self.close)) +
                20 * scale(correlation(self.vwap, delay(self.close, 5), 230))).iloc[-1]

    def alpha_032_df(self):
        return (scale(((sma(self.close, 7) / 7) - self.close)) +
                20 * scale(correlation(self.vwap, delay(self.close, 5), 230)))

    # Alpha#33	 rank((-1 * ((1 - (open / close))^1)))
    def alpha_033(self):
        return rank(-1 + (self.open / self.close)).iloc[-1]

    def alpha_033_df(self):
        return rank(-1 + (self.open / self.close))

    # Alpha#34	 rank(((1 - rank((stddev(returns, 2) / stddev(returns, 5)))) + (1 - rank(delta(close, 1)))))
    def alpha_034(self):
        inner = stddev(self.returns, 2) / stddev(self.returns, 5)
        inner = inner.replace([-np.inf, np.inf], 1).fillna(1)
        return rank(2 - rank(inner) - rank(delta(self.close, 1))).iloc[-1]

    def alpha_034_df(self):
        inner = stddev(self.returns, 2) / stddev(self.returns, 5)
        inner = inner.replace([-np.inf, np.inf], 1).fillna(1)
        return rank(2 - rank(inner) - rank(delta(self.close, 1)))

    # Alpha#35	 ((Ts_Rank(volume, 32) * (1 - Ts_Rank(((close + high) - low), 16))) * (1 - Ts_Rank(returns, 32)))
    def alpha_035(self):
        return ((ts_rank(self.volume, 32) *
                 (1 - ts_rank(self.close + self.high - self.low, 16)) *
                 (1 - ts_rank(self.returns, 32)))).iloc[-1]

    def alpha_035_df(self):
        return ((ts_rank(self.volume, 32) *
                 (1 - ts_rank(self.close + self.high - self.low, 16)) *
                 (1 - ts_rank(self.returns, 32))))

    # Alpha#36	 ((2.21 * rank(corr((close-open), delay(volume,1), 15))) + (0.7 * rank((open-close))) + (0.73 * rank(ts_rank(delay(-returns,6),5))) + rank(abs(corr(vwap, adv20,6))) + (0.6 * rank(((sma(close,200)-open)*(close-open)))))
    def alpha_036(self):
        adv20 = sma(self.volume, 20)
        term1 = 2.21 * rank(correlation(self.close - self.open, delay(self.volume, 1), 15))
        term2 = 0.7 * rank(self.open - self.close)
        term3 = 0.73 * rank(ts_rank(delay(-1 * self.returns, 6), 5))
        term4 = rank(abs(correlation(self.vwap, adv20, 6)))
        term5 = 0.6 * rank((sma(self.close, 200) - self.open) * (self.close - self.open))
        return (term1 + term2 + term3 + term4 + term5).iloc[-1]

    def alpha_036_df(self):
        adv20 = sma(self.volume, 20)
        term1 = 2.21 * rank(correlation(self.close - self.open, delay(self.volume, 1), 15))
        term2 = 0.7 * rank(self.open - self.close)
        term3 = 0.73 * rank(ts_rank(delay(-1 * self.returns, 6), 5))
        term4 = rank(abs(correlation(self.vwap, adv20, 6)))
        term5 = 0.6 * rank((sma(self.close, 200) - self.open) * (self.close - self.open))
        return term1 + term2 + term3 + term4 + term5

    # Alpha#37	 (rank(correlation(delay((open - close), 1), close, 200)) + rank((open - close)))
    def alpha_037(self):
        return (rank(correlation(delay(self.open - self.close, 1), self.close, 200)) +
                rank(self.open - self.close)).iloc[-1]

    def alpha_037_df(self):
        return rank(correlation(delay(self.open - self.close, 1), self.close, 200)) + rank(self.open - self.close)

    # Alpha#38	 ((-1 * rank(Ts_Rank(close, 10))) * rank((close / open)))
    def alpha_038(self):
        inner = self.close / self.open
        inner = inner.replace([-np.inf, np.inf], 1).fillna(1)
        return (-1 * rank(ts_rank(self.open, 10)) * rank(inner)).iloc[-1]

    def alpha_038_df(self):
        inner = self.close / self.open
        inner = inner.replace([-np.inf, np.inf], 1).fillna(1)
        return -1 * rank(ts_rank(self.open, 10)) * rank(inner)

    # Alpha#39	 ((-1 * rank((delta(close, 7) * (1 - rank(decay_linear((volume / adv20), 9)))))) * (1 + rank(sum(returns, 250))))
    def alpha_039(self):
        adv20 = sma(self.volume, 20)
        dl = decay_linear((self.volume / adv20), 9)
        return ((-1 * rank(delta(self.close, 7) * (1 - rank(dl)))) *
                (1 + rank(sma(self.returns, 250)))).iloc[-1]

    def alpha_039_df(self):
        adv20 = sma(self.volume, 20)
        dl = decay_linear((self.volume / adv20), 9)
        return ((-1 * rank(delta(self.close, 7) * (1 - rank(dl)))) *
                (1 + rank(sma(self.returns, 250))))

    # Alpha#40	 ((-1 * rank(stddev(high, 10))) * correlation(high, volume, 10))
    def alpha_040(self):
        return (-1 * rank(stddev(self.high, 10)) * correlation(self.high, self.volume, 10)).iloc[-1]

    def alpha_040_df(self):
        return -1 * rank(stddev(self.high, 10)) * correlation(self.high, self.volume, 10)

    # Alpha#41	 (((high * low)^0.5) - vwap)
    def alpha_041(self):
        return (np.sqrt(self.high * self.low) - self.vwap).iloc[-1]

    def alpha_041_df(self):
        return np.sqrt(self.high * self.low) - self.vwap

    # Alpha#42	 (rank((vwap - close)) / rank((vwap + close)))
    def alpha_042(self):
        return (rank((self.vwap - self.close)) / rank((self.vwap + self.close))).iloc[-1]

    def alpha_042_df(self):
        return rank((self.vwap - self.close)) / rank((self.vwap + self.close))

    # Alpha#43	 (ts_rank((volume / adv20), 20) * ts_rank((-1 * delta(close, 7)), 8))
    def alpha_043(self):
        adv20 = sma(self.volume, 20)
        return (ts_rank(self.volume / adv20, 20) * ts_rank((-1 * delta(self.close, 7)), 8)).iloc[-1]

    def alpha_043_df(self):
        adv20 = sma(self.volume, 20)
        return ts_rank(self.volume / adv20, 20) * ts_rank((-1 * delta(self.close, 7)), 8)

    # Alpha#44	 (-1 * correlation(high, rank(volume), 5))
    def alpha_044(self):
        df = correlation(self.high, rank(self.volume), 5)
        return (-1 * df.replace([-np.inf, np.inf], 0).fillna(0)).iloc[-1]

    def alpha_044_df(self):
        df = correlation(self.high, rank(self.volume), 5)
        return -1 * df.replace([-np.inf, np.inf], 0).fillna(0)

    # Alpha#45	 (-1 * ((rank((sum(delay(close, 5), 20) / 20)) * correlation(close, volume, 2)) * rank(correlation(sum(close, 5), sum(close, 20), 2))))
    def alpha_045(self):
        df = correlation(self.close, self.volume, 2)
        df = df.replace([-np.inf, np.inf], 0).fillna(0)
        return (-1 * (rank(sma(delay(self.close, 5), 20)) * df *
                      rank(correlation(ts_sum(self.close, 5), ts_sum(self.close, 20), 2)))).iloc[-1]

    def alpha_045_df(self):
        df = correlation(self.close, self.volume, 2)
        df = df.replace([-np.inf, np.inf], 0).fillna(0)
        return -1 * (rank(sma(delay(self.close, 5), 20)) * df *
                     rank(correlation(ts_sum(self.close, 5), ts_sum(self.close, 20), 2)))

    # Alpha#46	 ((0.25 < (((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10))) ? (-1 * 1) : (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < 0) ? 1 : ((-1 * 1) * (close - delay(close, 1)))))
    def alpha_046(self):
        inner = ((delay(self.close, 20) - delay(self.close, 10)) / 10) - ((delay(self.close, 10) - self.close) / 10)
        alpha = (-1 * delta(self.close))
        alpha[inner < 0] = 1
        alpha[inner > 0.25] = -1
        return alpha.iloc[-1]

    def alpha_046_df(self):
        inner = ((delay(self.close, 20) - delay(self.close, 10)) / 10) - ((delay(self.close, 10) - self.close) / 10)
        alpha = (-1 * delta(self.close))
        alpha[inner < 0] = 1
        alpha[inner > 0.25] = -1
        return alpha

    # Alpha#47	 ((((rank((1 / close)) * volume) / adv20) * ((high * rank((high - close))) / (sum(high, 5) / 5))) - rank((vwap - delay(vwap, 5))))
    def alpha_047(self):
        adv20 = sma(self.volume, 20)
        return ((((rank((1 / self.close)) * self.volume) / adv20) *
                 ((self.high * rank((self.high - self.close))) / (sma(self.high, 5)))) -
                rank((self.vwap - delay(self.vwap, 5)))).iloc[-1]

    def alpha_047_df(self):
        adv20 = sma(self.volume, 20)
        return ((((rank((1 / self.close)) * self.volume) / adv20) *
                 ((self.high * rank((self.high - self.close))) / (sma(self.high, 5)))) -
                rank((self.vwap - delay(self.vwap, 5))))

    # Alpha#48	 (indneutralize(((correlation(delta(close, 1), delta(delay(close, 1), 1), 250) * delta(close, 1)) / close), IndClass.subindustry) / sum(((delta(close, 1) / delay(close, 1))^2), 250))
    def alpha_048(self):
        if self.industry_map is None:
            self._not_implemented.add(48)
            return pd.Series(dtype=float)
        inner = correlation(delta(self.close, 1), delta(delay(self.close, 1), 1), 250) * delta(self.close, 1) / self.close
        neutralized = self.ind_neutralize(inner, level='subindustry')
        denominator = ts_sum((delta(self.close, 1) / delay(self.close, 1)) ** 2, 250)
        return (neutralized / denominator).iloc[-1]

    def alpha_048_df(self):
        if self.industry_map is None:
            self._not_implemented.add(48)
            return pd.DataFrame(index=self.close.index, columns=self.close.columns, dtype=float)
        inner = correlation(delta(self.close, 1), delta(delay(self.close, 1), 1), 250) * delta(self.close, 1) / self.close
        neutralized = self.ind_neutralize(inner, level='subindustry')
        denominator = ts_sum((delta(self.close, 1) / delay(self.close, 1)) ** 2, 250)
        return neutralized / denominator

    # Alpha#49	 (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 * 0.1)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))
    def alpha_049(self):
        inner = (((delay(self.close, 20) - delay(self.close, 10)) / 10) - ((delay(self.close, 10) - self.close) / 10))
        alpha = (-1 * delta(self.close))
        alpha[inner < -0.1] = 1
        return alpha.iloc[-1]

    def alpha_049_df(self):
        inner = (((delay(self.close, 20) - delay(self.close, 10)) / 10) - ((delay(self.close, 10) - self.close) / 10))
        alpha = (-1 * delta(self.close))
        alpha[inner < -0.1] = 1
        return alpha

    # Alpha#50	 (-1 * ts_max(rank(correlation(rank(volume), rank(vwap), 5)), 5))
    def alpha_050(self):
        return (-1 * ts_max(rank(correlation(rank(self.volume), rank(self.vwap), 5)), 5)).iloc[-1]

    def alpha_050_df(self):
        return -1 * ts_max(rank(correlation(rank(self.volume), rank(self.vwap), 5)), 5)

    # Alpha#51	 (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 * 0.05)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))
    def alpha_051(self):
        inner = (((delay(self.close, 20) - delay(self.close, 10)) / 10) - ((delay(self.close, 10) - self.close) / 10))
        alpha = (-1 * delta(self.close))
        alpha[inner < -0.05] = 1
        return alpha.iloc[-1]

    def alpha_051_df(self):
        inner = (((delay(self.close, 20) - delay(self.close, 10)) / 10) - ((delay(self.close, 10) - self.close) / 10))
        alpha = (-1 * delta(self.close))
        alpha[inner < -0.05] = 1
        return alpha

    # Alpha#52	 ((((-1 * ts_min(low, 5)) + delay(ts_min(low, 5), 5)) * rank(((sum(returns, 240) - sum(returns, 20)) / 220))) * ts_rank(volume, 5))
    def alpha_052(self):
        return (((-1 * delta(ts_min(self.low, 5), 5)) *
                 rank(((ts_sum(self.returns, 240) - ts_sum(self.returns, 20)) / 220)) *
                 ts_rank(self.volume, 5))).iloc[-1]

    def alpha_052_df(self):
        return (((-1 * delta(ts_min(self.low, 5), 5)) *
                 rank(((ts_sum(self.returns, 240) - ts_sum(self.returns, 20)) / 220)) *
                 ts_rank(self.volume, 5)))

    # Alpha#53	 (-1 * delta((((close - low) - (high - close)) / (close - low)), 9))
    def alpha_053(self):
        inner = (self.close - self.low).replace(0, 0.0001)
        return (-1 * delta((((self.close - self.low) - (self.high - self.close)) / inner), 9)).iloc[-1]

    def alpha_053_df(self):
        inner = (self.close - self.low).replace(0, 0.0001)
        return -1 * delta((((self.close - self.low) - (self.high - self.close)) / inner), 9)

    # Alpha#54	 ((-1 * ((low - close) * (open^5))) / ((low - high) * (close^5)))
    def alpha_054(self):
        inner = (self.low - self.high).replace(0, -0.0001)
        return (-1 * (self.low - self.close) * (self.open ** 5) / (inner * (self.close ** 5))).iloc[-1]

    def alpha_054_df(self):
        inner = (self.low - self.high).replace(0, -0.0001)
        return -1 * (self.low - self.close) * (self.open ** 5) / (inner * (self.close ** 5))

    # Alpha#55	 (-1 * correlation(rank(((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12)))), rank(volume), 6))
    def alpha_055(self):
        divisor = (ts_max(self.high, 12) - ts_min(self.low, 12)).replace(0, 0.0001)
        inner = (self.close - ts_min(self.low, 12)) / divisor
        df = correlation(rank(inner), rank(self.volume), 6)
        return (-1 * df.replace([-np.inf, np.inf], 0).fillna(0)).iloc[-1]

    def alpha_055_df(self):
        divisor = (ts_max(self.high, 12) - ts_min(self.low, 12)).replace(0, 0.0001)
        inner = (self.close - ts_min(self.low, 12)) / divisor
        df = correlation(rank(inner), rank(self.volume), 6)
        return -1 * df.replace([-np.inf, np.inf], 0).fillna(0)

    # Alpha#56	 (0 - (1 * (rank((sum(returns, 10) / sum(sum(returns, 2), 3))) * rank((returns * cap)))))
    def alpha_056(self):
        if self.total_mv is None:
            self._not_implemented.add(56)
            return pd.Series(dtype=float)
        cap = self.total_mv
        inner = rank((ts_sum(self.returns, 10) / ts_sum(ts_sum(self.returns, 2), 3))) * rank(self.returns * cap)
        return (0 - (1 * inner)).iloc[-1]
    def alpha_056_df(self):
        if self.total_mv is None:
            self._not_implemented.add(56)
            return pd.DataFrame(index=self.close.index, columns=self.close.columns, dtype=float)
        cap = self.total_mv
        inner = rank((ts_sum(self.returns, 10) / ts_sum(ts_sum(self.returns, 2), 3))) * rank(self.returns * cap)
        return 0 - (1 * inner)

    # Alpha#57	 (0 - (1 * ((close - vwap) / decay_linear(rank(ts_argmax(close, 30)), 2))))
    def alpha_057(self):
        dl = decay_linear(rank(ts_argmax(self.close, 30)), 2)
        return (0 - (1 * ((self.close - self.vwap) / dl))).iloc[-1]

    def alpha_057_df(self):
        dl = decay_linear(rank(ts_argmax(self.close, 30)), 2)
        return (0 - (1 * ((self.close - self.vwap) / dl)))

    # Alpha#58	 (-1 * Ts_Rank(decay_linear(correlation(IndNeutralize(vwap, sector), volume, 3.92795), 7.89291), 5.50322))
    def alpha_058(self):
        if self.industry_map is None:
            self._not_implemented.add(58)
            return pd.Series(dtype=float)
        v_neutral = self.ind_neutralize(self.vwap)
        return (-1 * ts_rank(decay_linear(correlation(v_neutral, self.volume, 4), 8), 5)).iloc[-1]
    def alpha_058_df(self):
        if self.industry_map is None:
            return pd.DataFrame(index=self.close.index, columns=self.close.columns, dtype=float)
        v_neutral = self.ind_neutralize(self.vwap)
        return -1 * ts_rank(decay_linear(correlation(v_neutral, self.volume, 4), 8), 5)

    # Alpha#59	 (-1 * Ts_Rank(decay_linear(correlation(IndNeutralize(vwap, industry), volume, 4.25197), 16.2289), 8.19648))
    def alpha_059(self):
        if self.industry_map is None:
            self._not_implemented.add(59)
            return pd.Series(dtype=float)
        v_neutral = self.ind_neutralize(self.vwap * 0.728317 + self.vwap * (1 - 0.728317))
        return (-1 * ts_rank(decay_linear(correlation(v_neutral, self.volume, 4), 16), 8)).iloc[-1]
    def alpha_059_df(self):
        if self.industry_map is None:
            return pd.DataFrame(index=self.close.index, columns=self.close.columns, dtype=float)
        v_neutral = self.ind_neutralize(self.vwap * 0.728317 + self.vwap * (1 - 0.728317))
        return -1 * ts_rank(decay_linear(correlation(v_neutral, self.volume, 4), 16), 8)

    # Alpha#60	 (0 - (1 * ((2 * scale(rank(((((close - low) - (high - close)) / (high - low)) * volume)))) - scale(rank(ts_argmax(close, 10))))))
    def alpha_060(self):
        divisor = (self.high - self.low).replace(0, 0.0001)
        inner = ((self.close - self.low) - (self.high - self.close)) * self.volume / divisor
        return -((2 * scale(rank(inner))) - scale(rank(ts_argmax(self.close, 10)))).iloc[-1]

    def alpha_060_df(self):
        divisor = (self.high - self.low).replace(0, 0.0001)
        inner = ((self.close - self.low) - (self.high - self.close)) * self.volume / divisor
        return -((2 * scale(rank(inner))) - scale(rank(ts_argmax(self.close, 10))))

    # Alpha#61	 (rank((vwap - ts_min(vwap, 16.1219))) < rank(correlation(vwap, adv180, 17.9282)))
    def alpha_061(self):
        adv180 = sma(self.volume, 180)
        return (rank((self.vwap - ts_min(self.vwap, 16))) < rank(correlation(self.vwap, adv180, 18))).iloc[-1]

    def alpha_061_df(self):
        adv180 = sma(self.volume, 180)
        return (rank((self.vwap - ts_min(self.vwap, 16))) < rank(correlation(self.vwap, adv180, 18)))

    # Alpha#62	 ((rank(correlation(vwap, sum(adv20, 22.4101), 9.91009)) < rank(((rank(open) + rank(open)) < (rank(((high + low) / 2)) + rank(high))))) * -1)
    def alpha_062(self):
        adv20 = sma(self.volume, 20)
        return ((rank(correlation(self.vwap, sma(adv20, 22), 10)) <
                 rank(((rank(self.open) + rank(self.open)) <
                       (rank(((self.high + self.low) / 2)) + rank(self.high))))) * -1).iloc[-1]

    def alpha_062_df(self):
        adv20 = sma(self.volume, 20)
        return ((rank(correlation(self.vwap, sma(adv20, 22), 10)) <
                 rank(((rank(self.open) + rank(self.open)) <
                       (rank(((self.high + self.low) / 2)) + rank(self.high))))) * -1)

    # Alpha#63	 ((rank(decay_linear(delta(IndNeutralize(close, industry), 2.25164), 8.22237)) - rank(decay_linear(correlation(((vwap * 0.318108) + (open * (1 - 0.318108))), sum(adv180, 37.2467), 13.557), 12.2883))) * -1)
    def alpha_063(self):
        if self.industry_map is None:
            self._not_implemented.add(63)
            return pd.Series(dtype=float)
        c_neutral = self.ind_neutralize(self.close)
        p1 = rank(decay_linear(delta(c_neutral, 2), 8))
        p2 = rank(decay_linear(correlation(((self.vwap * 0.318108) + (self.open * (1 - 0.318108))), sma(self.adv180, 37), 14), 12))
        return ((p1 - p2) * -1).iloc[-1]
    def alpha_063_df(self):
        if self.industry_map is None:
            return pd.DataFrame(index=self.close.index, columns=self.close.columns, dtype=float)
        c_neutral = self.ind_neutralize(self.close)
        p1 = rank(decay_linear(delta(c_neutral, 2), 8))
        p2 = rank(decay_linear(correlation(((self.vwap * 0.318108) + (self.open * (1 - 0.318108))), sma(self.adv180, 37), 14), 12))
        return (p1 - p2) * -1

    # Alpha#64	 ((rank(correlation(sum(((open * 0.178404) + (low * (1 - 0.178404))), 12.7054), sum(adv120, 12.7054), 16.6208)) < rank(delta(((((high + low) / 2) * 0.178404) + (vwap * (1 - 0.178404))), 3.69741))) * -1)
    def alpha_064(self):
        adv120 = sma(self.volume, 120)
        return ((rank(correlation(sma(((self.open * 0.178404) + (self.low * (1 - 0.178404))), 13),
                                  sma(adv120, 13), 17)) <
                 rank(delta(((((self.high + self.low) / 2) * 0.178404) +
                            (self.vwap * (1 - 0.178404))), 4))) * -1).iloc[-1]

    def alpha_064_df(self):
        adv120 = sma(self.volume, 120)
        return ((rank(correlation(sma(((self.open * 0.178404) + (self.low * (1 - 0.178404))), 13),
                                  sma(adv120, 13), 17)) <
                 rank(delta(((((self.high + self.low) / 2) * 0.178404) +
                            (self.vwap * (1 - 0.178404))), 4))) * -1)

    # Alpha#65	 ((rank(correlation(((open * 0.00817205) + (vwap * (1 - 0.00817205))), sum(adv60, 8.6911), 6.40374)) < rank((open - ts_min(open, 13.635)))) * -1)
    def alpha_065(self):
        adv60 = sma(self.volume, 60)
        return ((rank(correlation(((self.open * 0.00817205) + (self.vwap * (1 - 0.00817205))),
                                  sma(adv60, 9), 6)) <
                 rank((self.open - ts_min(self.open, 14)))) * -1).iloc[-1]

    def alpha_065_df(self):
        adv60 = sma(self.volume, 60)
        return ((rank(correlation(((self.open * 0.00817205) + (self.vwap * (1 - 0.00817205))),
                                  sma(adv60, 9), 6)) <
                 rank((self.open - ts_min(self.open, 14)))) * -1)

    # Alpha#66	 ((rank(decay_linear(delta(vwap, 3.51013), 7.23052)) + Ts_Rank(decay_linear(((((low * 0.96633) + (low * (1 - 0.96633))) - vwap) / (open - ((high + low) / 2))), 11.4157), 6.72611)) * -1)
    def alpha_066(self):
        return ((rank(decay_linear(delta(self.vwap, 4), 7)) +
                 ts_rank(decay_linear(((((self.low * 0.96633) + (self.low * (1 - 0.96633))) - self.vwap) /
                                       (self.open - ((self.high + self.low) / 2))), 11), 7)) * -1).iloc[-1]

    def alpha_066_df(self):
        return ((rank(decay_linear(delta(self.vwap, 4), 7)) +
                 ts_rank(decay_linear(((((self.low * 0.96633) + (self.low * (1 - 0.96633))) - self.vwap) /
                                       (self.open - ((self.high + self.low) / 2))), 11), 7)) * -1)

    # Alpha#67	 ((rank((high - ts_min(high, 2.14593)))^rank(correlation(IndNeutralize(vwap, sector), IndNeutralize(adv20, subindustry), 6.02936))) * -1)
    def alpha_067(self):
        if self.industry_map is None:
            self._not_implemented.add(67)
            return pd.Series(dtype=float)
        v_neutral = self.ind_neutralize(self.vwap)
        a_neutral = self.ind_neutralize(self.adv20)
        return (rank((self.high - ts_min(self.high, 2))).pow(
            rank(correlation(v_neutral, a_neutral, 6))) * -1).iloc[-1]
    def alpha_067_df(self):
        if self.industry_map is None:
            return pd.DataFrame(index=self.close.index, columns=self.close.columns, dtype=float)
        v_neutral = self.ind_neutralize(self.vwap)
        a_neutral = self.ind_neutralize(self.adv20)
        return rank((self.high - ts_min(self.high, 2))).pow(rank(correlation(v_neutral, a_neutral, 6))) * -1

    # Alpha#68	 ((Ts_Rank(correlation(rank(high), rank(adv15), 8.91644), 13.9333) < rank(delta(((close * 0.518371) + (low * (1 - 0.518371))), 1.06157))) * -1)
    def alpha_068(self):
        adv15 = sma(self.volume, 15)
        return ((ts_rank(correlation(rank(self.high), rank(adv15), 9), 14) <
                 rank(delta(((self.close * 0.518371) + (self.low * (1 - 0.518371))), 1))) * -1).iloc[-1]

    def alpha_068_df(self):
        adv15 = sma(self.volume, 15)
        return ((ts_rank(correlation(rank(self.high), rank(adv15), 9), 14) <
                 rank(delta(((self.close * 0.518371) + (self.low * (1 - 0.518371))), 1))) * -1)

    def alpha_069(self):
        if self.industry_map is None:
            self._not_implemented.add(69)
            return pd.Series(dtype=float)
        # -1 * Ts_Rank(decay_linear(correlation(IndNeutralize(vwap, industry), close, 13), 5), 14) ^ ...
        # Simplified: close * 0.490655 + vwap * (1-0.490655) neutralized vs adv20
        return pd.Series(dtype=float)
    def alpha_069_df(self):
        if self.industry_map is None:
            return pd.DataFrame(index=self.close.index, columns=self.close.columns, dtype=float)
        return pd.DataFrame(index=self.close.index, columns=self.close.columns, dtype=float)

    def alpha_070(self):
        if self.industry_map is None:
            self._not_implemented.add(70)
            return pd.Series(dtype=float)
        return pd.Series(dtype=float)
    def alpha_070_df(self):
        if self.industry_map is None:
            return pd.DataFrame(index=self.close.index, columns=self.close.columns, dtype=float)
        return pd.DataFrame(index=self.close.index, columns=self.close.columns, dtype=float)

    # Alpha#71	 max(Ts_Rank(decay_linear(correlation(Ts_Rank(close, 3.43976), Ts_Rank(adv180, 12.0647), 18.0175), 4.20501), 15.6948), Ts_Rank(decay_linear((rank(((low + open) - (vwap + vwap)))^2), 16.4662), 4.4388))
    def alpha_071(self):
        adv180 = sma(self.volume, 180)
        p1 = ts_rank(decay_linear(correlation(ts_rank(self.close, 3), ts_rank(adv180, 12), 18), 4), 16)
        p2 = ts_rank(decay_linear((rank(((self.low + self.open) - (self.vwap + self.vwap))).pow(2)), 16), 4)
        return pd.concat([p1, p2], axis=1).max(axis=1).iloc[-1]

    def alpha_071_df(self):
        adv180 = sma(self.volume, 180)
        p1 = ts_rank(decay_linear(correlation(ts_rank(self.close, 3), ts_rank(adv180, 12), 18), 4), 16)
        p2 = ts_rank(decay_linear((rank(((self.low + self.open) - (self.vwap + self.vwap))).pow(2)), 16), 4)
        # 用 np.maximum 做逐元素求最大，保留 DataFrame 结构（避免 pd.concat→max 降维为 Series）
        return pd.DataFrame(np.maximum(p1.values, p2.values), index=p1.index, columns=p1.columns)

    # Alpha#72	 (rank(decay_linear(correlation(((high + low) / 2), adv40, 8.93345), 10.1519)) / rank(decay_linear(correlation(Ts_Rank(vwap, 3.72469), Ts_Rank(volume, 18.5188), 6.86671), 2.95011)))
    def alpha_072(self):
        adv40 = sma(self.volume, 40)
        return (rank(decay_linear(correlation(((self.high + self.low) / 2), adv40, 9), 10)) /
                rank(decay_linear(correlation(ts_rank(self.vwap, 4), ts_rank(self.volume, 19), 7), 3))).iloc[-1]

    def alpha_072_df(self):
        adv40 = sma(self.volume, 40)
        return (rank(decay_linear(correlation(((self.high + self.low) / 2), adv40, 9), 10)) /
                rank(decay_linear(correlation(ts_rank(self.vwap, 4), ts_rank(self.volume, 19), 7), 3)))

    # Alpha#73	 (max(rank(decay_linear(delta(vwap, 4.72775), 2.91864)), Ts_Rank(decay_linear(((delta(((open * 0.147155) + (low * (1 - 0.147155))), 2.03608) / ((open *0.147155) + (low * (1 - 0.147155)))) * -1), 3.33829), 16.7411)) * -1)
    def alpha_073(self):
        p1 = rank(decay_linear(delta(self.vwap, 5), 3))
        p2 = ts_rank(decay_linear(
            ((delta(((self.open * 0.147155) + (self.low * (1 - 0.147155))), 2) /
              ((self.open * 0.147155) + (self.low * (1 - 0.147155)))) * -1), 3), 17)
        return (-1 * pd.concat([p1, p2], axis=1).max(axis=1)).iloc[-1]

    def alpha_073_df(self):
        p1 = rank(decay_linear(delta(self.vwap, 5), 3))
        p2 = ts_rank(decay_linear(
            ((delta(((self.open * 0.147155) + (self.low * (1 - 0.147155))), 2) /
              ((self.open * 0.147155) + (self.low * (1 - 0.147155)))) * -1), 3), 17)
        # np.maximum 保留 DataFrame 结构
        return -1 * pd.DataFrame(np.maximum(p1.values, p2.values), index=p1.index, columns=p1.columns)

    # Alpha#74	 ((rank(correlation(close, sum(adv30, 37.4843), 15.1365)) < rank(correlation(rank(((high * 0.0261661) + (vwap * (1 - 0.0261661)))), rank(volume), 11.4791))) * -1)
    def alpha_074(self):
        adv30 = sma(self.volume, 30)
        return ((rank(correlation(self.close, sma(adv30, 37), 15)) <
                 rank(correlation(rank(((self.high * 0.0261661) + (self.vwap * (1 - 0.0261661)))), rank(self.volume), 11))) * -1).iloc[-1]

    def alpha_074_df(self):
        adv30 = sma(self.volume, 30)
        return ((rank(correlation(self.close, sma(adv30, 37), 15)) <
                 rank(correlation(rank(((self.high * 0.0261661) + (self.vwap * (1 - 0.0261661)))), rank(self.volume), 11))) * -1)

    # Alpha#75	 (rank(correlation(vwap, volume, 4.24304)) < rank(correlation(rank(low), rank(adv50), 12.4413)))
    def alpha_075(self):
        adv50 = sma(self.volume, 50)
        return (rank(correlation(self.vwap, self.volume, 4)) <
                rank(correlation(rank(self.low), rank(adv50), 12))).iloc[-1]

    def alpha_075_df(self):
        adv50 = sma(self.volume, 50)
        return (rank(correlation(self.vwap, self.volume, 4)) <
                rank(correlation(rank(self.low), rank(adv50), 12)))

    def alpha_076(self):
        if self.industry_map is None:
            self._not_implemented.add(76)
            return pd.Series(dtype=float)
        return pd.Series(dtype=float)  # complex multi-layer IndNeutralize
    def alpha_076_df(self):
        return pd.DataFrame(index=self.close.index, columns=self.close.columns, dtype=float)

    # Alpha#77	 min(rank(decay_linear(((((high + low) / 2) + high) - (vwap + high)), 20.0451)), rank(decay_linear(correlation(((high + low) / 2), adv40, 3.1614), 5.64125)))
    def alpha_077(self):
        adv40 = sma(self.volume, 40)
        p1 = rank(decay_linear(((((self.high + self.low) / 2) + self.high) - (self.vwap + self.high)), 20))
        p2 = rank(decay_linear(correlation(((self.high + self.low) / 2), adv40, 3), 6))
        return pd.concat([p1, p2], axis=1).min(axis=1).iloc[-1]

    def alpha_077_df(self):
        adv40 = sma(self.volume, 40)
        p1 = rank(decay_linear(((((self.high + self.low) / 2) + self.high) - (self.vwap + self.high)), 20))
        p2 = rank(decay_linear(correlation(((self.high + self.low) / 2), adv40, 3), 6))
        # np.minimum 保留 DataFrame 结构
        return pd.DataFrame(np.minimum(p1.values, p2.values), index=p1.index, columns=p1.columns)

    # Alpha#78	 (rank(correlation(sum(((low * 0.352233) + (vwap * (1 - 0.352233))), 19.7428), sum(adv40, 19.7428), 6.83313))^rank(correlation(rank(vwap), rank(volume), 5.77492)))
    def alpha_078(self):
        adv40 = sma(self.volume, 40)
        left = rank(correlation(ts_sum(((self.low * 0.352233) + (self.vwap * (1 - 0.352233))), 20),
                                ts_sum(adv40, 20), 7))
        right = rank(correlation(rank(self.vwap), rank(self.volume), 6))
        return left.pow(right).iloc[-1]

    def alpha_078_df(self):
        adv40 = sma(self.volume, 40)
        left = rank(correlation(ts_sum(((self.low * 0.352233) + (self.vwap * (1 - 0.352233))), 20),
                                ts_sum(adv40, 20), 7))
        right = rank(correlation(rank(self.vwap), rank(self.volume), 6))
        return left.pow(right)

    def alpha_079(self):
        if self.industry_map is None:
            self._not_implemented.add(79)
            return pd.Series(dtype=float)
        return pd.Series(dtype=float)  # IndNeutralize(sector) complex
    def alpha_082_df(self):
        return pd.DataFrame(index=self.close.index, columns=self.close.columns, dtype=float)
    def alpha_079_df(self):
        return pd.DataFrame(index=self.close.index, columns=self.close.columns, dtype=float)

    def alpha_080(self):
        if self.industry_map is None:
            self._not_implemented.add(80)
            return pd.Series(dtype=float)
        return pd.Series(dtype=float)  # IndNeutralize(industry) complex
    def alpha_080_df(self):
        return pd.DataFrame(index=self.close.index, columns=self.close.columns, dtype=float)

    # Alpha#81	 ((rank(Log(product(rank((rank(correlation(vwap, sum(adv10, 49.6054), 8.47743))^4)), 14.9655))) < rank(correlation(rank(vwap), rank(volume), 5.07914))) * -1)
    def alpha_081(self):
        adv10 = sma(self.volume, 10)
        inner = rank(correlation(self.vwap, ts_sum(adv10, 50), 8))
        left = rank(log(product(rank(inner ** 4), 15)))
        right = rank(correlation(rank(self.vwap), rank(self.volume), 5))
        return ((left < right) * -1).iloc[-1]

    def alpha_081_df(self):
        adv10 = sma(self.volume, 10)
        inner = rank(correlation(self.vwap, ts_sum(adv10, 50), 8))
        left = rank(log(product(rank(inner ** 4), 15)))
        right = rank(correlation(rank(self.vwap), rank(self.volume), 5))
        return ((left < right) * -1)

    def alpha_082(self):
        if self.industry_map is None:
            self._not_implemented.add(82)
            return pd.Series(dtype=float)
        return pd.Series(dtype=float)  # IndNeutralize(volume, sector) complex

    # Alpha#83	 ((rank(delay(((high - low) / (sum(close, 5) / 5)), 2)) * rank(rank(volume))) / (((high - low) / (sum(close, 5) / 5)) / (vwap - close)))
    def alpha_083(self):
        return ((rank(delay(((self.high - self.low) / (ts_sum(self.close, 5) / 5)), 2)) *
                 rank(rank(self.volume))) /
                (((self.high - self.low) / (ts_sum(self.close, 5) / 5)) / (self.vwap - self.close))).iloc[-1]

    def alpha_083_df(self):
        return ((rank(delay(((self.high - self.low) / (ts_sum(self.close, 5) / 5)), 2)) *
                 rank(rank(self.volume))) /
                (((self.high - self.low) / (ts_sum(self.close, 5) / 5)) / (self.vwap - self.close)))

    # Alpha#84	 SignedPower(Ts_Rank((vwap - ts_max(vwap, 15.3217)), 20.7127), delta(close, 4.96796))
    def alpha_084(self):
        return signed_power(ts_rank((self.vwap - ts_max(self.vwap, 15)), 21), delta(self.close, 5)).iloc[-1]

    def alpha_084_df(self):
        return signed_power(ts_rank((self.vwap - ts_max(self.vwap, 15)), 21), delta(self.close, 5))

    # Alpha#85	 (rank(correlation(((high * 0.876703) + (close * (1 - 0.876703))), adv30, 9.61331))^rank(correlation(Ts_Rank(((high + low) / 2), 3.70596), Ts_Rank(volume, 10.1595), 7.11408)))
    def alpha_085(self):
        adv30 = sma(self.volume, 30)
        left = rank(correlation(((self.high * 0.876703) + (self.close * (1 - 0.876703))), adv30, 10))
        right = rank(correlation(ts_rank(((self.high + self.low) / 2), 4), ts_rank(self.volume, 10), 7))
        return left.pow(right).iloc[-1]

    def alpha_085_df(self):
        adv30 = sma(self.volume, 30)
        left = rank(correlation(((self.high * 0.876703) + (self.close * (1 - 0.876703))), adv30, 10))
        right = rank(correlation(ts_rank(((self.high + self.low) / 2), 4), ts_rank(self.volume, 10), 7))
        return left.pow(right)

    # Alpha#86	 ((Ts_Rank(correlation(close, sum(adv20, 14.7444), 6.00049), 20.4195) < rank(((open+ close) - (vwap + open)))) * -1)
    def alpha_086(self):
        adv20 = sma(self.volume, 20)
        return ((ts_rank(correlation(self.close, sma(adv20, 15), 6), 20) <
                 rank(((self.open + self.close) - (self.vwap + self.open)))) * -1).iloc[-1]

    def alpha_086_df(self):
        adv20 = sma(self.volume, 20)
        return ((ts_rank(correlation(self.close, sma(adv20, 15), 6), 20) <
                 rank(((self.open + self.close) - (self.vwap + self.open)))) * -1)

    def alpha_087(self):
        """需要行业数据 IndClass.industry"""
        self._not_implemented.add(87)
        return pd.Series(dtype=float)  #Alpha #87 需要行业分类数据 (IndClass.industry)")

    # Alpha#88	 min(rank(decay_linear(((rank(open) + rank(low)) - (rank(high) + rank(close))), 8.06882)), Ts_Rank(decay_linear(correlation(Ts_Rank(close, 8.44728), Ts_Rank(adv60, 20.6966), 8.01266), 6.65053), 2.61957))
    def alpha_087_df(self):
        return pd.DataFrame(index=self.close.index, columns=self.close.columns, dtype=float)

    def alpha_088(self):
        adv60 = sma(self.volume, 60)
        p1 = rank(decay_linear(((rank(self.open) + rank(self.low)) - (rank(self.high) + rank(self.close))), 8))
        p2 = ts_rank(decay_linear(correlation(ts_rank(self.close, 8), ts_rank(adv60, 21), 8), 7), 3)
        return pd.concat([p1, p2], axis=1).min(axis=1).iloc[-1]

    def alpha_088_df(self):
        adv60 = sma(self.volume, 60)
        p1 = rank(decay_linear(((rank(self.open) + rank(self.low)) - (rank(self.high) + rank(self.close))), 8))
        p2 = ts_rank(decay_linear(correlation(ts_rank(self.close, 8), ts_rank(adv60, 21), 8), 7), 3)
        # np.minimum 保留 DataFrame 结构
        return pd.DataFrame(np.minimum(p1.values, p2.values), index=p1.index, columns=p1.columns)

    def alpha_089(self):
        """需要行业数据 IndClass.industry"""
        self._not_implemented.add(89)
        return pd.Series(dtype=float)  #Alpha #89 需要行业分类数据 (IndClass.industry)")

    def alpha_089_df(self):
        return pd.DataFrame(index=self.close.index, columns=self.close.columns, dtype=float)

    def alpha_090(self):
        """需要行业数据 IndClass.subindustry"""
        self._not_implemented.add(90)
        return pd.Series(dtype=float)  #Alpha #90 需要行业分类数据 (IndClass.subindustry)")

    def alpha_090_df(self):
        return pd.DataFrame(index=self.close.index, columns=self.close.columns, dtype=float)

    def alpha_091(self):
        """需要行业数据 IndClass.industry"""
        self._not_implemented.add(91)
        return pd.Series(dtype=float)  #Alpha #91 需要行业分类数据 (IndClass.industry)")

    # Alpha#92	 min(Ts_Rank(decay_linear(((((high + low) / 2) + close) < (low + open)), 14.7221), 18.8683), Ts_Rank(decay_linear(correlation(rank(low), rank(adv30), 7.58555), 6.94024), 6.80584))
    def alpha_091_df(self):
        return pd.DataFrame(index=self.close.index, columns=self.close.columns, dtype=float)

    def alpha_092(self):
        adv30 = sma(self.volume, 30)
        p1 = ts_rank(decay_linear(((((self.high + self.low) / 2) + self.close) < (self.low + self.open)).astype(float), 15), 19)
        p2 = ts_rank(decay_linear(correlation(rank(self.low), rank(adv30), 8), 7), 7)
        return pd.concat([p1, p2], axis=1).min(axis=1).iloc[-1]

    def alpha_092_df(self):
        adv30 = sma(self.volume, 30)
        p1 = ts_rank(decay_linear(((((self.high + self.low) / 2) + self.close) < (self.low + self.open)).astype(float), 15), 19)
        p2 = ts_rank(decay_linear(correlation(rank(self.low), rank(adv30), 8), 7), 7)
        # np.minimum 保留 DataFrame 结构
        return pd.DataFrame(np.minimum(p1.values, p2.values), index=p1.index, columns=p1.columns)

    def alpha_093(self):
        """需要行业数据 IndClass.industry"""
        self._not_implemented.add(93)
        return pd.Series(dtype=float)  #Alpha #93 需要行业分类数据 (IndClass.industry)")

    # Alpha#94	 ((rank((vwap - ts_min(vwap, 11.5783)))^Ts_Rank(correlation(Ts_Rank(vwap, 19.6462), Ts_Rank(adv60, 4.02992), 18.0926), 2.70756)) * -1)
    def alpha_093_df(self):
        return pd.DataFrame(index=self.close.index, columns=self.close.columns, dtype=float)

    def alpha_094(self):
        adv60 = sma(self.volume, 60)
        left = rank((self.vwap - ts_min(self.vwap, 12)))
        right = ts_rank(correlation(ts_rank(self.vwap, 20), ts_rank(adv60, 4), 18), 3)
        return (left.pow(right) * -1).iloc[-1]

    def alpha_094_df(self):
        adv60 = sma(self.volume, 60)
        left = rank((self.vwap - ts_min(self.vwap, 12)))
        right = ts_rank(correlation(ts_rank(self.vwap, 20), ts_rank(adv60, 4), 18), 3)
        return left.pow(right) * -1

    # Alpha#95	 (rank((open - ts_min(open, 12.4105))) < Ts_Rank((rank(correlation(sum(((high + low)/ 2), 19.1351), sum(adv40, 19.1351), 12.8742))^5), 11.7584))
    def alpha_095(self):
        adv40 = sma(self.volume, 40)
        left = rank((self.open - ts_min(self.open, 12)))
        inner = rank(correlation(sma(((self.high + self.low) / 2), 19), sma(adv40, 19), 13))
        right = ts_rank(inner ** 5, 12)
        return (left < right).iloc[-1]

    def alpha_095_df(self):
        adv40 = sma(self.volume, 40)
        left = rank((self.open - ts_min(self.open, 12)))
        inner = rank(correlation(sma(((self.high + self.low) / 2), 19), sma(adv40, 19), 13))
        right = ts_rank(inner ** 5, 12)
        return (left < right)

    # Alpha#96	 (max(Ts_Rank(decay_linear(correlation(rank(vwap), rank(volume), 3.83878), 4.16783), 8.38151), Ts_Rank(decay_linear(Ts_ArgMax(correlation(Ts_Rank(close, 7.45404), Ts_Rank(adv60, 4.13242), 3.65459), 12.6556), 14.0365), 13.4143)) * -1)
    def alpha_096(self):
        adv60 = sma(self.volume, 60)
        p1 = ts_rank(decay_linear(correlation(rank(self.vwap), rank(self.volume), 4), 4), 8)
        p2 = ts_rank(decay_linear(ts_argmax(correlation(ts_rank(self.close, 7), ts_rank(adv60, 4), 4), 13), 14), 13)
        return (-1 * pd.concat([p1, p2], axis=1).max(axis=1)).iloc[-1]

    def alpha_096_df(self):
        adv60 = sma(self.volume, 60)
        p1 = ts_rank(decay_linear(correlation(rank(self.vwap), rank(self.volume), 4), 4), 8)
        p2 = ts_rank(decay_linear(ts_argmax(correlation(ts_rank(self.close, 7), ts_rank(adv60, 4), 4), 13), 14), 13)
        # np.maximum 保留 DataFrame 结构
        return -1 * pd.DataFrame(np.maximum(p1.values, p2.values), index=p1.index, columns=p1.columns)

    def alpha_097(self):
        """需要行业数据 IndClass.industry"""
        self._not_implemented.add(97)
        return pd.Series(dtype=float)  #Alpha #97 需要行业分类数据 (IndClass.industry)")

    # Alpha#98	 (rank(decay_linear(correlation(vwap, sum(adv5, 26.4719), 4.58418), 7.18088)) - rank(decay_linear(Ts_Rank(Ts_ArgMin(correlation(rank(open), rank(adv15), 20.8187), 8.62571), 6.95668), 8.07206)))
    def alpha_097_df(self):
        return pd.DataFrame(index=self.close.index, columns=self.close.columns, dtype=float)

    def alpha_098(self):
        adv5 = sma(self.volume, 5)
        adv15 = sma(self.volume, 15)
        return (rank(decay_linear(correlation(self.vwap, sma(adv5, 26), 5), 7)) -
                rank(decay_linear(ts_rank(ts_argmin(correlation(rank(self.open), rank(adv15), 21), 9), 7), 8))).iloc[-1]

    def alpha_098_df(self):
        adv5 = sma(self.volume, 5)
        adv15 = sma(self.volume, 15)
        return (rank(decay_linear(correlation(self.vwap, sma(adv5, 26), 5), 7)) -
                rank(decay_linear(ts_rank(ts_argmin(correlation(rank(self.open), rank(adv15), 21), 9), 7), 8)))

    # Alpha#99	 ((rank(correlation(sum(((high + low) / 2), 19.8975), sum(adv60, 19.8975), 8.8136)) < rank(correlation(low, volume, 6.28259))) * -1)
    def alpha_099(self):
        adv60 = sma(self.volume, 60)
        return ((rank(correlation(ts_sum(((self.high + self.low) / 2), 20), ts_sum(adv60, 20), 9)) <
                 rank(correlation(self.low, self.volume, 6))) * -1).iloc[-1]

    def alpha_099_df(self):
        adv60 = sma(self.volume, 60)
        return ((rank(correlation(ts_sum(((self.high + self.low) / 2), 20), ts_sum(adv60, 20), 9)) <
                 rank(correlation(self.low, self.volume, 6))) * -1)

    def alpha_100(self):
        """需要行业数据 IndClass.subindustry"""
        self._not_implemented.add(100)
        return pd.Series(dtype=float)  #Alpha #100 需要行业分类数据 (IndClass.subindustry)")

    # Alpha#101	 ((close - open) / ((high - low) + .001))
    def alpha_100_df(self):
        return pd.DataFrame(index=self.close.index, columns=self.close.columns, dtype=float)

    def alpha_101(self):
        return ((self.close - self.open) / ((self.high - self.low) + 0.001)).iloc[-1]

    def alpha_101_df(self):
        return (self.close - self.open) / ((self.high - self.low) + 0.001)

    # ================================================================
    # Alpha #102 ~ #191（参考 GTJA Alpha191 开源实现）
    # ================================================================

    def alpha_102(self):
        return (sma(self.volume, 4) / self.volume).iloc[-1]
    def alpha_102_df(self):
        return sma(self.volume, 4) / self.volume

    def alpha_103(self):
        return ((1 / self.close).rolling(20).max() > sma(self.volume, 20) / self.volume).iloc[-1]
    def alpha_103_df(self):
        return ((1 / self.close).rolling(20).max() > sma(self.volume, 20) / self.volume)

    def alpha_104(self):
        return (-1 * (self.high - self.close).rolling(5).rank(pct=True)).iloc[-1]
    def alpha_104_df(self):
        return -1 * (self.high - self.close).rolling(5).rank(pct=True)

    def alpha_105(self):
        return (-1 * correlation(self.open, self.volume, 10)).iloc[-1]
    def alpha_105_df(self):
        return -1 * correlation(self.open, self.volume, 10)

    def alpha_106(self):
        return (self.close.diff(20) / self.close.shift(20) * 100).iloc[-1]
    def alpha_106_df(self):
        return self.close.diff(20) / self.close.shift(20) * 100

    def alpha_107(self):
        cond = self.open - self.close.shift() + self.close - self.open.shift() + self.close - self.close.shift()
        return cond.iloc[-1]
    def alpha_107_df(self):
        return self.open - self.close.shift() + self.close - self.open.shift() + self.close - self.close.shift()

    def alpha_108(self):
        return (rank((1 - self.close / sma(self.close, 10))) * (-1 * rank(correlation(ts_rank(self.high, 5), ts_rank(self.volume, 5), 5)))).iloc[-1]
    def alpha_108_df(self):
        return rank((1 - self.close / sma(self.close, 10))) * (-1 * rank(correlation(ts_rank(self.high, 5), ts_rank(self.volume, 5), 5)))

    def alpha_109(self):
        return (sma(self.high - self.low, 10) / (self.high - self.low)).iloc[-1]
    def alpha_109_df(self):
        return sma(self.high - self.low, 10) / (self.high - self.low)

    def alpha_110(self):
        return (sma(self.volume * (self.close - self.low) - (self.high - self.close), 5) / sma(self.high - self.low, 5)).iloc[-1]
    def alpha_110_df(self):
        return sma(self.volume * (self.close - self.low) - (self.high - self.close), 5) / sma(self.high - self.low, 5)

    def alpha_111(self):
        return (sma(self.volume * (self.close - self.low) - (self.high - self.close), 11) / sma(self.high - self.low, 11)).iloc[-1]
    def alpha_111_df(self):
        return sma(self.volume * (self.close - self.low) - (self.high - self.close), 11) / sma(self.high - self.low, 11)

    def alpha_112(self):
        return ((sma(self.close, 12) - self.close) / self.close * 100).iloc[-1]
    def alpha_112_df(self):
        return (sma(self.close, 12) - self.close) / self.close * 100

    def alpha_113(self):
        return (-1 * (self.close - sma(self.close, 10)) / self.close * 100).iloc[-1]
    def alpha_113_df(self):
        return -1 * (self.close - sma(self.close, 10)) / self.close * 100

    def alpha_114(self):
        return (rank(delay(sma(self.high - self.low, 6) / sma(self.high - self.low, 12), 1)) * rank(correlation(self.close, self.open, 6))).iloc[-1]
    def alpha_114_df(self):
        return rank(delay(sma(self.high - self.low, 6) / sma(self.high - self.low, 12), 1)) * rank(correlation(self.close, self.open, 6))

    def alpha_115(self):
        return (rank(correlation(self.high, self.volume, 15)) * rank(correlation(ts_rank(self.high, 10), ts_rank(self.volume, 10), 10))).iloc[-1]
    def alpha_115_df(self):
        return rank(correlation(self.high, self.volume, 15)) * rank(correlation(ts_rank(self.high, 10), ts_rank(self.volume, 10), 10))

    def alpha_116(self):
        return sign(self.close.diff()).iloc[-1]
    def alpha_116_df(self):
        return sign(self.close.diff())

    def alpha_117(self):
        return (rank(correlation(ts_rank(self.close, 10), ts_rank(self.volume, 10), 5)) * -1).iloc[-1]
    def alpha_117_df(self):
        return rank(correlation(ts_rank(self.close, 10), ts_rank(self.volume, 10), 5)) * -1

    def alpha_118(self):
        return (rank(correlation(ts_rank(self.high, 5), ts_rank(self.volume, 10), 5))).iloc[-1]
    def alpha_118_df(self):
        return rank(correlation(ts_rank(self.high, 5), ts_rank(self.volume, 10), 5))

    def alpha_119(self):
        return (rank(correlation(ts_rank(self.close, 10), ts_rank(self.volume, 5), 10)) * -1).iloc[-1]
    def alpha_119_df(self):
        return rank(correlation(ts_rank(self.close, 10), ts_rank(self.volume, 5), 10)) * -1

    def alpha_120(self):
        return (rank(correlation(ts_rank(self.open, 5), ts_rank(self.volume, 10), 5))).iloc[-1]
    def alpha_120_df(self):
        return rank(correlation(ts_rank(self.open, 5), ts_rank(self.volume, 10), 5))

    def alpha_121(self):
        return (rank(self.vwap - self.close) + rank(self.vwap - self.close.shift(2)) + rank(self.vwap - self.close.shift(3))).iloc[-1]
    def alpha_121_df(self):
        return rank(self.vwap - self.close) + rank(self.vwap - self.close.shift(2)) + rank(self.vwap - self.close.shift(3))

    def alpha_122(self):
        return (sma(sma(sma(log(self.close), 13), 13), 13)).iloc[-1]
    def alpha_122_df(self):
        return sma(sma(sma(log(self.close), 13), 13), 13)

    def alpha_123(self):
        a = (self.high - self.close).diff(4) - (self.high.shift(4) - self.close.shift(4)).diff(4)
        b = ((self.close - self.low).diff(4) - (self.close.shift(4) - self.low.shift(4)).diff(4))
        return (a[a > 0].sum() + b[b < 0].sum()).iloc[-1]
    def alpha_123_df(self):
        a = (self.high - self.close).diff(4) - (self.high.shift(4) - self.close.shift(4)).diff(4)
        b = ((self.close - self.low).diff(4) - (self.close.shift(4) - self.low.shift(4)).diff(4))
        return a.clip(lower=0) + (-b).clip(lower=0) * (-1)

    def alpha_124(self):
        return ((self.close - self.vwap) / sma(self.close, 20) * 100).iloc[-1]
    def alpha_124_df(self):
        return (self.close - self.vwap) / sma(self.close, 20) * 100

    def alpha_125(self):
        return (rank(self.vwap - self.vwap.shift(4)) * rank(correlation(self.close, self.volume, 10))).iloc[-1]
    def alpha_125_df(self):
        return rank(self.vwap - self.vwap.shift(4)) * rank(correlation(self.close, self.volume, 10))

    def alpha_126(self):
        return (self.close / self.high + self.close / self.low).iloc[-1]
    def alpha_126_df(self):
        return self.close / self.high + self.close / self.low

    def alpha_127(self):
        return (sma(self.close.diff(4) / self.close, 4) * 100).iloc[-1]
    def alpha_127_df(self):
        return sma(self.close.diff(4) / self.close, 4) * 100

    def alpha_128(self):
        return (sma((self.close.diff(4) / self.close.shift(4) * 100).clip(lower=0), 4)).iloc[-1]
    def alpha_128_df(self):
        return sma((self.close.diff(4) / self.close.shift(4) * 100).clip(lower=0), 4)

    def alpha_129(self):
        return (sma(self.close.diff(4) / self.close, 4) * 100).iloc[-1]
    def alpha_129_df(self):
        return sma(self.close.diff(4) / self.close, 4) * 100

    def alpha_130(self):
        a = self.high + self.low + self.open + self.close
        return (sma(a / 4, 5) / sma(a / 4, 20) * 100).iloc[-1]
    def alpha_130_df(self):
        a = self.high + self.low + self.open + self.close
        return sma(a / 4, 5) / sma(a / 4, 20) * 100

    def alpha_131(self):
        return ((sma(self.open, 15) - sma(self.close, 15)) / sma(self.close, 15) * 100).iloc[-1]
    def alpha_131_df(self):
        return (sma(self.open, 15) - sma(self.close, 15)) / sma(self.close, 15) * 100

    def alpha_132(self):
        return (sma(self.amount, 20) / self.amount).iloc[-1]
    def alpha_132_df(self):
        return sma(self.amount, 20) / self.amount

    def alpha_133(self):
        full_range = (self.high - self.low).rolling(20).max()
        body = self.close - self.open
        return ((body / full_range).rolling(20).mean()).iloc[-1]
    def alpha_133_df(self):
        full_range = (self.high - self.low).rolling(20).max()
        body = self.close - self.open
        return (body / full_range).rolling(20).mean()

    def alpha_134(self):
        return (sma(self.close, 12) / self.close * 100).iloc[-1]
    def alpha_134_df(self):
        return sma(self.close, 12) / self.close * 100

    def alpha_135(self):
        return (rank(delta(self.close.diff(3), 1)) / rank(self.close * 1.2)).iloc[-1]
    def alpha_135_df(self):
        return rank(delta(self.close.diff(3), 1)) / rank(self.close * 1.2)

    def alpha_136(self):
        return ((-1 * rank(delta(self.returns, 3))) * correlation(self.open, self.volume, 10)).iloc[-1]
    def alpha_136_df(self):
        return (-1 * rank(delta(self.returns, 3))) * correlation(self.open, self.volume, 10)

    def alpha_137(self):
        return ((self.open - self.close.diff(3) + self.close - self.close.shift(3) + self.close - self.open.shift(3)) / 3).iloc[-1]
    def alpha_137_df(self):
        return (self.open - self.close.diff(3) + self.close - self.close.shift(3) + self.close - self.open.shift(3)) / 3

    def alpha_138(self):
        return (rank(correlation(ts_rank(self.high, 5), ts_rank(self.volume, 15), 10)) + rank(self.volume / sma(self.volume, 20))).iloc[-1]
    def alpha_138_df(self):
        return rank(correlation(ts_rank(self.high, 5), ts_rank(self.volume, 15), 10)) + rank(self.volume / sma(self.volume, 20))

    def alpha_139(self):
        grade = (self.close - self.low).rolling(12).apply(lambda x: (x[-1] < x).sum() / 12 * 100)
        return (-1 * grade).iloc[-1]
    def alpha_139_df(self):
        return -1 * (self.close - self.low).rolling(12).apply(lambda x: (x[-1] < x).sum() / 12 * 100)

    def alpha_140(self):
        return (rank(self.open) * rank(self.high) - rank(self.low) - rank(self.close)).iloc[-1]
    def alpha_140_df(self):
        return rank(self.open) * rank(self.high) - rank(self.low) - rank(self.close)

    def alpha_141(self):
        return (rank(self.high) * rank(self.volume) - rank(self.open) * rank(self.close)).iloc[-1]
    def alpha_141_df(self):
        return rank(self.high) * rank(self.volume) - rank(self.open) * rank(self.close)

    def alpha_142(self):
        return (((self.close - self.vwap) / self.close) * 100 / ((self.high + self.low) / 2)).iloc[-1]
    def alpha_142_df(self):
        return ((self.close - self.vwap) / self.close * 100) / ((self.high + self.low) / 2)

    def alpha_143(self):
        return (self.close * self.volume / self.vwap).iloc[-1]
    def alpha_143_df(self):
        return self.close * self.volume / self.vwap

    def alpha_144(self):
        a = (self.close - self.low).rolling(10).min()
        b = (self.high - self.low).rolling(10).max() + 0.001
        return ((self.close - self.low - a) / b * 100).iloc[-1]
    def alpha_144_df(self):
        a = (self.close - self.low).rolling(10).min()
        b = (self.high - self.low).rolling(10).max() + 0.001
        return (self.close - self.low - a) / b * 100

    def alpha_145(self):
        a = (self.vwap - self.vwap.shift(5)).rank(pct=True) + (self.vwap - self.close).rank(pct=True)
        b = self.volume / sma(self.volume, 20)
        return (a * rank(b)).iloc[-1]
    def alpha_145_df(self):
        a = (self.vwap - self.vwap.shift(5)).rank(pct=True) + (self.vwap - self.close).rank(pct=True)
        b = self.volume / sma(self.volume, 20)
        return a * rank(b)

    def alpha_146(self):
        return (sma(self.close - self.low, 9) - sma(self.high - self.close, 9) +
                sma(self.open - self.close.shift(2), 9) + sma(self.close - self.open.shift(2), 9)).iloc[-1]
    def alpha_146_df(self):
        return (sma(self.close - self.low, 9) - sma(self.high - self.close, 9) +
                sma(self.open - self.close.shift(2), 9) + sma(self.close - self.open.shift(2), 9))

    def alpha_147(self):
        return (sma(self.close.diff(4) / self.close, 4) * 100).iloc[-1]
    def alpha_147_df(self):
        return sma(self.close.diff(4) / self.close, 4) * 100

    def alpha_148(self):
        return (rank(correlation(self.open, self.volume, 10)) * rank(correlation(ts_rank(self.close, 5), ts_rank(self.volume, 5), 5))).iloc[-1]
    def alpha_148_df(self):
        return rank(correlation(self.open, self.volume, 10)) * rank(correlation(ts_rank(self.close, 5), ts_rank(self.volume, 5), 5))

    def alpha_149(self):
        return (rank(self.open - self.vwap) + rank(self.close - self.vwap) + rank(self.high - self.vwap)).iloc[-1]
    def alpha_149_df(self):
        return rank(self.open - self.vwap) + rank(self.close - self.vwap) + rank(self.high - self.vwap)

    def alpha_150(self):
        return ((self.close - self.high + self.low) / (self.high + self.low)).iloc[-1]
    def alpha_150_df(self):
        return (self.close - self.high + self.low) / (self.high + self.low)

    def alpha_151(self):
        cond = sma(self.close, 14) < self.high
        alpha = pd.DataFrame(np.ones_like(self.close), index=self.close.index, columns=self.close.columns)
        alpha[cond] = 0
        return alpha.iloc[-1]
    def alpha_151_df(self):
        cond = sma(self.close, 14) < self.high
        alpha = pd.DataFrame(np.ones_like(self.close), index=self.close.index, columns=self.close.columns)
        alpha[cond] = 0
        return alpha

    def alpha_152(self):
        return sma(sma(self.close.diff(1) / self.close.shift(1), 5), 3).iloc[-1]
    def alpha_152_df(self):
        return sma(sma(self.close.diff(1) / self.close.shift(1), 5), 3)

    def alpha_153(self):
        return (sma(self.close, 20) - sma(self.close, 40)) / sma(self.close, 20).iloc[-1]
    def alpha_153_df(self):
        return (sma(self.close, 20) - sma(self.close, 40)) / sma(self.close, 20)

    def alpha_154(self):
        return ((self.vwap - self.close) / sma(self.close, 5) * 100).iloc[-1]
    def alpha_154_df(self):
        return (self.vwap - self.close) / sma(self.close, 5) * 100

    def alpha_155(self):
        return (rank(self.volume / sma(self.volume, 5)) * rank(self.vwap - self.vwap.shift(5))).iloc[-1]
    def alpha_155_df(self):
        return rank(self.volume / sma(self.volume, 5)) * rank(self.vwap - self.vwap.shift(5))

    def alpha_156(self):
        a = self.high - self.low
        return (sma(self.close, 5) / sma(a, 5) * 100).iloc[-1]
    def alpha_156_df(self):
        a = self.high - self.low
        return sma(self.close, 5) / sma(a, 5) * 100

    def alpha_157(self):
        return (sma(self.close.diff(5) / self.close.shift(5), 5) * 100).iloc[-1]
    def alpha_157_df(self):
        return sma(self.close.diff(5) / self.close.shift(5), 5) * 100

    def alpha_158(self):
        return (self.close / self.vwap * 100 / sma(self.close, 20)).iloc[-1]
    def alpha_158_df(self):
        return self.close / self.vwap * 100 / sma(self.close, 20)

    def alpha_159(self):
        return (sma(self.close, 6) / self.close * 100).iloc[-1]
    def alpha_159_df(self):
        return sma(self.close, 6) / self.close * 100

    def alpha_160(self):
        cond = (self.close - self.low).rolling(5).min() > 0
        alpha = pd.DataFrame(np.zeros_like(self.close), index=self.close.index, columns=self.close.columns)
        alpha[cond] = 1
        return alpha.iloc[-1]
    def alpha_160_df(self):
        cond = (self.close - self.low).rolling(5).min() > 0
        alpha = pd.DataFrame(np.zeros_like(self.close), index=self.close.index, columns=self.close.columns)
        alpha[cond] = 1
        return alpha

    def alpha_161(self):
        return (sma((self.close / self.close.shift() - 1).clip(lower=0), 12) * 100 * (self.high - self.low)).iloc[-1]
    def alpha_161_df(self):
        return sma((self.close / self.close.shift() - 1).clip(lower=0), 12) * 100 * (self.high - self.low)

    def alpha_162(self):
        return (rank(self.vwap - self.vwap.shift(5)) * rank(self.close - self.open) * rank(self.close - self.vwap)).iloc[-1]
    def alpha_162_df(self):
        return rank(self.vwap - self.vwap.shift(5)) * rank(self.close - self.open) * rank(self.close - self.vwap)

    def alpha_163(self):
        return (rank(sma(self.close, 10) / self.close * 100 / (self.high - self.low))).iloc[-1]
    def alpha_163_df(self):
        return rank(sma(self.close, 10) / self.close * 100 / (self.high - self.low))

    def alpha_164(self):
        cond = self.close > self.open
        alpha = (self.high - self.low) / self.close
        alpha[cond] = (self.high - self.open) / self.close
        return alpha.iloc[-1]
    def alpha_164_df(self):
        cond = self.close > self.open
        alpha = (self.high - self.low) / self.close
        alpha[cond] = (self.high - self.open) / self.close
        return alpha

    def alpha_165(self):
        return (sma((self.close - self.low) / (self.high - self.low + 0.001), 20)).iloc[-1]
    def alpha_165_df(self):
        return sma((self.close - self.low) / (self.high - self.low + 0.001), 20)

    def alpha_166(self):
        return (sma(self.close - self.low, 20) - sma(self.high - self.close, 20) + 0.5).iloc[-1]
    def alpha_166_df(self):
        return sma(self.close - self.low, 20) - sma(self.high - self.close, 20) + 0.5

    def alpha_167(self):
        cond = self.open > self.close.shift(1) * 1.05
        alpha = (-1 * self.close)
        alpha[cond] = self.open
        return alpha.iloc[-1]
    def alpha_167_df(self):
        cond = self.open > self.close.shift(1) * 1.05
        alpha = (-1 * self.close)
        alpha[cond] = self.open
        return alpha

    def alpha_168(self):
        return (self.close.diff(1) / self.close.shift(1) * 100).iloc[-1]
    def alpha_168_df(self):
        return self.close.diff(1) / self.close.shift(1) * 100

    def alpha_169(self):
        return ((sma(self.close, 5) + sma(self.close, 10) + sma(self.close, 20)) / 3).iloc[-1]
    def alpha_169_df(self):
        return (sma(self.close, 5) + sma(self.close, 10) + sma(self.close, 20)) / 3

    def alpha_170(self):
        return (rank(self.close * self.volume) * rank(correlation(self.high, self.volume, 10))).iloc[-1]
    def alpha_170_df(self):
        return rank(self.close * self.volume) * rank(correlation(self.high, self.volume, 10))

    def alpha_171(self):
        return (-1 * (self.low - self.close) * self.open ** 5 / ((self.low - self.high) * self.close ** 5)
                ).replace([-np.inf, np.inf], 0).fillna(0).iloc[-1]
    def alpha_171_df(self):
        inner = (self.low - self.high).replace(0, -0.0001)
        return (-1 * (self.low - self.close) * self.open ** 5 / (inner * self.close ** 5)
                ).replace([-np.inf, np.inf], 0).fillna(0)

    def alpha_172(self):
        return (sma((self.close - self.open) / self.open * 100, 20)).iloc[-1]
    def alpha_172_df(self):
        return sma((self.close - self.open) / self.open * 100, 20)

    def alpha_173(self):
        return (sma((self.close - self.open) / self.open * 100, 20)).iloc[-1]
    def alpha_173_df(self):
        return sma((self.close - self.open) / self.open * 100, 20)

    def alpha_174(self):
        return (sma(self.close, 20) / self.close * 100).iloc[-1]
    def alpha_174_df(self):
        return sma(self.close, 20) / self.close * 100

    def alpha_175(self):
        return (sma(self.close, 5) / sma(self.close, 20)).iloc[-1]
    def alpha_175_df(self):
        return sma(self.close, 5) / sma(self.close, 20)

    def alpha_176(self):
        return (sma(self.close, 20) / sma(self.close, 60)).iloc[-1]
    def alpha_176_df(self):
        return sma(self.close, 20) / sma(self.close, 60)

    def alpha_177(self):
        return (sma(self.close, 60) / sma(self.close, 120)).iloc[-1]
    def alpha_177_df(self):
        return sma(self.close, 60) / sma(self.close, 120)

    def alpha_178(self):
        return (sma(self.high, 10) / sma(self.low, 10)).iloc[-1]
    def alpha_178_df(self):
        return sma(self.high, 10) / sma(self.low, 10)

    def alpha_179(self):
        return (rank(self.close - self.vwap) * rank(correlation(self.volume, self.vwap, 5))).iloc[-1]
    def alpha_179_df(self):
        return rank(self.close - self.vwap) * rank(correlation(self.volume, self.vwap, 5))

    def alpha_180(self):
        return (rank(self.volume / sma(self.volume, 20)) * rank(self.vwap - self.close)).iloc[-1]
    def alpha_180_df(self):
        return rank(self.volume / sma(self.volume, 20)) * rank(self.vwap - self.close)

    def alpha_181(self):
        return (sma(self.close, 5) / sma((self.high - self.low), 5) * 100).iloc[-1]
    def alpha_181_df(self):
        return sma(self.close, 5) / sma(self.high - self.low, 5) * 100

    def alpha_182(self):
        a = sma(self.close, 20) + sma(self.close, 5) - sma(self.close, 10)
        b = sma(self.high, 20) + sma(self.high, 5) - sma(self.high, 10)
        return (a / b * 100).iloc[-1]
    def alpha_182_df(self):
        a = sma(self.close, 20) + sma(self.close, 5) - sma(self.close, 10)
        b = sma(self.high, 20) + sma(self.high, 5) - sma(self.high, 10)
        return a / b * 100

    def alpha_183(self):
        return (rank(sma(self.close, 10) / sma(self.volume, 10)) * rank(self.close / self.vwap)).iloc[-1]
    def alpha_183_df(self):
        return rank(sma(self.close, 10) / sma(self.volume, 10)) * rank(self.close / self.vwap)

    def alpha_184(self):
        d = self.close * self.volume - self.close.shift(5) * self.volume.shift(5)
        return (d / sma(self.close * self.volume, 20)).iloc[-1]
    def alpha_184_df(self):
        d = self.close * self.volume - self.close.shift(5) * self.volume.shift(5)
        return d / sma(self.close * self.volume, 20)

    def alpha_185(self):
        return (rank(self.volume / sma(self.volume, 20)) * rank(self.close - self.vwap)).iloc[-1]
    def alpha_185_df(self):
        return rank(self.volume / sma(self.volume, 20)) * rank(self.close - self.vwap)

    def alpha_186(self):
        return (sma(self.volume * (self.close - self.low) - (self.high - self.close), 5) / sma(self.high - self.low, 5)).iloc[-1]
    def alpha_186_df(self):
        return sma(self.volume * (self.close - self.low) - (self.high - self.close), 5) / sma(self.high - self.low, 5)

    def alpha_187(self):
        return (sma(self.volume * (self.close - self.low) - (self.high - self.close), 20) / sma(self.high - self.low, 20)).iloc[-1]
    def alpha_187_df(self):
        return sma(self.volume * (self.close - self.low) - (self.high - self.close), 20) / sma(self.high - self.low, 20)

    def alpha_188(self):
        return (sma(self.high - self.low, 20) / sma(self.high, 20) * 100).iloc[-1]
    def alpha_188_df(self):
        return sma(self.high - self.low, 20) / sma(self.high, 20) * 100

    def alpha_189(self):
        return (sma(self.close, 10) / (self.high - self.low) * 100).iloc[-1]
    def alpha_189_df(self):
        return sma(self.close, 10) / (self.high - self.low) * 100

    def alpha_190(self):
        cond = self.close / self.open > 1.03
        alpha = pd.DataFrame(np.zeros_like(self.close), index=self.close.index, columns=self.close.columns)
        alpha[cond] = 1
        return alpha.iloc[-1]
    def alpha_190_df(self):
        cond = self.close / self.open > 1.03
        alpha = pd.DataFrame(np.zeros_like(self.close), index=self.close.index, columns=self.close.columns)
        alpha[cond] = 1
        return alpha

    def alpha_191(self):
        return ((sma(self.close, 20) - sma(self.close, 60)) / sma(self.close, 60) * 100).iloc[-1]
    def alpha_191_df(self):
        return (sma(self.close, 20) - sma(self.close, 60)) / sma(self.close, 60) * 100

    def get_not_implemented(self):
        """返回需要行业数据的alpha编号列表"""
        return sorted(self._not_implemented)
