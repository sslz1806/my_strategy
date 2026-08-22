"""
alpha191/calculator.py - Alpha191 因子计算器（高层接口）

整合数据加载、因子计算和本地因子回测接口。

用法:
    from 因子回测.alpha_191.calculator import Alpha191Calculator
    calc = Alpha191Calculator()
    calc.load_data('2025-01-01', '2025-07-01')

    # 计算单个因子
    alpha5 = calc.compute(5)              # pd.Series, 最新日横截面值
    alpha5_df = calc.compute_df(5)        # pd.DataFrame, 全时段

    # 批量计算
    results = calc.compute_all([1, 5, 10])

    # IC分析（使用 因子回测/alpha.py 的本地接口）
    ic_result = calc.analyze_ic(5, return_period=5)
"""

import sys
import os
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from 因子回测.alpha_191.adapter import load_factor_data, LocalDataAdapter
from 因子回测.alpha_191.alpha_formulas import Alpha191Formulas


class Alpha191Calculator:
    """
    Alpha191 因子计算器

    组合数据加载和因子计算的统一入口。
    """

    def __init__(self, data_dict: Optional[Dict[str, pd.DataFrame]] = None):
        """
        参数:
            data_dict: 可选，预加载的数据字典（来自 LocalDataAdapter）
                       None 则后续需要调用 load_data()
        """
        self.data = data_dict
        self.formulas: Optional[Alpha191Formulas] = None
        self._is_loaded = False

        if data_dict is not None:
            self.formulas = Alpha191Formulas(data_dict)
            self._is_loaded = True

    def load_data(
        self,
        start_date: str,
        end_date: str,
        stock_list: Optional[List[str]] = None,
        min_records: int = 100,
    ) -> None:
        """
        加载本地数据。

        参数:
            start_date: 开始日期 'YYYY-MM-DD'
            end_date: 结束日期 'YYYY-MM-DD'
            stock_list: 股票列表，None=全部
            min_records: 最小记录数
        """
        self.data = load_factor_data(
            start_date=start_date,
            end_date=end_date,
            stock_list=stock_list,
            min_records=min_records,
        )
        self.formulas = Alpha191Formulas(self.data)
        self._is_loaded = True
        print(f"数据加载完成: {self.data['close'].shape[0]}天 × {self.data['close'].shape[1]}只股票")

    def _check_loaded(self):
        if not self._is_loaded:
            raise RuntimeError("请先调用 load_data() 加载数据")

    @property
    def trading_dates(self):
        """交易日列表"""
        self._check_loaded()
        return self.data['close'].index

    @property
    def stock_codes(self):
        """股票代码列表"""
        self._check_loaded()
        return self.data['close'].columns

    def compute(self, alpha_num: int) -> pd.Series:
        """
        计算单个因子（仅最新日横截面值）。

        返回: pd.Series(index=股票代码, values=因子值)
        """
        self._check_loaded()
        return getattr(self.formulas, f'alpha_{alpha_num:03d}')()

    def compute_df(self, alpha_num: int) -> pd.DataFrame:
        """
        计算单个因子（全时段）。

        返回: pd.DataFrame(index=日期, columns=股票代码)
        """
        self._check_loaded()
        return getattr(self.formulas, f'alpha_{alpha_num:03d}_df')()

    def compute_all(
        self,
        alpha_list: Optional[List[int]] = None,
    ) -> Dict[str, pd.Series]:
        """
        批量计算多个因子（最新日值）。

        参数:
            alpha_list: alpha编号列表，None=全部已实现的

        返回: {alpha_name: pd.Series}
        """
        self._check_loaded()
        return self.formulas.compute_all(alpha_list)

    def get_latest_factor_wide(
        self,
        alpha_num: int,
    ) -> pd.DataFrame:
        """
        获取因子当前最新交易日的宽表数据。

        返回格式: 第一列 trading_date，其余列为股票代码
                 可直接传给 因子回测/alpha.py 的 analyze_ic()
        """
        self._check_loaded()
        # 获取全时段因子值
        df = self.compute_df(alpha_num)
        # 只保留最新日
        latest = df.iloc[[-1]].copy()
        latest.insert(0, 'trading_date', latest.index)
        return latest

    def get_factor_wide(
        self,
        alpha_num: int,
    ) -> pd.DataFrame:
        """
        获取因子在全时段的宽表数据。

        返回格式: 第一列 trading_date，其余列为股票代码
        """
        self._check_loaded()
        df = self.compute_df(alpha_num)
        df = df.copy()
        df.insert(0, 'trading_date', df.index)
        return df

    # ================================================================
    # 因子分析集成
    # ================================================================

    def analyze_ic(
        self,
        alpha_num: int,
        return_periods: List[int] = [1, 5, 10, 20],
        adjust_freq: int = 1,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> dict:
        """
        使用本地 因子回测/alpha.py 的 analyze_ic() 进行IC分析。

        参数:
            alpha_num: alpha编号
            return_periods: 收益周期列表
            adjust_freq: 调仓频率（天）
            start_date: 分析开始日期，None=数据开始日期
            end_date: 分析结束日期，None=数据结束日期

        返回:
            {'ic_df': ..., 'ic_decay_df': ...}
        """
        self._check_loaded()
        from 因子回测.alpha import analyze_ic

        # 获取因子宽表
        factor_wide = self.get_factor_wide(alpha_num)

        # 获取日线长表数据
        stock_data = self._get_long_format_stock_data()

        # 确定日期范围
        start = start_date or factor_wide['trading_date'].min().strftime('%Y-%m-%d')
        end = end_date or factor_wide['trading_date'].max().strftime('%Y-%m-%d')

        print(f"\n===== Alpha #{alpha_num} IC分析 =====")
        result = analyze_ic(
            factor_data=factor_wide,
            stock_data=stock_data,
            start_date=start,
            end_date=end,
            adjust_freq=adjust_freq,
            return_periods=return_periods,
            ret_col='returns',  # 使用日收益率列（由 adapter 从 close.pct_change 算得）
        )

        return result

    def analyze_factor(
        self,
        alpha_num: int,
        return_period: int = 5,
        adjust_freq: int = 1,
        group_num: int = 5,
    ) -> dict:
        """
        使用本地 alpha.py 保留的旧宽表接口进行因子分析。

        这是极简版宽表分析，返回 IC/分组收益/净值等。
        """
        self._check_loaded()
        from 因子回测.alpha import analyze_factor_bak

        # 获取因子值（宽表，index=日期, columns=股票）
        factor_df = self.compute_df(alpha_num)

        # 预计算未来N日收益宽表（从收盘价算）
        close_df = self.data['close']
        ret_data = close_df.shift(-return_period) / close_df - 1

        print(f"\n===== Alpha #{alpha_num} 宽表因子分析 =====")
        result = analyze_factor_bak(
            factor_data=factor_df,
            ret_data=ret_data,
            start_date=factor_df.index[0].strftime('%Y-%m-%d'),
            end_date=factor_df.index[-1].strftime('%Y-%m-%d'),
            adjust_freq=adjust_freq,
            return_period=return_period,
            group_num=group_num,
        )

        result['alpha_num'] = alpha_num
        return result

    def _get_long_format_stock_data(self) -> pd.DataFrame:
        """将内部宽表数据转为长格式，供 analyze_ic 使用"""
        close = self.data['close']
        volume = self.data['volume']
        amount = self.data['amount']
        returns = self.data.get('returns')        # 日收益率（adapter 已计算）
        pre_close = self.data.get('pre_close')    # 昨收价

        codes = close.columns.tolist()
        records = []
        for date in close.index:
            for code in codes:
                if pd.notna(close.loc[date, code]):
                    rec = {
                        'trading_date': date,
                        'code': code,
                        'close': close.loc[date, code],
                        'volume': volume.loc[date, code],
                        'amount': amount.loc[date, code],
                    }
                    if returns is not None:
                        rec['returns'] = returns.loc[date, code]
                    if pre_close is not None:
                        rec['pre_close'] = pre_close.loc[date, code]
                    records.append(rec)
        return pd.DataFrame(records)

    def get_not_implemented(self) -> List[int]:
        """返回需要行业数据等附加信息的alpha编号"""
        self._check_loaded()
        return self.formulas.get_not_implemented()
