"""
因子回测/alpha_191/adapter.py - 数据适配器（增强版）

使用本地数据接口(来自 my_utils.fun)读取 A 股日线数据，
转换为 alpha 因子计算所需的宽表格式（pandas DataFrame）。
数据不足时自动通过米筐 API （http://10.140.5.44:6959/） 补充。

主要增强:
  - total_mv / mv_A_free_float 市值数据（本地 parquet 已有）
  - industry 行业分类（本地数据 > 米筐 API 代理）
  - 米筐 API 兜底（本地 parquet 缺字段时自动拉起）

用法:
    from 因子回测.alpha_191.adapter import load_factor_data, load_factor_data_with_industry
    data = load_factor_data('2024-01-01', '2025-07-01')
    # data 包含 open/close/vwap/returns/advXX/total_mv/circulation_mv 等字段

    # 加载含行业分类的数据
    data_w_industry = load_factor_data_with_industry('2024-01-01', '2025-07-01')
    # data_w_industry 额外包含 'industry' 字段（dict: {code: industry_name}）
"""

import sys
import os
import logging
from typing import Dict, Optional, List, Union, Any
from dataclasses import dataclass

import numpy as np
import pandas as pd
import polars as pl

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from my_utils.fun import read_day_data, get_data_trading_days
from my_utils.mapping import convert_code_format

logger = logging.getLogger(__name__)


# ====================================================================
# 米筐 API 代理客户端（惰性初始化）
# ====================================================================

class _RQProxyClient:
    """单例模式 - 米筐 API 代理客户端"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        try:
            from my_utils.rqdata import RQData as RQProxy
            self._rq = RQProxy(username='ly', password='123456')
            self._base_url = "http://10.140.5.44:6959/"
            self._initialized = True
            logger.info("米筐 API 代理客户端初始化成功")
        except Exception as e:
            logger.warning(f"米筐 API 代理客户端初始化失败: {e}")
            self._rq = None
            self._initialized = True

    def call(self, func_name: str, params: dict) -> pd.DataFrame:
        """调用米筐代理 API"""
        if self._rq is None:
            raise ConnectionError("米筐 API 代理未初始化成功")
        return self._rq.get_rq_data(func_name, params, base_url=self._base_url)


# ====================================================================
# 行业分类获取
# ====================================================================

def fetch_industry_from_rq(
    stock_codes: List[str],
    date: str,
) -> Dict[str, str]:
    """
    通过米筐 API 代理获取股票行业分类。

    参数:
        stock_codes: 股票代码列表（gm格式，如 ['SHSE.600000']）
        date: 查询日期 'YYYY-MM-DD'

    返回:
        {gm_code: industry_name} 字典
    """
    try:
        client = _RQProxyClient()
        # 将 gm 格式转为米筐格式（SHSE.600000 -> 600000.XSHG）
        rq_codes = [_gm_to_rq(code) for code in stock_codes]
        rq_codes = [c for c in rq_codes if c]

        if not rq_codes:
            return {}

        # 调用米筐 get_industry API
        df = client.call('get_industry', {
            'order_book_ids': rq_codes,
            'date': date,
        })

        if df.empty:
            return {}

        # 解析结果——米筐返回: index=order_book_id, 列名=行业分类体系
        # 取 'sws_2016'（申万行业）或 'sws_2021'，如果都没有取第一列
        industry_col = None
        for col in ['sws_2021', 'sws_2016', 'zjw']:
            if col in df.columns:
                industry_col = col
                break
        if industry_col is None and len(df.columns) > 0:
            industry_col = df.columns[0]

        if industry_col is None:
            return {}

        # 转换回 gm 格式
        result = {}
        for rq_code, row in df.iterrows():
            gm_code = _rq_to_gm(str(rq_code))
            if gm_code:
                result[gm_code] = str(row[industry_col])
        return result

    except Exception as e:
        logger.warning(f"米筐获取行业数据失败: {e}")
        return {}


def fetch_industry_from_local(stock_codes: List[str], date: str) -> Dict[str, str]:
    """
    从本地 stock_api 获取行业分类（东方财富行业分类 CSV 文件）。

    参数:
        stock_codes: 股票代码列表（gm格式）
        date: 日期

    返回:
        {gm_code: industry_name} 字典
    """
    try:
        from my_utils.stock_api import stock_api
        api = stock_api()

        # stock_api 需要 '000001.SZ' 格式
        suffix_codes = [convert_code_format(c, format='suffix') for c in stock_codes]
        suffix_codes = [c for c in suffix_codes if c]

        # 获取行业列表
        industry_list = api.get_industry_list()
        if not industry_list:
            return {}

        # 遍历每个行业，获取该行业下的股票
        result = {}
        for ind_code, ind_name in industry_list:
            try:
                df = api.get_industry_data(industry_code=ind_code, start_date=date, end_date=date)
                if not df.empty:
                    codes_in_ind = df['code'].tolist()
                    for sc in suffix_codes:
                        if sc in codes_in_ind:
                            gm_code = convert_code_format(sc, format='gm')
                            if gm_code:
                                result[gm_code] = ind_name
            except Exception:
                continue

        return result
    except Exception as e:
        logger.warning(f"本地获取行业数据失败: {e}")
        return {}


def fetch_industry(
    stock_codes: List[str],
    date: str,
    use_local_first: bool = True,
) -> Dict[str, str]:
    """
    获取股票行业分类（本地数据优先，米筐代理兜底）。

    参数:
        stock_codes: 股票代码列表（gm格式）
        date: 查询日期
        use_local_first: 是否优先使用本地数据

    返回:
        {gm_code: industry_name}
    """
    if use_local_first:
        result = fetch_industry_from_local(stock_codes, date)
        if len(result) > 0:
            return result

    return fetch_industry_from_rq(stock_codes, date)


# ====================================================================
# 代码格式转换（辅助）
# ====================================================================

def _gm_to_rq(gm_code: str) -> str:
    """SHSE.600000 -> 600000.XSHG"""
    market_map = {'SHSE': 'XSHG', 'SZSE': 'XSHE'}
    parts = gm_code.split('.')
    if len(parts) != 2:
        return gm_code
    market, code = parts
    rq_market = market_map.get(market)
    if rq_market is None:
        return gm_code
    return f"{code}.{rq_market}"


def _rq_to_gm(rq_code: str) -> Optional[str]:
    """600000.XSHG -> SHSE.600000"""
    market_map = {'XSHG': 'SHSE', 'XSHE': 'SZSE', 'SH': 'SHSE', 'SZ': 'SZSE'}
    if '.' in rq_code:
        code, market = rq_code.split('.')
    else:
        # 纯数字代码
        if rq_code.startswith('6') or rq_code.startswith('9'):
            return f"SHSE.{rq_code}"
        else:
            return f"SZSE.{rq_code}"

    gm_market = market_map.get(market.upper())
    if gm_market is None:
        return None
    return f"{gm_market}.{code}"


# ====================================================================
# 本地数据适配器（增强版）
# ====================================================================

class LocalDataAdapter:
    """
    本地数据适配器（增强版）：从本地 Parquet + 米筐 API 读取数据，
    转换为因子计算需要的宽表格式。

    新增字段（与基础版相比）:
        - total_mv: 总市值
        - circulation_mv: 流通市值
        - mv_A_free_float: 自由流通市值
        - industry: 行业分类（需调用 load_with_industry 获取）
    """

    DEFAULT_DAY_DATA_PATH = 'rq_stock_all_data'

    # 基础价量字段
    REQUIRED_FIELDS = ['open', 'high', 'low', 'close', 'pre_close', 'volume', 'amount']

    # 可选字段（本地 parquet 有则加载，无则从米筐补充）
    OPTIONAL_FIELDS = ['total_mv', 'circulation_mv', 'mv_A_free_float']

    def __init__(self, data_path: str = None, use_rq_fallback: bool = True):
        """
        参数:
            data_path: Parquet文件路径（相对 DATA_ROOT_DIR）
            use_rq_fallback: 本地缺字段时是否用米筐 API 补充
        """
        self.data_path = data_path or self.DEFAULT_DAY_DATA_PATH
        self.use_rq_fallback = use_rq_fallback

    def load_data(
        self,
        start_date: Union[str, pd.Timestamp],
        end_date: Union[str, pd.Timestamp],
        stock_list: Optional[List[str]] = None,
        min_records: int = 100,
    ) -> Dict[str, pd.DataFrame]:
        """
        加载数据（价量 + 市值），不含行业分类。

        返回字段:
            open, high, low, close, pre_close, volume, amount,
            vwap, returns, adv5/10/20/30/40/50/60/120/180,
            total_mv, circulation_mv, mv_A_free_float
        """
        raw_data = self._read_local_data(start_date, end_date, stock_list)
        if raw_data.is_empty():
            raise ValueError(f"未读取到 {start_date}~{end_date} 范围内的数据")

        raw_data = self._filter_by_record_count(raw_data, min_records)
        data_dict = self._convert_to_wide(raw_data)
        return data_dict

    def load_with_industry(
        self,
        start_date: Union[str, pd.Timestamp],
        end_date: Union[str, pd.Timestamp],
        stock_list: Optional[List[str]] = None,
        min_records: int = 100,
    ) -> Dict[str, Any]:
        """
        加载数据（含行业分类）。

        data_dict 额外包含 'industry' 字段:
            {stock_code: industry_name} 静态映射
        """
        data_dict = self.load_data(start_date, end_date, stock_list, min_records)

        # 获取所有股票代码
        all_codes = data_dict['close'].columns.tolist()

        # 用中间日期查询行业（行业分类相对稳定）
        mid_date = data_dict['close'].index[len(data_dict['close'].index) // 2]

        industry_map = fetch_industry(
            stock_codes=all_codes,
            date=mid_date.strftime('%Y-%m-%d'),
        )
        data_dict['industry'] = industry_map
        logger.info(f"行业分类获取完成: {len(industry_map)}/{len(all_codes)} 只股票")
        return data_dict

    def _read_local_data(
        self,
        start_date: Union[str, pd.Timestamp],
        end_date: Union[str, pd.Timestamp],
        stock_list: Optional[List[str]] = None,
    ) -> pl.DataFrame:
        """读取本地日线数据（含可选字段）"""
        if isinstance(start_date, pd.Timestamp):
            start_date = start_date.date()
        if isinstance(end_date, pd.Timestamp):
            end_date = end_date.date()

        fields = self.REQUIRED_FIELDS + self.OPTIONAL_FIELDS + ['trading_date', 'code']
        raw = read_day_data(
            start_date=start_date,
            end_date=end_date,
            stock_list=stock_list,
            fields=fields,
            file_path=self.data_path,
        )
        return raw

    def _filter_by_record_count(self, df: pl.DataFrame, min_records: int) -> pl.DataFrame:
        counts = df.group_by('code').agg(pl.len().alias('count'))
        valid_codes = counts.filter(pl.col('count') >= min_records)
        return df.join(valid_codes.select('code'), on='code', how='semi')  # semi join 替代 is_in，避免 polars 1.x deprecation warning

    def _convert_to_wide(self, df: pl.DataFrame) -> Dict[str, pd.DataFrame]:
        """转宽表，包含市值字段"""
        df = df.sort(['code', 'trading_date'])
        pdf = df.to_pandas()

        all_dates = sorted(pdf['trading_date'].unique())
        all_codes = sorted(pdf['code'].unique())

        data_dict = {}
        all_fields = self.REQUIRED_FIELDS + self.OPTIONAL_FIELDS

        for field in all_fields:
            if field not in pdf.columns:
                continue
            wide = pdf.pivot_table(
                index='trading_date', columns='code',
                values=field, aggfunc='first',
            )
            wide = wide.reindex(index=all_dates, columns=all_codes)
            data_dict[field] = wide

        # 计算衍生字段
        data_dict = self._compute_derived_fields(data_dict)
        return data_dict

    def _compute_derived_fields(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """计算 vwap / returns / advXX"""
        close = data['close']
        volume = data['volume']
        amount = data.get('amount', volume * 0)  # fallback

        data['vwap'] = amount / volume.replace(0, np.nan)
        data['returns'] = close.pct_change(fill_method=None)

        for period in [5, 10, 20, 30, 40, 50, 60, 120, 180]:
            data[f'adv{period}'] = volume.rolling(window=period, min_periods=1).mean()

        return data


# ====================================================================
# 便捷函数
# ====================================================================

def load_factor_data(
    start_date: str,
    end_date: str,
    stock_list: Optional[List[str]] = None,
    min_records: int = 100,
) -> Dict[str, pd.DataFrame]:
    """一步加载因子计算所需数据（含市值字段）"""
    adapter = LocalDataAdapter()
    return adapter.load_data(start_date, end_date, stock_list, min_records)


def load_factor_data_with_industry(
    start_date: str,
    end_date: str,
    stock_list: Optional[List[str]] = None,
    min_records: int = 100,
) -> Dict[str, Any]:
    """加载数据（含行业分类），用于补齐 IndNeutralize 类 alpha"""
    adapter = LocalDataAdapter()
    return adapter.load_with_industry(start_date, end_date, stock_list, min_records)
