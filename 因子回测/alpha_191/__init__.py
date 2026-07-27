"""
因子回测/alpha_191 - Alpha191 因子本地化适配

将一个开源的 Alpha-101 / GTJA-191 因子库适配到本项目的数据接口和回测框架。

用法:
    from 因子回测.alpha_191 import Alpha191Calculator

    calc = Alpha191Calculator()
    calc.load_data('2025-01-01', '2025-07-01')
    alpha5 = calc.compute(5)
    result = calc.analyze_factor(5, return_period=5)

依赖:
    - my_utils.fun.read_day_data()  (本地数据接口)
    - 因子回测.alpha.analyze_ic()   (本地IC分析)
    - 因子回测.alpha.analyze_factor() (本地宽表因子分析)
    - my_utils.rqdata.RQData       (米筐API，用于补充数据)
"""

from .adapter import load_factor_data, LocalDataAdapter
from .alpha_formulas import Alpha191Formulas
from .calculator import Alpha191Calculator

__all__ = [
    'load_factor_data',
    'LocalDataAdapter',
    'Alpha191Formulas',
    'Alpha191Calculator',
]
