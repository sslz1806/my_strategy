"""
my_utils - 策略项目核心接口函数包

提供数据接口、特征计算、回测框架、实盘交易等功能
"""
from . import fun
from . import trade_fun
from . import stock_api
from . import mapping
from . import pd_fun
from . import email_fun
from . import stock_plot
from . import my_qmt

# stock_db 需要 pymysql 依赖，非核心模块，导入失败不影响其他模块
try:
    from . import stock_db
    _has_stock_db = True
except ImportError:
    _has_stock_db = False

__all__ = [
    "fun", "trade_fun", "stock_api", "mapping",
    "pd_fun", "email_fun", "stock_plot", "my_qmt",
]
if _has_stock_db:
    __all__.append("stock_db")