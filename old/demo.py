import polars as pl
import plotly
import datetime as dt
import pandas as pd
import math
from scipy import stats
from gq_stock.dataclass import Interval
from gq_stock import FactorStrategy

class SH099_demo_entroy(FactorStrategy):
    # 必要的属性
    name = "SH099_demo_entroy"
    version = "0.0.1"
    last_edit_date = "2025-06-25"
    category = "量价"
    author = "shq"
    code_mapping = False
    calc_latency = 1
    baseline = 'zz500'
    description = """
    股票因子demo
    """
    def __init__(self):
        # 允许父类做一些处理
        super().__init__(
            author = self.author,
            last_edit_date = self.last_edit_date,
            description = self.description,
            code_mapping = self.code_mapping
            )
        pass 

    def before_calculation(self, ctx):
        return True

    def after_calculation(self, ctx):
        return True

    def on_init(self, ctx):
        pass

   
    def on_calculation(self, ctx):
        #ctx.m.get_l2_tick()
        #mtb = ctx.m.get_bar(interval=Interval.M1, start_date= ctx.start_date(), end_date= ctx.end_date(), fields=["trading_date", "datetime","code", "volume", "turnover"])
        mtb = ctx.m.get_bar(interval=Interval.M1, start_date= ctx.start_date(), end_date= ctx.end_date())
        day_data = ctx.m.get_bar(interval=Interval.D1, start_date= ctx.start_date(), end_date= ctx.end_date())
        print("取数结束")
        
        ctx.info.get_all_code(ctx.start_date(), ctx.end_date())
        #  计算每天的成交量汇总
        mtb = mtb.with_columns(
            sumVolume = pl.col("volume").sum().over("trading_date", "code"),
            sumTurnover = pl.col("turnover").sum().over("trading_date", "code")
        )
        mtb = mtb.with_columns(
            volRatio = pl.col("volume") / pl.col("sumVolume"),
            turnoverRatio = pl.col("turnover") / pl.col("sumTurnover")
        )
        #熵值计算
        factor = mtb.group_by(
            "code",
            "trading_date"
        ).agg(
            volEntropy = (-1 * pl.col("volRatio") * pl.col("volRatio").log()).fill_nan(0).sum(),
            amtEntropy = (-1 * pl.col("turnoverRatio") * pl.col("turnoverRatio").log()).fill_nan(0).sum()
        )
        
        factor = factor.with_columns(
            diff = pl.col("amtEntropy") - pl.col("volEntropy")
        ).with_columns(
            datetime = pl.col("trading_date").cast(pl.Datetime).cast(pl.Datetime(time_unit="ms", time_zone="Asia/Shanghai")).dt.replace(hour=15, minute=0, second=0)
        ).sort("datetime", "code")
        
        factor = factor.select(
            pl.col("trading_date"),
            pl.col("code"),
            pl.col("datetime"),
            pl.col("diff").alias("factor_value"),
        )
        

        factor = factor.with_columns(
            # 计算滚动均值 (过去21天)
            rolling_mean=pl.col("factor_value")
                .rolling_mean(window_size=21, min_samples=21)
                .over("code"),
            
            # 计算滚动标准差 (过去21天)
            rolling_std=pl.col("factor_value")
                .rolling_std(window_size=21, min_samples=21)
                .over("code")
        ).with_columns(
            # 计算Z-Score并处理除零错误
            factor_value=pl.when(pl.col("rolling_std") != 0)
                .then((pl.col("factor_value") - pl.col("rolling_mean")) / pl.col("rolling_std"))
                .otherwise(None)
        )
        
        factor = factor.filter(
            pl.col("factor_value").is_not_null()  # 只保留非空行
        ).select(
            pl.col("trading_date"),
            pl.col("code"),
            pl.col("datetime"),
            pl.col("factor_value").cast(pl.Float32).alias("factor_value"),
        )


        return factor
    




if __name__ == "__main__":

    from gq_stock.dataclass import Interval, PriceType, WeightType
    from gq_stock import factor_operation as fo
    from gq_stock import weight_limit as wl
    import gq_stock as gs
    import datetime as dt

    gs.login("", "")
    start_date = dt.date(2022, 11, 1)
    end_date = dt.date(2022, 12, 30) 
    
    engine = gs.StockEngine(
        start_date = start_date,
        end_date = end_date,
    )
    
    stra = SH099_demo_entroy()
    engine.add_stra(stra)
    factor = engine.run()
    
    print(factor)
    
