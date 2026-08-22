# 我的简易回测框架代码
"""
:先传入已经按照策略生成的交割数据(pandas.dataframe包含datetime,code,direction,price，可选volume/cash_ratio/sell_volume字段),排序。
然后按照交易日循环,执行买入卖出操作并记录资金变化。

my_backtrader.my_backtester 
类功能:
1.可以设置初始资金,佣金,滑点等参数。默认为初始资金100000,佣金0.001,滑点0.001
2.全局变量记录当前资金和持仓情况.
3.回测逻辑next:可以自定义实现,逻辑就是每个交易日需要执行的操作
4.交易逻辑buy/sell:实现买入卖出操作,更新资金和持仓。两个函数,有按照数量买的,也有按照资金比例买的。
5.回测执行run:按照交易日循环调用next函数,直到结束(使用 get_data_trading_days 获取交易日;需传入 start_time/end_time)。
6.回测记录结果:记录每个交易日的资金变化,持仓情况等,方便后续分析。
"""

from __future__ import annotations
# 将目录加入到 sys.path 以便导入 fun 模块 C:\Users\20561\Desktop\策略\
import sys
sys.path.append("C://Users/20561/Desktop/策略")

from dataclasses import dataclass
from typing import Dict, Optional, Iterable, List
from my_utils.fun import get_data_trading_days,read_day_data
import logging
import pandas as pd
import numpy as np


@dataclass
class Position:
    """持仓信息,其中close,market_value,pnl,profit字段按照每日price更新"""
    code: str
    name: str = ""
    trade_log: list = None # 用于记录该段时间内这只股票的交易记录,有datetime,colse,volume,direction字段
    volume: int = 0
    avail_volume: int = 0  # 可用仓位，考虑t+1规则
    avg_cost: float = 0.0
    close: float = 0.0
    market_value: float = 0.0  # 持仓市值
    pnl:float = 0.0  # 持仓盈亏
    profit: float = 0.0 # 持仓盈亏比例



class Backtester:
    """Minimal backtest engine similar to common retail platforms."""

    def __init__(
        self,
        orders: pd.DataFrame,
        initial_cash: float = 100_000.0,
        commission: float = 0.001,
        slippage: float = 0.001,
        max_single_position: float = None,
    ) -> None:
        """
        __init__ 的 参数说明
        :param orders: pandas.DataFrame,交易指令数据,包括交易的时间,代码,方向,价格等信息; volume 可选
        :param initial_cash: 初始资金
        :param commission: 佣金比例
        :param slippage: 滑点比例
        :param max_single_position: 单票最大仓位上限（占总资金比例），None 表示不限制。
            例如 0.24 表示单票买入金额不超过总资金的 24%。
            该约束作用于每笔买入的资金分配上，与 orders 中的 cash_ratio（每日总仓位上限）独立。
        """
        if orders is None:
            raise ValueError("orders dataframe is required")
        # volume 非必需；策略可在 next 中自行决定成交量
        required_cols = {"datetime", "code", "direction", "price"}
        missing = required_cols - set(orders.columns)
        if missing:
            raise ValueError(f"orders dataframe missing columns: {missing}")

        self.orders = orders.copy()
        self.orders["datetime"] = pd.to_datetime(self.orders["datetime"])
        self.orders = self.orders.sort_values("datetime").reset_index(drop=True)

        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.total_value = float(initial_cash)
        self.commission = float(commission)
        self.slippage = float(slippage)
        self.max_single_position = (
            float(max_single_position) if max_single_position is not None else None
        )

        self.positions: Dict[str, Position] = {}
        self.price_cache: Dict[str, float] = {}
        # result 用于记录每日资金快照
        self.result: Optional[pd.DataFrame] = None
        self.trade_log = []
        # pos_log  # 用于记录每日持仓快照
        self.pos_log: Optional[pd.DataFrame] = pd.DataFrame(
            columns=[
                "datetime","code","name","volume","avg_cost","close","market_value","pnl","profit"
            ]
        )

    # --- trading helpers -------------------------------------------------
    def _add_trade_log(self, dt, code, side, price, volume, fee) -> None:
        log ={
                "datetime": dt,
                "code": code,
                "side": side,
                "price": price,
                "volume": volume,
                "fee": fee,
            }
        self.trade_log.append(log)
        #self.positions[code].trade_log.append(log) if code in self.positions else None

    def buy(self, code: str, price: float, volume: int, dt) -> None:
        if volume <= 0:
            return
        adj_price = price * (1 + self.slippage)
        cost = adj_price * volume
        fee = cost * self.commission
        total_cost = cost + fee
        if total_cost - self.cash > 1e-9:
            logging.info(f"资金不足,无法买入 {code} {volume} 股，所需资金 {total_cost:.2f}，当前现金 {self.cash:.2f}")

        pos = self.positions.get(code, Position(code=code))
        new_volume = pos.volume + volume
        pos.avg_cost = (pos.avg_cost * pos.volume + adj_price * volume) / new_volume
        pos.volume = new_volume
        self.positions[code] = pos

        self.cash -= total_cost
        self.price_cache[code] = price
        self._add_trade_log(dt, code, "buy", adj_price, volume, fee)

    def buy_value(self, code: str, price: float, buy_value, dt) -> int:
        # 按照市值买入(考虑100股整数倍)
        alloc = buy_value
        unit_cost = price * (1 + self.slippage)
        volume = int(alloc // unit_cost)
        if volume <= 50:
            return 0
        volume_real = (volume // 100) * 100  # 向下取整到100股整数倍
        self.buy(code, price, volume_real, dt)
        return volume_real

    def sell(self, code: str, price: float, volume: int, dt) -> None:
        if volume <= 0:
            return
        pos = self.positions.get(code)
        if pos is None or pos.volume < volume:
            raise ValueError("insufficient position for sell")

        adj_price = price * (1 - self.slippage)
        proceeds = adj_price * volume
        fee = proceeds * self.commission

        pos.volume -= volume
        if pos.volume == 0:
            del self.positions[code]
        else:
            self.positions[code] = pos

        self.cash += proceeds - fee
        self.price_cache[code] = price
        self._add_trade_log(dt, code, "sell", adj_price, volume, fee)

    # --- user extension point -------------------------------------------
    def next(self, dt: pd.Timestamp, daily_orders: pd.DataFrame) -> None:
        """默认日逻辑：执行传入的指令。
        daily_orders: datetime当天的交易指令DataFrame，包含code,direction,price,等字段。
        """
        if daily_orders.empty or daily_orders is None:
            return
        total_value = self.total_value
        cash = self.cash
        # 获取交易指令中的资金仓位信息(获取weight的平均值作为当天的总仓位)
        cash_ratio = daily_orders["cash_ratio"].mean() if "cash_ratio" in daily_orders.columns else 0.0
        cash = min(cash, total_value * cash_ratio)
        num_buy = sum(1 for _, row in daily_orders.iterrows() if int(row["direction"]) == 1)
        alloc_per_buy = cash / num_buy if num_buy > 0 else 0.

        # 单票仓位上限：凯利公式给出的最大单票风险暴露，约束每笔买入不超过总资金的 max_single_position
        if self.max_single_position is not None and alloc_per_buy > 0:
            single_cap = total_value * self.max_single_position
            alloc_per_buy = min(alloc_per_buy, single_cap)

        # 先卖后买，避免同日换仓时先买入污染卖出匹配基准
        daily_orders_exec = daily_orders.sort_values("direction", ascending=True, kind="mergesort")
        for _, row in daily_orders_exec.iterrows():
            direction = int(row["direction"])
            code = str(row["code"])
            price = float(row["price"])

            # 按照资金分配买入
            if direction == 1:
                buy_time = row["buy_time"] if "buy_time" in row else None
                self.buy_value(code, price, alloc_per_buy, buy_time)
            elif direction == -1:
                buy_time = row["buy_time"] if "buy_time" in row else None
                if pd.isna(buy_time):
                    buy_time = None
                sell_time = row["sell_time"] if "sell_time" in row else None
                # 利用self.trade_log追踪对应的买入记录,获取当时买入数量作为卖出数量
                sell_volume = 0
                for log in reversed(self.trade_log):
                    if log["code"] == code and log["side"] == "buy":
                        if buy_time is not None and log["datetime"] != buy_time:
                            continue
                        sell_volume = log["volume"]
                        break
                if sell_volume == 0 and code in self.positions and self.positions[code].volume > 0:
                    sell_volume = int(self.positions[code].volume)
                    logging.info(f"卖出数量为0,当前持仓数量为{sell_volume}")
                current_pos_volume = int(self.positions.get(code).volume) if code in self.positions else 0
                if sell_volume > current_pos_volume:
                    sell_volume = current_pos_volume
                    logging.info(f"卖出数量超过当前持仓数量,卖出数量为{sell_volume},当前持仓数量为{current_pos_volume}")
                if sell_volume > 0:
                    self.sell(code, price, sell_volume, sell_time)

    # --- bookkeeping -----------------------------------------------------
    def _snapshot_value(self, dt: pd.Timestamp) -> None:
        """
        _snapshot_value:更新资金快照。
        """
        holding = 0.0
        for code, pos in self.positions.items():
            # 获取最新市值
            market_value = pos.market_value
            holding += market_value
        total = self.cash + holding
        self.total_value = total
        snap = {
            "datetime": dt,
            "cash": self.cash,
            "holding_value": holding,
            "total_value": total,
        }
        return pd.DataFrame([snap])  # 返回单行DataFrame

    def _snapshot_positions(self, dt: pd.Timestamp) -> pd.DataFrame:
        """记录当前持仓情况的快照。其中close,market_value,pnl,profit字段利用read_day_data按照每日price更新"""
        records = []
        pos_list = list(self.positions.keys())
        pos_data = read_day_data(start_date=dt, end_date=dt, stock_list=pos_list, fields=["trading_date","code","close","name"])
        # 将polars转成pandas DataFrame以便查询
        pos_data = pos_data.to_pandas()
        for code, pos in self.positions.items():
            if code in pos_data["code"].values:
                daily_close = float(pos_data[pos_data["code"] == code]["close"].iloc[0])
            else:
                daily_close = float(pos.close) if pos.close and pos.close > 0 else float(self.price_cache.get(code, pos.avg_cost))
            market_value = daily_close * pos.volume
            pnl = market_value - pos.avg_cost * pos.volume
            profit = pnl / (pos.avg_cost * pos.volume) if pos.avg_cost * pos.volume > 0 else 0.0
            name = pos_data[pos_data["code"] == code]["name"].iloc[0] if code in pos_data["code"].values else ""
            # 更新持仓信息
            pos.close = daily_close
            pos.market_value = market_value
            pos.pnl = pnl
            pos.profit = profit
            pos.name = name
            # 记录持仓快照,返回当日的持仓情况DataFrame
            record = {
                "datetime": dt,
                "code": code,
                "name": pos.name,
                "volume": pos.volume,
                "avg_cost": pos.avg_cost,
                "close": daily_close,
                "market_value": market_value,
                "pnl": pnl,
                "profit": profit,
            }
            records.append(record)
        return pd.DataFrame(records)

    # --- analysis helpers ---------------------------------------------
    def _compute_trade_stats(self) -> Dict[str, float]:
        """基于 trade_log 计算简单的成交统计（胜率、盈亏比、已实现收益）。"""
        if not self.trade_log:
            return {
                "realized_profit": 0.0,
                "win_rate": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "profit_loss_ratio": 0.0,
            }

        logs = sorted(self.trade_log, key=lambda x: x.get("datetime"))
        holdings: Dict[str, Dict[str, float]] = {}
        realized: List[float] = []

        for log in logs:
            code = str(log.get("code"))
            side = log.get("side")
            price = float(log.get("price", 0.0))
            volume = int(log.get("volume", 0))
            fee = float(log.get("fee", 0.0))
            pos = holdings.get(code, {"vol": 0, "avg": 0.0})

            if side == "buy":
                new_vol = pos["vol"] + volume
                new_avg = price if pos["vol"] == 0 else (pos["avg"] * pos["vol"] + price * volume) / new_vol
                holdings[code] = {"vol": new_vol, "avg": new_avg}
            elif side == "sell":
                if pos["vol"] < volume:
                    logging.warning(f"sell volume exceeds holding for {code}, skip this log")
                    continue
                profit = (price - pos["avg"]) * volume - fee
                realized.append(profit)
                pos["vol"] -= volume
                if pos["vol"] > 0:
                    holdings[code] = pos
                else:
                    holdings.pop(code, None)

        realized_profit = float(np.sum(realized)) if realized else 0.0
        wins = [p for p in realized if p > 0]
        losses = [p for p in realized if p < 0]
        win_rate = len(wins) / len(realized) if realized else 0.0
        avg_win = float(np.mean(wins)) if wins else 0.0
        avg_loss = float(abs(np.mean(losses))) if losses else 0.0
        profit_loss_ratio = (avg_win / avg_loss) if avg_loss > 0 else float("inf") if avg_win > 0 else 0.0

        return {
            "realized_profit": realized_profit,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_loss_ratio": profit_loss_ratio,
        }

    def analyze(self, risk_free_rate: float = 0.02, trading_days_per_year: int = 252) -> Dict[str, float]:
        """回测结果分析，返回核心指标字典。

        指标包括：总收益率、年化收益、波动率、夏普、最大回撤、
        胜率、盈亏比、已实现收益、最终净值等。
        """
        if self.result is None or self.result.empty:
            raise ValueError("no backtest result to analyze; run() first")

        df = self.result.sort_values("datetime").reset_index(drop=True).copy()
        df["net_value"] = df["total_value"] / float(self.initial_cash)
        df["daily_return"] = df["net_value"].pct_change()

        daily_ret = df["daily_return"].dropna()
        total_return = float(df["net_value"].iloc[-1] - 1)
        if len(df) >= 2:
            days = (pd.to_datetime(df["datetime"].iloc[-1]) - pd.to_datetime(df["datetime"].iloc[0])).days
            years = days / 365.0 if days > 0 else 0.0
        else:
            years = 0.0

        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 and total_return > -1 else 0.0
        volatility = float(daily_ret.std() * np.sqrt(trading_days_per_year)) if not daily_ret.empty else 0.0
        excess_daily = daily_ret - risk_free_rate / trading_days_per_year if not daily_ret.empty else pd.Series(dtype=float)
        sharpe = float((excess_daily.mean() / excess_daily.std()) * np.sqrt(trading_days_per_year)) if not excess_daily.empty and excess_daily.std() > 0 else 0.0

        roll_max = df["net_value"].cummax()
        drawdown = (df["net_value"] - roll_max) / roll_max
        max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0

        trade_stats = self._compute_trade_stats()

        metrics = {
            "start_date": pd.to_datetime(df["datetime"].iloc[0]),
            "end_date": pd.to_datetime(df["datetime"].iloc[-1]),
            "total_return": total_return,
            "annualized_return": annualized_return,
            "volatility": volatility,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown,
            "final_net_value": float(df["net_value"].iloc[-1]),
            "realized_profit": trade_stats["realized_profit"],
            "win_rate": trade_stats["win_rate"],
            "avg_win": trade_stats["avg_win"],
            "avg_loss": trade_stats["avg_loss"],
            "profit_loss_ratio": trade_stats["profit_loss_ratio"],
        }
        return metrics


    def report(
        self,
        start_date=None,
        end_date=None,
        benchmark_code: Optional[str] = 'SHSE.000001',
        risk_free_rate: float = 0.02,
        trading_days_per_year: int = 252,
        return_method: str = "compound",
        plot: bool = True,
    ):
        """生成与 report_backtest_full 类似的指标与图表。

        benchmark_curve: 可选的基准净值曲线（pd.Series，index 为日期或 datetime，值为净值）。
        return_method: compound 复利；其他值走单利累计。
        返回 (metrics_df, fig)；如果 plot=False，则 fig 为 None。
        """
        import matplotlib.pyplot as plt
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        from my_utils.stock_api import stock_api

        if self.result is None or self.result.empty:
            raise ValueError("no backtest result to analyze; run() first")

        # result为每日资金快照
        df = self.result.sort_values("datetime").reset_index(drop=True).copy()
        df["datetime"] = pd.to_datetime(df["datetime"])
        df["net_value"] = df["total_value"] / float(self.initial_cash)
        df["date"] = df["datetime"].dt.date

        # 日期过滤
        if start_date is not None:
            start_ts = pd.to_datetime(start_date)
            df = df[df["datetime"] >= start_ts]
        else:
            start_date = df["datetime"].iloc[0].date()
        if end_date is not None:
            end_ts = pd.to_datetime(end_date)
            df = df[df["datetime"] <= end_ts]
        else:
            end_date = df["datetime"].iloc[-1].date()
        df = df.reset_index(drop=True)
        if df.empty:
            raise ValueError("no data after applying date filter")

        # 策略净值曲线
        daily_returns = df.set_index("datetime")["net_value"].pct_change().dropna()
        strategy_curve = df.set_index("datetime")["net_value"]
        if return_method != "compound":
            strategy_curve = 1 + daily_returns.cumsum()

        roll_max = strategy_curve.cummax()
        drawdown = (strategy_curve - roll_max) / roll_max

        # 基准净值曲线处理
        api = stock_api()
        index_data = api.gm_get_index_day_data(index_code=benchmark_code,start_date=start_date.strftime('%Y-%m-%d'),end_date=end_date.strftime('%Y-%m-%d'))
        index_data['trading_date'] = pd.to_datetime(index_data['trading_date'])
        index_data['trading_date_date'] = index_data['trading_date'].dt.date
        if not index_data.empty:
            index_data['net_value'] = index_data['close'] / index_data['close'].iloc[0]
            index_curve = index_data.set_index('trading_date_date')['net_value']
        else:
            raise ValueError("未获取到有效的指数数据")

        # 指标
        total_return = strategy_curve.iloc[-1] - 1 if len(strategy_curve) else 0.0
        if len(strategy_curve) >= 2:
            first_date = strategy_curve.index[0]
            last_date = strategy_curve.index[-1]
            days = (last_date - first_date).days
            years = days / 365 if days > 0 else 0
        else:
            years = 0
            first_date = df["datetime"].iloc[0]
            last_date = df["datetime"].iloc[-1]
        annualized_return = (strategy_curve.iloc[-1]) ** (1 / years) - 1 if years > 0 and strategy_curve.iloc[-1] > 0 else 0

        daily_ret = strategy_curve.pct_change().dropna()
        rf_daily = risk_free_rate / trading_days_per_year
        excess_daily = daily_ret - rf_daily
        sharpe_ratio = (excess_daily.mean() / excess_daily.std()) * np.sqrt(trading_days_per_year) if excess_daily.std() > 0 else 0
        max_drawdown = drawdown.min() if not drawdown.empty else 0
        max_drawdown_end = drawdown.idxmin() if not drawdown.empty else None
        max_drawdown_start = roll_max.loc[:max_drawdown_end].idxmax() if max_drawdown_end is not None else None

        trade_stats = self._compute_trade_stats()

        metrics_df = pd.DataFrame({
            "指标名称": [
                "回测开始日期", "回测结束日期", "策略胜率", "策略盈亏比",
                "策略总收益率", "策略年化收益率",
                "最大回撤", "最大回撤开始日期", "最大回撤结束日期",
                "夏普比率", "最终净值",
            ],
            "指标值": [
                first_date, last_date,
                f"{trade_stats['win_rate']:.2%}",
                f"{trade_stats['profit_loss_ratio']:.2f}",
                f"{total_return:.2%}",
                f"{annualized_return:.2%}",
                f"{max_drawdown:.2%}",
                max_drawdown_start, max_drawdown_end,
                f"{sharpe_ratio:.2f}",
                f"{strategy_curve.iloc[-1]:.4f}" if len(strategy_curve) else "1.0000",
            ]
        })

        fig = None
        if plot:
            fig = make_subplots(
                rows=3,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.08,
                subplot_titles=("净值曲线对比", "策略每日收益率", "策略回撤"),
                row_heights=[0.5, 0.25, 0.25],
            )

            # 策略净值
            fig.add_trace(
                go.Scatter(
                    x=strategy_curve.index,
                    y=strategy_curve.values,
                    name="策略净值,初始资金=%.2f" % self.initial_cash,
                    line=dict(color="#1f77b4", width=2),
                    hovertemplate="日期: %{x}<br>策略净值: %{y:.4f}<extra></extra>",
                ),
                row=1, col=1,
            )

            # 基准净值
            if index_curve is not None and not index_curve.empty:
                fig.add_trace(
                    go.Scatter(
                        x=index_curve.index,
                        y=index_curve.values,
                        name=f"基准({benchmark_code})",
                        line=dict(color="#ff7f0e", width=2, dash="dash"),
                        hovertemplate="日期: %{x}<br>基准净值: %{y:.4f}<extra></extra>",
                    ),
                    row=1, col=1,
                )

            # 每日收益率柱状
            ret_colors = ["#d62728" if x > 0 else "#2ca02c" for x in daily_ret]
            fig.add_trace(
                go.Bar(
                    x=daily_ret.index,
                    y=daily_ret.values,
                    name="每日收益率",
                    marker_color=ret_colors,
                    hovertemplate="日期: %{x}<br>收益率: %{y:.2%}<extra></extra>",
                ),
                row=2, col=1,
            )

            # 回撤（面积图）
            fig.add_trace(
                go.Scatter(
                    x=drawdown.index,
                    y=drawdown.values,
                    name="回撤",
                    fill="tozeroy",
                    line=dict(color="#d62728", width=1),
                    fillcolor="rgba(211, 39, 39, 0.2)",
                    hovertemplate="日期: %{x}<br>回撤率: %{y:.2%}<extra></extra>",
                ),
                row=3, col=1,
            )

            fig.update_layout(
                height=800,
                title_text="回测结果可视化",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                plot_bgcolor="rgba(248,248,248,1)",
                paper_bgcolor="white",
            )

            fig.update_yaxes(title_text="净值", row=1, col=1, tickformat=".2f")
            fig.update_yaxes(title_text="收益率", row=2, col=1, tickformat=".2%")
            fig.update_yaxes(title_text="回撤率", row=3, col=1, tickformat=".2%")
            fig.update_xaxes(title_text="日期", row=3, col=1)

        return metrics_df, fig

    def run(self, start_time, end_time) -> pd.DataFrame:
        """按交易日历回测。
        :param start_time: 支持 str/pandas.Timestamp/datetime
        :param end_time:   支持 str/pandas.Timestamp/datetime
        """
        if self.orders is None:
            raise ValueError("orders dataframe is required")

        # 获取交易日列表（fun.get_data_trading_days 应返回 datetime/timestamp 列表）
        trading_days: Iterable = get_data_trading_days(start_time, end_time)
        if trading_days is None:
            raise ValueError("get_data_trading_days returned None")

        orders = self.orders

        for day in trading_days:
            day_ts = pd.Timestamp(day).normalize()
            day_start = day_ts
            day_end = day_start + pd.Timedelta(days=1)

            # 取该交易日的订单
            daily_orders = orders[(orders["datetime"] >= day_start) & (orders["datetime"] < day_end)]
            # 执行当日策略逻辑（可在子类覆盖 next）
            self.next(day_ts, daily_orders)

            # 用当日最后价更新持仓估值价格
            if not daily_orders.empty:
                for code, rows in daily_orders.groupby("code"):
                    self.price_cache[str(code)] = float(rows.iloc[-1]["price"])

            # 每个交易日都做资金快照（即便当天没有订单）
            self.pos_log = pd.concat([self.pos_log, self._snapshot_positions(day_ts)], ignore_index=True)
            self.result = pd.concat([self.result, self._snapshot_value(day_ts)], ignore_index=True) if self.result is not None else self._snapshot_value(day_ts)

        if self.result is not None:
            self.result = self.result.sort_values("datetime").reset_index(drop=True)
        return self.result if self.result is not None else pd.DataFrame()

    # --- convenience accessors ------------------------------------------
    def portfolio_value(self) -> float:
        if self.result is None or self.result.empty:
            return self.initial_cash
        return float(self.result.iloc[-1]["total_value"])

    def history(self) -> pd.DataFrame:
        return pd.DataFrame() if self.result is None else self.result.copy()

    def trades(self) -> pd.DataFrame:
        return pd.DataFrame(self.trade_log)

if __name__ == "__main__":
    # 读取交割数据
    交割数据 = pd.read_csv("信号文件/断板低开-5--2.5 20260108_225026(sma7).csv",encoding='utf-8-sig')  # 假设有一个包含交易
    # 将每一条数据分成两条指令对应buy_time和sell_time,buy_price,sell_price
    orders_list = []
    for _, row in 交割数据.iterrows():
        buy_order = {
            "datetime": pd.to_datetime(row["buy_time"]),
            "code": row["code"],
            "direction": 1,
            "price": row["buy_price"],
            # "volume": row["volume"],  # 可选
            "cash_ratio":row["weight"], # 资金仓位
            "buy_time":row["buy_time"], # 用于标识买入时间
            "sell_time":row["sell_time"],
        }
        sell_order = {
            "datetime": pd.to_datetime(row["sell_time"]),
            "code": row["code"],
            "direction": -1,
            "price": row["sell_price"],
            # "volume": row["volume"],  # 可选
            "cash_ratio":row["weight"], # 资金仓位
            "buy_time":row["buy_time"], # 用于标识卖出对应的买入时间,便于追踪
            "sell_time":row["sell_time"],
        }
        orders_list.append(buy_order)
        orders_list.append(sell_order)
    orders_df = pd.DataFrame(orders_list)
    # 初始化回测引擎
    backtester = Backtester(orders=orders_df, initial_cash=1000000, commission=0.0001, slippage=0.0005)
    # 运行回测
    回测结果 = backtester.run(start_time="2024-01-01", end_time="2026-01-16")
    # 可视化回测结果


    # 保存pos_log,trade_log,result等数据
    backtester.pos_log.to_csv("持仓快照.csv",index=False)
    pd.DataFrame(backtester.trade_log).to_csv("交易记录.csv",index=False)
    回测结果.to_csv("资金变化.csv",index=False)
    
    
    