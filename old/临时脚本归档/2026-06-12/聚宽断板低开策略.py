# -*- coding: utf-8 -*-
"""
聚宽平台版本：断板/炸板低开策略

迁移来源：
1. 本地 `回测demo.ipynb` 中的主策略信号：
   - 昨日状态为「断板」或「炸板」
   - 今日开盘低开区间为 -5% 至 -2.5%
   - 昨日收盘价在昨日 7 日均线之上
   - 最近一次涨停描述不为「1天1板」且非空
   - 剔除 ST、创业板、科创板、B 股、北交所等股票
   - 流通市值在 35 亿至 1000 亿之间
   - 今日开盘价 / 前 30 日最低价 <= 3
2. 本地 `my_utils.trade_fun.trade` 的卖出逻辑：
   - T+1 起允许卖出
   - 次日 09:30 若大幅低开 <= -7%，优先卖出；若一字跌停则等待
   - 盘中/尾盘若跌破买入价 9%，止损卖出；跌停无法卖出则等待
   - 11:28 和 14:50 若价格未接近涨停，卖出

注意：
聚宽平台字段与本地数据字段不完全一致。本地 `mv_A_free_float` 是自由流通市值，
聚宽这里使用 `valuation.circulating_market_cap` 作为近似流通市值口径过滤。
请在聚宽回测页选择「分钟级」回测，以便 09:30 / 11:28 / 14:50 的定时交易逻辑生效。
"""

from jqdata import *

import datetime as dt
import pandas as pd


def initialize(context):
    """
    聚宽策略初始化函数。

    这里只保存可以序列化的简单参数到 g，避免聚宽模拟盘重启后全局对象无法恢复。
    """
    set_benchmark('000300.XSHG')
    set_option('use_real_price', True)
    set_option('avoid_future_data', True)
    log.set_level('system', 'error')

    # A 股常用费用假设：买卖佣金万三，卖出印花税千一，最低佣金 5 元。
    # 如果你的本地回测希望完全贴近 `fee_rate=0.004` 的双边粗略费率，可在聚宽页面另行调整。
    set_order_cost(
        OrderCost(
            close_tax=0.001,
            open_commission=0.0003,
            close_commission=0.0003,
            min_commission=5,
        ),
        type='stock',
    )

    # 信号参数集中放在 g，方便后续在聚宽页面直接修改参数做敏感性测试。
    g.open_pct_low = -5.0
    g.open_pct_high = -2.5
    g.mv_min = 35
    g.mv_max = 1000
    g.sma_window = 7
    g.lowest_window = 30
    g.limit_desc_lookback = 20
    g.relative_lowest_max = 3.0
    g.stop_loss_pct = 0.09

    # 单日最大策略仓位。你的 notebook 里最终常用 `profit * 0.4` 评估，
    # 因此这里默认每日总开仓资金不超过组合总资产的 40%。
    g.max_daily_position = 0.4

    # 卖出函数必须先于买入函数在 09:30 注册，避免旧持仓未处理时新信号占用资金。
    run_daily(sell_at_open, time='09:30')
    run_daily(buy, time='09:30')
    run_daily(sell_intraday, time='11:28')
    run_daily(sell_intraday, time='14:50')


def buy(context):
    """
    每日 09:30 选股并等权买入。

    由于聚宽 09:30 可通过 `get_current_data()` 读取当日开盘价，本函数使用昨日以前的
    日线数据生成状态类特征，再用当日开盘价计算低开幅度，避免使用未来数据。
    """
    prev_date = to_date_str(context.previous_date)
    current_data = get_current_data()

    candidate_stocks = prepare_stock_list(prev_date)
    if not candidate_stocks:
        return

    # 先获取昨日满足「断板/炸板」及其他日线条件的股票，再用今日开盘低开条件二次过滤。
    signal_df = get_signal_df(candidate_stocks, prev_date)
    if signal_df.empty:
        return

    buy_list = []
    for stock in list(signal_df.index):
        if stock not in current_data:
            continue

        snapshot = current_data[stock]
        yesterday_close = signal_df.loc[stock, 'close']
        day_open = snapshot.day_open

        # 聚宽在极端停牌、复牌或数据缺失时可能给出空开盘价，空值直接跳过，防止异常下单。
        if pd.isnull(day_open) or pd.isnull(yesterday_close) or yesterday_close <= 0:
            continue

        open_pct = (day_open / yesterday_close - 1) * 100
        if not (g.open_pct_low <= open_pct <= g.open_pct_high):
            continue

        # 盘中实时状态再过滤一次停牌、ST、涨跌停不可交易等异常情况。
        if is_untradable_at_open(stock, snapshot):
            continue

        buy_list.append(stock)

    if not buy_list:
        return

    # 当天信号等权买入，并限制整体策略仓位为组合总资产的 g.max_daily_position。
    # 这里按总资产而不是可用现金分配，是为了更接近本地 `weight=0.4` 的资金暴露口径。
    target_value = context.portfolio.total_value * g.max_daily_position / len(buy_list)
    for stock in buy_list:
        order_target_value(stock, target_value)
        log.info('买入: %s %s, 目标市值 %.2f' % (get_security_info(stock).display_name, stock, target_value))


def sell_at_open(context):
    """
    T+1 后 09:30 开盘卖出检查。

    本地回测中 09:30 只处理「次日大幅低开 <= -7%」的特殊卖出场景；
    如果开盘即一字跌停，聚宽市价卖单通常无法成交，因此这里延后等待后续时点。
    """
    current_data = get_current_data()
    for stock in list(context.portfolio.positions.keys()):
        position = context.portfolio.positions[stock]
        if position.closeable_amount <= 0 or stock not in current_data:
            continue

        snapshot = current_data[stock]
        day_open = snapshot.day_open
        last_price = snapshot.last_price

        if pd.isnull(day_open) or pd.isnull(snapshot.high_limit) or pd.isnull(snapshot.low_limit):
            continue

        prev_close = get_previous_close(stock, context.previous_date)
        if prev_close is None or prev_close <= 0:
            continue

        open_pct = (day_open / prev_close - 1) * 100

        # 跌停且没有打开空间时不强行下单，避免无意义卖单污染日志。
        if last_price <= snapshot.low_limit and day_open <= snapshot.low_limit:
            continue

        if open_pct <= -7:
            order_target_value(stock, 0)
            log.info('大幅低开卖出: %s %s, open_pct=%.2f%%' % (
                get_security_info(stock).display_name,
                stock,
                open_pct,
            ))


def sell_intraday(context):
    """
    11:28 与 14:50 的常规卖出检查。

    本地 `trade()` 在 11:30 / 15:00 检查，这里使用聚宽样例常用的 11:28 / 14:50，
    避免临近午收和收盘时部分分钟撮合不可控。
    """
    current_data = get_current_data()
    now_str = context.current_dt.strftime('%H:%M:%S')

    for stock in list(context.portfolio.positions.keys()):
        position = context.portfolio.positions[stock]
        if position.closeable_amount <= 0 or stock not in current_data:
            continue

        snapshot = current_data[stock]
        last_price = snapshot.last_price
        high_limit = snapshot.high_limit
        low_limit = snapshot.low_limit

        if pd.isnull(last_price) or pd.isnull(high_limit) or pd.isnull(low_limit):
            continue

        # 跌停附近卖出通常无法成交，先等待后续时点。
        if last_price <= low_limit:
            continue

        # 止损优先级高于未涨停卖出：只要 T+1 可卖且跌破买入价 9%，立即清仓。
        if position.avg_cost > 0 and last_price <= position.avg_cost * (1 - g.stop_loss_pct):
            order_target_value(stock, 0)
            log.info('止损卖出: %s %s, time=%s, price=%.3f, cost=%.3f' % (
                get_security_info(stock).display_name,
                stock,
                now_str,
                last_price,
                position.avg_cost,
            ))
            continue

        # 本地回测条件为 current_price < limit_up * 0.97，即离涨停价超过约 3% 就卖出。
        if last_price < high_limit * 0.97:
            order_target_value(stock, 0)
            log.info('未接近涨停卖出: %s %s, time=%s, price=%.3f, high_limit=%.3f' % (
                get_security_info(stock).display_name,
                stock,
                now_str,
                last_price,
                high_limit,
            ))


def get_signal_df(stock_list, date):
    """
    生成昨日收盘后已知的信号候选表。

    返回值 index 为聚宽股票代码，至少包含：
    - close: 昨日收盘价，用于今日 09:30 计算低开幅度
    - limit_status: 昨日涨停状态，要求为「断板」或「炸板」
    - last_limit_desc: 最近一次有效涨停描述，要求不为「1天1板」
    - sma_7: 昨日 7 日均线
    - lowest_30: 昨日前 30 日最低价
    """
    if not stock_list:
        return pd.DataFrame()

    # 性能优化：先用最近 3 个交易日做严格的「昨日断板/炸板」预筛。
    # 这个预筛只减少后续 45 天历史行情的请求范围，不改变最终信号口径；
    # 后面仍会用完整历史数据重新计算并确认 `limit_status`。
    stock_list = filter_recent_limit_candidates(stock_list, date)
    if not stock_list:
        return pd.DataFrame()

    # 多取一些历史天数，保证 30 日最低价、7 日均线、最近一次涨停描述都有足够样本。
    count = max(g.lowest_window + 5, g.limit_desc_lookback + 5, 45)
    price_df = get_price(
        stock_list,
        end_date=date,
        frequency='daily',
        fields=['open', 'close', 'high', 'low', 'pre_close', 'high_limit', 'low_limit', 'paused'],
        count=count,
        panel=False,
        fill_paused=False,
        skip_paused=False,
    )

    if price_df is None or price_df.empty:
        return pd.DataFrame()

    price_df = price_df.dropna(subset=['code', 'close', 'high', 'low', 'high_limit', 'low_limit'])
    if price_df.empty:
        return pd.DataFrame()

    enriched = []
    for stock, df in price_df.groupby('code'):
        df = df.sort_values('time').copy()
        if df.empty:
            continue

        df = add_limit_features(df)
        last_row = df.iloc[-1].copy()

        # 昨日必须是断板或炸板。注意：本地 notebook 用的是当日行里的 `prev_limit_status`，
        # 对聚宽 09:30 买入来说，昨日就是可观测的上一交易日，因此这里直接判断昨日状态。
        if last_row['limit_status'] not in ['断板', '炸板']:
            continue

        if pd.isnull(last_row['sma_7']) or last_row['close'] < last_row['sma_7']:
            continue

        # 本地 notebook 的 `last_limit_desc` 是在买入日行上通过昨日状态回看得到的；
        # 聚宽 09:30 尚无完整日线行，所以这里显式用「截至昨日」的数据模拟买入日视角。
        last_limit_desc = calc_signal_last_limit_desc(df)
        last_row['last_limit_desc'] = last_limit_desc
        if pd.isnull(last_row['last_limit_desc']) or last_row['last_limit_desc'] == '1天1板':
            continue

        if pd.isnull(last_row['lowest_30']) or last_row['lowest_30'] <= 0:
            continue

        # 绝对位置过滤：开盘价要到 09:30 才知道，这里先用昨日收盘价做粗过滤；
        # 买入函数里会用真实开盘价再次过滤 `day_open / lowest_30 <= 3`。
        if last_row['close'] / last_row['lowest_30'] > g.relative_lowest_max * 1.1:
            continue

        enriched.append(last_row)

    if not enriched:
        return pd.DataFrame()

    result = pd.DataFrame(enriched).set_index('code')
    result = filter_by_market_cap(result, date)

    # 今日 09:30 开盘价已知后，补做一次精确的 `open / lowest_30 <= 3` 过滤。
    current_data = get_current_data()
    keep = []
    for stock in list(result.index):
        if stock not in current_data:
            continue
        day_open = current_data[stock].day_open
        lowest_30 = result.loc[stock, 'lowest_30']
        if pd.notnull(day_open) and pd.notnull(lowest_30) and lowest_30 > 0:
            if day_open / lowest_30 <= g.relative_lowest_max:
                keep.append(stock)

    return result.loc[keep] if keep else pd.DataFrame()


def add_limit_features(df):
    """
    给单只股票日线数据添加本地策略所需的涨停状态和描述字段。

    字段含义对齐本地 `my_utils.fun`：
    - is_limit_up: 收盘价接近涨停价，视为涨停
    - is_broken_limit: 盘中触及涨停但收盘未封住，视为炸板
    - limit_status: 涨停 / 炸板 / 断板 / 未涨停
    - limit_desc: 对涨停日描述为 n天m板
    - last_limit_desc: 昨日之前最近一次有效涨停描述
    """
    df = df.copy()
    df['is_limit_up'] = df['close'] >= df['high_limit'] * 0.999
    df['is_broken_limit'] = (df['high'] >= df['high_limit'] * 0.999) & (~df['is_limit_up'])
    df['sma_7'] = df['close'].rolling(g.sma_window, min_periods=1).mean()

    # 本地 `cal_n_lowest(include_today=False)` 不包含当天，所以这里先 rolling 再 shift。
    df['lowest_30'] = df['low'].rolling(g.lowest_window, min_periods=1).min().shift(1)

    status_list = []
    for i in range(len(df)):
        if df.iloc[i]['is_limit_up']:
            status_list.append('涨停')
        elif df.iloc[i]['is_broken_limit']:
            status_list.append('炸板')
        else:
            # 本地 mark_limit_status 默认 db_days=2：最近 2 天内有涨停，且今日未涨停/炸板，则标为断板。
            start = max(0, i - 2)
            recent_limit = df.iloc[start:i]['is_limit_up'].any()
            status_list.append('断板' if recent_limit else '未涨停')

    df['limit_status'] = status_list
    df['limit_desc'] = calc_limit_desc(df)
    df['last_limit_desc'] = calc_last_limit_desc(df)
    return df


def calc_limit_desc(df):
    """
    计算「n天m板」描述。

    这里按本地函数逻辑处理：从最近一个「未涨停」后的下一天开始计数，
    涨停/炸板日记录区间天数和区间内涨停次数。
    """
    desc_list = []
    period_start = 0

    for i in range(len(df)):
        status = df.iloc[i]['limit_status']
        total_days = i - period_start + 1
        up_days = int(df.iloc[period_start:i + 1]['is_limit_up'].sum())

        if status in ['涨停', '炸板']:
            desc_list.append('%d天%d板' % (total_days, up_days))
        elif status == '断板':
            desc_list.append('断板')
        else:
            desc_list.append('未涨停')
            period_start = i + 1

    return desc_list


def calc_last_limit_desc(df):
    """
    记录当前行之前最近一次「涨停」对应的 n天m板描述。

    本地策略用它排除最近一次涨停为 `1天1板` 的股票，保留更接近多板/反包形态的样本。
    """
    result = []
    period_start = 0
    last_valid_desc = None

    for i in range(len(df)):
        if i == 0:
            result.append(None)
            continue

        prev_status = df.iloc[i - 1]['limit_status']
        if prev_status in ['涨停', '炸板', '断板']:
            for j in range(i - 1, period_start - 1, -1):
                if df.iloc[j]['limit_status'] == '涨停':
                    last_valid_desc = df.iloc[j]['limit_desc']
                    break
            result.append(last_valid_desc)
        else:
            result.append(None)
            period_start = i - 1
            last_valid_desc = None

    return result


def calc_signal_last_limit_desc(df):
    """
    为「下一交易日买入信号」计算最近一次涨停描述。

    本地 `mark_last_limit_desc` 的当前行会查看前一日状态；聚宽在 09:30 只能拿到昨日
    完整日线，因此这里把昨日当作“当前买入日的前一日”，从昨日向前寻找最近一次真正涨停。
    若在找到涨停前先遇到「未涨停」，说明当前连板/断板周期已经结束，返回 None。
    """
    if df.empty:
        return None

    last_status = df.iloc[-1]['limit_status']
    if last_status not in ['涨停', '炸板', '断板']:
        return None

    for i in range(len(df) - 1, -1, -1):
        status = df.iloc[i]['limit_status']
        if status == '涨停':
            return df.iloc[i]['limit_desc']
        if status == '未涨停':
            return None

    return None


def filter_by_market_cap(signal_df, date):
    """
    用聚宽估值表过滤流通市值。

    本地字段 `mv_A_free_float` 更接近自由流通市值；聚宽策略环境中通常可直接使用
    `valuation.circulating_market_cap`，单位一般为亿元。若你的聚宽账号字段命名不同，
    可在这里替换为实际可用字段。
    """
    if signal_df.empty:
        return signal_df

    q = query(
        valuation.code,
        valuation.circulating_market_cap,
    ).filter(
        valuation.code.in_(list(signal_df.index)),
        valuation.circulating_market_cap >= g.mv_min,
        valuation.circulating_market_cap <= g.mv_max,
    )
    valuation_df = get_fundamentals(q, date=date)

    if valuation_df is None or valuation_df.empty:
        return pd.DataFrame()

    valid_codes = set(valuation_df['code'].tolist())
    return signal_df.loc[[code for code in signal_df.index if code in valid_codes]]


def filter_by_market_cap_list(stock_list, date):
    """
    批量按流通市值过滤股票列表。

    这个函数放在取 45 天历史行情之前执行，目的是先缩小股票池，减少后续
    `get_price(..., count=45)` 的数据量；过滤字段和阈值与最终信号保持一致。
    """
    if not stock_list:
        return []

    q = query(
        valuation.code,
        valuation.circulating_market_cap,
    ).filter(
        valuation.code.in_(stock_list),
        valuation.circulating_market_cap >= g.mv_min,
        valuation.circulating_market_cap <= g.mv_max,
    )
    valuation_df = get_fundamentals(q, date=date)
    if valuation_df is None or valuation_df.empty:
        return []

    return valuation_df['code'].tolist()


def filter_recent_limit_candidates(stock_list, date):
    """
    用最近 3 个交易日预筛昨日为「断板」或「炸板」的股票。

    本地 `mark_limit_status(db_days=2)` 判断昨日断板只需要知道：
    - 昨日没有涨停，也没有炸板
    - 昨日前 2 个交易日内至少有 1 天涨停
    因此 3 日数据足够完成这个预筛；后续完整 45 日数据仍会再次确认信号。
    """
    if not stock_list:
        return []

    df = get_price(
        stock_list,
        end_date=date,
        frequency='daily',
        fields=['close', 'high', 'high_limit'],
        count=3,
        panel=False,
        fill_paused=False,
        skip_paused=False,
    )
    if df is None or df.empty:
        return []

    df = df.dropna(subset=['code', 'close', 'high', 'high_limit'])
    if df.empty:
        return []

    keep = []
    for stock, sub_df in df.groupby('code'):
        sub_df = sub_df.sort_values('time')
        if sub_df.empty:
            continue

        is_limit_up = sub_df['close'] >= sub_df['high_limit'] * 0.999
        is_broken_limit = (sub_df['high'] >= sub_df['high_limit'] * 0.999) & (~is_limit_up)

        yesterday_is_limit = bool(is_limit_up.iloc[-1])
        yesterday_is_broken = bool(is_broken_limit.iloc[-1])
        recent_has_limit = bool(is_limit_up.iloc[:-1].tail(2).any())

        if yesterday_is_broken:
            keep.append(stock)
        elif (not yesterday_is_limit) and (not yesterday_is_broken) and recent_has_limit:
            keep.append(stock)

    return keep


def prepare_stock_list(date):
    """
    构建每日基础股票池。

    过滤顺序尽量保持轻量：
    1. 全市场股票
    2. 剔除北交所、科创板、创业板、B 股等不在本策略交易范围内的股票
    3. 剔除上市不足 250 个自然日的新股
    4. 剔除 ST 与停牌股票
    """
    securities_df = get_all_securities('stock', date=date)
    initial_list = securities_df.index.tolist()
    initial_list = filter_board_stock(initial_list)
    initial_list = filter_new_stock(initial_list, date, days=250, securities_df=securities_df)
    initial_list = filter_st_stock(initial_list, date)
    initial_list = filter_paused_stock(initial_list, date)
    initial_list = filter_by_market_cap_list(initial_list, date)
    return initial_list


def filter_board_stock(stock_list):
    """
    剔除当前策略不交易的板块。

    聚宽代码格式示例：
    - 300xxx.XSHE: 创业板
    - 688xxx.XSHG: 科创板
    - 200xxx.XSHE / 900xxx.XSHG: B 股
    - 4/8 开头: 北交所常见代码段
    """
    result = []
    for stock in stock_list:
        code = stock.split('.')[0]
        if code.startswith(('300', '688', '200', '900', '4', '8')):
            continue
        result.append(stock)
    return result


def filter_new_stock(stock_list, date, days=250, securities_df=None):
    """
    剔除上市时间不足指定天数的新股。

    性能优化点：优先复用 `get_all_securities()` 已经返回的 start_date 批量过滤，
    避免对几千只股票逐只调用 `get_security_info()`。
    """
    if not stock_list:
        return []

    current_date = transform_date(date, 'd')
    if securities_df is not None and 'start_date' in securities_df.columns:
        df = securities_df.loc[[stock for stock in stock_list if stock in securities_df.index]]
        df = df[df['start_date'].apply(
            lambda x: current_date - pd.to_datetime(x).date() > dt.timedelta(days=days)
        )]
        return df.index.tolist()

    # 兼容兜底：如果聚宽环境返回的证券表没有 start_date，再退回逐只查询。
    result = []
    for stock in stock_list:
        info = get_security_info(stock)
        if info is not None and current_date - info.start_date > dt.timedelta(days=days):
            result.append(stock)
    return result


def filter_st_stock(stock_list, date):
    """按聚宽 ST 扩展字段剔除 ST 股票。"""
    if not stock_list:
        return []

    df = get_extras('is_st', stock_list, start_date=date, end_date=date, df=True)
    if df is None or df.empty:
        return stock_list

    st_series = df.iloc[0]
    return [stock for stock in stock_list if stock in st_series.index and not bool(st_series[stock])]


def filter_paused_stock(stock_list, date):
    """剔除昨日停牌股票，避免日线特征和今日开盘状态不连续。"""
    if not stock_list:
        return []

    df = get_price(
        stock_list,
        end_date=date,
        frequency='daily',
        fields=['paused'],
        count=1,
        panel=False,
        fill_paused=True,
    )
    if df is None or df.empty:
        return []

    df = df[df['paused'] == 0]
    return list(df['code'])


def is_untradable_at_open(stock, snapshot):
    """
    09:30 买入前的实时可交易性检查。

    低开策略理论上不会在涨停买入，但仍保留涨跌停和停牌检查，避免极端数据导致错误下单。
    """
    if snapshot.paused:
        return True
    if snapshot.is_st:
        return True

    day_open = snapshot.day_open
    if pd.isnull(day_open):
        return True

    if pd.notnull(snapshot.high_limit) and day_open >= snapshot.high_limit:
        return True
    if pd.notnull(snapshot.low_limit) and day_open <= snapshot.low_limit:
        return True

    return False


def get_previous_close(stock, previous_date):
    """获取上一交易日收盘价，用于 09:30 大幅低开判断。"""
    df = get_price(
        stock,
        end_date=previous_date,
        frequency='daily',
        fields=['close'],
        count=1,
        panel=False,
        fill_paused=False,
        skip_paused=False,
    )
    if df is None or df.empty:
        return None
    return df.iloc[-1]['close']


def transform_date(date, date_type):
    """
    日期格式转换工具，兼容聚宽 context 中常见的 str / date / datetime。

    date_type:
    - 'str': YYYY-MM-DD 字符串
    - 'dt': datetime.datetime
    - 'd': datetime.date
    """
    if isinstance(date, str):
        str_date = date
        dt_date = dt.datetime.strptime(date, '%Y-%m-%d')
        d_date = dt_date.date()
    elif isinstance(date, dt.datetime):
        str_date = date.strftime('%Y-%m-%d')
        dt_date = date
        d_date = date.date()
    elif isinstance(date, dt.date):
        str_date = date.strftime('%Y-%m-%d')
        dt_date = dt.datetime.strptime(str_date, '%Y-%m-%d')
        d_date = date
    else:
        raise TypeError('不支持的日期类型: %s' % type(date))

    return {
        'str': str_date,
        'dt': dt_date,
        'd': d_date,
    }[date_type]


def to_date_str(date):
    """将 date / datetime / str 统一转为聚宽 API 常用的 YYYY-MM-DD 字符串。"""
    return transform_date(date, 'str')
