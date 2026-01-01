# 画图函数

def plot_interactive_kline(df, title='K线图', add_line_list=['sma_5','sma_10','sma_20']):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from pd_fun import add_sma
    # 筛选指定股票数据
    df_plot = df.copy()
    
    # 自动生成标题（股票名称+代码）
    if 'name' in df_plot.columns and 'code' in df_plot.columns:
        title = f'{df_plot["name"].iloc[0]}({df_plot["code"].iloc[0]})'
    
    # 添加均线数据
    for line in add_line_list:
        # 如果均线不存在，则计算并添加
        if line not in df_plot.columns:
            period = int(line.split('_')[1])
            df_plot = add_sma(df_plot, period)  # 调用你的add_sma函数
    
    # 定义涨/跌颜色（涨红、跌绿）
    df_plot['color'] = df_plot.apply(
        lambda x: 'red' if x['close'] >= x['open'] else 'green', axis=1
    )

    # 创建子图（上：K线+均线；下：成交量）
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,  # 共享X轴
        vertical_spacing=0.05,  # 子图间距
        row_heights=[0.7, 0.3],  # 上70%、下30%
        #subplot_titles=('K线图', "成交量")
    )

    # ------------------- 上子图：K线 + 均线 -------------------
    # 绘制K线（蜡烛图，X轴用索引）
    fig.add_trace(
        go.Candlestick(
            x=df_plot.index,  # 直接用索引（非日期解析）
            open=df_plot['open'],
            high=df_plot['high'],
            low=df_plot['low'],
            close=df_plot['close'],
            increasing_line_color='red',  # 涨红
            decreasing_line_color='green',  # 跌绿
            name='K线',
            # 悬浮提示自定义（显示索引+高开低收+涨跌幅）
            hovertext=df_plot.apply(
                lambda x: (
                    f"时间: {x.name}<br>"
                    f"开盘: {x['open']:.2f}<br>"
                    f"最高: {x['high']:.2f}<br>"
                    f"最低: {x['low']:.2f}<br>"
                    f"收盘: {x['close']:.2f}<br>"
                    f"涨跌幅: {x['pct']:.2f}%"
                ), axis=1
            ),
            hoverinfo='text'
        ),
        row=1, col=1
    )

    # 绘制均线（X轴用索引）
    for line in add_line_list:
        fig.add_trace(
            go.Scatter(
                x=df_plot.index,
                y=df_plot[line],
                mode='lines',
                name=line.upper(),
                line=dict(width=1.5),
                # 均线悬浮提示
                hovertext=df_plot.apply(
                    lambda x, l=line: f"{l.upper()}: {x[l]:.2f}", axis=1
                ),
                hoverinfo='text'
            ),
            row=1, col=1
        )
    

    # 标注买卖点（signal=1 买入，signal=-1 卖出）
    if 'signal' in df_plot.columns:
        buy_df = df_plot[df_plot['signal'] == 1]
        sell_df = df_plot[df_plot['signal'] == -1]

        fig.add_trace(
            go.Scatter(
                x=buy_df.index,
                y=buy_df['low'],
                mode='markers',
                marker=dict(
                    symbol='arrow-bar-up',   # 更长的上箭头
                    color='dodgerblue',      # 与K线红绿区分
                    size=14,
                    line=dict(width=1, color='black')
                ),
                name='买入',
                hovertext=buy_df.apply(
                    lambda x: f"时间: {x.name}<br>买入价: {x['close']:.2f}", axis=1
                ),
                hoverinfo='text'
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=sell_df.index,
                y=sell_df['high'],
                mode='markers',
                marker=dict(
                    symbol='arrow-bar-down', # 更长的下箭头
                    color='orange',          # 与K线红绿区分
                    size=14,
                    line=dict(width=1, color='black')
                ),
                name='卖出',
                hovertext=sell_df.apply(
                    lambda x: f"时间: {x.name}<br>卖出价: {x['close']:.2f}", axis=1
                ),
                hoverinfo='text'
            ),
            row=1, col=1
        )
    # ------------------- 下子图：成交量 -------------------
    fig.add_trace(
        go.Bar(
            x=df_plot.index,
            y=df_plot['volume'],
            marker_color=df_plot['color'],
            name='成交量',
            # 成交量悬浮提示
            hovertext=df_plot.apply(
                lambda x: f"时间: {x.name}<br>成交量: {x['volume']}", axis=1
            ),
            hoverinfo='text'
        ),
        row=2, col=1
    )

    # ------------------- 布局配置（核心修改：图例+X轴字体） -------------------
    fig.update_layout(

        title=dict(
            text=title,
            x=0.05,  # 左上角（0为最左，1为最右）
            y=0.95,  # 顶部（0为最下，1为最上）
            xanchor='left',
            yanchor='top',
            font=dict(size=16)
        ),
        legend=dict(
            x=0.75, y=1.05,  # 图例放在图表顶部居中（y>1表示图外顶部）
            xanchor='center',
            yanchor='top',
            bgcolor='rgba(255,255,255,0.8)',
            orientation='h',  # 图例横向排列（更适配顶部布局）
            font=dict(size=10)  # 图例字体大小（可选调整）
        ),
        
        # 关闭X轴的日期解析，强制为类别轴 + 缩小X轴字体
        xaxis=dict(
            type='category',
            rangeslider=dict(visible=False),
            showgrid=True,
            gridcolor='lightgray',
            tickangle=45,
            tickmode='linear',
            dtick=15,
            tickfont=dict(size=8)  # 核心：X轴刻度字体缩小（可调整为9/10）
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='lightgray'
        ),
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor='white',
            font=dict(size=12)
        ),
        height=800,
        width=1200
    )

    # 子图坐标轴配置（同步下子图X轴字体+样式）
    fig.update_xaxes(
        type='category',
        title_text='时间',
        row=2, col=1,
        tickangle=45,
        tickmode='linear',
        dtick=15,
        tickfont=dict(size=8)  # 下子图X轴字体同步缩小
    )
    fig.update_yaxes(
        title_text='价格',
        row=1, col=1
    )
    fig.update_yaxes(
        title_text='成交量',
        row=2, col=1
    )

    # 显示图像
    fig.show()
