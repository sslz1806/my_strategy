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

def plot_interactive_curve(df, title="净值折线图",max_ticks=20):
    import pandas as pd
    import plotly.graph_objects as go
    """
    直接基于宽格式DataFrame绘制可交互折线图（每列一条线，无需转长格式）
    参数：
        df: 输入DataFrame（索引为x轴，每列为一条折线）
        title: 图表标题（可选）
    """
    x_categories = df.index.astype(str)  # 日期转字符串，作为分类轴
    # 初始化绘图对象
    fig = go.Figure()
    
    # 遍历每一列，添加折线（核心：宽格式直接绘制）
    for col in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,          # x轴：原DF索引（日期/时间等）
                y=df[col],           # y轴：当前列的数值
                name=col,            # 折线名称=列名
                mode="lines+markers",# 显示折线+数据点（更清晰）
                hovertemplate=f"{col}: %{{y}}<br>%{{x}}<extra></extra>"  # 悬浮提示格式
            )
        )
    
    # 基础样式配置（极简但够用）
    fig.update_layout(
        title=title,
        xaxis_title=df.index.name or "X轴",  # 自动取索引名作为x轴标签
        yaxis_title="数值",
        width=1400,
        height=600,
        template="plotly_white",
        hovermode="x unified",  # 同x轴位置悬浮时显示所有列数值
    )
    n = len(df.index)
    step = max(1, n // max_ticks)  # 根据数据量自动稀疏
    # 优化x轴标签显示（避免重叠）
    fig.update_xaxes(
        tickangle=-45,            # 标签旋转45°
        tickfont=dict(size=6),   # 标签字体大小
        type="category",           # 显式指定x轴为分类轴（关键！）
        dtick=step
    )
    # 显示交互图表
    fig.show()

def plot_interactive_bar_chart(df, title="净值柱状图", x_label=None, y_label="数值", max_ticks=20):
    import pandas as pd
    import plotly.graph_objects as go
    """
    基于宽格式DataFrame绘制可交互柱状图（每列一组柱子，同索引位置并列展示）
    参数：
        df: 输入DataFrame（索引为x轴分类，每列为一组柱子）
        title: 图表标题（可选）
        x_label: x轴标签（可选，默认用df索引名）
        y_label: y轴标签（可选，默认"数值"）
    """
    # 初始化绘图对象
    fig = go.Figure()

    # 遍历每一列，添加柱状图系列（宽格式直接绘制）
    for col in df.columns:
        fig.add_trace(
            go.Bar(
                x=df.index,          # x轴：原DF索引（日期/分类等）
                y=df[col],           # y轴：当前列的数值
                name=col,            # 柱子系列名称=列名
                opacity=0.85,        # 透明度（避免遮挡）
                # 悬浮提示格式（显示系列名、数值、索引）
                hovertemplate=f"{col}: %{{y:.4f}}<br>{df.index.name or '分类'}: %{{x}}<extra></extra>"
            )
        )

    # 布局配置（适配宽画布+交互体验）
    fig.update_layout(
        title=title,
        xaxis_title=x_label or df.index.name or "X轴",
        yaxis_title=y_label,
        width=1500,  # 宽度拉满（可根据需求调整，比如2000/3000）
        height=600,
        template="plotly_white",  # 清爽的浅色模板
        barmode="group",          # 并列柱状图（核心：同x轴位置多列并列）
        bargap=0.1,               # 不同x轴分类间的间距
        bargroupgap=0.1,          # 同x轴分类内不同列的间距
        # 图例放在顶部横向排列（适配宽画布）
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode="x unified"     # 鼠标悬浮时，同x轴位置的所有系列数值都显示
    )

    # 优化x轴标签显示（避免重叠）
    fig.update_xaxes(
        tickangle=-45,            # 标签旋转45°
        tickfont=dict(size=6),     # 标签字体大小
        type = "category",
        dtick=max(1, len(df.index) // max_ticks)  # 根据数据量自动稀疏
    )

    # 显示可交互图表
    fig.show()

def plot_bar_chart(df, title="净值条形图", x_label=None, y_label="数值"):
    """
    基于宽格式DataFrame绘制非交互条形图（每列一组条形，同索引位置并列展示）
    参数：
        df: 输入DataFrame（索引为x轴分类，每列为一组条形）
        title: 图表标题（可选）
        x_label: x轴标签（可选，默认用df索引名）
        y_label: y轴标签（可选，默认"数值"）
    """
    import matplotlib.pyplot as plt
    import numpy as np
    # 设置中文字体（避免中文乱码，根据系统调整）
    plt.rcParams["font.sans-serif"] = ["SimHei"]  # Windows
    # plt.rcParams["font.sans-serif"] = ["PingFang SC"]  # Mac
    plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

    # 初始化画布
    fig, ax = plt.subplots(figsize=(18, 7))  

    # 核心修改1：将日期索引转为字符串（消除时间轴间隔）
    index_str = df.index.astype(str)  # 日期→字符串，作为分类标签
    x_pos = np.arange(len(index_str))  # x轴位置仅对应有数据的索引

    # 计算条形宽度和位置（实现并列显示）
    bar_width = 0.8 / len(df.columns)  

    # 遍历每一列，绘制条形
    for i, col in enumerate(df.columns):
        # 计算当前列条形的x轴偏移（实现并列）
        bar_pos = x_pos + (i - len(df.columns)/2 + 0.5) * bar_width
        # 绘制条形
        ax.bar(
            bar_pos, 
            df[col].values, 
            width=bar_width, 
            label=col,  # 图例名称=列名
            alpha=0.8   # 透明度（避免遮挡）
        )

    # 样式配置
    ax.set_title(title, fontsize=16, pad=20)  # 标题
    ax.set_xlabel(x_label or df.index.name or "X轴", fontsize=9)  # x轴标签
    ax.set_ylabel(y_label, fontsize=12)  # y轴标签
    ax.set_xticks(x_pos)  # x轴刻度仅对应有数据的位置
    # 核心修改2：使用字符串化的索引作为刻度标签
    ax.set_xticklabels(index_str, rotation=60, ha="right")  
    ax.legend(title="数据系列", bbox_to_anchor=(1.05, 1), loc="upper left")  # 图例靠右显示
    ax.grid(axis="y", linestyle="--", alpha=0.3)  # 显示y轴网格线
    plt.tight_layout()  # 自动调整布局，避免标签截断

    # 显示图表
    plt.show()
