import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _color_spectrum(num):
    num = 2
    colors = [[0, 0, 0, 0] for _ in range(num)]

    r = [194 for _ in range(num)]
    g = list(reversed(range(140, 202, int((202-140)/num))))
    g = g[0: 15]
    b = g
    a = [1 for _ in range(num)]

    rgba = {'r': r, 'g': g, 'b': b, 'a': a}
    rgba = pd.DataFrame(rgba)
    colors = []      # Create an empty list

    for index, rows in rgba.iterrows():             # Iterate over each row
        my_list =[rows.r, rows.g, rows.b, rows.a]   # Create list for the current row
        colors.append(my_list)                      # append the list to the final list
    colors = [','.join(list(map(str, rgba))) for rgba in colors]
    colors = [f'rgba({rgba})' for rgba in colors]
    return colors

def plot_strategy(
        symbol, data, equity, 
        balance, pnl, orderbook, 
        freq, stats, add_plot, 
        add_plot_separate, min_bars=0, 
        same_timeframe=True, rel_equity=False):
    """ Interactive graph using the plotly framework
    """

    if 'min' in freq:
        dfreq = freq
        n = int(freq.split('m')[0]) / 60
    elif 'hour' in freq:
        dfreq = freq.split('h')[0] + 'h'
        n = int(freq.split('h')[0])
    elif 'daily' in freq:
        dfreq = 'D'
        n = 24
    
    if bool(add_plot_separate) == False:
        subtitle_4 = 'Volume'
    else:
        subtitle_4 = 'Indicator/Signal'

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True, 
        row_heights=[2, 1, 4, 4],
        vertical_spacing=0.05,
        subplot_titles=('Equity', 'P&L', None, subtitle_4),
        specs=[
            [{"secondary_y": True}], 
            [{"secondary_y": True}], 
            [{"secondary_y": True}], 
            [{"secondary_y": True}]
            ]
    )

    fig.update_layout(
        title_text="Backtest results",
        legend=dict(
                y=1,
                x=0.95,
                groupclick="togglegroup"
            ),
        template='plotly_dark', font=dict(family="Courier", size=15),
    )

    fig.update_xaxes(
        rangeslider_visible=False,
        mirror=True,
        ticks='outside',
        showline=True,
    )

    #Equity Curve
    if rel_equity:              #mdd duration
        initial_equity = equity.iloc[0]
        equity = equity/initial_equity
    
    dd_start = stats.loc['Max. Drawdown start']
    dd_end = stats.loc['Max. Drawdown end']
    mdd_d = stats.loc['Max. Drawdown Duration']
    equity.index = pd.to_datetime(equity.index)
    equity_dd = equity[dd_start]
    dd_i_start = equity.index.get_loc(dd_start) + 1
    dd_i_end = equity.index.get_loc(dd_end) + 1
    dd = list(range(dd_i_start, dd_i_end))

    mdd = stats.loc['Max. Drawdown [%]']        #max dd
    mdd_date = stats.loc['Max. Drawdown date']

    try:
        equity_mdd = equity[mdd_date]
        mdd_i_date = equity.index.get_loc(mdd_date)
    except KeyError:
        equity_mdd = None
        mdd_i_date = None

    fig.add_trace(go.Scatter(y=equity,
        customdata=equity.index,
        line=dict(
            color='rgba(8,84,189,1)',
            ), 
        showlegend=True,
        legendgroup='equity',
        legendgrouptitle_text='Equity',
        name='Net equity',
        hovertemplate="<br>".join([
                "<b>%{customdata}</b><br>"
                "<b>value</b>: %{y}$"
                ])
        ),
        row=1, col=1)

    fig.add_trace(go.Scatter(y=balance,
        customdata=balance.index,
        line=dict(
            color='rgba(7,208,36,0.74)',
            ), 
        showlegend=True,
        legendgroup='equity',
        legendgrouptitle_text='Equity',
        name='Net balance',
        hovertemplate="<br>".join([
                "<b>%{customdata}</b><br>"
                "<b>value</b>: %{y}$"
                ])
        ),
        row=1, col=1)

    fig.add_trace(go.Scatter(y=equity.expanding().max(),
        line=dict(
            color='rgba(199,186,17,0.66)', 
            # width=4
            ), 
        showlegend=False,
        hoverinfo='none',
        legendgroup='equity',
        legendgrouptitle_text='Equity',
        name='Peaks',
            ),
    row=1, col=1
    )

    fig.add_trace(go.Scatter(x=dd, y=[equity_dd for _ in dd],
        line=dict(
            color='rgba(225,20,3,1)', 
            # width=4
            ),
        hoverinfo='none', 
        showlegend=True,
        legendgroup='equity',
        legendgrouptitle_text='Equity',
        name=f'Max. Drawdown Duration ({mdd_d.days} days)'
            ), 
    row=1, col=1
    )

    if equity_mdd != None:
        fig.add_trace(go.Scatter(x=[mdd_i_date], y=[equity_mdd],
            customdata=[mdd_date],
            mode='markers',
            marker=dict(
                color='rgba(225,20,3,1)', 
                size=8,
                ), 
            showlegend=True,
            legendgroup='equity',
            legendgrouptitle_text='Equity',
            name=f'Max. Drawdown ({np.round(mdd, 2)}%)',
            hovertemplate="<br>".join([
                    "<b>%{customdata}</b>",
                    "<b>price</b>: %{y}$",
                    ])
            ), 
        row=1, col=1
        )

    #PnL
    if len(pnl) != 0:
        pnl_pos = pnl[pnl['gain'] > 0]
        pnl_neg = pnl[pnl['gain'] <= 0]
        max_gain = pnl_pos['gain'].max()
        max_loss = pnl_neg['gain'].min()

        pnl_avg = pnl['pnl'].mean() + np.zeros(data[symbol[0]]['date'].shape[0])

        fig.add_trace(go.Scatter(y=pnl_avg,
                        line=dict(
                            color='rgba(255,255,255,0.76)', 
                            # width=4, 
                            dash='dash'), 
                        showlegend=False,
                        legendgroup='P&L',
                    name=None), 
                    row=2, col=1, secondary_y=False)

        #Profits
        pnl_i_pos = [equity.index.get_loc(_) + 1 for _ in pnl_pos['date']]

        fig.add_trace(go.Scatter(x=pnl_i_pos, y=pnl_pos['pnl'], 
                    customdata=np.stack((pnl_pos['date'], pnl_pos['gain']), axis=-1),
                    mode="markers", 
                        marker = dict(
                            size = 10 * pnl_pos['gain']/max_gain + 10, 
                            symbol = 'triangle-up', 
                            color ='rgba(6,221,0,0.76)',
                            ), 
                    name='Profits', 
                    showlegend=True, 
                    legendgroup='P&L',
                    legendgrouptitle_text='P&L',
                    hovertemplate="<br>".join([
                            "<b>%{customdata[0]}</b>",
                            "%{y}%",
                            ])
                    ), 
                row=2, col=1, secondary_y=False)
        
        #losses
        pnl_i_neg = [equity.index.get_loc(_) + 1 for _ in pnl_neg['date']]

        fig.add_trace(go.Scatter(x=pnl_i_neg, y=pnl_neg['pnl'], 
                    customdata=np.stack((pnl_neg['date'], pnl_neg['gain']), axis=-1),
                    mode="markers", 
                        marker = dict(
                            size = 10 * pnl_neg['gain']/max_loss + 10, 
                            symbol = 'triangle-down', 
                            color ='rgba(231,0,0,0.76)',
                            ), 
                    name='Losses', 
                    showlegend=True, 
                    legendgroup='P&L',
                    legendgrouptitle_text='P&L',
                    hovertemplate="<br>".join([
                            "<b>%{customdata[0]}</b>",
                            "%{y}%",
                            ])
                    ), 
                row=2, col=1, secondary_y=False)

    #OHLCV data and buy and sell points
    c = 0
    colors = _color_spectrum(len(symbol))
    for sym in symbol:
        df = data[sym]
        if df.shape[0] < 10000:
            fig.add_trace(go.Candlestick(
                    # x=df["date"],
                    open=df["open"], 
                    high=df["high"],
                    low=df["low"], 
                    close=df["close"], 
                    legendgroup=sym,
                    legendgrouptitle_text=sym,
                    name='OHLC',
                    text=[f'<b>{_}</b>' for _ in df["date"]]
                    ), 
                row=3, col=1, 
                secondary_y=False,
            )
        else:
            fig.add_trace(go.Scatter(
                    # x=df["date"], 
                    y=df["close"], 
                    legendgroup=sym,
                    legendgrouptitle_text=sym,
                    name='Close',
                    ), 
                row=3, col=1, 
                secondary_y=False,
            )

        if len(orderbook) != 0:
            ob = orderbook[orderbook['symbol'] == sym]
            BUY = ob[ob['order-type'] == 'BUY']
            SELL = ob[ob['order-type'] == 'SELL']

            i_BUY = [equity.index.get_loc(_) for _ in BUY['date']]
            
            fig.add_trace(go.Scatter(x=i_BUY , y=BUY['price'], 
                    customdata=np.stack((BUY['date'], BUY['units'], BUY['units'] * BUY['price']), axis=-1),
                    mode="markers", 
                        marker = dict(
                            size = 15, 
                            symbol = 'triangle-up', 
                            color ='rgba(6,221,0,0.76)',
                            ), 
                    name='BUY', 
                    showlegend=True, 
                    legendgroup=sym,
                    legendgrouptitle_text=sym,
                    hovertemplate="<br>".join([
                            "<b>%{customdata[0]}</b><br>"
                            "<b>price</b>: %{y}$",   
                            "<b>units</b>: %{customdata[1]}",
                            "<b>total</b>: %{customdata[2]}",
                            ])
                    ), 
                row=3, col=1, secondary_y=False)
            
            i_SELL = [equity.index.get_loc(_) for _ in SELL['date']]

            fig.add_trace(go.Scatter(x=i_SELL , y=SELL['price'], 
                    customdata=np.stack((SELL['date'], SELL['units'], SELL['units'] * SELL['price']), axis=-1),
                    mode="markers", 
                        marker = dict(
                            size = 15, 
                            symbol = 'triangle-down', 
                            color ='rgba(231,0,0,0.76)',
                            ), 
                    name='SELL', 
                    showlegend=True, 
                    legendgroup=sym,
                    legendgrouptitle_text=sym,
                    hovertemplate="<br>".join([
                            "<b>%{customdata[0]}</b><br>"
                            "<b>price</b>: %{y}$",   
                            "<b>units</b>: %{customdata[1]}",
                            "<b>total</b>: %{customdata[2]}",
                            ])
                    ),  
                row=3, col=1, secondary_y=False)

        if bool(add_plot_separate) == False:
            fig.add_trace(go.Bar(
                    customdata=df['date'], 
                    y=df['volume'], 
                    name='Volume', 
                    showlegend=True, 
                    legendgroup=sym,
                    legendgrouptitle_text=sym,
                    marker_color=colors[c],
                    hovertemplate="<br>".join([
                            "<b>%{customdata}</b><br>"
                            "%{y}",
                            ])
                    ), 
                row=4, col=1)
            
        c+=1
    
    nans = pd.Series([np.nan for _ in range(min_bars)])
    for p in add_plot: 
        ppp = pd.concat([nans, p])
        fig.add_trace(go.Scatter(
            customdata=ppp.index, 
            y=ppp,
            # line=dict(
            #     color='rgba(8,84,189,1)',
            #     ), 
            showlegend=True,
            legendgroup='Indicators',
            legendgrouptitle_text='Indicators',
            name=p.name,
            hovertemplate="<br>".join([
                    "<b>%{customdata}</b><br>"
                    "%{y}$",
                    ])
                ), 
            row=3, col=1    
            )

    for p in add_plot_separate: 
        ppp = pd.concat([nans, p])
        fig.add_trace(go.Scatter(
            customdata=ppp.index, 
            y=ppp,
            # line=dict(
            #     color='rgba(8,84,189,1)',
            #     ), 
            showlegend=True,
            legendgroup='Indicators',
            legendgrouptitle_text='Indicators',
            name=p.name,
            hovertemplate="<br>".join([
                    "<b>%{customdata}</b><br>"
                    "%{y}",
                    ])
                ), 
            row=4, col=1    
            )
    
    return fig

def plot_scatter(symbol, data):
    fig = make_subplots(
    rows=1, cols=1,
    shared_xaxes=True, 
    # row_heights=[1],
    vertical_spacing=0.05,
    subplot_titles=('Scatter'),
    specs=[
        [{"secondary_y": True}], 
       ])

    fig.update_layout(
        title_text="Analysis",
        legend=dict(
                y=1,
                x=0.95,
                groupclick="togglegroup"
            ),
        template='plotly_dark', font=dict(family="Courier", size=15),
    )

    fig.update_xaxes(
        rangeslider_visible=False,
        mirror=True,
        ticks='outside',
        showline=True,
    )

    X, Y = data[symbol[0]]['close'], data[symbol[1]]['close']

    fig.add_trace(go.Scatter(x=X, y=Y,
            mode='markers',
            marker=dict(
                color=list(range(len(X))), 
                size=8,
                ), 
            showlegend=True,
            legendgroup='analysis',
            legendgrouptitle_text='Analysis',
            name='',
            # hovertemplate="<br>".join([
            #         "<b>%{customdata}</b><br>"
            #         "<b>price</b>: %{y}$"
            #         ])
            ), 
        row=1, col=1
        )
        
    return fig

if __name__ == '__main__':
    from backtester import Backtest
    # bt = Backtest(['amd', 'tsla'], '10min', '2022-02-14', '2022-02-28', 10000, verbose=False)
    # bt.place_buy_order('AMD', 330, amount=bt.net_wealth)
    # bt.place_sell_order('AMD', 420, amount=bt.net_wealth)
    # bt.place_buy_order('AMD', 430, amount=bt.net_wealth)
    # bt.close_out(437)
    # bt = Backtest(['amd', 'tsla'], '30min', '2022-02-14', '2022-02-28', 10000, verbose=False)
    # bt.place_buy_order('AMD', 4, amount=bt.net_wealth)
    # bt.place_sell_order('AMD', 25, amount=bt.net_wealth)
    # bt.place_buy_order('AMD', 80, amount=bt.net_wealth)
    # bt.close_out(150)
    # bt = Backtest(['amd', 'tsla'], '1hour', '2022-02-14', '2022-02-28', 10000)
    # print(bt.data)
    # bt.place_buy_order('AMD', 10, amount=bt.net_wealth)
    # bt.place_sell_order('AMD', 25, amount=bt.net_wealth)
    # bt.place_buy_order('AMD', 50, amount=bt.net_wealth)
    # bt.close_out(72)
    bt = Backtest(['amd', 'tsla'], 'daily', '2022-02-14', '2022-05-28', 10000 ,verbose=False)
    bt.place_buy_order('AMD', 10, amount=bt.net_wealth)
    bt.place_sell_order('AMD', 25, amount=bt.net_wealth)
    bt.place_buy_order('AMD', 50, amount=bt.net_wealth)
    # bt.min_bars = 10
    bt.close_out(72)
    plot_strategy(
        symbol=bt.symbol, 
        data=bt.data,
        equity=bt.equity, 
        balance=bt.balance,
        pnl=bt.pnl, 
        same_timeframe=bt.same_timeframe, 
        orderbook=bt.orderbook,
        stats=bt._statistics(),
        freq=bt.freq,
    )



