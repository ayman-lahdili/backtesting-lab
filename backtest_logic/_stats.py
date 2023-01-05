from dateutil import parser
import pandas as pd
import numpy as np


def performance(name: str, symbol: list, data: dict, equity: pd.Series, tradebook: pd.DataFrame, freq, min_bars=0, Rfr=0):
    ''' Calculates the stastistics used to measure performance
    '''


    if 'min' in freq:
        freq = int(freq.split('m')[0])
        min_lookback = str(min_bars * freq) + 'min'
    elif 'hour' in freq:
        freq = int(freq.split('h')[0]) * 60
        min_lookback = str(min_bars * freq/60) + 'hour'
    elif freq == 'daily':
        freq = 6.5 * 60
        min_lookback = str(min_bars) + 'days'
    
    #create a new dict to analyse the data 
    dt = {}
    for sym in symbol:
        dt[sym] = data[sym].iloc[min_bars: data[sym].shape[0]]

    equity = equity.iloc[min_bars: equity.shape[0]]

    #strategy_returns
    strategy_ret = np.log(equity).diff()
    strategy_ret = strategy_ret.dropna()

    #benchmark_returns equal ratio for each stock in a buy and hold strategy
    benchmark_ret = np.zeros_like(dt[sym]['close'])
    for sym in symbol:
        ratio = 1/len(symbol)
        ret = (np.log(dt[sym]['close']).diff() * ratio)
        benchmark_ret += ret
    benchmark_ret = benchmark_ret.dropna()
    benchmark_ret.index = strategy_ret.index

    #constants
    stats = pd.Series(dtype=object)
    trading_minutes = 6.5 * 60
    year = 252 * (trading_minutes/freq)
    month = (252/12) * (trading_minutes/freq)
    Rfr = Rfr


    ###Statistics
    abs_start_date = data[symbol[0]]['date'].iloc[0]
    start_date = parser.parse(dt[sym]['date'].iloc[0])
    end_date = parser.parse(dt[sym]['date'].iloc[-1])
    duration = end_date - start_date
    initial_equity = equity.iloc[0]
    final_equity = equity.iloc[-1]
    total_returns = (np.exp(strategy_ret.sum()) - 1) 
    buy_hold = (np.exp(benchmark_ret.sum()) - 1) 
    ann_ret = (np.exp(np.mean(strategy_ret) * year) - 1) 
    ann_vol = strategy_ret.std() * np.sqrt(year) 
    sharpe = (ann_ret - Rfr) / ann_vol

    #max dd
    roll_max = strategy_ret.expanding(min_periods=1).max()
    daily_dd = (strategy_ret-roll_max)
    dd = daily_dd.expanding(min_periods=1).min()
    mdd = dd.min()

    try:
        mdd_date = parser.parse(dd.idxmin())
    except TypeError:
        mdd_date = None

    #max dd duration
    peaks = equity.expanding().max()
    frequent = peaks.mode()
    start = int(peaks.searchsorted(frequent, side='left')[0]) + min_bars
    end = int(peaks.searchsorted(frequent, side='right')[0]) - 1
    dd_start = parser.parse(dt[sym].loc[start, 'date'])
    dd_end = parser.parse(dt[sym].loc[end+min_bars, 'date'])
    mdd_d = dd_end - dd_start

    #calmar
    calmar = -(ann_ret-Rfr)/mdd

    #sortino
    ann_downside = strategy_ret.loc[strategy_ret<0].std() * np.sqrt(year)
    sortino = (ann_ret - Rfr) / ann_downside

    #tradebook stats
    tb = tradebook
    num_trades = tb.shape[0]

    try:
        pos = (tb['return'] > 0).value_counts()[True]
        neg = num_trades - pos
        win_rate = (pos)/(pos+neg) * 100
    except KeyError:
        pos = 0
        win_rate = None
    
    best = tb['return'].max()
    worst = tb['return'].min()
    avg = tb['return'].mean()
    max_dur = tb['duration'].max()
    min_dur = tb['duration'].min()
    avg_dur = tb['duration'].mean()


    #Add each statistics to a pd.Series
    stats.loc['Min. lookback'] = min_lookback
    stats.loc['Start (- min lookback)'] = start_date
    stats.loc['End'] = end_date
    stats.loc['Duration'] = duration
    stats.loc['Initial equity [$]'] = initial_equity
    stats.loc['Final equity [$]'] = final_equity
    stats.loc['Total return [%]'] = total_returns * 100
    stats.loc['Buy and Hold [%]'] = buy_hold * 100
    stats.loc['Ann. Return [%]'] = ann_ret * 100
    stats.loc['Ann. Volatility [%]'] = ann_vol * 100
    stats.loc['Sharpe ratio'] = sharpe
    stats.loc['Sortino Ratio '] = sortino
    stats.loc['Calmar Ratio'] = calmar
    stats.loc['Max. Drawdown [%]'] = mdd * 100
    stats.loc['Max. Drawdown date'] = mdd_date
    stats.loc['Max. Drawdown Duration'] = mdd_d
    stats.loc['Max. Drawdown start'] = dd_start
    stats.loc['Max. Drawdown end'] = dd_end
    stats.loc['# Trades'] = num_trades
    stats.loc['Win rate [%]'] = win_rate
    stats.loc['Best Trade [%]'] = best
    stats.loc['Worst Trade [%]'] = worst
    stats.loc['Avg. Trade [%]'] = avg
    stats.loc['Max. Trade Duration'] = max_dur
    stats.loc['Min. Trade Duration'] = min_dur
    stats.loc['Avg. Trade Duration'] = avg_dur
    stats.loc['Strategy'] = name


    return stats

if __name__ == '__main__':
    ###test minute tf
    from backtester import Backtest
    bt = Backtest(['AMD', 'tsla', 'msft'], '10min', '2022-02-14', '2022-02-28', 10000, verbose=False)
    bt.place_buy_order('AMD', 330, amount=bt.net_wealth)
    bt.place_sell_order('AMD', 420, amount=bt.net_wealth)
    bt.place_buy_order('AMD', 430, amount=bt.net_wealth)
    bt.close_out(437)
    bt.plot()
    # bt.plot()

    ###HOUR
    # bt = Backtest(['AMD', 'tsla'], '1hour', '2022-02-14', '2022-02-28', 10000, verbose=False)
    # bt.place_buy_order('AMD', 10, amount=bt.net_wealth)
    # bt.place_sell_order('AMD', 20, amount=bt.net_wealth)
    # bt.place_buy_order('AMD', 30, amount=bt.net_wealth)
    # bt.close_out(30)
    # bt.plot()

    ###DAILY
    # bt = Backtest(['AMD', 'tsla'], 'daily', '2022-02-14', '2022-02-28', 10000, verbose=False)
    # bt.place_buy_order('AMD', 2, amount=bt.net_wealth)
    # bt.place_sell_order('AMD', 3, amount=bt.net_wealth)
    # bt.place_buy_order('AMD', 5, amount=bt.net_wealth)
    # bt.close_out(9)
    # bt.plot()

    #print(parser.isoparse(bt.freq))

    st = performance(name=bt,
        symbol=bt.symbol, 
        data=bt.data,
        equity=bt.equity,
        tradebook=bt.tradebook,
        freq=bt.freq,
        min_bars=20,
        )

    print(st)
    


