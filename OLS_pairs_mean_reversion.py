import numpy as np
import pandas as pd
from backtest_logic.backtester import Backtest

#---------#
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import coint



class Pairs_Mean_Reversion_OLS(Backtest):
    def run_mr_strategy(self):
        ''' Backtesting an SMA-based strategy.

        Parameters
        ==========
        SMA1, SMA2: int
        shorter and longer term simple moving average (with the given timeframe)
        '''
        msg = '\n\nRunning a pairs mean reversion strategy using OLS'
        msg += f'\nfixed costs {self.ftc} | '
        msg += f'proportional costs {self.ptc}'
        print(msg)
        print('=' * 55)

        lookback = 100

        symbol = self.symbol
        S1 = self.symbol[0].upper()
        S2 = self.symbol[1].upper()
        S1_position = 0
        S2_position = 0
        data = self.data
        model = LinearRegression()

        plot_z_score, plot_gamma, plot_pvalue = [], [], []
        plot_z_date = []

        S = []

        self.min_bars = lookback


        for bar in range(lookback, data[S1].shape[0]):
            S1_log_prices = np.log(data[S1]['open'].iloc[bar-lookback: bar])
            S2_log_prices = np.log(data[S2]['open'].iloc[bar-lookback: bar])

            score, pvalue, _ = coint(S1_log_prices, S2_log_prices)
            plot_pvalue.append(1- pvalue)

            X = np.array(S2_log_prices).reshape(-1, 1)
            Y = np.array(S1_log_prices)

            regression = model.fit(X, Y)

            gamma = regression.coef_[0]
            mu = regression.intercept_

            S1_log_price = np.log(data[S1]['open'].iloc[bar])
            S2_log_price = np.log(data[S2]['open'].iloc[bar])

            spread_t = S1_log_price - (gamma * S2_log_price) - mu

            if bar == lookback:
                # spread_mean = spread.mean()
                # spread_std = spread.std()
                spread = S1_log_prices - (gamma * S2_log_prices) - mu
                S = list(spread)
                       
            S.append(spread_t)


            spread_mean = pd.Series(data=S).mean() 
            spread_std = pd.Series(data=S).std()

            z_score_t = (spread_t - spread_mean) / spread_std

            # threshold = 0.7
            threshold = 20.6

            plot_z_date.append(self.get_date_price(S1, bar)[0])
            plot_z_score.append(z_score_t)
            plot_gamma.append(gamma)

            if S1_position == 0:
                if z_score_t < -threshold:
                    # self.place_buy_order(S1, bar, units=1)
                    # self.place_buy_order(S2, bar, units=1)
                    self.place_buy_order(S1, bar, amount=self.net_wealth)
                    S1_position = 1
                    if S2_position == 1:
                        # self.place_sell_order(S2, bar, units=1)
                        S2_position = 0
            
            if S1_position == 1:
                if z_score_t > threshold:
                    # self.place_sell_order(S1, bar, units=1)
                    # self.place_sell_order(S2, bar, units=1)
                    self.place_sell_order(S1, bar, amount=self.net_wealth)
                    S1_position = 0
                    if S2_position == 0:
                        # self.place_buy_order(S2, bar, units=1)
                        S2_position = 1

            
        
        plot_z_score = pd.Series(data=plot_z_score, index=plot_z_date, name='z-score')
        plot_pvalue = pd.Series(data=plot_pvalue, index=plot_z_date, name='p_value')
        plot_gamma = pd.Series(data=plot_gamma, index=plot_z_date, name='gamma')
        
        self.add_plot_separate(plot_z_score, plot_pvalue, plot_gamma)      
        self.close_out(bar)

    

if __name__ == '__main__':
    # bt = Pairs_Mean_Reversion_OLS(['pep', 'ko'], 'daily', '2020-01-01', '2022-01-01', 10000, verbose=False)
    # bt = Pairs_Mean_Reversion_OLS(['amd', 'tsla'], '10min', '2022-02-14', '2022-02-28', 10000, verbose=False)
    # bt = Pairs_Mean_Reversion_OLS(['gdx', 'gld'], 'daily', '2020-10-01', '2022-04-28', 10000, verbose=False)
    # bt = Pairs_Mean_Reversion_OLS(['paa', 'pagp'], 'daily', '2020-10-01', '2022-04-28', 10000, verbose=False)
    bt = Pairs_Mean_Reversion_OLS(['EWH', 'EWZ'], 'daily', '2000-08-01', '2002-01-31', 10000, verbose=False)
    bt.run_mr_strategy()
    print(bt.stats)
    # print(bt.data)
    # print(bt.statistics())
    bt.plot()
    # bt.scatter_plot()