import numpy as np
import pandas as pd
from backtest_logic.backtester import Backtest

#---------#
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import coint, add_constant
from pykalman import KalmanFilter


class Pairs_Mean_Reversion_OLS(Backtest):
    def run_mr_strategy(self):
        ''' Backtesting an SMA-based strategy.

        Parameters
        ==========
        SMA1, SMA2: int
        shorter and longer term simple moving average (with the given timeframe)
        '''
        msg = '\n\nRunning a pairs mean reversion strategy using Kalman Filters'
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
        plot_std, plot_mean = [], []
        plot_z_date = []

        S = []

        self.min_bars = lookback


        n = data[S1]['open'].shape[0]
        n_dim_state = 2

        state_means = np.zeros((n, n_dim_state))
        state_covs = np.zeros((n, n_dim_state, n_dim_state))

        for bar in range(lookback, data[S1].shape[0]):
            if bar == lookback:
                S1_log_prices = np.log(data[S1]['open'].iloc[bar-lookback: bar])
                S2_log_prices = np.log(data[S2]['open'].iloc[bar-lookback: bar])

                # score, pvalue, _ = coint(S1_log_prices, S2_log_prices)
                # plot_pvalue.append(1- pvalue)


                obs_mat = add_constant(S2_log_prices.values, prepend=False)[:, np.newaxis]
                print(len(obs_mat))
                obs_cov = np.cov(S1_log_prices)

                mu1, mu2 = np.var(S1_log_prices), np.var(S2_log_prices)

                trans_cov = np.array([[mu1, 0], [0, mu2]])
                # delta = 1e-4 #To estimate
                # trans_cov = delta / (1 - delta) * np.eye(2)
                
                kf = KalmanFilter(n_dim_obs=1, n_dim_state=2, # y is 1-dimensional, (alpha, beta) is 2-dimensional
                            initial_state_mean=np.ones(2),
                            initial_state_covariance=np.ones((2, 2)),
                            transition_matrices=np.eye(2),
                            observation_matrices=obs_mat,
                            observation_covariance=obs_cov,#1e-4,#10**2,
                            transition_covariance=trans_cov)

                state_means[0: bar], state_covs[0: bar] = kf.filter(S1_log_prices)
                gamma = state_means[bar, 0]
                mu = state_means[bar, 1]
            else:
                new_bar_S1 = np.log(data[S1]['open'].iloc[bar])
                new_bar_S2 = np.log(data[S2]['open'].iloc[bar])
                obs_mat = np.asarray([[new_bar_S2, 1]])

                state_means[bar], state_covs[bar] = kf.filter_update(state_means[bar-1], 
                                                        state_covs[bar-1], 
                                                        observation = new_bar_S1, 
                                                        observation_matrix=obs_mat)
                
                gamma = state_means[bar, 0]
                mu = state_means[bar, 1]

    
            S1_log_price = np.log(data[S1]['open'].iloc[bar])
            S2_log_price = np.log(data[S2]['open'].iloc[bar])

            spread = S1_log_prices[-25:] - (state_means[bar-25:bar, 0] * S2_log_prices[-25:]) - state_means[bar-25:bar, 0]
            spread_t = S1_log_price - (gamma * S2_log_price) - mu

            if bar == lookback:
                spread_mean = spread.mean()
                spread_std = spread.std()
                S = list(spread)
            S.append(spread_t)


            spread_mean = pd.Series(data=S).mean() 
            # spread_std = pd.Series(data=S).std()

            # spread_mean = spread.mean()
            spread_std = spread.std() #* 0.05


            # z_score_t = (spread_t - spread_mean) / spread_std
            z_score_t = spread_t

            # threshold = 0.7
            # threshold = 1.7
            # threshold = 0.7
            # threshold = 2.2
            # threshold = 0.5
            threshold = 0.002
            # threshold = 0.00076 #daily for gld 
            # threshold = 40e-6 #10min tf for gld
            # threshold = 0.002 #daily tf for EWH
            # threshold = 50e-6 #10min tf
            # threshold = spread_std

            plot_z_date.append(self.get_date_price(S1, bar)[0])
            plot_z_score.append(z_score_t)
            plot_gamma.append(gamma)
            plot_mean.append(spread_mean)
            plot_std.append(spread_std)


            if S1_position == 0:
                if z_score_t < -threshold:
                    # self.place_buy_order(S1, bar, units=50)
                    self.place_buy_order(S1, bar, amount=self.net_wealth)
                    # self.place_buy_order(S2, bar, units=1)
                    S1_position = 1
                    if S2_position == 1:
                        # self.place_sell_order(S2, bar, units=50)
                        # self.place_sell_order(S2, bar, amount=self.net_wealth)
                        S2_position = 0
            
            if S1_position == 1:
                if z_score_t > threshold:
                    # self.place_sell_order(S1, bar, units=50)
                    self.place_sell_order(S1, bar, amount=self.net_wealth)
                    # self.place_sell_order(S2, bar, units=1)
                    S1_position = 0
                    if S2_position == 0:
                        # self.place_buy_order(S2, bar, units=50)
                        # self.place_buy_order(S2, bar,amount=self.net_wealth)
                        S2_position = 1

        print(state_means[100])
        
        plot_z_score = pd.Series(data=plot_z_score, index=plot_z_date, name='z-score')
        # plot_pvalue = pd.Series(data=plot_pvalue, index=plot_z_date, name='p_value')
        plot_gamma = pd.Series(data=plot_gamma, index=plot_z_date, name='gamma')
        plot_mean = pd.Series(data=plot_mean, index=plot_z_date, name='mean')
        plot_std_up = pd.Series(data=plot_std, index=plot_z_date, name='std')
        plot_std_low = - pd.Series(data=plot_std, index=plot_z_date, name='std')

        # self.add_plot_separate(plot_z_score, plot_std_up, plot_std_low)#, plot_mean, plot_std_up, plot_std_low)
        self.add_plot_separate(plot_gamma)
        self.close_out(bar)

    

if __name__ == '__main__':
    # bt = Pairs_Mean_Reversion_OLS(['pep', 'ko'], 'daily', '2020-01-01', '2022-01-01', 10000, verbose=False)
    # bt = Pairs_Mean_Reversion_OLS(['amd', 'tsla'], '10min', '2022-02-14', '2022-02-28', 10000, verbose=False)
    # bt = Pairs_Mean_Reversion_OLS(['gdx', 'gld'], 'daily', '2020-10-01', '2022-04-28', 10000, verbose=False)
    bt = Pairs_Mean_Reversion_OLS(['paa', 'pagp'], 'daily', '2020-10-01', '2022-04-28', 10000, verbose=False)
    # bt = Pairs_Mean_Reversion_OLS(['gdx', 'gld'], '10min', '2022-01-01', '2022-06-20', 10000, verbose=False)
    # bt = Pairs_Mean_Reversion_OLS(['EWH', 'EWZ'], 'daily', '2000-08-01', '2002-01-31', 10000, verbose=False)
    bt.run_mr_strategy()
    print(bt.stats)
    # print(bt.data)
    # print(bt.statistics())
    bt.plot()
    # bt.scatter_plot()