import numpy as np
import pandas as pd
from backtest_logic.backtester import Backtest



class SMA_Crossover(Backtest):
    def run_sma_strategy(self, SMA1: int, SMA2: int):
        ''' Backtesting an SMA-based strategy.

        Parameters
        ==========
        SMA1, SMA2: int
        shorter and longer term simple moving average (with the given timeframe)
        '''
        msg = f'\n\nRunning SMA crossover strategy | SMA1={SMA1} & SMA2={SMA2}'
        msg += f'\nfixed costs {self.ftc} | '
        msg += f'proportional costs {self.ptc}'
        print(msg)
        print('=' * 55)
        self.position = 0 # initial neutral position
        self.trades = 0 # no trades yet
        self.amount = self.initial_amount # reset initial capital
        self.min_bars = SMA2 #minimum number of bars needed

        symbol = self.symbol[0].upper()

        data = self.data[symbol]

        plot_SMA1, plot_SMA2 = [], []
        plot_date = []

        for bar in range(SMA2, data.shape[0]):
            short_ma = data['open'].iloc[bar-SMA1: bar].mean()
            long_ma = data['open'].iloc[bar-SMA2: bar].mean()
            
            plot_SMA1.append(short_ma)
            plot_SMA2.append(long_ma)
            plot_date.append(self.get_date_price(symbol, bar)[0])
            

            if self.position == 0:
                if short_ma >= long_ma:
                    self.place_buy_order(symbol, bar, amount=self.net_wealth)
                    self.position = 1

            if self.position == 1:
                if short_ma < long_ma:
                    self.place_sell_order(symbol, bar, amount=self.net_wealth)
                    self.position = 0

        plot_SMA1 = pd.Series(data=plot_SMA1, index=plot_date, name=f'SMA{SMA1}')
        plot_SMA2 = pd.Series(data=plot_SMA2, index=plot_date, name=f'SMA{SMA2}')

        self.add_plot(plot_SMA1, plot_SMA2)
        self.close_out(bar)

if __name__ == '__main__':
    bt = SMA_Crossover('msft', '10min', '2022-02-14', '2022-02-28', 10000, verbose=False)
    bt.run_sma_strategy(10, 25)
    print(bt._statistics())
    bt.plot()