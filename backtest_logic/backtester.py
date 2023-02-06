import csv
import os

from dateutil import parser
from json import JSONDecodeError
import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError
import plotly.graph_objects as go
import requests

from ._plotting import plot_strategy, plot_scatter
from ._stats import performance
from ._TIINGO_TOKEN import TOKEN ###Add a file in the same directory as backtester.py named as TIINGO_TOKEN


script_dir = os.path.dirname(os.path.abspath(__file__))


class Data():

    @staticmethod
    def _request_data(ticker: str, freq: str, start: str, end: str) -> pd.DataFrame:
        ''' 
        Historical Intraday Prices Endpoint from IEX:

        To request historical intraday prices for a stock, use the following REST endpoint.

        Parametres
        ==========
        ticker: str
            unique ticker symbol
            
        freq: 'str'
            The frequency of the data can be in the following format.
                min: str
                    ex: '1min', '10min', '300min', ..., '{n}min' (n: int)
                    n < 2563
                hour: str
                    ex: '1hour', '10hour', '300hour', ..., '{n}hour' (n: int) 
                    n < 24
                daily: str
                    ex: 'daily' *no more than daily timeframe can be sampled
        '''
        ticker = ticker.upper()
        rel_path = f"raw_market_data/{ticker}_{start}_{end}.csv"
        data_file_path = os.path.join(script_dir, rel_path)

        if freq == 'daily':
            requestResponse = requests.get(f"https://api.tiingo.com/tiingo/daily/{ticker}/prices?startDate={start}&endDate={end}&token={TOKEN}", headers={
                'Content-Type': 'application/json'
            })
        else:
            requestResponse = requests.get(f"https://api.tiingo.com/iex/{ticker}/prices?startDate={start}&endDate={end}&resampleFreq={freq}&columns=open,high,low,close,volume&token={TOKEN}", headers={
                'Content-Type': 'application/json'
            })
        
        #Verification 1: check input timeframe unit
        try:
            stock_data = requestResponse.json()
        except JSONDecodeError:
            raise Exception("Change the timeframe: only 'min', 'hour', 'daily' are supported ")
        
        #Verification 2: check input timeframe size 
        if len(stock_data) == 0:
            if int(freq.split('h')[0]) >= 24:
                raise Exception('hours must be under 24')
            elif int(freq.split('m')[0]) >= 2563:
                raise Exception('minutes must be under 2563')
            else:
                raise Exception("Change the timeframe: only 'min', 'hour', 'daily' are supported ")
        
        #Verification 3: verify the ticker's existence 
        if len(stock_data) == 1:
            if stock_data['detail'] == 'Not found.':
                raise Exception(f'Unable to find the symbol {ticker}')
            elif 'Error: resampleFreq' in stock_data['detail']:
                raise Exception('resampleFreq format was not correct. It must be an int followed by Min or Hour, e.g. 5Min')

        #convert the data into csv format
        with open(data_file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            count = 0
            for dt in stock_data:
                if count == 0:
                    header = dt.keys()
                    writer.writerow(header)
                    count += 1
                writer.writerow(dt.values())
        
        raw = pd.read_csv(data_file_path)
       
        return raw

    @staticmethod
    def raw_data(ticker: str, freq: str, start: str, end:str) -> pd.DataFrame:
        """
        Returns a pandas.DataFrame() of the OHLCV market Data of a ticker.
        
        Parametres
        ==========
        ticker: str
            unique ticker symbol
        
        freq: str
            The frequency of the data can be in the following format.
                min: str
                    ex: '1min', '10min', '300min', ..., '{n}min' (n: int)
                    n < 2563
                hour: str
                    ex: '1hour', '10hour', '300hour', ..., '{n}hour' (n: int) 
                    n < 24
                daily: str
                    ex: 'daily' *no more than daily timeframe can be sampled
        """
        ticker = ticker.upper()
        rel_path = f"raw_market_data/{ticker}_{start}_{end}.csv"
        data_file_path = os.path.join(script_dir, rel_path)

        def get_date(dt) -> str: 
            ''' returns only the date of a ISO-8601 date representation
            '''
            return dt.split('T')[0]

        def find_date_iloc(df, dt) -> int:
            ''' returns the index value of a specific date
            '''
            return df[df==dt].index.values
        
        def to_seconds(t: str) -> int:
            ''' supports only {}min and {}hour
            '''
            if 'min' in t:
                t = int(t.split('m')[0]) * 60
            elif 'hour' in t:
                t = int(t.split('h')[0]) * 60 * 60
            elif 'day' in t:
                t = int(t.split('d')[0]) * 60 * 60 * 24
            return t
        
        #verify if the ticker file exists in the directory
        try: 
            raw = pd.read_csv(data_file_path)
            try:
                #try slicing if the requested start and end are within the dataset
                dates = raw['date'].apply(get_date)
                i_start = find_date_iloc(dates, start)[0]
                i_end = find_date_iloc(dates, end)[-1]
                
                raw = raw[i_start:i_end]

                #verify the frequency of the data
                dt1, dt2 = parser.parse(raw['date'].iloc[0]), parser.parse(raw['date'].iloc[1])

                time_diff = (dt2 - dt1).total_seconds()
                if time_diff != to_seconds(freq):
                    Data._request_data(ticker, freq, start, end)
                    raw = pd.read_csv(data_file_path)
 
            #if the start and end are the same
            except IndexError:
                Data._request_data(ticker, freq, start, end)
                raw = pd.read_csv(data_file_path)

        #If the file exist but is empty or if the file doesn't exist
        except (EmptyDataError, FileNotFoundError) as _: 
            Data._request_data(ticker, freq, start, end)
            raw = pd.read_csv(data_file_path)

        return raw


class Backtest(object):
    def __init__(self, symbol: str or list, freq: str, start: str, end: str, amount: float, 
                currency="CAD", ftc=0.0, ptc=0.0, 
                fractions=True, verbose=True, same_timeframe=True, min_bar=0):
        ''' Base class event-based backtesting of trading strategies
        
        Attributes
        ==========
        symbol: str
            TR RIC (financial instrument) to be used
        start: str
            start date for data selection
        end: str
            end date for data selection
        amount: float
            amount to be invested either once or per trade
        currency: str
            currency used for transactions
        ftc: float
            fixed transaction costs per trade (buy or sell)
        ptc: float
            proportional transaction costs per trade (buy or sell)
        
        DataFrames
        ==========
        tradebook: list -> pd.DataFrame (see close_out methode)
            contains trade information:
            'date': str
                date that a trade was executed
            'order-type': 
                the type of order executed
            'symbol': str
                the security that was bought
            'units': float
                the number of units bought
            'price': float
                the price that the security was bought at  

        pnl: list -> pd.DataFrame (see close_out methode)
                contains all the profit or loss from each trade
            'date': str
                date of the realized return
            'pnl': float
                return


        Methods
        =======
        get_data:
            retrieves and prepares the base data set
        plot_data:
            plots the closing price for the symbol
        get_date_price:
            returns the date and price for the given bar
        print_balance:
            prints out the current (cash) balance
        print_net_wealth:
            prints out the current net wealth
        place_buy_order:
            places a buy order
        place_sell_order:
            places a sell order
        close_out:
            closes out a long or short position
            need to be executed to analyse the strategy's performance.
        '''

        if type(symbol) is str:
            self.symbol = [symbol.upper()]
        elif type(symbol) is list:
            self.symbol = [sym.upper() for sym in symbol]
    
        self.freq = freq
        self.start = start
        self.end = end
        self.initial_amount = amount
        self.currency = currency
        self.ftc = ftc
        self.ptc = ptc
        self.current_units = {sym: 0 for sym in self.symbol}
        self.net_wealth = amount
        self.fraction = fractions
        self.amount = amount
        self.position = 0
        self.trades = 0
        self.orderbook = []
        self.tradebook = []
        self.pnl = []
        self._add_plot = []
        self._add_plot_separate = []
        self.verbose = verbose
        self.same_timeframe  = same_timeframe
        self.min_bars = min_bar

        self.get_data()
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}'
    
    def get_data(self):
        ''' Retrieve OHLCV data and filter through it
        
        Attributes
        ==========
        data: dict -> pd.DataFrame
            index: symbol\n
            The historical OHLCV data (see class Data())
        _prices: dict -> pd.Series
            Timeseries of the prices. Used to plot the equity
        units: dict -> pd.Series
            Timeseries of the prices. Used to plot the equity
        balance: pd.DataFrame
            Timeseries of the current balance information. Used to plot the equity
        '''
        raw, prices, units = {}, {}, {}

        #loop through each symbol requested to fetch their respective market data and to create a empty tradebook
        for sym in self.symbol:
            raw[sym] = Data.raw_data(sym, self.freq, self.start, self.end)

        #remove all non trading hours
        # for sym in self.symbol:
        #     raw[sym] = raw[sym][raw[sym]['volume'] != 0].reset_index(drop=True)

        if self.same_timeframe:         #remove any differente dates from each dataset
            compare_date = [pd.Index(raw[sym]['date']) for sym in self.symbol]
            differente_date = []

            for i in range(len(self.symbol)):
                for n in range(len(self.symbol)):
                    diff = compare_date[i].difference(compare_date[n])
                    for d in diff:
                        if d not in differente_date:
                            differente_date.append(d)
            
            for diffs in differente_date:
                for sym in self.symbol:
                    raw[sym] = raw[sym][raw[sym]["date"].str.contains(diffs) == False].reset_index(drop=True)

            if bool(differente_date):
                print('Removing the following dates:')
                print(differente_date)

        for sym in self.symbol:
            prices[sym] = pd.Series(
                data=list(raw[sym]['close']), 
                index=raw[sym]['date']
            )
            units[sym] = pd.Series(
                data=np.nan, 
                index=raw[sym]['date']
            )
            units[sym].iloc[0] = 0      #initialize 0 units
    
        balance = pd.Series(
                        data=np.nan, 
                        index=raw[sym]['date']
        )
        balance.iloc[0] = self.initial_amount

        #initialize
        self.data = raw
        self.units = units
        self._prices = prices
        self.balance = balance


    def get_date_price(self, symbol: str, bar: int) -> tuple:
        ''' Return date and price for bar.
        '''
        symbol = symbol.upper()
        date = self.data[symbol].loc[bar, 'date']
        price = self.data[symbol].loc[bar, 'open']

        return date, price  

    def get_date_buy_price(self, symbol: str, bar: int) -> tuple:
        ''' Return date and buying price for bar.
        Set at the highest ask
        '''
        symbol = symbol.upper()
        date = self.data[symbol].loc[bar, 'date']
        price = self.data[symbol].loc[bar, 'high']
        price = self.data[symbol].loc[bar, 'open']

        return date, price
    
    def get_date_sell_price(self, symbol: str, bar: int) -> tuple:
        ''' Return date and price for bar.
        Set at the lowest bid
        '''
        symbol = symbol.upper()
        date = self.data[symbol].loc[bar, 'date']
        price = self.data[symbol].loc[bar, 'low']
        price = self.data[symbol].loc[bar, 'open']

        return date, price    

    def print_balance(self, bar):
        ''' Print out current cash balance info.
        '''
        first_symbol = list(self.data.keys())[0]
        date, _ = self.get_date_price(first_symbol, bar)
        print(f'{date} | current balance {self.amount:.2f}')  ####ADD currency
    
    def get_net_wealth(self, bar):
        ''' Print out current cash balance info.
        ''' 
        P, U = [], []
        for sym in self.symbol:
            _, price = self.get_date_price(sym, bar)
            P.append(price)
            units = self.units[sym].iloc[bar]
            U.append(units)
        P = np.array([P])
        U = np.array([U])
        U[np.isnan(U)] = 0

        self.net_wealth = float(np.dot(U, P.T)[0] + self.amount)

    def place_buy_order(self, symbol, bar, units=None, amount=None):
        ''' Place a buy order.
        '''
        symbol = symbol.upper()
        date, price = self.get_date_buy_price(symbol, bar)
        if units is None:
            if self.fraction is True:
                units = (amount - 1) / price
                units = np.round(units, 4)
            else:
                units = int((amount - 1) / price)
        
        self.orderbook.append({
            'date':date,
            'order-type':'BUY',
            'symbol': symbol,
            'units':units,
            'price':price,
        })

        self.amount -= (units * price) * (1 + self.ptc) + self.ftc
        if self.amount < 0:
            raise Exception(f'Balance is negative {self.amount}')

        self.current_units[symbol] += units
        self.units[symbol].iloc[bar] = self.current_units[symbol]
        self._prices[symbol].iloc[bar] = price
        self.balance.iloc[bar] = self.amount
        self.trades += 1

        self.get_net_wealth(bar)

        if self.verbose:
            print(f'{date} | buying {units} units of {symbol} at {price:.2f}')
            self.print_balance(bar)
            print(f'{date} | current net wealth {self.net_wealth:.2f}')
      
    def place_sell_order(self, symbol, bar, units=None, amount=None):
        ''' Place a sell order.
        '''
        symbol = symbol.upper()
        date, price = self.get_date_sell_price(symbol, bar)

        if units is None:
            if self.fraction is True:
                if amount == self.net_wealth:
                    units = self.current_units[symbol]
                else:
                    units = amount / price
                    units = np.round(units, 4)
            else:
                units = int(amount / price)

        self.orderbook.append({
            'date':date,
            'order-type':'SELL',
            'symbol': symbol,
            'units':units,
            'price':price,
        })

        if units > self.current_units[symbol]:
            raise Exception(f'Attempted to sell {units} units of {symbol} when you have currently {self.current_units[symbol]} units of {symbol}')


        past_trades = 0
        remainder_units = units
        for trade in reversed(self.orderbook):
            if trade['order-type'] == 'BUY':
                if trade['symbol'] == symbol:
                    if remainder_units <= trade['units']:
                        past_trades += trade['price'] * remainder_units
                        self.tradebook.append({
                            'entry-date':trade['date'],
                            'exit-date':date,
                            'units':remainder_units,
                            'entry-price':trade['price'],
                            'exit-price':price,
                        })
                        break
                    else:
                        past_trades += trade['price'] * trade['units']
                        remainder_units -= trade['units']
                        self.tradebook.append({
                            'entry-date':trade['date'],
                            'exit-date':date,
                            'units':trade['units'],
                            'entry-price':trade['price'],
                            'exit-price':price,
                        })
                        continue
        
        if past_trades == 0:
            pnl = 0
        else:
            pnl = (price*units / past_trades) - 1
            gain = price*units - past_trades

        self.pnl.append({
            'date':date, 
            'pnl':pnl,
            'gain':gain
        })
        
        self.amount += (units * price) * (1 - self.ptc) - self.ftc
        self.current_units[symbol] -= units
        self.units[symbol].iloc[bar] = self.current_units[symbol]
        self._prices[symbol].iloc[bar] = price
        self.balance.iloc[bar] = self.amount
        self.trades += 1
        self.get_net_wealth(bar)

        if self.verbose:
            print(f'{date} | selling {units} units of {symbol} at {price:.2f}')
            self.print_balance(bar)
            print(f'{date} | current net wealth {self.net_wealth:.2f}')


    def close_out(self, bar):
        ''' Closing out positions. Needs to be executed before calculating the statistics
        
        Attributes
        ==========
        data: dict -> pd.DataFrame
            index: symbol\n
            The historical OHLCV data (see class Data())\n
            contains (T - min_num_bars) rows
        _prices: dict -> pd.Series
            Timeseries of the prices. Used to plot the equity\n
            contains (T - min_num_bars) rows
        units: dict -> pd.Series
            Timeseries of the prices. Used to plot the equity\n
            contains (T - min_num_bars) rows
        balance: pd.DataFrame
            Timeseries of the current balance information. Used to plot the equity\n
            contains (T - min_num_bars) rows
        pnl: pd.DataFrame
            profit resoluting from each trades (from buying to selling)
        '''

        for sym in self.symbol:
            if self.current_units[sym] == 0:
                continue
            else:
                self.place_sell_order(sym, bar, units=self.current_units[sym])

        for sym in self.symbol:
            self.units[sym] = self.units[sym].fillna(method='ffill')
            
        self.balance = self.balance.fillna(method='ffill')
        
        holding = []
        for bar in range(self.data[sym]['date'].shape[0]):
            p = 0
            for sym in self.symbol:
                units = self.units[sym].iloc[bar]
                price = self._prices[sym].iloc[bar]
                p += units * price
            holding.append(p)
        holding = pd.Series(data=holding, index=self.data[sym]['date'])

        self.equity = holding + self.balance
        self.pnl = pd.DataFrame(self.pnl)
        self.tradebook = pd.DataFrame(self.tradebook)

        try:
            self.tradebook['return'] = self.tradebook['exit-price'] / self.tradebook['entry-price'] - 1
            self.tradebook['duration'] = self.tradebook['exit-date'].apply(parser.parse) - self.tradebook['entry-date'].apply(parser.parse)
        except KeyError:
            self.tradebook['return'] = np.nan
            self.tradebook['duration'] = np.nan
        
        self.orderbook = pd.DataFrame(self.orderbook)

        self._statistics()  #compute statistics

    def _statistics(self) -> pd.Series:
        ''' Calculates the stastistics used to measure performance
        '''
        st = performance(
            name=self,
            symbol=self.symbol, 
            data=self.data,
            equity=self.equity,
            tradebook=self.tradebook,
            freq=self.freq,
            min_bars=self.min_bars,
        )
        self.stats = st

        return st

    def add_plot(self, *plot: pd.Series):
        ''' Additional serie(s) that can be graphed over the price function.
        '''
        for p in plot:
            self._add_plot.append(p)

    def add_plot_separate(self, *plot: pd.Series):
        ''' Additional serie(s) that can be graphed separated from the price function.
        '''
        for p in plot:
            self._add_plot_separate.append(p)

    def plot(self):
        ''' Graphs an interface to view how to strategy performed
        '''
        fig = plot_strategy(
            symbol=self.symbol, 
            data=self.data,
            equity=self.equity, 
            balance=self.balance,
            pnl=self.pnl, 
            same_timeframe=self.same_timeframe, 
            orderbook=self.orderbook,
            freq=self.freq,
            stats=self.stats,
            add_plot=self._add_plot,
            add_plot_separate=self._add_plot_separate,
            min_bars=self.min_bars,
        )

        fig.show()
    
    def _scatter_plot(self):
        fig = plot_scatter(symbol=self.symbol, data=self.data)

        fig.show()