import os
import csv
import pandas as pd
import requests

from dateutil import parser
from json import JSONDecodeError
from pandas.errors import EmptyDataError

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

