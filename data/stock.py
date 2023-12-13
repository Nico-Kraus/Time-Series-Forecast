import yfinance as yf
import pandas as pd
import os
from pandas.tseries.offsets import BDay
from pandas.tseries.offsets import Minute, Hour, Day, Week, MonthEnd

def fetch_stock_data(symbol = "AAPL", data_type = "Close", interval = "1d", start_date = "2019-01-01", size = 1000, **params):
    """
    Fetch stock data from Yahoo Finance.

    :param symbol: The stock symbol (e.g., 'AAPL' for Apple).
    :param data_type: The type of data ('close', 'high', etc.).
    :param interval: The data interval ('1d' for daily, '1m' for minute, etc.).
    :param start_date: The starting date for the data.
    :param size: The number of business days to fetch.
    :return: Numpy array of requested stock data.
    """
    path = "data/yfinance_data/"
    filename = f"{path}{symbol}_{data_type}_{interval}_{start_date}_{size}.csv"
    if not os.path.exists(path):
        os.makedirs(path)

    # Check if data already exists in the file
    if os.path.exists(filename):
        df = pd.read_csv(filename, index_col=0)
        if len(df) >= size:
            return df[data_type].to_numpy()[:size]

    total_days = int(size * 1.1) # 10% extra days
    end_date = calculate_end_date(start_date, total_days, interval)

    df = yf.download(symbol, start=start_date, end=end_date, interval=interval)
    if data_type not in df.columns:
        raise ValueError(f"Data type '{data_type}' not found in fetched data.")
    data = df[data_type]
    df.to_csv(filename)

    return data.to_numpy()[:size]


def calculate_end_date(start_date, size, interval):
    """
    Calculate the end date for a given interval type.
    
    :param start_date: The starting date for the data.
    :param size: The number of data points to fetch.
    :param interval: The data interval (e.g., '1d', '1m', etc.).
    :return: The end date as a string in YYYY-MM-DD format.
    """
    start = pd.to_datetime(start_date)
    
    if interval.endswith('m'):
        return (start + BDay(5)).strftime('%Y-%m-%d')
    elif interval.endswith('h'):
        hours = int(interval[:-1]) * size
        days = hours//4
        return (start + BDay(days)).strftime('%Y-%m-%d')
    elif interval == '1d':
        return (start + BDay(size)).strftime('%Y-%m-%d')
    elif interval == '5d':
        return (start + BDay(5*size)).strftime('%Y-%m-%d')
    elif interval == '1wk':
        return (start + Week(size)).strftime('%Y-%m-%d')
    elif interval == '1mo':
        return (start + MonthEnd(size)).strftime('%Y-%m-%d')
    elif interval == '3mo':
        return (start + MonthEnd(3 * size)).strftime('%Y-%m-%d')
    else:
        raise ValueError(f"Interval '{interval}' not recognized.")

    return end_date