# backend/data_fetching.py
import requests
import yfinance as yf
import pandas as pd
import streamlit as st

def fetch_quotes(symbols, api_key):
    url = "https://api.12data.com/quote"
    params = {
        "symbol": ",".join(symbols),
        "apikey": api_key
    }
    response = requests.get(url, params=params)
    data = response.json()
    return data

def get_latest_non_nan_value(series):
    """
    Get the most recent non-NaN value from a pandas Series.
    If the last value is not NaN, return it. Otherwise, return the most recent non-NaN value.
    """
    if series.empty:
        return None
    
    non_nan_values = series.dropna()
    
    if non_nan_values.empty:
        return None
    
    return non_nan_values.iloc[-1]

@st.cache_data(show_spinner=False)
def get_company_name(ticker):
    try:
        info = yf.Ticker(ticker).info
        return info.get('shortName', ticker)
    except Exception:
        return ticker

@st.cache_data(show_spinner=False)
def get_top_stocks_quotes(tickers):
    """
    Fetches the latest quote data for stocks using Yahoo Finance (yfinance),
    including company name, change, and percent change. Uses caching for efficiency.
    Handles multiple days of data by merging rows and taking the most recent non-NaN values.
    Returns a DataFrame with columns: ticker, name, last_price, day_high, day_low, open, volume, change, change_pct.
    """
    
    data = yf.download(tickers, period="2d", interval="1d", group_by='ticker', auto_adjust=True, threads=True)
    
    quotes = []
    successful_tickers = []
    failed_tickers = []
    
    for ticker in tickers:
        try:
            if len(tickers) == 1:
                ticker_data = data
            else:
                if ticker not in data.columns:
                    raise ValueError(f"Ticker {ticker} not found in data columns")
                ticker_data = data[ticker]
            
            last_close = get_latest_non_nan_value(ticker_data['Close'])
            high = get_latest_non_nan_value(ticker_data['High'])
            low = get_latest_non_nan_value(ticker_data['Low'])
            open_ = get_latest_non_nan_value(ticker_data['Open'])
            volume = get_latest_non_nan_value(ticker_data['Volume'])
            
            
            if last_close is None or pd.isna(last_close):
                raise ValueError(f"No valid close price for {ticker}")
            
            # Get company name
            name = get_company_name(ticker)
            
            change = last_close - open_ if open_ is not None else None
            change_pct = None
            if change is not None and open_ is not None and open_ != 0:
                change_pct = (change / open_) * 100
            
            quotes.append({
                'ticker': ticker,
                'name': name,
                'last_price': float(last_close),
                'day_high': float(high) if high is not None else None,
                'day_low': float(low) if low is not None else None,
                'open': float(open_) if open_ is not None else None,
                'volume': int(volume) if volume is not None else None,
                'change': float(change) if change is not None else None,
                'change_pct': float(change_pct) if change_pct is not None else None,
                'currency': 'USD',
                'bid': None,
                'ask': None,
                'year_high': None,
                'year_low': None,
                'market_cap': None
            })
            
            successful_tickers.append(ticker)
            
        except Exception as e:
            print(e)
            failed_tickers.append(ticker)
            
            quotes.append({
                'ticker': ticker,
                'name': ticker,
                'last_price': None,
                'day_high': None,
                'day_low': None,
                'open': None,
                'volume': None,
                'change': None,
                'change_pct': None,
                'currency': None,
                'bid': None,
                'ask': None,
                'year_high': None,
                'year_low': None,
                'market_cap': None
            })
    
    return pd.DataFrame(quotes)