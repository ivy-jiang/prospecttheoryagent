import pandas as pd
import yfinance as yf
import pandas_datareader.data as web
from datetime import datetime, timedelta
import os

def generate_full_history_csv():
    print("--- Starting Full History Data Fetch ---")
    
    # 1. Define Timeframe (Last 5 Years)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365) 
    
    print(f"1. Fetching raw data from {start_date.date()} to {end_date.date()}...")
    
    tickers = ["QQQ", "^VIX", "^TNX"]
    raw_data = yf.download(tickers, start=start_date, end=end_date, progress=False)
    
    # Handle yfinance data structure
    try:
        df = raw_data['Close'].copy()
    except KeyError:
        df = raw_data['Adj Close'].copy()

    # 2. Fetch 2-Year Treasury Yield
    print("2. Fetching 2-Year Treasury Yields...")
    try:
        us_2y = web.DataReader('DGS2', 'fred', start_date, end_date)
        df = df.join(us_2y)
        df.rename(columns={'DGS2': 'US_2y_yield'}, inplace=True)
    except Exception as e:
        print(f"   (FRED source unavailable, using Yahoo backup for Yields)")
        fvx = yf.download("^FVX", start=start_date, end=end_date, progress=False)['Close']
        df = df.join(fvx)
        df.rename(columns={'^FVX': 'US_2y_yield'}, inplace=True)

    # Fill missing weekends/holidays
    df = df.ffill().dropna()

    # 3. Calculate Indicators
    print("3. Calculating Technical Indicators...")
    
    df['price'] = df['QQQ']
    df['US_10y_yield'] = df['^TNX']
    df['market_sentiment'] = df['^VIX']
    
    # 100-Day Moving Average
    df['100D_MA'] = df['price'].rolling(window=100).mean()
    
    # RSI (14-Day)
    delta = df['price'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, min_periods=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD (12, 26, 9)
    ema12 = df['price'].ewm(span=12, adjust=False).mean()
    ema26 = df['price'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26

    # 4. Export
    final_cols = ['price', 'RSI', '100D_MA', 'MACD', 'US_10y_yield', 'US_2y_yield', 'market_sentiment']
    final_df = df[final_cols].dropna()
    
    filename = 'qqq_market_data_with_dates.csv'
    final_df.to_csv(filename, index=True)
    
    print(f"\nSUCCESS: Generated '{filename}' with {len(final_df)} rows.")

if __name__ == "__main__":
    generate_full_history_csv()
