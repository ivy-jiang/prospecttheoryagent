import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

def analyze_stats():
    # 1. Load Data
    print("Loading qqq_market_data.csv...")
    df = pd.read_csv('qqq_market_data.csv')
    
    # Check length
    n_rows = len(df)
    print(f"Data rows: {n_rows}")
    
    # 2. Reconstruct Dates
    # Best match start date was 2015-11-27
    start_date = "2015-11-27"
    print(f"Fetching QQQ dates starting from {start_date}...")
    
    # Fetch enough data
    dates_data = yf.download("QQQ", start=start_date, progress=False)
    
    # Align lengths
    # The CSV might be a subset or have slight differences.
    # We'll assume the CSV corresponds to the first N trading days from start_date.
    # Or if the CSV ends recently, we can align from the end?
    # Let's assume start alignment for now.
    
    if len(dates_data) < n_rows:
        print(f"Warning: Fetched data ({len(dates_data)}) is shorter than CSV ({n_rows}).")
        # Try fetching from earlier just in case
        dates_data = yf.download("QQQ", start="2015-01-01", progress=False)
        # Find the index of the start date
        try:
            start_idx = dates_data.index.get_loc(pd.Timestamp(start_date))
            dates_data = dates_data.iloc[start_idx:]
        except:
            pass

    # Take the first n_rows dates
    # Handle MultiIndex in new yfinance
    if isinstance(dates_data.columns, pd.MultiIndex):
        dates = dates_data.index
    else:
        dates = dates_data.index
        
    if len(dates) > n_rows:
        dates = dates[:n_rows]
    elif len(dates) < n_rows:
        print("Warning: Not enough dates fetched. Padding with dummy dates.")
        # This shouldn't happen if we fetched enough
        
    print(f"Aligned dates: {dates[0].date()} to {dates[-1].date()}")
    
    # Create a mapping dataframe
    df['Date'] = dates
    df['Year'] = df['Date'].dt.year
    
    # Backtest Window Offset
    # The backtest results start after window_size (20)
    # Step 0 in backtest -> Index 20 in df
    window_size = 20
    
    # 3. Analyze Daily Results
    print("\n" + "="*40)
    print("DAILY STRATEGY BUY % BY YEAR")
    print("="*40)
    try:
        daily_results = pd.read_csv('full_backtest_results.csv')
        
        # Map steps to dates
        # daily_results['step'] i corresponds to df index i + window_size
        
        # Add Date column to results
        # We need to be careful with indices
        result_dates = []
        for step in daily_results['step']:
            idx = step + window_size
            if idx < len(df):
                result_dates.append(df['Date'].iloc[idx])
            else:
                result_dates.append(None)
                
        daily_results['Date'] = result_dates
        daily_results = daily_results.dropna(subset=['Date'])
        daily_results['Year'] = daily_results['Date'].dt.year
        
        # Calculate Stats
        daily_stats = daily_results.groupby('Year')['action_name'].value_counts(normalize=True).unstack().fillna(0)
        if 'BUY' in daily_stats.columns:
            daily_stats['BUY_Pct'] = (daily_stats['BUY'] * 100).round(1)
            print(daily_stats[['BUY_Pct']])
            print(f"\nOverall Daily Buy %: {(daily_results['action_name'] == 'BUY').mean() * 100:.1f}%")
        else:
            print("No BUY signals found.")
            
    except FileNotFoundError:
        print("full_backtest_results.csv not found.")
        
    # 4. Analyze Weekly Results
    print("\n" + "="*40)
    print("WEEKLY STRATEGY BUY % BY YEAR")
    print("="*40)
    try:
        weekly_results = pd.read_csv('weekly_full_backtest.csv')
        
        # Map entry_day to dates
        # entry_day is the index in the original df (which we aligned with dates)
        
        w_dates = []
        for day in weekly_results['entry_day']:
            if day < len(df):
                w_dates.append(df['Date'].iloc[day])
            else:
                w_dates.append(None)
                
        weekly_results['Date'] = w_dates
        weekly_results = weekly_results.dropna(subset=['Date'])
        weekly_results['Year'] = weekly_results['Date'].dt.year
        
        # Calculate Stats
        weekly_stats = weekly_results.groupby('Year')['action'].value_counts(normalize=True).unstack().fillna(0)
        if 'BUY' in weekly_stats.columns:
            weekly_stats['BUY_Pct'] = (weekly_stats['BUY'] * 100).round(1)
            print(weekly_stats[['BUY_Pct']])
            print(f"\nOverall Weekly Buy %: {(weekly_results['action'] == 'BUY').mean() * 100:.1f}%")
        else:
            print("No BUY signals found.")
            
    except FileNotFoundError:
        print("weekly_full_backtest.csv not found.")

if __name__ == "__main__":
    analyze_stats()
