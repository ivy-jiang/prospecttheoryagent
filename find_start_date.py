import yfinance as yf
import pandas as pd

def find_date():
    target_price = 105.98359680175781
    print(f"Searching for price: {target_price}")
    
    # Fetch QQQ data from 2015 (QQQ was around 100 in 2015-2016)
    # QQQ was ~100 in 2015.
    data = yf.download("QQQ", start="2015-01-01", end="2020-01-01", progress=False)
    
    # Check Close and Adj Close
    # The data might be Adj Close.
    
    # Iterate and find closest match
    min_diff = 1.0
    best_date = None
    best_col = None
    
    for col in ['Close', 'Adj Close']:
        if col in data.columns:
            # Handle MultiIndex if present (yfinance update)
            # yfinance might return (Price, Ticker) columns
            try:
                series = data[col]
                if isinstance(series, pd.DataFrame):
                    series = series.iloc[:, 0] # Take first column (QQQ)
                
                for date, price in series.items():
                    diff = abs(price - target_price)
                    if diff < min_diff:
                        min_diff = diff
                        best_date = date
                        best_col = col
            except Exception as e:
                print(f"Error checking {col}: {e}")

    print(f"Best match: {best_date} in {best_col} with diff {min_diff}")

if __name__ == "__main__":
    find_date()
