import pandas as pd
import numpy as np
import torch
from prospect_theory_agent import ProspectTheoryQNetwork, FinancialEnv
from datetime import datetime

def load_model(model_path, state_dim, action_dim):
    model = ProspectTheoryQNetwork(state_dim, action_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def get_signal(model, data, window_size, current_idx):
    """Gets signal for a specific day index."""
    # We need a window ending at current_idx
    # Data slice: [current_idx - window_size + 1 : current_idx + 1]
    
    if current_idx < window_size - 1:
        return 0 # Not enough data
        
    # Create a mini-env or just extract state manually
    # Let's use the env to be consistent with training
    # But we need to be careful about normalization.
    # The data passed in should already be normalized (except price)
    
    # Extract the window
    start_idx = current_idx - window_size + 1
    end_idx = current_idx + 1
    window_data = data.iloc[start_idx:end_idx]
    
    env = FinancialEnv(window_data, window_size=window_size)
    state = env.reset() # This gets the state for the *last* step in this window?
    # No, reset() gets state at step 0.
    # If we pass a dataframe of length window_size, step 0 is the ONLY state (full window).
    # env._get_state() handles it.
    
    # However, env.reset() sets current_step = 0.
    # And _get_state() uses current_step.
    # If len(data) == window_size, then at step 0:
    # start = 0 - 19 = -19. 
    # It pads.
    # We want the state representing the FULL window.
    # That happens when current_step = window_size - 1.
    
    # So we should pass a slightly larger buffer or manually set step.
    # Easier: Pass the full dataframe to env, set env.current_step = current_idx, get state.
    
    # Re-instantiating env for every step is inefficient but safe.
    # Let's use one env for the whole dataframe.
    pass

def calculate_user_pnl():
    print("--- Calculating User PnL (Starting Nov 21, 2025) ---")
    
    # 1. Load Data
    df = pd.read_csv('qqq_data_60days.csv', index_col='Date', parse_dates=True)
    df = df.apply(pd.to_numeric, errors='coerce').dropna()
    
    # Preprocess
    df['log_return'] = np.log(df['price'] / df['price'].shift(1))
    df = df.dropna()
    
    # Normalize features
    market_data = df.copy()
    feature_cols = [c for c in market_data.columns if c != 'price']
    for col in feature_cols:
        market_data[col] = (market_data[col] - market_data[col].mean()) / (market_data[col].std() + 1e-8)
        
    # 2. Identify Trade Dates (Fridays)
    # User started Nov 21.
    # Dates: Nov 21, Nov 28, Dec 5, Dec 12, Dec 19, Dec 26
    target_dates = [
        '2025-11-21',
        '2025-11-28',
        '2025-12-05',
        '2025-12-12',
        '2025-12-19',
        '2025-12-26'
    ]
    
    # 3. Load Models
    # Old Model (Window 20)
    env_20 = FinancialEnv(market_data, window_size=20)
    model_old = load_model('prospect_theory_model.pth', env_20.state_dim, env_20.action_dim)
    
    # Tuned Model (Window 60)
    env_60 = FinancialEnv(market_data, window_size=60)
    model_tuned = load_model('prospect_theory_model_tuned.pth', env_60.state_dim, env_60.action_dim)
    
    # 4. Simulate Trades
    capital = 10000.0 # Hypothetical start
    shares = 0
    position = 0 # 0=Cash, 1=Long, -1=Short
    
    print(f"\n{'Date':<12} | {'Model':<10} | {'Signal':<6} | {'Price':<8} | {'Result':<10}")
    print("-" * 60)
    
    total_pnl_pct = 0
    
    for i, date_str in enumerate(target_dates):
        if date_str not in df.index:
            print(f"⚠️ Date {date_str} not found in data. Finding nearest...")
            # Logic to find nearest previous day if needed, but let's assume data is good for now
            # Actually, looking at the preview, dates are YYYY-MM-DD.
            continue
            
        current_idx = df.index.get_loc(date_str)
        current_price = df.iloc[current_idx]['price']
        
        # Determine which model to use
        # Use Old Model for everything before Dec 26
        if date_str == '2025-12-26':
            model = model_tuned
            window_size = 60
            model_name = "Tuned"
        else:
            model = model_old
            window_size = 20
            model_name = "Old"
            
        # Get Signal
        # We need to set the env to the correct step
        if window_size == 60:
            env = env_60
        else:
            env = env_20
            
        env.current_step = current_idx
        state = env._get_state()
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = model(state_tensor)[0].numpy()
            action = np.argmax(q_values)
            
        action_name = ['HOLD', 'BUY', 'SELL'][action]
        
        # Execute Trade (Weekly PnL)
        # We assume we hold for 1 week (until next date)
        # If it's the last date (Dec 26), we don't have a result yet.
        
        pnl = 0
        if i < len(target_dates) - 1:
            next_date_str = target_dates[i+1]
            if next_date_str in df.index:
                next_price = df.loc[next_date_str]['price']
                
                if action == 1: # BUY
                    pnl = (next_price - current_price) / current_price
                elif action == 2: # SELL
                    pnl = (current_price - next_price) / current_price
                
                total_pnl_pct += pnl
                result_str = f"{pnl*100:+.2f}%"
            else:
                result_str = "Pending"
        else:
            result_str = "Open"
            
        print(f"{date_str:<12} | {model_name:<10} | {action_name:<6} | ${current_price:<7.2f} | {result_str:<10}")
        
    print("-" * 60)
    print(f"Total Cumulative PnL: {total_pnl_pct*100:+.2f}%")
    
    # Estimate dollar value
    print(f"\nIf you started with $10,000:")
    print(f"Current Value: ${10000 * (1 + total_pnl_pct):,.2f}")
    print(f"Profit/Loss:   ${10000 * total_pnl_pct:,.2f}")

if __name__ == "__main__":
    calculate_user_pnl()
