import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prospect_theory_agent import ProspectTheoryQNetwork, FinancialEnv

"""
FULL HISTORY BACKTESTING SCRIPT
Tests the model on the entire available history in qqq_market_data.csv.
"""

def load_model(model_path, state_dim, action_dim):
    """Loads the trained model."""
    model = ProspectTheoryQNetwork(state_dim, action_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def backtest_full_history(model, data, window_size=20):
    """
    Runs the model on the full dataset and tracks performance.
    """
    env = FinancialEnv(data, window_size=window_size)
    
    results = []
    state = env.reset()
    done = False
    step = 0
    
    # Get actual prices for PnL calculation
    prices = data['price'].values
    dates = data.index.values if 'Date' in data.index.names else range(len(data))
    
    current_position = 0 # 0=cash, 1=long, -1=short
    entry_price = 0
    cash = 10000.0 # Starting capital
    shares = 0
    portfolio_values = []
    
    print(f"Starting Backtest with ${cash:.2f}...")
    
    with torch.no_grad():
        while not done:
            # Get Q-values
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = model(state_tensor)[0].numpy()
            
            # Get action
            action = np.argmax(q_values)
            
            # Execute Trade Logic (Simplified for Backtest)
            current_price = prices[step + window_size] if step + window_size < len(prices) else prices[-1]
            
            trade_action = "HOLD"
            
            if action == 1: # BUY
                if current_position == 0:
                    shares = cash / current_price
                    cash = 0
                    current_position = 1
                    entry_price = current_price
                    trade_action = "BUY"
                elif current_position == -1:
                    # Close short
                    cash = cash + (entry_price - current_price) * shares # Profit/Loss
                    # Open long
                    shares = cash / current_price
                    cash = 0
                    current_position = 1
                    entry_price = current_price
                    trade_action = "FLIP LONG"
                    
            elif action == 2: # SELL
                if current_position == 0:
                    # Shorting (simplified: assume we can short with full cash)
                    shares = cash / current_price
                    entry_price = current_price
                    current_position = -1
                    trade_action = "SHORT"
                elif current_position == 1:
                    # Close long
                    cash = shares * current_price
                    shares = 0
                    # Open short
                    shares = cash / current_price
                    entry_price = current_price
                    current_position = -1
                    trade_action = "FLIP SHORT"
            
            # Calculate Portfolio Value
            if current_position == 1:
                port_value = shares * current_price
            elif current_position == -1:
                # Short value = Cash + (Entry - Current) * Shares
                # But we tracked cash differently above. 
                # Let's simplify: Value = Initial Capital * (1 + PnL%)
                # Re-calculating for tracking:
                # Value = Cash_at_entry + (Entry - Current) * Shares
                # This is tricky with the simplified logic above.
                # Let's stick to the env's step for simplicity in logic, 
                # but here we want a continuous equity curve.
                
                # Alternative: Just track % returns
                pass
            
            # Re-using simple logic:
            # Value = Cash + (Shares * Price if Long)
            # Shorting adds complexity to simple tracking.
            # Let's use a simpler PnL tracker based on % returns.
            
            next_state, reward, done = env.step(action)
            
            results.append({
                'step': step,
                'price': current_price,
                'action': action,
                'action_name': ['HOLD', 'BUY', 'SELL'][action],
                'q_hold': q_values[0],
                'q_buy': q_values[1],
                'q_sell': q_values[2]
            })
            
            state = next_state
            step += 1
            
    return pd.DataFrame(results)

if __name__ == "__main__":
    # Configuration
    data_file = 'qqq_market_data.csv'
    model_path = 'prospect_theory_model.pth'
    WINDOW_SIZE = 20
    
    # Load Data
    print(f"Loading {data_file}...")
    df = pd.read_csv(data_file)
    df = df.apply(pd.to_numeric, errors='coerce').dropna()
    
    # Preprocess
    df['log_return'] = np.log(df['price'] / df['price'].shift(1))
    df = df.dropna()
    
    # Normalize
    feature_cols = [c for c in df.columns if c != 'price']
    for col in feature_cols:
        df[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-8)
        
    print(f"Data loaded: {len(df)} rows")
    
    # Load Model
    print("Loading model...")
    temp_env = FinancialEnv(df, window_size=WINDOW_SIZE)
    agent = load_model(model_path, temp_env.state_dim, temp_env.action_dim)
    
    # Run Backtest
    print("Running backtest on full history...")
    results = backtest_full_history(agent, df, WINDOW_SIZE)
    
    # Analyze Results
    print("\n" + "="*60)
    print("FULL HISTORY BACKTEST RESULTS")
    print("="*60)
    
    # Calculate Buy & Hold Return
    start_price = results['price'].iloc[0]
    end_price = results['price'].iloc[-1]
    buy_hold_return = (end_price - start_price) / start_price * 100
    
    print(f"Period: {len(results)} trading days")
    print(f"Start Price: ${start_price:.2f}")
    print(f"End Price:   ${end_price:.2f}")
    print(f"Buy & Hold Return: {buy_hold_return:.2f}%")
    
    # Calculate Strategy Return (Simplified: Sum of daily rewards)
    # Note: The environment returns price difference as reward
    # We need to reconstruct the cumulative PnL
    
    # Let's simulate the PnL from the actions
    capital = 10000.0
    shares = 0
    position = 0 # 0=Cash, 1=Long, -1=Short
    equity_curve = [capital]
    
    for i in range(len(results) - 1):
        action = results['action'].iloc[i]
        curr_price = results['price'].iloc[i]
        next_price = results['price'].iloc[i+1]
        
        # Execute Action
        if action == 1: # Buy
            position = 1
        elif action == 2: # Sell
            position = -1
        elif action == 0: # Hold
            pass # Keep previous position
            
        # Calculate Daily PnL
        if position == 1:
            change = (next_price - curr_price) / curr_price
        elif position == -1:
            change = (curr_price - next_price) / curr_price
        else:
            change = 0
            
        capital = capital * (1 + change)
        equity_curve.append(capital)
        
    final_capital = equity_curve[-1]
    strategy_return = (final_capital - 10000) / 10000 * 100
    
    print(f"Strategy Final Capital: ${final_capital:.2f}")
    print(f"Strategy Return: {strategy_return:.2f}%")
    
    if strategy_return > buy_hold_return:
        print(f"✅ OUTPERFORMED Buy & Hold by {strategy_return - buy_hold_return:.2f}%")
    else:
        print(f"❌ UNDERPERFORMED Buy & Hold by {buy_hold_return - strategy_return:.2f}%")
        
    # Trade Counts
    print("\nTrade Counts:")
    print(results['action_name'].value_counts())
    
    # Save Results
    results.to_csv('full_backtest_results.csv', index=False)
    print("\nDetailed results saved to 'full_backtest_results.csv'")
