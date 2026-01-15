import torch
import numpy as np
import pandas as pd
from prospect_theory_agent import ProspectTheoryQNetwork, FinancialEnv

"""
FULL HISTORY WEEKLY BACKTEST (NON-OVERLAPPING)
Tests the model on the full history but only trades once every 5 trading days (1 week).
This ensures no overlapping decision windows.
"""

def load_model(model_path, state_dim, action_dim):
    model = ProspectTheoryQNetwork(state_dim, action_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def backtest_weekly(model, data, window_size=20, holding_period=5):
    """
    Runs the model with a fixed holding period (e.g., 5 days for weekly).
    """
    # We can't use the standard env.step() easily for skipping days 
    # because it advances 1 by 1. We'll manually iterate.
    
    prices = data['price'].values
    
    # Initialize Environment just to get states
    env = FinancialEnv(data, window_size=window_size)
    
    results = []
    capital = 10000.0
    shares = 0
    
    # Start loop
    # We need enough data for the first window
    # env.reset() gives us the state at step 0 (which uses data from 0 to window_size)
    # Actually env.reset() sets current_step = 0.
    # The state is built from indices [0 : window_size].
    # The price at that moment is prices[window_size-1] ?? 
    # Let's look at env._get_state():
    #   start = self.current_step - (self.window_size - 1)
    #   end = self.current_step + 1
    # So at step 0, it pads with zeros.
    # We want to start when we have full data? 
    # The env handles padding, so starting at 0 is fine.
    
    # We will iterate with a step size of 'holding_period'
    
    total_steps = len(prices)
    
    print(f"Starting Weekly Backtest with ${capital:.2f}...")
    print(f"Holding Period: {holding_period} trading days")
    
    # We iterate through indices i
    # i represents the decision day.
    # We execute at Close of day i (or Open of i+1, simplified to Close of i)
    # We hold until day i + holding_period
    
    for i in range(0, total_steps - holding_period, holding_period):
        
        # 1. Get State for day i
        # We need to manually set the env to this step to get the correct state
        env.current_step = i
        state = env._get_state()
        
        # 2. Get Signal
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = model(state_tensor)[0].numpy()
            action = np.argmax(q_values)
            
        # 3. Execute Trade
        # Price at decision time
        entry_price = prices[i]
        # Price at end of holding period
        exit_price = prices[i + holding_period]
        
        # Calculate Return for this period
        period_return = 0
        trade_type = "HOLD"
        
        if action == 1: # BUY
            period_return = (exit_price - entry_price) / entry_price
            trade_type = "BUY"
        elif action == 2: # SELL
            period_return = (entry_price - exit_price) / entry_price
            trade_type = "SELL"
        else: # HOLD
            # If we hold cash, return is 0 (ignoring interest)
            period_return = 0
            trade_type = "HOLD"
            
        # Update Capital
        capital = capital * (1 + period_return)
        
        results.append({
            'entry_day': i,
            'exit_day': i + holding_period,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'action': trade_type,
            'period_return': period_return * 100,
            'capital': capital
        })
        
    return pd.DataFrame(results)

if __name__ == "__main__":
    # Configuration
    data_file = 'qqq_market_data.csv'
    model_path = 'prospect_theory_model.pth'
    WINDOW_SIZE = 20
    HOLDING_PERIOD = 5 # 1 week (5 trading days)
    
    # Load Data
    print(f"Loading {data_file}...")
    df = pd.read_csv(data_file)
    df = df.apply(pd.to_numeric, errors='coerce').dropna()
    
    # Preprocess (Same as training)
    df['log_return'] = np.log(df['price'] / df['price'].shift(1))
    df = df.dropna()
    
    feature_cols = [c for c in df.columns if c != 'price']
    for col in feature_cols:
        df[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-8)
        
    # Load Model
    temp_env = FinancialEnv(df, window_size=WINDOW_SIZE)
    agent = load_model(model_path, temp_env.state_dim, temp_env.action_dim)
    
    # Run Backtest
    results = backtest_weekly(agent, df, WINDOW_SIZE, HOLDING_PERIOD)
    
    # Analysis
    print("\n" + "="*60)
    print("WEEKLY (NON-OVERLAPPING) BACKTEST RESULTS")
    print("="*60)
    
    start_price = results['entry_price'].iloc[0]
    end_price = results['exit_price'].iloc[-1]
    buy_hold_return = (end_price - start_price) / start_price * 100
    
    final_capital = results['capital'].iloc[-1]
    strategy_return = (final_capital - 10000) / 10000 * 100
    
    print(f"Period: {len(results)} weeks ({len(results)*5} days)")
    print(f"Buy & Hold Return: {buy_hold_return:.2f}%")
    print(f"Strategy Return:   {strategy_return:.2f}%")
    
    if strategy_return > buy_hold_return:
        print(f"✅ OUTPERFORMED Buy & Hold by {strategy_return - buy_hold_return:.2f}%")
    else:
        print(f"❌ UNDERPERFORMED Buy & Hold by {buy_hold_return - strategy_return:.2f}%")
        
    print("\nTrade Counts:")
    print(results['action'].value_counts())
    
    results.to_csv('weekly_full_backtest.csv', index=False)
    print("\nSaved to weekly_full_backtest.csv")
