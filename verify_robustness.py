import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prospect_theory_agent import ProspectTheoryQNetwork, FinancialEnv, ReplayBuffer, select_action, train_step

def evaluate_performance(agent, data, window_size, name):
    """
    Evaluates the agent on a specific dataset and calculates metrics.
    """
    env = FinancialEnv(data, window_size=window_size)
    state = env.reset()
    done = False
    
    # Tracking
    actions = []
    daily_returns = []
    prices = data['price'].values
    positions = [0] # 0=Cash, 1=Long, -1=Short
    
    # We need to align steps with prices
    # The environment window ends at index (step + window_size)
    
    step = 0
    with torch.no_grad():
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = agent(state_tensor)
            action = q_values.argmax(dim=1).item()
            actions.append(action)
            
            # Identify current price and next price for PnL
            # Env step: returns next_state, reward, done
            curr_idx = step + window_size - 1
            if curr_idx >= len(prices) - 1:
                break
                
            curr_price = prices[curr_idx]
            next_price = prices[curr_idx + 1]
            
            # Simple PnL Logic
            pct_change = (next_price - curr_price) / curr_price
            
            # Determine position based on Action (0=Hold, 1=Buy, 2=Sell)
            # Assuming simplified "Hold" maintains previous position for this calculation,
            # BUT the model treats 0 as specific "stay out/do nothing" usually.
            # Let's stick to the strict interpretation:
            # 0 = Cash/Neutral
            # 1 = Long
            # 2 = Short
            
            if action == 1:
                daily_ret = pct_change
            elif action == 2:
                daily_ret = -pct_change
            else:
                daily_ret = 0.0
            
            daily_returns.append(daily_ret)
            
            next_state, _, done = env.step(action)
            state = next_state
            step += 1

    # --- Metrics Calculation ---
    daily_returns = np.array(daily_returns)
    total_pnl = np.sum(daily_returns) * 100 # Simple sum for approx %
    
    # Annualized Sharpe (Assuming 252 trading days)
    if np.std(daily_returns) == 0:
        sharpe = 0
    else:
        sharpe = (np.mean(daily_returns) / np.std(daily_returns)) * np.sqrt(252)
        
    # Hit Rate (Win Rate)
    # Only count days where we took an active position (Buy or Sell)
    active_trades = daily_returns[daily_returns != 0]
    if len(active_trades) > 0:
        hit_rate = len(active_trades[active_trades > 0]) / len(active_trades)
    else:
        hit_rate = 0
        
    print(f"\n--- {name} RESULTS ---")
    print(f"Days: {len(daily_returns)}")
    print(f"Total Return: {total_pnl:.2f}%")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Hit Rate: {hit_rate*100:.1f}%")
    print(f"Trades Taken: {len(active_trades)} / {len(daily_returns)} days")
    
    return {
        'sharpe': sharpe,
        'return': total_pnl,
        'hit_rate': hit_rate,
        'actions': actions
    }

if __name__ == "__main__":
    print("WARNING: This script performs a robust Train/Test split.")
    print("It will TRAIN A NEW MODEL from scratch on the first 80% of data.")
    print("This may take 1-2 minutes.\n")
    
    # 1. Prepare Data
    df = pd.read_csv('qqq_market_data.csv')
    df = df.apply(pd.to_numeric, errors='coerce').dropna()
    df['log_return'] = np.log(df['price'] / df['price'].shift(1))
    df = df.dropna()
    
    # Feature Normalize
    market_data = df.copy()
    feature_cols = [c for c in market_data.columns if c != 'price']
    for col in feature_cols:
        market_data[col] = (market_data[col] - market_data[col].mean()) / (market_data[col].std() + 1e-8)

    # 2. Split Data (80% Train, 20% Test)
    split_idx = int(len(market_data) * 0.8)
    train_data = market_data.iloc[:split_idx].reset_index(drop=True)
    test_data = market_data.iloc[split_idx:].reset_index(drop=True)
    
    print(f"Total Data: {len(market_data)} rows")
    print(f"Train Set:  {len(train_data)} rows (2015-Mid 2023 approx)")
    print(f"Test Set:   {len(test_data)} rows (Mid 2023-Present)")
    
    # 3. Train Model (Simplified for Speed - 300 Episodes)
    WINDOW_SIZE = 20
    env = FinancialEnv(train_data, window_size=WINDOW_SIZE)
    agent = ProspectTheoryQNetwork(env.state_dim, env.action_dim)
    target_agent = ProspectTheoryQNetwork(env.state_dim, env.action_dim)
    target_agent.load_state_dict(agent.state_dict())
    
    optimizer = optim.Adam(agent.parameters(), lr=0.001)
    buffer = ReplayBuffer(capacity=10000)
    
    print("\nTraining on In-Sample Data (5 Episodes - QUICK CHECK)...")
    
    epsilon = 1.0
    for episode in range(5):
        state = env.reset()
        done = False
        while not done:
            action = select_action(state, agent, epsilon)
            next_state, reward, done = env.step(action)
            buffer.push(state, action, reward, next_state, done)
            state = next_state
            train_step(agent, target_agent, buffer, 64, optimizer, 0.99)
            
        epsilon = max(0.01, epsilon * 0.50) # Very fast decay
        print(f"Episode {episode+1}/5 complete.")
        if episode % 10 == 0:
            target_agent.load_state_dict(agent.state_dict())
            
    print("Training Complete.")
    
    # 4. Evaluate
    print("\n" + "="*40)
    print("📊 ROBUSTNESS CHECK REPORT")
    print("="*40)
    
    train_results = evaluate_performance(agent, train_data, WINDOW_SIZE, "IN-SAMPLE (Training)")
    test_results = evaluate_performance(agent, test_data, WINDOW_SIZE, "OUT-OF-SAMPLE (Test)")
    
    # 5. Interpret
    print("\n" + "="*40)
    print("📈 VERDICT")
    print("="*40)
    
    sharp_diff = train_results['sharpe'] - test_results['sharpe']
    
    if test_results['sharpe'] > 0.5 and test_results['return'] > 0:
        print("✅ PASS: Strategy shows positive expectancy on unseen data.")
    else:
        print("⚠️  FAIL: Strategy struggles on unseen data.")
        
    if abs(sharp_diff) > 1.0:
        print(f"⚠️  OVERFITTING DETECTED: Large drop in Sharpe ({train_results['sharpe']:.2f} -> {test_results['sharpe']:.2f})")
    else:
        print("✅ STABLE: Performance is consistent between Train and Test.")
