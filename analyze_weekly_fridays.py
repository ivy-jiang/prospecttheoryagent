import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from prospect_theory_agent import ProspectTheoryQNetwork, FinancialEnv

def load_model(model_path, state_dim, action_dim):
    model = ProspectTheoryQNetwork(state_dim, action_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def analyze_weekly(data_file, model_path):
    # 1. Load Data
    print(f"Loading {data_file}...")
    df = pd.read_csv(data_file, index_col=0, parse_dates=True)
    
    # Preprocess
    df['log_return'] = np.log(df['price'] / df['price'].shift(1))
    df = df.dropna()
    
    # Normalize
    feature_cols = [c for c in df.columns if c != 'price']
    for col in feature_cols:
        df[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-8)
        
    print(f"Data loaded: {len(df)} rows")
    
    # 2. Load Model
    WINDOW_SIZE = 60
    env = FinancialEnv(df, window_size=WINDOW_SIZE)
    agent = load_model(model_path, env.state_dim, env.action_dim)
    
    # 3. Run Inference
    print("Running inference...")
    decisions = []
    dates = []
    
    state = env.reset()
    done = False
    step = 0
    
    valid_dates = df.index[WINDOW_SIZE:]
    
    with torch.no_grad():
        while not done and step < len(valid_dates):
            current_date = valid_dates[step]
            
            # Only record decision if it's a Friday (weekday == 4)
            if current_date.weekday() == 4:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = agent(state_tensor)[0].numpy()
                action = np.argmax(q_values)
                
                decisions.append(action)
                dates.append(current_date)
            
            # Always step the environment to keep state updated
            next_state, _, done = env.step(0) # Action doesn't matter for state update in this env
            state = next_state
            step += 1
            
    # Create DataFrame
    results = pd.DataFrame({
        'Date': dates,
        'Action': decisions,
        'Action_Name': [['HOLD', 'BUY', 'SELL'][a] for a in decisions]
    })
    
    # 4. Metrics
    print("\n" + "="*50)
    print("WEEKLY (FRIDAY) ANALYSIS RESULTS")
    print("="*50)
    print(f"Total Fridays Analyzed: {len(results)}")
    
    counts = results['Action_Name'].value_counts()
    print("\nDecision Counts:")
    print(counts)
    
    print("\nDecision Percentages:")
    print(counts / len(results) * 100)
    
    # 5. Visualization
    print("\nGenerating visualization...")
    plt.figure(figsize=(15, 6))
    
    colors = {0: 'gray', 1: 'green', 2: 'red'}
    labels = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
    
    # Plot bars
    # Since these are weekly, we can use a wider bar width
    for action in [0, 1, 2]:
        mask = results['Action'] == action
        if mask.any():
            plt.bar(results.loc[mask, 'Date'], 1, width=5.0, 
                   color=colors[action], label=labels[action], alpha=0.8)
    
    plt.title('Weekly (Friday) Trading Decisions', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.yticks([])
    plt.legend(loc='upper left')
    plt.grid(axis='x', alpha=0.3)
    
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.gcf().autofmt_xdate()
    
    plt.tight_layout()
    plt.savefig('weekly_decision_streaks.png', dpi=150)
    print("Saved chart to 'weekly_decision_streaks.png'")
    
    results.to_csv('weekly_decision_history.csv', index=False)
    print("Saved data to 'weekly_decision_history.csv'")

if __name__ == "__main__":
    analyze_weekly('qqq_market_data_with_dates.csv', 'prospect_theory_model_tuned.pth')
