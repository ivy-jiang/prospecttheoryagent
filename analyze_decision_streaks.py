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

def analyze_streaks(data_file, model_path):
    # 1. Load Data
    print(f"Loading {data_file}...")
    df = pd.read_csv(data_file, index_col=0, parse_dates=True)
    
    # Preprocess (same as training)
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
    
    # Get dates corresponding to the steps
    # The environment starts at window_size
    valid_dates = df.index[WINDOW_SIZE:]
    
    with torch.no_grad():
        while not done and step < len(valid_dates):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = agent(state_tensor)[0].numpy()
            action = np.argmax(q_values)
            
            decisions.append(action)
            dates.append(valid_dates[step])
            
            next_state, _, done = env.step(action)
            state = next_state
            step += 1
            
    # Create DataFrame
    results = pd.DataFrame({
        'Date': dates,
        'Action': decisions,
        'Action_Name': [['HOLD', 'BUY', 'SELL'][a] for a in decisions]
    })
    
    # 4. Calculate Streaks
    print("Calculating streaks...")
    results['group'] = (results['Action'] != results['Action'].shift()).cumsum()
    streaks = results.groupby(['group', 'Action_Name']).size().reset_index(name='length')
    
    # Metrics
    avg_streaks = streaks.groupby('Action_Name')['length'].mean()
    max_streaks = streaks.groupby('Action_Name')['length'].max()
    change_freq = len(streaks) / len(results)
    avg_days_between_changes = len(results) / len(streaks)
    
    print("\n" + "="*50)
    print("STREAK ANALYSIS RESULTS")
    print("="*50)
    print(f"Total Days Analyzed: {len(results)}")
    print(f"Total Decision Changes: {len(streaks)}")
    print(f"Average Days Between Changes: {avg_days_between_changes:.1f} days")
    print("\nAverage Streak Length (Days):")
    print(avg_streaks)
    print("\nMax Streak Length (Days):")
    print(max_streaks)
    
    # 5. Visualization
    print("\nGenerating visualization...")
    plt.figure(figsize=(15, 6))
    
    # Map actions to colors
    colors = {0: 'gray', 1: 'green', 2: 'red'} # Hold, Buy, Sell
    labels = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
    
    # Create colored bar chart
    # We use a bar chart where height is 1, just to show the color strip
    # Or better: a step plot or scatter plot
    
    # Let's do a colored bar chart for the timeline
    for action in [0, 1, 2]:
        mask = results['Action'] == action
        if mask.any():
            plt.bar(results.loc[mask, 'Date'], 1, width=1.0, 
                   color=colors[action], label=labels[action], alpha=0.8, align='edge')
    
    plt.title('Model Trading Decisions Over Time', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.yticks([]) # Hide y-axis numbers
    plt.legend(loc='upper left')
    plt.grid(axis='x', alpha=0.3)
    
    # Format x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.gcf().autofmt_xdate()
    
    plt.tight_layout()
    plt.savefig('decision_streaks.png', dpi=150)
    print("Saved chart to 'decision_streaks.png'")
    
    # Save CSV
    results.to_csv('decision_history.csv', index=False)
    print("Saved data to 'decision_history.csv'")

if __name__ == "__main__":
    analyze_streaks('qqq_market_data_with_dates.csv', 'prospect_theory_model_tuned.pth')
