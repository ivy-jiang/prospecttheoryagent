import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prospect_theory_agent import ProspectTheoryQNetwork, FinancialEnv, ReplayBuffer, select_action, train_step

"""
TUNED TRAINING SCRIPT
Retrains the model with hyperparameters optimized for trend capturing and lower risk aversion.
Changes:
- Window Size: 20 -> 60 (Capture longer term trends)
- Episodes: 1000 -> 3000 (More training time)
- Batch Size: 32 -> 64 (More stable updates)
"""

def plot_results(rewards, filename='tuned_results.png'):
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, alpha=0.3, color='blue', label='Episode Reward')
    
    # Rolling average
    if len(rewards) >= 50:
        rolling_avg = pd.Series(rewards).rolling(50).mean()
        plt.plot(rolling_avg, label='50-Episode Rolling Avg', color='red', linewidth=2)
        
    plt.title('Tuned Model Training Performance')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    print(f"Plot saved to {filename}")

if __name__ == "__main__":
    
    # --- 1. Load Data ---
    print("Loading QQQ market data...")
    try:
        df = pd.read_csv('qqq_market_data.csv')
        df = df.apply(pd.to_numeric, errors='coerce').dropna()
        
        # Feature Engineering
        df['log_return'] = np.log(df['price'] / df['price'].shift(1))
        df = df.dropna()
        
        market_data = df.copy()
        feature_cols = [c for c in market_data.columns if c != 'price']
        
        # Normalize
        for col in feature_cols:
            market_data[col] = (market_data[col] - market_data[col].mean()) / (market_data[col].std() + 1e-8)
            
        print(f"Data loaded. Shape: {market_data.shape}")

    except FileNotFoundError:
        print("Error: qqq_market_data.csv not found.")
        exit()

    # --- 2. Tuned Hyperparameters ---
    NUM_EPISODES = 500       # Reduced for speed (was 3000)
    BATCH_SIZE = 128         # Increased for speed
    GAMMA = 0.99            
    EPS_START = 1.0         
    EPS_END = 0.01          
    EPS_DECAY = 0.99        # Faster decay (0.99^500 ~ 0.006)
    TARGET_UPDATE = 10       
    WINDOW_SIZE = 60         # Keep larger window
    
    print(f"Training with Window Size: {WINDOW_SIZE}, Episodes: {NUM_EPISODES}")

    # --- 3. Initialization ---
    env = FinancialEnv(market_data, window_size=WINDOW_SIZE)
    state_dim = env.state_dim
    action_dim = env.action_dim
    
    agent = ProspectTheoryQNetwork(state_dim, action_dim)
    target_agent = ProspectTheoryQNetwork(state_dim, action_dim)
    target_agent.load_state_dict(agent.state_dict())
    target_agent.eval()
    
    optimizer = optim.Adam(agent.parameters(), lr=0.0001) # Lower LR for stability
    buffer = ReplayBuffer(capacity=50000) # Larger buffer
    
    episode_rewards = []
    epsilon = EPS_START

    # --- 4. Training Loop ---
    print("Starting training...")
    for episode in range(NUM_EPISODES):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = select_action(state, agent, epsilon)
            next_state, reward, done = env.step(action)
            buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            train_step(agent, target_agent, buffer, BATCH_SIZE, optimizer, GAMMA)
            
        episode_rewards.append(total_reward)
        epsilon = max(EPS_END, epsilon * EPS_DECAY)
        
        if episode % TARGET_UPDATE == 0:
            target_agent.load_state_dict(agent.state_dict())
            
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode: {episode+1}/{NUM_EPISODES} | Avg Reward: {avg_reward:.2f} | Epsilon: {epsilon:.2f}")

    print("Training complete.")
    
    # --- Save Tuned Model ---
    torch.save(agent.state_dict(), 'prospect_theory_model_tuned.pth')
    print("Model saved to prospect_theory_model_tuned.pth")
    
    plot_results(episode_rewards)
