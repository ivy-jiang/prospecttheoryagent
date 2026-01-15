import pandas as pd
import numpy as np
import torch
from prospect_theory_agent import ProspectTheoryQNetwork, FinancialEnv
from analyze_60day_weekly import get_all_signals
from datetime import datetime, timedelta

# Load Data
data_file = 'qqq_data_60days.csv'
df = pd.read_csv(data_file)
df = df.apply(pd.to_numeric, errors='coerce').dropna()

# Preprocess
df['log_return'] = np.log(df['price'] / df['price'].shift(1))
df = df.dropna()
feature_cols = [c for c in df.columns if c != 'price']
for col in feature_cols:
    df[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-8)

# Load Model
model_path = 'prospect_theory_model.pth'
env = FinancialEnv(df, window_size=20)
model = ProspectTheoryQNetwork(env.state_dim, env.action_dim)
model.load_state_dict(torch.load(model_path))
model.eval()

# Get Signals
signals = get_all_signals(model, df, window_size=20)

# Filter for Day 5 (Mondays)
# Start date: Aug 28, 2024 (Wednesday) -> Day 0
# Day 5 is Monday.
# Indices: 5, 12, 19, ...
indices = list(range(5, len(signals), 7))
monday_signals = signals.iloc[indices]

print(f"Total Mondays in period: {len(monday_signals)}")
print("Action Counts on Mondays:")
print(monday_signals['action'].value_counts())

# Decode actions
action_names = {0: 'Hold', 1: 'Buy', 2: 'Sell'}
counts = monday_signals['action'].value_counts().rename(index=action_names)
print("\nReadable Counts:")
print(counts)
