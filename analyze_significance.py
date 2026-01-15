import pandas as pd
import numpy as np
from scipy import stats

def calculate_max_drawdown(cumulative_returns):
    """Calculates Maximum Drawdown from a series of cumulative returns."""
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    return drawdown.min()

def analyze_significance(results_file):
    print(f"Loading results from {results_file}...")
    df = pd.read_csv(results_file)
    
    # 1. Prepare Returns Series
    # Strategy Return (convert from % to decimal)
    strategy_returns = df['period_return'] / 100.0
    
    # Benchmark (Buy & Hold) Return for the same periods
    benchmark_returns = (df['exit_price'] - df['entry_price']) / df['entry_price']
    
    # 2. Basic Metrics
    n = len(df)
    print(f"\n--- Basic Metrics (N={n} weeks) ---")
    
    strat_mean = strategy_returns.mean()
    strat_std = strategy_returns.std()
    bench_mean = benchmark_returns.mean()
    bench_std = benchmark_returns.std()
    
    print(f"Avg Weekly Return: Strategy={strat_mean:.4f} ({strat_mean*52:.2%}/yr) | Benchmark={bench_mean:.4f} ({bench_mean*52:.2%}/yr)")
    print(f"Std Dev (Risk):    Strategy={strat_std:.4f} | Benchmark={bench_std:.4f}")
    
    # 3. Sharpe Ratio (Annualized, assuming risk-free rate ~0 for comparison)
    # Weekly data -> sqrt(52)
    strat_sharpe = (strat_mean / strat_std) * np.sqrt(52)
    bench_sharpe = (bench_mean / bench_std) * np.sqrt(52)
    
    print(f"Sharpe Ratio:      Strategy={strat_sharpe:.4f} | Benchmark={bench_sharpe:.4f}")
    
    # 4. Maximum Drawdown
    # Construct equity curves
    strat_equity = (1 + strategy_returns).cumprod()
    bench_equity = (1 + benchmark_returns).cumprod()
    
    strat_dd = calculate_max_drawdown(strat_equity)
    bench_dd = calculate_max_drawdown(bench_equity)
    
    print(f"Max Drawdown:      Strategy={strat_dd:.2%} | Benchmark={bench_dd:.2%}")
    
    # 5. Statistical Significance (Paired T-test)
    # Null Hypothesis: Mean difference between Strategy and Benchmark is 0
    diff = strategy_returns - benchmark_returns
    t_stat, p_value = stats.ttest_rel(strategy_returns, benchmark_returns)
    
    print(f"\n--- Statistical Significance (Paired T-test) ---")
    print(f"T-Statistic: {t_stat:.4f}")
    print(f"P-Value:     {p_value:.4f}")
    
    if p_value < 0.05:
        print("✅ Result is Statistically Significant (p < 0.05)")
        if t_stat > 0:
            print("   Strategy significantly OUTPERFORMS Benchmark.")
        else:
            print("   Strategy significantly UNDERPERFORMS Benchmark.")
    else:
        print("❌ Result is NOT Statistically Significant (p >= 0.05)")
        print("   Performance difference could be due to random chance.")

    # 6. Robustness Check (Bootstrap)
    print(f"\n--- Robustness Check (Bootstrap 10,000 runs) ---")
    print("Resampling returns to estimate probability of outperformance...")
    
    outperform_count = 0
    n_iterations = 10000
    
    # We resample the *difference* in returns to see how often it's positive
    # Or we can resample total returns. Let's resample total returns.
    
    for _ in range(n_iterations):
        # Sample indices with replacement
        indices = np.random.choice(n, n, replace=True)
        
        sample_strat = strategy_returns.iloc[indices].mean()
        sample_bench = benchmark_returns.iloc[indices].mean()
        
        if sample_strat > sample_bench:
            outperform_count += 1
            
    prob_outperform = outperform_count / n_iterations
    print(f"Probability Strategy > Benchmark: {prob_outperform:.2%}")
    
    # Win Rate Significance (Binomial Test approximation)
    # Count weeks where Strategy > 0
    wins = (strategy_returns > 0).sum()
    win_rate = wins / n
    print(f"\n--- Win Rate Analysis ---")
    print(f"Win Rate (Weeks > 0%): {win_rate:.2%} ({wins}/{n})")
    
    # Compare to Benchmark Win Rate
    bench_wins = (benchmark_returns > 0).sum()
    bench_win_rate = bench_wins / n
    print(f"Benchmark Win Rate:    {bench_win_rate:.2%} ({bench_wins}/{n})")

if __name__ == "__main__":
    analyze_significance('weekly_full_backtest.csv')
