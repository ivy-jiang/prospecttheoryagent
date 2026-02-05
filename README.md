# Prospect Theory Agent 🚀

**Prospect Theory Agent** is an AI-powered trading assistant designed to trade **QQQ** (Nasdaq-100 ETF). It uses **Prospect Theory** and **Reinforcement Learning (DQN)** to make trading decisions that account for human-like risk perception (loss aversion), maximizing *Subjective Utility* rather than just raw expected value.

---

## ⚡️ Daily Signal: How to Trade

1.  **Run the Assistant**:
    Run the following command to fetch the latest data and get today's recommendation:
    ```bash
    python fetch_qqq.py && python get_recommendation.py
    ```

2.  **Read the Signal**:
    The system will output:
    *   📈 **BUY**: Model predicts an uptrend.
    *   📉 **SELL**: Model predicts a downtrend.
    *   ⏸️ **HOLD**: Market is uncertain; stay neutral.
    *   **Confidence %**: Assessing how strong the signal is.

3.  **Automatic Logging**:
    Every recommendation is automatically saved to **`trade_log.csv`**.

---

## 📚 Model Documentation

We believe in "White Box" AI. All inner workings of the model are fully documented:

*   **[`MODEL_DOCUMENTATION.md`](MODEL_DOCUMENTATION.md)**:  
    **The Theory Manual.** Explains the Neural Network architecture, why we use Prospect Theory, why specific variables (like Log Returns and VIX) were chosen, and the rationale behind engineering choices (ReLU, Adam, Window Sizes).
    
*   **[`MODEL_PERFORMANCE_REPORT.md`](MODEL_PERFORMANCE_REPORT.md)**:  
    **The Test Scores.** Details the rigourous "In-Sample vs Out-of-Sample" testing methodology used to verify that the model is robust and not just memorizing the past.

---

## 📂 Repository Guide

### 🚀 Core Production Files
*   **`prospect_theory_agent.py`**: **The Brain.** Defines the Dual-Stream DQN architecture and the custom Prospect Theory loss function.
*   **`fetch_qqq.py`**: **The Eyes.** Fetches live market data (Price, VIX, Treasury Yields) from Yahoo Finance/FRED.
*   **`get_recommendation.py`**: **The Mouth.** Loads the trained model (`.pth`), analyzes the latest data, and speaks the actionable recommendation.
*   **`trade_log.csv`**: **The Journal.** Persistent history of all model recommendations.
*   **`prospect_theory_model_tuned.pth`**: **The Knowledge.** The saved weights of the currently active, fully-trained model (500 episodes).

### 🧪 Training & Verification
*   **`train_tuned_model.py`**: The script used to train the production model. It runs for 500 episodes with optimized hyperparameters (Window=60, Batch=128).
*   **`verify_robustness.py`**: A diagnostic tool. It performs a strict **Train/Test Split** (80/20) to check for overfitting. *Run this if you want to stress-test the strategy logic yourself.*
*   **`backtest_full_history.py`**: Runs the trained model over the entire history of data to generate an equity curve and calculating total PnL.

### 📊 Research & Analysis Tools
*   **`analyze_60day_weekly.py`**: Specific backtest focused on the last 2 months.
*   **`analyze_spread.py`**: Studies the "Spread" (gap) between the Q-values of Buy vs Sell. A wider spread = higher confidence.
*   **`analyze_confidence.py`**: Visualizes how "sure" the model has been over time.
*   **`analyze_decision_streaks.py`**: Checks if the model gets stuck in "Buy" or "Sell" loops too often.
*   **`analyze_weekly_fridays.py`**: Research script investigating if Friday-only trading performed better.

### 💾 Data Files
*   **`qqq_market_data.csv`**: Deep historical data used for training.
*   **`qqq_data_60days.csv`**: rolling window of recent data used for daily inference.

---

## 🛠 Setup & Requirements

### Prerequisites
*   Python 3.8+
*   Packages: `pip install pandas yfinance torch numpy pandas_datareader matplotlib`

### GitHub Workflow
To update the repo with your latest daily recommendations:
```bash
git add .
git commit -m "Update model results for $(date +%Y-%m-%d)"
git push
```
