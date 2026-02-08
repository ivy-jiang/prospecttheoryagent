# Prospect Theory Dual-Stream Q-Network: Model Documentation

**Date:** February 5, 2026
**Author:** Antigravity (Agentic AI)
**Project:** Prospect Theory Agent

---

## 1. Model Overview & Design Philosophy

This trading system utilizes a **Deep Q-Network (DQN)** enhanced with **Prospect Theory** layers. Unlike standard RL agents that maximize expected objective value ($E[x]$), this agent maximizes **Subjective Utility** ($\sum w(p) \cdot v(x)$), mimicking how human traders perceive risk and reward (e.g., loss aversion, probability weighting).

The model is designed as a **Model-Based RL** agent, where it explicitly learns to predict the probability distribution of future market moves ("outcomes") before applying subjective value judgments.

---

## 2. Input Variables & Feature Engineering

The model takes a **20-day lookback window** of market data.

### 2.1 Raw Inputs
The raw data is sourced from `yfinance` and FRED (Federal Reserve Economic Data):
*   **QQQ (Price)**: Nasdaq-100 ETF derived price.
*   **^VIX**: CBOE Volatility Index (Market Sentiment).
*   **^TNX**: 10-Year Treasury Yield.
*   **DGS2 / ^FVX**: 2-Year Treasury Yield (Short-term risk-free rate proxy).

### 2.2 Feature Transformations
Neural networks require stationary, normalized data to learn effectively. We apply the following transformations:

| Feature | Transformation | Reasoning |
| :--- | :--- | :--- |
| **Price** | $\ln(P_t / P_{t-1})$ (Log Returns) | **Stationarity**. Raw prices are unbounded and non-stationary. Returns capture the *rate of change*, which is statistically stable. |
| **RSI** | 14-Day Calculation | **Momentum**. Normalized 0-100 scale indicating overbought/oversold conditions. |
| **MACD** | EMA(12) - EMA(26) | **Trend Following**. Captures convergence/divergence of moving averages. |
| **Moving Avg** | 100-Day SMA (Z-Scored) | **Trend Baseline**. The raw moving average is standardized to indicate the relative level of the trend line within the dataset. |
| **Yields/VIX** | Z-Score Normalization | **Scaling**. These inputs are on different scales (e.g., VIX ~20, Yield ~0.04). |

### 2.3 Normalization
**Method:** Z-Score Standardization ($z = \frac{x - \mu}{\sigma}$)
*   **Applied to:** All input features (Returns, RSI, MACD, MA, VIX, Yields).
*   **Excluded:** The raw `price` column (which is fed separately to the network for absolute price context).
*   **Reasoning:** Deep learning networks fail if one input is 0.01 (Returns) and another is 600 (MA). Z-scoring forces all inputs into the same ~(-3 to +3) range, ensuring the optimizer treats them with equal importance.

---

## 3. Model Architecture

The network is a **Dual-Stream DQN** with custom Prospect Theory heads.

### 3.1 Network Structure
1.  **Input Layer (The "State")**:
    *   The state vector is a concatenation ($S = [F, P]$) of two distinct parts:
        *   **Part A: Features ($F$)**: The Z-Scored variables (Returns, RSI, etc.). These provide the **Pattern** (Relative Market Shape).
        *   **Part B: Context ($P$)**: The Raw Prices ($). These provide the **Scale** (Absolute Dollar Magnitude).
    *   **Why?** The Features tell the model *if* it should buy (Technical Analysis), while the Absolute Price helps the Utility Layer calculate *how much* joy/pain a % move will cause in dollar terms (Prospect Theory).
2.  **"World Model" (Core Layers)**:
    *   **Layer 1**: Linear (Input $\to$ 128) + ReLU
    *   **Layer 2**: Linear (128 $\to$ 128) + ReLU
    *   **Output**: Projects to `Num_Outcomes × 2` (Outcomes + Probabilities).
3.  **Prospect Theory Heads**:
    *   **Utility Layer ($v(x)$)**: Small MLP (1 $\to$ 32 $\to$ 1) that learns the value function. It likely learns a convex/concave shape to represent risk preferences (e.g., loss aversion).
    *   **Probability Weighting Layer ($w(p)$)**: Small MLP (1 $\to$ 32 $\to$ 1) + Sigmoid. Learns to overweight/underweight tail probabilities.
4.  **Aggregation**:
    *   $SQ(s, a) = \sum_{k} w(p_k) \cdot v(x_k)$
    *   The model chooses the action that maximizes this **Subjective** Q-Value.

### 3.2 Engineering & Hyperparameter Rationale
Specific technical choices were made to balance model capacity with stability:

| Parameter | Value | Decision Rationale |
| :--- | :--- | :--- |
| **Hidden Units** | 128 | **Capacity vs. Overfitting**. The input dimension is ~140 (20 days * 7 features). 128 neurons is a heuristic rule-of-thumb (approx input size) to allow sufficient representation power without creating a massive "memory bank" that memorizes noise. |
| **Activation** | ReLU | **Gradient Flow**. Rectified Linear Units identify non-linearities efficiently and avoid the "vanishing gradient" problem common with Sigmoid/Tanh in deep networks. |
| **Optimizer** | Adam | **Adaptive Learning**. Financial data is noisy. Adam adapts the learning rate for each parameter, ensuring stable convergence even when some features (like VIX) fluctuate more than others. |
| **Gamma** | 0.99 | **Long-Term Focus**. A discount factor of 0.99 tells the agent that a reward 10 days from now is almost as valuable as a reward today. This prevents short-sighted "scalping" and encourages holding for trends. |
| **Window Size** | 20 | **Monthly Context**. 20 trading days $\approx$ 1 calendar month. This captures enough history to see a short-term trend formation (Momentum) without carrying stale data from months ago. |

---

## 4. Performance & Backtesting Methodology

### 4.1 Testing Protocol
To test for **overfitting**, we use a strict Time-Series Split:
*   **Training Set**: Historical data (e.g., 2015-2023).
*   **Test Set**: Out-of-sample data (e.g., 2024-Present).

**Signs of Overfitting:**
*   **High Variance**: Training Sharpe Ratio >> Test Sharpe Ratio.
*   **Pattern Memorization**: The model performs perfectly on specific historical dates but fails on new data with similar statistical properties.

### 4.2 Performance Metrics
We evaluate the model using the following hierarchy of metrics:

1.  **Sharpe Ratio** (Winner): $\frac{R_p - R_f}{\sigma_p}$
    *   The "Gold Standard". It penalizes volatility. A high PnL with massive drawdowns is inferior to a moderate PnL with a smooth curve.
2.  **Profit Factor**: $\frac{\text{Gross Profit}}{\text{Gross Loss}}$
    *   Measures the efficiency of the strategy. > 1.5 is good; > 2.0 is excellent.
3.  **Hit Rate (Accuracy)**:
    *   $\frac{\text{Winning Trades}}{\text{Total Trades}}$.
    *   *Caveat*: A 40% hit rate is acceptable if the winning trades are 3x larger than losing trades (Trend Following).
4.  **Total PnL (Equity Curve)**:
    *   Absolute return. Important, but meaningless without considering risk (Drawdown).

### 4.3 Sizing Assumptions
In the current backtesting framework (`backtest_full_history.py`):
*   **Sizing**: Fixed Fractional (100% Equity).
    *   When a **BUY** signal is generated, **100%** of available cash is converted to shares.
    *   When a **SELL** (Short) signal is generated, the portfolio flips to **100% Short** (modeled as Inverse Return).
*   **No Partial Scaling**: The model outputs discrete actions (Buy/Sell/Hold), not continuous weights (0.5, 0.2). This is a "Binary" sizing approach.

---

## 5. Professional Quant Assessment

**Strengths:**
*   **Novelty**: Integrating Prospect Theory makes the agent "psychologically aware," potentially allowing it to exploit human biases in the market (e.g., panic selling).
*   **Regime Adaptation**: By using Rolling Windows, the inputs carry context about volatility regimes (High VIX vs Low VIX).

**Weaknesses / Areas for Improvement:**
*   **Discrete Action Space**: Institutional models usually output target *weights* (e.g., +0.8 exposure) rather than binary Buy/Sell.
*   **Transaction Costs**: Current backtests likely assume 0 slippage and 0 commission. This inflates performance of high-frequency strategies.
*   **Stationarity Risk**: While returns are stationary, correlations (e.g., Stock vs Bond yields) define market regimes. The model may need periodic retraining to adapt to new correlation regimes (e.g., 2022 Inflation vs 2019 Low Rates).

**Verdict:**
To trust this model for capital allocation, we require a **Forward Test (Paper Trading)** period of 3-6 months to verify that the **Test Set Sharpe Ratio** holds up in live conditions.
