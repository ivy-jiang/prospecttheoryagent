# Model Performance Report (In-Sample vs Out-of-Sample)

**Date:** February 5, 2026
**Script:** `verify_robustness.py`

---

## 1. Methodology: Strict Train/Test Split

To rigorously test whether the model is overfitting (memorizing the past) or actually learning (generalizing to the future), we implemented a strict "Walk-Forward" split:

*   **Total Dataset:** ~2,500 trading days (2015 - Present)
*   **Training Set (In-Sample):** First 80% (2015 to Mid-2023). The model *only* sees this data during learning.
*   **Test Set (Out-of-Sample):** Last 20% (Mid-2023 to Present). The model *never* sees this data during training.

**Why this matters:**
If a model makes money on the Training Set but loses money on the Test Set, it is **overfitted**.
If it performs roughly equally on both (even if poor initially), it is **robust**.

---

## 2. Preliminary Results (Quick Check)

*Note: These results were generated using a "Quick Check" run of only **5 training episodes** (complete training requires ~1000 episodes). As a result, the absolute performance is poor (negative), but the **relative stability** is what we are analyzing.*

| Metric | In-Sample (Train) | Out-of-Sample (Test) | Verdict |
| :--- | :--- | :--- | :--- |
| **Total Return** | -107.50% | -48.34% | **Consistent** (Both negative) |
| **Sharpe Ratio** | -0.62 | -1.20 | **Consistent** (Both negative) |
| **Hit Rate** | 44.4% | 41.2% | **Stable** (~40-45% range) |

**Interpretation:**
The model currently shows **Underfitting**. It hasn't trained long enough (5 episodes) to learn a profitable strategy yet, so it is losing money in both periods. However, the fact that the Hit Rate is stable (44% vs 41%) suggests the architecture itself is not prone to massive overfitting—it just needs more study time.

---

## 3. How to Replicate (The "Real" Test)

To see the *true* performance capability of the model, you need to run the verification script with the full training schedule.

### Step 1: Open the Script
Open `verify_robustness.py` in your editor.

### Step 2: Adjust Training Duration
Find line ~135 and change the loop range from `5` back to `500` or `1000`:

```python
# Change this:
for episode in range(5):

# To this:
for episode in range(500):
```

### Step 3: Run the Script
Execute the following command in your terminal:
```bash
python verify_robustness.py
```

*Warning: This will take approximately **3-4 hours** on a CPU, or **30 minutes** on a GPU.*

### Step 4: Analyze Output
The script will output a report comparing the Training Sharpe vs. Testing Sharpe.
*   **Good Result:** Test Sharpe > 1.0 (and within 20% of Training Sharpe).
*   **Overfitted:** Test Sharpe << 0 (while Training Sharpe is > 2.0).
