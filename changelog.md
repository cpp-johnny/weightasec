# Changelog

---

## v1.0
**Prototype / Exploration**

### Core Additions
*   **Basic Dashboard:** Initial Streamlit implementation.
*   **Visualizations:** Raw time series plotting (weight over time).
*   **Smoothing:** Simple moving averages (MA7, MA30).
*   **Analysis:** Basic linear trend via OLS regression.

### Limitations
*   No proper data validation or model evaluation framework.
*   No uncertainty handling or separation between signal and noise.

---

## v2.0
**Performance & Statistics Major Upgrades**

### Major Upgrades
*   **Optimization:** Rolling OLS vectorized via NumPy ($O(n)$ vs $O(n^2)$ loops).
*   **Parameter Tuning:** Holt $\alpha/\beta$ fitted via SSE minimization (`scipy.optimize`) instead of hardcoded values.
*   **Uncertainty:** Removed bootstrap PI damping factors for statistically honest bands.
*   **Refactoring:** Moved `_chip()` and all helper functions to the top of the file for better readability.

### Fixes
*   Removed hardcoded smoothing constants.
*   Improved performance through vectorization over Python loops.

### Still Missing
*   No model comparison logic or error measurement.
*   No validation framework.

---

## v3.0
**Robustness & Edge Case Handling**

### Fixes
*   **Slope Stability:** Added "stable" guard at $<0.01$ kg/wk threshold to prevent near-zero slope explosion.
*   **Data Integrity:** Implemented `fillna(0)` on Delta NaN edge cases to prevent silent propagation.
*   **Bootstrap Logic:** Capped residual window at `min(30, n//2)` for short time series.
*   **Alignment:** Aligned bootstrap residuals strictly to the active projection model.

### Improvements
*   Enhanced handling of short or noisy datasets.
*   Smoother trend behavior in edge cases.

### Still Missing
*   No baseline comparison or predictive accuracy measurement.

---

## v4.0
**Validation Framework & Model Integrity**

### Major Upgrades
*   **Evaluation:** Added Walk-forward 1-step MAE to eliminate in-sample optimism.
*   **Benchmarking:** Introduced Naive baseline ($y_t = y_{t-1}$) to establish an irreducible noise floor.
*   **Comparative Metrics:** Side-by-side MAE display (Naive vs. OLS vs. Holt) to surface model performance.
*   **Data Quality:** New "Data Quality" card tracking gap counts, sample size, and variance stability with a 3-tier color signal (Green/Amber/Red).
*   **Transparency:** Added "Model Integrity" tab and residuals-over-time charts for both models.

### UX & UI Improvements
*   **Constraints:** Reduced projection slider max from 90 to 60 days (limiting noise-heavy long-term forecasts).
*   **Context:** Added PI captions naming residual sources and explaining interval widening.

---

## v5.0 (working on it)

might consider addiing Kalman Filter and heuristics like "youre gaining most weight over weekennds, weight spike during xxx"