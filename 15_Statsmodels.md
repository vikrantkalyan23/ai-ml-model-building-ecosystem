# Statsmodels

## 1. What is Statsmodels?

**Statsmodels** is a Python library for statistical modeling.

It is used when you need statistical tests, interpretable regression results, confidence intervals, p-values, and time-series models.

> **Simple definition:** Statsmodels helps you understand relationships in data using statistics, not just make predictions.

---

## 2. Basic Information

| Item | Details |
|---|---|
| **Name** | Statsmodels |
| **Type** | Statistical modeling library |
| **Best for** | Statistics, regression analysis, time series |
| **Common users** | Analysts, economists, researchers, data scientists |
| **Main strength** | Detailed statistical summaries |
| **Common models** | OLS, GLM, ARIMA, SARIMAX |

---

# 3. What Problem Does Statsmodels Solve?

Scikit-learn focuses mostly on prediction.

Statsmodels focuses on statistical understanding.

```text
Question:
    Does advertising spend affect sales?

Statsmodels helps answer:
    coefficient
    p-value
    confidence interval
    model summary
```

Example:

```text
Salary = base + age effect + experience effect + education effect
```

Statsmodels helps estimate and explain those effects.

---

# 4. OLS Regression

OLS means Ordinary Least Squares.

It is a classic linear regression method.

```python
import statsmodels.api as sm

X = sm.add_constant(X)

model = sm.OLS(y, X)
results = model.fit()

print(results.summary())
```

The summary includes:

- Coefficients
- Standard errors
- t-values
- p-values
- R-squared
- Confidence intervals

---

# 5. Formula API

Statsmodels can use formula syntax.

```python
import statsmodels.formula.api as smf

model = smf.ols("sales ~ advertising + price", data=df)
results = model.fit()

print(results.summary())
```

Formula meaning:

```text
sales ~ advertising + price

sales is predicted using advertising and price.
```

---

# 6. Important Statistical Terms

| Term | Meaning |
|---|---|
| **Coefficient** | Estimated effect of a feature |
| **p-value** | Evidence against no effect |
| **Confidence interval** | Likely range for coefficient |
| **R-squared** | Variance explained by model |
| **Residual** | Actual value minus predicted value |
| **OLS** | Ordinary Least Squares regression |

---

# 7. Time-Series Models

Statsmodels also supports time-series analysis.

Common models:

| Model | Used for |
|---|---|
| **ARIMA** | Forecasting non-seasonal time series |
| **SARIMAX** | Seasonal/exogenous time-series forecasting |
| **Exponential smoothing** | Trend/seasonal smoothing |
| **VAR** | Multivariate time series |

Example:

```python
from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(y, order=(1, 1, 1))
results = model.fit()

forecast = results.forecast(steps=5)
print(forecast)
```

---

# 8. Statsmodels vs Scikit-learn

| Feature | Statsmodels | Scikit-learn |
|---|---|---|
| Main goal | Statistical analysis | Prediction |
| Model summary | Excellent | Limited |
| p-values | Yes | Usually no |
| Confidence intervals | Yes | Usually no |
| ML algorithms | Limited | Many |
| Production ML pipelines | Limited | Strong |

---

# 9. Advantages

- Excellent statistical summaries
- Great for interpretable regression
- Supports p-values and confidence intervals
- Useful for econometrics and research
- Good time-series models
- Formula syntax is convenient

---

# 10. Disadvantages

- Not focused on modern ML algorithms
- Less convenient for production ML pipelines
- Not for deep learning
- Can be harder for beginners without statistics background

---

# 11. When to Use Statsmodels

Use Statsmodels when:

```text
You need statistical interpretation
You need p-values and confidence intervals
You are doing regression analysis
You need econometrics-style modeling
You need ARIMA/SARIMAX time-series models
```

Avoid it when:

```text
You only need best predictive accuracy
You need Random Forest, SVM, KNN, or neural networks
You need deep learning
```

---

# 12. Quick Revision Table

| Topic | Meaning |
|---|---|
| **Statsmodels** | Statistical modeling library |
| **OLS** | Linear regression by least squares |
| **Coefficient** | Feature effect estimate |
| **p-value** | Statistical significance signal |
| **Residual** | Prediction error |
| **ARIMA** | Time-series forecasting model |
| **SARIMAX** | Seasonal time-series model |

---

# 13. Final Summary

```text
Statsmodels is for statistical understanding.

Use it for:
    - Regression analysis
    - p-values
    - Confidence intervals
    - Statistical summaries
    - Time-series models

Scikit-learn predicts.
Statsmodels explains.
```
