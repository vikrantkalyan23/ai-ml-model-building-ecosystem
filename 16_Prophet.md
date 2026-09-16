# Prophet

## 1. What is Prophet?

**Prophet** is a forecasting library designed for time-series data.

It is commonly used for business forecasting where the data has trends, seasonality, and holidays.

> **Simple definition:** Prophet predicts future values from historical time-series data with an easy forecasting workflow.

---

## 2. Basic Information

| Item | Details |
|---|---|
| **Name** | Prophet |
| **Type** | Time-series forecasting library |
| **Best for** | Business forecasting |
| **Input data** | Date column + value column |
| **Required columns** | ds and y |
| **Handles** | Trend, seasonality, holidays |
| **Common tasks** | Sales, traffic, demand, revenue forecasting |

---

# 3. What Problem Does Prophet Solve?

Many businesses need to predict future values.

Examples:

```text
Past sales       -> future sales
Website traffic  -> future traffic
Demand history   -> future demand
Revenue history  -> future revenue
```

Prophet expects data like this:

```text
ds          y
2024-01-01  100
2024-01-02  120
2024-01-03  115
```

`ds` means date/time.

`y` means value to forecast.

---

# 4. Forecasting Flow

```text
Historical time series
        |
        v
Fit Prophet model
        |
        v
Create future dates
        |
        v
Predict future values
        |
        v
Forecast chart/table
```

---

# 5. Basic Example

```python
from prophet import Prophet

model = Prophet()
model.fit(df)

future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

print(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail())
```

Important output columns:

| Column | Meaning |
|---|---|
| **ds** | Date |
| **yhat** | Forecast value |
| **yhat_lower** | Lower uncertainty bound |
| **yhat_upper** | Upper uncertainty bound |

---

# 6. Trend and Seasonality

Prophet breaks forecasting into understandable parts.

```text
Forecast = trend + seasonality + holidays + error
```

Examples:

```text
Trend:
    Sales are increasing over years

Weekly seasonality:
    More traffic on weekends

Yearly seasonality:
    More sales during holidays
```

---

# 7. Plotting Forecasts

```python
fig1 = model.plot(forecast)
fig2 = model.plot_components(forecast)
```

Component plots can show:

- Trend
- Weekly seasonality
- Yearly seasonality
- Holiday effects

---

# 8. Holidays

Prophet can include holiday effects.

```python
import pandas as pd
from prophet import Prophet

holidays = pd.DataFrame({
    "holiday": ["sale_day", "sale_day"],
    "ds": pd.to_datetime(["2024-11-01", "2024-12-01"]),
    "lower_window": [0, 0],
    "upper_window": [1, 1],
})

model = Prophet(holidays=holidays)
model.fit(df)
```

---

# 9. Prophet vs ARIMA

| Feature | Prophet | ARIMA |
|---|---|---|
| Ease of use | Easier | More statistical setup |
| Seasonality | Built in | Must configure carefully |
| Holidays | Built in | Manual |
| Business forecasting | Excellent | Good |
| Statistical control | Medium | High |

---

# 10. Advantages

- Easy to use
- Good for business forecasting
- Handles trend and seasonality
- Supports holidays
- Gives uncertainty intervals
- Good visualization support

---

# 11. Disadvantages

- Not always best for every time series
- Needs clean historical data
- Not for general ML classification/regression
- Struggles with very irregular data
- Advanced forecasting may need more specialized models

---

# 12. When to Use Prophet

Use Prophet when:

```text
You have date/time data
You need future forecasts
Data has trend or seasonality
You want simple business forecasting
```

Avoid it when:

```text
You need image, text, or tabular classification
You do not have time-based data
You need low-level statistical control
```

---

# 13. Quick Revision Table

| Topic | Meaning |
|---|---|
| **Prophet** | Time-series forecasting library |
| **ds** | Date/time column |
| **y** | Value to forecast |
| **yhat** | Predicted value |
| **Trend** | Long-term direction |
| **Seasonality** | Repeating pattern |
| **Holiday effect** | Special date impact |

---

# 14. Final Summary

```text
Prophet is for forecasting future values.

Use it for:
    - Sales forecasting
    - Traffic forecasting
    - Demand forecasting
    - Business time series

Required columns:
    ds = date
    y  = value
```

---

# 15. Prophet Model Components

Prophet models time series using components.

```text
y(t) = trend(t) + seasonality(t) + holidays(t) + error
```

Simple meaning:

```text
trend:
    long-term increase or decrease

seasonality:
    repeating pattern

holidays:
    special date effects

error:
    noise the model cannot explain
```

This makes Prophet easier to interpret than many black-box forecasting models.

---

# 16. Trend

Trend is the long-term direction.

Examples:

```text
Sales slowly increasing
Website traffic declining
Revenue growing after product launch
```

Prophet can model trend changes.

```text
Before launch:
    slow growth

After launch:
    faster growth
```

These trend-change points are called changepoints.

---

# 17. Seasonality

Seasonality means repeating patterns.

Examples:

```text
Daily:
    traffic peaks at 8 PM

Weekly:
    lower sales on weekends

Yearly:
    higher demand during holidays
```

Prophet can include:

```python
model = Prophet(
    daily_seasonality=True,
    weekly_seasonality=True,
    yearly_seasonality=True
)
```

---

# 18. Forecast Uncertainty

Prophet returns uncertainty intervals.

```text
yhat_lower  <= likely forecast range <= yhat_upper
```

Example:

```text
yhat       = 100
yhat_lower = 85
yhat_upper = 120
```

Meaning:

```text
The model predicts around 100,
but reasonable future values may fall between 85 and 120.
```

Forecast uncertainty usually increases further into the future.

---

# 19. Regressors

Prophet can use extra variables.

Example:

```text
sales may depend on:
    date
    discount
    ad_spend
    holiday
```

Code:

```python
model = Prophet()
model.add_regressor("ad_spend")
model.add_regressor("discount")

model.fit(df)
```

The future dataframe must also include future values for those regressors.

---

# 20. Evaluation

Do not judge a forecast only by looking at the chart.

Use backtesting.

```text
Train on older data
Predict newer period
Compare forecast with actual values
```

Common metrics:

| Metric | Meaning |
|---|---|
| **MAE** | Average absolute error |
| **RMSE** | Penalizes large errors |
| **MAPE** | Percentage error |

---

# 21. Common Mistakes

| Mistake | Problem | Fix |
|---|---|---|
| Wrong column names | Prophet fails | Use `ds` and `y` |
| Missing future regressor values | Cannot predict correctly | Provide future regressor data |
| Too little history | Weak seasonality learning | Use more historical data |
| Ignoring outliers | Distorted forecast | Clean or mark outliers |
| Forecasting too far ahead | High uncertainty | Keep horizon realistic |

---

# 22. Practical Workflow

```text
1. Prepare ds/y dataframe
2. Plot data
3. Clean missing values and outliers
4. Fit baseline Prophet model
5. Add holidays/seasonality if needed
6. Backtest forecast
7. Tune changepoint and seasonality settings
8. Create future dataframe
9. Predict and explain components
```
