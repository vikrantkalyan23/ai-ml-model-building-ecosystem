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
