# CatBoost

## 1. What is CatBoost?

**CatBoost** is a gradient boosting library created to work especially well with categorical features.

The name comes from:

```text
CatBoost = Categorical Boosting
```

> **Simple definition:** CatBoost builds boosted decision trees and can handle categorical columns more naturally than many other ML libraries.

---

## 2. Basic Information

| Item | Details |
|---|---|
| **Name** | CatBoost |
| **Type** | Gradient boosting library |
| **Best for** | Tabular data with categorical features |
| **Main algorithm** | Gradient-boosted decision trees |
| **Python support** | Yes |
| **R support** | Yes |
| **GPU support** | Available on supported setups |
| **Common tasks** | Regression, classification, ranking |

---

# 3. What Problem Does CatBoost Solve?

Many real datasets contain categorical columns.

Example:

```text
City       Gender     Device      Purchased
Delhi      Male       Mobile      Yes
Mumbai     Female     Desktop     No
Pune       Female     Mobile      Yes
```

Many ML models need these categories converted into numbers first.

CatBoost can directly work with categorical columns when you tell it which columns are categorical.

```text
Raw categorical data
        |
        v
     CatBoost
        |
        v
Prediction
```

---

# 4. Main Idea

CatBoost uses boosting.

```text
Tree 1 -> makes prediction
Tree 2 -> improves errors
Tree 3 -> improves errors again
...
Final model -> strong prediction
```

Its special strength is categorical feature handling.

```text
Normal workflow:
    Encode categories manually -> train model

CatBoost workflow:
    Mark categorical columns -> train model
```

---

# 5. Important Features

| Feature | Meaning |
|---|---|
| **Categorical feature handling** | Works well with text categories |
| **Ordered boosting** | Helps reduce target leakage and overfitting |
| **Strong defaults** | Often works well without heavy tuning |
| **Regression/classification/ranking** | Supports common ML tasks |
| **GPU support** | Can train faster on supported hardware |
| **Feature importance** | Helps explain useful features |

---

# 6. Regression Example

```python
from catboost import CatBoostRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

X, y = load_diabetes(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = CatBoostRegressor(
    iterations=100,
    learning_rate=0.1,
    depth=6,
    verbose=0,
    random_seed=42
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

print(mean_absolute_error(y_test, predictions))
```

---

# 7. Classification Example

```python
from catboost import CatBoostClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = CatBoostClassifier(
    iterations=100,
    learning_rate=0.1,
    depth=6,
    verbose=0,
    random_seed=42
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

print(accuracy_score(y_test, predictions))
```

---

# 8. Categorical Feature Example

```python
from catboost import CatBoostClassifier

X = [
    ["Delhi", "Mobile", 25],
    ["Mumbai", "Desktop", 31],
    ["Delhi", "Mobile", 22],
    ["Pune", "Tablet", 40],
]
y = [1, 0, 1, 0]

cat_features = [0, 1]

model = CatBoostClassifier(verbose=0)
model.fit(X, y, cat_features=cat_features)

print(model.predict([["Delhi", "Mobile", 28]]))
```

---

# 9. Important Hyperparameters

| Hyperparameter | Meaning |
|---|---|
| **iterations** | Number of boosting rounds |
| **learning_rate** | Step size |
| **depth** | Tree depth |
| **l2_leaf_reg** | L2 regularization |
| **loss_function** | Objective to optimize |
| **cat_features** | Categorical column indexes/names |
| **random_seed** | Reproducibility |

---

# 10. CatBoost vs LightGBM vs XGBoost

| Feature | CatBoost | LightGBM | XGBoost |
|---|---|---|---|
| Categorical features | Excellent | Good with setup | Usually needs encoding |
| Speed | Fast | Very fast | Fast |
| Default performance | Strong | Strong | Strong |
| Tuning difficulty | Medium | Medium/high | Medium/high |
| Tabular data | Excellent | Excellent | Excellent |

---

# 11. Advantages

- Excellent for categorical data
- Strong default settings
- Good accuracy on tabular datasets
- Supports regression, classification, and ranking
- Can reduce manual preprocessing
- Provides feature importance tools

---

# 12. Disadvantages

- Can be slower than LightGBM on some very large datasets
- Still requires tuning for best results
- Not for deep learning or LLM training
- Model files can become large
- Categorical columns must be handled carefully

---

# 13. When to Use CatBoost

Use CatBoost when:

```text
Your data has many categorical columns
You want strong tabular ML performance
You want good default behavior
You need regression or classification
```

Avoid it when:

```text
You need neural networks
You are training computer vision models
You are fine-tuning LLMs
Your data is mostly unstructured text/images
```

---

# 14. Quick Revision Table

| Topic | Meaning |
|---|---|
| **CatBoost** | Gradient boosting library |
| **Best for** | Tabular data with categories |
| **Regression class** | CatBoostRegressor |
| **Classification class** | CatBoostClassifier |
| **Key strength** | Categorical feature handling |
| **Main risk** | Needs tuning and correct categorical setup |

---

# 15. Final Summary

```text
CatBoost is a powerful boosting library for tabular data.

Its biggest strength:
    categorical feature handling

Use it for:
    - Regression
    - Classification
    - Ranking
    - Business/tabular ML data
```
