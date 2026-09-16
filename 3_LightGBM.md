# LightGBM

## 1. What is LightGBM?

**LightGBM** stands for **Light Gradient Boosting Machine**.

LightGBM is a machine-learning library for fast gradient-boosted decision trees. It is especially strong for large structured/tabular datasets.

> **Simple definition:** LightGBM builds many decision trees step by step, but it is designed to train very fast and handle large data efficiently.

---

## 2. Basic Information

| Item | Details |
|---|---|
| **Name** | LightGBM |
| **Full form** | Light Gradient Boosting Machine |
| **Type** | Gradient boosting library |
| **Best for** | Large tabular datasets |
| **Main algorithm** | Gradient-boosted decision trees |
| **Python support** | Yes |
| **R support** | Yes |
| **GPU support** | Available on supported setups |
| **Common tasks** | Regression, classification, ranking |

---

# 3. What Problem Does LightGBM Solve?

LightGBM helps when you need accurate models on structured data, especially when the dataset is large.

Example:

```text
Customer age, income, visits, purchases
              |
              v
           LightGBM
              |
              v
      Churn: Yes / No
```

It is often used in:

- Fraud detection
- Credit scoring
- Customer churn
- Sales prediction
- Search ranking
- Competition tabular ML

---

# 4. Main Idea

LightGBM uses boosting.

```text
Dataset
   |
   v
Tree 1 learns first pattern
   |
   v
Tree 2 improves remaining error
   |
   v
Tree 3 improves again
   |
   v
Final strong model
```

Unlike many tree algorithms that grow level by level, LightGBM commonly grows trees leaf-wise.

```text
Level-wise:
    split all nodes level by level

Leaf-wise:
    split the leaf that gives the biggest improvement
```

This can make LightGBM fast and accurate, but it also means tuning is important to avoid overfitting.

---

# 5. Important Features

| Feature | Meaning |
|---|---|
| **Fast training** | Designed for speed on large datasets |
| **Low memory usage** | Efficient data handling |
| **Leaf-wise growth** | Can improve accuracy quickly |
| **Categorical support** | Can handle categorical features with proper setup |
| **Parallel learning** | Supports efficient computation |
| **Ranking support** | Useful for search and recommendation ranking |

---

# 6. Regression Example

```python
import lightgbm as lgb
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

X, y = load_diabetes(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = lgb.LGBMRegressor(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

print(mean_absolute_error(y_test, predictions))
```

---

# 7. Classification Example

```python
import lightgbm as lgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = lgb.LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

print(accuracy_score(y_test, predictions))
```

---

# 8. Important Hyperparameters

| Hyperparameter | Meaning |
|---|---|
| **n_estimators** | Number of boosting trees |
| **learning_rate** | Step size for each tree |
| **num_leaves** | Maximum number of leaves per tree |
| **max_depth** | Maximum tree depth |
| **min_child_samples** | Minimum samples in a leaf |
| **subsample** | Fraction of rows used |
| **colsample_bytree** | Fraction of columns used |
| **reg_alpha** | L1 regularization |
| **reg_lambda** | L2 regularization |

Simple tuning idea:

```text
More trees + smaller learning rate = often better, but slower
Too many leaves = possible overfitting
Regularization = helps control overfitting
```

---

# 9. LightGBM vs XGBoost

| Feature | LightGBM | XGBoost |
|---|---|---|
| Speed | Usually very fast | Fast |
| Tree growth | Often leaf-wise | Usually level-wise style |
| Large data | Excellent | Excellent |
| Tuning sensitivity | High | High |
| Beginner ease | Medium | Medium |
| Tabular accuracy | Excellent | Excellent |

---

# 10. Advantages

- Very fast on large tabular datasets
- Strong accuracy
- Good for regression and classification
- Supports ranking tasks
- Efficient memory usage
- Works well in competitions and business ML

---

# 11. Disadvantages

- Can overfit if not tuned carefully
- Not for deep learning
- Not for image, audio, or LLM training
- Some categorical-feature handling requires care
- Small datasets may not always benefit from it

---

# 12. When to Use LightGBM

Use LightGBM when:

```text
You have structured/tabular data
You need high accuracy
You have many rows
You want faster training than many other boosting methods
```

Avoid it when:

```text
You need neural networks
You are training large image models
You are building LLMs
Your dataset is tiny and simple
```

---

# 13. Quick Revision Table

| Topic | Meaning |
|---|---|
| **LightGBM** | Fast gradient boosting library |
| **Best for** | Large tabular data |
| **Main model type** | Boosted decision trees |
| **Regression class** | LGBMRegressor |
| **Classification class** | LGBMClassifier |
| **Key strength** | Speed and accuracy |
| **Main risk** | Overfitting if poorly tuned |

---

# 14. Final Summary

```text
LightGBM is a fast and powerful gradient boosting library.

Use it for:
    - Tabular data
    - Large datasets
    - Regression
    - Classification
    - Ranking

Remember:
    Fast training is its major strength.
    Careful tuning is important.
```
