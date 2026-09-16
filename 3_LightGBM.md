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

---

# 15. Decision Trees, Boosting, and LightGBM

To understand LightGBM properly, first understand three ideas:

```text
Decision Tree
    |
    v
Gradient Boosting
    |
    v
LightGBM optimization
```

## Decision tree

A decision tree predicts by asking questions.

```text
Income > 50000?
      /       \
    Yes        No
    /           \
Age > 30?     Low value
  /    \
High   Medium
```

One tree is easy to understand but can overfit.

## Gradient boosting

Gradient boosting builds many weak trees one after another.

```text
Tree 1 -> first prediction
Tree 2 -> fixes remaining mistakes
Tree 3 -> fixes remaining mistakes
...
Final model -> combination of all trees
```

LightGBM is a highly optimized implementation of this idea.

---

# 16. Leaf-wise Growth

Many tree algorithms grow level by level.

```text
Level-wise growth:

        root
       /    \
      A      B
     / \    / \
    C   D  E   F
```

LightGBM commonly uses leaf-wise growth.

```text
Leaf-wise growth:

        root
       /    \
      A      B
            / \
           C   D
              / \
             E   F
```

It chooses the leaf that gives the largest loss reduction.

### Why this matters

| Benefit | Risk |
|---|---|
| Can improve accuracy quickly | Can overfit if trees become too deep |
| Often faster on large data | Needs `num_leaves`, `max_depth`, and regularization |

Good beginner rule:

```text
If LightGBM overfits:
    reduce num_leaves
    reduce max_depth
    increase min_child_samples
    add regularization
```

---

# 17. Histogram-based Learning

LightGBM does not always evaluate every possible split value directly.

It groups continuous values into bins.

```text
Raw values:
12, 13, 14, 25, 26, 40, 42

Bins:
0-20, 21-35, 36-50
```

This makes training faster and reduces memory usage.

```text
Many exact values
      |
      v
Smaller number of bins
      |
      v
Faster split search
```

---

# 18. Categorical Features in LightGBM

LightGBM can handle categorical features, but you must prepare them correctly.

Typical options:

```text
Option 1: Use pandas category dtype
Option 2: Pass categorical_feature parameter
Option 3: Encode categories carefully
```

Example idea:

```python
import pandas as pd
import lightgbm as lgb

df["city"] = df["city"].astype("category")

model = lgb.LGBMClassifier()
model.fit(X_train, y_train, categorical_feature=["city"])
```

Be careful:

```text
Do not turn categories into fake ordered numbers unless that order is real.

Bad:
    Delhi = 1, Mumbai = 2, Pune = 3

The model may think Pune > Mumbai > Delhi.
```

---

# 19. LightGBM Training API Styles

LightGBM has two common styles.

## Scikit-learn style

Best for beginners.

```python
from lightgbm import LGBMClassifier

model = LGBMClassifier()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

## Native LightGBM style

More flexible for advanced training.

```python
import lightgbm as lgb

train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_test, label=y_test)

params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.05,
}

model = lgb.train(
    params,
    train_data,
    valid_sets=[valid_data],
    num_boost_round=100
)
```

---

# 20. Common Mistakes

| Mistake | Why it is a problem | Fix |
|---|---|---|
| Too many leaves | Overfitting | Reduce `num_leaves` |
| High learning rate | Unstable learning | Lower `learning_rate` |
| No validation set | Cannot detect overfitting | Use validation/cross-validation |
| Wrong categorical encoding | Misleading patterns | Use category dtype or proper encoding |
| Tuning only one parameter | Boosting parameters interact | Tune learning rate, leaves, depth, samples |

---

# 21. Practical Tuning Order

```text
1. Start with a simple baseline
2. Choose metric
3. Set validation split
4. Tune learning_rate and n_estimators
5. Tune num_leaves and max_depth
6. Tune min_child_samples
7. Add subsample/colsample_bytree
8. Add regularization if overfitting
```

Example:

```python
model = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.03,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
```
