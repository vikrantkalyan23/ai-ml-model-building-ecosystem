# Scikit-learn (sklearn)

## 1. What is Scikit-learn?

**Scikit-learn**, commonly imported as **sklearn**, is one of the most popular Python libraries for traditional machine learning.

It is mainly used for:

- Regression
- Classification
- Clustering
- Dimensionality reduction
- Data preprocessing
- Feature engineering
- Model selection
- Cross-validation
- Hyperparameter tuning
- Model evaluation

> **Simple definition:** Scikit-learn gives ready-made machine-learning tools so you can train, test, tune, and evaluate models with simple Python code.

---

## 2. Basic Information

| Item | Details |
|---|---|
| **Name** | Scikit-learn |
| **Import name** | sklearn |
| **Type** | Machine-learning library |
| **Language** | Python |
| **Built on** | NumPy, SciPy, Matplotlib |
| **Best for** | Traditional ML and tabular data |
| **Main data type** | Structured/tabular data |
| **Deep learning** | Limited |
| **GPU acceleration** | Usually no |
| **Open source** | Yes |
| **Beginner-friendly** | Very high |
| **Common use cases** | Regression, classification, clustering, preprocessing, evaluation |

---

# 3. What Problem Does Scikit-learn Solve?

Scikit-learn helps you build machine-learning models without writing algorithms from scratch.

For example:

```text
Age    Salary    Experience    Purchased
25     40000     2             No
31     60000     5             Yes
45     90000     15            Yes
28     50000     3             No
```

You can use this data to predict:

```text
Will a new customer purchase?
        |
        v
  Scikit-learn model
        |
        v
     Yes / No
```

Or:

```text
House size, rooms, location
          |
          v
  Scikit-learn model
          |
          v
   Predicted house price
```

---

# 4. Complete ML Workflow in Scikit-learn

Most Scikit-learn projects follow this flow:

```text
Raw Data
   |
   v
Clean Data
   |
   v
Preprocessing
   |
   v
Train/Test Split
   |
   v
Choose Model
   |
   v
Train Model
   |
   v
Evaluate Model
   |
   v
Tune Model
   |
   v
Prediction
```

In code, Scikit-learn models usually follow the same pattern:

```python
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

This consistent design is one big reason Scikit-learn is easy to learn.

---

# 5. Main Categories in Scikit-learn

```text
                         Scikit-learn
                              |
        ------------------------------------------------
        |                    |                         |
        v                    v                         v
 Supervised Learning   Unsupervised Learning      Preprocessing
        |                    |                         |
   -------------        -------------          -----------------
   |           |        |           |          |       |       |
   v           v        v           v          v       v       v
Regression Classification Clustering Dimensionality  Scaling Encoding
                                   Reduction
```

### Main idea

| Category | Meaning | Example |
|---|---|---|
| **Supervised Learning** | Learn from data with correct answers | Price prediction, spam detection |
| **Unsupervised Learning** | Find patterns without given answers | Customer grouping |
| **Preprocessing** | Prepare raw data for models | Scaling, encoding, missing values |
| **Model Selection** | Choose and tune the best model | Cross-validation, GridSearchCV |
| **Model Evaluation** | Check model performance | MAE, accuracy, F1-score |

---

# 6. Supervised Learning

Supervised Learning means the training data already contains the correct answer.

```text
Input features              Target answer
--------------              -------------
House size, rooms           House price
Email words                 Spam / Not spam
Age, income, history        Loan approved / rejected
```

Scikit-learn learns the relationship between input and answer.

```text
Training data + answers
          |
          v
       Model learns
          |
          v
Prediction for new data
```

Supervised Learning has two major types:

```text
Supervised Learning
        |
   -------------
   |           |
   v           v
Regression Classification
```

---

# 7. Regression

**Regression** predicts a continuous number.

```text
Question: How much?
Output: numeric value
```

Examples:

- Predict house price
- Predict sales
- Predict salary
- Predict temperature
- Predict delivery time

Simple example:

```text
Study hours -> Exam marks

1 hour  -> 35
2 hours -> 45
3 hours -> 55
4 hours -> 65

For 5 hours, the model may predict around 75.
```

### Regression code

```python
from sklearn.linear_model import LinearRegression

X = [[1], [2], [3], [4]]
y = [35, 45, 55, 65]

model = LinearRegression()
model.fit(X, y)

print(model.predict([[5]]))
```

---

# 8. Common Regression Models

| Model | Simple meaning | Best for |
|---|---|---|
| **LinearRegression** | Fits a straight-line relationship | Simple numeric prediction |
| **Ridge** | Linear regression with L2 regularization | Reducing overfitting |
| **Lasso** | Linear regression with L1 regularization | Feature selection |
| **ElasticNet** | Mix of Ridge and Lasso | Balanced regularization |
| **DecisionTreeRegressor** | Uses decision rules like a tree | Nonlinear patterns |
| **RandomForestRegressor** | Combines many decision trees | Strong general-purpose regression |
| **GradientBoostingRegressor** | Builds trees step by step to improve errors | Accurate tabular prediction |
| **SVR** | Support Vector Regression | Smaller datasets with complex patterns |
| **KNeighborsRegressor** | Predicts using nearby examples | Similarity-based prediction |

### Imports

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
```

### Try multiple regression models

```python
models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "ElasticNet": ElasticNet(),
    "DecisionTreeRegressor": DecisionTreeRegressor(random_state=42),
    "RandomForestRegressor": RandomForestRegressor(random_state=42),
    "GradientBoostingRegressor": GradientBoostingRegressor(random_state=42),
    "SVR": SVR(),
    "KNeighborsRegressor": KNeighborsRegressor(),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    print(name, model.score(X_test, y_test))
```

---

# 9. Classification

**Classification** predicts a category or class.

```text
Question: Which class?
Output: category
```

Examples:

- Spam or not spam
- Fraud or not fraud
- Pass or fail
- Disease or no disease
- Customer will churn or stay

Simple example:

```text
Study hours -> Result

1 hour  -> Fail
2 hours -> Fail
4 hours -> Pass
5 hours -> Pass

For 3.5 hours, the model may predict Pass.
```

### Classification code

```python
from sklearn.linear_model import LogisticRegression

X = [[1], [2], [4], [5]]
y = [0, 0, 1, 1]  # 0 = Fail, 1 = Pass

model = LogisticRegression()
model.fit(X, y)

print(model.predict([[3.5]]))
```

---

# 10. Common Classification Models

| Model | Simple meaning | Best for |
|---|---|---|
| **LogisticRegression** | Linear model for classification | Binary and multiclass classification |
| **DecisionTreeClassifier** | Uses decision rules like a tree | Explainable decisions |
| **RandomForestClassifier** | Combines many decision trees | Strong general-purpose classification |
| **SVC** | Support Vector Classifier | Clear class boundaries |
| **KNeighborsClassifier** | Classifies using nearest examples | Small/simple datasets |
| **GaussianNB** | Probability-based Naive Bayes | Fast baseline, text-like data |
| **GradientBoostingClassifier** | Builds trees step by step | Accurate tabular classification |

### Imports

```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
```

### Try multiple classification models

```python
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "DecisionTreeClassifier": DecisionTreeClassifier(random_state=42),
    "RandomForestClassifier": RandomForestClassifier(random_state=42),
    "SVC": SVC(),
    "KNeighborsClassifier": KNeighborsClassifier(),
    "GaussianNB": GaussianNB(),
    "GradientBoostingClassifier": GradientBoostingClassifier(random_state=42),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    print(name, model.score(X_test, y_test))
```

---

# 11. Unsupervised Learning

Unsupervised Learning means the data has no target answer.

```text
Input data                 Target answer
----------                 -------------
Customer behavior          Not given
Product usage              Not given
Many numeric columns       Not given
```

The model finds hidden structure.

```text
Unlabeled data
      |
      v
Model finds patterns
      |
      v
Groups or compressed features
```

Two important unsupervised tasks are:

```text
Unsupervised Learning
        |
   -------------------------
   |                       |
   v                       v
Clustering      Dimensionality Reduction
```

---

# 12. Clustering

**Clustering** groups similar data points together.

```text
Question: Which items are similar?
Output: groups/clusters
```

Examples:

- Customer segmentation
- Grouping documents by topic
- Grouping products by behavior
- Finding unusual data points

### Clustering example

```text
Customer A: low income, low spending
Customer B: low income, low spending
Customer C: high income, high spending
Customer D: high income, high spending

Possible result:
Cluster 0 -> A, B
Cluster 1 -> C, D
```

### Common clustering models

| Model | Simple meaning | Best for |
|---|---|---|
| **KMeans** | Creates K groups based on distance | Compact clusters |
| **DBSCAN** | Finds dense groups and noise points | Irregular clusters and outliers |
| **AgglomerativeClustering** | Builds clusters step by step | Hierarchical grouping |

### Code

```python
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

X = [
    [20, 200],
    [22, 220],
    [45, 1500],
    [48, 1600],
]

models = {
    "KMeans": KMeans(n_clusters=2, random_state=42, n_init="auto"),
    "DBSCAN": DBSCAN(eps=3, min_samples=2),
    "AgglomerativeClustering": AgglomerativeClustering(n_clusters=2),
}

for name, model in models.items():
    labels = model.fit_predict(X)
    print(name, labels)
```

---

# 13. Dimensionality Reduction

**Dimensionality Reduction** reduces many columns into fewer columns while keeping important information.

```text
Many features
     |
     v
Fewer useful features
```

Why it helps:

- Makes visualization easier
- Reduces noise
- Can make models faster
- Can reduce overfitting

Example:

```text
Original features:
age, salary, city, visits, clicks, purchases, time_on_site

Reduced features:
component_1, component_2
```

### Common dimensionality reduction models

| Model | Simple meaning | Best for |
|---|---|---|
| **PCA** | Creates principal components from numeric data | Dense numeric data |
| **TruncatedSVD** | PCA-like method for sparse data | Text and large sparse matrices |

### Code

```python
from sklearn.decomposition import PCA, TruncatedSVD

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

svd = TruncatedSVD(n_components=2)
X_svd = svd.fit_transform(X)
```

---

# 14. Data Preprocessing

Data Preprocessing means preparing raw data before model training.

Raw data may contain:

- Missing values
- Text categories
- Different numeric scales
- Duplicate rows
- Wrong data types
- Outliers

Good preprocessing often improves model quality more than changing the algorithm.

```text
Raw Data
   |
   v
Clean Data
   |
   v
Encoded and Scaled Data
   |
   v
Model
```

---

# 15. Encoding Categorical Data

Most machine-learning models need numbers, not text.

**OneHotEncoder** converts categories into numeric columns.

Example:

```text
Color
-----
Red
Blue
Green

After OneHotEncoder:

Color_Red  Color_Blue  Color_Green
1          0           0
0          1           0
0          0           1
```

### Code

```python
from sklearn.preprocessing import OneHotEncoder

X = [["Red"], ["Blue"], ["Green"]]

encoder = OneHotEncoder(sparse_output=False)
encoded = encoder.fit_transform(X)

print(encoded)
```

---

# 16. Scaling Numeric Data

Scaling makes numeric columns comparable.

Example:

```text
Age:       20 to 70
Salary:    20000 to 200000

Without scaling, salary may dominate distance-based models.
```

Scaling is especially useful for:

- LogisticRegression
- SVC
- SVR
- KNeighborsClassifier
- KNeighborsRegressor
- PCA

## StandardScaler

StandardScaler changes data so it has:

```text
mean = 0
standard deviation = 1
```

```python
from sklearn.preprocessing import StandardScaler

X = [[10], [20], [30], [40]]

scaler = StandardScaler()
scaled = scaler.fit_transform(X)

print(scaled)
```

## MinMaxScaler

MinMaxScaler converts values into a fixed range, usually 0 to 1.

```text
Original: 10, 20, 30, 40
Scaled:   0, 0.33, 0.67, 1
```

```python
from sklearn.preprocessing import MinMaxScaler

X = [[10], [20], [30], [40]]

scaler = MinMaxScaler()
scaled = scaler.fit_transform(X)

print(scaled)
```

---

# 17. Train/Test Split

Train/Test Split separates data into two parts.

```text
Training data -> used to teach the model
Testing data  -> used to check the model
```

Common split:

```text
80% training
20% testing
```

### Code

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

Why it matters:

```text
If you test on the same data used for training,
the score may look good but fail on real new data.
```

---

# 18. Model Evaluation

Model Evaluation means checking how well a model performs.

Different tasks need different metrics.

```text
Regression     -> numeric prediction metrics
Classification -> category prediction metrics
Clustering     -> grouping quality metrics
```

---

# 19. Regression Evaluation Metrics

Assume:

```text
Actual values:    100, 200, 300
Predicted values: 110, 190, 330
Errors:            10, -10, 30
```

| Metric | Full form | Simple meaning | Better value |
|---|---|---|---|
| **MAE** | Mean Absolute Error | Average absolute mistake | Lower |
| **MSE** | Mean Squared Error | Average squared mistake | Lower |
| **RMSE** | Root Mean Squared Error | Error in original unit | Lower |
| **R2 / R²** | R-squared | How much variance is explained | Higher |
| **MAPE** | Mean Absolute Percentage Error | Average percent mistake | Lower |

### MAE

```text
Errors: 10, 10, 30
MAE = (10 + 10 + 30) / 3 = 16.67
```

### MSE

```text
Squared errors: 100, 100, 900
MSE = (100 + 100 + 900) / 3 = 366.67
```

### RMSE

```text
RMSE = sqrt(MSE)
RMSE = sqrt(366.67) = 19.15
```

### R2 / R²

```text
R2/R² close to 1  -> very good
R2/R² close to 0  -> weak
R2/R² below 0     -> worse than simple average prediction
```

### MAPE

```text
Actual:    100
Predicted: 110
Error:     10%
```

### Code

```python
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score,
)

y_true = [100, 200, 300]
y_pred = [110, 190, 330]

mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_true, y_pred)
mape = mean_absolute_percentage_error(y_true, y_pred)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2/R²:", r2)
print("MAPE:", mape)
```

---

# 20. Classification Evaluation Metrics

Common classification metrics:

| Metric | Meaning |
|---|---|
| **Accuracy** | How many total predictions are correct |
| **Precision** | Of predicted positives, how many are actually positive |
| **Recall** | Of actual positives, how many were found |
| **F1-score** | Balance between precision and recall |
| **Confusion matrix** | Table of correct and wrong predictions |
| **ROC-AUC** | How well the model separates classes |

### Code

```python
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

y_true = [0, 1, 1, 0, 1]
y_pred = [0, 1, 0, 0, 1]

print("Accuracy:", accuracy_score(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred))
```

---

# 21. Cross-validation

Cross-validation tests the model multiple times using different train/test splits.

Example: 5-fold cross-validation

```text
Round 1: Fold 1 test, Folds 2-5 train
Round 2: Fold 2 test, Folds 1,3,4,5 train
Round 3: Fold 3 test, Folds 1,2,4,5 train
Round 4: Fold 4 test, Folds 1,2,3,5 train
Round 5: Fold 5 test, Folds 1-4 train

Final score = average of all scores
```

Why it helps:

- More reliable than one train/test split
- Uses data more efficiently
- Helps detect unstable models

### Code

```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

model = LinearRegression()
scores = cross_val_score(model, X, y, cv=5, scoring="r2")

print(scores)
print(scores.mean())
```

---

# 22. Hyperparameter Tuning

Hyperparameters are settings chosen before training.

Examples:

```text
RandomForestRegressor(n_estimators=100)
DecisionTreeClassifier(max_depth=3)
KNeighborsClassifier(n_neighbors=5)
KMeans(n_clusters=4)
```

Hyperparameter tuning means trying different settings to improve model performance.

```text
Default settings
      |
      v
Tune hyperparameters
      |
      v
Better model
```

---

# 23. GridSearchCV

**GridSearchCV** tries every combination from a parameter grid.

Use it when:

- The search space is small
- You want a complete search
- Training is not too expensive

Example:

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(random_state=42)

params = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 5, 10],
}

search = GridSearchCV(model, params, cv=5, scoring="neg_mean_absolute_error")
search.fit(X_train, y_train)

print(search.best_params_)
print(search.best_score_)
```

---

# 24. RandomizedSearchCV

**RandomizedSearchCV** tries random combinations from the parameter options.

Use it when:

- The search space is large
- You want faster tuning
- You do not need to test every combination

Example:

```python
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(random_state=42)

params = {
    "n_estimators": [50, 100, 200, 300, 500],
    "max_depth": [None, 5, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
}

search = RandomizedSearchCV(
    model,
    params,
    n_iter=10,
    cv=5,
    scoring="neg_mean_absolute_error",
    random_state=42,
)

search.fit(X_train, y_train)
print(search.best_params_)
```

### GridSearchCV vs RandomizedSearchCV

| Feature | GridSearchCV | RandomizedSearchCV |
|---|---|---|
| Search method | Tries all combinations | Tries random combinations |
| Speed | Slower | Faster |
| Best for | Small parameter grids | Large parameter spaces |
| Control | Very systematic | More flexible |

---

# 25. Feature Selection

Feature Selection means choosing the most useful input columns.

Example:

```text
Original features:
age, salary, city, customer_id, last_purchase, random_noise

Useful features:
age, salary, city, last_purchase
```

Why it helps:

- Removes useless columns
- Can improve accuracy
- Can reduce overfitting
- Can make training faster
- Makes models easier to understand

### Code

```python
from sklearn.feature_selection import SelectKBest, f_regression

selector = SelectKBest(score_func=f_regression, k=3)
X_selected = selector.fit_transform(X, y)

print(X_selected)
```

### Feature Selection vs Dimensionality Reduction

| Topic | Meaning | Output |
|---|---|---|
| **Feature Selection** | Keeps useful original columns | age, salary, city |
| **Dimensionality Reduction** | Creates new compressed columns | component_1, component_2 |

---

# 26. Pipelines

A **Pipeline** combines preprocessing and model training into one clean workflow.

Without a pipeline:

```text
Scale data
Train model
Predict
```

With a pipeline:

```text
Raw data
   |
   v
Scaler -> Model
   |
   v
Prediction
```

### Code

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression()),
])

pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

Pipelines are important because they help avoid data leakage and make the workflow cleaner.

---

# 27. Complete Mini Project

This example shows a simple regression workflow using built-in Scikit-learn data.

```python
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

X, y = load_diabetes(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestRegressor(random_state=42)),
])

pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = mse ** 0.5
r2 = r2_score(y_test, predictions)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2/R²:", r2)

scores = cross_val_score(pipeline, X, y, cv=5, scoring="r2")
print("Cross-validation R2:", scores.mean())

params = {
    "model__n_estimators": [50, 100],
    "model__max_depth": [None, 10],
}

search = GridSearchCV(pipeline, params, cv=3, scoring="r2")
search.fit(X_train, y_train)

print("Best parameters:", search.best_params_)
```

---

# 28. When to Use Scikit-learn

Use Scikit-learn for:

- Tabular data
- Small to medium datasets
- Regression
- Classification
- Clustering
- Dimensionality reduction
- Preprocessing
- Feature selection
- Model evaluation
- Beginner ML projects

Common real-world examples:

```text
Customer churn prediction
House price prediction
Fraud detection
Credit risk scoring
Customer segmentation
Sales forecasting
Basic recommendation systems
```

---

# 29. When Not to Use Scikit-learn

Scikit-learn is not the best choice for large deep-learning workloads.

Avoid Scikit-learn for:

```text
Large image classification
Complex computer vision
Large language models
Transformer training
Deep neural networks
GPU-heavy training
```

For those tasks, use libraries such as:

| Task | Better option |
|---|---|
| Deep learning | PyTorch, TensorFlow, Keras |
| LLMs | Hugging Face Transformers, PyTorch |
| Computer vision deep learning | PyTorch, TensorFlow, Ultralytics |
| Large gradient boosting | XGBoost, LightGBM, CatBoost |

---

# 30. Advantages of Scikit-learn

## Easy to learn

Most models use the same syntax:

```python
model.fit(X_train, y_train)
model.predict(X_test)
```

## Huge algorithm collection

```text
Linear Regression
Logistic Regression
Decision Trees
Random Forest
SVM
KNN
K-Means
PCA
Naive Bayes
Gradient Boosting
```

## Excellent preprocessing tools

```text
Scaling
Encoding
Imputation
Feature selection
Dimensionality reduction
Pipelines
```

## Excellent model evaluation

```text
MAE
MSE
RMSE
R2/R²
Accuracy
Precision
Recall
F1
ROC-AUC
Cross-validation
```

## Strong for tabular data

For many business problems, Scikit-learn is one of the best starting points.

---

# 31. Disadvantages of Scikit-learn

## Not designed for modern deep learning

Scikit-learn is not meant for:

```text
CNN
RNN
LSTM
GRU
Large Transformer
LLM
```

## Limited GPU support

Most Scikit-learn algorithms run on CPU.

## Neural-network support is basic

Scikit-learn includes:

```text
MLPClassifier
MLPRegressor
```

But for serious neural networks, PyTorch or TensorFlow is usually better.

---

# 32. Scikit-learn vs PyTorch

| Feature | Scikit-learn | PyTorch |
|---|---|---|
| Main purpose | Traditional ML | Deep learning |
| Beginner-friendly | Very high | Medium |
| Tabular ML | Excellent | Good |
| Deep learning | Limited | Excellent |
| Neural networks | Basic | Excellent |
| CNN/RNN/Transformers | No | Yes |
| LLM training/fine-tuning | No | Yes |
| GPU support | Limited | Strong |
| Typical syntax | High-level | Flexible and lower-level |

---

# 33. Scikit-learn vs TensorFlow/Keras

| Feature | Scikit-learn | TensorFlow/Keras |
|---|---|---|
| Main purpose | Traditional ML | Deep learning |
| Ease of learning | Very high | Medium |
| Tabular ML | Excellent | Good |
| Neural networks | Basic | Excellent |
| CNN | No | Yes |
| RNN/LSTM/GRU | No | Yes |
| Transformers | No | Yes |
| GPU training | Limited | Strong |
| Production deep learning | No | Yes |

---

# 34. Learning Roadmap

```text
1. Machine-learning basics
        |
2. Data preprocessing
        |
3. Train/Test Split
        |
4. Linear Regression
        |
5. Logistic Regression
        |
6. Decision Trees
        |
7. Random Forest
        |
8. KNN
        |
9. SVM
        |
10. Naive Bayes
        |
11. K-Means clustering
        |
12. PCA
        |
13. Pipelines
        |
14. Cross-validation
        |
15. Hyperparameter tuning
        |
16. Feature selection
        |
17. Model evaluation
        |
18. XGBoost / LightGBM / CatBoost
```

---

# 35. Quick Revision Table

| Topic | Meaning |
|---|---|
| **Scikit-learn** | Python library for traditional ML |
| **Supervised Learning** | Learns from data with answers |
| **Regression** | Predicts numbers |
| **Classification** | Predicts categories |
| **Unsupervised Learning** | Finds patterns without answers |
| **Clustering** | Groups similar data |
| **Dimensionality Reduction** | Reduces many columns to fewer columns |
| **Data Preprocessing** | Cleans, encodes, and scales data |
| **OneHotEncoder** | Converts categories to numeric columns |
| **StandardScaler** | Scales data to mean 0 and standard deviation 1 |
| **MinMaxScaler** | Scales data to 0-1 range |
| **Train/Test Split** | Separates training and testing data |
| **Cross-validation** | Tests model multiple times |
| **Hyperparameter Tuning** | Finds better model settings |
| **GridSearchCV** | Tries all parameter combinations |
| **RandomizedSearchCV** | Tries random parameter combinations |
| **Feature Selection** | Selects useful columns |
| **Pipeline** | Combines preprocessing and model steps |
| **Model Evaluation** | Measures model performance |

---

# 36. Final Summary

```text
Scikit-learn is the best starting library for traditional machine learning.

Use it when:
    - Your data is structured or tabular
    - You need regression, classification, or clustering
    - You want preprocessing and evaluation tools
    - You are learning ML fundamentals

Do not use it as the main tool for:
    - Large neural networks
    - LLMs
    - Complex computer vision
    - GPU-heavy deep learning

Most important pattern:
    model.fit(X_train, y_train)
    model.predict(X_test)
```
