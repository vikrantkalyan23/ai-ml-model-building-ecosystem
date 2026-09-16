# Machine Learning Fundamentals

Machine Learning means teaching a computer to learn patterns from data and use those patterns to make predictions or discover structure.

Simple idea:

```text
Past data + correct method = model
model + new data = prediction or insight
```

Example:

```text
House size, rooms, location -> House price
Email words -> Spam or not spam
Customer purchase history -> Customer groups
Many columns -> Fewer important columns
```

## Category Chart

```text
Machine Learning
|
|-- Supervised Learning
|   |
|   |-- Regression
|   |   |-- Predict a number
|   |   |-- Example: house price, salary, temperature
|   |   |-- Models:
|   |       LinearRegression
|   |       Ridge
|   |       Lasso
|   |       ElasticNet
|   |       DecisionTreeRegressor
|   |       RandomForestRegressor
|   |       GradientBoostingRegressor
|   |       SVR
|   |       KNeighborsRegressor
|   |
|   |-- Classification
|       |-- Predict a category/class
|       |-- Example: spam/not spam, fraud/not fraud
|       |-- Models:
|           LogisticRegression
|           DecisionTreeClassifier
|           RandomForestClassifier
|           SVC
|           KNeighborsClassifier
|           GaussianNB
|           GradientBoostingClassifier
|
|-- Unsupervised Learning
|   |
|   |-- Clustering
|   |   |-- Find groups without labels
|   |   |-- Example: customer segmentation
|   |   |-- Models:
|   |       KMeans
|   |       DBSCAN
|   |       AgglomerativeClustering
|   |
|   |-- Dimensionality Reduction
|       |-- Reduce many columns to fewer useful columns
|       |-- Example: visualization, noise reduction
|       |-- Models:
|           PCA
|           TruncatedSVD
|
|-- Data Preprocessing
|   |
|   |-- OneHotEncoder
|   |-- MinMaxScaler
|   |-- StandardScaler
|
|-- Model Evaluation
|   |
|   |-- Regression metrics:
|   |   MAE, MSE, RMSE, R2/R², MAPE
|   |
|   |-- Validation:
|       Train/Test Split
|       Cross-validation
|
|-- Model Improvement
    |
    |-- Hyperparameter Tuning
    |-- GridSearchCV
    |-- RandomizedSearchCV
    |-- Feature Selection
```

## Basic ML Workflow

```text
1. Collect data
       |
2. Clean data
       |
3. Preprocess data
       |
4. Split into train/test data
       |
5. Choose model
       |
6. Train model
       |
7. Evaluate model
       |
8. Tune model
       |
9. Use model for prediction
```

Example in Python:

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

X = [[500], [800], [1000], [1200], [1500]]
y = [150000, 220000, 300000, 360000, 450000]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(predictions)
print(mean_absolute_error(y_test, predictions))
```

## Supervised Learning

Supervised Learning uses data where the answer is already known.

```text
Input data                 Known answer
----------                 ------------
House size, rooms          Price
Email text                 Spam or not spam
Patient measurements       Disease or no disease
```

The model learns from examples.

```text
Training examples -> model learns pattern -> model predicts new answer
```

Supervised Learning has two main types:

```text
Supervised Learning
|
|-- Regression
|-- Classification
```

## Regression

Regression predicts a continuous numeric value.

```text
Question: How much?
Output: number
```

Examples:

```text
Predict house price
Predict car price
Predict exam score
Predict monthly sales
Predict temperature
```

Small demonstration:

```text
Study hours -> Exam marks

1 hour  -> 35 marks
2 hours -> 45 marks
3 hours -> 55 marks
4 hours -> 65 marks

If a student studies 5 hours, the model may predict around 75 marks.
```

Regression code:

```python
from sklearn.linear_model import LinearRegression

X = [[1], [2], [3], [4]]
y = [35, 45, 55, 65]

model = LinearRegression()
model.fit(X, y)

print(model.predict([[5]]))
```

## Classification

Classification predicts a class or category.

```text
Question: Which type?
Output: category
```

Examples:

```text
Spam or not spam
Pass or fail
Fraud or not fraud
Cat, dog, or horse
Customer will leave or stay
```

Small demonstration:

```text
Study hours -> Result

1 hour  -> Fail
2 hours -> Fail
4 hours -> Pass
5 hours -> Pass

If a student studies 3.5 hours, the model may predict Pass.
```

Classification code:

```python
from sklearn.linear_model import LogisticRegression

X = [[1], [2], [4], [5]]
y = [0, 0, 1, 1]  # 0 = Fail, 1 = Pass

model = LogisticRegression()
model.fit(X, y)

print(model.predict([[3.5]]))
```

## Unsupervised Learning

Unsupervised Learning uses data where answers are not given.

```text
Input data                 Known answer
----------                 ------------
Customer age, income       No label
Product behavior           No label
Many numeric columns       No label
```

The model tries to find hidden patterns.

```text
Data without labels -> model finds groups or structure
```

Unsupervised Learning has two common types:

```text
Unsupervised Learning
|
|-- Clustering
|-- Dimensionality Reduction
```

## Clustering

Clustering groups similar data points together.

```text
Question: Which points are similar?
Output: groups/clusters
```

Examples:

```text
Group customers by shopping behavior
Group documents by topic
Group images by visual similarity
Find unusual data points
```

Small demonstration:

```text
Customers:

Customer A: low income, low spending
Customer B: low income, low spending
Customer C: high income, high spending
Customer D: high income, high spending

The model may create:
Cluster 0 -> A, B
Cluster 1 -> C, D
```

KMeans code:

```python
from sklearn.cluster import KMeans

X = [
    [20, 200],
    [22, 220],
    [45, 1500],
    [48, 1600],
]

model = KMeans(n_clusters=2, random_state=42, n_init="auto")
labels = model.fit_predict(X)

print(labels)
```

## Dimensionality Reduction

Dimensionality Reduction reduces many columns into fewer columns while keeping important information.

```text
Many features -> fewer features
```

Why it is useful:

```text
Makes data easier to visualize
Removes noise
Can make models faster
Can reduce overfitting
```

Example:

```text
Original data:
height, weight, age, income, spending, visits, clicks, purchases

Reduced data:
component_1, component_2
```

PCA code:

```python
from sklearn.decomposition import PCA

X = [
    [1, 2, 3],
    [2, 3, 4],
    [8, 9, 10],
    [9, 10, 11],
]

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

print(X_reduced)
```

## Data Preprocessing

Data Preprocessing means preparing raw data before giving it to a model.

Raw data often has problems:

```text
Missing values
Text categories
Different scales
Outliers
Duplicate rows
Wrong data types
```

Clean data helps models learn better.

```text
Raw data -> cleaned data -> encoded/scaled data -> model
```

## OneHotEncoder

Most ML models understand numbers, not text categories.

OneHotEncoder converts categories into numeric columns.

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

Code:

```python
from sklearn.preprocessing import OneHotEncoder

X = [["Red"], ["Blue"], ["Green"]]

encoder = OneHotEncoder(sparse_output=False)
encoded = encoder.fit_transform(X)

print(encoded)
```

## StandardScaler

StandardScaler changes data so it has:

```text
mean = 0
standard deviation = 1
```

It is useful for models such as:

```text
LogisticRegression
SVC
SVR
KNeighborsClassifier
KNeighborsRegressor
PCA
```

Code:

```python
from sklearn.preprocessing import StandardScaler

X = [[10], [20], [30], [40]]

scaler = StandardScaler()
scaled = scaler.fit_transform(X)

print(scaled)
```

## MinMaxScaler

MinMaxScaler converts values into a fixed range, usually 0 to 1.

Example:

```text
Original: 10, 20, 30, 40
Scaled:   0, 0.33, 0.67, 1
```

Code:

```python
from sklearn.preprocessing import MinMaxScaler

X = [[10], [20], [30], [40]]

scaler = MinMaxScaler()
scaled = scaler.fit_transform(X)

print(scaled)
```

## Train/Test Split

Train/Test Split divides data into two parts.

```text
Training data -> used to teach the model
Testing data  -> used to check the model
```

Common split:

```text
80% training
20% testing
```

Code:

```python
from sklearn.model_selection import train_test_split

X = [[1], [2], [3], [4], [5]]
y = [10, 20, 30, 40, 50]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

## Model Evaluation

Model Evaluation means checking how good or bad a model is.

For regression, common metrics are:

```text
MAE
MSE
RMSE
R2/R²
MAPE
```

For classification, common metrics are:

```text
Accuracy
Precision
Recall
F1-score
Confusion matrix
ROC-AUC
```

## Regression Metrics

Assume:

```text
Actual values:    100, 200, 300
Predicted values: 110, 190, 330
Errors:            10, -10, 30
```

## MAE

MAE means Mean Absolute Error.

```text
MAE = average of absolute errors
```

Simple meaning:

```text
On average, how far are predictions from actual values?
```

Example:

```text
Actual:    100, 200, 300
Predicted: 110, 190, 330
Errors:    10,  10,  30

MAE = (10 + 10 + 30) / 3 = 16.67
```

Code:

```python
from sklearn.metrics import mean_absolute_error

y_true = [100, 200, 300]
y_pred = [110, 190, 330]

print(mean_absolute_error(y_true, y_pred))
```

## MSE

MSE means Mean Squared Error.

```text
MSE = average of squared errors
```

Simple meaning:

```text
Large mistakes are punished more strongly.
```

Example:

```text
Errors: 10, 10, 30
Squared errors: 100, 100, 900

MSE = (100 + 100 + 900) / 3 = 366.67
```

Code:

```python
from sklearn.metrics import mean_squared_error

y_true = [100, 200, 300]
y_pred = [110, 190, 330]

print(mean_squared_error(y_true, y_pred))
```

## RMSE

RMSE means Root Mean Squared Error.

```text
RMSE = square root of MSE
```

Simple meaning:

```text
Like MSE, but converted back to the original unit.
```

Example:

```text
MSE = 366.67
RMSE = sqrt(366.67) = 19.15
```

Code:

```python
from sklearn.metrics import mean_squared_error

y_true = [100, 200, 300]
y_pred = [110, 190, 330]

rmse = mean_squared_error(y_true, y_pred) ** 0.5
print(rmse)
```

## R2 / R²

R2, also written as R², means R-squared.

```text
R2/R² shows how much variance the model explains.
```

Simple meaning:

```text
R2/R² close to 1  -> very good
R2/R² close to 0  -> weak model
R2/R² below 0     -> worse than simple average prediction
```

Code:

```python
from sklearn.metrics import r2_score

y_true = [100, 200, 300]
y_pred = [110, 190, 330]

print(r2_score(y_true, y_pred))
```

## MAPE

MAPE means Mean Absolute Percentage Error.

```text
MAPE = average percentage error
```

Simple meaning:

```text
On average, predictions are wrong by what percent?
```

Example:

```text
Actual:    100
Predicted: 110

Error = 10%
```

Code:

```python
from sklearn.metrics import mean_absolute_percentage_error

y_true = [100, 200, 300]
y_pred = [110, 190, 330]

print(mean_absolute_percentage_error(y_true, y_pred))
```

## Metric Selection Guide

| Metric | Use when | Lower or higher is better |
| ------ | -------- | ------------------------- |
| MAE | You want easy-to-understand average error | Lower |
| MSE | You want to punish large errors strongly | Lower |
| RMSE | You want error in original unit | Lower |
| R2/R² | You want explained variance score | Higher |
| MAPE | You want percentage error | Lower |

## Cross-validation

Cross-validation checks model performance more reliably than one train/test split.

Instead of testing only once, it trains and tests multiple times.

```text
Example: 5-fold cross-validation

Round 1: Fold 1 test, Folds 2-5 train
Round 2: Fold 2 test, Folds 1,3,4,5 train
Round 3: Fold 3 test, Folds 1,2,4,5 train
Round 4: Fold 4 test, Folds 1,2,3,5 train
Round 5: Fold 5 test, Folds 1-4 train

Final score = average of all 5 scores
```

Code:

```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

model = LinearRegression()
scores = cross_val_score(model, X, y, cv=5, scoring="r2")

print(scores)
print(scores.mean())
```

## Hyperparameter Tuning

Hyperparameters are settings chosen before training.

Examples:

```text
RandomForestRegressor(n_estimators=100)
KNeighborsClassifier(n_neighbors=5)
DecisionTreeClassifier(max_depth=3)
KMeans(n_clusters=4)
```

Hyperparameter Tuning means trying different settings to find the best model.

```text
Default settings -> okay model
Better settings  -> better model
```

## GridSearchCV

GridSearchCV tries every combination of given hyperparameters.

Good when:

```text
You have small search space
You want a careful search
```

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

## RandomizedSearchCV

RandomizedSearchCV tries random combinations of hyperparameters.

Good when:

```text
You have many possible values
You want faster tuning
```

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

## Feature Selection

Feature Selection means choosing the most useful columns.

Example:

```text
Original features:
age, salary, city, customer_id, last_purchase, random_noise

Useful features:
age, salary, city, last_purchase
```

Why it helps:

```text
Removes useless columns
Can improve accuracy
Can reduce overfitting
Can make model faster
```

Code:

```python
from sklearn.feature_selection import SelectKBest, f_regression

selector = SelectKBest(score_func=f_regression, k=3)
X_selected = selector.fit_transform(X, y)

print(X_selected)
```

## Feature Selection vs Dimensionality Reduction

| Topic | Meaning | Example |
| ----- | ------- | ------- |
| Feature Selection | Select original useful columns | Keep age, salary, city |
| Dimensionality Reduction | Create new compressed columns | PCA component 1, PCA component 2 |

## Regression Models

Regression models predict numbers.

| Model | Simple meaning | Best for |
| ----- | -------------- | -------- |
| LinearRegression | Fits a straight-line relationship | Simple numeric prediction |
| Ridge | Linear regression with L2 regularization | Many features, less overfitting |
| Lasso | Linear regression with L1 regularization | Feature selection and simple models |
| ElasticNet | Mix of Ridge and Lasso | Balanced regularization |
| DecisionTreeRegressor | Uses decision rules like a tree | Nonlinear patterns |
| RandomForestRegressor | Many decision trees together | Strong general-purpose regression |
| GradientBoostingRegressor | Trees built step by step to fix errors | High accuracy on tabular data |
| SVR | Support Vector Regression | Smaller datasets, complex boundaries |
| KNeighborsRegressor | Predicts using nearby examples | Simple similarity-based prediction |

Regression model imports:

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
```

Regression model example:

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
    score = model.score(X_test, y_test)
    print(name, score)
```

## Classification Models

Classification models predict categories.

| Model | Simple meaning | Best for |
| ----- | -------------- | -------- |
| LogisticRegression | Linear classifier for categories | Binary and multiclass classification |
| DecisionTreeClassifier | Uses decision rules like a tree | Easy-to-explain classification |
| RandomForestClassifier | Many decision trees together | Strong general-purpose classification |
| SVC | Support Vector Classifier | Clear class boundaries |
| KNeighborsClassifier | Classifies using nearby examples | Small/simple datasets |
| GaussianNB | Probability-based Naive Bayes | Fast baseline, text-like data |
| GradientBoostingClassifier | Trees built step by step to fix errors | Accurate tabular classification |

Classification model imports:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
```

Classification model example:

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
    score = model.score(X_test, y_test)
    print(name, score)
```

## Clustering Models

| Model | Simple meaning | Best for |
| ----- | -------------- | -------- |
| KMeans | Creates K groups based on distance | Round/compact clusters |
| DBSCAN | Finds dense groups and outliers | Irregular clusters and noise |
| AgglomerativeClustering | Builds clusters step by step | Hierarchical grouping |

Clustering imports:

```python
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
```

Clustering example:

```python
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

models = {
    "KMeans": KMeans(n_clusters=2, random_state=42, n_init="auto"),
    "DBSCAN": DBSCAN(eps=3, min_samples=2),
    "AgglomerativeClustering": AgglomerativeClustering(n_clusters=2),
}

for name, model in models.items():
    labels = model.fit_predict(X)
    print(name, labels)
```

## Dimensionality Reduction Models

| Model | Simple meaning | Best for |
| ----- | -------------- | -------- |
| PCA | Compresses numeric data into principal components | Dense numeric data |
| TruncatedSVD | PCA-like method for sparse data | Text data, large sparse matrices |

Imports:

```python
from sklearn.decomposition import PCA, TruncatedSVD
```

Example:

```python
from sklearn.decomposition import PCA, TruncatedSVD

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

svd = TruncatedSVD(n_components=2)
X_svd = svd.fit_transform(X)
```

## Complete Mini Demonstration

This example shows a normal supervised regression workflow.

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

## Choosing the Right Algorithm

```text
Need to predict a number?
    Use Regression

Need to predict a category?
    Use Classification

Need to find groups without labels?
    Use Clustering

Need to reduce many columns?
    Use Dimensionality Reduction

Need better performance?
    Use Cross-validation and Hyperparameter Tuning

Need cleaner input data?
    Use Data Preprocessing
```

## Quick Revision Table

| Topic | Meaning | Example |
| ----- | ------- | ------- |
| Supervised Learning | Learn from data with answers | Price prediction, spam detection |
| Regression | Predict number | House price |
| Classification | Predict category | Spam or not spam |
| Unsupervised Learning | Learn from data without answers | Customer grouping |
| Clustering | Find groups | Customer segments |
| Dimensionality Reduction | Reduce columns | PCA visualization |
| Train/Test Split | Separate training and testing data | 80/20 split |
| Cross-validation | Test model multiple times | 5-fold CV |
| Hyperparameter Tuning | Find best settings | max_depth, n_neighbors |
| GridSearchCV | Try all parameter combinations | Careful search |
| RandomizedSearchCV | Try random parameter combinations | Faster search |
| Feature Selection | Choose useful columns | SelectKBest |
| OneHotEncoder | Convert categories to numbers | Red/Blue/Green columns |
| StandardScaler | Mean 0, std 1 scaling | SVM, KNN, PCA |
| MinMaxScaler | Scale to 0-1 range | Normalized numeric data |
| MAE | Average absolute error | Easy regression error |
| MSE | Average squared error | Punishes big mistakes |
| RMSE | Root of MSE | Error in original unit |
| R2/R² | Explained variance score | Higher is better |
| MAPE | Percentage error | Business-friendly error |

## Final Simple Summary

```text
Machine Learning learns patterns from data.

Supervised Learning:
    Data has answers.
    Regression predicts numbers.
    Classification predicts categories.

Unsupervised Learning:
    Data has no answers.
    Clustering finds groups.
    Dimensionality Reduction reduces columns.

Preprocessing:
    Clean, encode, and scale data.

Evaluation:
    Measure model performance using metrics.

Improvement:
    Use cross-validation, tuning, and feature selection.
```
