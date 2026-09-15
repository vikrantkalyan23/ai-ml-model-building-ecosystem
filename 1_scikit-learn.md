# Scikit-learn (sklearn)

Scikit-learn is  primary toolkit for traditional machine learning, especially structured/tabular data.

Scikit-learn is an open-source Python library that provides ready-to-use tools for:

Regression
Classification
Clustering
Dimensionality reduction
Feature engineering
Data preprocessing
Model selection
Cross-validation
Hyperparameter tuning
Model evaluation

```
Raw Data
   ↓
Clean Data
   ↓
Preprocessing
   ↓
Train/Test Split
   ↓
Choose Model
   ↓
Train
   ↓
Evaluate
   ↓
Tune
   ↓
Prediction
```
Scikit-learn provides tools for almost every step except things like deep neural-network training.

## What kind of ML does Scikit-learn handle?
```
                 Scikit-learn
                      │
       ┌──────────────┼──────────────┐
       │              │              │
       ▼              ▼              ▼
  Supervised     Unsupervised    Preprocessing
    Learning       Learning
       │              │
   ┌───┴───┐      ┌───┴────┐
   ▼       ▼      ▼        ▼
Regression Classification Clustering PCA
```

## Common Scikit-learn regression models:
```
Linear Regression
Ridge Regression
Lasso Regression
Decision Tree Regressor
Random Forest Regressor
Gradient Boosting Regressor
Random Forest
Support Vector Regression
```

## Popular Scikit-learn algorithms:
```
K-Means
DBSCAN
Agglomerative Clustering
Gaussian Mixture Models
```

## Best use cases for Scikit-learn
```
Customer churn
House price prediction
Fraud detection
Credit risk
Customer segmentation
```

## When NOT to use Scikit-learn
```
❌ Large image classification
❌ Complex computer vision
❌ LLMs
❌ Large generative AI models
```

## Advantages of Scikit-learn
#### Very easy to learn
```
model.fit(X_train, y_train)
model.predict(X_test)
```
#### Huge collection of algorithms
```
Random Forest
SVM
KNN
K-Means
PCA
```
### Excellent documentation
One reason Scikit-learn is so popular is its documentation and examples.
### Excellent for beginners
You can focus on ML concepts rather than neural-network implementation details.
### Excellent preprocessing tools
This is one of its biggest strengths:
```
Scaling
Encoding
Imputation
Feature selection
Dimensionality reduction
Pipelines
```
### Excellent model evaluation
```
MAE
MSE
R²
Accuracy
Precision
Recall
F1
ROC-AUC
Cross-validation
```
### Excellent for tabular data
For business applications, this is extremely valuable.

## Disadvantages
### Not designed for modern deep learning
Don't use Scikit-learn to build:
```
CNN
LSTM
GRU
Large Transformer
LLM
```
### Limited GPU/deep-learning capabilities
Scikit-learn's core ecosystem is primarily CPU-oriented
### Not an LLM framework
For:
```
Llama
Mistral
BERT
GPT-style models
```
look at PyTorch + Hugging Face.
### Neural networks are limited
Scikit-learn has some neural-network support such as:
```
MLPClassifier
MLPRegressor
```
but these aren't intended to replace dedicated deep-learning frameworks.

## Scikit-learn vs PyTorch
|                   | Scikit-learn | PyTorch |
| ----------------- | ------------ | ------- |
| Traditional ML    | ⭐⭐⭐⭐⭐        | ⭐       |
| Deep Learning     | ⭐            | ⭐⭐⭐⭐⭐   |
| Neural Networks   | Limited      | ⭐⭐⭐⭐⭐   |
| CNN               | ❌            | ✅       |
| RNN/LSTM          | ❌            | ✅       |
| Transformers      | ❌            | ✅       |
| LLM               | ❌            | ⭐⭐⭐⭐⭐   |
| Tabular ML        | ⭐⭐⭐⭐⭐        | ⭐⭐⭐     |
| Research          | ⭐⭐⭐          | ⭐⭐⭐⭐⭐   |
| Beginner-friendly | ⭐⭐⭐⭐⭐        | ⭐⭐⭐     |
| GPU               | Limited      | ⭐⭐⭐⭐⭐   |

## Scikit-learn vs TensorFlow/Keras
|                   | Scikit-learn   | TensorFlow/Keras                         |
| ----------------- | -------------- | ---------------------------------------- |
| Main purpose      | Traditional ML | Deep Learning                            |
| Neural networks   | Limited/simple | ⭐⭐⭐⭐⭐                                    |
| Random Forest     | ✅              | ❌                                        |
| SVM               | ✅              | ❌/not typical                            |
| KNN               | ✅              | ❌                                        |
| Linear Regression | ✅              | ✅ possible, but not its main strength    |
| CNN               | ❌              | ✅                                        |
| RNN               | ❌              | ✅                                        |
| LSTM              | ❌              | ✅                                        |
| GRU               | ❌              | ✅                                        |
| Transformers      | ❌              | ✅                                        |
| GPU training      | Limited        | ✅                                        |
| Ease of learning  | ⭐⭐⭐⭐⭐          | ⭐⭐⭐                                      |
| Tabular data      | ⭐⭐⭐⭐⭐          | ⭐⭐⭐                                      |
| Image data        | ⭐              | ⭐⭐⭐⭐⭐                                    |
| LLM               | ❌              | Possible, but PyTorch/HF often preferred |


## Scikit-learn learning roadmap
```
1. ML fundamentals
       ↓
2. Train/Test Split
       ↓
3. Linear Regression
       ↓
4. Logistic Regression
       ↓
5. Decision Trees
       ↓
6. Random Forest
       ↓
7. KNN
       ↓
8. SVM
       ↓
9. K-Means
       ↓
10. Scaling
       ↓
11. Encoding
       ↓
12. Missing values
       ↓
13. Feature engineering
       ↓
14. Pipelines
       ↓
15. Cross-validation
       ↓
16. Hyperparameter tuning
       ↓
17. Model evaluation
       ↓
18. XGBoost
```