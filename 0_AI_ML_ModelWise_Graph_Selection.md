# AI/ML Data Type, Problem Type, Model, Graph & Library

## Complete practical documentation for choosing models, metrics, graphs and Python libraries

---

# Table of Contents

1. Decision framework
2. Data types
3. Problem types
4. Data type → graphs matrix
5. Problem type → models matrix
6. Classification
7. Regression
8. Time series forecasting
9. Clustering
10. Dimensionality reduction
11. Anomaly detection
12. Recommendation systems
13. Ranking/search
14. NLP
15. Next-word prediction & LLM fine-tuning
16. Computer vision
17. Audio
18. Video
19. Geospatial
20. Graph/network ML
21. Reinforcement learning
22. Generative AI
23. Hyperparameter tuning
24. Explainable AI
25. Production monitoring
26. Library guide
27. Complete project workflows
28. Graph selection cheat sheet
29. Model selection cheat sheet
30. Official documentation

---

# 1. Decision Framework

Always select visualization using this sequence.

```text
DATA TYPE
    |
    v
PROBLEM TYPE
    |
    v
MODEL FAMILY
    |
    v
METRICS
    |
    v
GRAPHS
    |
    v
EXPLAINABILITY
    |
    v
PRODUCTION MONITORING
```

A graph is selected because it answers a machine-learning question.

Examples:

| Question | Best graph |
|---|---|
| What is the distribution? | Histogram |
| Are there outliers? | Box Plot |
| Are two variables related? | Scatter Plot |
| Are features correlated? | Correlation Heatmap |
| Is the model overfitting? | Train vs Validation Loss |
| Which class is misclassified? | Confusion Matrix |
| Is regression biased? | Residual Plot |
| Which features matter? | SHAP / Feature Importance |
| Did tuning improve the model? | Optimization History |
| Has production data changed? | Drift Distribution Plot |

---

# 2. Data Types in Machine Learning

## 2.1 Numerical Data

Examples:

- Age
- Salary
- Revenue
- Temperature
- Height
- Sensor values

Recommended graphs:

| Graph | Purpose |
|---|---|
| Histogram | Distribution |
| KDE | Density |
| Box Plot | Outliers |
| Violin Plot | Group comparison |
| Scatter | Relationship |
| Correlation Heatmap | Correlation |

Best libraries:

```text
NumPy
Pandas
Matplotlib
Seaborn
Plotly
Scikit-learn
```

---

## 2.2 Categorical Data

Examples:

- Gender
- Country
- Product
- Department
- Education

Graphs:

| Graph | Purpose |
|---|---|
| Count Plot | Frequency |
| Bar Chart | Comparison |
| Stacked Bar | Group comparison |
| Grouped Bar | Multiple categories |
| Pie/Donut | Small category proportions |

Libraries:

```text
Pandas
Seaborn
Matplotlib
Plotly
```

---

## 2.3 Ordinal Data

Examples:

```text
Low
Medium
High

Bronze
Silver
Gold
```

Use ordered bar charts.

Always preserve the logical order.

---

## 2.4 Mixed Tabular Data

Typical dataset:

| Age | Salary | City | Experience | Purchased |
|---|---|---|---|---|

Required EDA graphs:

```text
Histogram
Box Plot
Count Plot
Correlation Heatmap
Pair Plot
Missing Value Heatmap
Scatter Plot
```

Primary libraries:

```text
Pandas
Seaborn
Matplotlib
Scikit-learn
```

---

## 2.5 Time Series

Examples:

- Weather
- Stock prices
- Electricity
- Website traffic
- Temperature
- Sales

Required graphs:

| Graph | Purpose |
|---|---|
| Line Plot | Trend |
| Rolling Mean | Smoothed trend |
| Seasonal Plot | Seasonality |
| Lag Plot | Temporal relation |
| ACF | Autocorrelation |
| PACF | Partial autocorrelation |
| Forecast Plot | Prediction |

Libraries:

```text
Pandas
Matplotlib
Plotly
Statsmodels
Prophet
Scikit-learn
PyTorch
```

---

## 2.6 Text

Examples:

- Reviews
- Emails
- Chat
- Articles
- Tweets
- Documents

Graphs:

| Graph | Purpose |
|---|---|
| Token Frequency Bar | Vocabulary |
| Sequence Length Histogram | Tokenization analysis |
| Word Length Histogram | Text quality |
| Class Distribution | Labels |
| PCA | Embeddings |
| t-SNE | Embeddings |
| UMAP | Embeddings |

Libraries:

```text
Transformers
Datasets
Tokenizers
SpaCy
NLTK
Sentence-Transformers
Scikit-learn
```

---

## 2.7 Image

Examples:

- Face recognition
- OCR
- Medical images
- Document scanner
- Satellite images

Graphs:

```text
Sample Image Grid
Class Distribution
Image Width Histogram
Image Height Histogram
Aspect Ratio Histogram
Confusion Matrix
Bounding Box Visualization
Mask Overlay
```

Libraries:

```text
OpenCV
Pillow
Torchvision
PyTorch
TensorFlow
Matplotlib
```

---

## 2.8 Audio

Examples:

- Speech
- Music
- Voice assistant

Graphs:

```text
Waveform
Spectrogram
Mel Spectrogram
MFCC Heatmap
Duration Histogram
Sample Rate Distribution
```

Libraries:

```text
Librosa
Torchaudio
PyTorch
Matplotlib
```

---

## 2.9 Video

Graphs:

```text
Frame Grid
Duration Histogram
FPS Distribution
Motion Magnitude
Object Tracking Trajectory
```

Libraries:

```text
OpenCV
FFmpeg-python
Torchvision
PyTorch
```

---

## 2.10 Geospatial

Graphs:

```text
Scatter Map
Heat Map
Choropleth
Route Map
Hexbin Spatial Density
```

Libraries:

```text
GeoPandas
Folium
Plotly
Shapely
```

---

## 2.11 Graph / Network Data

Examples:

- Social networks
- Fraud graph
- Knowledge graph

Graphs:

```text
Network Graph
Degree Distribution
Centrality Bar
Community Visualization
```

Libraries:

```text
NetworkX
PyVis
iGraph
Plotly
```

---

# 3. Problem Types

| Problem | Output |
|---|---|
| Binary Classification | Yes/No |
| Multiclass Classification | One class |
| Multilabel Classification | Multiple labels |
| Regression | Continuous value |
| Time Series Forecasting | Future value |
| Clustering | Unknown groups |
| Dimensionality Reduction | Fewer dimensions |
| Anomaly Detection | Rare event |
| Recommendation | Suggested item |
| Ranking | Ordered results |
| Object Detection | Bounding boxes |
| Segmentation | Pixel classes |
| OCR | Extract text |
| NLP | Understand language |
| Next Token Prediction | Next word/token |
| Generative AI | Generate content |
| Reinforcement Learning | Reward policy |
| Survival Analysis | Time until event |

---

# 4. Data Type → Graph Matrix

| Data | Required graphs |
|---|---|
| Numerical | Histogram, Box, Scatter, Heatmap |
| Categorical | Count, Bar, Stacked Bar |
| Mixed Tabular | Histogram, Box, Pair Plot, Heatmap |
| Time Series | Line, Rolling Mean, ACF, PACF |
| Text | Token Bar, Length Histogram, Embedding Scatter |
| Image | Image Grid, Resolution Histogram, Class Bar |
| Audio | Waveform, Spectrogram, Mel Spectrogram |
| Video | Frame Grid, Motion Plot |
| Geospatial | Map, Choropleth, Density |
| Network | Network Graph, Degree Histogram |

---

# 5. Problem Type → Model Matrix

| Problem | Common models |
|---|---|
| Binary Classification | Logistic Regression, Random Forest, XGBoost, Neural Net |
| Multiclass | Random Forest, XGBoost, CNN, Transformer |
| Regression | Linear, Ridge, Lasso, Random Forest, XGBoost |
| Forecasting | ARIMA, SARIMA, Prophet, LSTM, Transformer |
| Clustering | KMeans, DBSCAN, Hierarchical, GMM |
| Anomaly | Isolation Forest, LOF, One-Class SVM, Autoencoder |
| Recommendation | ALS, Matrix Factorization, Neural CF |
| Ranking | BM25, LambdaMART, BERT Ranker |
| Object Detection | YOLO, SSD, Faster R-CNN, DETR |
| Segmentation | U-Net, DeepLab, Mask R-CNN |
| NLP | BERT, RoBERTa, DistilBERT |
| Next Token | RNN, LSTM, GRU, GPT |
| RL | DQN, PPO, A2C, SAC |

---

# 6. Classification Documentation

## Objective

Predict categories.

Examples:

```text
Spam / Not Spam
Fraud / Normal
Positive / Negative
Dog / Cat / Bird
```

## Models

Beginner:

```text
Logistic Regression
KNN
Decision Tree
Naive Bayes
Random Forest
```

Advanced:

```text
XGBoost
LightGBM
CatBoost
Neural Network
Transformer
```

## Required metrics

```text
Accuracy
Precision
Recall
F1 Score
ROC-AUC
PR-AUC
Log Loss
```

## Required graphs

| Graph | Why |
|---|---|
| Confusion Matrix | Error types |
| ROC Curve | Threshold performance |
| Precision-Recall Curve | Imbalanced data |
| Calibration Curve | Probability quality |
| Learning Curve | Under/overfitting |
| Validation Curve | Hyperparameter effect |
| Feature Importance | Model interpretation |
| SHAP | Explain predictions |

## How to read

Confusion matrix:

```text
                 Predicted
               Negative Positive

Actual Negative    TN       FP
Actual Positive    FN       TP
```

Interpretation:

- FP = false alarm
- FN = missed positive
- TP = correct positive
- TN = correct negative

---

# 7. Regression Documentation

## Objective

Predict continuous values.

Examples:

- House price
- Temperature
- Revenue
- Demand

## Models

```text
Linear Regression
Ridge
Lasso
ElasticNet
Random Forest Regressor
XGBoost Regressor
Neural Network
```

## Metrics

```text
MAE
MSE
RMSE
R²
MAPE
```

## Required graphs

| Graph | Purpose |
|---|---|
| Actual vs Predicted | Prediction quality |
| Residual Plot | Bias detection |
| Error Histogram | Error distribution |
| Q-Q Plot | Residual diagnostics |
| Learning Curve | Generalization |
| SHAP | Explainability |

## Reading residuals

Good:

```text
•  •   •
----0---------
  •   •
```

Bad:

```text
      •
    •
  •
•
```

Patterned residuals suggest missing structure.

---

# 8. Time Series Forecasting

## Problems

- Weather forecast
- Sales forecast
- Electricity demand
- Traffic prediction
- Temperature prediction

## Models

```text
ARIMA
SARIMA
Prophet
LSTM
GRU
Transformer
XGBoost with lag features
```

## Required graphs

| Graph | Purpose |
|---|---|
| Line Plot | Trend |
| Rolling Mean | Smoothing |
| Seasonal Decomposition | Trend + season |
| ACF | Lag relation |
| PACF | Direct lag effect |
| Actual vs Forecast | Evaluation |
| Residual Plot | Forecast errors |
| Confidence Interval | Forecast uncertainty |

## Workflow

```text
Raw Time Series
      |
      v
Line Plot
      |
      +--> Rolling Mean
      +--> Seasonality
      +--> ACF
      +--> PACF
      |
      v
Model
      |
      v
Forecast Plot
      |
      v
Residual Analysis
```

---

# 9. Clustering

## Objective

Find hidden groups.

Examples:

- Customer segmentation
- Document grouping
- Product clusters

## Models

```text
KMeans
DBSCAN
Hierarchical
Gaussian Mixture
```

## Metrics

```text
Silhouette Score
Davies-Bouldin
Calinski-Harabasz
```

## Required graphs

```text
Elbow Plot
Silhouette Plot
Cluster Scatter
PCA Scatter
UMAP
Dendrogram
```

---

# 10. Dimensionality Reduction

Methods:

```text
PCA
t-SNE
UMAP
```

Required graphs:

| Graph | Purpose |
|---|---|
| Explained Variance | PCA components |
| PCA Scatter | 2D visualization |
| t-SNE Scatter | Local clusters |
| UMAP Scatter | Embedding structure |

Use PCA for linear compression.

Use t-SNE or UMAP mainly for visualization.

---

# 11. Anomaly Detection

## Use cases

- Fraud
- Network intrusion
- Manufacturing defects
- Sensor failure

## Models

```text
Isolation Forest
Local Outlier Factor
One-Class SVM
Autoencoder
```

## Graphs

```text
Anomaly Score Histogram
Threshold Plot
Scatter with anomalies
ROC Curve
PR Curve
Time-series anomaly chart
```

Primary library:

```text
PyOD
Scikit-learn
Matplotlib
Plotly
```

---

# 12. Recommendation Systems

## Models

```text
Collaborative Filtering
Matrix Factorization
ALS
Neural Collaborative Filtering
```

## Metrics

```text
Precision@K
Recall@K
MAP
NDCG
Hit Rate
Coverage
```

## Graphs

```text
Precision@K Curve
Recall@K Curve
NDCG@K Curve
Popularity Distribution
User Activity Histogram
Item Interaction Heatmap
```

Libraries:

```text
Implicit
Surprise
PyTorch
Pandas
```

---

# 13. Ranking/Search Systems

Metrics:

```text
MRR
MAP
NDCG
Precision@K
Recall@K
```

Graphs:

```text
NDCG Comparison
MRR Bar
Query Score Distribution
Rank Position Histogram
```

Models:

```text
BM25
LambdaMART
Cross Encoder
BERT Ranker
```

---

# 14. NLP

## Dataset analysis

Graphs:

```text
Class Distribution
Sequence Length Histogram
Token Frequency
Vocabulary Size
Word Length Histogram
```

## Classification evaluation

```text
Confusion Matrix
Precision
Recall
F1
ROC
PR
```

## Embedding visualization

```text
PCA
t-SNE
UMAP
```

Libraries:

```text
Transformers
Datasets
Tokenizers
SpaCy
NLTK
Sentence-Transformers
Scikit-learn
```

---

# 15. Next Word Prediction & LLM Fine-Tuning

This is the recommended visualization set for RNN, LSTM, GRU and Transformer language models.

## Dataset graphs

```text
Sequence Length Histogram
Vocabulary Frequency
Token Frequency
Unknown Token Count
```

## Training graphs

```text
Train Loss
Validation Loss
Learning Rate
Gradient Norm
Weight Histogram
Activation Histogram
```

## Evaluation graphs

```text
Perplexity
Top-1 Accuracy
Top-5 Accuracy
Token Confidence Bar
Wrong Token Frequency
Error Category Bar
```

## Hyperparameter graphs

```text
Learning Rate vs Validation Loss
Batch Size vs Score
Sequence Length vs Score
Optimization History
Parameter Importance
Contour Plot
```

## Complete dashboard

```text
DATASET
   |
   +--> Sequence Histogram
   +--> Vocabulary
   |
TRAINING
   |
   +--> Train Loss
   +--> Eval Loss
   +--> Learning Rate
   +--> Gradient Norm
   |
EVALUATION
   |
   +--> Perplexity
   +--> Top-1
   +--> Top-5
   +--> Confidence
   |
TUNING
   |
   +--> Optimization History
   +--> Parameter Importance
```

Libraries:

```text
PyTorch
Transformers
PEFT
TRL
TensorBoard
MLflow
Weights & Biases
Optuna
```

---

# 16. Computer Vision

## Classification

Required graphs:

```text
Sample Grid
Class Distribution
Train Loss
Validation Loss
Accuracy Curve
Confusion Matrix
Grad-CAM
```

## Object Detection

Metrics:

```text
mAP
Precision
Recall
IoU
```

Graphs:

```text
Bounding Box Samples
PR Curve
Precision Curve
Recall Curve
IoU Histogram
Class AP Bar
```

## Segmentation

Graphs:

```text
Mask Overlay
Dice Score Curve
IoU Curve
Pixel Confusion Matrix
```

Libraries:

```text
OpenCV
Pillow
Torchvision
Ultralytics
PyTorch
```

---

# 17. Audio

Dataset graphs:

```text
Waveform
Spectrogram
Mel Spectrogram
MFCC Heatmap
Duration Histogram
```

Model evaluation:

```text
Loss Curve
Accuracy
Confusion Matrix
Embedding UMAP
```

Libraries:

```text
Librosa
Torchaudio
PyTorch
Matplotlib
```

---

# 18. Video

Graphs:

```text
Frame Sampling Grid
FPS Histogram
Duration Histogram
Object Trajectory
Motion Magnitude
```

Libraries:

```text
OpenCV
FFmpeg-python
PyTorch
```

---

# 19. Geospatial ML

Graphs:

```text
Scatter Map
Choropleth
Density Heatmap
Route Visualization
Hexbin Density
```

Libraries:

```text
GeoPandas
Folium
Plotly
Shapely
```

---

# 20. Graph / Network ML

Use cases:

- Fraud network
- Social graph
- Knowledge graph

Graphs:

```text
Network Graph
Degree Histogram
Centrality Bar
Community Detection Plot
```

Libraries:

```text
NetworkX
PyVis
iGraph
```

---

# 21. Reinforcement Learning

Algorithms:

```text
Q-Learning
DQN
PPO
A2C
SAC
```

Required graphs:

| Graph | Purpose |
|---|---|
| Episode Reward | Main learning metric |
| Moving Average Reward | Smoothed progress |
| Episode Length | Policy behavior |
| Policy Loss | Actor optimization |
| Value Loss | Critic optimization |
| Entropy | Exploration |
| Epsilon | Exploration schedule |

Libraries:

```text
Gymnasium
Stable-Baselines3
PyTorch
TensorBoard
MLflow
```

---

# 22. Generative AI

## Text generation

Graphs:

```text
Train Loss
Eval Loss
Perplexity
Reward Score
Latency
Token Usage
Human Preference Score
```

## Image generation

Graphs:

```text
Generator Loss
Discriminator Loss
FID
IS
Generated Sample Grid
Embedding Visualization
```

Libraries:

```text
PyTorch
Diffusers
Transformers
MLflow
TensorBoard
```

---

# 23. Hyperparameter Tuning

## Hyperparameters

```text
learning_rate
batch_size
epochs
dropout
hidden_size
num_layers
weight_decay
sequence_length
tree_depth
n_estimators
```

## Required graphs

| Graph | Purpose |
|---|---|
| Optimization History | Trial improvement |
| Validation Curve | One parameter |
| Slice Plot | Parameter effect |
| Contour Plot | Two parameter interaction |
| Parallel Coordinate | Multiple parameters |
| Heatmap | Grid search |
| Parameter Importance | Most influential parameter |

Libraries:

```text
Optuna
Scikit-learn
Plotly
Matplotlib
```

---

# 24. Explainable AI

## Feature Importance

Best for tree models.

Graph:

```text
Horizontal Bar Chart
```

## Permutation Importance

Graph:

```text
Performance Drop
```

## SHAP

| Plot | Use |
|---|---|
| Summary | Global importance |
| Bar | Ranking |
| Waterfall | Single prediction |
| Dependence | Feature relationship |
| Force | Local explanation |

Libraries:

```text
SHAP
LIME
ELI5
Scikit-learn
```

---

# 25. Production Monitoring

Monitor continuously after deployment.

## Data drift

Graphs:

```text
Histogram Comparison
KDE Comparison
Feature Distribution Trend
Missing Value Trend
PSI
```

## Prediction drift

Graphs:

```text
Probability Distribution
Prediction Histogram
Class Percentage Trend
```

## Model performance

Graphs:

```text
Accuracy over Time
F1 over Time
RMSE over Time
Latency
Error Rate
Throughput
```

Libraries:

```text
MLflow
Evidently
Prometheus
Grafana
Plotly
```

---

# 26. Library Guide

## Core

| Library | Purpose |
|---|---|
| NumPy | Arrays |
| Pandas | DataFrames |
| Polars | Fast DataFrames |
| SciPy | Scientific computing |
| Statsmodels | Statistics |

## Visualization

| Library | Best use |
|---|---|
| Matplotlib | Foundation plotting |
| Seaborn | Statistical plots |
| Plotly | Interactive charts |
| Altair | Declarative visualization |
| Bokeh | Interactive dashboards |

## ML

| Library | Use |
|---|---|
| Scikit-learn | ML, metrics, CV |
| XGBoost | Boosting |
| LightGBM | Fast boosting |
| CatBoost | Categorical features |
| imbalanced-learn | Imbalanced classes |

## Deep Learning

| Library | Use |
|---|---|
| PyTorch | Deep learning |
| TensorFlow | Deep learning |
| Keras | High-level DL |
| Torchvision | Vision |
| Torchaudio | Audio |

## NLP

| Library | Use |
|---|---|
| Transformers | LLMs |
| Datasets | Datasets |
| Tokenizers | Tokenization |
| SpaCy | NLP pipeline |
| NLTK | Classical NLP |
| Sentence-Transformers | Embeddings |

## Experiment tracking

| Library | Use |
|---|---|
| MLflow | Lifecycle |
| TensorBoard | Training curves |
| Weights & Biases | Team experiments |

## Explainability

```text
SHAP
LIME
ELI5
```

## Monitoring

```text
Evidently
Prometheus
Grafana
MLflow
```

---

# 27. Complete Project Workflows

## Tabular Classification

```text
CSV
 |
 v
Histogram + Box + Heatmap
 |
 v
Preprocessing
 |
 v
Random Forest / XGBoost
 |
 v
Confusion Matrix
 |
 +--> ROC
 +--> PR
 +--> Calibration
 |
 v
SHAP
 |
 v
MLflow
```

## Regression

```text
CSV
 |
 v
Histogram + Scatter + Heatmap
 |
 v
Regression Model
 |
 v
Actual vs Predicted
 |
 v
Residual Plot
 |
 v
Error Histogram
 |
 v
SHAP
```

## Time Series

```text
Date + Value
 |
 v
Line Plot
 |
 +--> Rolling Mean
 +--> Seasonality
 +--> ACF
 +--> PACF
 |
 v
ARIMA / LSTM
 |
 v
Forecast Plot
 |
 v
Residual Analysis
```

## Sentiment Analysis

```text
Reviews
 |
 v
Sequence Histogram
 |
 v
Tokenizer
 |
 v
BERT
 |
 +--> Train Loss
 +--> Eval Loss
 +--> Confusion
 +--> F1
 +--> PR
 |
 v
SHAP
```

## Next Word Prediction

```text
Corpus
 |
 v
Vocabulary
 |
 v
Sequence Length Histogram
 |
 v
LSTM / Transformer
 |
 +--> Train Loss
 +--> Validation Loss
 +--> Learning Rate
 +--> Gradient Norm
 |
 v
Perplexity
 |
 +--> Top-1
 +--> Top-5
 +--> Confidence Distribution
 +--> Wrong Token Analysis
```

## Object Detection

```text
Images
 |
 v
Sample Grid
 |
 v
YOLO
 |
 +--> Loss
 +--> Precision
 +--> Recall
 +--> PR Curve
 +--> IoU
 +--> mAP
 |
 v
Bounding Box Review
```

---

# 28. Graph Selection Cheat Sheet

| If you need to know... | Use this graph |
|---|---|
| Distribution | Histogram |
| Density | KDE |
| Outliers | Box Plot |
| Group distribution | Violin |
| Category count | Bar / Count |
| Relationship | Scatter |
| Correlation | Heatmap |
| Trend | Line |
| Seasonality | Seasonal Plot |
| Lag relationship | ACF/PACF |
| Overfitting | Train vs Validation Loss |
| Classification errors | Confusion Matrix |
| Threshold behavior | ROC |
| Imbalanced classification | PR Curve |
| Probability reliability | Calibration |
| Regression quality | Actual vs Predicted |
| Regression bias | Residual |
| Cluster count | Elbow |
| Cluster quality | Silhouette |
| Embeddings | PCA / t-SNE / UMAP |
| Hyperparameter search | Optimization History |
| Parameter interaction | Contour |
| Feature explanation | SHAP |
| Production drift | Distribution comparison |
| RL learning | Reward Curve |
| LLM quality | Eval Loss + Perplexity |

---

# 29. Model Selection Cheat Sheet

| Data | Problem | Start with |
|---|---|---|
| Tabular | Classification | Logistic Regression, Random Forest |
| Tabular | Regression | Linear Regression, Random Forest |
| Tabular | Imbalanced | XGBoost + class weighting |
| Time Series | Forecast | ARIMA / Prophet |
| Long sequence | Forecast | LSTM / Transformer |
| Text | Classification | BERT |
| Text | Next token | GPT / LSTM |
| Image | Classification | CNN / ResNet |
| Image | Detection | YOLO |
| Image | Segmentation | U-Net |
| Audio | Classification | CNN on spectrogram |
| Recommendation | User-item | ALS |
| Network | Graph prediction | GNN |
| RL | Control | PPO / DQN |

---

# 30. Official Documentation

Core visualization and ML libraries:

- Matplotlib: https://matplotlib.org/stable/
- Seaborn: https://seaborn.pydata.org/
- Plotly Python: https://plotly.com/python/
- Scikit-learn: https://scikit-learn.org/stable/
- Scikit-learn Visualizations: https://scikit-learn.org/stable/visualizations.html
- Optuna Visualization: https://optuna.readthedocs.io/en/latest/reference/visualization/
- SHAP: https://shap.readthedocs.io/
- MLflow: https://mlflow.org/docs/latest/
- TensorBoard: https://www.tensorflow.org/tensorboard
- PyTorch: https://pytorch.org/
- Hugging Face Transformers: https://huggingface.co/docs/transformers/
- Evidently: https://docs.evidentlyai.com/
- Yellowbrick: https://www.scikit-yb.org/

---

# Final Rule

Use this formula in every project:

```text
DATA TYPE
      +
PROBLEM TYPE
      +
MODEL TYPE
      +
METRIC
      =
CORRECT GRAPH
      +
CORRECT LIBRARY
```

Examples:

```text
Mixed Tabular + Classification
→ Random Forest
→ Confusion Matrix + ROC + PR + SHAP
→ Pandas + Seaborn + Scikit-learn + SHAP

Time Series + Forecasting
→ ARIMA / LSTM
→ Line + ACF + PACF + Forecast Plot
→ Pandas + Statsmodels + Matplotlib

Text + Next Word Prediction
→ LSTM / Transformer
→ Train Loss + Eval Loss + Perplexity + Top-K
→ PyTorch + Transformers + TensorBoard + MLflow
```

This guide should be used as a practical reference whenever starting a new AI/ML project.
