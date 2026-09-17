# AI/ML Model Training, Evaluation, Tuning & Visualization

## Table of Contents

1. [Introduction](#1-introduction)
2. [Why Graphs Matter in AI/ML](#2-why-graphs-matter-in-aiml)
3. [ML Lifecycle](#3-ml-lifecycle)
4. [Quick Reference: Major Plot Types](#4-quick-reference-major-plot-types)
5. [Data Exploration Visualizations](#5-data-exploration-visualizations)
   - Histogram
   - KDE Plot
   - Box Plot
   - Violin Plot
   - Bar Chart
   - Scatter Plot
   - Correlation Heatmap
6. [Training Visualizations](#6-training-visualizations)
   - Training Loss
   - Validation Loss
   - Training vs Validation Loss
   - Training Accuracy
   - Validation Accuracy
   - Learning Curves
   - Underfitting
   - Good Fit
   - Overfitting
7. [Classification Evaluation](#7-classification-evaluation)
   - Confusion Matrix
   - Accuracy
   - Precision
   - Recall
   - F1 Score
   - ROC Curve
   - AUC
   - Precision-Recall Curve
   - Calibration Curve
8. [Regression Evaluation](#8-regression-evaluation)
   - Actual vs Predicted
   - Residual Plot
   - Error Distribution
   - Q-Q Plot
   - R², MAE, MSE and RMSE
9. [Cross-Validation Visualizations](#9-cross-validation-visualizations)
10. [Hyperparameter Tuning Visualizations](#10-hyperparameter-tuning-visualizations)
11. [Feature Importance & Explainability](#11-feature-importance--explainability)
12. [Clustering Visualizations](#12-clustering-visualizations)
13. [Dimensionality Reduction](#13-dimensionality-reduction)
14. [Deep Learning Diagnostics](#14-deep-learning-diagnostics)
15. [NLP, RNNs, Transformers & LLM Fine-Tuning](#15-nlp-rnns-transformers--llm-fine-tuning)
16. [Next-Word Prediction: Practical Visualization Dashboard](#16-next-word-prediction-practical-visualization-dashboard)
17. [Production ML Monitoring](#17-production-ml-monitoring)
18. [How to Read Any ML Graph](#18-how-to-read-any-ml-graph)
19. [Diagnostic Combinations](#19-diagnostic-combinations)
20. [Recommended Learning Roadmap](#20-recommended-learning-roadmap)
21. [Common Mistakes](#21-common-mistakes)
22. [Final Cheat Sheet](#22-final-cheat-sheet)

---

# 1. Introduction

Graphs and plots are essential tools for understanding machine-learning systems.

A metric tells you **what happened**.

A visualization often helps explain **why it happened**.

For example:

```text
Training accuracy   = 94%
Validation accuracy = 78%
```

The numbers show a gap, but a training/validation curve can reveal whether the gap developed gradually, suddenly, or after a particular epoch.

In practical ML work, visualizations help you:

- Understand your dataset.
- Find outliers.
- Understand feature distributions.
- Discover relationships between variables.
- Monitor model training.
- Detect underfitting.
- Detect overfitting.
- Compare training and validation performance.
- Evaluate classification models.
- Evaluate regression models.
- Tune hyperparameters.
- Understand model predictions.
- Analyze embeddings.
- Diagnose neural networks.
- Monitor production models.
- Analyze model failures.

The goal is not to memorize every plot.

The goal is to develop the ability to look at a graph and answer:

> What does this graph tell me about my data or model, what problem is visible, and what should I investigate next?

---

# 2. Why Graphs Matter in AI/ML

Consider a neural network that produces:

```text
Training loss      = 0.12
Validation loss    = 0.75
Training accuracy  = 97%
Validation accuracy = 78%
```

This suggests a generalization problem.

A graph can make the behavior obvious:

```text
Loss
 |
 |\
 | \
 |  \
 |   \          Training
 |    \_________
 |
 |      \____
 |           \__
 |              \__
 +----------------------> Epoch
          Validation
```

The model is improving on training data while validation performance eventually worsens.

This is a classic overfitting pattern.

---

# 3. ML Lifecycle

A useful mental model is:

```text
DATA
  |
  v
EXPLORATION
  |
  v
PREPROCESSING
  |
  v
TRAINING
  |
  v
EVALUATION
  |
  +------------------+
  |                  |
  v                  |
DIAGNOSIS            |
  |                  |
  v                  |
HYPERPARAMETER ------+
TUNING
  |
  v
FINE-TUNING
  |
  v
FINAL EVALUATION
  |
  v
DEPLOYMENT
  |
  v
MONITORING
```

Different graphs become useful at different stages.

---

# 4. Quick Reference: Major Plot Types

| Category | Important plots |
|---|---|
| Data exploration | Histogram, KDE, Box Plot, Violin Plot |
| Feature relationships | Scatter Plot, Pair Plot, Correlation Heatmap |
| Classification | Confusion Matrix, ROC, PR Curve, Calibration |
| Regression | Actual vs Predicted, Residual Plot, Error Distribution, Q-Q Plot |
| Training | Loss Curve, Accuracy Curve, Learning Curve |
| Hyperparameter tuning | Parameter-vs-score plot, Heatmap, Optimization History |
| Cross-validation | Fold Score Plot, CV Distribution |
| Explainability | Feature Importance, Permutation Importance, SHAP |
| Clustering | Elbow Plot, Silhouette Plot |
| Dimensionality reduction | PCA, t-SNE, UMAP |
| Deep learning | Loss, Accuracy, Learning Rate, Gradient Norm, Weight/Activation Distributions |
| NLP/LLM | Train Loss, Eval Loss, Perplexity, Top-K Accuracy, LR Schedule |
| Production | Data Drift, Prediction Drift, Performance Over Time |

---

# 5. Data Exploration Visualizations

Before training a model, understand your data.

## 5.1 Histogram

### Purpose

A histogram shows the distribution of a numerical variable.

Typical examples:

- Age
- Salary
- Price
- Temperature
- House area
- Model errors

### Python

```python
import matplotlib.pyplot as plt

plt.hist(df["age"], bins=20)
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.title("Age Distribution")
plt.show()
```

### How to read it

Look for:

- Center
- Spread
- Skewness
- Multiple peaks
- Outliers
- Long tails

A roughly symmetric distribution:

```text
frequency
   |
   |       ███
   |      █████
   |    ████████
   |  ██████████
   | ███████████
   +----------------
       value
```

A right-skewed distribution:

```text
████████
██████
████
██
█
-------------------->
```

### ML use

Histograms help identify:

- Feature distributions.
- Highly skewed features.
- Suspicious values.
- Potential transformations.
- Differences between train/validation/test data.

---

## 5.2 KDE Plot

KDE means **Kernel Density Estimation**.

Instead of bars, it provides a smooth estimate of a distribution.

Conceptually:

```text
density
  |
  |       /\
  |      /  \
  |_____/    \_____
  +---------------->
```

KDE is useful for comparing distributions.

For example:

```text
Training distribution
Validation distribution
Test distribution
```

If these distributions are substantially different, investigate possible distribution shift.

---

## 5.3 Box Plot

A box plot summarizes:

- Minimum
- First quartile (Q1)
- Median
- Third quartile (Q3)
- Maximum
- Outliers

Conceptually:

```text
       |
   ----+----
      | |
      | |
   +--+---+--+
   |         |
   +---------+
       |
       |
```

### Why it matters

It is particularly useful for outlier detection.

Example:

```text
Salary:

30k
35k
40k
42k
45k
48k
50k
5000k   <-- suspicious value
```

A box plot can make this obvious.

---

## 5.4 Violin Plot

A violin plot combines:

- Distribution
- Density
- Central tendency

Conceptually:

```text
       ██
      ████
     ██████
     ██████
      ████
       ██
```

It is useful when comparing distributions across several groups.

---

## 5.5 Bar Chart

Bar charts are best for categorical comparisons.

Examples:

```text
Model A   ████████
Model B   ██████████
Model C   █████████
```

Typical ML uses:

- Class counts.
- Feature importance.
- Model metric comparison.
- Category frequencies.

Do not use a bar chart when the primary purpose is to show continuous time-series behavior; a line chart is usually more appropriate.

---

## 5.6 Scatter Plot

A scatter plot shows the relationship between two numerical variables.

Example:

```text
X = Study Hours
Y = Exam Score
```

Conceptually:

```text
Score
  |
90|              *
80|           *
70|        *
60|     *
50|  *
  +--------------------> Hours
```

### Interpretation

Upward pattern:

```text
Positive relationship
```

Downward pattern:

```text
Negative relationship
```

Random cloud:

```text
Weak/no obvious linear relationship
```

### Important

Correlation does not automatically mean causation.

---

## 5.7 Correlation Heatmap

Correlation measures the strength and direction of a relationship.

Typical range:

```text
+1  strong positive linear relationship
 0  no linear relationship
-1  strong negative linear relationship
```

Example:

```text
              Age Salary Experience
Age            1    .72     .65
Salary        .72    1      .82
Experience    .65   .82      1
```

A heatmap makes this easier to inspect visually.

### ML uses

- Detect redundant features.
- Identify multicollinearity.
- Discover relationships.
- Help with feature selection.

### Important limitation

Correlation mainly captures linear relationships. A nonlinear relationship can have low Pearson correlation.

---

# 6. Training Visualizations

Training graphs are among the most important graphs in deep learning.

---

## 6.1 Training Loss Curve

Loss represents how wrong the model is according to its objective function.

Example:

```text
Epoch    Loss

1        1.20
2        0.95
3        0.72
4        0.55
5        0.43
6        0.36
7        0.31
```

Conceptually:

```text
Loss
 |
 |\
 | \
 |  \
 |   \____
 +--------------> Epoch
```

### Interpretation

Generally:

```text
Loss decreasing
       |
       v
Model is learning the training objective
```

A loss curve that becomes noisy or unstable can indicate:

- Learning rate is too high.
- Batch size is small.
- Data is noisy.
- Optimization is unstable.

---

## 6.2 Validation Loss

Validation loss measures performance on data not used to update the model's weights.

It is an important indicator of generalization.

Healthy behavior:

```text
Training loss     ↓
Validation loss   ↓
```

Potential overfitting:

```text
Training loss     ↓↓↓↓↓
Validation loss      ↓ ↑ ↑ ↑
```

---

## 6.3 Training vs Validation Loss

This is one of the most important deep-learning graphs.

Example:

```text
Epoch     Train Loss    Val Loss

1            1.20         1.15
2            0.90         0.95
3            0.68         0.78
4            0.51         0.67
5            0.40         0.61
6            0.33         0.64
7            0.27         0.71
8            0.23         0.81
```

Interpretation:

```text
Epoch 1-5:
Both improve.

After epoch 5:
Training loss continues decreasing.
Validation loss starts increasing.
```

This indicates overfitting.

### Practical action

The best checkpoint may be around the epoch where validation loss is lowest, rather than the final epoch.

This is often combined with **early stopping**.

---

## 6.4 Training Accuracy

For classification:

```text
Epoch
1 -> 60%
2 -> 68%
3 -> 74%
4 -> 81%
5 -> 86%
```

Accuracy is:

```text
Correct predictions
-------------------
Total predictions
```

---

## 6.5 Validation Accuracy

Validation accuracy shows how the model performs on held-out validation data.

A common overfitting pattern:

```text
Training accuracy     -> 99%
Validation accuracy   -> 78%
```

A large persistent gap deserves investigation.

---

## 6.6 Training vs Validation Accuracy

Healthy:

```text
Training accuracy     ↑
Validation accuracy   ↑
```

Overfitting:

```text
Training accuracy     ↑↑↑
Validation accuracy   → or ↓
```

---

## 6.7 Underfitting

Typical pattern:

```text
Training performance     poor
Validation performance   poor
```

Possible solutions:

- Increase model capacity.
- Train longer.
- Improve features.
- Reduce excessive regularization.
- Improve architecture.
- Improve data quality.

---

## 6.8 Good Fit

Typical pattern:

```text
Training performance     good
Validation performance   good
Gap                      reasonable
```

---

## 6.9 Overfitting

Typical pattern:

```text
Training performance     excellent
Validation performance   substantially worse
```

Possible solutions:

- More training data.
- Data augmentation.
- Regularization.
- Dropout.
- Weight decay.
- Early stopping.
- Reduce model complexity.
- Better validation methodology.

---

# 7. Classification Evaluation

Classification models require more than accuracy.

---

## 7.1 Confusion Matrix

For binary classification:

```text
                 Predicted
               Negative Positive

Actual Negative    TN       FP
Actual Positive    FN       TP
```

Example:

```text
                 Predicted
                 0       1

Actual 0        850      50
Actual 1         30      70
```

Therefore:

```text
TN = 850
FP = 50
FN = 30
TP = 70
```

### Why it matters

It shows the types of mistakes your model makes.

---

## 7.2 Accuracy

```text
Accuracy = (TP + TN) / Total
```

Example:

```text
(70 + 850) / 1000
= 92%
```

### Important warning

Accuracy can be misleading with imbalanced classes.

Suppose:

```text
99% = negative
1%  = positive
```

A model that always predicts negative gets:

```text
99% accuracy
```

but:

```text
Positive recall = 0%
```

Therefore, evaluate the metric that matches the problem.

---

## 7.3 Precision

```text
Precision = TP / (TP + FP)
```

Question:

> When the model predicts positive, how often is it correct?

Precision matters when false positives are costly.

Example:

- Spam detection.
- Recommendation filtering.
- Certain alerting systems.

---

## 7.4 Recall

```text
Recall = TP / (TP + FN)
```

Question:

> Of all actual positive cases, how many did the model find?

Recall matters when false negatives are costly.

---

## 7.5 F1 Score

```text
F1 = 2 * Precision * Recall
     ------------------------
       Precision + Recall
```

F1 balances precision and recall.

It can be useful when both types of error matter and class distribution is not adequately summarized by accuracy.

---

## 7.6 ROC Curve

ROC = Receiver Operating Characteristic.

It plots:

```text
True Positive Rate
against
False Positive Rate
```

Conceptually:

```text
TPR
1 |          ______
  |       __/
  |     _/
  |   _/
  |__/
  +--------------------> FPR
```

The ROC curve shows performance across different classification thresholds.

---

## 7.7 AUC

AUC = Area Under the ROC Curve.

AUC measures ranking/separation ability across thresholds.

Rough interpretation:

```text
AUC near 1.0 -> strong discrimination
AUC near 0.5 -> roughly random ranking
```

The exact usefulness of AUC depends on the task, class distribution, and decision costs.

---

## 7.8 Precision-Recall Curve

A Precision-Recall curve shows the trade-off between:

```text
Precision
Recall
```

It is particularly useful for imbalanced classification problems where the positive class is rare.

Example:

```text
Precision
1 |\
  | \
  |  \
  |   \____
  +----------------> Recall
```

As the threshold changes, recall often increases while precision decreases.

---

## 7.9 Calibration Curve

A classification model can output confidence.

For example:

```text
Confidence = 90%
```

Calibration asks whether predictions made with approximately 90% confidence are correct approximately 90% of the time.

A well-calibrated model:

```text
Predicted probability ≈ Observed frequency
```

Calibration is particularly relevant when predicted probabilities are used for decisions or risk estimates.

---

# 8. Regression Evaluation

Regression predicts continuous values.

Examples:

- Temperature.
- House price.
- Revenue.
- Demand.
- Sensor readings.

---

## 8.1 Actual vs Predicted Plot

Ideal behavior:

```text
Predicted
   |
40 |          *
35 |       *
30 |    *
25 | *
   +-----------------> Actual
```

Points should generally lie near:

```text
Predicted = Actual
```

Large deviations indicate larger errors.

---

## 8.2 Residual Plot

Residual:

```text
Residual = Actual - Predicted
```

Good residual pattern:

```text
Residual
  +
  |  •   •
0 |--------------------
  |    •   •
  -
  +--------------------> Prediction
```

Residuals should generally be randomly scattered around zero.

Bad pattern:

```text
       •
     •
   •
 •
```

A systematic pattern can indicate that the model is missing structure.

---

## 8.3 Error Distribution

Plot the distribution of:

```text
Actual - Predicted
```

Look for:

- Center around zero.
- Skew.
- Heavy tails.
- Outliers.

A large tail may indicate occasional severe prediction errors.

---

## 8.4 Q-Q Plot

A Q-Q plot compares the quantiles of your data against a theoretical distribution, often the normal distribution.

It is useful for examining whether residuals approximately follow an assumed distribution.

Approximate straight line:

```text
        /
      /
    /
  /
 /
```

Strong deviations from the line indicate deviations from the assumed distribution.

Do not automatically require normally distributed residuals for every ML model; whether normality matters depends on the model and the statistical assumptions involved.

---

## 8.5 Common Regression Metrics

### MAE

```text
MAE = mean(|Actual - Predicted|)
```

Easy to interpret.

### MSE

```text
MSE = mean((Actual - Predicted)^2)
```

Penalizes large errors more strongly.

### RMSE

```text
RMSE = sqrt(MSE)
```

Same units as the target.

### R²

Measures how much variation in the target is explained relative to a baseline formulation.

Important:

> R² should not be interpreted as a universal "percentage of correctness."

---

# 9. Cross-Validation Visualizations

Cross-validation tests how stable model performance is across multiple train/validation splits.

Example:

```text
Fold 1 = 0.88
Fold 2 = 0.91
Fold 3 = 0.87
Fold 4 = 0.90
Fold 5 = 0.89
```

A fold score plot can show:

```text
Fold 1  █████████
Fold 2  ██████████
Fold 3  ████████
Fold 4  █████████
Fold 5  █████████
```

### What to look for

You generally want:

- Strong average performance.
- Reasonably stable performance.
- No suspiciously large fold-to-fold variation.

Large variation may indicate:

- Small dataset.
- Heterogeneous data.
- Unstable model.
- Poor splitting strategy.
- Leakage problems.

Always choose a cross-validation strategy appropriate to the data. For example, time-series data should not generally be evaluated using ordinary random K-fold splitting.

---

# 10. Hyperparameter Tuning Visualizations

Hyperparameters are settings chosen outside the model's learned parameters.

Examples:

```text
learning_rate
batch_size
dropout
number_of_layers
hidden_size
tree_depth
number_of_estimators
```

---

## 10.1 Hyperparameter vs Validation Score

Example:

```text
Learning Rate     Validation Accuracy

0.0001            0.72
0.0005            0.79
0.001             0.84
0.003             0.87
0.01              0.76
0.03              0.61
```

Interpretation:

```text
Very small LR
    |
    v
Slow learning

Middle region
    |
    v
Better validation result in this experiment

Very large LR
    |
    v
Potential instability / poorer optimization
```

Do not assume one globally optimal learning rate exists; it depends on architecture, optimizer, data, batch size, schedule, and training setup.

---

## 10.2 Hyperparameter Heatmap

Suppose you vary learning rate and batch size:

```text
                Batch Size
              16    32    64

LR 0.0001     .72   .74   .71
LR 0.001      .81   .86   .84
LR 0.01       .75   .78   .70
```

A heatmap helps identify promising regions.

Useful combinations include:

```text
Learning Rate × Batch Size
Depth × Number of Estimators
Dropout × Learning Rate
Hidden Size × Sequence Length
```

---

## 10.3 Optimization History

Tools such as Optuna can perform many trials.

Example:

```text
Trial 1 -> 0.71
Trial 2 -> 0.76
Trial 3 -> 0.81
Trial 4 -> 0.78
Trial 5 -> 0.85
...
```

A plot of trial number versus validation score helps show whether optimization is finding better configurations.

---

## 10.4 Hyperparameter Tuning Strategy

Common approaches:

### Grid Search

Try all combinations from a predefined grid.

### Random Search

Sample combinations randomly.

### Bayesian Optimization

Use previous results to choose promising future trials.

### Optuna

A practical optimization framework that can use advanced samplers and pruning.

---

# 11. Feature Importance & Explainability

Model performance alone does not tell you what the model uses.

---

## 11.1 Feature Importance

Tree-based models often expose feature importance.

Example:

```text
Feature          Importance

income           ███████████
age              ████████
experience       ██████
education        ████
city             ██
```

It answers:

> Which features contributed to the model's decision according to this importance method?

Commonly available for:

- Decision Trees.
- Random Forest.
- Gradient boosting models.
- XGBoost.
- LightGBM.

### Important

Feature importance does not mean:

```text
Feature -> causes target
```

Importance is not causality.

---

## 11.2 Permutation Importance

Process:

1. Train the model.
2. Shuffle one feature.
3. Measure the performance change.
4. Repeat.

Example:

```text
Feature       Performance Drop

income        0.14
age           0.08
experience    0.06
city          0.01
```

A large drop suggests the model depends strongly on that feature under the chosen evaluation setup.

---

## 11.3 SHAP

SHAP provides model explanation based on Shapley-value concepts.

Common visualizations:

- SHAP summary plot.
- SHAP bar plot.
- SHAP dependence plot.
- SHAP waterfall plot.
- SHAP force plot.

A simplified interpretation:

```text
income       -> prediction +0.42
age          -> prediction +0.21
experience   -> prediction +0.11
debt         -> prediction -0.25
```

It can answer:

> Which features pushed this prediction higher or lower relative to a baseline?

---

# 12. Clustering Visualizations

Clustering is unsupervised learning.

Examples:

- Customer segmentation.
- Document grouping.
- User behavior analysis.

---

## 12.1 Elbow Plot

For K-Means, you can plot the within-cluster objective/inertia against K.

Example:

```text
K = 2 -> 900
K = 3 -> 600
K = 4 -> 400
K = 5 -> 380
K = 6 -> 370
```

Conceptually:

```text
Inertia
 |
900| *
   |  \
600|   *
   |    \
400|     *
   |      \__
370|         ***
   +-----------------> K
```

The "elbow" is a point where additional clusters provide diminishing improvement.

It is a heuristic, not a proof of the correct number of clusters.

---

## 12.2 Silhouette Score

Silhouette measures how similar an observation is to its own cluster compared with other clusters.

Approximate interpretation:

```text
+1 -> well separated
 0 -> overlapping / boundary region
-1 -> may be assigned poorly
```

Use it as one diagnostic, not as the only criterion for deciding cluster count.

---

# 13. Dimensionality Reduction

High-dimensional data cannot easily be visualized directly.

Common methods:

```text
PCA
t-SNE
UMAP
```

---

## 13.1 PCA Explained Variance

Suppose:

```text
PC1 = 42%
PC2 = 21%
PC3 = 12%
PC4 = 7%
```

Cumulative explained variance:

```text
Variance
100 |                  ______
 80 |            _____/
 60 |       ____/
 40 |   ___/
 20 |__/
    +----------------------> Components
```

This helps determine how much variance is retained as components are added.

---

## 13.2 PCA 2D Scatter Plot

Example:

```text
Original data
100 dimensions
       |
       v
     PCA
       |
       v
2 dimensions
```

Then visualize:

```text
        Cluster A
       • • •
      • • • •

                    Cluster B
                   • • •
                  • • •
```

Useful for visually inspecting separation.

---

## 13.3 t-SNE

t-SNE is commonly used for visualizing high-dimensional representations.

For example:

```text
       A A A
      A A A

                    B B B
                   B B B

     C C C
```

It can reveal local grouping.

### Important caution

Do not interpret t-SNE axes as meaningful physical dimensions.

Do not assume that every apparent distance or cluster size has a direct quantitative interpretation.

---

## 13.4 UMAP

UMAP is another dimensionality-reduction technique.

It is commonly used for:

- Embeddings.
- NLP.
- Computer vision.
- Clustering.
- Large datasets.

It often provides useful visualization while scaling better than t-SNE in many situations.

---

# 14. Deep Learning Diagnostics

Neural networks provide additional diagnostic plots.

---

## 14.1 Training Loss

Answers:

> Is the optimization objective improving on the training data?

---

## 14.2 Validation Loss

Answers:

> Is the model generalizing to held-out validation data?

---

## 14.3 Learning Rate Curve

Learning-rate schedules may look like:

```text
Learning Rate
 |
 |      /\
 |     /  \
 |____/    \________
 +----------------------> Steps
```

This could represent a warmup followed by decay.

Useful schedules include:

- Warmup.
- Cosine decay.
- One-cycle policy.
- ReduceLROnPlateau.
- Step decay.

---

## 14.4 Gradient Norm

A gradient norm plot can help identify optimization problems.

Conceptually:

```text
Gradient Norm
 |
 |     /\
 |    /  \
 |___/    \____
 +----------------> Steps
```

Potential problems:

### Vanishing gradients

```text
Gradient -> near zero
```

### Exploding gradients

```text
Gradient -> extremely large
```

This is particularly relevant to deep networks and historically important for RNNs.

---

## 14.5 Weight Distribution

Plot weight distributions for layers.

Useful for detecting:

- Weight explosion.
- Weight collapse.
- Unusual parameter distributions.
- Poor initialization.
- Training instability.

---

## 14.6 Activation Distribution

Inspect activations from layers.

Useful for diagnosing:

- Dead ReLU neurons.
- Saturation.
- Poor normalization.
- Distribution changes through the network.

---

# 15. NLP, RNNs, Transformers & LLM Fine-Tuning

This section is especially important for next-word prediction and language-model work.

---

## 15.1 Training Loss vs Steps

Language models are commonly monitored using training steps.

Example:

```text
Step 100  -> loss 2.8
Step 200  -> loss 2.4
Step 300  -> loss 2.1
```

Plot:

```text
Loss
 |
 |\
 | \
 |  \
 |   \____
 +----------------> Steps
```

---

## 15.2 Evaluation Loss

Evaluation/validation loss shows how the model performs on held-out data.

Typical pattern:

```text
Train loss -> decreases
Eval loss  -> decreases
```

Potential overfitting:

```text
Train loss -> keeps decreasing
Eval loss  -> eventually increases
```

---

## 15.3 Perplexity

For language models:

```text
Perplexity = exp(loss)
```

under the usual natural-log language-model loss convention.

Lower perplexity generally means the model assigns higher probability to the observed tokens.

Example:

```text
Step 1   -> PPL 40
Step 2   -> PPL 30
Step 3   -> PPL 22
Step 4   -> PPL 18
```

The decreasing trend indicates improvement on the evaluated data.

---

## 15.4 Learning Rate vs Steps

Important for fine-tuning.

Example:

```text
LR
 |
 |       /\
 |      /  \
 |_____/    \________
 +--------------------> Steps
```

Inspect this together with loss.

For example:

```text
LR increases
     |
     v
Loss suddenly explodes
```

This can be a clue that the learning rate or optimization configuration needs investigation.

---

## 15.5 Token-Level Accuracy

For next-token prediction:

```text
Actual:
I love machine learning

Prediction:
I love machine learning
```

You can evaluate whether the correct token appears as:

```text
Top-1
Top-5
Top-10
```

---

## 15.6 Top-K Accuracy

Suppose actual next word:

```text
learning
```

Model predictions:

```text
1. AI
2. deep
3. learning   <-- correct
4. model
5. neural
```

Then:

```text
Top-1 = failure
Top-5 = success
```

Top-K metrics can be particularly informative for next-token prediction because multiple plausible continuations may exist.

---

## 15.7 Prediction Probability Distribution

For a given input:

```text
learning -> 0.45
model    -> 0.20
AI       -> 0.15
deep     -> 0.10
data     -> 0.05
...
```

A bar chart of the top predictions can answer:

> Is the model confident or uncertain?

It also helps inspect whether the probability distribution is overly concentrated.

---

## 15.8 Calibration for Language Models

Calibration can also be studied for token probabilities.

A model that assigns:

```text
90% confidence
```

should ideally be correct at roughly that frequency for comparable predictions.

Calibration analysis is more advanced and should be designed carefully for the language-model setting.

---

## 15.9 Error Analysis by Token

You can analyze:

- Most frequently missed tokens.
- Rare tokens.
- Unknown/OOV tokens.
- Punctuation.
- Capitalization.
- Long-context cases.
- Short-context cases.
- Common words.
- Domain-specific terms.

Example:

```text
Error count

"the"       ██████████
"machine"   ███████
"learning"  █████
"database"  ███
```

This often reveals more actionable information than one aggregate accuracy number.

---

# 16. Next-Word Prediction: Practical Visualization Dashboard

For a next-word prediction RNN/LSTM/Transformer, a useful training dashboard could contain:

```text
                 MODEL TRAINING
                       |
       +---------------+---------------+
       |               |               |
       v               v               v
   Train Loss      Eval Loss      Learning Rate
       |               |               |
       +---------------+---------------+
                       |
                       v
                 Training Curves
                       |
       +---------------+---------------+
       |               |               |
       v               v               v
   Top-1 Accuracy   Top-5 Accuracy   Perplexity
       |               |               |
       +---------------+---------------+
                       |
                       v
                  Error Analysis
                       |
       +---------------+---------------+
       |               |               |
       v               v               v
  Wrong Tokens     Confidence       Examples
                       |
                       v
              Hyperparameter Tuning
                       |
       +---------------+---------------+
       |               |               |
       v               v               v
 Learning Rate    Batch Size     Sequence Length
```

For sequence-length experiments:

```text
SEQUENCE_LENGTH = 14
SEQUENCE_LENGTH = 30
SEQUENCE_LENGTH = 60
```

Compare:

```text
Sequence Length -> Validation Loss
Sequence Length -> Top-1 Accuracy
Sequence Length -> Top-5 Accuracy
Sequence Length -> Perplexity
Training time
Memory usage
```

This gives a more complete picture than choosing a sequence length based on one metric.

---

# 17. Production ML Monitoring

Training is not the end of ML.

Once deployed, monitor the model.

---

## 17.1 Data Drift

Suppose training data contains:

```text
Age: mostly 20-60
```

Production data later becomes:

```text
Age: mostly 30-90
```

The input distribution has changed.

Conceptually:

```text
Training:
      /\
     /  \

Production:
          /\
         /  \
```

This may indicate data drift.

---

## 17.2 Prediction Drift

Monitor model outputs.

Example:

```text
January:
Positive predictions = 20%

June:
Positive predictions = 48%
```

Possible causes include:

- Data distribution changes.
- User behavior changes.
- Pipeline changes.
- Model changes.
- Real-world changes.

A drift signal does not automatically identify the cause.

---

## 17.3 Performance Over Time

For models with available ground truth, monitor metrics over time:

```text
Accuracy
 |
95|------
93|     ----
91|          ---
89|             ---
 +--------------------> Time
```

A decline can trigger investigation.

---

# 18. How to Read Any ML Graph

When looking at any ML graph, ask these questions.

## Question 1: What is the X-axis?

Examples:

```text
Epoch
Step
Time
Learning Rate
Feature
Threshold
Cluster Count
```

---

## Question 2: What is the Y-axis?

Examples:

```text
Loss
Accuracy
Precision
Recall
Error
Probability
Frequency
```

---

## Question 3: Is higher better or lower better?

Common examples:

| Metric | Usually better |
|---|---|
| Accuracy | Higher |
| Precision | Higher |
| Recall | Higher |
| F1 | Higher |
| AUC | Higher |
| R² | Higher |
| Loss | Lower |
| MAE | Lower |
| MSE | Lower |
| RMSE | Lower |
| Perplexity | Lower |

"Usually" matters: the metric must be interpreted in the context of the task and objective.

---

## Question 4: Training or validation/test?

This is critical.

Training performance alone cannot establish generalization.

---

## Question 5: Is there a gap?

Example:

```text
Training accuracy = 99%
Validation accuracy = 78%
```

A large gap can indicate overfitting.

---

## Question 6: Is there a trend?

Ask:

```text
Increasing?
Decreasing?
Stable?
Oscillating?
Sudden jump?
Sudden collapse?
```

---

## Question 7: Are there outliers?

Outliers can indicate:

- Bad data.
- Rare examples.
- Measurement errors.
- Model failures.
- Distribution changes.

---

## Question 8: Is there a systematic pattern?

Random scatter can be desirable in some diagnostics, such as residual plots.

A systematic pattern may indicate that the model is missing structure.

---

# 19. Diagnostic Combinations

Do not think about graphs only individually.

Use combinations.

---

## 19.1 Neural Network Training

Start with:

```text
Train Loss
+
Validation Loss
```

Then:

```text
Train Accuracy
+
Validation Accuracy
```

Then inspect:

```text
Learning Rate
+
Gradient Norm
```

when diagnosing optimization issues.

---

## 19.2 Classification

Use:

```text
Confusion Matrix
+
Precision
+
Recall
+
F1
+
ROC-AUC
+
Precision-Recall Curve
```

Add calibration when probabilities matter.

---

## 19.3 Regression

Use:

```text
Actual vs Predicted
+
Residual Plot
+
MAE
+
RMSE
+
R²
+
Error Distribution
```

---

## 19.4 Hyperparameter Tuning

Use:

```text
Hyperparameter
       |
       v
Validation Score
       +
Heatmap
       +
Cross-Validation
       +
Optimization History
```

---

## 19.5 Explainability

Use:

```text
Feature Importance
+
Permutation Importance
+
SHAP
```

---

## 19.6 LLM / NLP Fine-Tuning

Use:

```text
Training Loss
+
Evaluation Loss
+
Learning Rate
+
Perplexity
+
Top-K Accuracy
+
Qualitative Error Analysis
```

---

# 20. Recommended Learning Roadmap

If you are learning Python, NumPy, Pandas, Matplotlib and then moving toward ML/deep learning, use this progression.

## Level 1 — Visualization Fundamentals

Master:

```text
1. Line Chart
2. Bar Chart
3. Scatter Plot
4. Histogram
5. Box Plot
6. KDE
7. Correlation Heatmap
```

Learn:

- X-axis.
- Y-axis.
- Scale.
- Distribution.
- Trend.
- Relationship.
- Outliers.

---

## Level 2 — Model Training

Master:

```text
8. Training Loss
9. Validation Loss
10. Training Accuracy
11. Validation Accuracy
12. Learning Curves
```

Learn to identify:

```text
Underfitting
Good Fit
Overfitting
```

---

## Level 3 — Classification

Master:

```text
13. Confusion Matrix
14. Accuracy
15. Precision
16. Recall
17. F1
18. ROC
19. AUC
20. Precision-Recall Curve
21. Calibration
```

---

## Level 4 — Regression

Master:

```text
22. Actual vs Predicted
23. Residual Plot
24. Error Distribution
25. Q-Q Plot
26. MAE
27. MSE
28. RMSE
29. R²
```

---

## Level 5 — Cross-Validation & Hyperparameter Tuning

Master:

```text
30. Fold Scores
31. Parameter vs Validation Score
32. Hyperparameter Heatmap
33. Optimization History
34. CV Score Distribution
```

Then learn:

```text
Grid Search
Random Search
Bayesian Optimization
Optuna
```

---

## Level 6 — Explainability

Master:

```text
35. Feature Importance
36. Permutation Importance
37. SHAP Summary
38. SHAP Dependence
39. SHAP Waterfall
```

---

## Level 7 — Unsupervised Learning

Master:

```text
40. Elbow Plot
41. Silhouette Plot
42. PCA Explained Variance
43. PCA Scatter
44. t-SNE
45. UMAP
```

---

## Level 8 — Deep Learning

Master:

```text
46. Train Loss
47. Validation Loss
48. Accuracy Curves
49. Learning Rate
50. Gradient Norm
51. Weight Distributions
52. Activation Distributions
53. Embedding Visualization
```

---

## Level 9 — NLP / LLM

Master:

```text
54. Training Loss vs Steps
55. Evaluation Loss
56. Perplexity
57. Learning Rate Schedule
58. Token Accuracy
59. Top-K Accuracy
60. Probability Distribution
61. Calibration
62. Embedding Visualization
63. Error Analysis
```

---

## Level 10 — Production ML

Learn:

```text
64. Data Drift
65. Prediction Drift
66. Model Performance Over Time
67. Feature Distribution Monitoring
68. Latency Monitoring
69. Error Rate Monitoring
70. Data Quality Monitoring
```

---

# 21. Common Mistakes

## Mistake 1: Looking only at accuracy

Accuracy can hide:

- Class imbalance.
- False positives.
- False negatives.
- Poor minority-class performance.

---

## Mistake 2: Looking only at training performance

A model can memorize training data.

Always consider validation/test performance.

---

## Mistake 3: Choosing the final epoch automatically

The final epoch is not necessarily the best checkpoint.

Validation performance can peak earlier.

---

## Mistake 4: Assuming lower training loss always means a better model

Lower training loss can coexist with worse generalization.

---

## Mistake 5: Assuming correlation means causation

A correlation heatmap identifies associations, not causal relationships.

---

## Mistake 6: Treating feature importance as causality

A feature can be important to a predictive model without causing the target.

---

## Mistake 7: Using only one metric

Use metrics appropriate to:

- Business objective.
- Error costs.
- Class balance.
- Model type.
- Deployment requirements.

---

## Mistake 8: Ignoring the data split

A beautiful training curve is not useful if your validation/test setup contains leakage.

---

## Mistake 9: Ignoring scale and axes

Always check:

- Axis range.
- Units.
- Log vs linear scale.
- Whether the plot is truncated.

---

## Mistake 10: Treating t-SNE/UMAP visualization as proof

Embedding visualization is exploratory.

It does not automatically prove that clusters are objectively meaningful.

---

# 22. Final Cheat Sheet

## Data

```text
Histogram
    -> Distribution

KDE
    -> Smooth distribution

Box Plot
    -> Quartiles + outliers

Violin
    -> Distribution + density

Scatter
    -> Relationship

Heatmap
    -> Correlation / matrix values
```

## Training

```text
Train Loss
    -> Is optimization improving?

Validation Loss
    -> Is generalization improving?

Train vs Validation Loss
    -> Underfitting / overfitting diagnosis

Train Accuracy
    -> Training classification performance

Validation Accuracy
    -> Held-out classification performance

Learning Rate
    -> Optimization schedule
```

## Classification

```text
Confusion Matrix
    -> Types of errors

Accuracy
    -> Overall correctness

Precision
    -> Reliability of positive predictions

Recall
    -> Ability to find actual positives

F1
    -> Precision/recall balance

ROC
    -> Threshold-based discrimination

AUC
    -> Overall ranking/separation measure

PR Curve
    -> Precision/recall trade-off

Calibration
    -> Reliability of predicted probabilities
```

## Regression

```text
Actual vs Predicted
    -> Prediction quality

Residual Plot
    -> Systematic errors

Error Distribution
    -> Error shape/outliers

Q-Q Plot
    -> Distributional assumption check

MAE
    -> Average absolute error

MSE
    -> Squared error

RMSE
    -> Error in target units

R²
    -> Relative explained-variance-style measure
```

## Tuning

```text
Parameter vs Score
    -> Sensitivity to a hyperparameter

Heatmap
    -> Interaction between two hyperparameters

Optimization History
    -> Progress across trials

CV Plot
    -> Stability across folds
```

## Explainability

```text
Feature Importance
    -> Model-level feature contribution measure

Permutation Importance
    -> Performance dependence on feature

SHAP
    -> Local/global prediction explanations
```

## Clustering

```text
Elbow
    -> Diminishing returns as K increases

Silhouette
    -> Cluster separation/cohesion
```

## Dimensionality Reduction

```text
PCA Explained Variance
    -> Variance retained

PCA Scatter
    -> Low-dimensional visualization

t-SNE
    -> Local embedding visualization

UMAP
    -> Embedding visualization / dimensionality reduction
```

## Deep Learning

```text
Loss
    -> Learning

Validation Loss
    -> Generalization

Learning Rate
    -> Optimization behavior

Gradient Norm
    -> Gradient stability

Weight Distribution
    -> Parameter behavior

Activation Distribution
    -> Internal network behavior
```

## NLP / LLM

```text
Training Loss
    -> Language-model training progress

Eval Loss
    -> Generalization

Perplexity
    -> Language-model uncertainty/performance measure

Top-K Accuracy
    -> Whether correct token appears in top K

Probability Distribution
    -> Model confidence/uncertainty

Error Analysis
    -> What kinds of tokens/examples fail
```

## Production

```text
Data Drift
    -> Input distribution changes

Prediction Drift
    -> Output distribution changes

Performance Over Time
    -> Model quality degradation

Latency
    -> Serving performance

Error Rate
    -> Operational/model failures
```

---

# The Core Mental Model

When working with an ML model, think:

```text
                  DATA
                    |
                    v
              DISTRIBUTION
                    |
                    v
                TRAINING
                    |
          +---------+---------+
          |                   |
          v                   v
       TRAIN                VALIDATION
       LOSS                   LOSS
          |                   |
          +---------+---------+
                    |
                    v
               DIAGNOSIS
                    |
          +---------+---------+
          |         |         |
          v         v         v
      UNDERFIT   GOOD FIT  OVERFIT
                    |
                    v
             HYPERPARAMETER
                 TUNING
                    |
                    v
              FINAL MODEL
                    |
                    v
                TESTING
                    |
                    v
               DEPLOYMENT
                    |
                    v
               MONITORING
                    |
          +---------+---------+
          |         |         |
          v         v         v
        DRIFT    ERRORS   PERFORMANCE
```

The most important principle is:

> **Do not ask only "Is my model accurate?" Ask "What does the evidence tell me about how the model is learning, generalizing, failing, and behaving on real data?"**

That mindset is the foundation of practical ML model development.
