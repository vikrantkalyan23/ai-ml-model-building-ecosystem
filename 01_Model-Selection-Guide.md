# Model Selection Guide

## 1. What is Model Selection?

**Model selection** means choosing the right machine-learning model for your data type, problem type, data size, accuracy need, interpretability need, and compute budget.

> **Simple definition:** Model selection answers: "For this data and this problem, which model should I try first?"

---

## 2. Quick Decision Chart

```text
What type of data do you have?
|
|-- Tabular data
|   |
|   |-- Need simple baseline?          Scikit-learn
|   |-- Need best accuracy?            XGBoost / LightGBM / CatBoost
|   |-- Many categorical columns?      CatBoost
|   |-- Very large dataset?            LightGBM
|   |-- Need statistical explanation?  Statsmodels
|
|-- Time-series data
|   |
|   |-- Business forecasting?          Prophet
|   |-- Statistical forecasting?       Statsmodels
|   |-- ML-style forecasting?          Scikit-learn / XGBoost
|
|-- Text data
|   |
|   |-- NLP pipeline/NER/parsing?      spaCy
|   |-- Topic modeling?                Gensim
|   |-- Embeddings/classification?     Transformers
|   |-- Text generation/LLMs?          Hugging Face Transformers
|   |-- Efficient LLM fine-tuning?     PEFT
|
|-- Image/video data
|   |
|   |-- Basic image processing?        OpenCV
|   |-- Object detection?              Ultralytics
|   |-- Custom deep learning?          PyTorch / TensorFlow / Keras
|
|-- Agent/environment data
    |
    |-- Actions + rewards?             Stable-Baselines3
```

---

## 3. First Ask These Questions

Before choosing a model, ask:

| Question | Why it matters |
|---|---|
| What is the data type? | Tabular, text, image, time series, audio, environment |
| What is the target? | Number, category, group, future value, generated text, action |
| How much data do you have? | Small data and large data need different models |
| Do you need explanation? | Some models are easier to explain |
| Do you need best accuracy? | Boosting/deep learning may help |
| Do you have GPU? | Deep learning and LLMs often need GPU |
| Is prediction speed important? | Some models are too slow for real-time use |

---

# 4. Tabular Data

Tabular data means rows and columns.

Example:

```text
Age    Salary    City      Purchased
25     40000     Delhi     No
31     60000     Mumbai    Yes
45     90000     Pune      Yes
```

Tabular data is common in business, finance, healthcare, operations, and analytics.

## Best models for tabular data

| Problem | Best first model | Stronger models | Use when |
|---|---|---|---|
| Regression | LinearRegression, RandomForestRegressor | XGBoost, LightGBM, CatBoost | Predict a number |
| Classification | LogisticRegression, RandomForestClassifier | XGBoost, LightGBM, CatBoost | Predict a category |
| Clustering | KMeans | DBSCAN, AgglomerativeClustering | Find groups |
| Feature reduction | PCA | TruncatedSVD | Reduce columns |
| Statistical explanation | OLS in Statsmodels | GLM, Logit | Need p-values and coefficients |

Simple choice:

```text
Beginner baseline:
    Scikit-learn

Best tabular accuracy:
    XGBoost / LightGBM / CatBoost

Many categorical columns:
    CatBoost

Very large tabular dataset:
    LightGBM

Need p-values:
    Statsmodels
```

---

# 5. Tabular Regression

Regression predicts a number.

Examples:

```text
House price
Monthly sales
Insurance cost
Customer lifetime value
Delivery time
```

## Recommended models

| Situation | Best model |
|---|---|
| Simple relationship | LinearRegression |
| Linear model but overfitting | Ridge, Lasso, ElasticNet |
| Nonlinear relationship | DecisionTreeRegressor, RandomForestRegressor |
| Strong accuracy needed | XGBoost, LightGBM, CatBoost |
| Large dataset | LightGBM |
| Many categorical features | CatBoost |
| Need explanation/statistics | Statsmodels OLS |

## Practical order

```text
1. LinearRegression
2. RandomForestRegressor
3. XGBoost / LightGBM / CatBoost
4. Tune best model
5. Explain with feature importance or Statsmodels if needed
```

## Metrics

| Metric | Use when |
|---|---|
| MAE | You want easy average error |
| RMSE | You want to punish large errors |
| R2/R² | You want explained variance |
| MAPE | You want percentage error |

---

# 6. Tabular Classification

Classification predicts a category.

Examples:

```text
Spam or not spam
Fraud or not fraud
Customer churn or stay
Loan approved or rejected
Disease or no disease
```

## Recommended models

| Situation | Best model |
|---|---|
| Simple baseline | LogisticRegression |
| Easy explanation | DecisionTreeClassifier |
| Strong general baseline | RandomForestClassifier |
| Best tabular accuracy | XGBoost, LightGBM, CatBoost |
| Many categories | CatBoost |
| Large dataset | LightGBM |
| Text-like sparse features | LogisticRegression, LinearSVC, GaussianNB |

## Practical order

```text
1. LogisticRegression
2. RandomForestClassifier
3. XGBoost / LightGBM / CatBoost
4. Tune model
5. Check precision, recall, F1, ROC-AUC
```

## Metrics

| Metric | Use when |
|---|---|
| Accuracy | Classes are balanced |
| Precision | False positives are costly |
| Recall | False negatives are costly |
| F1-score | Need balance between precision and recall |
| ROC-AUC | Need ranking/separation quality |
| PR-AUC | Data is highly imbalanced |

---

# 7. Imbalanced Classification

Imbalanced data means one class is much more common.

Example:

```text
Fraud:     1%
Not fraud: 99%
```

Accuracy can be misleading.

```text
Model predicts "not fraud" always.
Accuracy = 99%
But fraud detection = useless
```

## Best model choices

| Situation | Recommended models |
|---|---|
| Simple baseline | LogisticRegression with class weights |
| Strong tabular model | XGBoost, LightGBM, CatBoost |
| Need interpretability | LogisticRegression, DecisionTree |
| Need high recall | Tune threshold, use class weights |

## Best metrics

```text
Precision
Recall
F1-score
PR-AUC
Confusion matrix
```

Avoid relying only on accuracy.

---

# 8. Time-Series Data

Time-series data has time order.

Example:

```text
Date        Sales
2024-01-01  100
2024-01-02  120
2024-01-03  115
```

## Recommended models

| Problem | Best model |
|---|---|
| Business forecasting | Prophet |
| Statistical time-series modeling | ARIMA, SARIMAX in Statsmodels |
| Forecasting with many features | XGBoost, LightGBM, CatBoost |
| Deep sequence modeling | LSTM/GRU/Transformer in TensorFlow or PyTorch |
| Quick baseline | Moving average, linear regression |

Simple choice:

```text
Easy business forecast:
    Prophet

Need statistical time-series model:
    Statsmodels

Need many external features:
    XGBoost / LightGBM / CatBoost

Need deep learning sequence model:
    PyTorch / TensorFlow / Keras
```

## Important warning

Do not randomly shuffle time-series data before splitting.

Correct split:

```text
Past data      -> train
Future period  -> test
```

---

# 9. Text Data

Text data includes reviews, emails, documents, support tickets, articles, and prompts.

## Recommended tools

| Problem | Best tool/model |
|---|---|
| Tokenization, NER, POS tagging | spaCy |
| Topic modeling | Gensim LDA |
| Word embeddings | Gensim Word2Vec, FastText |
| Document similarity | Gensim, sentence-transformers |
| Text classification baseline | Scikit-learn + TF-IDF + LogisticRegression |
| Strong text classification | Hugging Face Transformers |
| Summarization | Hugging Face Transformers |
| Translation | Hugging Face Transformers |
| Text generation | Hugging Face Transformers |
| Efficient LLM fine-tuning | PEFT/LoRA |

## Practical order

```text
Need fast NLP extraction?
    spaCy

Need topic discovery?
    Gensim

Need high accuracy text classification?
    Transformers

Need text generation or LLM behavior?
    Hugging Face Transformers

Need to adapt LLM cheaply?
    PEFT
```

---

# 10. Image Data

Image data includes photos, scans, screenshots, medical images, product images, and camera frames.

## Recommended tools

| Problem | Best tool/model |
|---|---|
| Resize/crop/filter image | OpenCV |
| Read webcam/video | OpenCV |
| Draw boxes/text | OpenCV |
| Edge/contour detection | OpenCV |
| Image classification | TensorFlow/Keras or PyTorch |
| Object detection | Ultralytics YOLO |
| Image segmentation | Ultralytics YOLO segmentation, PyTorch |
| Custom vision research | PyTorch |
| Production vision deployment | TensorFlow, ONNX, TensorRT |

Simple choice:

```text
Image processing:
    OpenCV

Object detection:
    Ultralytics

Custom neural-network vision model:
    PyTorch / TensorFlow / Keras
```

---

# 11. Video Data

Video is a sequence of images.

## Recommended tools

| Problem | Best tool |
|---|---|
| Read video frames | OpenCV |
| Save/write video | OpenCV |
| Object detection in video | Ultralytics |
| Object tracking | Ultralytics + OpenCV |
| Action recognition | PyTorch / TensorFlow |
| Real-time camera pipeline | OpenCV + optimized model |

Typical workflow:

```text
Video
  |
OpenCV reads frames
  |
Ultralytics detects objects
  |
OpenCV draws boxes
  |
Save/display result
```

---

# 12. Audio and Speech Data

Audio data includes speech, music, sound events, and acoustic signals.

For English-language training or language assessment, audio usually means **spoken English**.

Examples:

```text
Student speaking English
Pronunciation practice
IELTS/TOEFL-style speaking answer
Reading aloud
Conversation recording
Accent/pronunciation feedback
Fluency scoring
Grammar and vocabulary assessment
```

## Recommended tools

| Problem | Best model/tool |
|---|---|
| Speech-to-text transcription | Whisper / Hugging Face Transformers |
| Pronunciation assessment | Wav2Vec2 / HuBERT / Whisper features + custom scoring model |
| Fluency assessment | Audio features + transcript features + Scikit-learn/XGBoost |
| Grammar assessment | Transcript + Transformers / LLM |
| Vocabulary assessment | Transcript + Transformers / LLM |
| Speaking score prediction | Multimodal model using audio + transcript |
| Accent classification | Wav2Vec2 / audio classifier |
| Audio classification | PyTorch / TensorFlow |
| Signal feature extraction | Librosa |
| Simple audio ML | Scikit-learn with extracted features |

Simple choice:

```text
Need transcription:
    Whisper

Need pronunciation scoring:
    Wav2Vec2 / HuBERT / Whisper embeddings + custom model

Need grammar/vocabulary scoring:
    Transcript + Transformers / LLM

Need full English speaking assessment:
    Audio model + speech-to-text + text scoring model
```

## English Language Training Model

An English language training model helps learners improve speaking, pronunciation, grammar, vocabulary, and fluency.

This is usually not one model. It is a pipeline.

```text
Student audio
      |
      v
Speech-to-text model
      |
      v
Transcript
      |
      |------> Grammar/vocabulary feedback
      |
      |------> Pronunciation/fluency analysis
      |
      v
Personalized learning feedback
```

## Best model choices

| Training feature | Best model/tool | Why |
|---|---|---|
| Convert speech to text | Whisper | Strong general speech recognition |
| Check spoken grammar | Transformer/LLM on transcript | Understands language structure |
| Give writing/speaking feedback | LLM | Good at explanation and correction |
| Pronunciation practice | Wav2Vec2/HuBERT/audio embeddings | Uses actual sound, not only text |
| Fluency feedback | Audio features + ML model | Uses pauses, speed, hesitation |
| Vocabulary level | Transformer/LLM or rule features | Can judge word choice and complexity |
| Lesson recommendation | Scikit-learn/XGBoost/recommender | Maps learner weakness to next lesson |

## Example English training pipeline

```text
1. Record learner speech
2. Clean audio
3. Transcribe speech with Whisper
4. Compare transcript with target sentence if reading aloud
5. Extract audio features:
       pauses
       speaking rate
       pitch
       energy
       pronunciation embeddings
6. Analyze transcript:
       grammar
       vocabulary
       sentence complexity
7. Generate feedback:
       pronunciation issue
       grammar correction
       better vocabulary
       next exercise
```

## Model recommendation

| Data available | Best approach |
|---|---|
| Only audio | Whisper + audio feature model |
| Audio + expected sentence | Forced alignment / pronunciation model |
| Audio + human speaking scores | Train XGBoost/LightGBM/PyTorch scoring model |
| Transcript only | Transformers / LLM text assessment |
| Audio + transcript | Best option for full speaking assessment |

---

# 12.1 Language Assessment Model

A language assessment model gives a score for language ability.

Example scores:

```text
Pronunciation:  7/10
Fluency:        6/10
Grammar:        8/10
Vocabulary:     7/10
Overall:        7/10
```

## Assessment dimensions

| Dimension | Data needed | Best model type |
|---|---|---|
| Pronunciation | Audio | Wav2Vec2/HuBERT/audio embeddings |
| Fluency | Audio timing + transcript | Feature model + XGBoost/LightGBM |
| Grammar | Transcript | Transformer/LLM |
| Vocabulary | Transcript | Transformer/LLM + lexical features |
| Coherence | Transcript | Transformer/LLM |
| Overall score | Audio + transcript + human labels | Supervised scoring model |

## Best architecture

For serious English-speaking assessment, use a multi-part architecture.

```text
Audio input
   |
   |-- Speech-to-text model
   |       |
   |       v
   |   Transcript features
   |
   |-- Audio embedding model
   |       |
   |       v
   |   Pronunciation/fluency features
   |
   v
Scoring model
   |
   v
Pronunciation, fluency, grammar, vocabulary, overall score
```

## Recommended models by assessment type

| Assessment problem | Best fit |
|---|---|
| Read-aloud pronunciation scoring | Audio alignment model + Wav2Vec2/HuBERT |
| Free speaking assessment | Whisper + Transformers/LLM + scoring model |
| IELTS-style speaking score | Audio + transcript + supervised scoring model |
| Grammar-only assessment | Transcript + LLM/Transformer |
| Fluency-only assessment | Pause/speed features + XGBoost/LightGBM |
| Accent detection | Wav2Vec2 classifier |
| Beginner app prototype | Whisper + LLM feedback |
| Production scoring system | Human-labeled data + custom supervised model |

## Useful features for scoring

| Feature type | Examples |
|---|---|
| Audio timing | pause count, pause duration, words per minute |
| Voice features | pitch, energy, speech rate |
| Pronunciation features | phoneme similarity, word stress, mispronounced sounds |
| Transcript features | grammar errors, vocabulary level, sentence complexity |
| Semantic features | relevance, coherence, answer completeness |

## Which model is best?

```text
If you only need transcription:
    Whisper

If you need pronunciation scoring:
    Wav2Vec2 / HuBERT + pronunciation scoring head

If you need grammar and vocabulary feedback:
    Transcript + Transformer/LLM

If you need final speaking score:
    Combine audio features + transcript features
    Train XGBoost/LightGBM/CatBoost or a neural network

If you need production-level assessment:
    Collect human-scored learner audio
    Train a supervised scoring model
    Validate against expert scores
```

## Important warning

For language assessment, do not rely only on speech-to-text.

Speech-to-text can tell **what was said**, but not fully **how it was spoken**.

```text
Transcript can judge:
    grammar
    vocabulary
    coherence

Audio can judge:
    pronunciation
    fluency
    pauses
    rhythm
    accent patterns
```

Best result:

```text
Audio + transcript + human-labeled scores
```

---

# 13. Recommendation Data

Recommendation data usually contains users, items, and interactions.

Example:

```text
user_id   item_id   rating
1         10        5
1         12        4
2         10        3
```

## Recommended models

| Problem | Best model/tool |
|---|---|
| Simple recommendation | Popularity-based baseline |
| Similar users/items | KNN, cosine similarity |
| Tabular recommendation features | XGBoost, LightGBM, CatBoost |
| Embedding-based recommendation | PyTorch / TensorFlow |
| Text-based recommendation | Transformers embeddings |

Practical order:

```text
1. Popularity baseline
2. Similarity model
3. Matrix factorization or embeddings
4. Gradient boosting with user/item features
5. Deep recommender if data is large
```

---

# 14. Graph Data

Graph data contains nodes and edges.

Examples:

```text
Social networks
Fraud rings
Knowledge graphs
Molecular structures
Recommendation graphs
```

## Recommended models

| Problem | Best model/tool |
|---|---|
| Simple graph features | NetworkX + Scikit-learn |
| Node classification | Graph Neural Networks |
| Link prediction | Graph embeddings / GNNs |
| Knowledge graph embeddings | Specialized graph libraries |
| Fraud network analysis | Graph features + XGBoost/LightGBM |

Common deep-learning tools:

```text
PyTorch Geometric
DGL
PyTorch
```

---

# 15. Reinforcement Learning Problems

Use reinforcement learning when an agent learns actions from rewards.

Examples:

```text
Game playing
Robot control
Trading simulation
Route optimization
Resource allocation
```

## Recommended models

| Situation | Best algorithm/tool |
|---|---|
| Beginner RL | Stable-Baselines3 + PPO |
| Discrete actions | DQN, PPO |
| Continuous actions | PPO, SAC, TD3 |
| Custom deep RL research | PyTorch |
| Need reliable baselines | Stable-Baselines3 |

Simple choice:

```text
State + action + reward + environment:
    Stable-Baselines3

Normal labeled dataset:
    Not reinforcement learning
```

---

# 16. Small Data vs Large Data

## Small data

Good choices:

```text
LinearRegression
LogisticRegression
Ridge
Lasso
DecisionTree
RandomForest
Statsmodels
```

Avoid:

```text
Large deep neural networks
Huge LLM fine-tuning
Overly complex models
```

## Large data

Good choices:

```text
LightGBM
XGBoost
CatBoost
PyTorch
TensorFlow
JAX
Transformers
```

Use large models only when the data and compute justify them.

---

# 17. Need Explainability?

Some projects need clear explanations.

Examples:

```text
Credit approval
Healthcare
Finance
Policy decisions
Scientific research
```

## Best choices

| Need | Best models/tools |
|---|---|
| Coefficients and p-values | Statsmodels |
| Simple feature effects | LinearRegression, LogisticRegression |
| Easy rule explanation | DecisionTree |
| Feature importance | RandomForest, XGBoost, LightGBM, CatBoost |
| Model-agnostic explanation | SHAP, permutation importance |

Simple rule:

```text
Need statistical explanation:
    Statsmodels

Need predictive accuracy with explanation:
    XGBoost/LightGBM/CatBoost + SHAP
```

---

# 18. Need Production Deployment?

Model choice also depends on deployment.

| Deployment need | Good choices |
|---|---|
| Simple API for tabular model | Scikit-learn, XGBoost, LightGBM |
| Mobile model | TensorFlow Lite, ONNX |
| Browser model | TensorFlow.js, ONNX Runtime Web |
| Real-time object detection | Ultralytics export + TensorRT/ONNX |
| LLM serving | Hugging Face, vLLM, TGI |
| Edge image processing | OpenCV |

Questions to ask:

```text
How fast must prediction be?
How large can the model be?
CPU or GPU?
Batch prediction or real-time?
Cloud, mobile, browser, or edge?
```

---

# 19. Model Selection by Problem

| Problem | Best starting model/tool | Stronger option |
|---|---|---|
| House price prediction | LinearRegression, RandomForest | XGBoost/LightGBM/CatBoost |
| Customer churn | LogisticRegression, RandomForest | XGBoost/LightGBM/CatBoost |
| Fraud detection | LogisticRegression with class weights | XGBoost/LightGBM/CatBoost |
| Sales forecasting | Prophet | Statsmodels, LightGBM with features |
| Statistical regression report | Statsmodels OLS | GLM/SARIMAX |
| Customer segmentation | KMeans | DBSCAN, AgglomerativeClustering |
| Text classification | TF-IDF + LogisticRegression | Transformers |
| Topic modeling | Gensim LDA | BERTopic/Transformers embeddings |
| NER extraction | spaCy | Transformers |
| Text generation | Transformers | Fine-tuned LLM + PEFT |
| Image preprocessing | OpenCV | OpenCV + deep model |
| Object detection | Ultralytics YOLO | Custom PyTorch model |
| RL agent | Stable-Baselines3 PPO | Custom PyTorch RL |

---

# 20. Models to Try First

## For tabular regression

```text
1. LinearRegression
2. RandomForestRegressor
3. XGBoost / LightGBM / CatBoost
```

## For tabular classification

```text
1. LogisticRegression
2. RandomForestClassifier
3. XGBoost / LightGBM / CatBoost
```

## For forecasting

```text
1. Prophet
2. Statsmodels ARIMA/SARIMAX
3. LightGBM/XGBoost with time features
```

## For NLP

```text
1. spaCy or TF-IDF + Scikit-learn
2. Hugging Face Transformers
3. PEFT if LLM fine-tuning is needed
```

## For computer vision

```text
1. OpenCV for preprocessing
2. Ultralytics for object detection
3. PyTorch/TensorFlow for custom neural networks
```

---

# 21. What to Avoid

| Situation | Avoid |
|---|---|
| Small tabular dataset | Huge neural networks |
| Need p-values | XGBoost/LightGBM alone |
| Need object detection | Scikit-learn |
| Need LLM text generation | spaCy/Gensim alone |
| Need image filtering | Transformers |
| Need business forecasting quickly | Custom deep learning first |
| No environment/reward | Reinforcement learning |

---

# 22. Final Summary

```text
Choose model by data type first:
    tabular, text, image, time series, audio, graph, environment

Then choose by problem:
    regression, classification, clustering, forecasting,
    generation, detection, recommendation, reinforcement learning

Begin simple:
    baseline model first

Then improve:
    stronger model, tuning, feature engineering, better data

Best model is not always the most complex model.
Best model is the one that solves the problem reliably.
```
