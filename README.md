# Major ML/AI model-building ecosystem

Scikit-learn, TensorFlow, PyTorch, XGBoost, LightGBM, etc. are not all the same kind of library.
Some build neural networks, some build tree models, some specialize in LLMs, and some are mainly for specific types of data.


| Package / Framework           | Best for                            | Difficulty | Typical models                                                       | GPU       | Main strength            |
| ----------------------------- | ----------------------------------- | ---------: | -------------------------------------------------------------------- | --------- | ------------------------ |
| **Scikit-learn**              | Traditional ML                      |          ⭐ | Linear Regression, Logistic Regression, Random Forest, SVM, KNN, PCA | Usually ❌ | Easy ML                  |
| **XGBoost**                   | Tabular ML                          |         ⭐⭐ | Gradient Boosting, Regression, Classification                        | ✅         | Excellent accuracy       |
| **LightGBM**                  | Large tabular data                  |         ⭐⭐ | Gradient Boosting, Classification, Regression                        | ✅         | Very fast                |
| **CatBoost**                  | Tabular data + categorical features |         ⭐⭐ | Gradient Boosting                                                    | ✅         | Categorical data         |
| **TensorFlow**                | Deep Learning                       |         ⭐⭐ | CNN, RNN, LSTM, GRU, Transformer                                     | ✅         | Production ecosystem     |
| **Keras**                     | Easy Deep Learning                  |         ⭐⭐ | ANN, CNN, RNN, LSTM, GRU, Transformer                                | ✅         | Simpler neural networks  |
| **PyTorch**                   | Deep Learning                       |         ⭐⭐ | CNN, RNN, LSTM, Transformer                                          | ✅         | Flexibility/research     |
| **JAX**                       | High-performance ML/research        |        ⭐⭐⭐ | Neural Networks, scientific ML                                       | ✅         | Fast numerical computing |
| **Hugging Face Transformers** | LLM/NLP                             |        ⭐⭐⭐ | BERT, GPT-style, T5, Llama, etc.                                     | ✅         | Pretrained models        |
| **Transformers + PEFT**       | LLM fine-tuning                     |        ⭐⭐⭐ | LoRA/adapter fine-tuning                                             | ✅         | Efficient fine-tuning    |
| **spaCy**                     | NLP applications                    |         ⭐⭐ | NER, text classification, NLP pipelines                              | Sometimes | Production NLP           |
| **OpenCV**                    | Computer Vision                     |         ⭐⭐ | Image processing, detection pipelines                                | Optional  | Image/video processing   |
| **Ultralytics**               | Object Detection/Vision             |         ⭐⭐ | YOLO models                                                          | ✅         | Easy object detection    |
| **Stable-Baselines3**         | Reinforcement Learning              |        ⭐⭐⭐ | PPO, DQN, A2C, SAC                                                   | ✅         | RL experimentation       |
| **Statsmodels**               | Statistical modeling                |         ⭐⭐ | ARIMA, regression, time series                                       | ❌         | Statistics               |
| **Prophet**                   | Business forecasting                |          ⭐ | Time-series forecasting                                              | ❌         | Easy forecasting         |
| **Gensim**                    | NLP/embeddings                      |         ⭐⭐ | Word2Vec, Doc2Vec                                                    | Optional  | Text representations     |

## The most important packages

| Priority | Package                       | Why                    |
| -------: | ----------------------------- | ---------------------- |
|     🥇 1 | **Scikit-learn**              | ML fundamentals        |
|     🥈 2 | **Pandas**                    | Data processing        |
|     🥉 3 | **NumPy**                     | Numerical computing    |
|        4 | **XGBoost**                   | Powerful tabular ML    |
|        5 | **PyTorch**                   | Modern deep learning   |
|        6 | **Keras**                     | Easy deep learning     |
|        7 | **Hugging Face Transformers** | LLMs                   |
|        8 | **OpenCV**                    | Computer vision        |
|        9 | **LightGBM**                  | Large/tabular ML       |
|       10 | **CatBoost**                  | Categorical/tabular ML |

## Similar tools comparison

These tools are easier to understand when compared with other tools used for similar problems.

## XGBoost vs LightGBM vs CatBoost vs Statsmodels

These are often used with structured/tabular data, but they are not the same type of tool.

| Feature | XGBoost | LightGBM | CatBoost | Statsmodels |
| ------- | ------- | -------- | -------- | ----------- |
| Main purpose | High-accuracy tabular ML | Fast large-scale tabular ML | Tabular ML with categorical features | Statistical modeling and inference |
| Best for | Regression, classification, ranking | Large datasets, fast training | Datasets with many categorical columns | Regression analysis, p-values, confidence intervals |
| Core idea | Gradient-boosted decision trees | Fast gradient boosting with leaf-wise growth | Gradient boosting with strong categorical handling | Statistical models such as OLS, GLM, ARIMA |
| Prediction focus | Very strong | Very strong | Very strong | Medium |
| Explanation/statistics focus | Medium | Medium | Medium | Very strong |
| Categorical features | Usually needs encoding | Supports with setup | Excellent built-in handling | Usually needs encoding/model design |
| Speed | Fast | Very fast | Fast | Usually fast for statistical models |
| GPU support | Yes | Yes | Yes | Usually no |
| Use when | You want strong tabular accuracy | You need speed on large tabular data | Your data has many categories | You need statistical interpretation |
| Avoid when | You need p-values or causal explanation | Dataset is tiny and simple | No categorical advantage needed | You need modern ML algorithms |

Simple choice:

```text
Need best tabular ML accuracy?         XGBoost / LightGBM / CatBoost
Need very fast training?               LightGBM
Need categorical feature handling?     CatBoost
Need p-values and statistical summary? Statsmodels
```

## Scikit-learn vs Prophet vs Statsmodels

These can all appear in prediction workflows, but they solve different problems.

| Feature | Scikit-learn | Prophet | Statsmodels |
| ------- | ------------ | ------- | ----------- |
| Main purpose | Traditional machine learning | Business time-series forecasting | Statistical modeling |
| Best for | Regression, classification, clustering | Forecasting future values over time | Explaining relationships statistically |
| Data type | Tabular data | Date/time + value data | Tabular and time-series data |
| Time-series support | Limited/general ML style | Main focus | Strong statistical time-series models |
| Statistical summary | Limited | Limited | Excellent |
| Beginner-friendly | Very high | High | Medium |
| Common models | Linear models, trees, SVM, KNN, PCA | Trend + seasonality + holidays | OLS, GLM, ARIMA, SARIMAX |
| Use when | You need general ML | You need easy forecasting | You need statistical interpretation |

Simple choice:

```text
General ML problem?       Scikit-learn
Business forecast?        Prophet
Statistical explanation?  Statsmodels
```

## TensorFlow vs Keras vs PyTorch vs JAX

These are deep-learning and numerical-computing tools.

| Feature | TensorFlow | Keras | PyTorch | JAX |
| ------- | ---------- | ----- | ------- | --- |
| Main purpose | Deep learning framework | Easy neural-network API | Flexible deep learning | High-performance numerical ML |
| Best for | Production deep learning | Beginners and fast prototypes | Research, LLMs, custom training | Research, scientific ML, accelerators |
| Ease of use | Medium | High | Medium | Harder |
| Flexibility | High | Medium | Very high | Very high |
| GPU/TPU support | Strong | Through backend | Strong | Strong |
| Typical style | Framework ecosystem | Simple model building | Pythonic training loops | Functional transformations |
| Common use | Vision, NLP, deployment | Quick neural networks | Research, LLMs, vision | JIT, grad, vmap, TPU work |
| Use when | You need production ML deployment | You are learning deep learning | You need control and flexibility | You need fast compiled research code |

Simple choice:

```text
Easiest neural networks?       Keras
Production deep learning?      TensorFlow
Research/custom training?      PyTorch
High-performance ML research?  JAX
```

## Hugging Face Transformers vs PEFT vs spaCy vs Gensim

These are NLP-related tools, but each has a different role.

| Feature | Hugging Face Transformers | Transformers + PEFT | spaCy | Gensim |
| ------- | ------------------------- | ------------------- | ----- | ------ |
| Main purpose | Pretrained transformer models | Efficient LLM fine-tuning | Production NLP pipelines | Topic modeling and embeddings |
| Best for | LLMs, NLP, summarization, translation | LoRA/adapters for large models | NER, tokenization, parsing, text pipelines | LDA, Word2Vec, Doc2Vec |
| Text generation | Yes | Yes, through base model | No | No |
| Fine-tuning | Yes | Efficient fine-tuning | Possible for NLP tasks | Not for LLM fine-tuning |
| Speed | Depends on model size | Depends on base model | Very fast | Fast for topic modeling |
| Best data type | Text, images, audio depending on model | Text/LLM instruction data | Text documents | Large text collections |
| Use when | You need pretrained AI models | Full LLM fine-tuning is too costly | You need production NLP extraction | You need topics or classic embeddings |

Simple choice:

```text
Use pretrained LLM/NLP model?  Hugging Face Transformers
Fine-tune LLM cheaply?         PEFT
Fast NLP extraction pipeline?  spaCy
Topic modeling/Word2Vec?       Gensim
```

## Ultralytics vs OpenCV

Both are used in computer vision, but they work at different levels.

| Feature | Ultralytics | OpenCV |
| ------- | ----------- | ------ |
| Main purpose | YOLO deep-learning vision models | Image/video processing |
| Best for | Object detection, segmentation, pose | Reading, resizing, filtering, drawing, camera/video |
| Model training | Yes | Limited |
| Image preprocessing | Basic | Excellent |
| Object detection | Excellent with YOLO | Traditional methods or model integration |
| GPU use | Common for training/inference | Optional |
| Use when | You need to detect objects with YOLO | You need to process images/videos |
| Works together? | Yes | Yes |

Simple choice:

```text
Need object detection model?        Ultralytics
Need image/video processing tools?  OpenCV
Best practical combo?               OpenCV + Ultralytics
```

## Stable-Baselines3 vs Other ML Libraries

Stable-Baselines3 is different because it is for reinforcement learning.

| Feature | Stable-Baselines3 | Scikit-learn | PyTorch/TensorFlow |
| ------- | ----------------- | ------------ | ------------------ |
| Main purpose | Reinforcement learning | Traditional supervised/unsupervised ML | Deep learning |
| Learns from | Rewards from environment | Labeled/unlabeled data | Data + loss function |
| Best for | Agents, games, robotics simulations | Tabular ML | Neural networks |
| Common algorithms | PPO, DQN, A2C, SAC | Linear models, trees, SVM, KNN | CNN, RNN, Transformers |
| Needs environment | Yes | No | No, unless RL/custom setup |
| Use when | An agent must learn actions | You have normal ML data | You need neural networks |

Simple choice:

```text
Agent + action + reward + environment?  Stable-Baselines3
Rows/columns + labels?                  Scikit-learn
Neural network training?                PyTorch/TensorFlow
```

## Quick grouped recommendation

| Problem type | Best starting tools |
| ------------ | ------------------- |
| Traditional tabular ML | Scikit-learn |
| High-accuracy tabular ML | XGBoost, LightGBM, CatBoost |
| Statistical interpretation | Statsmodels |
| Business forecasting | Prophet |
| Deep learning beginner projects | Keras |
| Deep learning research/custom models | PyTorch |
| Production deep learning | TensorFlow |
| High-performance ML research | JAX |
| LLM/NLP pretrained models | Hugging Face Transformers |
| Efficient LLM fine-tuning | Transformers + PEFT |
| Production NLP pipelines | spaCy |
| Topic modeling and embeddings | Gensim |
| Image/video processing | OpenCV |
| Object detection | Ultralytics |
| Reinforcement learning | Stable-Baselines3 |
