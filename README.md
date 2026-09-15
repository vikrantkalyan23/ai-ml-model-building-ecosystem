# Major ML/AI model-building ecosystem

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
