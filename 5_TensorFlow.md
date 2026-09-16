# TensorFlow

## 1. What is TensorFlow?

**TensorFlow** is an open-source machine-learning and deep-learning framework.

It is used to build, train, evaluate, and deploy neural networks.

> **Simple definition:** TensorFlow helps you build deep-learning models such as image classifiers, text models, time-series models, and production ML systems.

---

## 2. Basic Information

| Item | Details |
|---|---|
| **Name** | TensorFlow |
| **Type** | Deep-learning framework |
| **Best for** | Neural networks and production ML |
| **Main API** | Keras inside TensorFlow |
| **GPU support** | Yes |
| **Common tasks** | Vision, NLP, forecasting, recommendation |
| **Deployment tools** | TensorFlow Serving, TensorFlow Lite, TensorFlow.js |

---

# 3. What Problem Does TensorFlow Solve?

Traditional ML libraries are excellent for tabular data, but deep learning needs flexible neural-network tools.

TensorFlow helps with:

```text
Images -> CNN -> class label
Text -> neural network -> sentiment
Audio -> neural network -> command
Time series -> neural network -> forecast
```

Example:

```text
Image pixels
     |
     v
TensorFlow model
     |
     v
Cat / Dog / Car
```

---

# 4. What is a Tensor?

A tensor is a multi-dimensional array.

```text
Scalar: 5
Vector: [1, 2, 3]
Matrix:
    [[1, 2],
     [3, 4]]
Image tensor:
    height x width x channels
```

TensorFlow works by passing tensors through operations and neural-network layers.

---

# 5. Neural Network Idea

```text
Input data
   |
   v
Layer 1
   |
   v
Layer 2
   |
   v
Output layer
   |
   v
Prediction
```

Each layer learns useful patterns.

For example, in image models:

```text
Early layers  -> edges
Middle layers -> shapes
Later layers  -> objects
```

---

# 6. Simple TensorFlow/Keras Model

```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation="relu", input_shape=(4,)),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(X_train, y_train, epochs=10, batch_size=32)
predictions = model.predict(X_test)
```

---

# 7. Common TensorFlow Layers

| Layer | Meaning | Used for |
|---|---|---|
| **Dense** | Fully connected layer | Tabular data |
| **Conv2D** | Convolution layer | Images |
| **MaxPooling2D** | Downsampling layer | Images |
| **Embedding** | Converts tokens to vectors | NLP |
| **LSTM** | Sequence layer | Text/time series |
| **Dropout** | Randomly disables neurons during training | Regularization |
| **BatchNormalization** | Normalizes layer activations | Stable training |

---

# 8. Training Workflow

```text
Prepare data
    |
Build model
    |
Compile model
    |
Train model
    |
Evaluate model
    |
Use for prediction
```

TensorFlow/Keras code follows this pattern:

```python
model.compile(optimizer="adam", loss="mse")
model.fit(X_train, y_train, epochs=10)
model.evaluate(X_test, y_test)
model.predict(X_new)
```

---

# 9. Image Classification Example

```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
```

---

# 10. Advantages

- Strong for deep learning
- Good production ecosystem
- GPU/TPU support
- Works with Keras
- Supports mobile and browser deployment
- Large community and many examples

---

# 11. Disadvantages

- More complex than Scikit-learn
- Not usually the first choice for simple tabular ML
- Deep learning needs more data and compute
- Debugging can be harder than traditional ML
- PyTorch is often preferred in research workflows

---

# 12. TensorFlow vs Scikit-learn

| Feature | TensorFlow | Scikit-learn |
|---|---|---|
| Main purpose | Deep learning | Traditional ML |
| Neural networks | Excellent | Limited |
| Tabular ML | Good | Excellent |
| Computer vision | Excellent | Limited |
| LLM work | Possible | No |
| Beginner ease | Medium | Very high |
| GPU support | Strong | Limited |

---

# 13. When to Use TensorFlow

Use TensorFlow when:

```text
You need neural networks
You work with images, text, audio, or sequences
You need production deployment
You need GPU/TPU acceleration
```

Avoid it when:

```text
You only need simple tabular ML
You want quick classical ML baselines
You have very small data
```

---

# 14. Quick Revision Table

| Topic | Meaning |
|---|---|
| **TensorFlow** | Deep-learning framework |
| **Tensor** | Multi-dimensional array |
| **Keras** | High-level TensorFlow API |
| **Dense** | Fully connected layer |
| **Conv2D** | Image-processing layer |
| **Epoch** | One full pass over training data |
| **Batch size** | Number of samples per training step |

---

# 15. Final Summary

```text
TensorFlow is used for deep learning.

Use it for:
    - Neural networks
    - Computer vision
    - NLP
    - Time series
    - Production ML deployment

Main pattern:
    build model
    compile model
    fit model
    evaluate model
    predict
```
