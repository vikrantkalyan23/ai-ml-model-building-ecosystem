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

---

# 16. Computation Graph Idea

TensorFlow is built around tensor operations.

In simple terms:

```text
Input tensors
      |
      v
Operations and layers
      |
      v
Output tensors
```

Older TensorFlow code focused heavily on static computation graphs. Modern TensorFlow uses eager execution by default, so operations behave more like normal Python.

```python
import tensorflow as tf

x = tf.constant([1, 2, 3])
y = x * 2

print(y)
```

But TensorFlow can still compile functions for speed.

```python
@tf.function
def double(x):
    return x * 2
```

---

# 17. What Happens During Training?

Training a neural network means updating weights to reduce loss.

```text
Input data
   |
Forward pass
   |
Prediction
   |
Compute loss
   |
Backpropagation
   |
Update weights
   |
Repeat
```

Important pieces:

| Piece | Meaning |
|---|---|
| **Weights** | Learnable numbers inside the model |
| **Loss** | How wrong the prediction is |
| **Optimizer** | Updates weights |
| **Gradient** | Direction for improving weights |
| **Epoch** | One full pass through training data |
| **Batch** | Small group of samples used per update |

---

# 18. Model Building APIs

TensorFlow/Keras gives three common ways to build models.

## Sequential API

Best for simple layer-by-layer models.

```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(1)
])
```

## Functional API

Best for models with multiple inputs, outputs, or branches.

```python
inputs = tf.keras.Input(shape=(10,))
x = tf.keras.layers.Dense(64, activation="relu")(inputs)
outputs = tf.keras.layers.Dense(1)(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)
```

## Subclassing

Best for advanced custom behavior.

```python
class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs):
        return self.dense(inputs)
```

---

# 19. Dataset Pipeline

For larger datasets, TensorFlow uses `tf.data`.

```python
dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
dataset = dataset.shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)

model.fit(dataset, epochs=10)
```

Pipeline idea:

```text
Read data
   |
Shuffle
   |
Batch
   |
Prefetch
   |
Train efficiently
```

---

# 20. Regularization Techniques

Neural networks can overfit.

Common solutions:

| Technique | Meaning |
|---|---|
| **Dropout** | Randomly disables neurons during training |
| **L2 regularization** | Penalizes large weights |
| **Early stopping** | Stops when validation score stops improving |
| **Data augmentation** | Creates varied training examples |
| **Batch normalization** | Stabilizes activations |

Example:

```python
callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=100,
    callbacks=[callback]
)
```

---

# 21. Saving and Loading Models

```python
model.save("my_model.keras")

loaded_model = tf.keras.models.load_model("my_model.keras")
predictions = loaded_model.predict(X_test)
```

Deployment options:

| Tool | Use |
|---|---|
| **TensorFlow Serving** | Server deployment |
| **TensorFlow Lite** | Mobile/edge devices |
| **TensorFlow.js** | Browser/JavaScript |

---

# 22. Common Mistakes

| Mistake | Problem | Fix |
|---|---|---|
| Too many epochs | Overfitting | Use validation and early stopping |
| Wrong loss function | Model learns wrong objective | Match loss to task |
| No scaling | Training becomes unstable | Normalize input data |
| Too large learning rate | Loss may explode | Lower learning rate |
| Too little data | Poor generalization | Use simpler model or augmentation |

---

# 23. Loss Function Selection

| Task | Output layer | Loss |
|---|---|---|
| Regression | Dense(1) | mse or mae |
| Binary classification | Dense(1, sigmoid) | binary_crossentropy |
| Multiclass integer labels | Dense(classes, softmax) | sparse_categorical_crossentropy |
| Multiclass one-hot labels | Dense(classes, softmax) | categorical_crossentropy |

This matching is very important.
