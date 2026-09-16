# Keras

## 1. What is Keras?

**Keras** is a high-level deep-learning API for building neural networks with simple, readable code.

Modern Keras is commonly used through TensorFlow as `tf.keras`.

> **Simple definition:** Keras is the beginner-friendly interface for creating neural networks.

---

## 2. Basic Information

| Item | Details |
|---|---|
| **Name** | Keras |
| **Type** | High-level neural-network API |
| **Common usage** | `tf.keras` |
| **Best for** | Fast deep-learning model building |
| **Main strength** | Simple model-building syntax |
| **Common tasks** | Classification, regression, vision, NLP, time series |

---

# 3. Why Keras Exists

Deep learning can require a lot of low-level code.

Keras simplifies the workflow:

```text
Define layers
     |
Compile model
     |
Train model
     |
Evaluate model
     |
Predict
```

Instead of managing every detail manually, you can focus on model structure.

---

# 4. Sequential Model

The simplest Keras model is `Sequential`.

Use it when layers flow one after another.

```text
Input -> Layer 1 -> Layer 2 -> Output
```

Example:

```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation="relu", input_shape=(10,)),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])
```

---

# 5. Compile, Fit, Evaluate, Predict

Keras models usually follow this pattern:

```python
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(X_train, y_train, epochs=10, batch_size=32)
model.evaluate(X_test, y_test)
predictions = model.predict(X_new)
```

Meaning:

| Step | Meaning |
|---|---|
| **compile** | Choose optimizer, loss, metrics |
| **fit** | Train the model |
| **evaluate** | Test model performance |
| **predict** | Use model on new data |

---

# 6. Common Layers

| Layer | Meaning | Used for |
|---|---|---|
| **Dense** | Fully connected layer | General neural networks |
| **Conv2D** | Detects image patterns | Computer vision |
| **MaxPooling2D** | Reduces image size | Computer vision |
| **Flatten** | Converts matrix to vector | Before Dense layers |
| **Embedding** | Converts words/tokens to vectors | NLP |
| **LSTM** | Learns sequence patterns | Text/time series |
| **Dropout** | Reduces overfitting | Regularization |

---

# 7. Activation Functions

Activation functions add nonlinearity.

| Activation | Simple meaning | Common use |
|---|---|---|
| **relu** | Keeps positive values | Hidden layers |
| **sigmoid** | Output between 0 and 1 | Binary classification |
| **softmax** | Probabilities across classes | Multiclass classification |
| **tanh** | Output between -1 and 1 | Sequence models |

---

# 8. Loss Functions

Loss tells the model how wrong it is.

| Loss | Used for |
|---|---|
| **mse** | Regression |
| **mae** | Regression |
| **binary_crossentropy** | Binary classification |
| **categorical_crossentropy** | Multiclass one-hot labels |
| **sparse_categorical_crossentropy** | Multiclass integer labels |

---

# 9. Regression Example

```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation="relu", input_shape=(8,)),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(1),
])

model.compile(optimizer="adam", loss="mse", metrics=["mae"])
model.fit(X_train, y_train, epochs=20, batch_size=32)

loss, mae = model.evaluate(X_test, y_test)
print(mae)
```

---

# 10. Classification Example

```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation="relu", input_shape=(10,)),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(X_train, y_train, epochs=10, batch_size=32)
```

---

# 11. Advantages

- Very beginner-friendly
- Clean and readable syntax
- Great for quick neural-network experiments
- Integrated with TensorFlow
- Supports many model types
- Good for education and production prototypes

---

# 12. Disadvantages

- Less low-level control than raw PyTorch or TensorFlow
- Not ideal for every research experiment
- Still needs deep-learning knowledge
- Not usually best for simple tabular ML

---

# 13. Keras vs TensorFlow

| Feature | Keras | TensorFlow |
|---|---|---|
| Level | High-level API | Full framework |
| Ease of use | Very easy | More flexible |
| Main role | Build models simply | Train/deploy ML systems |
| Best for | Beginners and fast prototypes | Production and custom workflows |

---

# 14. When to Use Keras

Use Keras when:

```text
You are learning deep learning
You want clean neural-network code
You need quick prototypes
You are building common model architectures
```

Avoid it when:

```text
You need very custom training logic
You only need classical ML
You are doing very low-level research experiments
```

---

# 15. Quick Revision Table

| Topic | Meaning |
|---|---|
| **Keras** | High-level neural-network API |
| **Sequential** | Simple stack of layers |
| **compile** | Set optimizer, loss, metrics |
| **fit** | Train model |
| **evaluate** | Test model |
| **predict** | Make predictions |
| **Dense** | Fully connected layer |

---

# 16. Final Summary

```text
Keras makes deep learning easier.

Use it for:
    - Simple neural networks
    - Image models
    - Text models
    - Regression/classification with neural networks

Most important pattern:
    model = Sequential([...])
    model.compile(...)
    model.fit(...)
```
