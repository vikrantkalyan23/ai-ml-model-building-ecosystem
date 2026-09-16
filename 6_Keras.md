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

---

# 17. Sequential vs Functional API

Keras has more than one way to create models.

## Sequential API

Use it when the model is a simple stack.

```text
Input -> Layer -> Layer -> Output
```

```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(1)
])
```

## Functional API

Use it when the model has branches, shared layers, or multiple inputs/outputs.

```text
Input A ----\
             -> Combined layers -> Output
Input B ----/
```

```python
inputs = tf.keras.Input(shape=(10,))
x = tf.keras.layers.Dense(64, activation="relu")(inputs)
outputs = tf.keras.layers.Dense(1)(x)

model = tf.keras.Model(inputs, outputs)
```

---

# 18. Choosing Output Layers

The final layer depends on the task.

| Task | Final layer | Meaning |
|---|---|---|
| Regression | `Dense(1)` | Predict one number |
| Binary classification | `Dense(1, activation="sigmoid")` | Probability of class 1 |
| Multiclass classification | `Dense(num_classes, activation="softmax")` | Probability for each class |

Example:

```python
# Regression
tf.keras.layers.Dense(1)

# Binary classification
tf.keras.layers.Dense(1, activation="sigmoid")

# Multiclass classification
tf.keras.layers.Dense(10, activation="softmax")
```

---

# 19. Optimizers

Optimizers update model weights.

| Optimizer | Simple meaning |
|---|---|
| **SGD** | Basic gradient descent |
| **Adam** | Popular adaptive optimizer |
| **RMSprop** | Useful for some sequence tasks |
| **Adagrad** | Adapts learning rate per parameter |

Most beginner projects start with Adam.

```python
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="mse"
)
```

---

# 20. Callbacks

Callbacks run during training.

Common callbacks:

| Callback | Use |
|---|---|
| **EarlyStopping** | Stop when validation stops improving |
| **ModelCheckpoint** | Save best model |
| **ReduceLROnPlateau** | Lower learning rate when stuck |
| **TensorBoard** | Visualize training |

Example:

```python
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint("best.keras", save_best_only=True),
]

model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=100,
    callbacks=callbacks
)
```

---

# 21. Overfitting and Underfitting

```text
Underfitting:
    training score bad
    validation score bad

Overfitting:
    training score good
    validation score bad

Good fit:
    training score good
    validation score good
```

Fixes:

| Problem | Possible fix |
|---|---|
| Underfitting | Bigger model, train longer, better features |
| Overfitting | Dropout, regularization, early stopping, more data |

---

# 22. Image Data Augmentation

Data augmentation creates varied image examples.

```python
augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])
```

Why it helps:

```text
Original image
      |
      v
Flipped / rotated / zoomed images
      |
      v
Model learns more robust patterns
```

---

# 23. Common Mistakes

| Mistake | Problem | Fix |
|---|---|---|
| Wrong loss function | Bad learning objective | Match loss to task |
| Too many epochs | Overfitting | Use EarlyStopping |
| No validation data | Cannot monitor generalization | Use validation split |
| Not scaling inputs | Slow/unstable training | Normalize data |
| Huge model for small data | Overfitting | Use smaller model |
