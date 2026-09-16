# PyTorch

## 1. What is PyTorch?

**PyTorch** is an open-source deep-learning framework widely used for research, production, and modern AI development.

It is popular because it feels natural to write Python code and gives strong control over neural-network training.

> **Simple definition:** PyTorch helps you build and train neural networks with flexible Python code.

---

## 2. Basic Information

| Item | Details |
|---|---|
| **Name** | PyTorch |
| **Type** | Deep-learning framework |
| **Best for** | Research, deep learning, LLMs, computer vision |
| **Core data object** | Tensor |
| **GPU support** | Yes |
| **Main module** | torch |
| **Neural network module** | torch.nn |
| **Common tasks** | Vision, NLP, audio, recommendation, LLMs |

---

# 3. What Problem Does PyTorch Solve?

PyTorch helps you build models where you need flexibility.

Examples:

```text
Image -> CNN -> class
Text -> Transformer -> answer
Audio -> neural network -> command
User history -> recommender -> product
```

It is especially common in:

- Research experiments
- Custom training loops
- Computer vision
- NLP and LLMs
- Generative AI

---

# 4. What is a Tensor?

A tensor is a multi-dimensional array.

```python
import torch

x = torch.tensor([[1, 2], [3, 4]])
print(x)
```

Tensors can run on CPU or GPU.

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
x = x.to(device)
```

---

# 5. Neural Network Idea

```text
Input tensor
    |
    v
Layer 1
    |
    v
Activation
    |
    v
Layer 2
    |
    v
Prediction
```

PyTorch models are usually classes that inherit from `torch.nn.Module`.

---

# 6. Simple Model

```python
import torch
from torch import nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.layers(x)

model = SimpleModel()
```

---

# 7. Training Loop

PyTorch gives you direct control over training.

```python
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    predictions = model(X_train)
    loss = loss_fn(predictions, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(epoch, loss.item())
```

Meaning:

| Step | Meaning |
|---|---|
| **forward pass** | Make prediction |
| **loss** | Measure error |
| **zero_grad** | Clear old gradients |
| **backward** | Compute gradients |
| **step** | Update weights |

---

# 8. Common PyTorch Modules

| Module | Meaning |
|---|---|
| **torch** | Tensor operations |
| **torch.nn** | Neural-network layers |
| **torch.optim** | Optimizers |
| **torch.utils.data** | Dataset and DataLoader |
| **torchvision** | Computer vision tools |
| **torchaudio** | Audio tools |
| **torchtext** | Text utilities |

---

# 9. Common Layers

| Layer | Used for |
|---|---|
| **nn.Linear** | Fully connected networks |
| **nn.Conv2d** | Image models |
| **nn.MaxPool2d** | Image downsampling |
| **nn.Embedding** | Token embeddings |
| **nn.LSTM** | Sequence models |
| **nn.Transformer** | Transformer models |
| **nn.Dropout** | Regularization |

---

# 10. PyTorch vs TensorFlow

| Feature | PyTorch | TensorFlow |
|---|---|---|
| Style | Pythonic and flexible | Framework/ecosystem focused |
| Research | Very popular | Popular |
| Production | Strong | Strong |
| Beginner ease | Medium | Medium |
| Custom training loops | Very natural | Possible |
| LLM ecosystem | Very strong | Good |

---

# 11. Advantages

- Flexible and Python-friendly
- Excellent for research
- Strong GPU support
- Strong LLM and generative AI ecosystem
- Easy custom training loops
- Large community

---

# 12. Disadvantages

- More code than Keras for simple models
- Requires understanding training loops
- Not the simplest choice for beginner tabular ML
- Deployment requires extra planning

---

# 13. When to Use PyTorch

Use PyTorch when:

```text
You need deep learning
You need custom model logic
You are working with LLMs
You are doing research or experiments
You need GPU acceleration
```

Avoid it when:

```text
You only need basic tabular ML
You want the simplest possible beginner workflow
You can solve the task with Scikit-learn
```

---

# 14. Quick Revision Table

| Topic | Meaning |
|---|---|
| **PyTorch** | Deep-learning framework |
| **Tensor** | Main data object |
| **nn.Module** | Base class for models |
| **forward** | Defines prediction flow |
| **loss.backward** | Computes gradients |
| **optimizer.step** | Updates model weights |
| **DataLoader** | Loads data in batches |

---

# 15. Final Summary

```text
PyTorch is a flexible deep-learning framework.

Use it for:
    - Neural networks
    - Computer vision
    - NLP
    - LLMs
    - Research
    - Custom training loops

Core pattern:
    define model
    compute loss
    backward
    optimizer step
```

---

# 16. Autograd

Autograd is PyTorch's automatic differentiation system.

It records operations on tensors and calculates gradients.

```python
import torch

x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 3 * x

y.backward()

print(x.grad)
```

Flow:

```text
Tensor with requires_grad=True
        |
Operations
        |
Loss
        |
backward()
        |
Gradients
```

Gradients tell the optimizer how to update weights.

---

# 17. Dataset and DataLoader

For real projects, data is loaded in batches.

```python
from torch.utils.data import TensorDataset, DataLoader

dataset = TensorDataset(X_train, y_train)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for X_batch, y_batch in loader:
    predictions = model(X_batch)
```

Why batching matters:

```text
Full dataset may be too large
Batches make training memory-friendly
Shuffling improves learning
```

---

# 18. Training and Evaluation Mode

PyTorch models have modes.

```python
model.train()
```

Use during training.

```python
model.eval()
```

Use during evaluation.

Why it matters:

| Layer | Train mode | Eval mode |
|---|---|---|
| Dropout | Randomly drops units | Uses all units |
| BatchNorm | Uses batch statistics | Uses stored statistics |

Evaluation should usually use:

```python
model.eval()

with torch.no_grad():
    predictions = model(X_test)
```

---

# 19. Device Management

PyTorch does not automatically move everything to GPU.

```python
device = "cuda" if torch.cuda.is_available() else "cpu"

model = model.to(device)
X_batch = X_batch.to(device)
y_batch = y_batch.to(device)
```

Common error:

```text
Expected all tensors to be on the same device
```

Fix:

```text
Move model and data to the same device.
```

---

# 20. Complete Training Skeleton

```python
import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SimpleModel().to(device)

loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    model.train()

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        predictions = model(X_batch)
        loss = loss_fn(predictions, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        # validation code here
        pass
```

---

# 21. Saving and Loading

Recommended beginner style:

```python
torch.save(model.state_dict(), "model.pt")

model = SimpleModel()
model.load_state_dict(torch.load("model.pt"))
model.eval()
```

Meaning:

```text
state_dict = learned weights and buffers
```

---

# 22. Common Mistakes

| Mistake | Problem | Fix |
|---|---|---|
| Forgetting `zero_grad()` | Gradients accumulate | Call before backward |
| Forgetting `model.eval()` | Wrong validation behavior | Set eval mode |
| No `torch.no_grad()` | Wastes memory in evaluation | Use no_grad |
| Device mismatch | Runtime error | Move model/data to same device |
| Wrong tensor shape | Layer mismatch | Print shapes often |
