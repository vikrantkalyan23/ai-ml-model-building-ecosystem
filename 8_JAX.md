# JAX

## 1. What is JAX?

**JAX** is a high-performance numerical computing library for Python.

It is popular in machine learning research because it combines NumPy-like code with automatic differentiation and accelerator support.

> **Simple definition:** JAX lets you write NumPy-style code that can run fast on CPU, GPU, or TPU and automatically compute gradients.

---

## 2. Basic Information

| Item | Details |
|---|---|
| **Name** | JAX |
| **Type** | Numerical computing and ML research library |
| **Best for** | High-performance research code |
| **Syntax style** | NumPy-like |
| **Automatic differentiation** | Yes |
| **JIT compilation** | Yes |
| **GPU/TPU support** | Yes |
| **Common ecosystem tools** | Flax, Haiku, Optax |

---

# 3. What Problem Does JAX Solve?

JAX helps when you need fast mathematical computation and automatic gradients.

Examples:

```text
Math function
    |
    v
Automatic gradient
    |
    v
Fast compiled execution
```

It is often used for:

- ML research
- Scientific computing
- Differentiable programming
- Custom neural networks
- Large-scale accelerator experiments

---

# 4. NumPy-like API

JAX code often looks like NumPy code.

```python
import jax.numpy as jnp

x = jnp.array([1.0, 2.0, 3.0])
y = x * 2

print(y)
```

The difference is that JAX can transform functions.

---

# 5. Automatic Differentiation

JAX can calculate gradients automatically.

```python
import jax
import jax.numpy as jnp

def f(x):
    return x ** 2 + 3 * x

grad_f = jax.grad(f)

print(grad_f(2.0))
```

Simple meaning:

```text
Function -> jax.grad -> derivative/gradient
```

---

# 6. JIT Compilation

JIT means Just-In-Time compilation.

It can make functions faster.

```python
import jax
import jax.numpy as jnp

@jax.jit
def multiply_add(x):
    return x * 2 + 1

print(multiply_add(jnp.array([1, 2, 3])))
```

Flow:

```text
Python function
      |
      v
jax.jit
      |
      v
Compiled fast function
```

---

# 7. Vectorization with vmap

`vmap` applies a function over batches automatically.

```python
import jax
import jax.numpy as jnp

def square(x):
    return x ** 2

batched_square = jax.vmap(square)

print(batched_square(jnp.array([1, 2, 3])))
```

---

# 8. Main JAX Transformations

| Transformation | Meaning |
|---|---|
| **grad** | Compute gradients |
| **jit** | Compile function for speed |
| **vmap** | Vectorize function over batches |
| **pmap** | Parallelize across devices |

---

# 9. JAX and Neural Networks

JAX is lower-level than Keras or PyTorch.

For neural networks, people often use libraries built on top of JAX:

| Library | Purpose |
|---|---|
| **Flax** | Neural-network library |
| **Haiku** | Neural-network library |
| **Optax** | Optimizers and losses |

Example ecosystem flow:

```text
JAX -> math, gradients, compilation
Flax/Haiku -> model layers
Optax -> optimization
```

---

# 10. JAX vs PyTorch

| Feature | JAX | PyTorch |
|---|---|---|
| Style | Functional transformations | Object-oriented modules |
| Research | Excellent | Excellent |
| Beginner ease | Harder | Easier |
| Neural network API | Usually via Flax/Haiku | Built in with torch.nn |
| Compilation | Central feature | Available but less central |
| Accelerator work | Excellent | Excellent |

---

# 11. Advantages

- Very fast for numerical computation
- Automatic differentiation
- JIT compilation
- Good GPU/TPU support
- Powerful for research
- NumPy-like syntax

---

# 12. Disadvantages

- Harder for beginners
- Smaller ecosystem than PyTorch for many practical tasks
- Neural networks need extra libraries
- Functional programming style takes time to learn
- Debugging compiled functions can be tricky

---

# 13. When to Use JAX

Use JAX when:

```text
You need high-performance numerical code
You need automatic differentiation
You are doing ML research
You need GPU/TPU acceleration
You are comfortable with functional programming
```

Avoid it when:

```text
You are a beginner learning basic ML
You need simple tabular models
You want the easiest deep-learning API
```

---

# 14. Quick Revision Table

| Topic | Meaning |
|---|---|
| **JAX** | Fast numerical computing library |
| **jax.numpy** | NumPy-like API |
| **grad** | Automatic differentiation |
| **jit** | Compile for speed |
| **vmap** | Batch/vectorize functions |
| **pmap** | Parallelize across devices |
| **Flax** | Neural-network library for JAX |
| **Optax** | Optimizers for JAX |

---

# 15. Final Summary

```text
JAX is for fast numerical computing and ML research.

Use it for:
    - Gradients
    - JIT compilation
    - GPU/TPU work
    - Scientific ML
    - Research experiments

Remember:
    JAX is powerful, but less beginner-friendly than Keras or Scikit-learn.
```
