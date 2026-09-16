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

---

# 16. Functional Programming Style

JAX works best when functions are pure.

Pure function idea:

```text
Same input -> same output
No hidden changes
No unexpected side effects
```

Good JAX style:

```python
def add_one(x):
    return x + 1
```

Less ideal style:

```python
total = 0

def add_to_total(x):
    global total
    total += x
    return total
```

Why it matters:

```text
JAX transformations such as jit, grad, and vmap
work best with predictable pure functions.
```

---

# 17. Random Numbers in JAX

JAX handles randomness differently from NumPy.

You explicitly pass random keys.

```python
import jax

key = jax.random.PRNGKey(42)
values = jax.random.normal(key, shape=(3,))

print(values)
```

To generate more random values, split the key.

```python
key, subkey = jax.random.split(key)
values = jax.random.normal(subkey, shape=(3,))
```

Simple meaning:

```text
Randomness is explicit.
This makes experiments easier to reproduce.
```

---

# 18. JAX Arrays Are Immutable

JAX arrays should not be changed in place like normal Python lists.

Instead of:

```python
# Not JAX style
x[0] = 10
```

Use:

```python
x = x.at[0].set(10)
```

Why:

```text
Immutable-style updates help JAX compile and transform code safely.
```

---

# 19. Training Loop Idea in JAX

A JAX training step is usually written as a function.

```python
import jax
import jax.numpy as jnp

def loss_fn(params, X, y):
    predictions = X @ params
    return jnp.mean((predictions - y) ** 2)

@jax.jit
def train_step(params, X, y, learning_rate):
    grads = jax.grad(loss_fn)(params, X, y)
    params = params - learning_rate * grads
    return params
```

Flow:

```text
params + data
      |
loss_fn
      |
grad
      |
update params
```

---

# 20. Why JIT Can Be Surprising

`jax.jit` compiles a function.

The first call can be slower because compilation happens.

```text
First call:
    compile + run

Later calls:
    run compiled function
```

JIT works best when shapes are stable.

Common issue:

```text
Changing input shapes repeatedly can cause repeated compilation.
```

---

# 21. JAX Ecosystem

JAX itself is low-level. Many projects use extra libraries.

| Library | Purpose |
|---|---|
| **Flax** | Neural-network models |
| **Haiku** | Neural-network models |
| **Optax** | Optimizers and losses |
| **Orbax** | Checkpointing |
| **Equinox** | Neural networks with PyTree style |

Typical stack:

```text
JAX      -> arrays, grad, jit, vmap
Flax     -> model layers
Optax    -> optimizer
Orbax    -> checkpoints
```

---

# 22. Common Mistakes

| Mistake | Problem | Fix |
|---|---|---|
| Writing side-effect-heavy code | Hard for JAX transformations | Use pure functions |
| Mutating arrays directly | Not JAX style | Use `.at[].set()` |
| Forgetting random keys | Repeated or invalid randomness | Split keys |
| Changing shapes under jit | Recompilation | Keep shapes stable |
| Expecting PyTorch style | Different mental model | Think functions + transformations |
