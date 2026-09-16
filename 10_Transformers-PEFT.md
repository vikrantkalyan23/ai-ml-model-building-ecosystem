# Transformers + PEFT

## 1. What is PEFT?

**PEFT** stands for **Parameter-Efficient Fine-Tuning**.

PEFT is a set of techniques for fine-tuning large pretrained models by training only a small number of extra parameters.

> **Simple definition:** PEFT lets you adapt large models without updating all model weights.

---

## 2. Basic Information

| Item | Details |
|---|---|
| **Name** | PEFT |
| **Full form** | Parameter-Efficient Fine-Tuning |
| **Common library** | Hugging Face PEFT |
| **Best for** | Efficient LLM fine-tuning |
| **Popular method** | LoRA |
| **Works with** | Hugging Face Transformers |
| **Main benefit** | Lower memory and compute needs |

---

# 3. Why PEFT is Needed

Large language models have billions of parameters.

Full fine-tuning can require:

```text
Large GPU memory
Long training time
Large model checkpoints
High cost
```

PEFT reduces this burden.

```text
Frozen base model
       +
Small trainable adapter
       |
       v
Task-adapted model
```

---

# 4. Full Fine-tuning vs PEFT

| Feature | Full Fine-tuning | PEFT |
|---|---|---|
| Train all model weights | Yes | No |
| Memory usage | High | Lower |
| Checkpoint size | Large | Small |
| Training cost | Higher | Lower |
| Good for LLMs | Expensive | Common choice |

---

# 5. What is LoRA?

**LoRA** stands for **Low-Rank Adaptation**.

LoRA adds small trainable matrices to selected model layers.

```text
Original model weights: frozen
LoRA adapter weights: trainable
```

Simple flow:

```text
Pretrained LLM
     |
     v
Add LoRA adapters
     |
     v
Train only adapters
     |
     v
Task-specific model
```

---

# 6. Common PEFT Methods

| Method | Simple meaning |
|---|---|
| **LoRA** | Adds low-rank trainable adapters |
| **QLoRA** | LoRA with quantized base model |
| **Prefix tuning** | Learns special prefix vectors |
| **Prompt tuning** | Learns soft prompt embeddings |
| **Adapter tuning** | Adds small adapter modules |

---

# 7. Simple LoRA Example

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

model_name = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["c_attn"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)
model.print_trainable_parameters()
```

---

# 8. Important LoRA Hyperparameters

| Hyperparameter | Meaning |
|---|---|
| **r** | Rank of LoRA matrices |
| **lora_alpha** | Scaling factor |
| **target_modules** | Layers where LoRA is added |
| **lora_dropout** | Dropout for LoRA layers |
| **task_type** | Task such as causal LM or sequence classification |

Simple idea:

```text
Higher r = more trainable capacity, more memory
Lower r = smaller adapter, less capacity
```

---

# 9. Adapter Saving

PEFT usually saves only adapter weights.

```text
Base model: downloaded separately
Adapter: small trained file
```

This makes sharing and storage easier.

```python
model.save_pretrained("my-lora-adapter")
tokenizer.save_pretrained("my-lora-adapter")
```

---

# 10. Advantages

- Lower memory use than full fine-tuning
- Smaller checkpoints
- Faster experiments
- Useful for large language models
- Multiple adapters can share one base model
- Works well with Hugging Face Transformers

---

# 11. Disadvantages

- Not always as flexible as full fine-tuning
- Requires correct target modules
- Still needs GPU for larger models
- Adapter/base-model version matching matters
- Quantized training can be more complex

---

# 12. When to Use PEFT

Use PEFT when:

```text
You want to fine-tune an LLM
Full fine-tuning is too expensive
You need small adapter checkpoints
You want multiple task-specific adapters
```

Avoid it when:

```text
The model is small enough for full fine-tuning
You need to change every model parameter
You are not using pretrained transformer models
```

---

# 13. Quick Revision Table

| Topic | Meaning |
|---|---|
| **PEFT** | Parameter-efficient fine-tuning |
| **LoRA** | Low-rank adapter tuning |
| **QLoRA** | Quantized LoRA training |
| **Adapter** | Small trainable module |
| **Frozen model** | Base model weights are not updated |
| **target_modules** | Layers where adapters are attached |

---

# 14. Final Summary

```text
PEFT makes LLM fine-tuning cheaper.

Main idea:
    freeze base model
    train small adapters

Most common method:
    LoRA

Use PEFT when full fine-tuning is too costly.
```
