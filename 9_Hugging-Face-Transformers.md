# Hugging Face Transformers

## 1. What is Hugging Face Transformers?

**Hugging Face Transformers** is a Python library for using pretrained transformer models.

It is widely used for NLP, LLMs, computer vision, audio, and multimodal AI.

> **Simple definition:** Transformers lets you download and use powerful pretrained AI models with a few lines of Python.

---

## 2. Basic Information

| Item | Details |
|---|---|
| **Name** | Hugging Face Transformers |
| **Type** | Pretrained model library |
| **Best for** | NLP, LLMs, vision, audio, multimodal AI |
| **Core model type** | Transformer |
| **Backend support** | PyTorch, TensorFlow, JAX/Flax |
| **Common platform** | Hugging Face Hub |
| **Common tasks** | Text classification, generation, translation, summarization |

---

# 3. What Problem Does It Solve?

Training large AI models from scratch is expensive.

Transformers lets you use models that are already trained.

```text
Pretrained model
       |
       v
Use directly or fine-tune
       |
       v
Your task
```

Examples:

```text
Text -> sentiment
Text -> summary
Question + context -> answer
Prompt -> generated text
Image -> label
Audio -> transcription
```

---

# 4. What is a Transformer?

A transformer is a neural-network architecture based on attention.

Simple idea:

```text
Input tokens
    |
    v
Attention layers
    |
    v
Context-aware representations
    |
    v
Prediction or generation
```

Attention helps the model understand which parts of the input are important.

---

# 5. Pipeline API

The easiest way to use Transformers is `pipeline`.

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")

result = classifier("I love machine learning.")
print(result)
```

Other pipelines:

```python
from transformers import pipeline

summarizer = pipeline("summarization")
generator = pipeline("text-generation")
translator = pipeline("translation_en_to_fr")
qa = pipeline("question-answering")
```

---

# 6. Tokenizer and Model

Transformers usually use two parts:

```text
Tokenizer -> converts text to numbers
Model     -> makes prediction from numbers
```

Example:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

inputs = tokenizer("Hello world", return_tensors="pt")
outputs = model(**inputs)

print(outputs.logits)
```

---

# 7. Common Auto Classes

| Class | Meaning |
|---|---|
| **AutoTokenizer** | Loads correct tokenizer |
| **AutoModel** | Loads base model |
| **AutoModelForSequenceClassification** | Text classification model |
| **AutoModelForCausalLM** | Text generation model |
| **AutoModelForSeq2SeqLM** | Translation/summarization model |
| **AutoModelForQuestionAnswering** | Question answering model |

---

# 8. Common Tasks

| Task | Example |
|---|---|
| **Text classification** | Sentiment, spam detection |
| **Text generation** | Chat, autocomplete |
| **Summarization** | Long article to short summary |
| **Translation** | English to Hindi |
| **Question answering** | Answer from context |
| **Named entity recognition** | Find names, dates, places |
| **Embeddings** | Convert text to vectors |

---

# 9. Fine-tuning Idea

Fine-tuning means adapting a pretrained model to your own task.

```text
Pretrained model
       |
       v
Your labeled data
       |
       v
Fine-tuned model
       |
       v
Better task-specific predictions
```

Fine-tuning is useful when a general model is not enough.

---

# 10. Advantages

- Huge collection of pretrained models
- Very easy pipeline API
- Supports PyTorch, TensorFlow, and JAX/Flax
- Strong for NLP and LLM work
- Active ecosystem
- Integrates with datasets, tokenizers, and PEFT tools

---

# 11. Disadvantages

- Large models need strong hardware
- Fine-tuning can be expensive
- Model choice can be confusing
- Some models require careful licensing checks
- Deployment can be complex for large models

---

# 12. When to Use Transformers

Use it when:

```text
You need pretrained NLP or LLM models
You want text generation or summarization
You need embeddings
You want to fine-tune transformer models
```

Avoid it when:

```text
You only need simple tabular ML
You need small classical ML models
You cannot handle model size/compute
```

---

# 13. Quick Revision Table

| Topic | Meaning |
|---|---|
| **Transformers** | Library for pretrained transformer models |
| **Pipeline** | Easiest high-level API |
| **Tokenizer** | Converts text to model inputs |
| **AutoModel** | Loads correct model architecture |
| **Fine-tuning** | Train pretrained model on your data |
| **Hugging Face Hub** | Model and dataset repository |

---

# 14. Final Summary

```text
Hugging Face Transformers makes pretrained AI models easy to use.

Use it for:
    - NLP
    - LLMs
    - Text generation
    - Summarization
    - Translation
    - Fine-tuning

Most important idea:
    tokenizer + model = transformer workflow
```

---

# 15. Why Pretrained Models Matter

Large transformer models are trained on huge datasets.

This training teaches general language or vision patterns.

```text
Massive pretraining data
        |
        v
General pretrained model
        |
        v
Use directly or fine-tune
```

Instead of training from zero, you reuse learned knowledge.

Example:

```text
BERT already understands many language patterns.
You fine-tune it for sentiment, spam, or intent classification.
```

---

# 16. Tokenization in More Detail

Models do not read raw text directly.

They read token IDs.

```text
"I love ML"
      |
Tokenizer
      |
[101, 1045, 2293, 19875, 102]
```

Tokenizer output usually includes:

| Field | Meaning |
|---|---|
| **input_ids** | Numeric token IDs |
| **attention_mask** | Which tokens are real vs padding |
| **token_type_ids** | Segment IDs for some models |

Example:

```python
inputs = tokenizer(
    ["I love ML", "Transformers are useful"],
    padding=True,
    truncation=True,
    return_tensors="pt"
)
```

---

# 17. Model Families

| Model family | Main use |
|---|---|
| **BERT** | Understanding, classification, embeddings |
| **GPT-style** | Text generation |
| **T5** | Text-to-text tasks |
| **BART** | Summarization and generation |
| **DistilBERT** | Smaller/faster BERT-style model |
| **Vision Transformer** | Image tasks |
| **Whisper** | Speech recognition |

Simple choice:

```text
Need classification?   BERT-style
Need generation?       GPT/T5/BART-style
Need summarization?    T5/BART
Need speech-to-text?   Whisper
```

---

# 18. Trainer API

For fine-tuning, Hugging Face provides `Trainer`.

```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="results",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    eval_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

trainer.train()
```

Why it helps:

```text
Handles training loop
Handles evaluation
Handles saving checkpoints
Handles batching
```

---

# 19. Inference vs Fine-tuning

| Topic | Inference | Fine-tuning |
|---|---|---|
| Meaning | Use existing model | Train model on your data |
| Data needed | Optional prompt/input | Labeled or task data |
| Compute need | Lower | Higher |
| Code complexity | Lower | Higher |
| Output | General model behavior | Task-specific behavior |

Beginner advice:

```text
Try pipeline/inference first.
Fine-tune only when the pretrained model is not good enough.
```

---

# 20. Common Mistakes

| Mistake | Problem | Fix |
|---|---|---|
| Using wrong model head | Task mismatch | Use correct AutoModelFor... class |
| No truncation | Input too long errors | Set `truncation=True` |
| No padding for batches | Shape mismatch | Set `padding=True` |
| Huge model on small GPU | Out of memory | Use smaller model or quantization |
| Ignoring license | Legal/project risk | Check model license |

---

# 21. Practical Model Selection

```text
Small and fast:
    distilbert-base-uncased

Text classification:
    bert-base-uncased
    roberta-base

Text generation:
    GPT-style causal language model

Summarization:
    T5 or BART style model

Embeddings:
    sentence-transformers models
```

Always match:

```text
task + model type + hardware + license
```
