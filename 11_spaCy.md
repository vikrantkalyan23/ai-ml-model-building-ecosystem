# spaCy

## 1. What is spaCy?

**spaCy** is a Python library for practical Natural Language Processing (NLP).

It is designed for production NLP pipelines, not just research experiments.

> **Simple definition:** spaCy helps you process text, find linguistic features, extract entities, and build NLP applications.

---

## 2. Basic Information

| Item | Details |
|---|---|
| **Name** | spaCy |
| **Type** | NLP library |
| **Best for** | Production text processing |
| **Language** | Python |
| **Common tasks** | Tokenization, POS tagging, NER, parsing, text classification |
| **Model style** | Pretrained language pipelines |

---

# 3. What Problem Does spaCy Solve?

Raw text is hard for programs to understand.

spaCy converts text into structured information.

```text
Raw text
   |
   v
spaCy pipeline
   |
   v
Tokens, entities, sentences, parts of speech
```

Example:

```text
"Apple opened an office in Mumbai."

Entities:
Apple  -> ORG
Mumbai -> GPE
```

---

# 4. Basic Usage

```python
import spacy

nlp = spacy.load("en_core_web_sm")

doc = nlp("Apple opened an office in Mumbai.")

for token in doc:
    print(token.text, token.pos_, token.dep_)
```

---

# 5. spaCy Pipeline

A spaCy pipeline is a sequence of processing steps.

```text
Text
 |
 v
Tokenizer
 |
 v
Tagger
 |
 v
Parser
 |
 v
NER
 |
 v
Processed Doc
```

You can inspect pipeline components:

```python
print(nlp.pipe_names)
```

---

# 6. Important Objects

| Object | Meaning |
|---|---|
| **nlp** | Loaded language pipeline |
| **Doc** | Processed text document |
| **Token** | One token/word/punctuation |
| **Span** | Slice of a document |
| **Entity** | Named entity such as person or place |

---

# 7. Tokenization

Tokenization splits text into tokens.

```python
doc = nlp("I love machine learning.")

for token in doc:
    print(token.text)
```

Output idea:

```text
I
love
machine
learning
.
```

---

# 8. Named Entity Recognition

NER finds names, places, organizations, dates, money values, and more.

```python
doc = nlp("Google was founded in California.")

for ent in doc.ents:
    print(ent.text, ent.label_)
```

Example output:

```text
Google ORG
California GPE
```

---

# 9. Part-of-Speech Tagging

POS tagging identifies word roles.

```python
doc = nlp("The cat sleeps.")

for token in doc:
    print(token.text, token.pos_)
```

Example:

```text
The DET
cat NOUN
sleeps VERB
```

---

# 10. Text Classification

spaCy can also train text classifiers.

Example use cases:

```text
Review sentiment
Support ticket category
Spam detection
Intent detection
```

Flow:

```text
Text examples + labels
        |
        v
Train text classifier
        |
        v
Predict label for new text
```

---

# 11. spaCy vs Transformers

| Feature | spaCy | Transformers |
|---|---|---|
| Main use | Production NLP pipelines | Pretrained transformer models |
| Speed | Very fast | Depends on model size |
| Classic NLP tasks | Excellent | Good |
| LLM text generation | No | Yes |
| Beginner usage | Easy | Easy to medium |
| Custom pipelines | Excellent | Possible |

---

# 12. Advantages

- Fast text processing
- Production-friendly
- Good tokenization and NER
- Easy pipeline system
- Supports custom components
- Good for practical NLP applications

---

# 13. Disadvantages

- Not an LLM framework
- Not mainly for generative AI
- Accuracy depends on selected language model
- Training custom models needs labeled data

---

# 14. When to Use spaCy

Use spaCy when:

```text
You need fast NLP preprocessing
You need tokenization, NER, parsing, or POS tags
You are building production NLP pipelines
You need rule-based + statistical NLP
```

Avoid it when:

```text
You need chatbots based on LLMs
You need large text generation
You need transformer fine-tuning as the main task
```

---

# 15. Quick Revision Table

| Topic | Meaning |
|---|---|
| **spaCy** | Production NLP library |
| **nlp** | Language processing pipeline |
| **Doc** | Processed document |
| **Token** | Word/punctuation unit |
| **NER** | Named entity recognition |
| **POS** | Part-of-speech tagging |
| **Pipeline** | Ordered text-processing steps |

---

# 16. Final Summary

```text
spaCy is for practical NLP pipelines.

Use it for:
    - Tokenization
    - Named entity recognition
    - POS tagging
    - Dependency parsing
    - Text classification

Remember:
    spaCy is excellent for NLP processing, but not an LLM framework.
```
