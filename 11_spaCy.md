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

---

# 17. Rule-based Matching

spaCy is not only statistical models. It also supports rule-based matching.

Use the `Matcher` when you know exact token patterns.

```python
import spacy
from spacy.matcher import Matcher

nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)

pattern = [{"LOWER": "machine"}, {"LOWER": "learning"}]
matcher.add("ML_TERM", [pattern])

doc = nlp("I am learning machine learning.")
matches = matcher(doc)

for match_id, start, end in matches:
    print(doc[start:end].text)
```

Rule-based matching is useful for:

- Product names
- Codes
- Domain-specific phrases
- Simple extraction rules

---

# 18. EntityRuler

`EntityRuler` lets you add custom named entities.

```python
import spacy

nlp = spacy.load("en_core_web_sm")
ruler = nlp.add_pipe("entity_ruler", before="ner")

patterns = [
    {"label": "SKILL", "pattern": "machine learning"},
    {"label": "SKILL", "pattern": "deep learning"},
]

ruler.add_patterns(patterns)

doc = nlp("She knows machine learning.")

for ent in doc.ents:
    print(ent.text, ent.label_)
```

This is useful when pretrained NER does not know your domain terms.

---

# 19. Dependency Parsing

Dependency parsing shows grammatical relationships.

```python
doc = nlp("The student reads a book.")

for token in doc:
    print(token.text, token.dep_, token.head.text)
```

Example meaning:

```text
student -> subject of reads
book    -> object of reads
```

Use cases:

- Information extraction
- Relation extraction
- Grammar analysis
- Search/query understanding

---

# 20. Custom Pipeline Components

You can add your own processing step.

```python
from spacy.language import Language

@Language.component("length_component")
def length_component(doc):
    doc.user_data["length"] = len(doc)
    return doc

nlp.add_pipe("length_component", last=True)

doc = nlp("This is a sentence.")
print(doc.user_data["length"])
```

Pipeline idea:

```text
Tokenizer -> tagger -> parser -> NER -> your custom component
```

---

# 21. Training Custom NER

If built-in models are not enough, train a custom NER model.

Data format idea:

```text
Text:
    "OpenAI is in San Francisco"

Entity annotations:
    OpenAI -> ORG
    San Francisco -> GPE
```

Training requires:

```text
Many labeled examples
Consistent annotation rules
Validation data
Careful evaluation
```

---

# 22. Common Mistakes

| Mistake | Problem | Fix |
|---|---|---|
| Expecting LLM behavior | spaCy is not a chatbot | Use Transformers/LLMs |
| Using wrong model size | Poor accuracy or slow speed | Choose sm/md/lg/trf appropriately |
| No domain examples | Bad custom model | Label domain data |
| Too many rules after NER | Conflicting entities | Order pipeline carefully |
| Ignoring tokenization | Bad spans | Check token boundaries |

---

# 23. Model Size Choices

```text
sm  -> small, fast, lower accuracy
md  -> medium, includes word vectors in many pipelines
lg  -> larger, better vectors
trf -> transformer-based, slower but stronger
```

Choose based on:

```text
speed requirement
accuracy requirement
available memory
language/domain
```
