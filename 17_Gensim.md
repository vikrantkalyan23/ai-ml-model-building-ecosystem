# Gensim

## 1. What is Gensim?

**Gensim** is a Python library for topic modeling, document similarity, and word embeddings.

It is often used for unsupervised NLP on large text collections.

> **Simple definition:** Gensim helps find topics in documents and create vector representations of words or texts.

---

## 2. Basic Information

| Item | Details |
|---|---|
| **Name** | Gensim |
| **Type** | NLP and topic-modeling library |
| **Best for** | Topic modeling and embeddings |
| **Common models** | Word2Vec, Doc2Vec, LDA, LSI |
| **Main strength** | Efficient unsupervised text modeling |
| **Common tasks** | Topics, similarity, embeddings |

---

# 3. What Problem Does Gensim Solve?

Large text collections are hard to explore manually.

Gensim can help answer:

```text
What topics exist in these documents?
Which documents are similar?
What words have similar meanings?
How can text be represented as vectors?
```

Example:

```text
Many news articles
        |
        v
Gensim topic model
        |
        v
Politics, sports, finance, technology topics
```

---

# 4. Text Preprocessing Flow

Before using Gensim, text is usually cleaned.

```text
Raw documents
      |
      v
Lowercase
      |
      v
Tokenize
      |
      v
Remove stopwords/punctuation
      |
      v
Train topic or embedding model
```

---

# 5. Dictionary and Corpus

Gensim often uses:

```text
Dictionary -> maps words to IDs
Corpus     -> bag-of-words representation
```

Example:

```python
from gensim.corpora import Dictionary

texts = [
    ["machine", "learning", "model"],
    ["deep", "learning", "neural", "network"],
    ["topic", "modeling", "documents"],
]

dictionary = Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

print(corpus)
```

---

# 6. LDA Topic Modeling

LDA means Latent Dirichlet Allocation.

It finds hidden topics in documents.

```python
from gensim.models import LdaModel

lda = LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=2,
    random_state=42
)

topics = lda.print_topics()
for topic in topics:
    print(topic)
```

Simple idea:

```text
Documents -> LDA -> topics made of important words
```

---

# 7. Word2Vec

Word2Vec learns word embeddings.

Words with similar meanings get similar vectors.

```python
from gensim.models import Word2Vec

sentences = [
    ["king", "queen", "royal"],
    ["man", "woman", "person"],
    ["dog", "cat", "animal"],
]

model = Word2Vec(
    sentences=sentences,
    vector_size=50,
    window=3,
    min_count=1,
    workers=2
)

print(model.wv.most_similar("king"))
```

---

# 8. Common Gensim Models

| Model | Meaning |
|---|---|
| **Word2Vec** | Word embeddings |
| **Doc2Vec** | Document embeddings |
| **LDA** | Topic modeling |
| **LSI/LSA** | Latent semantic indexing |
| **FastText** | Word embeddings with subword information |

---

# 9. Gensim vs spaCy vs Transformers

| Feature | Gensim | spaCy | Transformers |
|---|---|---|---|
| Topic modeling | Excellent | Limited | Not main focus |
| Word embeddings | Good | Available | Contextual embeddings |
| NLP pipelines | Limited | Excellent | Good |
| Text generation | No | No | Yes |
| Large pretrained LLMs | No | No | Yes |

---

# 10. Advantages

- Excellent for topic modeling
- Efficient for large text collections
- Good Word2Vec/Doc2Vec support
- Useful for document similarity
- Works well for unsupervised NLP

---

# 11. Disadvantages

- Not an LLM library
- Not for modern text generation
- Requires preprocessing
- Topic quality can require tuning
- Less useful for supervised NLP than newer libraries

---

# 12. When to Use Gensim

Use Gensim when:

```text
You need topic modeling
You need Word2Vec or Doc2Vec
You need document similarity
You are exploring large text collections
```

Avoid it when:

```text
You need chatbots or LLMs
You need transformer fine-tuning
You need production NLP pipelines with NER/parsing
```

---

# 13. Quick Revision Table

| Topic | Meaning |
|---|---|
| **Gensim** | Topic modeling and embedding library |
| **Dictionary** | Maps tokens to IDs |
| **Corpus** | Numeric document representation |
| **LDA** | Topic modeling algorithm |
| **Word2Vec** | Word embedding model |
| **Doc2Vec** | Document embedding model |
| **Similarity** | Compare document/vector closeness |

---

# 14. Final Summary

```text
Gensim is useful for unsupervised NLP.

Use it for:
    - Topic modeling
    - Word embeddings
    - Document embeddings
    - Document similarity

Remember:
    Gensim explores text.
    Transformers generate and understand text with large pretrained models.
```
