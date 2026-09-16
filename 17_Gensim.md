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

---

# 15. Bag-of-Words Representation

Gensim often starts with bag-of-words.

Bag-of-words ignores word order and counts words.

Example:

```text
Document:
    "machine learning machine"

Vocabulary:
    machine -> 0
    learning -> 1

Bag-of-words:
    [(0, 2), (1, 1)]
```

Meaning:

```text
word ID 0 appears 2 times
word ID 1 appears 1 time
```

This representation is simple and useful for topic modeling.

---

# 16. TF-IDF

TF-IDF gives higher weight to important words and lower weight to very common words.

```text
TF  = term frequency
IDF = inverse document frequency
```

Simple meaning:

```text
Word appears often in this document
but not in every document
        |
        v
Important word
```

Code:

```python
from gensim.models import TfidfModel

tfidf = TfidfModel(corpus)
tfidf_corpus = tfidf[corpus]
```

---

# 17. Understanding LDA Topics

LDA topics are groups of words.

Example:

```text
Topic 0:
    bank, loan, credit, money, finance

Topic 1:
    team, match, goal, player, league
```

Interpretation:

```text
Topic 0 may be finance.
Topic 1 may be sports.
```

Important:

```text
LDA does not name topics automatically.
Humans inspect top words and assign meaning.
```

---

# 18. Choosing Number of Topics

`num_topics` is important.

Too few topics:

```text
Different themes get mixed together.
```

Too many topics:

```text
Topics become tiny, noisy, or repeated.
```

Practical approach:

```text
Try several values:
    5, 10, 15, 20

Inspect topics:
    Are they coherent?
    Are they useful?
    Are they too similar?
```

---

# 19. Word Embeddings

Word embeddings represent words as dense vectors.

```text
king  -> [0.21, -0.14, 0.88, ...]
queen -> [0.20, -0.10, 0.84, ...]
```

Similar words have similar vectors.

Use cases:

- Word similarity
- Recommendation
- Search
- Clustering words/documents
- Feature creation for ML models

---

# 20. Document Similarity

Gensim can compare documents.

Simple idea:

```text
Convert documents to vectors
        |
        v
Compare vector similarity
        |
        v
Find similar documents
```

Example use cases:

```text
Find duplicate articles
Recommend similar documents
Search related reports
Group similar support tickets
```

---

# 21. Common Mistakes

| Mistake | Problem | Fix |
|---|---|---|
| No preprocessing | Noisy topics | Clean/tokenize text |
| Too many stopwords | Useless topics | Remove common words |
| Wrong num_topics | Mixed or repeated topics | Try multiple values |
| Expecting LLM behavior | Gensim is not generative AI | Use Transformers for generation |
| Tiny corpus | Weak topics/embeddings | Use more documents |

---

# 22. Practical Workflow

```text
1. Collect documents
2. Clean text
3. Tokenize
4. Remove stopwords
5. Create Dictionary
6. Create Corpus
7. Train TF-IDF, LDA, Word2Vec, or Doc2Vec
8. Inspect results
9. Tune preprocessing/model parameters
10. Use topics or vectors in application
```
