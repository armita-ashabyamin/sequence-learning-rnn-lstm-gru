# Recurrent Networks: From Formal Patterns to Language Understanding

> **Course:** Large Language Models  
> **Author:** آرمیتا اصحاب یمین

---

## 📝 Overview

This repository explores the capabilities and limitations of recurrent neural networks (RNNs) across three distinct sequence learning tasks:

1. **Formal Pattern Recognition** — Can LSTMs learn and generalize regular language patterns?
2. **Creative Text Generation** — Can GRUs generate coherent text from character-level training?
3. **Semantic Understanding** — Can LSTMs predict sentiment from sentence embeddings?

The implementations reveal fundamental insights about RNN generalization, memorization vs. learning, and the gap between simple pattern recognition and true language understanding.

---

## 📁 Project Structure

```
.
├── RNN_Experiment.ipynb        # Complete implementation notebook
├── Book.txt                   # Required: Text corpus
├── dictionary.txt             # Required: Sentiment treebank dictionary
├── sentiment_labels.txt       # Required: Sentiment treebank labels
├── README.md                  # This file
└── .gitignore                 # Git ignore rules
```

---

## 🚀 Getting Started

### Prerequisites

```bash
python >= 3.8
tensorflow >= 2.0
numpy >= 1.19
pandas >= 1.0
gensim >= 4.0
scikit-learn >= 0.24
```

### Installation

```bash
pip install tensorflow numpy pandas gensim scikit-learn
```

### Dataset Setup

| File | Description | Source |
|------|-------------|--------|
| `Book.txt` | Plain text book | Any public domain book |
| `dictionary.txt` | Phrase-to-ID mapping | Stanford Sentiment Treebank |
| `sentiment_labels.txt` | Sentiment scores (0-1) | Stanford Sentiment Treebank |

---

## 🧪 Tasks & Methodology

### Task 1: Pattern Recognition — Testing LSTM Generalization

**Formal Language:** Can an LSTM learn the rule behind a pattern, or just memorize examples?

#### 1a. Pattern `a^k N b^k`

```
Examples: aN b E, aaN bb E, aaaN bbb E, ...
```

- **Model:** LSTM with 10 hidden units
- **Training:** k = 1..10 (all prefixes, next-character prediction)
- **Input:** 40 chars, left-padded
- **Test:** k = 15 (unseen length)

#### 1b. Pattern `(ab)^k N (ab)^k`

```
Examples: abN abE, ababN ababE, abababN abababE, ...
```

- **Model:** LSTM with 20 hidden units
- **Training:** k = 1..15
- **Input:** 80 chars, left-padded
- **Test:** k > 15

**Key Question:** Does the LSTM learn the underlying pattern or simply memorize training examples?

---

### Task 2: Text Generation — Character-Level GRU Language Model

**Methodology:**
- Sliding windows: 40 characters, 35 overlap (stride = 5)
- **Architecture:** GRU(128) → Dense(vocab_size, softmax)
- **Optimizer:** RMSProp (learning rate = 0.01)
- **Generation:** Temperature sampling (T=0.8) for 400 characters

**Evaluation:**
- Word count in generated text
- Overlap with book vocabulary (semantic coherence)

**Sample Generation:**
> "e the meaning of it? It was impossible that me, and tauph shate offere, 'Mr. Colling hards him coured to be in Lady plise is me, 'the asded losited, ard a for a she had been amiable..."

---

### Task 3: Sentiment Analysis — Semantic Understanding with LSTM

**Architecture:**
```
Input (max_words, 100d embeddings)
    ↓
Masking (ignore padded zeros)
    ↓
LSTM(256)
    ↓
Dense(1, sigmoid) → Sentiment score [0, 1]
```

**Data:** Stanford Sentiment Treebank (60% train / 40% test)

**Handling Variable Lengths:**
1. Pad all sentences to max length with zero vectors
2. Masking layer tells LSTM to ignore padded positions
3. LSTM processes only real words

**Metrics:**
- Mean Squared Error (MSE)
- Accuracy: `|predicted - true| < 0.05`

---

## 📊 Results & Insights

| Task | Architecture | Key Finding |
|------|-------------|-------------|
| Pattern A (k=15) | LSTM(10) | ❌ **Fails:** Generates endless 'b's, no 'E' token |
| Pattern B (k=15) | LSTM(20) | ❌ **Fails:** Repeats 'ab' instead of pattern |
| Text Generation | GRU(128) | ✅ **Partial:** 67.9% words in book vocab |
| Sentiment Analysis | LSTM(256) | ⚠️ **Moderate:** MSE=0.0308, Acc=34.5% |

### Key Takeaways

1. **Small LSTMs memorize, not learn rules** — They fail spectacularly on unseen lengths, revealing the gap between pattern recognition and true generalization.

2. **Character-level modeling works locally** — GRUs can learn character-level patterns and spelling, but global coherence remains elusive.

3. **Sentiment is nuanced** — Even with 256 LSTM units, sentence-level sentiment prediction is challenging, with only 34.5% of predictions within 5% of the true label.

4. **Temperature matters** — The balance between creativity (T > 1) and determinism (T < 1) dramatically affects generation quality.

---

## 🔧 Implementation Details

| Feature | Implementation |
|---------|---------------|
| Pattern encoding | One-hot vectors for {a, b, N, E} |
| Text preprocessing | Sliding windows, character-level |
| Word embeddings | Word2Vec (100d, trained on corpus) |
| Sequence padding | Left-padding for patterns, right-padding for sentences |
| Masking | Keras `Masking` layer for sentiment task |
| Autoregressive generation | Greedy decoding + temperature sampling |
| Reproducibility | Fixed random seeds (42) |

---

## 🎯 What This Repository Teaches

- **Generalization limits** of RNNs on structured sequences
- **Practical implementation** of LSTM, GRU with TensorFlow/Keras
- **Autoregressive generation** techniques
- **Handling variable-length sequences** with padding and masking
- **Evaluation methodologies** for generative and predictive tasks

---

## 📚 References

1. Chung, J., Gulcehre, C., Cho, K., & Bengio, Y. (2014). *Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling*. arXiv:1412.3555

2. Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). *Efficient Estimation of Word Representations in Vector Space*. arXiv:1301.3781

3. Socher, R., Perelygin, A., Wu, J., Chuang, J., Manning, C. D., Ng, A. Y., & Potts, C. (2013). *Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank*. EMNLP 2013.

4. Hochreiter, S., & Schmidhuber, J. (1997). *Long Short-Term Memory*. Neural Computation, 9(8), 1735-1780.

---

## 🧑‍💻 Author

**آرمیتا اصحاب یمین**  
Armita Ashabyamin

---


## 🤝 Contributing

This is a course assignment; contributions are not expected. However, feel free to fork and experiment!

---

## ⚠️ Disclaimer

This code is for educational purposes. The pattern recognition models deliberately use small architectures to demonstrate generalization limitations, not state-of-the-art performance.
```

---

## LICENSE

```text
MIT License

Copyright (c) 2026 آرمیتا اصحاب یمین

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

