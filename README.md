# 🤖 Natural Language Processing Portfolio

> A collection of NLP projects demonstrating text classification, sentiment analysis, named entity recognition, and more using modern deep learning techniques.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12+-orange.svg)](https://www.tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/transformers/)

![NLP Projects Overview](assets/nlp_overview.png)

## 📚 Projects Overview

This repository contains 8+ NLP projects covering various domains and techniques, from traditional machine learning to state-of-the-art transformer models.

## 🎯 Featured Projects

### 1. Sentiment Analysis with BERT
**Goal:** Multi-class sentiment classification of product reviews

**Highlights:**
- Fine-tuned BERT model achieving **92% accuracy**
- Handles 5-star rating prediction
- Real-time inference API with FastAPI
- Deployed on AWS Lambda

**Tech:** PyTorch, Transformers, FastAPI, Docker

[📂 View Project](./sentiment-analysis-bert/)

---

### 2. Named Entity Recognition (NER) System
**Goal:** Extract entities (Person, Organization, Location, Date) from news articles

**Highlights:**
- BiLSTM-CRF architecture with 89% F1-score
- Custom entity types for domain-specific use cases
- Spacy integration for production deployment
- Trained on CoNLL-2003 dataset + custom annotations

**Tech:** TensorFlow, Spacy, CRF, Jupyter

[📂 View Project](./named-entity-recognition/)

---

### 3. Text Classification with Transfer Learning
**Goal:** Multi-label classification of research paper abstracts

**Highlights:**
- Compared BERT, RoBERTa, and DistilBERT
- Achieved 94% accuracy with ensemble approach
- Label: Computer Science subfields (ML, NLP, CV, etc.)
- Handles imbalanced datasets with focal loss

**Tech:** Transformers, Scikit-learn, Weights & Biases

[📂 View Project](./text-classification/)

---

### 4. Question Answering System
**Goal:** Build extractive QA system for customer support

**Highlights:**
- Fine-tuned BERT-QA on SQuAD 2.0
- 86% F1-score on test set
- Interactive Streamlit demo
- Context-aware answer extraction

**Tech:** Transformers, Streamlit, Elasticsearch

[📂 View Project](./question-answering/)

---

### 5. Text Summarization (Abstractive)
**Goal:** Generate concise summaries of long documents

**Highlights:**
- Implemented T5 and BART models
- Evaluated with ROUGE scores (ROUGE-L: 0.42)
- Beam search and nucleus sampling for generation
- Applied to news articles and research papers

**Tech:** Transformers, NLTK, PyTorch

[📂 View Project](./text-summarization/)

---

### 6. Chatbot with Intent Classification
**Goal:** Rule-based + ML hybrid chatbot for FAQs

**Highlights:**
- Intent classification with 95% accuracy
- Entity extraction for slot filling
- Dialogue state tracking
- Deployed as Slack bot

**Tech:** Rasa, BERT, FastAPI, Redis

[📂 View Project](./chatbot/)

---

### 7. Topic Modeling with LDA & BERTopic
**Goal:** Discover hidden topics in large text corpus

**Highlights:**
- Compared LDA vs BERTopic
- Visualized topic distributions with pyLDAvis
- Applied to 100K+ Reddit posts
- Identified 20 coherent topics

**Tech:** Gensim, BERTopic, Scikit-learn

[📂 View Project](./topic-modeling/)

---

### 8. Spam Detection System
**Goal:** Binary classification of spam vs ham messages

**Highlights:**
- Ensemble of Naive Bayes, SVM, and LSTM
- 98.5% accuracy on test set
- Real-time prediction API
- Deployed with 99.9% uptime

**Tech:** Scikit-learn, TensorFlow, Flask

[📂 View Project](./spam-detection/)

## 🛠️ Common Tech Stack

- **Languages:** Python 3.9+
- **DL Frameworks:** PyTorch, TensorFlow, Keras
- **NLP Libraries:** 
  - Transformers (Hugging Face)
  - spaCy
  - NLTK
  - Gensim
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Deployment:** FastAPI, Docker, AWS

## 📊 Performance Summary

| Project | Model | Accuracy/F1 | Dataset Size |
|---------|-------|-------------|--------------|
| Sentiment Analysis | BERT | 92% | 50K reviews |
| NER | BiLSTM-CRF | 89% F1 | CoNLL-2003 |
| Text Classification | RoBERTa | 94% | 20K papers |
| Question Answering | BERT-QA | 86% F1 | SQuAD 2.0 |
| Summarization | T5-base | 0.42 ROUGE-L | CNN/DailyMail |
| Intent Classification | BERT | 95% | 10K utterances |
| Spam Detection | Ensemble | 98.5% | 5K messages |

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.9+
CUDA 11.8+ (for GPU support)
```

### Installation

1. Clone the repository
```bash
git clone https://github.com/OSP06/NLP-Projects.git
cd NLP-Projects
```

2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Download required models
```bash
python scripts/download_models.py
```

## 📁 Repository Structure
```
nlp-projects/
├── sentiment-analysis-bert/
│   ├── notebooks/
│   ├── src/
│   ├── models/
│   └── README.md
├── named-entity-recognition/
├── text-classification/
├── question-answering/
├── text-summarization/
├── chatbot/
├── topic-modeling/
├── spam-detection/
├── common/
│   ├── utils.py
│   ├── preprocessing.py
│   └── evaluation.py
├── requirements.txt
└── README.md
```

## 🎓 Key Learnings

- **Transformer Architecture:** Deep understanding of attention mechanisms
- **Transfer Learning:** Fine-tuning pre-trained models for specific tasks
- **Deployment:** Production-ready NLP systems with APIs
- **Evaluation:** Proper metrics for different NLP tasks
- **Data Augmentation:** Techniques for handling limited labeled data

## 📈 Skills Demonstrated

✅ Text Preprocessing & Feature Engineering
✅ Classical ML (Naive Bayes, SVM, Random Forest)
✅ Deep Learning (RNN, LSTM, BiLSTM, Transformers)
✅ Transfer Learning (BERT, RoBERTa, T5, GPT)
✅ Model Evaluation & Hyperparameter Tuning
✅ API Development & Deployment
✅ MLOps (Docker, CI/CD, Monitoring)

## 🔮 Future Work

- [ ] Add multilingual NLP projects
- [ ] Implement GPT-based projects
- [ ] Add speech-to-text integration
- [ ] Build end-to-end NLP pipeline
- [ ] Add real-time processing examples

## 📚 Resources & References

- [Hugging Face Course](https://huggingface.co/course)
- [Stanford CS224N: NLP with Deep Learning](http://web.stanford.edu/class/cs224n/)
- [Papers with Code - NLP](https://paperswithcode.com/area/natural-language-processing)

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit PRs.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 👤 Author

**Om Patel**
- GitHub: [@OSP06](https://github.com/OSP06)
- LinkedIn: [om-sanjay-patel](https://linkedin.com/in/om-sanjay-patel)
- Email: your.email@example.com

---

⭐️ If you find these projects helpful, please star the repo!
