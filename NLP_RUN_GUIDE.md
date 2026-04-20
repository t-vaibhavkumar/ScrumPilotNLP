# ScrumPilotNLP — NLP Modules: Run Guide

## Setup (already done)
```bash
# 1. Install NLP dependencies
uv pip install -r nlp_requirements.txt   # ✅ done

# 2. Download spaCy model
python -m spacy download en_core_web_sm  # ✅ done
```

---

## Run Each Demo (in order)

### Unit 1 — Preprocessing
```bash
python nlp_demos/demo_unit1_preprocessing.py
```
**What it shows:** Word tokenization → Sentence tokenization → BERT subword tokenization → Text normalization → Lemmatization + POS tagging → NER  
**Status: ✅ VERIFIED WORKING**

---

### Unit 1 — Representations
```bash
python nlp_demos/demo_unit1_representations.py
```
**What it shows:** BoW matrix + cosine similarity → TF-IDF retrieval (story mapping) → N-gram language model + perplexity  
**Status: ✅ VERIFIED WORKING**

---

### Unit 2 — Neural Models
```bash
python nlp_demos/demo_unit2_neural.py
```
**What it shows:** Word2Vec embeddings → LSTM classifier (trains live) → GRU classifier (trains live) → Bahdanau Attention + weight visualization → F1/BLEU/ROUGE evaluation  
**Status: ✅ VERIFIED WORKING**

---

### Unit 3 — Transformers
```bash
python nlp_demos/demo_unit3_transformers.py
```
**What it shows:** BERT contextual embeddings → Sentence-BERT semantic search → CNN text classifier → Model comparison table  
**Note:** Downloads DistilBERT (~250MB) + all-MiniLM-L6-v2 (~90MB) on first run. Cached after that.  
**Time:** ~3-5 min first run, ~1 min after cache

---

### Unit 4 — Applications
```bash
python nlp_demos/demo_unit4_applications.py
```
**What it shows:** Extractive summarization (TF-IDF) → Abstractive summarization (DistilBART) → Extractive QA (DistilBERT-SQuAD) → Text generation (Flan-T5)  
**Note:** Downloads DistilBART (~900MB) + Flan-T5-small (~300MB) + DistilBERT-SQuAD (~250MB) on first run.  
**Time:** ~5-10 min first run

---

### Unit 5 — Speech
```bash
python nlp_demos/demo_unit5_speech.py
```
**What it shows:** ASR pipeline overview (Whisper) → TTS via pyttsx3 (speaks aloud) → gTTS saves MP3 → full Scrum sprint summary read aloud  
**Time:** ~10 sec

---

## Run Individual Modules Directly

Each module also has its own `__main__` demo:

```bash
# Unit 1
python backend/nlp/unit1_preprocessing/tokenizer.py
python backend/nlp/unit1_preprocessing/normalizer.py
python backend/nlp/unit1_preprocessing/lemmatizer.py
python backend/nlp/unit1_preprocessing/ner.py
python backend/nlp/unit1_representations/bow.py
python backend/nlp/unit1_representations/tfidf.py
python backend/nlp/unit1_representations/ngram_lm.py

# Unit 2
python backend/nlp/unit2_models/word_embeddings.py
python backend/nlp/unit2_models/lstm_classifier.py
python backend/nlp/unit2_models/gru_classifier.py
python backend/nlp/unit2_models/attention.py
python backend/nlp/unit2_models/evaluator.py

# Unit 3
python backend/nlp/unit3_transformers/bert_embeddings.py
python backend/nlp/unit3_transformers/sentence_bert.py
python backend/nlp/unit3_transformers/bert_classifier.py
python backend/nlp/unit3_transformers/cnn_text_classifier.py

# Unit 4
python backend/nlp/unit4_applications/summarizer.py
python backend/nlp/unit4_applications/qa_system.py
python backend/nlp/unit4_applications/text_generator.py

# Unit 5
python backend/nlp/unit5_speech/tts.py
```

---

## What Each Module Replaces (NLP vs Groq)

| Was Groq doing | Now handled by | Module |
|---|---|---|
| "What meeting type is this?" | LSTM classifier | `lstm_classifier.py` |
| "What action is this sentence?" | GRU classifier | `gru_classifier.py` |
| "Find the Epic in this text" | spaCy NER | `ner.py` |
| "Map phrase to story ID" | Sentence-BERT cosine similarity | `sentence_bert.py` |
| "Classify Epic priority" | DistilBERT + LogisticRegression | `bert_classifier.py` |
| "Summarize this meeting" | DistilBART abstractive / TF-IDF extractive | `summarizer.py` |
| "Answer questions about backlog" | DistilBERT-SQuAD extractive QA | `qa_system.py` |
| "Generate Epic description" | Flan-T5-small seq2seq | `text_generator.py` |

---

## Syllabus Coverage Checklist

| Topic | Module | Status |
|---|---|---|
| Tokenization (word, sentence, subword) | `tokenizer.py` | ✅ |
| Text Normalization | `normalizer.py` | ✅ |
| Lemmatization + POS Tagging | `lemmatizer.py` | ✅ |
| Named Entity Recognition | `ner.py` | ✅ |
| Bag of Words (BoW) | `bow.py` | ✅ |
| TF-IDF | `tfidf.py` | ✅ |
| N-gram Language Model + Perplexity | `ngram_lm.py` | ✅ |
| Word Embeddings (Word2Vec) | `word_embeddings.py` | ✅ |
| LSTM Text Classification | `lstm_classifier.py` | ✅ |
| GRU Text Classification | `gru_classifier.py` | ✅ |
| Attention Mechanism (Bahdanau) | `attention.py` | ✅ |
| Evaluation: P/R/F1, BLEU, ROUGE | `evaluator.py` | ✅ |
| CNN for Text Classification | `cnn_text_classifier.py` | ✅ |
| BERT Contextual Embeddings | `bert_embeddings.py` | ✅ |
| Sentence-BERT (Semantic Search) | `sentence_bert.py` | ✅ |
| Fine-tuning BERT (Transfer Learning) | `bert_classifier.py` | ✅ |
| Text Summarization (Extractive + Abstractive) | `summarizer.py` | ✅ |
| Question Answering System | `qa_system.py` | ✅ |
| Language Generation (T5) | `text_generator.py` | ✅ |
| ASR (Whisper) | `backend/speech/whisperai/` | ✅ (existing) |
| Text-to-Speech | `tts.py` | ✅ |
