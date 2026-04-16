"""
=============================================================
Unit 1 — Text Tokenization
=============================================================
Demonstrates three levels of tokenization:
  1. Word tokenization       (NLTK)
  2. Sentence tokenization   (NLTK)
  3. Subword tokenization    (BERT WordPiece via HuggingFace)

Syllabus: Unit 1 — Text preprocessing: tokenization
Run     : python backend/nlp/unit1_preprocessing/tokenizer.py
=============================================================
"""

import re
import nltk
from typing import List, Dict

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
from nltk.tokenize import word_tokenize, sent_tokenize


# ─────────────────────────────────────────────────────────
# 1. Word Tokenization
# ─────────────────────────────────────────────────────────

def word_tok(text: str) -> List[str]:
    """
    Tokenize text into individual words using NLTK Punkt.
    Handles punctuation, contractions and special characters.
    """
    return word_tokenize(text)


# ─────────────────────────────────────────────────────────
# 2. Sentence Tokenization
# ─────────────────────────────────────────────────────────

def sentence_tok(text: str) -> List[str]:
    """
    Split a paragraph / transcript into individual sentences.
    Uses NLTK's trained Punkt sentence boundary detector.
    """
    return sent_tokenize(text)


# ─────────────────────────────────────────────────────────
# 3. BERT Subword (WordPiece) Tokenization
# ─────────────────────────────────────────────────────────

def bert_tok(text: str, model_name: str = "bert-base-uncased") -> Dict:
    """
    Tokenize text using BERT's WordPiece algorithm.

    WordPiece splits unknown/rare words into common subword pieces:
      "authentication" → ["authentication"]   (known word)
      "scrum-piloting" → ["scrum", "-", "pilot", "##ing"]   (subwords)
      '##' prefix means: this piece continues the previous token

    Special tokens added automatically:
      [CLS] — start of sequence (used for classification)
      [SEP] — end of sequence

    Returns:
        dict with tokens, input_ids, token_count
    """
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(model_name)

    encoded = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    )
    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])

    return {
        "tokens":      tokens,
        "input_ids":   encoded["input_ids"][0].tolist(),
        "token_count": len(tokens),
    }


# ─────────────────────────────────────────────────────────
# Standalone Demo
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    TRANSCRIPT = (
        "Sarah: I completed the user authentication API yesterday.\n"
        "Mike: I'm working on the payment gateway integration. "
        "Expected to finish by tomorrow.\n"
        "Scrum Master: Great! Any blockers?\n"
        "Sarah: No blockers from my side. I'll start the sprint review prep today."
    )

    print("=" * 65)
    print("          UNIT 1 — TEXT TOKENIZATION DEMO")
    print("=" * 65)

    # ── 1. Word Tokenization ──────────────────────────────
    print("\n[1] WORD TOKENIZATION  (NLTK Punkt)")
    print("-" * 45)
    words = word_tok(TRANSCRIPT)
    print(f"  First 15 tokens : {words[:15]}")
    print(f"  Total word tokens: {len(words)}")

    # ── 2. Sentence Tokenization ──────────────────────────
    print("\n[2] SENTENCE TOKENIZATION  (NLTK Punkt)")
    print("-" * 45)
    sents = sentence_tok(TRANSCRIPT)
    for i, s in enumerate(sents, 1):
        print(f"  Sent {i}: {s.strip()}")

    # ── 3. BERT Subword Tokenization ──────────────────────
    print("\n[3] SUBWORD TOKENIZATION  (BERT WordPiece)")
    print("-" * 45)
    test_cases = [
        "The backlog grooming session produced sprint-ready stories",
        "authentication microservice scrum-piloting",
    ]
    for text in test_cases:
        result = bert_tok(text)
        print(f"\n  Input  : {text}")
        print(f"  Tokens : {result['tokens']}")
        print(f"  Count  : {result['token_count']} subword tokens")
        print("  Note   : '##' = continuation of previous token")

    print()
