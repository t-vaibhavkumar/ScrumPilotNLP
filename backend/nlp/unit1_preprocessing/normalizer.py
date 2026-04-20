"""
=============================================================
Unit 1 — Text Normalization
=============================================================
Normalizes raw meeting transcript text before NLP processing:
  • Lowercasing
  • Contraction expansion     (I'm → I am)
  • Speaker-label removal     (Sarah: → "")
  • Filler-word removal       (um, uh, like …)
  • Punctuation removal
  • Stopword removal          (optional — skip for LSTM/BERT)
  • Whitespace cleanup

Syllabus: Unit 1 — Text preprocessing: normalization
Run     : python backend/nlp/unit1_preprocessing/normalizer.py
=============================================================
"""

import re
from typing import List

import nltk
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

# ── Constants ─────────────────────────────────────────────

STOP_WORDS = set(stopwords.words("english"))

FILLER_WORDS = {
    "um", "uh", "hmm", "like", "basically", "actually",
    "you know", "i mean", "so", "well", "right", "okay", "ok",
}

CONTRACTIONS = {
    "i'm": "i am",        "i've": "i have",    "i'll": "i will",   "i'd": "i would",
    "we're": "we are",    "we've": "we have",  "we'll": "we will", "we'd": "we would",
    "it's": "it is",      "it'll": "it will",  "isn't": "is not",  "aren't": "are not",
    "wasn't": "was not",  "weren't": "were not","don't": "do not",  "doesn't": "does not",
    "didn't": "did not",  "can't": "cannot",   "couldn't": "could not",
    "won't": "will not",  "wouldn't": "would not", "shouldn't": "should not",
    "that's": "that is",  "there's": "there is","they're": "they are","they've": "they have",
    "you're": "you are",  "you've": "you have", "you'll": "you will",
    "he's": "he is",      "she's": "she is",    "let's": "let us",
    "what's": "what is",  "where's": "where is","who's": "who is",
}


# ── Step-by-step functions ────────────────────────────────

def to_lowercase(text: str) -> str:
    """Convert all characters to lowercase."""
    return text.lower()


def expand_contractions(text: str) -> str:
    """Expand English contractions: can't → cannot, I'm → I am."""
    for contraction, expansion in CONTRACTIONS.items():
        text = re.sub(
            r"\b" + re.escape(contraction) + r"\b",
            expansion,
            text,
            flags=re.IGNORECASE,
        )
    return text


def remove_speaker_labels(text: str) -> str:
    """
    Remove 'SpeakerName:' prefixes from diarized transcripts.
    e.g. 'Sarah: I completed the API' → 'I completed the API'
    """
    return re.sub(r"^[A-Za-z][A-Za-z ]{1,30}:\s*", "", text, flags=re.MULTILINE)


def remove_filler_words(text: str) -> str:
    """Remove spoken filler words common in meeting recordings."""
    words = text.split()
    return " ".join(w for w in words if w.lower() not in FILLER_WORDS)


def remove_punctuation(text: str) -> str:
    """Remove all punctuation, keeping only alphanumeric and spaces."""
    return re.sub(r"[^\w\s]", "", text)


def remove_stopwords_from(text: str) -> str:
    """
    Remove English stopwords.
    Use this for BoW/TF-IDF — NOT for LSTM/BERT (those need full sentences).
    """
    words = text.split()
    return " ".join(w for w in words if w.lower() not in STOP_WORDS)


def clean_whitespace(text: str) -> str:
    """Collapse multiple spaces and strip leading/trailing whitespace."""
    return re.sub(r"\s+", " ", text).strip()


# ── Full pipeline ─────────────────────────────────────────

def normalize(text: str, remove_stops: bool = False) -> str:
    """
    Full text normalization pipeline.

    Args:
        text         : Raw meeting transcript text
        remove_stops : If True, removes stopwords.
                       Set False for LSTM/GRU/BERT (need full context).
                       Set True for BoW/TF-IDF (reduces noise).

    Returns:
        Clean normalized string
    """
    text = to_lowercase(text)
    text = expand_contractions(text)
    text = remove_speaker_labels(text)
    text = remove_filler_words(text)
    text = remove_punctuation(text)
    if remove_stops:
        text = remove_stopwords_from(text)
    text = clean_whitespace(text)
    return text


# ── Standalone Demo ───────────────────────────────────────

if __name__ == "__main__":
    RAW = (
        "Sarah: Um, I've completed the user auth API, you know.\n"
        "Mike: I'm working on payment gateway, it'll be done by tomorrow basically.\n"
        "Scrum Master: Great! We've got to make sure it's tested before Friday."
    )

    print("=" * 65)
    print("         UNIT 1 — TEXT NORMALIZATION DEMO")
    print("=" * 65)
    print(f"\nRAW INPUT:\n{RAW}")
    print("\nSTEP-BY-STEP PIPELINE:")
    print("-" * 45)

    t = RAW
    steps = [
        ("1. Lowercase",           to_lowercase(t)),
        ("2. Expand contractions", expand_contractions(to_lowercase(t))),
        ("3. Remove speaker tags", remove_speaker_labels(expand_contractions(to_lowercase(t)))),
        ("4. Remove filler words", remove_filler_words(
                                   remove_speaker_labels(expand_contractions(to_lowercase(t))))),
    ]
    for name, result in steps:
        print(f"\n  [{name}]")
        print(f"   → {result[:120]}")

    print("\n" + "-" * 45)
    full = normalize(RAW)
    full_no_sw = normalize(RAW, remove_stops=True)
    print(f"\nFINAL  (with stopwords) : {full}")
    print(f"FINAL  (no  stopwords)  : {full_no_sw}")
    print(f"\n  ℹ  Keep stopwords for LSTM/GRU/BERT (need full sentence context)")
    print(f"  ℹ  Remove stopwords for BoW/TF-IDF  (reduces vocabulary noise)")
    print()
