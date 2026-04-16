"""
=============================================================
Unit 1 — Lemmatization & POS Tagging
=============================================================
Uses spaCy's English NLP pipeline to:
  • Lemmatize tokens  (running→run, stories→story, built→build)
  • Assign Universal POS tags  (NOUN, VERB, ADJ, ADV, …)
  • Assign fine-grained tags   (NNS, VBG, JJ, …)
  • Identify stopwords and morphological features

Why lemmatize?
  → Reduces vocabulary size for BoW/TF-IDF
  → Groups inflected forms ("runs", "running", "ran") → "run"
  → Improves model generalisation

Syllabus: Unit 1 — Lemmatization, POS tagging and NER
Run     : python backend/nlp/unit1_preprocessing/lemmatizer.py
=============================================================
"""

from typing import List, Dict

import spacy

# Load spaCy model — download once with:
#   python -m spacy download en_core_web_sm
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess, sys
    print("Downloading spaCy model en_core_web_sm …")
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
    nlp = spacy.load("en_core_web_sm")


# ── Core functions ────────────────────────────────────────

def lemmatize_and_pos(text: str) -> List[Dict]:
    """
    Run spaCy pipeline on text and return token-level analysis.

    Returns:
        List of dicts:
          token    — original word form
          lemma    — dictionary (base) form
          pos      — Universal POS tag (NOUN, VERB, ADJ …)
          tag      — Fine-grained Penn Treebank tag (NN, VBG, JJ …)
          is_stop  — whether word is a stopword
    """
    doc = nlp(text)
    results = []
    for token in doc:
        if token.is_space:
            continue
        results.append({
            "token":   token.text,
            "lemma":   token.lemma_,
            "pos":     token.pos_,       # Universal POS (NOUN, VERB …)
            "tag":     token.tag_,       # Fine-grained (NNS, VBG …)
            "is_stop": token.is_stop,
        })
    return results


def extract_lemmas(text: str, pos_filter: List[str] = None) -> List[str]:
    """
    Extract lemmas from text, optionally filtered by POS.

    Args:
        text       : Input string
        pos_filter : e.g. ["NOUN","VERB"] — return only these POS classes
                     None → return all non-stop, non-punct lemmas

    Returns:
        Sorted list of unique lemmas (lowercase)
    """
    doc = nlp(text)
    lemmas = []
    for token in doc:
        if token.is_stop or token.is_punct or token.is_space:
            continue
        if pos_filter and token.pos_ not in pos_filter:
            continue
        lemmas.append(token.lemma_.lower())
    return lemmas


# POS tag descriptions for display
POS_DESCRIPTIONS = {
    "NOUN":  "Noun",
    "VERB":  "Verb",
    "ADJ":   "Adjective",
    "ADV":   "Adverb",
    "PROPN": "Proper Noun",
    "PRON":  "Pronoun",
    "DET":   "Determiner",
    "ADP":   "Preposition",
    "CONJ":  "Conjunction",
    "CCONJ": "Coordinating Conjunction",
    "SCONJ": "Subordinating Conjunction",
    "AUX":   "Auxiliary Verb",
    "NUM":   "Number",
    "PUNCT": "Punctuation",
}


# ── Standalone Demo ───────────────────────────────────────

if __name__ == "__main__":
    TEXT = (
        "The developers are building user stories and running sprint planning meetings "
        "to estimate the remaining backlog items and prioritize epics."
    )

    print("=" * 70)
    print("        UNIT 1 — LEMMATIZATION + POS TAGGING DEMO")
    print("=" * 70)
    print(f"\nInput: {TEXT}\n")

    results = lemmatize_and_pos(TEXT)

    # ── Token table ───────────────────────────────────────
    print(f"{'Token':<20} {'Lemma':<20} {'POS':<8} {'Tag':<8} {'Stop'}")
    print("-" * 68)
    for r in results:
        stop_flag = "✓" if r["is_stop"] else ""
        print(f"{r['token']:<20} {r['lemma']:<20} {r['pos']:<8} {r['tag']:<8} {stop_flag}")

    # ── Filtered lemmas ───────────────────────────────────
    print("\nNOUN + VERB lemmas only (keyword extraction input):")
    key_lemmas = extract_lemmas(TEXT, pos_filter=["NOUN", "VERB"])
    print(f"  → {key_lemmas}")

    # ── Scrum-domain examples ─────────────────────────────
    print("\nScrum-domain lemmatization examples:")
    examples = [
        "The teams are sprint planning and grooming stories",
        "Developers estimated twenty story points for the epics",
        "Running retrospectives helps teams improve their velocities",
        "The authentication stories were prioritised in Q2",
    ]
    for ex in examples:
        lemmas = extract_lemmas(ex, pos_filter=["NOUN", "VERB"])
        print(f"\n  INPUT : {ex}")
        print(f"  LEMMAS: {lemmas}")

    # ── POS distribution ──────────────────────────────────
    from collections import Counter
    pos_counts = Counter(r["pos"] for r in results if not r["is_stop"])
    print("\nPOS distribution (content words only):")
    for pos, count in pos_counts.most_common():
        desc = POS_DESCRIPTIONS.get(pos, pos)
        print(f"  {pos:<8} ({desc:<25}) : {count}")
    print()
