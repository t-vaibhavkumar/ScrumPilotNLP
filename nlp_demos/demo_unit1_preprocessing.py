"""
=============================================================
Demo — Unit 1: Preprocessing Pipeline
=============================================================
Runs all Unit 1 preprocessing steps on a full Scrum meeting
transcript and prints results to console.

Steps:
  1. Word & Sentence Tokenization (NLTK)
  2. Subword Tokenization (BERT WordPiece)
  3. Text Normalization (filler words, contractions, speaker labels)
  4. Lemmatization + POS Tagging (spaCy)
  5. Named Entity Recognition (spaCy)

Run: python nlp_demos/demo_unit1_preprocessing.py
=============================================================
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.nlp.unit1_preprocessing.tokenizer  import word_tok, sentence_tok, bert_tok
from backend.nlp.unit1_preprocessing.normalizer  import normalize
from backend.nlp.unit1_preprocessing.lemmatizer  import lemmatize_and_pos, extract_lemmas
from backend.nlp.unit1_preprocessing.ner         import extract_entities, extract_assignees, extract_estimates, extract_dates


# ── Full meeting transcript ───────────────────────────────
TRANSCRIPT = """
Sarah: Um, good morning everyone. Let's kick off today's standup.

Mike: So, yesterday I completed the Stripe payment gateway integration —
pushed it to staging. Today I'll basically be working on the unit tests
for the authentication module. No blockers from my side, you know.

Alice: I'm, uh, still working on the CI pipeline setup with GitHub Actions.
It should be done by tomorrow. I'm waiting on DevOps to give me AWS credentials —
that's my blocker actually.

Tom: Yesterday I reviewed Mike's pull request and estimated eight story points
for the new analytics dashboard epic. Today I'll start implementing the
REST API endpoints. No blockers.

Scrum Master: Great! Sprint ends this Friday April 25th. We have 34 story points
committed and the team capacity is 80 hours. Sarah will organize the sprint review.
"""


def separator(title: str) -> None:
    print(f"\n{'=' * 65}")
    print(f"  {title}")
    print("=" * 65)


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║      UNIT 1 — PREPROCESSING PIPELINE DEMO               ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"\nInput: Scrum Standup Transcript ({len(TRANSCRIPT.split())} words)\n")
    print(TRANSCRIPT.strip())

    # ── Step 1: Word Tokenization ─────────────────────────
    separator("STEP 1: WORD TOKENIZATION  (NLTK Punkt)")
    tokens = word_tok(TRANSCRIPT)
    print(f"  First 20 tokens : {tokens[:20]}")
    print(f"  Total tokens    : {len(tokens)}")

    # ── Step 2: Sentence Tokenization ────────────────────
    separator("STEP 2: SENTENCE TOKENIZATION  (NLTK Punkt)")
    sents = sentence_tok(TRANSCRIPT)
    for i, s in enumerate(sents, 1):
        print(f"  Sent {i:2d}: {s.strip()[:80]}")

    # ── Step 3: BERT Subword Tokenization ─────────────────
    separator("STEP 3: SUBWORD TOKENIZATION  (BERT WordPiece)")
    example = "CI/CD pipeline with GitHub-Actions authentication microservice"
    result  = bert_tok(example)
    print(f"  Input  : {example}")
    print(f"  Tokens : {result['tokens']}")
    print(f"  Count  : {result['token_count']} subword tokens")
    print("  Note   : '##' = subword continuation of previous token")

    # ── Step 4: Normalization ─────────────────────────────
    separator("STEP 4: TEXT NORMALIZATION")
    norm_full   = normalize(TRANSCRIPT)
    norm_nostop = normalize(TRANSCRIPT, remove_stops=True)
    print(f"  With stopwords : {norm_full[:150]}…")
    print(f"  No  stopwords  : {norm_nostop[:150]}…")

    # ── Step 5: Lemmatization + POS ───────────────────────
    separator("STEP 5: LEMMATIZATION + POS TAGGING  (spaCy)")
    sample_sent = "The developers are building user stories and running sprint planning"
    results     = lemmatize_and_pos(sample_sent)
    print(f"  Input: {sample_sent}\n")
    print(f"  {'Token':<20} {'Lemma':<20} {'POS':<8} {'Stop'}")
    print("  " + "-" * 55)
    for r in results:
        stop = "✓" if r["is_stop"] else ""
        print(f"  {r['token']:<20} {r['lemma']:<20} {r['pos']:<8} {stop}")
    key_lemmas = extract_lemmas(sample_sent, pos_filter=["NOUN", "VERB"])
    print(f"\n  Key lemmas (NOUN+VERB): {key_lemmas}")

    # ── Step 6: NER ───────────────────────────────────────
    separator("STEP 6: NAMED ENTITY RECOGNITION  (spaCy)")
    entities = extract_entities(TRANSCRIPT)
    if entities:
        print(f"  {'Entity':<25} {'Label':<12} {'Description'}")
        print("  " + "-" * 60)
        for e in entities:
            print(f"  {e['text']:<25} {e['label']:<12} {e['description']}")
    else:
        print("  No entities detected.")

    print(f"\n  Extracted:")
    print(f"    Assignees : {extract_assignees(TRANSCRIPT)}")
    print(f"    Estimates : {extract_estimates(TRANSCRIPT)}")
    print(f"    Dates     : {extract_dates(TRANSCRIPT)}")

    print("\n✓ Unit 1 Preprocessing demo complete!\n")
