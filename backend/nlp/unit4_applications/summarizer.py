"""
=============================================================
Unit 4 — Text Summarization
=============================================================
Implements both summarization approaches:

EXTRACTIVE (sentence scoring):
  Select the most important sentences from the original text.
  Uses TF-IDF sentence scoring + cosine similarity ranking.
  Output: a subset of original sentences (verbatim).
  Fast, no model needed.

ABSTRACTIVE (BART):
  Generate a new, fluent summary paragraph.
  Uses facebook/bart-large-cnn (or distilbart for CPU speed).
  Output: newly generated text, not in the original.
  Captures the meaning, not just copying sentences.

Application in ScrumPilot:
  After a PM meeting is processed:
  → Extractive summary → meeting minutes (verbatim key points)
  → Abstractive summary → executive summary paragraph

Syllabus: Unit 4 — Summarization: Extractive and Abstractive
Run     : python backend/nlp/unit4_applications/summarizer.py
=============================================================
"""

import re
from typing import List, Tuple
import numpy as np


# ════════════════════════════════════════════════════════
# EXTRACTIVE SUMMARIZATION (TF-IDF sentence scoring)
# ════════════════════════════════════════════════════════

def _sentence_tfidf_scores(sentences: List[str]) -> np.ndarray:
    """Score sentences by TF-IDF importance relative to the full document."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    if len(sentences) < 2:
        return np.ones(len(sentences))

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_mat  = vectorizer.fit_transform(sentences)

    # Score = average cosine similarity of each sentence to all others
    # (TextRank-like: important sentences are similar to many others)
    sim_matrix = cosine_similarity(tfidf_mat, tfidf_mat)
    scores = sim_matrix.sum(axis=1) - 1.0  # subtract self-similarity (= 1)
    return scores


def extractive_summarize(text: str, n_sentences: int = 3) -> str:
    """
    Select the top-n most important sentences from the text.

    Algorithm:
      1. Split into sentences
      2. Compute TF-IDF vectors for each sentence
      3. Rank by average cosine similarity to all other sentences
      4. Return top-n sentences in original order

    Args:
        text        : Input document / meeting transcript
        n_sentences : Number of sentences to extract

    Returns:
        Extracted summary (joined sentences)
    """
    import nltk
    nltk.download("punkt",     quiet=True)
    nltk.download("punkt_tab", quiet=True)
    from nltk.tokenize import sent_tokenize

    sentences = sent_tokenize(text)
    if len(sentences) <= n_sentences:
        return text

    scores   = _sentence_tfidf_scores(sentences)
    top_idx  = np.argsort(scores)[::-1][:n_sentences]
    top_idx  = sorted(top_idx)  # restore original order

    return " ".join(sentences[i] for i in top_idx)


# ════════════════════════════════════════════════════════
# ABSTRACTIVE SUMMARIZATION (BART)
# ════════════════════════════════════════════════════════

BART_MODEL = "sshleifer/distilbart-cnn-12-6"   # CPU-friendly DistilBART


def _load_bart():
    """
    Load DistilBART using the model class directly.
    (transformers v5 removed 'summarization' from pipeline registry)
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    print(f"  Loading {BART_MODEL} …")
    tokenizer = AutoTokenizer.from_pretrained(BART_MODEL)
    model     = AutoModelForSeq2SeqLM.from_pretrained(BART_MODEL)
    model.eval()
    return tokenizer, model


def abstractive_summarize(
    text: str,
    summarizer,
    max_length: int = 80,
    min_length: int = 20,
) -> str:
    """
    Generate a fluent, abstractive summary using DistilBART.

    BART architecture:
      Encoder: reads the full input text (like BERT)
      Decoder: generates the summary token by token (like GPT)
      Trained on CNN/DailyMail news summarization task.

    Args:
        text       : Input document
        summarizer : (tokenizer, model) tuple returned by _load_bart()
        max_length : Maximum summary token length
        min_length : Minimum summary token length

    Returns:
        Generated summary string
    """
    import torch
    tokenizer, model = summarizer

    # BART has a max input of 1024 tokens — truncate if needed
    if len(text.split()) > 900:
        text = " ".join(text.split()[:900])

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    )
    with torch.no_grad():
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=max_length,
            min_length=min_length,
            num_beams=4,
            early_stopping=True,
        )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


# ── Standalone Demo ───────────────────────────────────────

if __name__ == "__main__":
    import sys, os as _os
    # Allow running directly as: python backend/nlp/unit4_applications/summarizer.py
    sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))))

    MEETING_TRANSCRIPT = """
    Sarah: Good morning everyone. Let's start today's PM meeting.
    The main topic is the Q2 product roadmap and our Epic priorities.

    Product Manager: Thanks Sarah. Based on our stakeholder discussions,
    the payment gateway integration is our highest priority for Q2.
    The business value is rated at nine out of ten because it directly impacts revenue.
    Without this feature, we cannot process customer payments online.

    Tech Lead: I agree. The authentication system should also be prioritized.
    We currently have security vulnerabilities that need to be addressed.
    The business value is eight and the time criticality is nine — we risk
    losing enterprise customers if this is not fixed before the audit.

    Product Manager: Exactly. The analytics dashboard is important but can wait
    for Q3. It has a business value of six and is more of a nice-to-have for now.

    Sarah: What about the notification system?

    Tech Lead: That one is medium priority. We estimated five story points and
    the business value is around seven. It improves user retention but is not critical.

    Product Manager: Let's finalize: payment gateway first, then authentication,
    then notifications. The dashboard goes to Q3. Are we aligned?

    Team: Yes, agreed.

    Sarah: Perfect. I'll update the backlog and create the epics in Jira.
    Next meeting is scheduled for Monday at 10 AM. Thanks everyone!
    """

    print("=" * 65)
    print("       UNIT 4 — TEXT SUMMARIZATION DEMO")
    print("=" * 65)
    print(f"\nOriginal transcript: {len(MEETING_TRANSCRIPT.split())} words")

    # ── Extractive summarization ──────────────────────────
    print("\n[1] EXTRACTIVE SUMMARIZATION  (TF-IDF sentence scoring)")
    print("-" * 55)
    print("    Approach: Select top-3 most important sentences verbatim\n")
    ext_summary = extractive_summarize(MEETING_TRANSCRIPT, n_sentences=3)
    print(f"  Summary ({len(ext_summary.split())} words):")
    print(f"  {ext_summary}")

    print("\n  Approach notes:")
    print("  + Fast, no model needed")
    print("  + Factually accurate (verbatim sentences)")
    print("  − Choppy, may miss context across sentence boundaries")

    # ── Abstractive summarization ─────────────────────────
    print("\n[2] ABSTRACTIVE SUMMARIZATION  (BART / DistilBART)")
    print("-" * 55)
    print("    Approach: Generate a new fluent summary paragraph")
    print("    Model   : sshleifer/distilbart-cnn-12-6 (CPU-friendly)\n")

    try:
        bart = _load_bart()
        abs_summary = abstractive_summarize(MEETING_TRANSCRIPT.strip(), bart)
        print(f"  Summary ({len(abs_summary.split())} words):")
        print(f"  {abs_summary}")

        print("\n  Approach notes:")
        print("  + Fluent, reads like a human-written summary")
        print("  + Can combine information from multiple sentences")
        print("  − Slower (requires neural model), can hallucinate")
    except Exception as e:
        print(f"  BART not available: {e}")
        print("  Install: pip install transformers")

    # ── ROUGE comparison ──────────────────────────────────
    print("\n[3] ROUGE SCORE COMPARISON")
    print("-" * 55)
    reference = (
        "The payment gateway integration is the highest priority for Q2 with business value nine. "
        "Authentication security vulnerabilities must also be addressed urgently. "
        "The analytics dashboard is deferred to Q3."
    )
    try:
        from backend.nlp.unit2_models.evaluator import rouge_score
        ext_rouge = rouge_score(ext_summary, reference)
        print(f"  Extractive ROUGE-1 F1 : {ext_rouge['ROUGE-1']['f1']}")
        print(f"  Extractive ROUGE-2 F1 : {ext_rouge['ROUGE-2']['f1']}")
        print(f"  Extractive ROUGE-L F1 : {ext_rouge['ROUGE-L']['f1']}")
    except ImportError:
        print("  (Run from project root to enable ROUGE evaluation)")
    print()
