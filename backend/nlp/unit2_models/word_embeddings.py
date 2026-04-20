"""
=============================================================
Unit 2 — Word Embeddings (Word2Vec)
=============================================================
Trains a Word2Vec model on a Scrum meeting corpus and demonstrates:
  • Distributed word representations (dense vectors, low-dimension)
  • Semantic similarity: sprint ↔ iteration, epic ↔ feature
  • Vector arithmetic: king - man + woman ≈ queen (analogy)
  • Comparison with BoW: BoW is sparse, Word2Vec is dense & semantic

Architecture:
  Word2Vec Skip-gram: given a word, predict surrounding context words
  Each word → learned dense vector (embedding_dim=100)

Syllabus: Unit 2 — Distributed Word Representations
Run     : python backend/nlp/unit2_models/word_embeddings.py
=============================================================
"""

import os
import re
from typing import List, Tuple

import nltk
nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
from nltk.tokenize import word_tokenize


# ── Training corpus: Scrum meeting sentences ──────────────

SCRUM_CORPUS = [
    "the sprint planning meeting defines the sprint goal and team commitment",
    "the product owner prioritizes items in the product backlog",
    "user stories describe features from the user perspective",
    "epics are large features that span multiple sprints",
    "the scrum master facilitates the daily standup meeting",
    "story points estimate the effort required for a user story",
    "the sprint review demonstrates completed work to stakeholders",
    "the retrospective helps the team improve their process",
    "the backlog grooming session refines and estimates stories",
    "velocity is the average story points completed per sprint",
    "acceptance criteria define when a user story is done",
    "the definition of done ensures consistent quality standards",
    "developers work on tasks within a sprint to complete stories",
    "the kanban board visualizes the workflow and progress",
    "authentication and authorization protect user data and access",
    "the payment gateway integrates with Stripe for checkout",
    "the REST API exposes endpoints for the frontend to consume",
    "continuous integration ensures code is tested on every commit",
    "docker containers package the application for deployment",
    "the database migration updates the schema for new features",
    "test driven development improves code quality and coverage",
    "the product backlog contains all features requirements and bugs",
    "sprint ceremonies include planning review retrospective and standup",
    "the team velocity helps forecast future sprint capacity",
    "burndown chart tracks remaining work against time in a sprint",
    "tech debt accumulates when shortcuts are taken during development",
    "refactoring improves code quality without changing functionality",
    "microservices split the application into independently deployable units",
    "the API gateway routes requests to the appropriate microservice",
    "feature flags allow gradual rollout of new functionality",
    "user onboarding helps new users get started with the product",
    "analytics dashboard shows key performance indicators and metrics",
    "push notifications alert users to important events in real time",
    "mobile responsive design ensures the app works on all devices",
    "security audit identifies vulnerabilities in the codebase",
    "performance testing ensures the application handles peak load",
    "code review improves quality and spreads knowledge in the team",
    "the sprint goal focuses the team on a single business objective",
    "estimation poker uses fibonacci numbers for story point voting",
    "the scrum board shows todo in progress done columns",
]


# ── Train Word2Vec ────────────────────────────────────────

def train_word2vec(corpus: List[str], embedding_dim: int = 100, window: int = 5):
    """
    Train a Word2Vec Skip-gram model on a list of sentences.

    Args:
        corpus       : List of sentences (strings)
        embedding_dim: Dimensionality of word vectors
        window       : Context window size

    Returns:
        Trained gensim Word2Vec model
    """
    from gensim.models import Word2Vec

    # Tokenize each sentence
    tokenized = [word_tokenize(sent.lower()) for sent in corpus]

    model = Word2Vec(
        sentences=tokenized,
        vector_size=embedding_dim,
        window=window,
        min_count=1,          # Include all words (small corpus)
        workers=1,            # CPU single-threaded
        sg=1,                 # Skip-gram (sg=1) vs CBOW (sg=0)
        epochs=200,           # More epochs for small corpus
        seed=42,
    )
    return model


def most_similar(model, word: str, top_k: int = 5) -> List[Tuple[str, float]]:
    """Return top-k most similar words by cosine similarity in embedding space."""
    try:
        return model.wv.most_similar(word, topn=top_k)
    except KeyError:
        return []


def word_similarity(model, word_a: str, word_b: str) -> float:
    """Cosine similarity between two word vectors."""
    try:
        return round(float(model.wv.similarity(word_a, word_b)), 4)
    except KeyError:
        return 0.0


def vector_arithmetic(model, pos: List[str], neg: List[str], top_k: int = 3):
    """
    Word vector arithmetic.
    Example: pos=["king","woman"], neg=["man"] → finds word closest to king-man+woman
    """
    try:
        return model.wv.most_similar(positive=pos, negative=neg, topn=top_k)
    except KeyError:
        return []


# ── Standalone Demo ───────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("       UNIT 2 — WORD EMBEDDINGS (Word2Vec) DEMO")
    print("=" * 65)

    print("\nTraining Word2Vec on Scrum meeting corpus …")
    model = train_word2vec(SCRUM_CORPUS, embedding_dim=100)
    vocab_size = len(model.wv)
    print(f"  Vocabulary  : {vocab_size} words")
    print(f"  Vector size : {model.vector_size} dimensions")
    print(f"  Architecture: Skip-gram")

    # ── Most similar words ────────────────────────────────
    print("\n[1] MOST SIMILAR WORDS (semantic neighbours)")
    print("-" * 45)
    query_words = ["sprint", "epic", "story", "backlog", "authentication"]
    for word in query_words:
        sims = most_similar(model, word, top_k=5)
        if sims:
            sim_str = ", ".join(f"{w}({s:.2f})" for w, s in sims)
            print(f"  '{word}'  →  {sim_str}")
        else:
            print(f"  '{word}'  →  not in vocabulary")

    # ── Pairwise similarities ─────────────────────────────
    print("\n[2] PAIRWISE WORD SIMILARITY (cosine)")
    print("-" * 45)
    pairs = [
        ("sprint",         "iteration",    "expected: HIGH (same concept)"),
        ("epic",           "story",        "expected: HIGH (both backlog items)"),
        ("backlog",        "sprint",       "expected: MED  (related concepts)"),
        ("authentication", "payment",      "expected: MED  (both features)"),
        ("retrospective",  "authentication","expected: LOW (unrelated)"),
    ]
    print(f"  {'Word A':<20} {'Word B':<20} {'Cosine':>7}  Note")
    print("  " + "-" * 65)
    for a, b, note in pairs:
        sim = word_similarity(model, a, b)
        print(f"  {a:<20} {b:<20} {sim:>7.4f}  {note}")

    # ── Vector arithmetic ─────────────────────────────────
    print("\n[3] VECTOR ARITHMETIC")
    print("-" * 45)
    print("  sprint - planning + review = ?")
    result = vector_arithmetic(model, ["sprint", "review"], ["planning"])
    for word, score in result:
        print(f"    → '{word}'  ({score:.4f})")

    print("\n  backlog - story + epic = ?")
    result2 = vector_arithmetic(model, ["backlog", "epic"], ["story"])
    for word, score in result2:
        print(f"    → '{word}'  ({score:.4f})")

    # ── BoW vs Word2Vec comparison ────────────────────────
    print("\n[4] BoW vs WORD2VEC COMPARISON")
    print("-" * 45)
    print("  'authentication' vs 'login':")
    sim = word_similarity(model, "authentication", "login")
    print(f"    Word2Vec cosine similarity : {sim}")
    print(f"    BoW cosine similarity      : 0.0  (no shared vocabulary)")
    print("  → Word2Vec captures semantic meaning; BoW does not")

    print("\n  BoW vector: {sprint: 1, planning: 1, ...}  → SPARSE ({} mostly zeros)")
    vec_example = model.wv["sprint"]
    print(f"  Word2Vec vector (first 10 of {model.vector_size} dims): {[round(x,3) for x in vec_example[:10]]}")
    print("  → Word2Vec: DENSE (every dimension has meaningful value)")
    print()
