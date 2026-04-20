"""
=============================================================
Unit 1 — Bag of Words (BoW)
=============================================================
Represents each document as a vector of word counts.
  • CountVectorizer builds the vocabulary
  • Each document → a sparse vector of length |vocab|
  • Cosine similarity measures document overlap

Limitations (important for exam/viva):
  • No word order: "dog bites man" = "man bites dog"
  • No semantics:  "login" and "authentication" are unrelated
  • High-dimensional sparse matrix (wasteful memory)

Syllabus: Unit 1 — Traditional Representations: BoW and limitations
Run     : python backend/nlp/unit1_representations/bow.py
=============================================================
"""

import numpy as np
from typing import List, Tuple
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ── BoW functions ─────────────────────────────────────────

def build_bow_matrix(documents: List[str]) -> Tuple[np.ndarray, CountVectorizer]:
    """
    Fit a CountVectorizer and transform documents into a BoW matrix.

    Returns:
        (matrix, vectorizer)
        matrix shape: (n_docs, vocab_size) — sparse, converted to dense on demand
    """
    vectorizer = CountVectorizer(
        stop_words="english",   # Remove common English stopwords
        max_features=500,       # Keep top-500 most frequent terms
        ngram_range=(1, 1),     # Unigrams only (pure BoW)
        min_df=1,               # Minimum document frequency
    )
    matrix = vectorizer.fit_transform(documents)
    return matrix, vectorizer


def doc_similarity(matrix, idx_a: int, idx_b: int) -> float:
    """Cosine similarity between documents at index idx_a and idx_b."""
    return float(cosine_similarity(matrix[idx_a], matrix[idx_b])[0][0])


def find_most_similar(
    query: str,
    documents: List[str],
    vectorizer: CountVectorizer,
    matrix,
    top_k: int = 3,
) -> List[Tuple[int, float, str]]:
    """
    Find the top-k documents most similar to a query string using BoW.

    Args:
        query     : Natural language query
        documents : Original document list
        vectorizer: Fitted CountVectorizer
        matrix    : Precomputed BoW matrix

    Returns:
        List of (doc_index, similarity_score, document_text)
    """
    query_vec = vectorizer.transform([query])
    sims = cosine_similarity(query_vec, matrix)[0]
    ranked = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)
    return [(idx, round(score, 4), documents[idx]) for idx, score in ranked[:top_k]]


def get_top_terms(
    matrix,
    vectorizer: CountVectorizer,
    doc_idx: int,
    top_k: int = 8,
) -> List[Tuple[str, int]]:
    """Return the top-k most frequent terms in a specific document."""
    feature_names = vectorizer.get_feature_names_out()
    counts = matrix[doc_idx].toarray()[0]
    top_idx = np.argsort(counts)[::-1][:top_k]
    return [(feature_names[i], int(counts[i])) for i in top_idx if counts[i] > 0]


# ── Standalone Demo ───────────────────────────────────────

if __name__ == "__main__":
    # Scrum backlog Epic descriptions
    EPICS = [
        "Build a user authentication system with SSO and JWT tokens for secure login",
        "Develop a payment gateway integration with Stripe for online checkout",
        "Create a real-time analytics dashboard showing sprint velocity and burndown",
        "Implement user login and registration with OAuth2 and social login support",  # Similar to #0
        "Set up CI/CD pipeline with GitHub Actions and automated Docker deployment",
        "Design a push notification system for email and SMS alert delivery",
        "Add multi-language internationalization and localization to the web app",
        "Build a REST API for user profile management and account settings",
    ]

    print("=" * 65)
    print("         UNIT 1 — BAG OF WORDS (BoW) DEMO")
    print("=" * 65)

    matrix, vectorizer = build_bow_matrix(EPICS)
    vocab = vectorizer.get_feature_names_out()
    dense = matrix.toarray()

    print(f"\nVocabulary size  : {len(vocab)} unique terms")
    print(f"Matrix shape     : {matrix.shape}  (docs × vocab)")
    print(f"Sparsity         : {(dense == 0).sum() / dense.size * 100:.1f}% zeros")

    # ── Top terms per Epic ────────────────────────────────
    print("\nTop terms per Epic:")
    for i, epic in enumerate(EPICS):
        top = get_top_terms(matrix, vectorizer, i, top_k=5)
        print(f"  E{i}: {epic[:45]:.<48} {[t for t, c in top]}")

    # ── Pairwise similarity ───────────────────────────────
    print("\nPairwise Cosine Similarity (BoW vectors):")
    print(f"  {'Pair':<10} {'Score':>7}  {'Expected'}")
    print("  " + "-" * 45)
    pairs = [
        (0, 3, "HIGH  — both about authentication/login"),
        (0, 1, "LOW   — auth vs. payment"),
        (0, 4, "LOW   — auth vs. CI/CD"),
        (1, 5, "LOW   — payment vs. notifications"),
        (2, 7, "MED   — dashboard vs. profile API"),
    ]
    for a, b, note in pairs:
        score = doc_similarity(matrix, a, b)
        print(f"  E{a} ↔ E{b}    {score:>7.4f}  {note}")

    # ── Query-based retrieval ─────────────────────────────
    print("\nBoW Story Retrieval (spoken phrase → matching Epic):")
    queries = [
        "user login and signup with token authentication",
        "deploy the app automatically using Docker",
    ]
    for q in queries:
        results = find_most_similar(q, EPICS, vectorizer, matrix, top_k=2)
        print(f"\n  Query: '{q}'")
        for rank, (idx, score, doc) in enumerate(results, 1):
            print(f"    {rank}. [score={score:.4f}]  E{idx}: {doc[:60]}")

    print("\n⚠  BoW LIMITATIONS:")
    print("  • No word order: 'sprint planning' = 'planning sprint'")
    print("  • No semantics: 'authentication' and 'login' score LOW similarity")
    print("  → TF-IDF (Unit 1) partially fixes weighting; Sentence-BERT (Unit 3) fixes semantics")
    print()
