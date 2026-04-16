"""
=============================================================
Unit 1 — TF-IDF Representation
=============================================================
Improves on BoW by weighting terms by importance:
  TF   = Term Frequency  — how often a word appears in a doc
  IDF  = Inverse Document Frequency — how rare it is globally
  Score = TF × IDF  (high = important AND distinctive)

Advantages over pure BoW:
  + Common words (sprint, team) get downweighted by IDF
  + Rare-but-important words (Stripe, JWT) score higher
  + Bigrams capture multi-word terms (payment gateway, sprint planning)

Application: Story retrieval for sprint planning
  Speaker says: "I'll handle the login feature"
  → TF-IDF cosine similarity finds the matching User Story

Syllabus: Unit 1 — Traditional Representations: BoW, TF-IDF and limitations
Run     : python backend/nlp/unit1_representations/tfidf.py
=============================================================
"""

import numpy as np
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ── TF-IDF functions ──────────────────────────────────────

def build_tfidf_matrix(documents: List[str]) -> Tuple[np.ndarray, TfidfVectorizer]:
    """
    Fit a TF-IDF vectorizer and transform documents into a TF-IDF matrix.

    Args:
        documents : List of User Story / Epic title+description strings

    Returns:
        (matrix, vectorizer)
        matrix shape: (n_docs, vocab_size)
    """
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=1000,
        ngram_range=(1, 2),   # Unigrams + bigrams (captures: "payment gateway")
        sublinear_tf=True,    # Use log(1 + TF) — dampens very high frequency terms
        min_df=1,
    )
    matrix = vectorizer.fit_transform(documents)
    return matrix, vectorizer


def top_tfidf_terms(
    matrix,
    vectorizer: TfidfVectorizer,
    doc_idx: int,
    top_k: int = 8,
) -> List[Tuple[str, float]]:
    """Get the most distinctive terms for a specific document."""
    feature_names = vectorizer.get_feature_names_out()
    scores = matrix[doc_idx].toarray()[0]
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [(feature_names[i], round(float(scores[i]), 4)) for i in top_idx if scores[i] > 0]


def retrieve_story(
    utterance: str,
    stories: List[str],
    vectorizer: TfidfVectorizer,
    matrix,
    top_k: int = 3,
) -> List[Tuple[int, float, str]]:
    """
    Given a natural language utterance from a meeting, find the best matching User Story.

    This is the NLP replacement for LLM-based story mapping:
      OLD: Ask Groq/LLM → 'which story does this phrase refer to?'
      NEW: TF-IDF cosine similarity — pure NLP, runs offline

    Args:
        utterance : Natural language phrase from the meeting
        stories   : List of User Story titles/descriptions
        vectorizer: Fitted TfidfVectorizer (call build_tfidf_matrix first)
        matrix    : Precomputed TF-IDF matrix

    Returns:
        List of (story_index, similarity_score, story_text)
    """
    query_vec = vectorizer.transform([utterance])
    sims = cosine_similarity(query_vec, matrix)[0]
    ranked = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)
    return [(idx, round(float(score), 4), stories[idx]) for idx, score in ranked[:top_k]]


def compare_bow_vs_tfidf(documents: List[str], query: str) -> None:
    """Side-by-side comparison of BoW vs TF-IDF retrieval."""
    from sklearn.feature_extraction.text import CountVectorizer

    # BoW retrieval
    bow_vec = CountVectorizer(stop_words="english")
    bow_matrix = bow_vec.fit_transform(documents)
    bow_query = bow_vec.transform([query])
    bow_sims = cosine_similarity(bow_query, bow_matrix)[0]

    # TF-IDF retrieval
    tfidf_vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    tfidf_matrix = tfidf_vec.fit_transform(documents)
    tfidf_query = tfidf_vec.transform([query])
    tfidf_sims = cosine_similarity(tfidf_query, tfidf_matrix)[0]

    print(f"\n  Query: '{query}'")
    print(f"  {'Doc':<5} {'BoW Score':>10} {'TF-IDF Score':>13}  Text (first 55 chars)")
    print("  " + "-" * 75)
    for i, doc in enumerate(documents):
        bow_s = round(float(bow_sims[i]), 4)
        tfidf_s = round(float(tfidf_sims[i]), 4)
        marker = "  ←best" if tfidf_s == max(tfidf_sims) else ""
        print(f"  D{i:<4} {bow_s:>10.4f} {tfidf_s:>13.4f}  {doc[:55]}{marker}")


# ── Standalone Demo ───────────────────────────────────────

if __name__ == "__main__":
    USER_STORIES = [
        "Implement JWT-based user authentication and secure login flow",
        "Integrate Stripe payment gateway for online checkout processing",
        "Build real-time sprint analytics dashboard with burndown charts",
        "Create push notification system for email and SMS alerts",
        "Set up CI/CD pipeline with GitHub Actions and Docker containers",
        "Design REST API for user profile management and account settings",
        "Add multi-language support and internationalization to the app",
        "Implement OAuth2 social login with Google and GitHub providers",
        "Build automated test suite with unit and integration tests",
        "Create admin dashboard for managing users and permissions",
    ]

    print("=" * 65)
    print("            UNIT 1 — TF-IDF DEMO")
    print("=" * 65)

    matrix, vectorizer = build_tfidf_matrix(USER_STORIES)
    vocab = vectorizer.get_feature_names_out()

    print(f"\nVocabulary size  : {len(vocab)} terms (including bigrams)")
    print(f"Matrix shape     : {matrix.shape}  (stories × terms)")

    # ── Top TF-IDF terms per story ────────────────────────
    print("\nTop TF-IDF terms per User Story (most distinctive words):")
    for i, story in enumerate(USER_STORIES):
        top = top_tfidf_terms(matrix, vectorizer, i, top_k=4)
        terms_str = str([t for t, _ in top])
        print(f"  S{i}: {story[:40]:.<43} {terms_str}")

    # ── Sprint planning: natural language → story ID ──────
    print("\n" + "=" * 65)
    print("  SPRINT PLANNING — Natural Language → Story Mapping")
    print("  (NLP alternative to LLM API calls)")
    print("=" * 65)

    utterances = [
        "I'll work on the login feature this sprint",
        "Let's pull in the payment work, it's high priority",
        "I can handle the authentication for users",
        "We should set up the automated CI pipeline",
        "Sarah will take the notification alerts story",
    ]

    for utt in utterances:
        results = retrieve_story(utt, USER_STORIES, vectorizer, matrix, top_k=2)
        print(f"\n  Speaker says: '{utt}'")
        for rank, (idx, score, story) in enumerate(results, 1):
            print(f"    Match {rank} [score={score:.4f}]  S{idx}: {story}")

    # ── BoW vs TF-IDF comparison ──────────────────────────
    print("\n" + "=" * 65)
    print("  BoW vs TF-IDF COMPARISON")
    print("=" * 65)
    compare_bow_vs_tfidf(
        USER_STORIES,
        "authenticate users with secure tokens"
    )

    print("\n⚠  TF-IDF LIMITATIONS:")
    print("  + Better than BoW: IDF downweights 'sprint', 'user', 'build'")
    print("  + Bigrams capture 'payment gateway', 'user authentication'")
    print("  − Still no semantics: 'auth' ≠ 'login' unless they share words")
    print("  → Sentence-BERT (Unit 3) overcomes this with contextual embeddings")
    print()
