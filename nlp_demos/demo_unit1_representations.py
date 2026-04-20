"""
=============================================================
Demo — Unit 1: Representations (BoW, TF-IDF, N-gram LM)
=============================================================
Run: python nlp_demos/demo_unit1_representations.py
=============================================================
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.nlp.unit1_representations.bow      import build_bow_matrix, doc_similarity, find_most_similar
from backend.nlp.unit1_representations.tfidf    import build_tfidf_matrix, top_tfidf_terms, retrieve_story
from backend.nlp.unit1_representations.ngram_lm import NgramLanguageModel


EPICS = [
    "Implement user authentication with JWT tokens and SSO single sign-on",
    "Integrate Stripe payment gateway for online checkout and billing",
    "Build real-time analytics dashboard with sprint velocity burndown charts",
    "Create push notification system for email and SMS user alerts",
    "Set up CI/CD pipeline with GitHub Actions and Docker containers",
    "Design REST API for user profile management and account settings",
    "Add OAuth2 social login with Google and GitHub identity providers",
    "Build automated test suite with unit and integration test coverage",
]

CORPUS = [
    "sprint planning meeting defines the sprint goal",
    "team capacity this sprint is eighty hours total",
    "user authentication is the highest priority epic",
    "payment gateway integration is critical for Q2 launch",
    "we completed the login feature and merged to main",
    "product owner prioritizes the backlog items by value",
    "sprint retrospective helped us improve our velocity",
    "daily standup keeps the team aligned on progress",
    "story points estimate the effort for each user story",
    "scrum master facilitates the ceremonies and removes blockers",
]


def section(title: str) -> None:
    print(f"\n{'─' * 65}")
    print(f"  {title}")
    print("─" * 65)


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  UNIT 1 — REPRESENTATIONS: BoW, TF-IDF, N-gram LM       ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # ── BoW ───────────────────────────────────────────────
    section("1. BAG OF WORDS (BoW)")
    bow_matrix, bow_vect = build_bow_matrix(EPICS)
    vocab = bow_vect.get_feature_names_out()
    print(f"  Vocabulary size : {len(vocab)}")
    print(f"  Matrix shape    : {bow_matrix.shape}  (epics × vocab)")

    print("\n  Pairwise cosine similarity (BoW vectors):")
    pairs = [(0, 6, "auth ↔ OAuth"), (0, 1, "auth ↔ payment"), (0, 2, "auth ↔ dashboard")]
    for a, b, label in pairs:
        score = doc_similarity(bow_matrix, a, b)
        print(f"    E{a} ↔ E{b} [{label}]  →  {score:.4f}")

    print("\n  BoW query retrieval:")
    query = "user login with secure tokens"
    results = find_most_similar(query, EPICS, bow_vect, bow_matrix, top_k=2)
    print(f"    Query: '{query}'")
    for idx, score, doc in results:
        print(f"      [{score:.4f}]  E{idx}: {doc[:60]}")

    # ── TF-IDF ────────────────────────────────────────────
    section("2. TF-IDF RETRIEVAL")
    tfidf_matrix, tfidf_vect = build_tfidf_matrix(EPICS)
    print(f"  Vocabulary size : {len(tfidf_vect.get_feature_names_out())} (incl. bigrams)")

    print("\n  Top TF-IDF terms per Epic (most distinctive):")
    for i, epic in enumerate(EPICS):
        top = top_tfidf_terms(tfidf_matrix, tfidf_vect, i, top_k=4)
        terms = [t for t, _ in top]
        print(f"    E{i}: {epic[:40]:.<43} {terms}")

    print("\n  TF-IDF story retrieval (sprint planning):")
    story_ids = [f"EP-{i+1:03d}" for i in range(len(EPICS))]
    utterances = [
        "I'll work on login and authentication this sprint",
        "Let's pull in the payment checkout story",
        "Can someone set up the automated deploy pipeline?",
    ]
    for utt in utterances:
        matches = retrieve_story(utt, EPICS, tfidf_vect, tfidf_matrix, top_k=2)
        print(f"\n    '{utt}'")
        for idx, score, story in matches:
            print(f"      [{score:.4f}]  E{idx}: {story[:55]}")

    # ── N-gram LM ─────────────────────────────────────────
    section("3. N-GRAM LANGUAGE MODEL + PERPLEXITY")
    bigram_lm = NgramLanguageModel(n=2)
    bigram_lm.train(CORPUS)

    print("\n  Top-5 bigrams:")
    for ngram, count in bigram_lm.most_common(5):
        print(f"    {ngram}  →  {count}")

    print("\n  Perplexity test (lower = more expected by the model):")
    tests = [
        ("sprint planning review retrospective standup scrum",   "Scrum domain"),
        ("the cat sat on the mat wearing a hat",                 "Out-of-domain"),
        ("pizza delivery service open twenty four hours daily",  "Out-of-domain"),
    ]
    for sent, kind in tests:
        pp = bigram_lm.perplexity(sent)
        print(f"    [{kind}] pp={pp:.1f}  '{sent[:50]}'")

    # ── BoW vs TF-IDF comparison ──────────────────────────
    section("4. BoW vs TF-IDF COMPARISON SUMMARY")
    print(f"  {'Method':<12} {'Captures order':<18} {'Weights by rarity':<20} {'Bigrams':<10}")
    print("  " + "-" * 60)
    print(f"  {'BoW':<12} {'No':<18} {'No':<20} {'No':<10}")
    print(f"  {'TF-IDF':<12} {'No':<18} {'Yes (IDF)':<20} {'Yes':<10}")
    print(f"  {'Word2Vec':<12} {'Limited':<18} {'Yes (dense)':<20} {'N/A':<10}")
    print(f"  {'SBERT':<12} {'Yes (attn)':<18} {'Yes':<20} {'N/A':<10}")

    print("\n✓ Unit 1 Representations demo complete!\n")
