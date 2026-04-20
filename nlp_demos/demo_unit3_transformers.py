"""
=============================================================
Demo — Unit 3: Transformers (BERT, Sentence-BERT, CNN)
=============================================================
Run: python nlp_demos/demo_unit3_transformers.py
=============================================================
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


STORIES   = [
    "Implement user authentication and secure login with JWT tokens",
    "Integrate Stripe payment gateway for online checkout processing",
    "Build real-time sprint analytics dashboard with burndown charts",
    "Set up CI/CD pipeline with GitHub Actions and Docker containers",
    "Create push notification system for email and SMS delivery",
    "Design REST API for user profile management and settings",
    "Add OAuth2 social login with Google and GitHub providers",
    "Build automated test suite with unit and integration tests",
]
STORY_IDS = [f"SP-{i+1:03d}" for i in range(len(STORIES))]

TRAINING_DATA_CNN = [
    ("The stakeholders want a payment gateway integrated by Q2", 0),
    ("Business value for the authentication epic is rated at nine", 0),
    ("Product owner reviewed the requirements with the client", 0),
    ("Management approved the budget for new mobile application", 0),
    ("Market research shows users need better notification features", 0),
    ("Our sprint goal is to complete the payment gateway integration", 1),
    ("Team capacity this sprint is eighty hours across five developers", 1),
    ("We are pulling in four stories totaling thirty story points", 1),
    ("Sprint starts Monday and ends Friday in two weeks", 1),
    ("Previous sprint velocity was thirty eight let us target that", 1),
    ("Yesterday I completed the login API and pushed to main", 2),
    ("Today I will work on the payment form component and tests", 2),
    ("I am blocked on the database migration waiting for approval", 2),
    ("No blockers from my side I will continue with the API work", 2),
    ("Yesterday I reviewed pull requests and helped Mike debug", 2),
]
CNN_CLASS_NAMES = ["PM_MEETING", "SPRINT_PLANNING", "STANDUP"]


def section(title: str) -> None:
    print(f"\n{'─' * 65}")
    print(f"  {title}")
    print("─" * 65)


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║     UNIT 3 — TRANSFORMERS DEMO                          ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # ── BERT Contextual Embeddings ────────────────────────
    section("1. BERT CONTEXTUAL EMBEDDINGS  (DistilBERT)")
    from backend.nlp.unit3_transformers.bert_embeddings import (
        _load_model, get_sentence_embedding, cosine_similarity, find_most_similar_bert
    )
    tokenizer, bert = _load_model()

    print("\n  Contextual similarity (same word, different context):")
    finance_bank = get_sentence_embedding("we need a bank account for the budget", tokenizer, bert)
    river_bank   = get_sentence_embedding("the team sat by the river bank", tokenizer, bert)
    tech_bank    = get_sentence_embedding("the sprint backlog is stored in the database", tokenizer, bert)
    print(f"    finance ↔ river  : {cosine_similarity(finance_bank, river_bank):.4f}")
    print(f"    finance ↔ tech   : {cosine_similarity(finance_bank, tech_bank):.4f}")

    print("\n  BERT sentence similarity:")
    pairs = [
        ("User authentication with JWT", "Login system with token validation"),
        ("Sprint planning meeting",      "User authentication with JWT"),
    ]
    for a, b in pairs:
        e_a = get_sentence_embedding(a, tokenizer, bert)
        e_b = get_sentence_embedding(b, tokenizer, bert)
        sim = cosine_similarity(e_a, e_b)
        print(f"    {sim:.4f}  '{a[:40]}' ↔ '{b[:40]}'")

    print("\n  BERT story retrieval (sprint planning):")
    query = "I'll work on the login feature this sprint"
    matches = find_most_similar_bert(query, STORIES, tokenizer, bert, top_k=3)
    print(f"    Query: '{query}'")
    for idx, score, doc in matches:
        print(f"      [{score:.4f}]  {STORY_IDS[idx]}: {doc[:55]}")

    # ── Sentence-BERT ─────────────────────────────────────
    section("2. SENTENCE-BERT  (all-MiniLM-L6-v2)")
    from backend.nlp.unit3_transformers.sentence_bert import (
        _load_sbert, find_matching_story, semantic_similarity
    )
    sbert = _load_sbert()

    utterances = [
        "I'll handle the login work",
        "Payment checkout needs to be done",
        "Let's pull in the CI deployment task",
    ]
    print("\n  Sprint planning — utterance → story mapping:")
    for utt in utterances:
        matches = find_matching_story(utt, STORIES, STORY_IDS, sbert, top_k=2, threshold=0.0)
        print(f"\n    '{utt}'")
        for m in matches[:2]:
            print(f"      [{m['score']:.4f}]  {m['story_id']}: {m['title'][:55]}")

    print("\n  SBERT vs TF-IDF for 'user login and signup':")
    q = "user login and signup with secure access"
    sbert_match = find_matching_story(q, STORIES, STORY_IDS, sbert, top_k=1, threshold=0.0)
    print(f"    SBERT    → {sbert_match[0]['story_id']} [{sbert_match[0]['score']:.4f}]  {sbert_match[0]['title']}")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as sk_cos
    import numpy as np
    tv = TfidfVectorizer(stop_words="english"); tfm = tv.fit_transform(STORIES)
    tfidf_sims = sk_cos(tv.transform([q]), tfm)[0]
    best = int(np.argmax(tfidf_sims))
    print(f"    TF-IDF   → {STORY_IDS[best]} [{tfidf_sims[best]:.4f}]  {STORIES[best]}")

    # ── CNN Classifier ────────────────────────────────────
    section("3. CNN TEXT CLASSIFIER  (kernel sizes 3,4,5)")
    from backend.nlp.unit3_transformers.cnn_text_classifier import (
        train as cnn_train, predict as cnn_predict
    )
    print(f"  Classes: {CNN_CLASS_NAMES}")
    print("  Architecture: Embedding → Conv1D[3,4,5] → MaxPool → FC")
    print("  Training CNN …")
    cnn_model, cnn_vocab = cnn_train(TRAINING_DATA_CNN, epochs=50)

    cnn_tests = [
        "investors expect the MVP before the funding round",
        "yesterday I deployed the feature to staging and fixed bugs",
        "we commit to three stories within our eighty hour capacity",
    ]
    print("\n  Inference:")
    for s in cnn_tests:
        r = cnn_predict(cnn_model, cnn_vocab, s)
        print(f"    [{r['prediction']:<16} {r['confidence']:.3f}]  {s[:60]}")

    # ── Architecture comparison ───────────────────────────
    section("4. MODEL COMPARISON SUMMARY")
    print(f"  {'Model':<12} {'Task':<30} {'Speed (CPU)':<12} {'Accuracy'}")
    print("  " + "-" * 65)
    print(f"  {'BoW':<12} {'Retrieval':<30} {'Very fast':<12} {'Low'}")
    print(f"  {'TF-IDF':<12} {'Retrieval':<30} {'Very fast':<12} {'Medium'}")
    print(f"  {'LSTM':<12} {'Classification':<30} {'Fast':<12} {'Medium-High'}")
    print(f"  {'GRU':<12} {'Classification':<30} {'Fast':<12} {'Medium-High'}")
    print(f"  {'CNN':<12} {'Classification':<30} {'Fastest':<12} {'Medium-High'}")
    print(f"  {'BERT':<12} {'Embeddings/QA':<30} {'Slow':<12} {'High'}")
    print(f"  {'Sentence-BERT':<12} {'Semantic search':<30} {'Medium':<12} {'Highest'}")

    print("\n✓ Unit 3 Transformers demo complete!\n")
