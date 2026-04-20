"""
=============================================================
Demo — Unit 2: Neural Models (Word2Vec, LSTM, GRU, Attention, Evaluation)
=============================================================
Run: python nlp_demos/demo_unit2_neural.py
=============================================================
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.nlp.unit2_models.word_embeddings  import train_word2vec, most_similar, word_similarity, SCRUM_CORPUS
from backend.nlp.unit2_models.lstm_classifier  import (
    train as lstm_train, predict as lstm_predict,
    TRAINING_DATA as LSTM_DATA, CLASS_NAMES as LSTM_CLASSES
)
from backend.nlp.unit2_models.gru_classifier   import (
    train as gru_train, predict as gru_predict,
    TRAINING_DATA as GRU_DATA, CLASS_NAMES as GRU_CLASSES
)
from backend.nlp.unit2_models.attention        import (
    train_attention_model, predict_with_attention,
    TRAINING_DATA as ATTN_DATA
)
from backend.nlp.unit2_models.evaluator        import (
    print_classification_report, bleu_score, rouge_score
)


def section(title: str) -> None:
    print(f"\n{'─' * 65}")
    print(f"  {title}")
    print("─" * 65)


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║     UNIT 2 — NEURAL MODELS DEMO                         ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # ── Word2Vec ──────────────────────────────────────────
    section("1. WORD2VEC EMBEDDINGS")
    print("  Training Word2Vec on Scrum corpus …")
    w2v = train_word2vec(SCRUM_CORPUS)
    for word in ["sprint", "story", "authentication"]:
        sims = most_similar(w2v, word, top_k=4)
        sim_str = ", ".join(f"{w}({s:.2f})" for w, s in sims)
        print(f"  '{word}'  →  {sim_str}")

    print("\n  Pairwise similarities:")
    pairs = [("sprint","iteration"), ("epic","story"), ("backlog","sprint")]
    for a, b in pairs:
        print(f"    {a} ↔ {b}  =  {word_similarity(w2v, a, b):.4f}")

    # ── LSTM Classifier ───────────────────────────────────
    section("2. LSTM MEETING TYPE CLASSIFIER")
    print(f"  Classes: {LSTM_CLASSES}")
    print("  Training LSTM (Embedding → LSTM → Dropout → Linear) …")
    lstm_model, lstm_vocab = lstm_train(LSTM_DATA, epochs=40)

    test_sents = [
        "The CEO wants the new features shipped before the product launch",
        "Today I finished the API endpoint and will start writing tests",
        "We have sixty hours of capacity and will commit to three stories",
    ]
    print("\n  Inference on unseen sentences:")
    for s in test_sents:
        r = lstm_predict(lstm_model, lstm_vocab, s)
        print(f"    [{r['prediction']:<16} {r['confidence']:.3f}]  {s[:60]}")

    # ── GRU Classifier ────────────────────────────────────
    section("3. GRU ACTION TYPE CLASSIFIER")
    print(f"  Classes: {GRU_CLASSES}")
    print("  Training Bidirectional GRU …")
    gru_model, gru_vocab = gru_train(GRU_DATA, epochs=50)

    gru_tests = [
        "We need a ticket for the performance testing work",
        "I wrapped up the database migration this morning",
        "Bob will take the DevOps automation task",
    ]
    print("\n  Inference:")
    for s in gru_tests:
        r = gru_predict(gru_model, gru_vocab, s)
        print(f"    [{r['prediction']:<15} {r['confidence']:.3f}]  {s[:60]}")

    # ── Attention ─────────────────────────────────────────
    section("4. ATTENTION MECHANISM — Which words matter?")
    print("  Training LSTM + Bahdanau Attention …")
    attn_model, attn_vocab = train_attention_model(ATTN_DATA, epochs=60)

    text = "stakeholders approved the payment gateway budget for Q2"
    r = predict_with_attention(attn_model, attn_vocab, text)
    print(f"\n  Input     : {text}")
    print(f"  Predicted : {r['prediction']}  (conf={r['confidence']:.4f})")
    print(f"  Attention weights (token → importance):")
    sorted_attn = sorted(r["attn_weights"].items(), key=lambda x: x[1], reverse=True)
    for tok, w in sorted_attn[:6]:
        bar = "█" * int(w * 40)
        print(f"    {tok:<20} {w:.4f}  {bar}")

    # ── Evaluation Metrics ────────────────────────────────
    section("5. EVALUATION METRICS  (P / R / F1 / BLEU / ROUGE)")

    # Simulated predictions
    y_true = [0]*5 + [1]*5 + [2]*5
    y_pred = [0, 0, 1, 0, 0,  1, 1, 0, 1, 1,  2, 2, 2, 1, 2]
    print_classification_report(y_true, y_pred, LSTM_CLASSES)

    print("\n  BLEU score (Epic description generation):")
    hyp = "Build a user authentication system with JWT tokens and SSO support"
    ref = "Implement user authentication using JWT tokens and single sign-on"
    b = bleu_score(hyp, ref)
    print(f"    Hypothesis: {hyp}")
    print(f"    Reference : {ref}")
    print(f"    BLEU-4    : {b['bleu']}  |  Precisions: {b['ngram_precisions']}")

    print("\n  ROUGE score (Meeting summarization):")
    gen_summary = "Payment gateway is top priority for Q2. Authentication must address security."
    ref_summary = "The team agreed to prioritize payment gateway for Q2 alongside authentication."
    r_scores = rouge_score(gen_summary, ref_summary)
    for metric, vals in r_scores.items():
        print(f"    {metric}: {vals}")

    print("\n✓ Unit 2 Neural Models demo complete!\n")
