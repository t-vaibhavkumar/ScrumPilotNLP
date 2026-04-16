"""
=============================================================
Unit 3 — BERT-based Text Classifier (Feature Extraction)
=============================================================
Classifies Epic priority (HIGH / MEDIUM / LOW) using:
  1. DistilBERT extracts CLS embeddings (feature extraction)
  2. Sklearn LogisticRegression trains on those features

This is the CPU-friendly approach to "fine-tuning BERT":
  Full fine-tuning: update all BERT weights via gradient descent (slow on CPU)
  Feature extraction: freeze BERT, train only a small classifier on top (FAST)

Both approaches use transfer learning — BERT's pre-trained knowledge
is leveraged for a new task without training from scratch.

Usage in ScrumPilot:
  Input:  Epic description text
  Output: Priority class (HIGH / MEDIUM / LOW) + confidence

Syllabus: Unit 3 — Fine-tuning BERT for Text Classification
          Unit 3 — Transfer learning and fine-tuning strategies
Run     : python backend/nlp/unit3_transformers/bert_classifier.py
=============================================================
"""

import torch
import numpy as np
from typing import List, Tuple, Dict


# ── Labels ────────────────────────────────────────────────
CLASS_NAMES = ["LOW", "MEDIUM", "HIGH"]
LABEL2IDX   = {name: i for i, name in enumerate(CLASS_NAMES)}


# ── Synthetic Training Data ───────────────────────────────
# Derived from WSJF scores: WSJF>=5 → HIGH, 3–5 → MEDIUM, <3 → LOW

TRAINING_DATA: List[Tuple[str, str]] = [
    # HIGH priority epics (critical, revenue-impacting, must-have)
    ("Implement secure user authentication with SSO and MFA to prevent unauthorized access", "HIGH"),
    ("Integrate Stripe payment gateway for Q2 launch — critical for revenue generation",     "HIGH"),
    ("Build real-time fraud detection system to protect customer financial transactions",     "HIGH"),
    ("Deliver core mobile app before investor demo — strategic priority for funding round",   "HIGH"),
    ("GDPR compliance implementation required by law before EU market launch in June",       "HIGH"),
    ("Complete checkout flow optimization to reduce cart abandonment and increase revenue",   "HIGH"),
    ("Emergency security patch for SQL injection vulnerability in production database",       "HIGH"),
    ("Launch MVP before Q3 — CEO and board have committed to this delivery date",            "HIGH"),
    ("Payment processing downtime fix — losing thousands of dollars per hour in sales",      "HIGH"),
    ("User authentication is broken in production affecting all existing customers today",    "HIGH"),

    # MEDIUM priority epics (important, valuable, planned)
    ("Build analytics dashboard to track sprint velocity and team performance metrics",      "MEDIUM"),
    ("Add email notification system for task assignment and status update alerts",           "MEDIUM"),
    ("Implement user profile management with settings and preference customization",         "MEDIUM"),
    ("Create admin panel for customer support team to manage accounts and tickets",          "MEDIUM"),
    ("Improve search functionality with filters and sorting for better user experience",     "MEDIUM"),
    ("Add multi-language support to prepare for international expansion next quarter",       "MEDIUM"),
    ("Build reporting module with export to PDF and CSV for business intelligence",          "MEDIUM"),
    ("Integrate Slack notifications to improve team communication and workflow",             "MEDIUM"),
    ("Implement automated testing pipeline to reduce regression bugs in releases",           "MEDIUM"),
    ("Add social media login with Google and GitHub OAuth2 for easier onboarding",          "MEDIUM"),

    # LOW priority epics (nice-to-have, minor, deferred)
    ("Update the color scheme and typography to match the new brand guidelines",             "LOW"),
    ("Add dark mode option to the web interface for user preference",                       "LOW"),
    ("Write additional developer documentation for the internal API endpoints",              "LOW"),
    ("Improve error message text to be more user-friendly and descriptive",                 "LOW"),
    ("Add keyboard shortcuts for power users to navigate the dashboard faster",             "LOW"),
    ("Refactor legacy codebase modules to improve maintainability and readability",         "LOW"),
    ("Add optional tutorial walkthrough for new users during first login experience",       "LOW"),
    ("Improve loading animation and transition effects for a more polished UI feel",        "LOW"),
    ("Add CSV export for sprint reports as an alternative to PDF format",                   "LOW"),
    ("Clean up unused database columns from the schema left by old feature flags",          "LOW"),
]


# ── Feature Extraction with DistilBERT ────────────────────

MODEL_NAME = "distilbert-base-uncased"

def _load_bert():
    from transformers import DistilBertTokenizer, DistilBertModel
    print(f"  Loading {MODEL_NAME} …")
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    model     = DistilBertModel.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model


def extract_features(texts: List[str], tokenizer, bert_model) -> np.ndarray:
    """
    Extract CLS token embeddings using DistilBERT (frozen, no gradient).

    BERT's CLS token summarises the whole sequence via self-attention.
    These 768-dim features capture rich semantic information.

    Args:
        texts: List of Epic description strings

    Returns:
        numpy array of shape (n_texts, 768)
    """
    features = []
    for text in texts:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True,
        )
        with torch.no_grad():
            outputs = bert_model(**inputs)
        cls_emb = outputs.last_hidden_state[0, 0, :].numpy()  # CLS token
        features.append(cls_emb)
    return np.array(features)


# ── Train Classifier ──────────────────────────────────────

def train_bert_classifier(
    data: List[Tuple[str, str]],
    tokenizer,
    bert_model,
):
    """
    Train a LogisticRegression classifier on BERT features.

    Strategy: Feature extraction (freeze BERT) + train small classifier
    This is CPU-friendly — BERT runs in inference mode, no backward pass.

    Returns:
        Trained sklearn LogisticRegression model
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder

    texts  = [text for text, _ in data]
    labels = [label for _, label in data]

    print(f"  Extracting BERT features for {len(texts)} training samples …")
    X = extract_features(texts, tokenizer, bert_model)
    y = np.array([LABEL2IDX[l] for l in labels])

    print("  Training LogisticRegression on BERT features …")
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf.fit(X, y)

    train_acc = clf.score(X, y) * 100
    print(f"  Training accuracy: {train_acc:.1f}%")

    return clf


# ── Inference ─────────────────────────────────────────────

def predict_priority(
    text: str,
    tokenizer,
    bert_model,
    clf,
) -> Dict:
    """
    Predict the priority class of an Epic description.

    Args:
        text: Epic description string

    Returns:
        {prediction, confidence, probabilities}
    """
    X   = extract_features([text], tokenizer, bert_model)
    idx = int(clf.predict(X)[0])
    probs = clf.predict_proba(X)[0]

    return {
        "text":          text[:80] + "…" if len(text) > 80 else text,
        "prediction":    CLASS_NAMES[idx],
        "confidence":    round(float(probs[idx]), 4),
        "probabilities": {CLASS_NAMES[i]: round(float(p), 4) for i, p in enumerate(probs)},
    }


# ── Standalone Demo ───────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("     UNIT 3 — BERT PRIORITY CLASSIFIER DEMO")
    print("=" * 65)
    print("\nApproach  : Transfer Learning (Feature Extraction)")
    print("BERT role : Extract 768-dim CLS embeddings (frozen)")
    print("Classifier: LogisticRegression trained on BERT features")
    print("Task      : Epic Priority → HIGH / MEDIUM / LOW")
    print(f"Train set : {len(TRAINING_DATA)} samples\n")

    from transformers import DistilBertTokenizer, DistilBertModel
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    bert_model = DistilBertModel.from_pretrained(MODEL_NAME)
    bert_model.eval()

    clf = train_bert_classifier(TRAINING_DATA, tokenizer, bert_model)

    # ── Test on unseen Epic descriptions ──────────────────
    print("\n" + "=" * 65)
    print("  INFERENCE ON UNSEEN EPIC DESCRIPTIONS")
    print("=" * 65)

    test_epics = [
        ("Payment system is completely down in production right now", "expect HIGH"),
        ("Add tooltips and help text to improve onboarding experience", "expect LOW"),
        ("Build team performance dashboard for sprint retrospectives",  "expect MEDIUM"),
        ("Security audit required before GDPR compliance submission",   "expect HIGH"),
        ("Refactor CSS to use a design system for consistency",         "expect LOW"),
        ("Integrate with third-party CRM for sales team reporting",     "expect MEDIUM"),
    ]

    for epic, expected in test_epics:
        result = predict_priority(epic, tokenizer, bert_model, clf)
        match  = "✓" if result["prediction"] in expected else "✗"
        print(f"\n  {match} Input      : {epic}")
        print(f"    Predicted  : {result['prediction']}  ({expected})")
        print(f"    Confidence : {result['confidence']}")
        print(f"    All probs  : {result['probabilities']}")

    print("\nTransfer Learning Explained:")
    print("  BERT was pre-trained on 3.3B words → learns general language patterns")
    print("  We add a tiny LogisticRegression on top → learns task-specific mapping")
    print("  Total training time: seconds (vs. hours for training from scratch)")
    print()
