"""
=============================================================
Unit 2 — Evaluation Metrics
=============================================================
Comprehensive NLP evaluation functions:
  Classification: Precision, Recall, F1, Confusion Matrix
  Generation    : BLEU score   (for generated text vs reference)
  Summarization : ROUGE-1, ROUGE-2, ROUGE-L scores

All metrics explained with formulas in docstrings.

Syllabus: Unit 2 — Evaluation metrics: precision, recall, F1,
          confusion matrix, BLEU, ROUGE
Run     : python backend/nlp/unit2_models/evaluator.py
=============================================================
"""

import math
from typing import List, Dict, Tuple
from collections import Counter


# ════════════════════════════════════════════════════════
# CLASSIFICATION METRICS
# ════════════════════════════════════════════════════════

def confusion_matrix(y_true: List[int], y_pred: List[int], n_classes: int) -> List[List[int]]:
    """
    Compute confusion matrix.

    CM[i][j] = number of instances where true class = i, predicted class = j.
    Correct predictions lie on the main diagonal.
    """
    cm = [[0] * n_classes for _ in range(n_classes)]
    for true, pred in zip(y_true, y_pred):
        cm[true][pred] += 1
    return cm


def precision_recall_f1(
    y_true: List[int],
    y_pred: List[int],
    n_classes: int,
) -> Dict:
    """
    Per-class and macro-average Precision, Recall, and F1 score.

    Precision = TP / (TP + FP)  — of all predicted positives, how many are correct?
    Recall    = TP / (TP + FN)  — of all actual positives, how many did we find?
    F1        = 2 × (P × R) / (P + R)  — harmonic mean of P and R

    Macro-average: average the per-class scores (treats all classes equally).
    """
    per_class = {}
    for c in range(n_classes):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        per_class[c] = {
            "precision": round(precision, 4),
            "recall":    round(recall,    4),
            "f1":        round(f1,        4),
            "tp": tp, "fp": fp, "fn": fn,
        }

    macro_p  = sum(v["precision"] for v in per_class.values()) / n_classes
    macro_r  = sum(v["recall"]    for v in per_class.values()) / n_classes
    macro_f1 = sum(v["f1"]        for v in per_class.values()) / n_classes

    accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)

    return {
        "per_class":    per_class,
        "macro_precision": round(macro_p,  4),
        "macro_recall":    round(macro_r,  4),
        "macro_f1":        round(macro_f1, 4),
        "accuracy":        round(accuracy, 4),
    }


def print_classification_report(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
) -> None:
    """Print a formatted classification report (like sklearn's)."""
    n = len(class_names)
    metrics = precision_recall_f1(y_true, y_pred, n)
    cm = confusion_matrix(y_true, y_pred, n)

    print("\nClassification Report:")
    print(f"  {'Class':<20} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print("  " + "-" * 50)
    for i, name in enumerate(class_names):
        m = metrics["per_class"][i]
        print(f"  {name:<20} {m['precision']:>10.4f} {m['recall']:>8.4f} {m['f1']:>8.4f}")

    print("  " + "-" * 50)
    print(f"  {'Macro avg':<20} {metrics['macro_precision']:>10.4f} "
          f"{metrics['macro_recall']:>8.4f} {metrics['macro_f1']:>8.4f}")
    print(f"\n  Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.1f}%)")

    print("\nConfusion Matrix (rows=true, cols=predicted):")
    header = "  " + " " * 18 + "".join(f"{n[:8]:>10}" for n in class_names)
    print(header)
    for i, name in enumerate(class_names):
        row_str = "".join(f"{v:>10}" for v in cm[i])
        print(f"  {name:<18} {row_str}")


# ════════════════════════════════════════════════════════
# BLEU SCORE
# ════════════════════════════════════════════════════════

def _ngrams(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))


def bleu_score(
    hypothesis: str,
    reference: str,
    max_n: int = 4,
) -> Dict:
    """
    Compute BLEU score (Bilingual Evaluation Understudy).

    BLEU measures n-gram overlap between a generated text (hypothesis)
    and a reference text. Used to evaluate NLP generation quality.

    BLEU = BP × exp(Σ wₙ × log pₙ)
      where pₙ = modified n-gram precision
            BP = brevity penalty (penalizes short hypotheses)
            wₙ = 1/max_n (uniform weights)

    Score range: 0.0 (no match) → 1.0 (perfect match)

    Args:
        hypothesis: Model-generated text
        reference : Ground truth reference text
        max_n     : Maximum n-gram order (default 4 = BLEU-4)
    """
    hyp_tokens = hypothesis.lower().split()
    ref_tokens = reference.lower().split()

    if not hyp_tokens:
        return {"bleu": 0.0, "ngram_precisions": {}}

    # Compute modified n-gram precision for each order
    precisions = {}
    log_sum = 0.0
    for n in range(1, max_n + 1):
        hyp_ngrams = _ngrams(hyp_tokens, n)
        ref_ngrams = _ngrams(ref_tokens, n)

        # Clipped count: min(hypothesis count, reference count)
        clipped = sum(min(count, ref_ngrams[ng]) for ng, count in hyp_ngrams.items())
        total   = max(sum(hyp_ngrams.values()), 1)

        p_n = clipped / total if total > 0 else 0.0
        precisions[f"BLEU-{n}"] = round(p_n, 4)

        if p_n > 0:
            log_sum += (1.0 / max_n) * math.log(p_n)

    # Brevity penalty
    r  = len(ref_tokens)
    c  = len(hyp_tokens)
    bp = 1.0 if c >= r else math.exp(1 - r / c)

    bleu = bp * math.exp(log_sum) if log_sum > -float("inf") else 0.0

    return {
        "bleu":             round(bleu, 4),
        "brevity_penalty":  round(bp, 4),
        "ngram_precisions": precisions,
    }


# ════════════════════════════════════════════════════════
# ROUGE SCORE
# ════════════════════════════════════════════════════════

def _lcs_length(a: List[str], b: List[str]) -> int:
    """Length of the Longest Common Subsequence (for ROUGE-L)."""
    m, n = len(a), len(b)
    dp   = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i-1][j-1] + 1 if a[i-1] == b[j-1] else max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]


def rouge_score(hypothesis: str, reference: str) -> Dict:
    """
    Compute ROUGE-1, ROUGE-2, and ROUGE-L scores.

    ROUGE = Recall-Oriented Understudy for Gisting Evaluation.
    Used to evaluate summarization quality.

    ROUGE-1: Unigram (word) overlap
    ROUGE-2: Bigram overlap
    ROUGE-L: Longest Common Subsequence (respects word order)

    F1 formula: 2 × (Precision × Recall) / (Precision + Recall)

    Args:
        hypothesis: Model-generated summary
        reference : Ground truth summary / reference
    """
    hyp = hypothesis.lower().split()
    ref = reference.lower().split()

    def _rouge_n(hyp_tokens, ref_tokens, n):
        hyp_ng = _ngrams(hyp_tokens, n)
        ref_ng = _ngrams(ref_tokens, n)
        overlap = sum(min(hyp_ng[ng], ref_ng[ng]) for ng in hyp_ng)
        precision = overlap / max(sum(hyp_ng.values()), 1)
        recall    = overlap / max(sum(ref_ng.values()), 1)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}

    # ROUGE-L
    lcs = _lcs_length(hyp, ref)
    p_l = lcs / max(len(hyp), 1)
    r_l = lcs / max(len(ref), 1)
    f_l = 2 * p_l * r_l / (p_l + r_l) if (p_l + r_l) > 0 else 0.0

    return {
        "ROUGE-1": _rouge_n(hyp, ref, 1),
        "ROUGE-2": _rouge_n(hyp, ref, 2),
        "ROUGE-L": {"precision": round(p_l, 4), "recall": round(r_l, 4), "f1": round(f_l, 4)},
    }


# ── Standalone Demo ───────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("         UNIT 2 — EVALUATION METRICS DEMO")
    print("=" * 65)

    # ── Classification Metrics ────────────────────────────
    print("\n[1] CLASSIFICATION METRICS  (Precision / Recall / F1 / Confusion Matrix)")
    print("-" * 55)

    # Simulating predictions from the LSTM/GRU classifiers
    y_true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
    y_pred = [0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 2, 2, 2, 1, 2]
    class_names = ["PM_MEETING", "SPRINT_PLANNING", "STANDUP"]

    print_classification_report(y_true, y_pred, class_names)

    # ── BLEU Score ────────────────────────────────────────
    print("\n[2] BLEU SCORE  (Generated text quality)")
    print("-" * 55)
    print("Use case: Evaluate AI-generated Epic descriptions vs. reference\n")

    bleu_cases = [
        (
            "Build a user authentication system with JWT tokens and SSO support",
            "Implement user authentication using JWT tokens and single sign-on",
            "High overlap",
        ),
        (
            "Create a payment gateway for Stripe checkout integration",
            "Implement user authentication using JWT tokens and single sign-on",
            "Low overlap",
        ),
        (
            "Implement user authentication",
            "Implement user authentication using JWT tokens and single sign-on",
            "Short hypothesis (brevity penalty)",
        ),
    ]

    for hyp, ref, note in bleu_cases:
        result = bleu_score(hyp, ref)
        print(f"  [{note}]")
        print(f"    Hypothesis : {hyp}")
        print(f"    Reference  : {ref}")
        print(f"    BLEU-4     : {result['bleu']}")
        print(f"    Precisions : {result['ngram_precisions']}")
        print(f"    Brevity BP : {result['brevity_penalty']}")
        print()

    # ── ROUGE Score ───────────────────────────────────────
    print("[3] ROUGE SCORE  (Summarization quality)")
    print("-" * 55)
    print("Use case: Evaluate AI meeting summary vs. manually written minutes\n")

    reference_summary = (
        "The team agreed to prioritize the payment gateway integration for Q2. "
        "Authentication and dashboard epics were also discussed and confirmed."
    )
    generated_summary = (
        "Payment gateway integration is the top priority for Q2 delivery. "
        "The authentication epic and dashboard were confirmed as important items."
    )
    bad_summary = "The weather is nice and the coffee was good today."

    for hyp, label in [
        (generated_summary, "Good AI summary"),
        (bad_summary,       "Poor irrelevant summary"),
    ]:
        scores = rouge_score(hyp, reference_summary)
        print(f"  [{label}]")
        print(f"    Hypothesis: {hyp[:70]}…")
        print(f"    Reference : {reference_summary[:70]}…")
        print(f"    ROUGE-1   : {scores['ROUGE-1']}")
        print(f"    ROUGE-2   : {scores['ROUGE-2']}")
        print(f"    ROUGE-L   : {scores['ROUGE-L']}")
        print()
