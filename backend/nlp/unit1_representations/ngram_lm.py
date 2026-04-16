"""
=============================================================
Unit 1 — N-gram Language Model + Perplexity
=============================================================
Demonstrates:
  • Bigram and trigram probability estimation from a corpus
  • Laplace (add-1) smoothing for unseen n-grams
  • Perplexity: measures how "surprised" the model is by new text
      PP(W) = exp(-1/N × Σ log P(wᵢ | context))
      Lower perplexity → text is more expected by this model
      Higher perplexity → text is out-of-domain or unexpected

Syllabus: Unit 1 — Language modelling: n-grams, probability, perplexity
Run     : python backend/nlp/unit1_representations/ngram_lm.py
=============================================================
"""

import math
from collections import Counter
from typing import List, Tuple, Dict

import nltk
nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)
from nltk.tokenize import word_tokenize


# ── N-gram Language Model ─────────────────────────────────

class NgramLanguageModel:
    """
    Bigram (n=2) or Trigram (n=3) language model with Laplace smoothing.

    Training: Counts all n-grams from a corpus.
    Inference: Estimates P(word | previous words) via:
        P(w_n | w_{n-1}) = (C(w_{n-1}, w_n) + 1) / (C(w_{n-1}) + V)
    where V = vocabulary size (Laplace add-1 smoothing).
    """

    def __init__(self, n: int = 2):
        """
        Args:
            n: Order of the model — 2 (bigram) or 3 (trigram)
        """
        assert n in (2, 3), "Only bigram (n=2) or trigram (n=3) supported"
        self.n = n
        self.ngram_counts: Counter = Counter()
        self.context_counts: Counter = Counter()
        self.vocab: set = set()
        self.V: int = 0   # Vocabulary size (for Laplace smoothing)

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize and wrap with start/end markers."""
        tokens = word_tokenize(text.lower())
        return ["<s>"] * (self.n - 1) + tokens + ["</s>"]

    def train(self, documents: List[str]) -> None:
        """
        Estimate n-gram probabilities from a list of text documents.

        Args:
            documents: List of training sentences / paragraphs
        """
        all_tokens: List[str] = []
        for doc in documents:
            tokens = self._tokenize(doc)
            self.vocab.update(tokens)
            all_tokens.extend(tokens)

        self.V = len(self.vocab)

        # Count all n-grams and their (n-1)-gram contexts
        for i in range(len(all_tokens) - self.n + 1):
            ngram   = tuple(all_tokens[i : i + self.n])
            context = ngram[: self.n - 1]
            self.ngram_counts[ngram]   += 1
            self.context_counts[context] += 1

        print(f"  {self.n}-gram LM trained | vocab={self.V} | n-grams={len(self.ngram_counts)}")

    def probability(self, ngram: Tuple[str, ...]) -> float:
        """
        Laplace-smoothed conditional probability of an n-gram.
        P(w_n | w_1…w_{n-1}) = (C(w_1…w_n) + 1) / (C(w_1…w_{n-1}) + V)
        """
        context      = ngram[: self.n - 1]
        count_ngram  = self.ngram_counts.get(ngram, 0)
        count_ctx    = self.context_counts.get(context, 0)
        return (count_ngram + 1) / (count_ctx + self.V)

    def perplexity(self, text: str) -> float:
        """
        Compute perplexity of a text string under this language model.

        PP(W) = exp( -1/N × Σ log P(wᵢ | context) )

        Lower = text is more "expected" by the model (in-domain)
        Higher = text is surprising / out-of-domain

        Args:
            text: A new text string to evaluate

        Returns:
            Perplexity score (float)
        """
        tokens = self._tokenize(text)
        N = len(tokens) - (self.n - 1)
        if N <= 0:
            return float("inf")

        log_prob_sum = 0.0
        for i in range(self.n - 1, len(tokens)):
            ngram = tuple(tokens[i - self.n + 1 : i + 1])
            p = self.probability(ngram)
            log_prob_sum += math.log(p)

        avg_log_prob = log_prob_sum / N
        return math.exp(-avg_log_prob)

    def most_common(self, top_k: int = 10) -> List[Tuple[Tuple, int]]:
        """Return the top-k most frequent n-grams."""
        return self.ngram_counts.most_common(top_k)

    def next_word_distribution(self, context: Tuple[str, ...], top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Given a context, return the top-k most probable next words.

        Args:
            context: (n-1)-gram context tuple, e.g. ("sprint", ) for bigram
            top_k:   Number of candidates to return
        """
        candidates = {}
        for ngram in self.ngram_counts:
            if ngram[: self.n - 1] == context:
                word = ngram[-1]
                candidates[word] = self.probability(ngram)
        sorted_cands = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        return sorted_cands[:top_k]


# ── Standalone Demo ───────────────────────────────────────

if __name__ == "__main__":
    # Scrum meeting corpus for training
    CORPUS = [
        "we need to build a user authentication system with jwt tokens",
        "the payment gateway integration is critical for q2 launch",
        "sprint planning meeting starts on monday morning at ten",
        "i completed the login feature and pushed to production",
        "we are working on the analytics dashboard this sprint",
        "the sprint review is scheduled for friday afternoon",
        "i am blocked on the database migration task",
        "let us pull in the payment gateway story this sprint",
        "user authentication is the highest priority epic",
        "the team velocity is thirty four story points per sprint",
        "we need to complete the backlog grooming before sprint planning",
        "the acceptance criteria for the payment story are defined",
        "developers estimated eight story points for the authentication epic",
        "the sprint retrospective helped us improve our workflow process",
        "our goal is to deliver the minimum viable product by q2",
        "the scrum master facilitated the daily standup meeting",
        "the product owner reviewed the user stories in the backlog",
        "we will have the deployment ready before the sprint ends",
        "the team discussed the risks and blockers during standup",
        "sprint planning helps the team commit to realistic goals",
    ]

    print("=" * 65)
    print("    UNIT 1 — N-GRAM LANGUAGE MODEL + PERPLEXITY DEMO")
    print("=" * 65)

    # ── Train bigram model ────────────────────────────────
    print("\n[1] BIGRAM MODEL (n=2)")
    print("-" * 45)
    bigram = NgramLanguageModel(n=2)
    bigram.train(CORPUS)

    print("\n  Top-10 most common bigrams:")
    for ngram, count in bigram.most_common(10):
        print(f"    {ngram}  →  count={count}")

    # ── Train trigram model ───────────────────────────────
    print("\n[2] TRIGRAM MODEL (n=3)")
    print("-" * 45)
    trigram = NgramLanguageModel(n=3)
    trigram.train(CORPUS)

    print("\n  Top-10 most common trigrams:")
    for ngram, count in trigram.most_common(10):
        print(f"    {ngram}  →  count={count}")

    # ── Sample probabilities ──────────────────────────────
    print("\n[3] CONDITIONAL PROBABILITIES  (Laplace-smoothed)")
    print("-" * 45)
    test_cases = [
        (("sprint",), "planning",   "seen in training"),
        (("payment",), "gateway",   "seen in training"),
        (("story",),   "points",    "seen in training"),
        (("random",),  "word",      "unseen — smoothed to near-zero"),
    ]
    print(f"  {'Context':<15} {'Next word':<15} {'P(next|ctx)':>12}  Note")
    print("  " + "-" * 60)
    for ctx, word, note in test_cases:
        p = bigram.probability(ctx + (word,))
        print(f"  {str(ctx):<15} {word:<15} {p:>12.8f}  {note}")

    # ── Next word predictions ─────────────────────────────
    print("\n[4] NEXT WORD PREDICTION (top-5 candidates)")
    print("-" * 45)
    contexts = [("sprint",), ("payment",), ("story",)]
    for ctx in contexts:
        candidates = bigram.next_word_distribution(ctx, top_k=5)
        print(f"\n  Context: {ctx[0]!r}")
        for word, prob in candidates:
            print(f"    → '{word}'  p={prob:.6f}")

    # ── Perplexity comparison ─────────────────────────────
    print("\n[5] PERPLEXITY COMPARISON (bigram model)")
    print("-" * 45)
    print("  Lower perplexity = more expected by the Scrum-trained model\n")

    test_sentences = [
        ("Sprint planning meeting scheduled for Monday",         "Scrum domain"),
        ("User authentication story points estimated at eight",  "Scrum domain"),
        ("The backlog grooming session went well today",          "Scrum domain"),
        ("The cat sat on the mat wearing a funny hat",            "Out-of-domain"),
        ("Pizza delivery service is open twenty four hours",      "Out-of-domain"),
        ("Stock market analysis shows Q3 growth of fifteen",      "Out-of-domain"),
    ]

    print(f"  {'Sentence':<52} {'Perplexity':>12}  Domain")
    print("  " + "-" * 75)
    for sent, domain in test_sentences:
        pp = bigram.perplexity(sent)
        print(f"  {sent:<52} {pp:>12.1f}  {domain}")

    print("\n  → Scrum sentences have much lower perplexity (the model 'knows' them)")
    print("  → Out-of-domain sentences have very high perplexity")
    print()
