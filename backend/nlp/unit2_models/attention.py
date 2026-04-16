"""
=============================================================
Unit 2 — Attention Mechanism
=============================================================
Implements Bahdanau (additive) attention over LSTM hidden states.
Shows WHICH words the model focuses on when making a classification.

Architecture (LSTM + Attention):
  Embedding → LSTM → Attention weights → Weighted context vector
    → Linear → class label

Why attention?
  Plain LSTM uses only the FINAL hidden state → information bottleneck
  for long sequences. Attention attends to ALL hidden states,
  weighting each by its relevance to the classification task.

Syllabus: Unit 2 — Attention mechanism for sequences
Run     : python backend/nlp/unit2_models/attention.py
=============================================================
"""

import re
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# ── Bahdanau Attention Layer ──────────────────────────────

class BahdanauAttention(nn.Module):
    """
    Additive (Bahdanau) attention mechanism.

    For each LSTM hidden state h_t, computes an attention score:
      e_t = v · tanh(W_h · h_t + b)     (energy / alignment score)
      α_t = softmax(e_t)                 (attention weight)
      c   = Σ α_t · h_t                 (context vector)

    The context vector c summarizes the relevant parts of the input
    sequence for the current decision.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            lstm_outputs: (batch, seq_len, hidden_dim) — all LSTM hidden states

        Returns:
            context      : (batch, hidden_dim) — weighted sum of hidden states
            attention_wts: (batch, seq_len)    — attention weight per time step
        """
        # Compute alignment scores
        energy = torch.tanh(self.W(lstm_outputs))   # (batch, seq, hidden)
        scores = self.v(energy).squeeze(-1)          # (batch, seq)

        # Attention weights via softmax
        attention_wts = F.softmax(scores, dim=1)    # (batch, seq)

        # Weighted context vector
        context = torch.bmm(
            attention_wts.unsqueeze(1),  # (batch, 1, seq)
            lstm_outputs                 # (batch, seq, hidden)
        ).squeeze(1)                     # (batch, hidden)

        return context, attention_wts


# ── LSTM + Attention Classifier ───────────────────────────

class LSTMWithAttention(nn.Module):
    """
    LSTM classifier enhanced with Bahdanau attention.

    Unlike plain LSTM (uses only final h_n), this model:
      1. Runs LSTM over all tokens → all hidden states
      2. Uses attention to compute a weighted sum of hidden states
      3. Classification is based on the attended context vector

    This allows interpretation: which words were most important?
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        num_classes: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm      = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.attention = BahdanauAttention(hidden_dim)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_dim, num_classes)
        self.log_soft  = nn.LogSoftmax(dim=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            log_probs   : (batch, num_classes)
            attn_weights: (batch, seq_len) — for interpretability
        """
        embedded   = self.embedding(x)                   # (batch, seq, embed)
        lstm_out, _ = self.lstm(embedded)                # (batch, seq, hidden)
        context, attn_wts = self.attention(lstm_out)     # (batch, hidden), (batch, seq)
        dropped    = self.dropout(context)
        logits     = self.fc(dropped)
        return self.log_soft(logits), attn_wts


# ── Minimal training data for demo ───────────────────────

TRAINING_DATA: List[Tuple[str, int]] = [
    # PM_MEETING=0
    ("stakeholders want payment gateway integrated by Q2", 0),
    ("business value for authentication epic is nine out of ten", 0),
    ("product owner reviewed requirements with client today", 0),
    ("management approved budget for the new mobile application", 0),
    ("roadmap includes three epics for Q3 delivery timeline", 0),
    # SPRINT_PLANNING=1
    ("sprint goal is to complete the payment gateway integration", 1),
    ("team capacity this sprint is eighty hours total", 1),
    ("we commit to authentication story and dashboard epic", 1),
    ("sprint starts monday and ends in two weeks on friday", 1),
    ("previous velocity was thirty eight story points per sprint", 1),
    # STANDUP=2
    ("yesterday I completed the login API and pushed to main", 2),
    ("today I will work on payment form component and tests", 2),
    ("I am blocked on database migration waiting for approval", 2),
    ("I finished authentication task and started dashboard widget", 2),
    ("no blockers from my side I will continue with API work", 2),
]
CLASS_NAMES = ["PM_MEETING", "SPRINT_PLANNING", "STANDUP"]


# ── Preprocessing utilities ───────────────────────────────

def tokenize(text: str) -> List[str]:
    return re.sub(r"[^\w\s]", "", text.lower()).split()


class SimpleVocab:
    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}

    def build(self, sentences):
        for sent in sentences:
            for word in sent:
                if word not in self.word2idx:
                    self.word2idx[word] = len(self.word2idx)

    def encode(self, tokens, max_len=20):
        ids = [self.word2idx.get(t, 1) for t in tokens]
        return (ids + [0] * max_len)[:max_len]

    def decode(self, ids):
        inv = {v: k for k, v in self.word2idx.items()}
        return [inv.get(i, "<UNK>") for i in ids if i != 0]

    def __len__(self):
        return len(self.word2idx)


# ── Training ──────────────────────────────────────────────

def train_attention_model(
    data: List[Tuple[str, int]],
    epochs: int = 40,
    max_len: int = 20,
) -> Tuple[LSTMWithAttention, SimpleVocab]:
    vocab = SimpleVocab()
    vocab.build([tokenize(t) for t, _ in data])

    X = [torch.tensor(vocab.encode(tokenize(t), max_len)) for t, _ in data]
    y = torch.tensor([l for _, l in data], dtype=torch.long)

    model     = LSTMWithAttention(vocab_size=len(vocab))
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        seqs       = torch.stack(X)
        log_probs, _ = model(seqs)
        loss         = criterion(log_probs, y)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            preds   = log_probs.argmax(dim=1)
            acc     = (preds == y).float().mean().item() * 100
            print(f"    Epoch {epoch:3d}/{epochs}  loss={loss.item():.4f}  acc={acc:.1f}%")

    return model, vocab


# ── Inference with attention visualization ─────────────────

def predict_with_attention(
    model: LSTMWithAttention,
    vocab: SimpleVocab,
    text: str,
    max_len: int = 20,
) -> Dict:
    """Predict class and return per-token attention weights."""
    model.eval()
    tokens  = tokenize(text)
    ids     = vocab.encode(tokens, max_len)
    tensor  = torch.tensor([ids], dtype=torch.long)

    with torch.no_grad():
        log_probs, attn_wts = model(tensor)
        probs    = torch.exp(log_probs)[0].tolist()
        pred_idx = int(log_probs.argmax(dim=1).item())
        weights  = attn_wts[0, :len(tokens)].tolist()

    return {
        "text":        text,
        "tokens":      tokens,
        "prediction":  CLASS_NAMES[pred_idx],
        "confidence":  round(probs[pred_idx], 4),
        "attn_weights": {tok: round(w, 4) for tok, w in zip(tokens, weights)},
    }


# ── Standalone Demo ───────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("      UNIT 2 — ATTENTION MECHANISM DEMO")
    print("=" * 65)
    print("\nArchitecture: Embedding → LSTM → Bahdanau Attention → Linear")
    print("Goal: Show which words the model attends to for classification\n")

    print("Training LSTM + Attention …")
    model, vocab = train_attention_model(TRAINING_DATA, epochs=60)
    print(f"  Vocabulary size: {len(vocab)}")

    # ── Show attention weights ────────────────────────────
    print("\n" + "=" * 65)
    print("  ATTENTION WEIGHT VISUALIZATION")
    print("  (higher weight = model focuses more on this word)")
    print("=" * 65)

    test_sentences = [
        "stakeholders approved the payment budget for Q3",
        "I completed the authentication task yesterday morning",
        "sprint goal is to deliver the login feature by friday",
    ]

    for text in test_sentences:
        result = predict_with_attention(model, vocab, text)
        print(f"\n  Input     : {text}")
        print(f"  Predicted : {result['prediction']}  (conf={result['confidence']:.4f})")
        print(f"  Attention weights (token → weight):")

        # Sort by attention weight  (highest first)
        sorted_attn = sorted(result["attn_weights"].items(), key=lambda x: x[1], reverse=True)
        for token, weight in sorted_attn:
            bar = "█" * int(weight * 40)
            print(f"    {token:<20} {weight:.4f}  {bar}")

    print("\n  Attention lets us interpret the model:")
    print("  → 'stakeholders', 'budget', 'payment' → high weight for PM_MEETING")
    print("  → 'completed', 'yesterday' → high weight for STANDUP")
    print("  → 'sprint', 'goal', 'deliver' → high weight for SPRINT_PLANNING")
    print()
