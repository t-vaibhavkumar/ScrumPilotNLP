"""
=============================================================
Unit 2 — LSTM Text Classifier
=============================================================
Classifies meeting transcripts into 3 types using a PyTorch LSTM:
  0: PM_MEETING       — product/stakeholder meetings
  1: SPRINT_PLANNING  — sprint planning sessions
  2: STANDUP          — daily standup meetings

Architecture:
  Embedding(vocab_size, 64) → LSTM(64, 128, layers=2) → Dropout(0.3)
    → Linear(128, 3) → LogSoftmax → class label

Why LSTM over plain RNN?
  LSTM has forget/input/output gates → handles long-range dependencies
  and avoids the vanishing gradient problem of simple RNNs.

Syllabus: Unit 2 — RNNs, vanishing gradients, LSTM architecture
Run     : python backend/nlp/unit2_models/lstm_classifier.py
=============================================================
"""

import re
from typing import List, Tuple, Dict
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# ── Synthetic Training Data ───────────────────────────────
# 3 classes: PM_MEETING=0, SPRINT_PLANNING=1, STANDUP=2

TRAINING_DATA: List[Tuple[str, int]] = [
    # ── PM_MEETING (label 0) ──────────────────────────────
    ("The stakeholders want a payment gateway integrated by Q2", 0),
    ("Business value for the authentication epic is rated at nine", 0),
    ("The product owner reviewed the requirements with the client", 0),
    ("We need to prioritize the analytics dashboard for the next release", 0),
    ("The Acme Corp contract requires SSO integration before launch", 0),
    ("Management approved the budget for the new mobile application", 0),
    ("The roadmap includes three major epics for Q3 delivery", 0),
    ("Stakeholders confirmed that payment feature is the highest priority", 0),
    ("The product vision aligns with user authentication and data security", 0),
    ("Market research shows users need better notification features", 0),
    ("Investors expect the MVP to be ready before the funding round", 0),
    ("The client requested multi-language support for international expansion", 0),
    ("Business requirements include advanced reporting and export features", 0),
    ("The legal team requires GDPR compliance for user data management", 0),
    ("Product strategy focuses on mobile first design and performance", 0),

    # ── SPRINT_PLANNING (label 1) ─────────────────────────
    ("Our sprint goal is to complete the payment gateway integration", 1),
    ("Team capacity this sprint is eighty hours across five developers", 1),
    ("Let's commit to the authentication story and the dashboard epic", 1),
    ("Sarah will take the frontend work and Mike handles the backend API", 1),
    ("We are pulling in four stories totaling thirty-four story points", 1),
    ("The sprint starts Monday and ends Friday in two weeks", 1),
    ("Bob is on leave for three days so total capacity is reduced", 1),
    ("Let us pull SP-187 and SP-188 into this sprint", 1),
    ("We need to break this story into subtasks before we can estimate", 1),
    ("Previous sprint velocity was thirty eight point so let's target that", 1),
    ("The definition of done requires unit tests and code review approval", 1),
    ("Risk identified: we depend on DevOps team for the webhook setup", 1),
    ("Alice can take the notification task and Tom handles deployment", 1),
    ("We commit to delivering the login feature by end of this sprint", 1),
    ("Sprint planning estimates show we have capacity for three stories", 1),

    # ── STANDUP (label 2) ─────────────────────────────────
    ("Yesterday I completed the login API and pushed the code to main", 2),
    ("Today I will work on the payment form component and unit tests", 2),
    ("I am blocked on the database migration waiting for DBA approval", 2),
    ("I finished the authentication task and started the dashboard widget", 2),
    ("No blockers from my side, I will continue with the API integration", 2),
    ("Yesterday I reviewed pull requests and helped Mike debug the issue", 2),
    ("I completed the user story for the notification system yesterday", 2),
    ("Today I plan to finish the REST endpoint and write integration tests", 2),
    ("I am blocked waiting for the staging environment to be set up", 2),
    ("Yesterday I worked on the frontend payment form, it is nearly done", 2),
    ("Today I will do the code review for Sarah and start my next task", 2),
    ("I completed the database schema migration and updated the tests", 2),
    ("Blocker: the third party API is returning errors in the sandbox", 2),
    ("I am working on the CI pipeline configuration and Docker setup", 2),
    ("Yesterday I deployed the feature to staging and fixed two bugs", 2),
]

CLASS_NAMES = ["PM_MEETING", "SPRINT_PLANNING", "STANDUP"]


# ── Preprocessing ─────────────────────────────────────────

def tokenize(text: str) -> List[str]:
    return re.sub(r"[^\w\s]", "", text.lower()).split()


class Vocabulary:
    """Simple word-to-index vocabulary."""

    def __init__(self, min_freq: int = 1):
        self.word2idx: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word: Dict[int, str] = {0: "<PAD>", 1: "<UNK>"}
        self.min_freq = min_freq

    def build(self, sentences: List[List[str]]) -> None:
        counts = Counter(word for sent in sentences for word in sent)
        for word, freq in counts.items():
            if freq >= self.min_freq and word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

    def encode(self, tokens: List[str]) -> List[int]:
        return [self.word2idx.get(t, 1) for t in tokens]  # 1 = <UNK>

    def __len__(self) -> int:
        return len(self.word2idx)


def pad_sequence(seq: List[int], max_len: int, pad_idx: int = 0) -> List[int]:
    if len(seq) >= max_len:
        return seq[:max_len]
    return seq + [pad_idx] * (max_len - len(seq))


# ── Dataset ───────────────────────────────────────────────

class MeetingDataset(Dataset):
    def __init__(self, data: List[Tuple[str, int]], vocab: Vocabulary, max_len: int = 30):
        self.samples = []
        for text, label in data:
            tokens = tokenize(text)
            ids    = vocab.encode(tokens)
            padded = pad_sequence(ids, max_len)
            self.samples.append((torch.tensor(padded, dtype=torch.long), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ── LSTM Model ────────────────────────────────────────────

class LSTMClassifier(nn.Module):
    """
    LSTM-based text classifier.

    Architecture:
        Embedding → LSTM (2 layers) → Dropout → Linear → LogSoftmax

    The final hidden state of the LSTM captures the sequence context
    and is passed to a linear layer for classification.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout  = nn.Dropout(dropout)
        self.fc       = nn.Linear(hidden_dim, num_classes)
        self.log_soft = nn.LogSoftmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len)
        embedded = self.embedding(x)                    # (batch, seq, embed_dim)
        out, (h_n, c_n) = self.lstm(embedded)          # h_n: (layers, batch, hidden)
        last_hidden = h_n[-1]                           # Take last layer's hidden state
        dropped     = self.dropout(last_hidden)
        logits      = self.fc(dropped)                  # (batch, num_classes)
        return self.log_soft(logits)


# ── Training ──────────────────────────────────────────────

def train(
    data: List[Tuple[str, int]],
    epochs: int = 30,
    lr: float = 1e-3,
    max_len: int = 30,
) -> Tuple[LSTMClassifier, Vocabulary]:
    """Train the LSTM classifier and return (model, vocabulary)."""

    # Build vocabulary
    vocab = Vocabulary()
    vocab.build([tokenize(text) for text, _ in data])

    dataset    = MeetingDataset(data, vocab, max_len)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    model     = LSTMClassifier(vocab_size=len(vocab))
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss, correct, total = 0.0, 0, 0
        for seqs, labels in dataloader:
            labels = torch.tensor(labels, dtype=torch.long) if not isinstance(labels, torch.Tensor) else labels
            optimizer.zero_grad()
            out  = model(seqs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            preds   = out.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += len(labels)

        if epoch % 10 == 0:
            acc = correct / total * 100
            print(f"    Epoch {epoch:3d}/{epochs}  loss={total_loss/len(dataloader):.4f}  acc={acc:.1f}%")

    return model, vocab


# ── Inference ─────────────────────────────────────────────

def predict(model: LSTMClassifier, vocab: Vocabulary, text: str, max_len: int = 30) -> Dict:
    """Predict the meeting type of a single text string."""
    model.eval()
    tokens  = tokenize(text)
    ids     = vocab.encode(tokens)
    padded  = pad_sequence(ids, max_len)
    tensor  = torch.tensor([padded], dtype=torch.long)

    with torch.no_grad():
        log_probs = model(tensor)
        probs     = torch.exp(log_probs)[0].tolist()
        pred_idx  = int(torch.argmax(log_probs, dim=1).item())

    return {
        "text":       text,
        "prediction": CLASS_NAMES[pred_idx],
        "confidence": round(probs[pred_idx], 4),
        "probabilities": {CLASS_NAMES[i]: round(p, 4) for i, p in enumerate(probs)},
    }


# ── Standalone Demo ───────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("        UNIT 2 — LSTM TEXT CLASSIFIER DEMO")
    print("=" * 65)
    print("\nArchitecture: Embedding → LSTM(2 layers) → Dropout → Linear")
    print(f"Task        : Classify meeting type ({', '.join(CLASS_NAMES)})")
    print(f"Training set: {len(TRAINING_DATA)} samples ({len(TRAINING_DATA)//3} per class)\n")

    print("Training LSTM …")
    model, vocab = train(TRAINING_DATA, epochs=40, lr=1e-3)
    print(f"\n  Vocabulary size: {len(vocab)} words")

    # ── Test on unseen sentences ──────────────────────────
    print("\n" + "=" * 65)
    print("  INFERENCE ON UNSEEN SENTENCES")
    print("=" * 65)

    test_inputs = [
        "The CEO wants the new features shipped before the product launch",
        "Today I finished the API endpoint and will start writing tests",
        "We have sixty hours of capacity and will take three user stories",
        "I am blocked on the infrastructure setup can someone help",
        "Business value for the payment epic is critical for revenue",
        "Our sprint goal for this week is the complete authentication flow",
    ]

    for text in test_inputs:
        result = predict(model, vocab, text)
        print(f"\n  Input      : {text}")
        print(f"  Predicted  : {result['prediction']}  (confidence={result['confidence']:.4f})")
        print(f"  All probs  : {result['probabilities']}")

    print()
