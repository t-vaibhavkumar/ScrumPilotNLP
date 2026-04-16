"""
=============================================================
Unit 2 — GRU Text Classifier
=============================================================
Classifies individual standup sentences into 5 scrum action types:
  0: create_task    — "We need to add a task for X"
  1: complete_task  — "I finished the login feature"
  2: update_status  — "I'm currently working on the payment form"
  3: assign_task    — "Sarah will handle the API work"
  4: no_action      — General discussion, no actionable item

Architecture:
  Embedding(vocab_size, 64) → GRU(64, 128) → Dropout(0.3)
    → Linear(128, 5) → LogSoftmax → class label

GRU vs LSTM:
  GRU has fewer gates (reset + update) vs LSTM (forget + input + output)
  GRU is simpler, faster to train, comparable performance on short text.
  For short single sentences → GRU often preferred.

Syllabus: Unit 2 — LSTM and GRU architectures
Run     : python backend/nlp/unit2_models/gru_classifier.py
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
# 5 classes: 0=create_task, 1=complete_task, 2=update_status, 3=assign_task, 4=no_action

TRAINING_DATA: List[Tuple[str, int]] = [
    # ── create_task (0) ───────────────────────────────────
    ("We need to create a task for the database migration", 0),
    ("Someone should write unit tests for the authentication module", 0),
    ("Let us add a task to set up the CI pipeline configuration", 0),
    ("We should create a ticket to fix the payment validation bug", 0),
    ("A new task is needed for the API documentation update", 0),
    ("We need to add integration tests for the notification service", 0),
    ("Let us create a subtask for the Docker container configuration", 0),
    ("I think we should track the performance optimization separately", 0),
    ("We need to log the security vulnerability as a task", 0),
    ("Someone needs to create a task for updating the dependencies", 0),

    # ── complete_task (1) ─────────────────────────────────
    ("I finished the login API yesterday and merged to main", 1),
    ("The payment form is done and deployed to staging", 1),
    ("I completed the database migration successfully this morning", 1),
    ("The authentication feature is finished and passed all tests", 1),
    ("I wrapped up the notification service implementation", 1),
    ("The sprint story for the dashboard is complete", 1),
    ("I closed the ticket for the API endpoint yesterday", 1),
    ("The user profile management feature is done and reviewed", 1),
    ("I finished the CI pipeline setup and it is running green", 1),
    ("The bug fix for the payment gateway is complete and verified", 1),

    # ── update_status (2) ─────────────────────────────────
    ("I am currently working on the payment gateway integration", 2),
    ("Today I will continue with the authentication module refactoring", 2),
    ("I started the database migration and expect to finish tomorrow", 2),
    ("The dashboard widget is in progress and about half done", 2),
    ("I am still working on the API endpoint but getting close", 2),
    ("This is taking longer than expected due to the third party issue", 2),
    ("I will pick up the notification task after my current work", 2),
    ("I'm pulling the login story into my current workload today", 2),
    ("I started working on the sprint planning items this afternoon", 2),
    ("The feature is in progress and should be done by tomorrow", 2),

    # ── assign_task (3) ───────────────────────────────────
    ("Sarah will handle the frontend authentication work", 3),
    ("Assign the payment integration story to Mike", 3),
    ("Tom can take the database migration task", 3),
    ("Let Alice do the notification system implementation", 3),
    ("The DevOps setup should go to Bob this sprint", 3),
    ("I will take the REST API endpoint work from the backlog", 3),
    ("Can you assign the dashboard story to the senior developer", 3),
    ("Mike is taking the payment gateway story for this sprint", 3),
    ("The security audit task should be assigned to a senior engineer", 3),
    ("Sarah and Tom will pair on the authentication feature together", 3),

    # ── no_action (4) ─────────────────────────────────────
    ("The overall architecture looks good to me", 4),
    ("I think the team is making solid progress this sprint", 4),
    ("The code quality has improved significantly since last month", 4),
    ("Good morning everyone how is everyone doing today", 4),
    ("The sprint is going well and we are on track", 4),
    ("I agree with the technical approach suggested by Tom", 4),
    ("The documentation is comprehensive and easy to follow", 4),
    ("Thanks everyone for a productive sprint planning session", 4),
    ("The retrospective feedback was very useful for the team", 4),
    ("I think the velocity is improving compared to last sprint", 4),
]

CLASS_NAMES = ["create_task", "complete_task", "update_status", "assign_task", "no_action"]


# ── Preprocessing (shared utilities) ─────────────────────

def tokenize(text: str) -> List[str]:
    return re.sub(r"[^\w\s]", "", text.lower()).split()


class Vocabulary:
    def __init__(self):
        self.word2idx: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word: Dict[int, str] = {0: "<PAD>", 1: "<UNK>"}

    def build(self, sentences: List[List[str]]) -> None:
        counts = Counter(word for sent in sentences for word in sent)
        for word in counts:
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx]  = word

    def encode(self, tokens: List[str]) -> List[int]:
        return [self.word2idx.get(t, 1) for t in tokens]

    def __len__(self) -> int:
        return len(self.word2idx)


def pad_sequence(seq: List[int], max_len: int) -> List[int]:
    return (seq + [0] * max_len)[:max_len]


# ── Dataset ───────────────────────────────────────────────

class ActionDataset(Dataset):
    def __init__(self, data: List[Tuple[str, int]], vocab: Vocabulary, max_len: int = 25):
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


# ── GRU Model ─────────────────────────────────────────────

class GRUClassifier(nn.Module):
    """
    GRU-based sentence classifier.

    Architecture:
        Embedding → GRU (1 layer, bidirectional) → Dropout → Linear → LogSoftmax

    Bidirectional GRU: reads the sentence left-to-right AND right-to-left,
    then concatenates both final hidden states for richer representation.

    GRU gates:
      Reset gate  — how much of past state to forget
      Update gate — how much to blend old and new state
      (simpler than LSTM's 3 gates, but often equally effective)
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        num_classes: int = 5,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(
            embed_dim,
            hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
        )
        fc_in = hidden_dim * 2 if bidirectional else hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(fc_in, num_classes)
        self.log_soft = nn.LogSoftmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)               # (batch, seq, embed_dim)
        _, h_n   = self.gru(embedded)              # h_n: (dirs, batch, hidden)
        if self.bidirectional:
            # Concatenate forward & backward final hidden states
            h_cat = torch.cat([h_n[0], h_n[1]], dim=1)   # (batch, hidden*2)
        else:
            h_cat = h_n[0]
        dropped = self.dropout(h_cat)
        logits  = self.fc(dropped)
        return self.log_soft(logits)


# ── Training ──────────────────────────────────────────────

def train(
    data: List[Tuple[str, int]],
    epochs: int = 40,
    lr: float = 1e-3,
    max_len: int = 25,
) -> Tuple[GRUClassifier, Vocabulary]:
    """Train the GRU classifier."""
    vocab = Vocabulary()
    vocab.build([tokenize(text) for text, _ in data])

    dataset    = ActionDataset(data, vocab, max_len)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    model     = GRUClassifier(vocab_size=len(vocab))
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

def predict(
    model: GRUClassifier,
    vocab: Vocabulary,
    text: str,
    max_len: int = 25,
) -> Dict:
    """Predict the scrum action type of a single sentence."""
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
        "text":          text,
        "prediction":    CLASS_NAMES[pred_idx],
        "confidence":    round(probs[pred_idx], 4),
        "probabilities": {CLASS_NAMES[i]: round(p, 4) for i, p in enumerate(probs)},
    }


# ── Standalone Demo ───────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("      UNIT 2 — GRU TEXT CLASSIFIER DEMO")
    print("=" * 65)
    print("\nArchitecture: Embedding → Bidirectional GRU → Dropout → Linear")
    print(f"Task        : Scrum action type classification")
    print(f"Classes     : {CLASS_NAMES}")
    print(f"Training set: {len(TRAINING_DATA)} samples ({len(TRAINING_DATA)//5} per class)\n")

    print("Training GRU …")
    model, vocab = train(TRAINING_DATA, epochs=50, lr=1e-3)
    print(f"\n  Vocabulary size: {len(vocab)} words")

    # ── Inference on unseen sentences ────────────────────
    print("\n" + "=" * 65)
    print("  INFERENCE ON UNSEEN SENTENCES")
    print("=" * 65)

    test_inputs = [
        "We need a ticket for the performance testing work",
        "I wrapped up the database migration this morning",
        "I am currently halfway through the sprint story",
        "Bob will take the DevOps automation task",
        "The code looks clean and well structured",
        "I'm still working on the authentication update",
        "Assign the notification feature to the junior developer",
    ]

    for text in test_inputs:
        result = predict(model, vocab, text)
        print(f"\n  Input      : {text}")
        print(f"  Predicted  : {result['prediction']}  (confidence={result['confidence']:.4f})")
        print(f"  All probs  : {result['probabilities']}")

    print("\n  GRU vs LSTM comparison:")
    print("    LSTM: 3 gates (forget, input, output) — better for long sequences")
    print("    GRU:  2 gates (reset, update)         — faster, good for short sentences")
    print("    For standup sentences (<30 words) → GRU is sufficient")
    print()
