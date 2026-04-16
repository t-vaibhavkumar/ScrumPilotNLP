"""
=============================================================
Unit 3 — CNN for Text Classification
=============================================================
1D Convolutional Neural Network for meeting type classification.
Uses multiple kernel sizes to capture n-gram patterns.

Architecture:
  Embedding(vocab_size, 64)
  → [Conv1D(128, k=3), Conv1D(128, k=4), Conv1D(128, k=5)]  (parallel)
  → MaxPool over time (extract dominant feature per filter)
  → Concat(3 × 128 = 384)
  → Dropout(0.3) → Linear(384, 3) → LogSoftmax

Why multiple kernel sizes?
  k=3: trigram patterns ("sprint planning meeting")
  k=4: 4-gram patterns  ("completed the login feature")
  k=5: 5-gram patterns  ("I am blocked on the database")

CNNs vs RNNs for text:
  CNN: parallel computation, fast, captures local n-gram features
  RNN/LSTM: sequential, handles long-range dependencies better
  For short sentences → CNN often matches LSTM performance, much faster.

Syllabus: Unit 2/3 — CNNs for text, CNN for Text Classification
Run     : python backend/nlp/unit3_transformers/cnn_text_classifier.py
=============================================================
"""

import re
from typing import List, Tuple, Dict
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# ── Training data (same as LSTM — enables model comparison) ──

TRAINING_DATA: List[Tuple[str, int]] = [
    # PM_MEETING = 0
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
    # SPRINT_PLANNING = 1
    ("Our sprint goal is to complete the payment gateway integration", 1),
    ("Team capacity this sprint is eighty hours across five developers", 1),
    ("Let's commit to the authentication story and the dashboard epic", 1),
    ("Sarah will take the frontend work and Mike handles the backend API", 1),
    ("We are pulling in four stories totaling thirty-four story points", 1),
    ("The sprint starts Monday and ends Friday in two weeks", 1),
    ("Bob is on leave for three days so total capacity is reduced", 1),
    ("Let us pull SP-187 and SP-188 into this sprint", 1),
    ("We need to break this story into subtasks before we can estimate", 1),
    ("Previous sprint velocity was thirty eight so let's target that", 1),
    ("The definition of done requires unit tests and code review approval", 1),
    ("Risk identified: we depend on DevOps team for the webhook setup", 1),
    ("Alice can take the notification task and Tom handles deployment", 1),
    ("We commit to delivering the login feature by end of this sprint", 1),
    ("Sprint planning estimates show we have capacity for three stories", 1),
    # STANDUP = 2
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
    def __init__(self):
        self.word2idx: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1}

    def build(self, sentences):
        counts = Counter(w for s in sentences for w in s)
        for w in counts:
            if w not in self.word2idx:
                self.word2idx[w] = len(self.word2idx)

    def encode(self, tokens, max_len=30):
        ids = [self.word2idx.get(t, 1) for t in tokens]
        return (ids + [0] * max_len)[:max_len]

    def __len__(self):
        return len(self.word2idx)


class TextDataset(Dataset):
    def __init__(self, data, vocab, max_len=30):
        self.samples = []
        for text, label in data:
            tokens = tokenize(text)
            ids    = vocab.encode(tokens, max_len)
            self.samples.append((torch.tensor(ids, dtype=torch.long), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ── CNN Model ─────────────────────────────────────────────

class TextCNN(nn.Module):
    """
    TextCNN (Kim 2014) for text classification.

    Parallel convolutional filters with kernel sizes [3, 4, 5]
    act as n-gram detectors. Max-over-time pooling extracts the
    most dominant feature from each filter.

    This architecture is widely used for short-text classification.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        num_filters: int = 128,
        kernel_sizes: List[int] = [3, 4, 5],
        num_classes: int = 3,
        dropout: float = 0.3,
        max_len: int = 30,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Parallel convolutional layers — one per kernel size
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels  = embed_dim,
                out_channels = num_filters,
                kernel_size  = k,
            )
            for k in kernel_sizes
        ])

        self.dropout = nn.Dropout(dropout)
        # After max-pool: num_filters features per kernel size
        self.fc      = nn.Linear(num_filters * len(kernel_sizes), num_classes)
        self.log_soft = nn.LogSoftmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len)
        embedded = self.embedding(x)          # (batch, seq_len, embed_dim)
        embedded = embedded.transpose(1, 2)   # (batch, embed_dim, seq_len)  for Conv1D

        pooled_outputs = []
        for conv in self.convs:
            # Conv1D → (batch, num_filters, seq_len - k + 1)
            conv_out = F.relu(conv(embedded))
            # Max-over-time pooling → (batch, num_filters, 1)
            pooled  = F.max_pool1d(conv_out, kernel_size=conv_out.size(2))
            pooled_outputs.append(pooled.squeeze(2))    # (batch, num_filters)

        # Concatenate features from all kernel sizes → (batch, num_filters * n_kernels)
        cat     = torch.cat(pooled_outputs, dim=1)
        dropped = self.dropout(cat)
        logits  = self.fc(dropped)
        return self.log_soft(logits)


# ── Training ──────────────────────────────────────────────

def train(
    data: List[Tuple[str, int]],
    epochs: int = 40,
    lr: float = 1e-3,
    max_len: int = 30,
) -> Tuple[TextCNN, Vocabulary]:
    """Train the CNN classifier."""
    vocab = Vocabulary()
    vocab.build([tokenize(text) for text, _ in data])

    dataset    = TextDataset(data, vocab, max_len)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    model     = TextCNN(vocab_size=len(vocab))
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

def predict(model: TextCNN, vocab: Vocabulary, text: str, max_len: int = 30) -> Dict:
    model.eval()
    tokens  = tokenize(text)
    ids     = vocab.encode(tokens, max_len)
    tensor  = torch.tensor([ids], dtype=torch.long)
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
    print("       UNIT 3 — CNN TEXT CLASSIFIER DEMO")
    print("=" * 65)
    print("\nArchitecture: Embedding → Conv1D(k=3,4,5) → MaxPool → Linear")
    print(f"Task        : Meeting type ({', '.join(CLASS_NAMES)})")
    print(f"Kernel sizes: [3, 4, 5]  — captures trigram, 4-gram, 5-gram patterns")
    print(f"Training set: {len(TRAINING_DATA)} samples\n")

    print("Training CNN …")
    cnn_model, vocab = train(TRAINING_DATA, epochs=50, lr=1e-3)
    print(f"\n  Vocab size        : {len(vocab)}")
    total_params = sum(p.numel() for p in cnn_model.parameters())
    print(f"  Total parameters  : {total_params:,}")

    # ── Test inference ────────────────────────────────────
    print("\n" + "=" * 65)
    print("  INFERENCE ")
    print("=" * 65)
    test_inputs = [
        "The CEO wants the new features shipped before the product launch",
        "Today I finished the API endpoint and will start writing tests",
        "We have sixty hours of capacity and will take three user stories",
        "I am blocked on the infrastructure setup can someone help",
        "Business value for the payment epic is critical for revenue",
    ]
    for text in test_inputs:
        result = predict(cnn_model, vocab, text)
        print(f"\n  Input     : {text}")
        print(f"  Predicted : {result['prediction']}  (conf={result['confidence']:.4f})")
        print(f"  All probs : {result['probabilities']}")

    # ── CNN vs LSTM architecture comparison ───────────────
    print("\n" + "=" * 65)
    print("  CNN vs LSTM vs GRU — Architecture Comparison")
    print("=" * 65)
    print(f"  {'Model':<12} {'Computation':<15} {'Long-range':<15} {'Best for'}")
    print("  " + "-" * 60)
    print(f"  {'CNN':<12} {'Parallel':<15} {'Local n-grams':<15} short sentences, speed")
    print(f"  {'LSTM':<12} {'Sequential':<15} {'Long sequences':<15} long documents, context")
    print(f"  {'GRU':<12} {'Sequential':<15} {'Medium range':<15} short-medium sentences")
    print(f"  {'BERT':<12} {'Parallel':<15} {'Full sequence':<15} best quality, slowest")
    print()
