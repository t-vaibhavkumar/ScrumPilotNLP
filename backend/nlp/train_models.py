"""
ScrumPilot — Model Training Script
Trains LSTM + GRU classifiers on Agile-domain data from
data/training/*.jsonl and saves weights to backend/nlp/models/

Target: LSTM ~88-92%, GRU ~88-92%  (avoid overfitting AND underfitting)

Run: python backend\nlp\train_models.py
  OR: python -m backend.nlp.train_models
"""

import os
import sys
import time
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Add project root so 'backend' package is importable
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# ── Wipe old saved models ─────────────────────────────────
for fname in ["lstm_meeting_type.pt", "lstm_meeting_type.vocab",
              "gru_action_type.pt",   "gru_action_type.vocab"]:
    p = os.path.join(MODELS_DIR, fname)
    if os.path.exists(p):
        os.remove(p)
        print(f"  Removed old: {fname}")

print("\n=== ScrumPilot Model Training ===\n")

from backend.nlp.training_data import (
    LSTM_TRAINING_DATA, GRU_TRAINING_DATA,
    LSTM_LABEL_MAP, GRU_LABEL_MAP,
)

print(f"LSTM: {len(LSTM_TRAINING_DATA)} samples | {len(LSTM_LABEL_MAP)} classes")
print(f"GRU : {len(GRU_TRAINING_DATA)} samples | {len(GRU_LABEL_MAP)} classes\n")


# ── Shared utilities ──────────────────────────────────────
def _save(model, vocab, name):
    torch.save(model.state_dict(), os.path.join(MODELS_DIR, f"{name}.pt"))
    with open(os.path.join(MODELS_DIR, f"{name}.vocab"), "wb") as f:
        pickle.dump(vocab, f)
    print(f"Saved -> backend/nlp/models/{name}.pt + .vocab")


# ═══════════════════════════════════════════════════════════
#  LSTM TRAINING (meeting type: 3 classes)
#  Target accuracy: 85-92%
# ═══════════════════════════════════════════════════════════
print("--- Training LSTM (meeting type) ---")

from backend.nlp.unit2_models.lstm_classifier import (
    LSTMClassifier, Vocabulary, MeetingDataset, tokenize as lstm_tok
)

# Proper ML approach: train on 80%, validate on 20%.
# Report BOTH train_acc and val_acc — divergence = overfitting.
# Early stop when val_acc reaches target (patient stop).
LSTM_HP = dict(
    epochs       = 200,
    lr           = 5e-4,      # lower LR — 1e-3 overfit too fast on 225 samples
    weight_decay = 5e-4,      # less L2 — more data provides its own regularisation
    batch_size   = 8,
    embed_dim    = 128,       # KEY: vocab grew 484→617, embed_dim=64 was underpowered
    num_layers   = 1,
    dropout      = 0.3,
    max_len      = 12,
    val_split    = 0.25,     # 25% val = 52 samples → ±1.9% steps (more stable than 42)
    patience     = 25,        # more patience for slower LR convergence
)

import random
random.seed(42)
torch.manual_seed(42)          # deterministic weights + dropout
data_shuffled = LSTM_TRAINING_DATA[:]
random.shuffle(data_shuffled)
val_size   = int(len(data_shuffled) * LSTM_HP["val_split"])
train_data = data_shuffled[val_size:]
val_data   = data_shuffled[:val_size]
print(f"  Train: {len(train_data)} | Val: {len(val_data)}")

t0 = time.time()
lstm_vocab = Vocabulary()
lstm_vocab.build([lstm_tok(text) for text, _ in train_data])  # vocab from TRAIN only

train_dataset = MeetingDataset(train_data, lstm_vocab, max_len=LSTM_HP["max_len"])
val_dataset   = MeetingDataset(val_data,   lstm_vocab, max_len=LSTM_HP["max_len"])
train_loader  = DataLoader(train_dataset, batch_size=LSTM_HP["batch_size"], shuffle=True)
val_loader    = DataLoader(val_dataset,   batch_size=LSTM_HP["batch_size"], shuffle=False)

lstm_model = LSTMClassifier(
    vocab_size  = len(lstm_vocab),
    embed_dim   = LSTM_HP["embed_dim"],   # 128 — matches expanded 617-word vocabulary
    num_classes = len(LSTM_LABEL_MAP),
    num_layers  = LSTM_HP["num_layers"],
    dropout     = LSTM_HP["dropout"],
)
lstm_criterion = nn.NLLLoss()
lstm_optimizer = optim.Adam(
    lstm_model.parameters(),
    lr           = LSTM_HP["lr"],
    weight_decay = LSTM_HP["weight_decay"],
)
lstm_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    lstm_optimizer, mode="max", factor=0.5, patience=10, min_lr=1e-5
)

# Gradient sanity check
lstm_model.train()
_s, _l = next(iter(train_loader))
_o = lstm_model(_s); _loss = lstm_criterion(_o, _l); _loss.backward()
_eg = lstm_model.embedding.weight.grad.norm().item()
lstm_model.zero_grad()
print(f"  Gradient check | embed grad norm = {_eg:.6f}  {'OK' if _eg > 1e-4 else 'WARN'}")

def _eval(model, loader, criterion):
    model.eval()
    correct, total, tloss = 0, 0, 0.0
    with torch.no_grad():
        for seqs, labels in loader:
            labels = labels if isinstance(labels, torch.Tensor) else torch.tensor(labels, dtype=torch.long)
            out = model(seqs)
            tloss  += criterion(out, labels).item()
            correct += (out.argmax(1) == labels).sum().item()
            total   += len(labels)
    return correct / total * 100, tloss / len(loader)

best_val_acc  = 0.0
best_epoch    = 0
no_improve    = 0
best_state    = None

for epoch in range(1, LSTM_HP["epochs"] + 1):
    lstm_model.train()
    correct, total, tloss = 0, 0, 0.0
    for seqs, labels in train_loader:
        labels = labels if isinstance(labels, torch.Tensor) else torch.tensor(labels, dtype=torch.long)
        lstm_optimizer.zero_grad()
        out  = lstm_model(seqs)
        loss = lstm_criterion(out, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(lstm_model.parameters(), max_norm=3.0)
        lstm_optimizer.step()
        tloss   += loss.item()
        correct += (out.argmax(1) == labels).sum().item()
        total   += len(labels)

    train_acc = correct / total * 100
    val_acc, _ = _eval(lstm_model, val_loader, lstm_criterion)
    lstm_scheduler.step(val_acc)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch   = epoch
        no_improve   = 0
        best_state   = {k: v.clone() for k, v in lstm_model.state_dict().items()}
    else:
        no_improve += 1

    if epoch % 10 == 0:
        lr_now = lstm_optimizer.param_groups[0]["lr"]
        print(f"  Epoch {epoch:3d}/{LSTM_HP['epochs']}  "
              f"train={train_acc:.1f}%  val={val_acc:.1f}%  lr={lr_now:.1e}")

    if no_improve >= LSTM_HP["patience"]:
        print(f"  Early stop at epoch {epoch} (patience={LSTM_HP['patience']}) — best val={best_val_acc:.1f}% @ ep{best_epoch}")
        break

# Restore best weights
if best_state:
    lstm_model.load_state_dict(best_state)

print(f"\nLSTM done in {time.time()-t0:.1f}s | best val_acc={best_val_acc:.1f}% @ epoch {best_epoch} | vocab={len(lstm_vocab)}")
_save(lstm_model, lstm_vocab, "lstm_meeting_type")





# ═══════════════════════════════════════════════════════════
#  GRU TRAINING (action type: 5 classes)
#  Target accuracy: 85-92%
# ═══════════════════════════════════════════════════════════
print("\n--- Training GRU (action type) ---")

from backend.nlp.unit2_models.gru_classifier import (
    GRUClassifier, Vocabulary as GRUVocabulary, ActionDataset, tokenize as gru_tok
)

# Hyperparameters tuned to prevent overfitting (was hitting 100%)
GRU_HP = dict(
    epochs       = 60,
    lr           = 3e-4,
    batch_size   = 8,
    dropout      = 0.5,     # stronger dropout → harder to memorise
    weight_decay = 1e-3,    # stronger L2 regularisation
    lr_step      = 20,
    lr_gamma     = 0.6,
    max_len      = 25,
    target_acc   = 91.0,    # stop early at this
    floor_acc    = 60.0,    # only start checking target after this floor
)

t0 = time.time()
gru_vocab = GRUVocabulary()
gru_vocab.build([gru_tok(text) for text, _ in GRU_TRAINING_DATA])

gru_dataset = ActionDataset(GRU_TRAINING_DATA, gru_vocab, max_len=GRU_HP["max_len"])
gru_loader  = DataLoader(gru_dataset, batch_size=GRU_HP["batch_size"], shuffle=True)

gru_model = GRUClassifier(
    vocab_size  = len(gru_vocab),
    num_classes = len(GRU_LABEL_MAP),
    dropout     = GRU_HP["dropout"],
)
gru_criterion = nn.NLLLoss()
gru_optimizer = optim.Adam(
    gru_model.parameters(),
    lr           = GRU_HP["lr"],
    weight_decay = GRU_HP["weight_decay"],
)
gru_scheduler = optim.lr_scheduler.StepLR(
    gru_optimizer,
    step_size = GRU_HP["lr_step"],
    gamma     = GRU_HP["lr_gamma"],
)

best_gru_acc = 0.0
for epoch in range(1, GRU_HP["epochs"] + 1):
    gru_model.train()
    total_loss, correct, total = 0.0, 0, 0
    for seqs, labels in gru_loader:
        labels = labels if isinstance(labels, torch.Tensor) else torch.tensor(labels, dtype=torch.long)
        gru_optimizer.zero_grad()
        out  = gru_model(seqs)
        loss = gru_criterion(out, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(gru_model.parameters(), max_norm=1.0)
        gru_optimizer.step()
        total_loss += loss.item()
        preds   = out.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += len(labels)
    gru_scheduler.step()

    acc = correct / total * 100
    if acc > best_gru_acc:
        best_gru_acc = acc

    if epoch % 10 == 0:
        lr_now = gru_optimizer.param_groups[0]["lr"]
        print(f"  Epoch {epoch:3d}/{GRU_HP['epochs']}  "
              f"loss={total_loss/len(gru_loader):.4f}  "
              f"acc={acc:.1f}%  lr={lr_now:.2e}")

    # Only trigger early stop once model has learned enough
    if acc >= GRU_HP["target_acc"] and acc > GRU_HP["floor_acc"]:
        print(f"  Early stop at epoch {epoch} — acc={acc:.1f}% >= {GRU_HP['target_acc']}%")
        break

print(f"\nGRU done in {time.time()-t0:.1f}s | best acc={best_gru_acc:.1f}% | vocab={len(gru_vocab)}")
_save(gru_model, gru_vocab, "gru_action_type")


# ═══════════════════════════════════════════════════════════
#  Smoke Test
# ═══════════════════════════════════════════════════════════
print("\n=== Smoke Test ===\n")
from backend.nlp.unit2_models.lstm_classifier import predict as lstm_predict
from backend.nlp.unit2_models.gru_classifier  import predict as gru_predict

lstm_model.eval(); gru_model.eval()

LSTM_TESTS = [
    ("Yesterday I completed the payment integration no blockers today",        "STANDUP"),
    ("We commit to thirty story points this sprint Alice takes the auth story", "SPRINT_PLANNING"),
    ("The CEO approved the GDPR epic business value is nine time criticality",  "PM_MEETING"),
    ("I am blocked on the AWS credentials from DevOps no update",               "STANDUP"),
    ("Let us pull in the analytics story for sprint fourteen capacity allows",   "SPRINT_PLANNING"),
    ("The product roadmap for Q3 focuses on retention epics with high WSJF",    "PM_MEETING"),
]

print("LSTM (meeting type):")
correct, total = 0, 0
for text, expected in LSTM_TESTS:
    r  = lstm_predict(lstm_model, lstm_vocab, text)
    ok = r["prediction"] == expected
    correct += ok
    status = "OK   " if ok else "WRONG"
    print(f"  [{status}] {r['prediction']:>16}  conf={r['confidence']:.2f}  | {text[:52]}")
print(f"  Test accuracy: {correct}/{total if total else len(LSTM_TESTS)} "
      f"= {correct/len(LSTM_TESTS)*100:.0f}%")

GRU_TESTS = [
    ("I finished the Stripe payment gateway integration",            "complete_task"),
    ("We need to create a ticket for the failing integration tests", "create_task"),
    ("Tom will take the authentication story this sprint",           "assign_task"),
    ("Still working on the CI pipeline setup blocked on Docker",     "update_status"),
    ("Good morning team let us keep this standup short today",       "no_action"),
    ("I completed the OAuth login feature and merged to main",       "complete_task"),
    ("Someone should open a bug for the checkout session timeout",   "create_task"),
    ("Alice is the lead on the payment redesign project",            "assign_task"),
]

print("\nGRU (action type):")
correct = 0
for text, expected in GRU_TESTS:
    r  = gru_predict(gru_model, gru_vocab, text)
    ok = r["prediction"] == expected
    correct += ok
    status = "OK   " if ok else "WRONG"
    print(f"  [{status}] {r['prediction']:>15}  conf={r['confidence']:.2f}  | {text[:52]}")
print(f"  Test accuracy: {correct}/{len(GRU_TESTS)} = {correct/len(GRU_TESTS)*100:.0f}%")

# File sizes
print("\nSaved files:")
for fname in sorted(os.listdir(MODELS_DIR)):
    size = os.path.getsize(os.path.join(MODELS_DIR, fname))
    print(f"  {fname:<40}  {size/1024:.1f} KB")

print("\nDone. Run the bot — models load from disk instantly.")
