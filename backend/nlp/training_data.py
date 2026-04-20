"""
=============================================================
ScrumPilot — NLP Training Data Loader
=============================================================
Loads LSTM and GRU training data from JSONL files in
data/training/ so data scientists can add samples without
touching Python code.

Files:
    data/training/lstm_meeting_type.jsonl   → LSTM
    data/training/gru_action_type.jsonl     → GRU

JSONL format (one JSON object per line):
    {"text": "sentence", "label": 0, "label_name": "PM_MEETING"}

Label maps
----------
LSTM (meeting type):
    0 = PM_MEETING
    1 = SPRINT_PLANNING
    2 = STANDUP

GRU (action type):
    0 = complete_task
    1 = create_task
    2 = update_status
    3 = assign_task
    4 = no_action
=============================================================
"""

import json
import os
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent.parent  # project root
_DATA = _ROOT / "data" / "training"

LSTM_DATA_FILE = _DATA / "lstm_meeting_type.jsonl"
GRU_DATA_FILE  = _DATA / "gru_action_type.jsonl"

# ── Label maps ────────────────────────────────────────────
LSTM_LABEL_MAP = {0: "PM_MEETING", 1: "SPRINT_PLANNING", 2: "STANDUP"}
GRU_LABEL_MAP  = {
    0: "complete_task",
    1: "create_task",
    2: "update_status",
    3: "assign_task",
    4: "no_action",
}
LSTM_REVERSE_MAP = {v: k for k, v in LSTM_LABEL_MAP.items()}
GRU_REVERSE_MAP  = {v: k for k, v in GRU_LABEL_MAP.items()}


# ── Loader ────────────────────────────────────────────────
def _load_jsonl(path: Path) -> list:
    """Load a JSONL file → list of (text, label) tuples."""
    data = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                obj = json.loads(line)
                data.append((obj["text"], int(obj["label"])))
            except (json.JSONDecodeError, KeyError) as e:
                print(f"  [training_data] Warning: skipping line {lineno} in {path.name}: {e}")
    return data


# ── Datasets (loaded once on import) ─────────────────────
if LSTM_DATA_FILE.exists():
    LSTM_TRAINING_DATA = _load_jsonl(LSTM_DATA_FILE)
else:
    print(f"[training_data] WARNING: {LSTM_DATA_FILE} not found — using empty dataset")
    LSTM_TRAINING_DATA = []

if GRU_DATA_FILE.exists():
    GRU_TRAINING_DATA = _load_jsonl(GRU_DATA_FILE)
else:
    print(f"[training_data] WARNING: {GRU_DATA_FILE} not found — using empty dataset")
    GRU_TRAINING_DATA = []


# ── Stats ─────────────────────────────────────────────────
def dataset_stats():
    """Print a summary of the loaded training datasets."""
    from collections import Counter

    print("\n-- LSTM training data -----------------------------")
    print(f"   File: {LSTM_DATA_FILE}")
    c = Counter(label for _, label in LSTM_TRAINING_DATA)
    for k in sorted(LSTM_LABEL_MAP):
        print(f"   {LSTM_LABEL_MAP[k]:<22} {c.get(k, 0):4d} samples")
    print(f"   {'TOTAL':<22} {sum(c.values()):4d} samples")

    print("\n-- GRU training data ------------------------------")
    print(f"   File: {GRU_DATA_FILE}")
    c = Counter(label for _, label in GRU_TRAINING_DATA)
    for k in sorted(GRU_LABEL_MAP):
        print(f"   {GRU_LABEL_MAP[k]:<22} {c.get(k, 0):4d} samples")
    print(f"   {'TOTAL':<22} {sum(c.values()):4d} samples")
    print()


if __name__ == "__main__":
    dataset_stats()
