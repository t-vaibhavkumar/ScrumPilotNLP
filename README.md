# ScrumPilot — AI-Powered Scrum Automation

> **End-to-end Agile meeting intelligence:** Transcripts in → Jira actions out, with human approval via Telegram.

ScrumPilot listens to your Scrum meetings (standup, sprint planning, PM meetings), understands them using a custom NLP pipeline (LSTM + GRU + SBERT + DistilBART), and automatically creates/updates Jira tickets — all with a human-in-the-loop Telegram approval step.

---

## ✅ End-to-End Status

| Component | Status | Notes |
|---|---|---|
| LSTM meeting classifier | ✅ Ready | 88% val accuracy, 300 Agile samples |
| GRU action classifier | ✅ Ready | 91% accuracy, 5 action types |
| SBERT story matching | ✅ Ready | Semantic matching to Jira tickets |
| DistilBART summarizer | ✅ Ready | Abstractive meeting summary |
| Whisper ASR | ✅ Ready | Voice → transcript |
| NLP pipeline orchestrator | ✅ Ready | All 4 units connected |
| Context loader (PostgreSQL) | ✅ Ready | Loads active sprint from DB |
| Telegram bot | ✅ Ready | `/meeting`, `/done`, `/sprint`, `/approvals` |
| Jira integration | ✅ Ready | Create/update/transition tickets |
| PostgreSQL persistence | ✅ Ready | Epics, stories, tasks, sprints |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  INPUT                                                           │
│  Voice (Whisper ASR)  OR  Text via Telegram /meeting            │
└───────────────────────────────┬──────────────────────────────────┘
                                │ transcript
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│  NLP PIPELINE  (pipeline_orchestrator.py)                        │
│                                                                  │
│  Unit 1 ─ Preprocessing                                         │
│    • Sentence tokenizer, normalizer                              │
│    • Named Entity Recognition (spaCy): assignees, dates          │
│    • TF-IDF extractive summarizer                                │
│                                                                  │
│  Unit 2 ─ Custom Trained Models                                  │
│    • LSTM  → meeting type   (STANDUP / SPRINT_PLANNING / PM)     │
│    • GRU   → action type    (complete / create / assign /        │
│                               update_status / no_action)         │
│                                                                  │
│  Unit 3 ─ Sentence-BERT (all-MiniLM-L6-v2)                      │
│    • Semantic similarity: action sentence → closest Jira story   │
│    • Corpus seeded from PostgreSQL sprint stories (or Jira API)  │
│    • DB actor resolution: "Alice" → SP-003 via SprintAssignment  │
│                                                                  │
│  Unit 4 ─ DistilBART                                             │
│    • Abstractive meeting summary (80 tokens)                     │
└───────────────────────────────┬──────────────────────────────────┘
                                │ structured actions + summary
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│  CONTEXT LAYER  (context_loader.py)                              │
│                                                                  │
│  PostgreSQL (active sprint) ──→  SprintContext                   │
│    • story_titles[], story_keys[]  →  SBERT corpus               │
│    • assignments{}  →  actor → Jira key resolution               │
│    • capacity_hours, velocity_target  →  sprint payload          │
│    • epic_titles[]  →  PM meeting epic enrichment                │
│                                                                  │
│  Fallback chain:  DB → Jira API → hardcoded defaults             │
└───────────────────────────────┬──────────────────────────────────┘
                                │ approval payload
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│  HUMAN-IN-THE-LOOP  (Telegram Bot)                               │
│                                                                  │
│  Bot sends formatted summary to PM/Scrum Master:                 │
│    • Meeting type + confidence                                   │
│    • Extracted actions with matched Jira tickets                 │
│    • Blockers detected                                           │
│    • [✅ Approve]  [❌ Reject]  inline buttons                   │
└───────────────────────────────┬──────────────────────────────────┘
                                │ approval
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│  JIRA INTEGRATION  (JiraAgent)                                   │
│                                                                  │
│  Standup       → update_status, complete_task, assign_task       │
│  Sprint Plan   → create sprint, move stories, set assignments    │
│  PM Meeting    → create epics (WSJF scored), decompose→stories   │
└──────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│  PostgreSQL  ←──────────────  sync Jira keys back to DB          │
│  Sprints, SprintStories, Epics, Stories, Tasks, ScrumActions     │
└──────────────────────────────────────────────────────────────────┘
```
---

---

## Quick Start

### 1. Clone & Install

```powershell
git clone https://github.com/your-username/ScrumPilotNLP.git
cd ScrumPilotNLP

python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -r nlp_requirements.txt
```

### 2. Configure Environment

```powershell
copy .env.example .env
# Edit .env with your credentials
```

`.env` required keys:

```env
# PostgreSQL
DATABASE_URL=postgresql://postgres:yourpassword@localhost:5432/scrumpilot

# Jira (https://id.atlassian.com/manage-profile/security/api-tokens)
JIRA_URL=https://your-domain.atlassian.net
JIRA_EMAIL=your-email@example.com
JIRA_API_TOKEN=your_jira_api_token
JIRA_PROJECT_KEY=SP

# Telegram (@BotFather)
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id

# Groq (https://console.groq.com) — for LLM-based pipelines
GROQ_API_KEY=your_groq_api_key
```

### 3. Set Up Database

```powershell
# Create the database
psql -U postgres -c "CREATE DATABASE scrumpilot;"

# Run migrations
alembic upgrade head
```

### 4. Train NLP Models (one-time, ~30 seconds)

```powershell
python backend\nlp\train_models.py
```

Expected output:
```
LSTM: 300 samples | 3 classes
GRU : 190 samples | 5 classes

--- Training LSTM (meeting type) ---
  Train: 225 | Val: 75
  Gradient check | embed grad norm = 0.010xxx  OK
  Epoch  10/200  train=97.3%  val=82.7%
  Early stop at epoch 34 — best val=88.0% @ ep9

--- Training GRU (action type) ---
  Early stop at epoch 19 — acc=93.7%

=== Smoke Test ===
  LSTM: 6/6 = 100%
  GRU : 8/8 = 100%

Done. Run the bot — models load from disk instantly.
```

Models saved to `backend/nlp/models/` — subsequent bot starts load instantly from disk.

### 5. Start the Bot

```powershell
python -m backend.telegram.bot
```

---

## Demo: End-to-End Walkthrough

### Demo 1 — Daily Standup

**In Telegram:**
```
/meeting
```
*Bot replies: "Recording started. Send your meeting transcript or use /done to process."*

```
Alice: Yesterday I completed the Stripe payment gateway integration, no blockers.
Tom: Still working on the CI/CD pipeline. Blocked on Docker credentials from DevOps.
Bob: I finished the OAuth login feature and merged to main this morning.
```

```
/done
```

**What happens internally:**
1. LSTM classifies → `STANDUP` (conf=0.99)
2. GRU classifies each sentence:
   - Alice's update → `complete_task`
   - Tom's update → `update_status` 
   - Bob's update → `complete_task`
3. SBERT/DB matches each action to a Jira story (e.g. `SP-003`, `SP-007`)
4. Blocker detected: "Blocked on Docker credentials"
5. DistilBART generates summary

**Bot sends approval to Telegram:**
```
📋 STANDUP  (confidence: 99%)

✅ complete_task  │ SP-003  │ Alice
   Stripe payment gateway integration

🔄 update_status  │ SP-007  │ Tom
   CI/CD pipeline setup

✅ complete_task  │ SP-001  │ Bob
   OAuth login feature merged to main

⚠️ BLOCKER: Tom — Docker credentials from DevOps

[✅ Approve & Execute]   [❌ Reject]
```

**On Approve:** Jira transitions `SP-003` → Done, `SP-007` → In Progress, logs blocker.

---

### Demo 2 — Sprint Planning

```
We're committing to sprint 14 starting Monday.
Alice will take the authentication story for thirty points.
Tom is handling the CI/CD pipeline setup at twenty points.
We have eighty hours total capacity this sprint.
```

**Bot sends:**
```
📅 SPRINT PLANNING

Sprint 14  |  Capacity: 80h  |  Target velocity: 50pts

Committed stories:
  SP-001  User Authentication  │ Alice  │ 30pts
  SP-007  CI/CD Pipeline       │ Tom    │ 20pts

[✅ Create Sprint & Assign]   [❌ Reject]
```

---

### Demo 3 — PM Meeting (Epic Creation)

```
The CEO approved the GDPR compliance epic. Business value is nine, 
time criticality is eight. The sales team needs CRM integration by end of month.
```

**Bot sends:**
```
🏆 PM MEETING — Epic Proposals

[1] GDPR Compliance Epic
    WSJF: 9.0  │  BV:9  TC:8  RR:5
    
[2] CRM Integration  
    WSJF: 6.4  │  BV:8  TC:7  RR:5

[✅ Create Epics in Jira]   [❌ Reject]
```

---

## NLP Model Details

| Model | Task | Training Data | Accuracy |
|---|---|---|---|
| **LSTM** | Meeting type classification | 300 Agile meeting sentences (3 classes) | 88% val |
| **GRU** | Action type classification | 190 Agile action phrases (5 classes) | 91% train |
| **SBERT** `all-MiniLM-L6-v2` | Semantic story matching | Pretrained (general) | — |
| **DistilBART** `distilbart-cnn-12-6` | Meeting summarization | Pretrained (general) | — |
| **Whisper** | Speech → text | Pretrained, OpenAI | — |

**Training data location:** `data/training/lstm_meeting_type.jsonl`, `data/training/gru_action_type.jsonl`

**Retrain anytime:**
```powershell
# Delete existing models and retrain
del backend\nlp\models\*.pt backend\nlp\models\*.vocab
python backend\nlp\train_models.py
```

**Extend training data:** Add lines to the `.jsonl` files — no code changes needed:
```json
{"text": "I finished the GraphQL migration", "label": 2, "label_name": "STANDUP"}
```

---

## Context-Aware Pipeline

Once a sprint is created in PostgreSQL, the pipeline automatically becomes context-aware:

```
Next orchestrator init
  └─ context_loader.py queries PostgreSQL
       └─ Active sprint found
            ├─ story_titles[] + story_keys[] → SBERT corpus (replaces Jira API)
            ├─ assignments{} → "alice" → "SP-003" (direct, no SBERT needed)
            ├─ capacity_hours + velocity_target → sprint planning payload
            └─ epic_titles[] → PM meeting enrichment
```

Test your context:
```powershell
python backend\nlp\context_loader.py
```

---

## Project Structure

```
ScrumPilotNLP/
├── backend/
│   ├── nlp/
│   │   ├── unit1_preprocessing/   # Tokenizer, NER, normalizer
│   │   ├── unit1_representations/ # TF-IDF
│   │   ├── unit2_models/          # LSTM + GRU classifiers
│   │   ├── unit3_transformers/    # Sentence-BERT, BERT embeddings
│   │   ├── unit4_applications/    # DistilBART summarizer
│   │   ├── unit5_speech/          # Whisper ASR
│   │   ├── models/                # Saved .pt + .vocab files (git-ignored)
│   │   ├── pipeline_orchestrator.py   # Main NLP entry point
│   │   ├── context_loader.py          # PostgreSQL sprint context
│   │   ├── jira_action_mapper.py      # NLP output → Jira actions
│   │   ├── train_models.py            # Standalone training script
│   │   └── training_data.py           # In-memory training data loader
│   ├── agents/                   # LLM agents (Groq/Llama)
│   ├── db/                       # SQLAlchemy models + CRUD
│   ├── pipelines/                # Backlog, sprint, standup pipelines
│   ├── telegram/                 # Bot handlers
│   └── tools/                    # Jira client
├── data/
│   ├── training/                 # JSONL training datasets
│   │   ├── lstm_meeting_type.jsonl   # 300 samples, 3 classes
│   │   └── gru_action_type.jsonl     # 190 samples, 5 classes
│   └── transcripts/              # Sample meeting transcripts
├── alembic/                      # DB migrations
├── .env.example
├── requirements.txt
├── nlp_requirements.txt
└── README.md
```

---

## Bot Commands

| Command | Description |
|---|---|
| `/start` | Register and link your Telegram account |
| `/meeting` | Start recording a meeting transcript |
| `/done` | Process the transcript through the full NLP pipeline |
| `/approvals` | List pending approval requests |
| `/sprint` | Show current sprint overview |
| `/status` | Show your task statuses |
| `/team` | Show team member assignments |
| `/help` | Command reference |

---

## Standalone Pipelines (no bot required)

```powershell
# Backlog pipeline — PM meeting → epics → Jira
python run_backlog_pipeline.py

# Sprint planning — transcript → sprint + stories in Jira
python run_sprint_planning_pipeline.py

# Standup — transcript → task status updates in Jira
python run_standup_pipeline.py
```

---

## Troubleshooting

| Error | Solution |
|---|---|
| `ModuleNotFoundError: No module named 'backend'` | Run from project root, not from a subdirectory |
| `No sprint found in DB` | Expected on first run — pipeline falls back to Jira API |
| `LSTM val_acc stuck at 36%` | max_len=35 causes gradient vanishing — fixed in current config |
| `JiraError HTTP 404` | Ticket key doesn't exist in your Jira project; check `JIRA_PROJECT_KEY` |
| `Database connection failed` | Check `DATABASE_URL` in `.env` and that PostgreSQL is running |
| `Telegram bot not responding` | Verify `TELEGRAM_BOT_TOKEN` and run `python -m backend.telegram.bot` |

---

## Key Design Decisions

- **Models trained from scratch** on Agile-domain data (not fine-tuned general models) — LSTM and GRU understand Scrum vocabulary natively
- **Train once, load from disk** — models persist to `backend/nlp/models/*.pt`; bot starts in <1s on subsequent runs
- **Graceful fallback chain** — PostgreSQL → Jira API → hardcoded defaults; system always has a story corpus
- **Human-in-the-loop always** — no Jira action executes without Telegram approval
- **Separation of concerns** — training data in `.jsonl` files, editable by non-developers

---

## License

[Your License Here]

---

**Version:** 2.0.0 — Context-Aware NLP Pipeline
