# ScrumPilot NLP 🚀

An end-to-end NLP-powered system that converts Scrum meeting transcripts into actionable Jira updates.

---

## 📌 Overview

ScrumPilot NLP processes meeting transcripts and automatically:

- ✅ Creates new Jira tasks  
- ✅ Updates task status (To Do → In Progress → Done)  
- ✅ Assigns tasks to team members  
- ✅ Adds comments to existing tickets  

⚡ Fully local NLP pipeline — **no LLM APIs required**

---

## 🧠 Architecture
Transcript
↓
NLP Pipeline (spaCy + DistilBERT)
↓
Structured Actions (JSON)
↓
Jira Executor (Deterministic)
↓
Jira Updates

---

## ⚙️ Features

- 🔍 Intent Classification (DistilBERT)
- 🧩 Entity Extraction (spaCy + rules)
- 🔁 Context-aware sentence merging
- 🎯 Jira integration (REST API)
- 📊 Evaluation system (Precision, Recall, F1)

---

## 📁 Project Structure
```env
backend/
├── agents/
│ ├── nlp_scrum_extractor.py
│ ├── jira_executor.py
│ └── intent_classifier.py
├── tools/
│ ├── jira_client.py
│ └── bootstrap_jira.py
├── config/
│ └── team_config.py
├── pipelines/
│ └── scrum_pipeline_nlp.py
├── evaluation/
│ ├── run_evaluation.py
│ ├── evaluation_transcript_.txt
│ └── evaluation_transcript_.json
```
---

## 🔧 Setup

### 1. Install dependencies

Or using pip:
```bash
pip install -r requirements_nlp.txt
```

### 2. Install spaCy model
```bash
python -m spacy download en_core_web_sm
```

### 3. Setup Jira credentials
Create `.env` file in root:

```env
JIRA_BASE_URL=https://your-domain.atlassian.net
JIRA_EMAIL=your_email@example.com
JIRA_API_TOKEN=your_api_token
```


---

## 🚀 Usage

🔹 **Step 1** — Bootstrap Jira (Create Base Tasks)
```bash
uv run python -m backend.tools.bootstrap_jira
```

🔹 **Step 2** — Run NLP Pipeline
```bash
uv run python backend/pipelines/scrum_pipeline_nlp.py
```

🔹 **Step 3** — Run Evaluation
```bash
uv run python -m backend.evaluation.run_evaluation
```

---

## 📊 Evaluation Metrics
```env
The system evaluates:
🎯 Precision
🔍 Recall
⚖️ F1 Score
```

**Example:**
```env
Precision: 0.78
Recall: 0.85
F1 Score: 0.81
```

---

## 🧪 Example Input
"Yesterday I completed the login UI. Today I will work on dashboard."


**Output:**
```json
[
  {"action": "complete_task", "summary": "login UI"},
  {"action": "update_status", "summary": "dashboard"}
]
```

---

## ⚠️ Notes
- Requires existing Jira project
- Assignment works only with valid Jira users
- Uses heuristic matching for ticket resolution

---

## 🧠 Key Design Decisions
- No LLM usage → fully offline + fast
- Rule-based + ML hybrid approach
- Deterministic Jira execution layer

---

## 🔮 Future Improvements
- Better coreference resolution ("it", "that")
- Fuzzy matching using embeddings
- UI dashboard
- Real-time meeting integration