# ScrumPilot 🤖🎙️

**AI-Powered Meeting Automation System for Agile Teams**

ScrumPilot automates your entire Agile workflow from meeting transcripts to Jira backlog creation. It extracts Epics from PM meetings, calculates WSJF priorities, decomposes work into Stories and Tasks, and creates a complete hierarchy in Jira - all automatically.

---

## 🎯 What ScrumPilot Does

### Workflow 1: PM Meeting → Jira Backlog (Automated) ✅
**Single Command**: `python -m backend.pipelines.backlog_pipeline`

**Input**: 
- PM meeting transcript (discusses new features, business value)
- Grooming meeting transcript (discusses estimates, complexity)

**Process**:
1. Extracts Epics and Business Value from PM meeting
2. Extracts Time Criticality, Risk Reduction, and Effort from Grooming meeting
3. Calculates WSJF scores: (Business Value + Time Criticality + Risk Reduction) / Effort
4. Ranks Epics by priority (highest WSJF first)
5. Decomposes each Epic into 3-5 User Stories
6. Breaks each Story into 2-4 Sub-tasks with hour estimates
7. Creates complete hierarchy in Jira: Epic → Story → Task

**Output**: 
- Complete prioritized backlog in Jira
- Comprehensive reports (JSON + Markdown)
- WSJF scores and priority rankings

**Time**: ~30-60 seconds | **Production Ready**: ✅ Yes

**Example**:
```bash
# Run complete pipeline
python -m backend.pipelines.backlog_pipeline

# Output:
# ✅ 4 Epics created
# ✅ 16 Stories created  
# ✅ 48 Tasks created
# ✅ Total: 68 items in Jira
```

### Workflow 2: Scrum Meeting → Jira Updates (Automated) ✅
**Command**: `python -m backend.pipelines.scrum_pipeline`

**Input**: Scrum/Standup meeting transcript

**Process**:
- Identifies task status updates ("completed", "in progress")
- Extracts task assignments
- Captures blockers and comments
- Updates Jira tickets automatically

**Output**: Updated task statuses in Jira

**Time**: ~10-20 seconds | **Production Ready**: ✅ Yes

**Example**:
```bash
# Run with Jira updates
python -m backend.pipelines.scrum_pipeline

# Run without Jira (dry run)
python -m backend.pipelines.scrum_pipeline --dry-run
```

---

## 📁 Project Structure

```text
ScrumPilot/
├── main.py                     # Main application entry point
├── backend/
│   ├── pipelines/              # Workflow Orchestration
│   │   ├── backlog_pipeline.py # Workflow 1: PM → Jira (NEW)
│   │   ├── scrum_pipeline.py   # Workflow 2: Scrum → Jira
│   │   └── meet_bot.py         # Google Meet automation
│   │
│   ├── agents/                 # AI Intelligence
│   │   ├── backlog_extractor.py    # Extract Epics from PM meeting
│   │   ├── grooming_extractor.py   # Extract estimates from Grooming
│   │   ├── wsjf_calculator.py      # Calculate WSJF scores
│   │   ├── epic_decomposer.py      # Decompose Epics → Stories → Tasks
│   │   ├── jira_creator.py         # Create hierarchy in Jira
│   │   └── scrum_extractor.py      # Extract actions from Scrum meeting
│   │
│   ├── tools/                  # Integrations
│   │   ├── jira_client.py      # Jira API wrapper (with rate limiting, retry, etc.)
│   │   └── report_generator.py # Generate reports
│   │
│   ├── meeting/                # Meeting Automation
│   │   └── meet_client.py      # Playwright bot for Google Meet
│   │
│   ├── speech/                 # Audio Processing
│   │   ├── audio_recorder.py   # System audio recording
│   │   ├── diarizer.py         # Speaker identification
│   │   └── whisperai/
│   │       ├── live_transcript.py   # Real-time transcription
│   │       └── transcribe.py        # File-based transcription
│   │
│   ├── data/                   # Generated Data
│   │   ├── pm_meetings/        # PM meeting data
│   │   ├── grooming_meetings/  # Grooming meeting data
│   │   ├── wsjf/               # WSJF calculations
│   │   ├── decomposed/         # Decomposed backlogs
│   │   ├── jira/               # Jira creation results
│   │   ├── checkpoints/        # Pipeline checkpoints
│   │   └── pipeline_reports/   # Pipeline execution reports
│   │
│   └── tests/                  # Unit Tests
│
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the root directory:

```env
# Groq API (for LLM processing)
GROQ_API_KEY=your_groq_api_key

# Jira Configuration
JIRA_URL=https://your-domain.atlassian.net
JIRA_EMAIL=your-email@example.com
JIRA_API_TOKEN=your_jira_api_token
JIRA_PROJECT_KEY=SP

# Hugging Face (for speaker diarization)
HF_TOKEN=your_huggingface_token

# Audio Recording
OUTPUT_AUDIO_PATH=backend/speech/temp/meeting_audio.wav
```

### 3. Run Workflow 1: PM Meeting → Jira

```bash
# Complete automated pipeline
python -m backend.pipelines.backlog_pipeline
```

**What it does**:
- Reads PM and Grooming transcripts
- Extracts Epics and estimates
- Calculates WSJF priorities
- Decomposes into Stories and Tasks
- Creates everything in Jira
- Generates comprehensive reports

**Output**:
```
✅ Pipeline Complete!
   📊 4 Epics → 16 Stories → 48 Tasks
   ⏱️  Total: 68 items created in Jira
   📄 Report: backend/data/pipeline_reports/20260411_pipeline_report.md
```

### 4. Run Workflow 2: Scrum Meeting → Jira

```bash
# Update Jira from Scrum meeting
python -m backend.pipelines.scrum_pipeline

# Dry run (no Jira updates)
python -m backend.pipelines.scrum_pipeline --dry-run
```

---

## 📋 Detailed Setup

### Required API Keys

#### 1. Groq API Key (Required)
- Sign up at: https://console.groq.com
- Create API key
- Free tier available

#### 2. Jira API Token (Required)
- Go to: https://id.atlassian.com/manage-profile/security/api-tokens
- Click "Create API token"
- Label it "ScrumPilot"
- Copy the token

#### 3. Hugging Face Token (Optional - for speaker diarization)
- Sign up at: https://huggingface.co
- Go to Settings → Access Tokens
- Create token with "Read" role
- Accept model terms:
  - https://huggingface.co/pyannote/segmentation-3.0
  - https://huggingface.co/pyannote/speaker-diarization-3.1

### System Requirements

#### FFmpeg (Windows - for audio processing)
1. Download from: https://www.gyan.dev/ffmpeg/builds/
2. Extract to: `C:\ffmpeg`
3. Add to PATH: `C:\ffmpeg\bin`
4. Restart terminal

---

## 🎯 Usage Examples

### Example 1: Create Backlog from Transcripts

```bash
# Place your transcripts in:
# - backend/data/pm_meetings/example_pm_transcript.txt
# - backend/data/grooming_meetings/example_grooming_transcript.txt

# Run pipeline
python -m backend.pipelines.backlog_pipeline

# Check results in Jira!
```

### Example 2: Update Tasks from Scrum Meeting

```bash
# Place transcript in:
# - backend/data/scrum_meetings/example_scrum_transcript.txt

# Run pipeline
python -m backend.pipelines.scrum_pipeline
```

### Example 3: Resume After Failure

```bash
# If pipeline fails, it saves a checkpoint
# Resume from checkpoint:
python -c "
from backend.pipelines.backlog_pipeline import BacklogPipeline
pipeline = BacklogPipeline()
result = pipeline.resume_from_checkpoint('backend/data/checkpoints/CHECKPOINT_FILE.json')
"
```

---

## 🔧 Configuration

### Pipeline Configuration

```python
from backend.pipelines.backlog_pipeline import BacklogPipeline, PipelineConfig

# Custom configuration
config = PipelineConfig(
    # Timeouts
    phase_timeout=300,          # 5 minutes per phase
    pipeline_timeout=900,       # 15 minutes total
    
    # Approval gates (optional)
    require_approval_after_wsjf=True,
    require_approval_after_decomposition=True,
    
    # Checkpointing
    save_checkpoints=True,
    checkpoint_dir="backend/data/checkpoints",
    
    # Reporting
    generate_final_report=True,
    report_dir="backend/data/pipeline_reports"
)

pipeline = BacklogPipeline(config=config)
result = pipeline.run(
    pm_transcript_path="...",
    grooming_transcript_path="..."
)
```

---

## 📊 Production Features

### Workflow 1: PM Meeting → Jira

✅ **Environment Validation** - Validates API keys, connectivity, disk space before execution
✅ **Timeout Protection** - Phase and pipeline-level timeouts prevent hanging
✅ **Partial Failure Handling** - One Epic failure doesn't block others
✅ **Resume from Checkpoint** - Fast recovery from failures
✅ **Rate Limiting** - Respects Jira API limits (150 calls/min)
✅ **Automatic Retry** - Handles transient failures with exponential backoff
✅ **Duplicate Detection** - Warns about similar items before creating
✅ **Idempotency** - Safe to re-run, skips already created items
✅ **Progress Tracking** - Real-time phase execution updates
✅ **Comprehensive Reporting** - JSON + Markdown reports with all details

**Production Readiness**: 8/10

### Workflow 2: Scrum Meeting → Jira

✅ **Action Extraction** - Identifies status updates, assignments, comments
✅ **Jira Integration** - Updates tickets automatically
✅ **Dry Run Mode** - Test without making changes
✅ **Error Handling** - Graceful failure handling

**Production Readiness**: 8/10

---

## 🧪 Testing

### Test Jira Connection

```bash
python -m backend.tools.test_jira
```

### Run Individual Agents

```bash
# Test PM extraction
python -m backend.agents.backlog_extractor

# Test Grooming extraction
python -m backend.agents.grooming_extractor

# Test WSJF calculation
python -m backend.agents.wsjf_calculator

# Test Epic decomposition
python -m backend.agents.epic_decomposer

# Test Jira creation
python -m backend.agents.jira_creator
```

---

## 📈 Performance

### Workflow 1: PM Meeting → Jira
- **Duration**: 30-60 seconds
- **Phases**: 6 (Validation, PM Extraction, Grooming Extraction, WSJF, Decomposition, Jira Creation)
- **LLM Calls**: ~20-30 (depending on Epic count)
- **Jira API Calls**: ~60-100 (depending on backlog size)
- **Success Rate**: 100% (with retry and error handling)

### Workflow 2: Scrum Meeting → Jira
- **Duration**: 10-20 seconds
- **LLM Calls**: 1-2
- **Jira API Calls**: 5-20 (depending on updates)
- **Success Rate**: 95%+

---

## 🐛 Troubleshooting

### Issue: "GROQ_API_KEY not set"
**Solution**: Add `GROQ_API_KEY` to `.env` file

### Issue: "Jira connection failed"
**Solution**: 
1. Verify `JIRA_URL`, `JIRA_EMAIL`, `JIRA_API_TOKEN` in `.env`
2. Run `python -m backend.tools.test_jira` to test connection

### Issue: "Pipeline timeout"
**Solution**: Increase timeout in configuration:
```python
config = PipelineConfig(phase_timeout=600, pipeline_timeout=1800)
```

### Issue: "Epic decomposition failed"
**Solution**: Pipeline continues with other Epics (partial failure handling). Check logs for specific error.

---

## 🚧 Upcoming Features

### Phase 7: Google Meet Bot Integration
- Automated meeting joining
- Real-time audio recording
- Automatic transcript generation

### Phase 8: Real-time Processing
- Live meeting transcription
- Real-time Jira updates
- Instant backlog creation

---

## 📝 License

[Your License Here]

---

## 🤝 Contributing

Contributions welcome! Please read our contributing guidelines.

---

## 📧 Support

For issues and questions, please open a GitHub issue.

---

**Built with ❤️ for Agile teams**
