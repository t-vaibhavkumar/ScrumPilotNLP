# ScrumPilot 🤖🎙️

ScrumPilot is an automated AI Scrum Master designed to join Google Meet sessions, record system audio, identify individual speakers (diarization), and automatically generate Jira tasks based on the meeting transcript.

---

## 📁 Project Structure

```text
ScrumPilot/
├── backend/
│   ├── speech/                 # Audio capture & Processing
│   │   ├── meet_bot.py         # Main: Joins Meet & records system audio
│   │   ├── diarizer.py         # Task 1: Identifies WHO spoke (Pyannote)
│   │   ├── diarized_transcription.py # Master script: Combines Who + What
│   │   └── whisperai/
│   │       ├── live_transcript.py   # Real-time transcription (faster-whisper)
│   │       └── transcribe.py        # File-based transcription (openai-whisper)
│   │
│   ├── agents/                 # AI Intelligence (Scrum Agent logic)
│   │
│   ├── tools/                  # Integrations & Scripts
│   │   ├── jira_client.py      # Jira API wrapper functions
│   │   └── test_jira.py        # Diagnostic tool for connection
│   │
│   └── .env                    # Local credentials (git-ignored)
├── requirements.txt
└── README.md
```


# 🚀 Setup & Usage

## 1. Environment Setup

Install the dependencies and set up the browser automation tools:

```bash
pip install -r requirements.txt
```

## ❗ System Requirement: FFmpeg (Windows)

The speaker diarization system requires **FFmpeg DLLs** to be available in your system path.

### Steps

1. Download the **"full-shared" build** from:  
   https://www.gyan.dev/ffmpeg/builds/

2. Extract the archive to:

```text
C:\ffmpeg
```

3. Add the following directory to System Environment Variables → Path

```text
C:\ffmpeg\bin
```

4. Restart VS Code and your terminal.

## 2. Hugging Face Access (Required for Diarization)

The pyannote diarization models are gated. If access is not granted you will receive a 403 Forbidden error.

### Step 1 — Accept Model Terms

Visit the following pages while logged into Hugging Face and click "Agree and access repository".

https://huggingface.co/pyannote/segmentation-3.0

https://huggingface.co/pyannote/speaker-diarization-3.1

https://huggingface.co/pyannote/speaker-diarization-community-1

You may be asked to provide name and university details.

### Step 2 — Create a Hugging Face Token

1. Go to Hugging Face → Settings → Tokens
2. Click Create New Token
3. Select Read role
4. Copy the generated token

## 3. Configuration (.env)

Create a `.env` file in the root directory. Use the following guide to retrieve your credentials. **Never share your API tokens.**

| Variable | Where to find it |
|---|---|
| **JIRA_URL** | The base domain in your browser address bar. Example: `https://yourname.atlassian.net` (Do not include paths like `/projects/...`). |
| **JIRA_EMAIL** | The email address you use to log in to your Atlassian/Jira account. |
| **JIRA_API_TOKEN** | Go to **Atlassian API Tokens**. Click **Create API token**, label it `ScrumPilot`, and copy the secret. |
| **JIRA_PROJECT_KEY** | Look at any task on your board (e.g., `KAN-1`). The letters before the hyphen (`KAN`) are your project key.  In our project rn it is KAN|

### Template

```env
# JIRA CONFIG
JIRA_URL=https://your-domain.atlassian.net
JIRA_EMAIL=your-email@example.com
JIRA_API_TOKEN=your_jira_api_token
JIRA_PROJECT_KEY=KAN

HF_TOKEN=your_huggingface_read_token
OUTPUT_AUDIO_PATH=backend/speech/temp/meeting_audio.wav
```

## 4. Execution Workflow
### Step 1 — Record the Meeting

Run the meeting bot:
```bash
python backend/speech/meet_bot.py
```
The bot will:
- Automatically join the Google Meet
- Record system audio
- Save a .wav recording once the browser closes

### Step 2 — Generate Diarized Transcript

Run:
```bash
python backend/speech/diarized_transcription.py
```
This script will:
- Load the recorded audio
- Run speaker diarization using Pyannote
- Transcribe speech using Whisper AI
- Produce a structured transcript file:
```text
meeting_transcript.txt
```
### Step 3 — Verify Jira Connection

Before running automation, verify that Jira credentials work:
```bash
python backend/tools/test_jira.py
```
This confirms:
- Jira API authentication
- Project access
- Permission to create issues
