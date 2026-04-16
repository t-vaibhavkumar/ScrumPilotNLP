"""
=============================================================
Demo — Unit 5: Speech (ASR + TTS)
=============================================================
Demonstrates the full Speech NLP pipeline:
  ASR: Audio → Text  (Whisper — already in the project)
  TTS: Text  → Audio (pyttsx3 / gTTS)

Run: python nlp_demos/demo_unit5_speech.py
=============================================================
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.nlp.unit5_speech.tts import speak_offline, speak_gtts, sprint_summary_to_speech


SPRINT_SUMMARY_EPICS = [
    {"priority_rank": 1, "title": "Payment Gateway Integration",  "wsjf_score": 8.5},
    {"priority_rank": 2, "title": "User Authentication System",   "wsjf_score": 7.2},
    {"priority_rank": 3, "title": "Analytics Dashboard",          "wsjf_score": 4.1},
    {"priority_rank": 4, "title": "Notification System",          "wsjf_score": 3.8},
]


def section(title: str) -> None:
    print(f"\n{'─' * 65}")
    print(f"  {title}")
    print("─" * 65)


if __name__ == "__main__":
    os.makedirs("nlp_demos/output", exist_ok=True)

    print("╔══════════════════════════════════════════════════════════╗")
    print("║     UNIT 5 — SPEECH (ASR + TTS) DEMO                   ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # ── ASR: Whisper (existing) ───────────────────────────
    section("1. ASR — Automatic Speech Recognition  (OpenAI Whisper)")
    print("""
  Whisper is a neural ASR model (encoder-decoder Transformer):
    • Encoder: processes mel-spectrogram frames of audio
    • Decoder: generates text tokens autoregressively

  Already integrated in: backend/speech/whisperai/transcribe.py

  Usage:
    from backend.speech.whisperai.transcribe import transcribe_audio
    text = transcribe_audio("meeting.mp3")

  Models available (by size/speed tradeoff):
    tiny   → fastest, CPU-friendly, lower accuracy
    base   → good balance for short meetings
    small  → better accuracy, still reasonable on CPU
    medium → high accuracy (recommended for production)
    large  → best accuracy, GPU recommended

  This is the SPEECH → TEXT step of the NLP pipeline.
    """)

    # ── TTS: Text → Speech ───────────────────────────────
    section("2. TTS — Text-to-Speech  (pyttsx3 / gTTS)")

    demo_text = (
        "Good morning team. This is the ScrumPilot NLP sprint summary. "
        "Priority one is the Payment Gateway Integration with a WSJF score of eight point five. "
        "Priority two is the User Authentication System with a score of seven point two. "
        "The sprint ends on April 28th. Team capacity is 80 hours."
    )

    print(f"\n  Text to be spoken:\n  '{demo_text}'\n")

    # Offline TTS
    print("  [OFFLINE TTS — pyttsx3]")
    try:
        speak_offline(demo_text, rate=145)
        print("  ✓ Spoken via system TTS engine")
    except ImportError:
        print("  ✗ pyttsx3 not installed. Run: pip install pyttsx3")
    except Exception as e:
        print(f"  ✗ pyttsx3 error: {e}")

    # Save to MP3 (gTTS)
    print("\n  [ONLINE TTS — gTTS (saves to MP3)]")
    try:
        out_path = speak_gtts(demo_text, save_path="nlp_demos/output/sprint_summary.mp3")
        print(f"  ✓ Saved to: {out_path}")
    except ImportError:
        print("  ✗ gTTS not installed. Run: pip install gTTS")
    except Exception as e:
        print(f"  ✗ gTTS error (internet required): {e}")

    # ── Full Sprint Summary ───────────────────────────────
    section("3. SPRINT BACKLOG → SPEECH  (End-to-End)")
    print("  Backlog epics:")
    for epic in SPRINT_SUMMARY_EPICS:
        print(f"    Rank {epic['priority_rank']}: {epic['title']} (WSJF={epic['wsjf_score']})")
    print("\n  Generating spoken summary …")
    sprint_summary_to_speech(SPRINT_SUMMARY_EPICS, method="offline")

    # ── Pipeline overview ─────────────────────────────────
    section("4. FULL NLP PIPELINE  (End-to-End Overview)")
    print("""
  SPEECH RECOGNITION (Unit 5)
    Audio file (.mp3/.wav) → Whisper ASR → Raw transcript text

  PREPROCESSING (Unit 1)
    Raw text → Normalize → Tokenize → Lemmatize → NER

  REPRESENTATIONS (Unit 1)
    Tokens → BoW / TF-IDF / Word2Vec embeddings

  CLASSIFICATION (Unit 2–3)
    Text → LSTM / GRU / BERT → Meeting type / Action type

  SEMANTIC SEARCH (Unit 3)
    Sentence → SBERT embedding → Cosine similarity → Story ID

  GENERATION / SUMMARIZATION (Unit 4)
    Transcript → BART → Meeting minutes
    Epic title  → T5   → Epic description

  TEXT-TO-SPEECH (Unit 5)
    Summary text → pyttsx3 / gTTS → Audio output
    """)

    print("✓ Unit 5 Speech demo complete!\n")
