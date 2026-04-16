"""
=============================================================
Unit 5 — Text-to-Speech (TTS)
=============================================================
Converts sprint summary text to spoken audio.

Options (ordered by quality / complexity):
  1. pyttsx3 — offline, no install required on most systems
  2. gTTS    — Google TTS, requires internet, better quality

Application in ScrumPilot:
  After the backlog pipeline runs, TTS reads the sprint summary:
  "Priority 1: Payment Gateway — WSJF score 8.5. Priority 2: …"

Syllabus: Unit 5 — TTS fundamentals; Tacotron2 and FastSpeech2 (concepts)
          (Whisper ASR already in backend/speech/whisperai/transcribe.py)
Run     : python backend/nlp/unit5_speech/tts.py
=============================================================
"""

import os
import tempfile
from typing import Optional, Literal


# ════════════════════════════════════════════════════════
# OFFLINE TTS — pyttsx3
# ════════════════════════════════════════════════════════

def speak_offline(text: str, rate: int = 150, volume: float = 1.0) -> None:
    """
    Speak text using the system TTS engine (offline, no internet needed).

    Uses pyttsx3 which wraps:
      Windows: SAPI5 (Microsoft TTS)
      macOS  : NSSpeechSynthesizer
      Linux  : espeak

    Args:
        text   : Text to speak
        rate   : Words per minute (default 150)
        volume : Volume 0.0–1.0 (default 1.0)
    """
    import pyttsx3
    engine = pyttsx3.init()
    engine.setProperty("rate",   rate)
    engine.setProperty("volume", volume)
    engine.say(text)
    engine.runAndWait()


def save_offline(text: str, output_path: str, rate: int = 150) -> str:
    """
    Save TTS audio to a file using pyttsx3.

    Returns:
        Path to the saved audio file
    """
    import pyttsx3
    engine = pyttsx3.init()
    engine.setProperty("rate", rate)
    engine.save_to_file(text, output_path)
    engine.runAndWait()
    return output_path


# ════════════════════════════════════════════════════════
# ONLINE TTS — gTTS (Google TTS)
# ════════════════════════════════════════════════════════

def speak_gtts(text: str, lang: str = "en", save_path: Optional[str] = None) -> str:
    """
    Convert text to speech using Google TTS (requires internet).

    Returns:
        Path to saved MP3 file
    """
    from gtts import gTTS
    tts = gTTS(text=text, lang=lang, slow=False)

    if save_path is None:
        save_path = os.path.join(tempfile.gettempdir(), "tts_output.mp3")

    tts.save(save_path)
    print(f"  Audio saved → {save_path}")
    return save_path


# ════════════════════════════════════════════════════════
# Scrum Sprint Summary → TTS
# ════════════════════════════════════════════════════════

def sprint_summary_to_speech(epics: list, method: Literal["offline", "gtts"] = "offline") -> str:
    """
    Generate a spoken summary of the sprint backlog priorities.

    Args:
        epics : List of Epic dicts with title, wsjf_score, priority_rank
        method: 'offline' (pyttsx3) or 'gtts' (Google TTS)

    Returns:
        The summary text that was spoken
    """
    lines = ["Sprint Backlog Priority Summary."]
    for epic in epics:
        rank  = epic.get("priority_rank", "?")
        title = epic.get("title",         "Unknown")
        score = epic.get("wsjf_score",    "?")
        lines.append(f"Priority {rank}: {title}. W S J F score: {score}.")
    lines.append("End of summary.")
    text = " ".join(lines)

    print(f"  Summary text: {text}")

    if method == "offline":
        try:
            speak_offline(text)
        except Exception as e:
            print(f"  pyttsx3 error: {e} — falling back to gtts")
            return speak_gtts(text)
    else:
        return speak_gtts(text)

    return text


# ── Standalone Demo ───────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("         UNIT 5 — TEXT-TO-SPEECH DEMO")
    print("=" * 65)
    print("\nSyllabus: Unit 5 — TTS fundamentals")
    print("Tool    : pyttsx3 (offline) + gTTS (Google, internet required)")

    # ── Demonstrate TTS conceptual architecture ───────────
    print("\n[1] TTS PIPELINE CONCEPTS")
    print("-" * 45)
    print("  Traditional TTS pipeline:")
    print("    Text → Text Analysis → Linguistic Features")
    print("    → Acoustic Model → Vocoder → Audio Waveform")
    print("")
    print("  Neural TTS models:")
    print("    Tacotron2  : Text → Mel-spectrogram → Griffin-Lim vocoder")
    print("    FastSpeech2: Parallel → Mel-spectrogram → HiFi-GAN vocoder")
    print("    (Both faster and more natural than traditional pipeline TTS)")
    print("")
    print("  Our implementation: pyttsx3 wraps OS TTS (SAPI5/espeak)")

    # ── Sprint summary TTS ────────────────────────────────
    print("\n[2] SPRINT SUMMARY → SPEECH")
    print("-" * 45)

    sample_epics = [
        {"priority_rank": 1, "title": "Payment Gateway Integration",  "wsjf_score": 8.5},
        {"priority_rank": 2, "title": "User Authentication System",   "wsjf_score": 7.2},
        {"priority_rank": 3, "title": "Analytics Dashboard",          "wsjf_score": 4.1},
    ]

    print("  Attempting offline TTS (pyttsx3) …")
    try:
        summary = sprint_summary_to_speech(sample_epics, method="offline")
        print("  ✓ Spoken successfully via pyttsx3")
    except ImportError:
        print("  pyttsx3 not installed. Run: pip install pyttsx3")
        print("  Text that would be spoken:")
        lines = ["Sprint Backlog Priority Summary."]
        for e in sample_epics:
            lines.append(f"Priority {e['priority_rank']}: {e['title']}. W S J F score: {e['wsjf_score']}.")
        print(f"  {' '.join(lines)}")

    # ── gTTS demo ─────────────────────────────────────────
    print("\n[3] gTTS — Save to MP3 File")
    print("-" * 45)
    text = "Hello, this is ScrumPilot. The payment gateway is priority one with a WSJF score of eight point five."
    print(f"  Input : {text}")
    try:
        out_path = speak_gtts(text, save_path="nlp_demos/output/tts_demo.mp3")
        print(f"  Saved : {out_path}")
        print("  Play  : open the MP3 in any media player")
    except ImportError:
        print("  gTTS not installed. Run: pip install gTTS")
    except Exception as e:
        print(f"  gTTS error (internet required?): {e}")

    # ── Multilingual TTS ──────────────────────────────────
    print("\n[4] MULTILINGUAL TTS NOTE (gTTS)")
    print("-" * 45)
    langs = [("en", "English"), ("hi", "Hindi"), ("fr", "French"), ("de", "German")]
    text_en = "Sprint planning is complete. Team is ready."
    print(f"  Same text in different languages (gTTS lang codes):")
    for code, name in langs:
        print(f"    lang='{code}' ({name})  → gTTS supports {code} synthesis")
    print("  → Aligns with Unit 5: NLP for Low-Resource Languages / Multilingual")
    print()
