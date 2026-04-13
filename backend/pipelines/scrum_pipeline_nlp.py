"""
scrum_pipeline.py — NLP-only Scrum Pipeline.

Wires NLPScrumExtractor (spaCy + BART zero-shot) and JiraExecutor
(deterministic routing) to replace the previous LLM-based pipeline.

No LangChain. No Groq API. No cloud calls.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.agents.nlp_scrum_extractor import NLPScrumExtractor
from backend.agents.jira_executor import JiraExecutor


def run_scrum_pipeline(transcript: str, dry_run: bool = False) -> dict:
    """
    Full pipeline: transcript → extract actions → execute on Jira.

    Args:
        transcript: The diarized meeting transcript text.
        dry_run:    If True, only extract actions without executing on Jira.

    Returns:
        dict with keys:
            actions  — list of ScrumAction dicts
            report   — str summary (None in dry_run mode)
    """
    print("=" * 50)
    print("SCRUM PIPELINE (NLP) — Starting")
    print("=" * 50)

    # ── Step 1: NLP extraction ────────────────────────────────────────────
    print("\n🔍 Step 1: Extracting scrum actions via NLP pipeline...")
    extractor = NLPScrumExtractor()
    actions = extractor.extract_actions(transcript)

    print(f"   Found {len(actions)} action(s):")
    for i, a in enumerate(actions, 1):
        conf_str = f"  (conf={a.get('confidence', 0):.2f})"
        flag = " [LOW CONF]" if a.get("low_confidence") else ""
        print(f"   {i}. [{a.get('action')}]{flag} {a.get('summary', 'N/A')!r}{conf_str}")

    if dry_run:
        print("\n⏸️  Dry-run mode — skipping Jira execution.")
        print("\n📋 Extracted actions:")
        print(json.dumps(actions, indent=2))
        return {"actions": actions, "report": None}

    # ── Step 2: Deterministic Jira execution ──────────────────────────────
    print("\n🚀 Step 2: Executing actions on Jira (deterministic routing)...")
    executor = JiraExecutor()
    report = executor.execute_actions(actions)

    print("\n📋 Execution Report:")
    print(report)

    print("\n" + "=" * 50)
    print("SCRUM PIPELINE (NLP) — Complete")
    print("=" * 50)

    return {"actions": actions, "report": report}


def run_from_transcript_file(path: str, dry_run: bool = False) -> dict:
    """Convenience: read a transcript file and run the pipeline."""
    with open(path, "r", encoding="utf-8") as f:
        transcript = f.read()
    return run_scrum_pipeline(transcript, dry_run=dry_run)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    here = os.path.dirname(__file__)
    default_path = os.path.join(here, "..", "agents", "example_transcript.txt")

    positional_args = [a for a in sys.argv[1:] if not a.startswith("--")]
    transcript_path = positional_args[0] if positional_args else default_path
    dry = "--dry-run" in sys.argv or "--dry" in sys.argv

    result = run_from_transcript_file(transcript_path, dry_run=dry)

    if "--json" in sys.argv:
        print(json.dumps(result, indent=2, default=str))