"""
=============================================================
Unit 1 — Named Entity Recognition (NER)
=============================================================
Uses spaCy's pre-trained English NER model to identify:
  PERSON   — speaker names, task assignees
  ORG      — team names, companies, tools
  DATE     — sprint dates, deadlines
  CARDINAL — numeric estimates (hours, story points)
  PRODUCT  — feature/software names

Application in ScrumPilot:
  → Extracts assignee names from standup transcripts
  → Extracts story-point estimates from grooming sessions
  → Extracts sprint dates from planning meetings

Syllabus: Unit 1 — Lemmatization, POS tagging and NER: concepts and tagsets
Run     : python backend/nlp/unit1_preprocessing/ner.py
=============================================================
"""

from typing import List, Dict

import spacy

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess, sys
    print("Downloading spaCy model en_core_web_sm …")
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
    nlp = spacy.load("en_core_web_sm")


# ── Label descriptions ────────────────────────────────────

LABEL_INFO = {
    "PERSON":   "Person / assignee name",
    "ORG":      "Organization / team / tool",
    "DATE":     "Date or time expression",
    "CARDINAL": "Numeric value (hours, points, count)",
    "PRODUCT":  "Product or software feature",
    "GPE":      "Geopolitical entity (country, city)",
    "MONEY":    "Monetary value",
    "PERCENT":  "Percentage",
    "TIME":     "Time expression",
    "ORDINAL":  "Ordinal number (first, second …)",
    "NORP":     "Nationality, religious or political group",
}


# ── Core functions ────────────────────────────────────────

def extract_entities(text: str) -> List[Dict]:
    """
    Run spaCy NER on text and return all identified entities.

    Returns:
        List of dicts: {text, label, start_char, end_char, description}
    """
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        entities.append({
            "text":        ent.text,
            "label":       ent.label_,
            "start":       ent.start_char,
            "end":         ent.end_char,
            "description": LABEL_INFO.get(ent.label_, ent.label_),
        })
    return entities


def extract_by_label(text: str, label: str) -> List[str]:
    """Extract all entities of a given NER label from text."""
    doc = nlp(text)
    return [ent.text for ent in doc.ents if ent.label_ == label]


def extract_assignees(text: str) -> List[str]:
    """
    Extract PERSON entities → likely task assignees in standup transcripts.
    Example: 'Sarah will handle the API' → ['Sarah']
    """
    return extract_by_label(text, "PERSON")


def extract_estimates(text: str) -> List[str]:
    """
    Extract CARDINAL entities → numeric estimates (story points, hours).
    Example: 'estimated 8 story points' → ['8']
    """
    return extract_by_label(text, "CARDINAL")


def extract_dates(text: str) -> List[str]:
    """
    Extract DATE/TIME entities → sprint dates and deadlines.
    Example: 'complete by Friday April 25th' → ['Friday', 'April 25th']
    """
    dates = extract_by_label(text, "DATE")
    dates += extract_by_label(text, "TIME")
    return dates


def extract_organizations(text: str) -> List[str]:
    """Extract ORG entities → team names, tools, companies."""
    return extract_by_label(text, "ORG")


# ── Standalone Demo ───────────────────────────────────────

if __name__ == "__main__":
    STANDUP = (
        "Sarah finished the login API yesterday and estimated 3 story points remaining. "
        "Mike is working on the Stripe payment gateway integration — he expects to complete "
        "it by Friday. The DevOps team will set up the AWS infrastructure by next Monday. "
        "We have a sprint review on April 25th at 2 PM. "
        "Sujay from the backend team will handle the database migration, "
        "estimated at 5 hours."
    )

    print("=" * 65)
    print("       UNIT 1 — NAMED ENTITY RECOGNITION DEMO")
    print("=" * 65)
    print(f"\nInput:\n{STANDUP}\n")

    entities = extract_entities(STANDUP)

    print(f"{'Entity Text':<28} {'Label':<12} {'Description'}")
    print("-" * 65)
    for e in entities:
        print(f"  {e['text']:<26} {e['label']:<12} {e['description']}")

    print("\nExtracted by category:")
    print(f"  PERSON   (assignees) : {extract_assignees(STANDUP)}")
    print(f"  CARDINAL (estimates) : {extract_estimates(STANDUP)}")
    print(f"  DATE/TIME            : {extract_dates(STANDUP)}")
    print(f"  ORG      (teams)     : {extract_organizations(STANDUP)}")

    # ── Sprint planning sentence ──────────────────────────
    print("\nSprint Planning NER example:")
    planning = (
        "Our sprint goal is to deliver the payment feature by April 30th. "
        "Alice will take the frontend work, Bob handles the backend API — "
        "together they have 80 hours of capacity this sprint."
    )
    print(f"  Input: {planning}")
    ents = extract_entities(planning)
    for e in ents:
        print(f"  [{e['label']}] {e['text']!r}")

    # ── Backlog grooming sentence ─────────────────────────
    print("\nGrooming session NER example:")
    grooming = (
        "The authentication epic has a business value of 9 and "
        "time criticality of 8. The team at Acme Corp estimated 5 effort points."
    )
    print(f"  Input: {grooming}")
    for e in extract_entities(grooming):
        print(f"  [{e['label']}] {e['text']!r}")

    print()
