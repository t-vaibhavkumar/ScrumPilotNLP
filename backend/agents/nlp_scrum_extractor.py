"""
nlp_scrum_extractor.py — Pure NLP/ML replacement for ScrumExtractorAgent.

Pipeline per sentence:
  1. parse_transcript()      — split diarized text into (speaker, text) pairs
  2. split_sentences()       — tokenise each turn into individual sentences
  3. merge_context()         — merge only genuine continuations (called ONCE)
  4. filter_actionable()     — drop greetings, filler, too-short sentences
  5. classify_intent()       — fine-tuned DistilBERT (falls back to BART)
  6. extract_entities()      — spaCy NER + regex rules
  7. build_action()          — assemble ScrumAction dataclass
  8. postprocess_actions()   — deduplicate, merge description sentences

No LLM API calls. No cloud. Runs fully offline.
"""

from __future__ import annotations

import re
import json
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict
from backend.config.team_config import TEAM_EMAIL_MAP

# ── Lazy model holders ────────────────────────────────────────────────────────
_nlp = None


def _get_spacy():
    global _nlp
    if _nlp is None:
        import spacy
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise RuntimeError(
                "spaCy model 'en_core_web_sm' not found. "
                "Run: python -m spacy download en_core_web_sm"
            )
    return _nlp


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class ScrumAction:
    action: str
    summary: str
    description: Optional[str] = None
    assignee: Optional[str] = None
    assignee_email: Optional[str] = None
    status: Optional[str] = None
    ticket_key: Optional[str] = None
    comment: Optional[str] = None
    confidence: float = 0.0
    low_confidence: bool = False
    source_speaker: Optional[str] = None
    source_sentence: Optional[str] = None
    deadline_change: Optional[str] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


# ── Step 1: Transcript parsing ────────────────────────────────────────────────

def parse_transcript(transcript: str) -> List[Dict[str, str]]:
    results = []
    pattern = re.compile(r"^(?:\[(.+?)\]|(.+?)):\s*(.+)$")
    for raw_line in transcript.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        m = pattern.match(line)
        if m:
            speaker = (m.group(1) or m.group(2)).strip()
            text = m.group(3).strip()
            results.append({"speaker": speaker, "text": text})
    return results


# ── Step 2: Sentence splitting ────────────────────────────────────────────────

def split_sentences(parsed_data: List[Dict[str, str]]) -> List[Dict[str, str]]:
    nlp = _get_spacy()
    results = []
    for entry in parsed_data:
        speaker = entry["speaker"]
        doc = nlp(entry["text"])
        for sent in doc.sents:
            text = sent.text.strip()
            if text:
                results.append({"speaker": speaker, "sentence": text})
    return results


# ── Step 3: Context merge (called ONCE) ───────────────────────────────────────

_WEAK_PRONOUNS = re.compile(r"^(it|that|this)\b", re.IGNORECASE)

_ACTION_VERBS_RE = re.compile(
    r"\b(fix|create|assign|complete|finish|update|add|comment|pick|start|"
    r"working|handle|need|should|mark|close|deploy|merge|review|also)\b",
    re.IGNORECASE,
)


def merge_context(sentences: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Merge a sentence into the previous one ONLY when it is a genuine
    pronoun-continuation with no independent action verb.

    Conservative: only merges bare pronoun fragments under 6 words.
    "Also, I think we need to create..." is NOT merged — it has action verbs.
    """
    merged = []
    prev = None

    for item in sentences:
        text = item["sentence"].strip()
        speaker = item["speaker"]
        word_count = len(text.split())

        is_continuation = (
            prev is not None
            and prev["speaker"] == speaker
            and word_count < 6
            and _WEAK_PRONOUNS.match(text)
            and not _ACTION_VERBS_RE.search(text)
        )

        if is_continuation:
            prev["sentence"] += " " + text
        else:
            if prev:
                merged.append(prev)
            prev = item.copy()

    if prev:
        merged.append(prev)

    return merged


# ── Step 4: Actionability filter ──────────────────────────────────────────────

def filter_actionable(sentences: List[Dict[str, str]]) -> List[Dict[str, str]]:
    from backend.config.team_config import FILTER_PHRASES, MIN_SENTENCE_TOKENS

    actionable = []
    for item in sentences:
        text = item["sentence"].strip().lower().rstrip(".")

        if len(text.split()) < MIN_SENTENCE_TOKENS:
            continue

        if any(text == fp or text.startswith(fp) for fp in FILTER_PHRASES):
            continue

        actionable.append(item)

    return actionable


# ── Step 5: Intent classification ─────────────────────────────────────────────

def classify_intent(sentences: List[Dict]) -> List[Dict]:
    from backend.agents.intent_classifier import ScrumIntentClassifier
    classifier = ScrumIntentClassifier()
    return classifier.predict(sentences)


# ── Step 6: Entity extraction ─────────────────────────────────────────────────

_TICKET_KEY_RE = re.compile(r"\b([A-Z][A-Z0-9]+-\d+)\b")


def _extract_ticket_key(text: str) -> Optional[str]:
    m = _TICKET_KEY_RE.search(text)
    return m.group(1) if m else None


# Completion words used for low-confidence override only
_DONE_OVERRIDE_RE = re.compile(
    r"\b(fixed|resolved|merged|deployed|shipped|closed|completed|finished)\b",
    re.IGNORECASE,
)


def _extract_status(text: str) -> Optional[str]:
    from backend.config.team_config import STATUS_PHRASE_MAP
    lower = text.lower()
    for canonical, phrases in STATUS_PHRASE_MAP.items():
        for phrase in phrases:
            if phrase in lower:
                return canonical
    return None


_FIRST_PERSON_RE = re.compile(
    r"\b(i|i've|i'm|i'll|i finished|i completed|i started|i will)\b",
    re.IGNORECASE,
)


def _extract_assignee(text: str, speaker: str, intent: str) -> Optional[str]:
    """
    Returns assignee display name. Email resolution is separate in build_action()
    so we never silently drop a valid assignee due to a missing email config.
    """
    nlp = _get_spacy()
    doc = nlp(text)

    persons = [
        ent.text for ent in doc.ents
        if ent.label_ == "PERSON" and ent.text.lower() in TEAM_EMAIL_MAP
    ]

    self_report_intents = {"complete_task", "update_status"}

    if persons:
        for p in persons:
            if p.lower() != speaker.lower():
                return p

    if intent in self_report_intents and _FIRST_PERSON_RE.search(text):
        return speaker

    if intent == "assign_task":
        if persons:
            return persons[0]
        return speaker

    return None


_DESCRIPTION_RE = re.compile(
    r'(?:description\s+(?:should\s+be|is)|described?\s+as)\s*["\']?(.+?)["\']?\s*$',
    re.IGNORECASE,
)


def _extract_description(text: str) -> Optional[str]:
    m = _DESCRIPTION_RE.search(text)
    if m:
        return m.group(1).strip().rstrip(".")
    return None


_DEADLINE_RE = re.compile(
    r"(deadline.*(extend|push)|won't be able to finish.*by|delay(ed)?|pushed to\s+\w+)",
    re.IGNORECASE,
)


def _extract_deadline(text: str) -> Optional[str]:
    if _DEADLINE_RE.search(text):
        return "Deadline Extended"
    return None


_NOISE_WORDS_RE = re.compile(
    r"\b(yesterday|today|last night|this morning|next|now|also|just|already)\b",
    re.IGNORECASE,
)

_TASK_KEYWORDS = {
    "task", "module", "page", "service", "migration", "integration",
    "pipeline", "script", "test", "tests", "fix", "bug", "feature",
    "setup", "configuration", "deployment", "authentication", "schema",
    "profile", "gateway", "ci", "cd", "cicd", "api", "ui", "flow",
    "dashboard", "component", "endpoint", "middleware", "cache", "job",
    "report", "search", "notification", "analytics", "export", "import",
}

_SUMMARY_PATTERNS = [
    re.compile(
        r"(?:create|open|add|log)\s+(?:a\s+)?(?:new\s+)?(?:task|story|ticket|issue|bug|spike)\s+"
        r"(?:for\s+|to\s+|about\s+)?(.+?)(?:\.|,|$)",
        re.IGNORECASE,
    ),
    re.compile(
        r"need\s+(?:a\s+)?(?:task|story|ticket)\s+(?:for\s+)?(.+?)(?:\.|,|$)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:working\s+on|pick(?:ing)?\s+up|taking\s+on|handle|handling|start(?:ing)?\s+on?)\s+"
        r"(?:the\s+)?(.+?)(?:\.|,|$)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:completed|finished|done\s+(?:with)?|resolved|closed(?:\s+out)?|shipped|deployed|merged)\s+"
        r"(?:the\s+)?(.+?)(?:\.|,|$)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:fixing|fixed)\s+(?:the\s+)?(.+?)(?:\.|,|$)",
        re.IGNORECASE,
    ),
    re.compile(
        r"mark\s+(?:the\s+)?(.+?)\s+as\s+(?:done|complete|finished|closed)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:take\s+ownership\s+of|assign(?:ed)?|own)\s+(?:the\s+)?(.+?)(?:\.|,|\?|$)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:still\s+)?(?:in\s+progress\s+on|working\s+on)\s+(?:the\s+)?(.+?)(?:\.|,|$)",
        re.IGNORECASE,
    ),
]

_TRAILING_NOISE_RE = re.compile(
    r"\s+(task|ticket|issue|and|that|it|this|since.*|as\s+done.*)\s*$",
    re.IGNORECASE,
)


def _clean_candidate(text: str) -> str:
    text = _NOISE_WORDS_RE.sub("", text)
    text = _TRAILING_NOISE_RE.sub("", text)
    return text.strip().rstrip(".,?!")


def _title_case_summary(text: str) -> str:
    if not text:
        return text
    return text[0].upper() + text[1:]


def _extract_summary(text: str, speaker: str, intent: str) -> str:
    nlp = _get_spacy()

    # Strategy 1: regex patterns
    for pat in _SUMMARY_PATTERNS:
        m = pat.search(text)
        if m:
            candidate = _clean_candidate(m.group(1).strip())
            if len(candidate) > 3:
                return _title_case_summary(candidate)

    # Strategy 2: spaCy noun chunks with task keywords
    doc = nlp(text)
    best_chunk = ""
    best_len = 0
    for chunk in doc.noun_chunks:
        chunk_lower = chunk.text.lower()
        has_keyword = any(kw in chunk_lower for kw in _TASK_KEYWORDS)
        if has_keyword and len(chunk.text) > best_len:
            best_chunk = chunk.text.strip()
            best_len = len(chunk.text)

    if best_chunk:
        candidate = _clean_candidate(best_chunk)
        if len(candidate) > 3:
            return _title_case_summary(candidate)

    # Strategy 3: strip filler prefix and truncate
    cleaned = re.sub(
        r"^(i\s+|we\s+|i'll\s+|we'll\s+|i'm\s+|i've\s+|let\s+me\s+|let's\s+)",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()
    cleaned = _clean_candidate(cleaned)

    if len(cleaned) > 60:
        cleaned = cleaned[:60].rsplit(" ", 1)[0]

    return _title_case_summary(cleaned) if cleaned else "Unknown task"


_COMMENT_PATTERNS = [
    re.compile(r"comment\s+(?:that|saying|to\s+say)\s+(.+)$", re.IGNORECASE),
    re.compile(r"note\s+(?:that|down\s+that)\s+(.+)$", re.IGNORECASE),
    re.compile(r"add\s+(?:a\s+)?comment\s+(?:on\s+\S+\s+)?that\s+(.+)$", re.IGNORECASE),
]


def _extract_comment(text: str) -> Optional[str]:
    for pat in _COMMENT_PATTERNS:
        m = pat.search(text)
        if m:
            return m.group(1).strip().rstrip(".")
    return text.strip()


# ── Step 7: Assemble ScrumAction ──────────────────────────────────────────────

def build_action(item: Dict) -> ScrumAction:
    from backend.config.team_config import CONFIDENCE_THRESHOLD

    text       = item["sentence"]
    speaker    = item["speaker"]
    intent     = item["intent"]
    confidence = item["confidence"]

    ticket_key  = _extract_ticket_key(text)
    status      = _extract_status(text)
    description = _extract_description(text)
    assignee    = _extract_assignee(text, speaker, intent)
    summary     = _extract_summary(text, speaker, intent)
    deadline    = _extract_deadline(text)

    # Deadline overrides intent
    if deadline:
        intent = "deadline_change"

    # Only override intent when classifier is uncertain AND completion word found
    elif (
        _DONE_OVERRIDE_RE.search(text)
        and intent != "complete_task"
        and confidence < 0.75
    ):
        intent = "complete_task"

    # Resolve email — keep assignee name even if email isn't configured
    assignee_email = None
    if assignee:
        assignee_email = TEAM_EMAIL_MAP.get(assignee.lower())

    comment = _extract_comment(text) if intent == "add_comment" else None

    if intent == "complete_task" and status is None:
        status = "Done"

    return ScrumAction(
        action=intent,
        summary=summary,
        description=description,
        assignee=assignee,
        assignee_email=assignee_email,
        status=status,
        ticket_key=ticket_key,
        comment=comment,
        deadline_change=deadline,
        confidence=confidence,
        low_confidence=confidence < CONFIDENCE_THRESHOLD,
        source_speaker=speaker,
        source_sentence=text,
    )


# ── Step 8: Postprocessing ────────────────────────────────────────────────────

_FILLER_SUMMARIES = {
    "think that covers everything",
    "assign that to you",
    "it", "that", "this", "it last night",
}


def postprocess_actions(actions: List[ScrumAction]) -> List[ScrumAction]:
    merged = []
    for action in actions:

        # Rule 1: merge description sentence into preceding create_task
        if (
            action.source_sentence
            and action.source_sentence.lower().startswith("the description should be")
            and merged
            and merged[-1].action == "create_task"
        ):
            merged[-1].description = action.description or action.summary
            continue

        # Rule 2: drop known filler summaries
        if not action.summary:
            action.summary = (action.source_sentence or "")[:50]

        if action.summary.lower().rstrip(".") in _FILLER_SUMMARIES:
            continue

        # Rule 3: deduplicate consecutive same-speaker/intent/status/summary
        if (
            merged
            and merged[-1].source_speaker == action.source_speaker
            and merged[-1].action == action.action
            and merged[-1].status == action.status
            and merged[-1].summary.lower() == action.summary.lower()
        ):
            continue

        merged.append(action)

    return merged


# ── Public API ────────────────────────────────────────────────────────────────

class NLPScrumExtractor:

    def extract_actions(self, transcript: str) -> List[dict]:
        print("\n[NLP] Step 1: Parsing transcript...")
        parsed = parse_transcript(transcript)
        print(f"       → {len(parsed)} speaker turns")

        print("[NLP] Step 2: Splitting into sentences...")
        sentences = split_sentences(parsed)
        print(f"       → {len(sentences)} sentences (before merge)")

        sentences = merge_context(sentences)          # called ONCE
        print(f"       → {len(sentences)} sentences (after merge)")

        print("[NLP] Step 3: Filtering non-actionable sentences...")
        actionable = filter_actionable(sentences)
        print(f"       → {len(actionable)} actionable sentences")

        print("[NLP] Step 4: Classifying intent...")
        classified = classify_intent(actionable)

        print("[NLP] Step 5 & 6: Extracting entities and building actions...")
        action_objects = []
        for item in classified:
            action = build_action(item)
            flag = " [LOW CONF]" if action.low_confidence else ""
            print(f"       [{action.action}]{flag} → {action.summary!r}  "
                  f"(conf={action.confidence:.2f})")
            action_objects.append(action)

        action_objects = postprocess_actions(action_objects)
        actions = [a.to_dict() for a in action_objects]

        print(f"\n[NLP] Done. {len(actions)} action(s) extracted.\n")
        return actions


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

    here = os.path.dirname(__file__)
    transcript_path = os.path.join(here, "..", "agents", "example_transcript.txt")

    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript = f.read()

    extractor = NLPScrumExtractor()
    actions = extractor.extract_actions(transcript)

    print("\nFinal Actions:\n")
    print(json.dumps(actions, indent=2))