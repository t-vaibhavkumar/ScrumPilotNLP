"""
team_config.py — Static configuration for the NLP pipeline.

Edit this file to match your actual Jira project and team members.
"""

# ── Name → Jira email mapping ─────────────────────────────────────────────────
# Keys are lowercase versions of names as they appear in transcripts.
# Values are the Jira account emails used for assignee lookups.
TEAM_EMAIL_MAP: dict[str, str] = {
    "sameer": "sameer@example.com",
    "alex":   "alex@example.com",
    "priya":  "priya@example.com",
}

# ── Status vocabulary → canonical Jira transition names ──────────────────────
# Each list contains surface forms that map to a canonical transition.
# Add synonyms your team uses naturally.
STATUS_PHRASE_MAP: dict[str, list[str]] = {
    "Done": [
        "done", "completed", "finished", "complete", "wrapped up",
        "closed", "resolved", "shipped", "deployed", "merged",
    ],
    "In Progress": [
        "in progress", "working on", "started", "picking up", "pick up",
        "taking on", "handling", "started working", "currently on",
        "going to work", "going to pick",
    ],
    "To Do": [
        "to do", "todo", "not started", "pending", "queued", "backlog",
    ],
    "In Review": [
        "in review", "reviewing", "under review", "ready for review",
        "up for review", "needs review",
    ],
    "Blocked": [
        "blocked", "stuck", "waiting on", "on hold", "paused",
    ],
}

# ── Intent label → zero-shot hypothesis templates ────────────────────────────
# These are the natural-language hypotheses fed to the BART zero-shot classifier.
# Tune these if classification accuracy is poor for specific action types.
INTENT_HYPOTHESES = {
    "create_task":   "This sentence announces that a new task, ticket, or work item needs to be created.",
    "complete_task": "This sentence states that a task or work item has been finished or completed.",
    "update_status": "This sentence describes the current progress or status of ongoing work.",
    "assign_task":   "This sentence assigns or delegates a task to a specific person.",
    "add_comment":   "This sentence requests adding a note or decision to an existing ticket.",
    "deadline_change": "This sentence indicates a deadline is extended or missed."
}
# ── Confidence threshold ──────────────────────────────────────────────────────
# Sentences whose top intent score falls below this threshold are flagged
# as low-confidence in the output (still included, but marked).
CONFIDENCE_THRESHOLD: float = 0.45

# ── Jira project key ──────────────────────────────────────────────────────────
JIRA_PROJECT_KEY: str = "SCRUM"

# ── Sentences to filter out (non-actionable) ─────────────────────────────────
# Short utterances that are greetings, filler, or purely social.
FILTER_PHRASES: list[str] = [
    "good morning", "good afternoon", "good evening",
    "let's start", "let's sync", "sync again", "that covers everything",
    "sounds good", "great", "perfect", "agreed", "nice", "sure",
    "alright", "okay", "ok", "thanks", "thank you","i think that covers","i agree with", "let's sync again",
    "right good call","that covers everything","i'll go first",
]

# Minimum token count for a sentence to be considered actionable.
MIN_SENTENCE_TOKENS: int = 5