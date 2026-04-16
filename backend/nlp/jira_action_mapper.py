"""
=============================================================
NLP → Jira Action Mapper
=============================================================
Converts the NLP pipeline output (GRU action dicts + SBERT
story IDs + NER actors) into the format expected by the
existing JiraAgent.execute_actions() method.

GRU action types    →   JiraAgent action type
─────────────────────────────────────────────
complete_task       →   complete_task
create_task         →   create_task
update_status       →   update_status
assign_task         →   assign_task
no_action           →   (skipped)

Syllabus connection: This is the application layer that
bridges NLP extraction (Units 1–3) to real-world automation.
=============================================================
"""

from typing import List, Dict, Optional
import re


# ── Status mapping ────────────────────────────────────────
# Maps GRU action → Jira workflow status
GRU_TO_JIRA_STATUS = {
    "complete_task":  "Done",
    "update_status":  "In Progress",
    "assign_task":    "In Progress",
    "create_task":    None,           # new ticket, no transition
    "no_action":      None,
}

# ── Priority from WSJF score ──────────────────────────────
def _wsjf_to_priority(wsjf: float) -> str:
    if wsjf >= 8.0:  return "Highest"
    if wsjf >= 6.0:  return "High"
    if wsjf >= 4.0:  return "Medium"
    return "Low"


# ── Sentence → ticket summary ─────────────────────────────
def _clean_summary(sentence: str, max_len: int = 80) -> str:
    """Extract a clean ticket summary from a raw sentence."""
    # Remove speaker labels (e.g. "Mike: Yesterday I ...")
    sentence = re.sub(r"^[A-Z][a-z]+:\s*", "", sentence).strip()
    # Remove filler words
    for filler in ["yesterday", "today", "basically", "actually", "you know", "um", "uh"]:
        sentence = re.sub(rf"\b{filler}\b", "", sentence, flags=re.IGNORECASE)
    sentence = re.sub(r"\s+", " ", sentence).strip()
    return sentence[:max_len].capitalize()


# ════════════════════════════════════════════════════════════
# STANDUP: GRU actions → Jira action dicts
# ════════════════════════════════════════════════════════════

def map_standup_actions(nlp_actions: List[Dict]) -> List[Dict]:
    """
    Convert GRU-extracted standup actions → JiraAgent action dicts.

    Input (from pipeline_orchestrator):
        [
          {
            "sentence": "Mike completed the Stripe payment integration",
            "action":   "complete_task",
            "actor":    "Mike",
            "story_id": "SP-002",
            "story_title": "Integrate Stripe payment gateway...",
          }, ...
        ]

    Output (for JiraAgent.execute_actions):
        [
          {
            "action":      "complete_task",
            "summary":     "SP-002 - Stripe payment gateway",
            "description": "Mike completed the Stripe payment integration",
            "assignee":    "Mike",
            "status":      "Done",
          }, ...
        ]
    """
    jira_actions = []

    for item in nlp_actions:
        gru_action = item.get("action", "no_action")
        if gru_action == "no_action":
            continue

        sentence    = item.get("sentence", "")
        actor       = item.get("actor", "")
        story_id    = item.get("story_id", "")
        story_title = item.get("story_title", "")

        # Build a clean summary: prefer story title, fallback to sentence snippet
        if story_id and story_title:
            summary = f"{story_id} - {story_title[:60]}"
        else:
            summary = _clean_summary(sentence)

        jira_dict = {
            "action":      gru_action,
            "summary":     summary,
            "description": f"[NLP extracted] {sentence[:120]}",
            "assignee":    actor,
            "status":      GRU_TO_JIRA_STATUS.get(gru_action, "In Progress"),
            "story_id":    story_id,
        }
        jira_actions.append(jira_dict)

    return jira_actions


# ════════════════════════════════════════════════════════════
# PM MEETING: NLP epics → epic_creation approval payload
# ════════════════════════════════════════════════════════════

def map_pm_meeting_epics(
    summary_text: str,
    entities: Dict,
    extracted_epics: Optional[List[Dict]] = None,
) -> Dict:
    """
    Build the 'epics_data' dict expected by:
        ApprovalService.create_epic_approval(epics_data, ...)

    Args:
        summary_text   : BART abstractive summary of the PM meeting
        entities       : NER output {assignees, estimates, dates}
        extracted_epics: Optional list of epics parsed from transcript
                         Each: {title, business_value, time_criticality, effort}

    Returns:
        {
          "epics": [...],
          "meeting_summary": "...",
          "source": "nlp_pipeline"
        }
    """
    if not extracted_epics:
        # Fallback: create a single epic from the summary
        extracted_epics = [{
            "title":             "Meeting Epic (auto-extracted by NLP)",
            "description":       summary_text,
            "business_value":    8,
            "time_criticality":  7,
            "risk_reduction":    5,
            "effort":            5,
            "wsjf": {
                "wsjf_score":    (8 + 7 + 5) / 5,
                "cost_of_delay": 8 + 7 + 5,
                "job_size":      5,
            },
            "assignees":         entities.get("assignees", []),
            "dates":             entities.get("dates", []),
        }]

    return {
        "epics":           extracted_epics,
        "meeting_summary": summary_text,
        "source":          "nlp_pipeline",
        "nlp_metadata": {
            "model":       "DistilBART + GRU + spaCy NER",
            "assignees":   entities.get("assignees", []),
            "estimates":   entities.get("estimates", []),
            "dates":       entities.get("dates", []),
        },
    }


# ════════════════════════════════════════════════════════════
# SPRINT PLANNING: NLP stories → sprint_planning approval payload
# ════════════════════════════════════════════════════════════

def map_sprint_planning(
    summary_text: str,
    entities:     Dict,
    actions:      List[Dict],
    meeting_type: str = "SPRINT_PLANNING",
) -> Dict:
    """
    Build sprint_data dict for:
        ApprovalService.create_sprint_approval(sprint_data, ...)

    Returns:
        {
          "sprint_name": "Sprint N",
          "sprint_goal": "...",
          "story_ids":   ["SP-001", ...],
          "stories":     [...],
          "capacity":    80,
          ...
        }
    """
    # Parse sprint name and goal from entities
    sprint_name = "Next Sprint"
    capacity    = 80

    # Extract capacity from estimates if present
    for est in entities.get("estimates", []):
        try:
            val = int(re.sub(r"[^\d]", "", str(est)))
            if 20 <= val <= 200:   # plausible capacity
                capacity = val
                break
        except ValueError:
            pass

    # Collect story IDs from NLP action mapping
    story_ids = list({a.get("story_id") for a in actions if a.get("story_id")})

    stories = []
    for a in actions:
        if a.get("story_id"):
            stories.append({
                "story_id":    a["story_id"],
                "title":       a.get("story_title", a.get("sentence", "")[:60]),
                "assignee":    a.get("actor", ""),
                "action_type": a.get("action", ""),
            })

    return {
        "sprint_name":     sprint_name,
        "sprint_goal":     summary_text[:150],
        "story_ids":       story_ids,
        "stories":         stories,
        "capacity":        capacity,
        "source":          "nlp_pipeline",
        "meeting_summary": summary_text,
        "nlp_metadata": {
            "model":     "LSTM + GRU + SBERT + DistilBART",
            "assignees": entities.get("assignees", []),
            "estimates": entities.get("estimates", []),
            "dates":     entities.get("dates", []),
        },
    }


# ════════════════════════════════════════════════════════════
# STANDUP: Build approval payload
# ════════════════════════════════════════════════════════════

def map_standup_approval_payload(
    actions:      List[Dict],
    summary_text: str,
    entities:     Dict,
    blockers:     List[str],
) -> Dict:
    """
    Build the payload for:
        ApprovalService.create_standup_approval(payload, ...)

    This is also what appears in the Telegram approval message
    via format_standup_approval() in approval_handler.py.
    """
    jira_actions = map_standup_actions(actions)

    action_counts: Dict[str, int] = {}
    for a in jira_actions:
        t = a["action"]
        action_counts[t] = action_counts.get(t, 0) + 1

    return {
        "actions":     jira_actions,
        "blockers":    blockers,
        "summary": {
            "total_actions":  len(jira_actions),
            "action_counts":  action_counts,
            "meeting_summary": summary_text,
            "participants":   entities.get("assignees", []),
            "dates":          entities.get("dates", []),
        },
        "source": "nlp_pipeline",
        "nlp_metadata": {
            "model": "GRU + NER + SBERT + DistilBART",
        },
    }
