"""
=============================================================
NLP Context Loader
=============================================================
Fetches the active sprint's stories, epics, and team assignments
from PostgreSQL and packages them into a SprintContext object
that the NLPOrchestrator uses to:

  1. Seed the SBERT semantic-search corpus with REAL sprint stories
     (replaces the slow/unreliable Jira API call).
  2. Resolve actor names (Alice, Tom...) → Jira ticket keys via
     the SprintAssignment table.
  3. Supply sprint metadata (goal, capacity, velocity) to the
     approval payload for sprint-planning meetings.
  4. Provide unassigned epic titles for PM-meeting processing.

Usage:
    from backend.nlp.context_loader import load_sprint_context

    ctx = load_sprint_context()          # uses active sprint
    ctx = load_sprint_context(sprint_id=14)  # specific sprint

Falls back gracefully to None if the database is unreachable or
no sprint data exists yet.
=============================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ── SprintContext dataclass ───────────────────────────────────────────────────

@dataclass
class SprintContext:
    """
    Rich context object assembled from PostgreSQL sprint data.
    Consumed by NLPOrchestrator to make every pipeline step context-aware.
    """

    # Sprint identity
    sprint_id:       int
    sprint_number:   Optional[int]
    sprint_name:     str
    sprint_goal:     str
    sprint_status:   str              # "active" | "planned" | "completed"

    # Capacity metadata (for sprint-planning payloads)
    capacity_hours:   Optional[int]
    velocity_target:  Optional[int]
    velocity_actual:  Optional[int]
    team_size:        Optional[int]

    # SBERT corpus — replaces Jira API story list
    story_titles: List[str] = field(default_factory=list)   # "User Authentication Story"
    story_keys:   List[str] = field(default_factory=list)   # "SP-001"
    story_ids:    List[int] = field(default_factory=list)   # internal DB IDs

    # Per-story detail dicts (for approval payload enrichment)
    stories: List[dict] = field(default_factory=list)

    # Epic corpus — used for PM-meeting epic extraction
    epic_titles: List[str] = field(default_factory=list)
    epic_keys:   List[str] = field(default_factory=list)

    # Actor resolution: lower-cased name → Jira story key
    # e.g. {"alice": "SP-003", "tom": "SP-001"}
    assignments: Dict[str, str] = field(default_factory=dict)

    # ── Convenience helpers ───────────────────────────────────────────────────

    def resolve_actor(self, name: str) -> Optional[str]:
        """
        Return the Jira key assigned to `name` (case-insensitive).
        Returns None if no assignment found.
        """
        return self.assignments.get(name.lower().strip())

    def sbert_corpus(self) -> tuple[List[str], List[str]]:
        """
        Return (titles, keys) ready to pass to SBERT find_matching_story().
        Falls back to an empty corpus if no stories were loaded.
        """
        return self.story_titles, self.story_keys

    def summary(self) -> str:
        """Human-readable one-liner for logging."""
        return (
            f"Sprint '{self.sprint_name}' ({self.sprint_status}) | "
            f"{len(self.story_titles)} stories | "
            f"{len(self.assignments)} assignments | "
            f"capacity={self.capacity_hours}h | vel_target={self.velocity_target}"
        )


# ── Loader ────────────────────────────────────────────────────────────────────

def load_sprint_context(sprint_id: Optional[int] = None) -> Optional[SprintContext]:
    """
    Query PostgreSQL for sprint context and return a SprintContext.

    Args:
        sprint_id: Specific sprint to load.  If None, the active (or most
                   recent) sprint is used automatically.

    Returns:
        SprintContext on success, None on any DB / data error.
    """
    try:
        from backend.db.connection import get_session
        from backend.db.crud import (
            get_active_sprint,
            get_sprint_stories_with_details,
            get_sprint_assignments,
        )
    except Exception as exc:
        logger.warning(f"[ContextLoader] DB imports failed — running without context: {exc}")
        return None

    try:
        with get_session() as db:
            # ── 1. Resolve the sprint ─────────────────────────────────────
            if sprint_id is not None:
                from backend.db.models import Sprint
                from sqlalchemy import select
                sprint = db.get(Sprint, sprint_id)
            else:
                sprint = get_active_sprint(db)

            if sprint is None:
                logger.info("[ContextLoader] No sprint found in DB — using fallback corpus")
                return None

            sid = sprint.sprint_id
            logger.info(f"[ContextLoader] Loading context for sprint {sid}: '{sprint.sprint_name}'")

            # ── 2. Stories ────────────────────────────────────────────────
            stories = get_sprint_stories_with_details(db, sid)

            story_titles: List[str] = []
            story_keys:   List[str] = []
            story_ids:    List[int] = []
            epic_titles:  List[str] = []
            epic_keys:    List[str] = []
            seen_epics:   set       = set()

            for s in stories:
                title = s.get("title") or ""
                key   = s.get("jira_key") or ""

                if title:
                    # Use description as supplementary text if title is short
                    desc = s.get("description") or ""
                    corpus_text = f"{title}. {desc}".strip() if desc else title
                    story_titles.append(corpus_text)
                    story_keys.append(key or f"DB-{s['story_id']}")
                    story_ids.append(s["story_id"])

                # Collect unique epics
                et = s.get("epic_title") or ""
                ek = s.get("epic_jira_key") or ""
                if et and et not in seen_epics:
                    epic_titles.append(et)
                    epic_keys.append(ek)
                    seen_epics.add(et)

            # ── 3. Assignments ────────────────────────────────────────────
            assignments = get_sprint_assignments(db, sid)

            # ── 4. Build context ─────────────────────────────────────────
            ctx = SprintContext(
                sprint_id       = sid,
                sprint_number   = sprint.sprint_number,
                sprint_name     = sprint.sprint_name,
                sprint_goal     = sprint.sprint_goal or "",
                sprint_status   = sprint.status or "unknown",
                capacity_hours  = sprint.team_capacity_hours,
                velocity_target = sprint.velocity_target,
                velocity_actual = sprint.velocity_actual,
                team_size       = sprint.team_size,
                story_titles    = story_titles,
                story_keys      = story_keys,
                story_ids       = story_ids,
                stories         = stories,
                epic_titles     = epic_titles,
                epic_keys       = epic_keys,
                assignments     = assignments,
            )

            logger.info(f"[ContextLoader] {ctx.summary()}")
            return ctx

    except Exception as exc:
        logger.warning(
            f"[ContextLoader] Failed to load sprint context ({exc}); "
            "pipeline will use Jira API fallback"
        )
        return None


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, __file__.replace("\\", "/").split("/backend/")[0])

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    ctx = load_sprint_context()
    if ctx is None:
        print("\nNo context loaded (DB unavailable or no sprint in DB).")
        print("The pipeline will fall back to the Jira API corpus.\n")
    else:
        print(f"\n{'='*60}")
        print(f"  Sprint:      {ctx.sprint_name} (#{ctx.sprint_number})")
        print(f"  Status:      {ctx.sprint_status}")
        print(f"  Goal:        {ctx.sprint_goal[:80]}")
        print(f"  Capacity:    {ctx.capacity_hours}h  |  Vel target: {ctx.velocity_target}")
        print(f"  Team size:   {ctx.team_size}")
        print(f"\n  Stories ({len(ctx.story_titles)}):")
        for t, k in zip(ctx.story_titles, ctx.story_keys):
            print(f"    [{k}] {t[:70]}")
        print(f"\n  Epics ({len(ctx.epic_titles)}):")
        for t, k in zip(ctx.epic_titles, ctx.epic_keys):
            print(f"    [{k}] {t[:70]}")
        print(f"\n  Assignments ({len(ctx.assignments)}):")
        for name, key in ctx.assignments.items():
            print(f"    {name.title():15} → {key}")
        print(f"{'='*60}\n")
