from __future__ import annotations
import os
import sys
from typing import List, Dict, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


class ActionResult:
    def __init__(self, action, success, message, ticket_key=None):
        self.action = action
        self.success = success
        self.message = message
        self.ticket_key = ticket_key

    def __repr__(self):
        status = "OK" if self.success else "FAIL"
        key = f" [{self.ticket_key}]" if self.ticket_key else ""
        return f"[{status}]{key} {self.message}"


class JiraExecutor:
    def __init__(self, jira_manager=None):
        if jira_manager is None:
            from backend.tools.jira_client import JiraManager
            jira_manager = JiraManager()

        self._jira = jira_manager
        self._last_ticket = None  # 🔥 IMPORTANT (context memory)

    # ─────────────────────────────────────────────
    # MAIN ENTRY
    # ─────────────────────────────────────────────
    def execute_actions(self, actions: List[Dict]) -> str:
        results = []

        for action in actions:
            print(f"\n→ Executing [{action['action']}]: {action.get('summary')}")

            try:
                result = self._route(action)
            except Exception as e:
                result = ActionResult(action, False, str(e))

            if result.ticket_key:
                self._last_ticket = result.ticket_key

            results.append(result)
            print("   ", result)

        return self._format_report(results)

    # ─────────────────────────────────────────────
    # ROUTER
    # ─────────────────────────────────────────────
    def _route(self, action):
        mapping = {
            "create_task": self._handle_create,
            "complete_task": self._handle_complete,
            "update_status": self._handle_update,
            "assign_task": self._handle_assign,
            "add_comment": self._handle_comment,
        }

        handler = mapping.get(action.get("action"))
        if not handler:
            return ActionResult(action, False, "Unknown action")

        return handler(action)

    # ─────────────────────────────────────────────
    # HANDLERS
    # ─────────────────────────────────────────────

    def _handle_create(self, action):
        summary = action.get("summary", "").strip()

        # 🔥 avoid duplicates
        existing = self._search(summary)
        if existing:
            return ActionResult(action, True, "Task already exists", existing)

        resp = self._jira.create_ticket(
            summary=summary,
            description=action.get("description", ""),
            assignee_email=action.get("assignee_email"),
        )

        if resp.get("key"):
            return ActionResult(action, True, "Created", resp["key"])

        return ActionResult(action, False, str(resp))

    def _handle_complete(self, action):
        ticket = self._resolve_ticket(action)
        if not ticket:
            return ActionResult(action, False, "No ticket found")

        resp = self._jira.update_ticket_status(ticket, "Done")
        return ActionResult(action, True, "Marked Done", ticket)

    def _handle_update(self, action):
        ticket = self._resolve_ticket(action)
        if not ticket:
            return ActionResult(action, False, "No ticket found")

        status = action.get("status") or "In Progress"
        resp = self._jira.update_ticket_status(ticket, status)

        return ActionResult(action, True, f"Updated to {status}", ticket)

    def _handle_assign(self, action):
        ticket = self._resolve_ticket(action)

        if not ticket:
            return ActionResult(action, False, "No ticket found")

        email = action.get("assignee_email")
        if not email:
            return ActionResult(action, False, "No assignee email", ticket)

        self._jira.assign_ticket(ticket, email)
        return ActionResult(action, True, f"Assigned to {email}", ticket)

    def _handle_comment(self, action):
        ticket = self._resolve_ticket(action)
        if not ticket:
            return ActionResult(action, False, "No ticket found")

        comment = action.get("comment") or action.get("summary")
        self._jira.add_comment(ticket, comment)

        return ActionResult(action, True, "Comment added", ticket)

    # ─────────────────────────────────────────────
    # SMART MATCHING (VERY IMPORTANT)
    # ─────────────────────────────────────────────

    def _resolve_ticket(self, action) -> Optional[str]:
        summary = (action.get("summary") or "").lower()

        # ❌ ignore garbage summaries
        bad = ["it", "that", "this", "yeah", "resolved it"]
        if any(b in summary for b in bad):
            return self._last_ticket  # fallback

        # try search
        match = self._search(summary)
        if match:
            return match

        # fallback to last used ticket
        return self._last_ticket

    def _search(self, summary):
        if not summary or len(summary.split()) < 2:
            return None

        resp = self._jira.search_tickets(summary_query=summary)
        issues = resp.get("issues", [])

        if not issues:
            return None

        # 🔥 scoring
        words = set(summary.split())

        best = None
        best_score = 0

        for issue in issues:
            issue_summary = issue.get("summary", "").lower()
            score = len(words & set(issue_summary.split()))

            if score > best_score:
                best_score = score
                best = issue

        return best["key"] if best else None

    # ─────────────────────────────────────────────
    # REPORT
    # ─────────────────────────────────────────────

    def _format_report(self, results):
        success = [r for r in results if r.success]
        fail = [r for r in results if not r.success]

        lines = [
            "=" * 40,
            "JIRA REPORT",
            "=" * 40,
            f"Success: {len(success)} | Failed: {len(fail)}",
        ]

        for r in results:
            lines.append(str(r))

        return "\n".join(lines)