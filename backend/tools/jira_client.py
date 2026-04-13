import os
from jira import JIRA
from typing import Optional, List, Dict


class JiraManager:
    def __init__(self):
        self.url         = os.getenv("JIRA_URL")
        self.email       = os.getenv("JIRA_EMAIL")
        self.token       = os.getenv("JIRA_API_TOKEN")
        self.project_key = os.getenv("JIRA_PROJECT_KEY")

        if not all([self.url, self.email, self.token]):
            raise ValueError("Jira credentials missing in environment variables.")

        self.client = JIRA(
            server=self.url,
            basic_auth=(self.email, self.token)
        )

        # Cache email → accountId lookups
        self._account_cache: Dict[str, str] = {}

    # ── Create ────────────────────────────────────────────────────────────

    def create_ticket(
        self,
        summary: str,
        description: str = "",
        issue_type: str = "Task",
        assignee_email: Optional[str] = None,
    ) -> Dict:
        try:
            fields = {
                "project":   {"key": self.project_key},   # FIX 1: must be dict
                "summary":   summary,
                "issuetype": {"name": issue_type},
            }

            if description:
                fields["description"] = self._to_adf(description)

            # FIX 2: Jira Cloud requires accountId, not email/name
            if assignee_email:
                account_id = self._resolve_account_id(assignee_email)
                if account_id:
                    fields["assignee"] = {"accountId": account_id}

            new_issue = self.client.create_issue(fields=fields)
            return {
                "success": True,
                "key":     new_issue.key,
                "summary": summary,
                "message": f"Ticket {new_issue.key} created successfully.",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ── Status transitions ────────────────────────────────────────────────

    def get_transitions(self, ticket_key: str) -> Dict:
        try:
            raw = self.client.transitions(ticket_key)
            transitions = [{"id": t["id"], "name": t["name"]} for t in raw]
            return {"success": True, "transitions": transitions}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def update_ticket_status(self, ticket_key: str, transition_name: str) -> Dict:
        """
        FIX 3: Looks up the transition ID by name before calling transition_issue().
        Jira requires the ID — passing the name string directly throws an error.
        """
        try:
            raw         = self.client.transitions(ticket_key)
            transitions = {t["name"].lower(): t["id"] for t in raw}

            # Exact match first
            transition_id = transitions.get(transition_name.lower())

            # Fuzzy match — "Done" matches "Mark as Done"
            if not transition_id:
                transition_id = next(
                    (tid for name, tid in transitions.items()
                     if transition_name.lower() in name),
                    None,
                )

            if not transition_id:
                return {
                    "success": False,
                    "error":   f"Transition '{transition_name}' not found. "
                               f"Available: {list(transitions.keys())}",
                }

            self.client.transition_issue(ticket_key, transition_id)
            return {
                "success": True,
                "message": f"Ticket {ticket_key} transitioned to '{transition_name}'.",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ── Assignment ────────────────────────────────────────────────────────

    def assign_ticket(self, ticket_key: str, assignee_email: str) -> Dict:
        try:
            account_id = self._resolve_account_id(assignee_email)
            if not account_id:
                return {
                    "success": False,
                    "error":   f"No Jira account found for: {assignee_email}",
                }
            self.client.assign_issue(ticket_key, account_id)
            return {
                "success": True,
                "message": f"Ticket {ticket_key} assigned to {assignee_email}.",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ── Search ────────────────────────────────────────────────────────────

    def search_tickets(
        self,
        summary_query: Optional[str] = None,
        assignee_email: Optional[str] = None,
        status: Optional[str] = None,
        max_results: int = 10,
    ) -> Dict:
        try:
            clauses = [f'project = "{self.project_key}"']

            if summary_query:
                safe = summary_query.replace('"', '\\"')
                clauses.append(f'summary ~ "{safe}"')

            if assignee_email:
                account_id = self._resolve_account_id(assignee_email)
                if account_id:
                    clauses.append(f'assignee = "{account_id}"')

            if status:
                clauses.append(f'status = "{status}"')

            jql    = " AND ".join(clauses) + " ORDER BY created DESC"
            issues = self.client.search_issues(jql, maxResults=max_results)

            results = []
            for i in issues:
                results.append({
                    "key":     i.key,
                    "summary": i.fields.summary,
                    "status":  i.fields.status.name,
                    "assignee": (
                        getattr(i.fields.assignee, "emailAddress", None)
                        or getattr(i.fields.assignee, "displayName", None)
                        if i.fields.assignee else None
                    ),
                })
            return {"success": True, "issues": results}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_user_tickets(self, email: str) -> Dict:
        return self.search_tickets(assignee_email=email, status=None)

    # ── Comments ──────────────────────────────────────────────────────────

    def add_comment(self, ticket_key: str, comment_text: str) -> Dict:
        try:
            self.client.add_comment(ticket_key, comment_text)
            return {
                "success": True,
                "message": f"Comment added to {ticket_key}.",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ── Helpers ───────────────────────────────────────────────────────────

    def _resolve_account_id(self, email: str) -> Optional[str]:
        """Resolve email → Jira Cloud accountId (cached per session)."""
        if email in self._account_cache:
            return self._account_cache[email]
        try:
            users = self.client.search_users(query=email)
            if users:
                account_id = users[0].accountId
                self._account_cache[email] = account_id
                return account_id
            print(f"[Jira] No user found for: {email}")
            return None
        except Exception as e:
            print(f"[Jira] User lookup failed for {email}: {e}")
            return None

    @staticmethod
    def _to_adf(text: str) -> dict:
        """Convert plain text to Atlassian Document Format (required by Jira Cloud)."""
        return {
            "version": 1,
            "type":    "doc",
            "content": [
                {
                    "type":    "paragraph",
                    "content": [{"type": "text", "text": text}],
                }
            ],
        }


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    manager = JiraManager()

    print("\n--- Recent tickets ---")
    result = manager.search_tickets(max_results=3)
    for issue in result.get("issues", []):
        print(f"  {issue['key']}: {issue['summary']} [{issue['status']}]")

    print("\n--- Available transitions on first ticket ---")
    if result.get("issues"):
        key = result["issues"][0]["key"]
        print(manager.get_transitions(key))