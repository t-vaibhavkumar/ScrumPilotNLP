from backend.tools.jira_client import JiraManager
from dotenv import load_dotenv
load_dotenv()
import os
print("URL:", os.getenv("JIRA_URL"))
print("EMAIL:", os.getenv("JIRA_EMAIL"))
print("TOKEN:", "SET" if os.getenv("JIRA_API_TOKEN") else "MISSING")


def bootstrap():
    jira = JiraManager()

    tasks = [
        "Authentication middleware",
        "Dashboard UI",
        "Payment service integration",
        "Cache invalidation bug",
        "Notifications module",
        "Analytics job",
    ]

    print("\n🚀 Bootstrapping Jira with base tasks...\n")

    for task in tasks:
        existing = jira.search_tickets(summary_query=task)

        if existing.get("issues"):
            print(f"⚠️ Already exists: {task}")
            continue

        resp = jira.create_ticket(summary=task)

        if resp.get("key"):
            print(f"✅ Created: {task} → {resp['key']}")
        else:
            print(f"❌ Failed: {task}")

    print("\nDone.\n")


if __name__ == "__main__":
    bootstrap()