"""
generate_data.py — Synthetic Scrum meeting data generator.

Generates ~200 sentences per class (1000 total) for fine-tuning
a DistilBERT intent classifier on 5 Scrum action types.

Run locally:
    python training/generate_data.py

Output:
    training/synthetic_scrum_data.csv
"""

import csv
import random
import os

random.seed(42)

# ── Vocabulary pools ──────────────────────────────────────────────────────────

NAMES = [
    "Alex", "Priya", "Sameer", "Jordan", "Taylor", "Morgan",
    "Jamie", "Riley", "Casey", "Devon", "Avery", "Quinn",
]

TASKS = [
    "the login page UI", "the API integration module", "the payment gateway",
    "the database schema migration", "the user profile page", "the CI/CD pipeline",
    "the authentication service", "the notification service", "the search feature",
    "the admin dashboard", "the reporting module", "the email service",
    "the unit tests for auth", "the deployment script", "the caching layer",
    "the onboarding flow", "the billing integration", "the data export feature",
    "the OAuth implementation", "the rate limiting middleware", "the error tracking setup",
    "the mobile responsive layout", "the API documentation", "the load balancer config",
    "the Redis integration", "the S3 bucket setup", "the webhook handler",
    "the password reset flow", "the dark mode feature", "the accessibility fixes",
]

TICKET_KEYS = [f"SCRUM-{n}" for n in range(1, 60)]

DONE_PHRASES = [
    "completed", "finished", "done with", "wrapped up", "closed out",
    "shipped", "merged", "resolved", "deployed", "signed off on",
]

IN_PROGRESS_PHRASES = [
    "working on", "still on", "in the middle of", "currently handling",
    "picking up", "taking care of", "making progress on", "halfway through",
    "continuing with", "actively working on",
]

STATUS_TARGETS = ["In Progress", "In Review", "Blocked", "Done", "To Do"]


# ── Template sets per class ───────────────────────────────────────────────────

def make_create_task() -> list[str]:
    sentences = []
    name = lambda: random.choice(NAMES)
    task = lambda: random.choice(TASKS)
    key  = lambda: random.choice(TICKET_KEYS)

    templates = [
        # Direct creation requests
        lambda: f"We need to create a new task for {task()}.",
        lambda: f"I think we should open a ticket for {task()}.",
        lambda: f"Can someone create a story for {task()}?",
        lambda: f"Let me create a task for {task()}.",
        lambda: f"We need a new ticket for {task()}.",
        lambda: f"I'll open a ticket for {task()} today.",
        lambda: f"Someone should create an issue for {task()}.",
        lambda: f"We should add {task()} to the backlog.",
        lambda: f"Let's create a task to handle {task()}.",
        lambda: f"Can you open a new story for {task()}?",
        # With descriptions
        lambda: f"We need a task for {task()}, the description should cover the implementation details.",
        lambda: f"Create a ticket for {task()} with a description about the approach.",
        lambda: f"Let me add a new task for {task()} — it should include setup instructions.",
        # With assignee
        lambda: f"Create a task for {task()} and assign it to {name()}.",
        lambda: f"Open a ticket for {task()} for {name()} to pick up.",
        lambda: f"We need a new story for {task()}, {name()} will own it.",
        # Bug/investigation variants
        lambda: f"We need to open a bug for {task()}.",
        lambda: f"I'll create an investigation ticket for {task()}.",
        lambda: f"Let's log a bug for the issue in {task()}.",
        lambda: f"We should create a spike for {task()}.",
        # Indirect/implicit
        lambda: f"That feature isn't tracked yet — we need a ticket.",
        lambda: f"{task()} isn't in Jira yet, let me add it.",
        lambda: f"I don't see a ticket for {task()}, I'll create one.",
        lambda: f"We're missing a story for {task()}, someone add it.",
        lambda: f"There's no task for {task()} yet.",
        # Formal standup phrasing
        lambda: f"Today I'm going to create a task for {task()}.",
        lambda: f"I'll be creating a ticket for {task()} this morning.",
        lambda: f"My plan is to open a story for {task()} today.",
        lambda: f"I need to log a task for {task()} before EOD.",
        lambda: f"Action item: create a ticket for {task()}.",
    ]

    while len(sentences) < 200:
        sentences.append(random.choice(templates)())

    return sentences[:200]


def make_complete_task() -> list[str]:
    sentences = []
    name = lambda: random.choice(NAMES)
    task = lambda: random.choice(TASKS)
    key  = lambda: random.choice(TICKET_KEYS)
    done = lambda: random.choice(DONE_PHRASES)

    templates = [
        # First person
        lambda: f"I {done()} {task()} yesterday.",
        lambda: f"I've {done()} {task()}.",
        lambda: f"I {done()} {task()} last night.",
        lambda: f"{task().capitalize()} is done.",
        lambda: f"I've finished {task()}, it's ready for review.",
        lambda: f"I wrapped up {task()} this morning.",
        lambda: f"Just {done()} {task()}.",
        lambda: f"I'm done with {task()}.",
        lambda: f"{task().capitalize()} has been {done()}.",
        lambda: f"I've {done()} {task()} and pushed the changes.",
        # Third person
        lambda: f"{name()} {done()} {task()} yesterday.",
        lambda: f"{name()} has {done()} {task()}.",
        lambda: f"{name()} finished {task()} last sprint.",
        # Ticket key references
        lambda: f"I {done()} {key()}.",
        lambda: f"{key()} is done.",
        lambda: f"I've closed {key()}.",
        lambda: f"I finished {key()} this morning.",
        lambda: f"{key()} has been {done()}.",
        lambda: f"Just closed out {key()}.",
        # Status confirmation
        lambda: f"That task is fully done and ready for review.",
        lambda: f"The work on {task()} is complete.",
        lambda: f"{task().capitalize()} is finished, nothing left to do.",
        lambda: f"We can mark {task()} as done.",
        lambda: f"I'm happy to say {task()} is complete.",
        lambda: f"All the work for {task()} is wrapped up.",
        # PR / merge confirmation
        lambda: f"My PR for {task()} got merged.",
        lambda: f"I merged {task()} into main.",
        lambda: f"The PR for {task()} is merged and deployed.",
        lambda: f"{task().capitalize()} is deployed to staging.",
        lambda: f"I deployed {task()} last night.",
    ]

    while len(sentences) < 200:
        sentences.append(random.choice(templates)())

    return sentences[:200]


def make_update_status() -> list[str]:
    sentences = []
    name = lambda: random.choice(NAMES)
    task = lambda: random.choice(TASKS)
    key  = lambda: random.choice(TICKET_KEYS)
    prog = lambda: random.choice(IN_PROGRESS_PHRASES)

    templates = [
        # In progress
        lambda: f"I'm still {prog()} {task()}.",
        lambda: f"I've been {prog()} {task()} since yesterday.",
        lambda: f"Still {prog()} {task()}, should be done by EOD.",
        lambda: f"Today I'll continue {prog()} {task()}.",
        lambda: f"I'm {prog()} {task()} right now.",
        lambda: f"{task().capitalize()} is in progress.",
        lambda: f"I'm about halfway through {task()}.",
        lambda: f"Making good progress on {task()}.",
        lambda: f"I'm actively {prog()} {task()}.",
        lambda: f"I picked up {task()} yesterday and it's going well.",
        # Blocked
        lambda: f"I'm blocked on {task()}, waiting for {name()}.",
        lambda: f"{task().capitalize()} is blocked — we need a decision first.",
        lambda: f"I can't move forward on {task()} until the API is ready.",
        lambda: f"Blocked on {task()} due to a dependency on {name()}.",
        lambda: f"{task().capitalize()} is on hold until we resolve the config issue.",
        # In review
        lambda: f"{task().capitalize()} is in review.",
        lambda: f"I've submitted {task()} for review.",
        lambda: f"My PR for {task()} is up for review.",
        lambda: f"{task().capitalize()} is ready for review, waiting on {name()}.",
        lambda: f"I put {task()} into review yesterday.",
        # Ticket key
        lambda: f"{key()} is currently in progress.",
        lambda: f"I'm {prog()} {key()}.",
        lambda: f"{key()} is blocked waiting on infra.",
        lambda: f"{key()} is in review.",
        lambda: f"Still working through {key()}.",
        # Progress update
        lambda: f"I've completed about 80% of {task()}.",
        lambda: f"{task().capitalize()} is almost done, just a few things left.",
        lambda: f"I'm close to finishing {task()}.",
        lambda: f"Mostly done with {task()}, just testing left.",
        lambda: f"Just need to finish the tests for {task()} and it's done.",
    ]

    while len(sentences) < 200:
        sentences.append(random.choice(templates)())

    return sentences[:200]


def make_assign_task() -> list[str]:
    sentences = []
    name = lambda: random.choice(NAMES)
    name2 = lambda: random.choice(NAMES)
    task = lambda: random.choice(TASKS)
    key  = lambda: random.choice(TICKET_KEYS)

    def two_names():
        n1, n2 = random.sample(NAMES, 2)
        return n1, n2

    templates = [
        # Direct assignment
        lambda: f"I'll assign {task()} to {name()}.",
        lambda: f"Can you assign {task()} to {name()}?",
        lambda: f"{task().capitalize()} should go to {name()}.",
        lambda: f"Let's assign {task()} to {name()}.",
        lambda: f"I'm assigning {task()} to {name()}.",
        # Ownership
        lambda: (lambda n1, n2: f"{n1}, can you take ownership of {task()}?")(
            *two_names()),
        lambda: f"Can {name()} own {task()}?",
        lambda: f"{name()} will own {task()} going forward.",
        lambda: f"I'd like {name()} to own {task()}.",
        lambda: f"{name()} should be the owner of {task()}.",
        # Picking up
        lambda: f"{name()} is going to pick up {task()}.",
        lambda: f"I'll have {name()} pick up {task()}.",
        lambda: f"Can {name()} pick this up?",
        lambda: f"{name()} said they'd pick up {task()}.",
        # Ticket key assignment
        lambda: f"Assigning {key()} to {name()}.",
        lambda: f"I'll assign {key()} to {name()}.",
        lambda: f"{key()} is now assigned to {name()}.",
        lambda: f"Can you move {key()} to {name()}?",
        # Reallocation
        lambda: (lambda n1, n2: f"Moving {task()} from {n1} to {n2}.")(
            *two_names()),
        lambda: (lambda n1, n2: f"Let's reassign {task()} from {n1} to {n2}.")(
            *two_names()),
        lambda: (lambda n1, n2: f"{n2} will take over {task()} from {n1}.")(
            *two_names()),
        # Delegation
        lambda: f"I need someone to take {task()} — {name()}, can you?",
        lambda: f"{name()}, would you be able to handle {task()}?",
        lambda: f"I'm going to delegate {task()} to {name()}.",
        lambda: f"Can we give {task()} to {name()}?",
        lambda: f"{name()} should handle {task()} this sprint.",
        # Self-assignment
        lambda: f"I'll take {task()}.",
        lambda: f"I'll pick up {task()} myself.",
        lambda: f"I can take ownership of {task()}.",
        lambda: f"I'll own {task()} going forward.",
        lambda: f"Put {task()} on me.",
    ]

    while len(sentences) < 200:
        sentences.append(random.choice(templates)())

    return sentences[:200]


def make_add_comment() -> list[str]:
    sentences = []
    name = lambda: random.choice(NAMES)
    task = lambda: random.choice(TASKS)
    key  = lambda: random.choice(TICKET_KEYS)

    decisions = [
        "we decided to use REST instead of GraphQL",
        "we agreed to postpone this to next sprint",
        "the approach was approved in the design review",
        "we changed the deadline to Friday",
        "the client requested a UI change",
        "we switched from MySQL to PostgreSQL for this",
        "the team agreed to use a third-party library",
        "we decided to split this into two tasks",
        "the PR needs another round of review",
        "we got sign-off from the product team",
    ]

    decision = lambda: random.choice(decisions)

    templates = [
        # Explicit comment requests
        lambda: f"Can someone add a comment on {task()} that {decision()}?",
        lambda: f"Add a note to {task()} saying {decision()}.",
        lambda: f"Please add a comment on {key()} — {decision()}.",
        lambda: f"Can you update the ticket for {task()} with a note that {decision()}?",
        lambda: f"Add a comment to {key()} that {decision()}.",
        lambda: f"Someone should note on {task()} that {decision()}.",
        lambda: f"Log a comment on {task()}: {decision()}.",
        lambda: f"Update {key()} with a comment — {decision()}.",
        lambda: f"Can you drop a note on {task()} that {decision()}?",
        lambda: f"Please document on {key()} that {decision()}.",
        # Decision logging
        lambda: f"We should document the decision about {task()} in the ticket.",
        lambda: f"I'll add a note to {task()} about the approach we discussed.",
        lambda: f"Let me comment on {key()} with the outcome of today's discussion.",
        lambda: f"I'll update the ticket for {task()} with what we decided.",
        lambda: f"Can you record the decision in {task()}?",
        # Context updates
        lambda: f"Add context to {key()} about the blockers we discussed.",
        lambda: f"Note in {task()} that we're waiting on the design team.",
        lambda: f"Update {task()} with a comment about the API changes.",
        lambda: f"I'll add a technical note to {key()} for future reference.",
        lambda: f"Can someone annotate {task()} with the edge cases we found?",
        # Implicit
        lambda: f"I'll note that in the ticket.",
        lambda: f"Let me add that to the ticket.",
        lambda: f"I'll document that decision in {key()}.",
        lambda: f"We should put that in the ticket for {task()}.",
        lambda: f"I'll leave a note on {key()} about this.",
        # Standup follow-ups
        lambda: f"After this call I'll update {task()} with the details.",
        lambda: f"I'll add a comment to {task()} after the standup.",
        lambda: f"Remind me to add a note on {key()} later.",
        lambda: f"I'll update the comments on {task()} with today's discussion.",
        lambda: f"Let's make sure to document this in {key()}.",
    ]

    while len(sentences) < 200:
        sentences.append(random.choice(templates)())

    return sentences[:200]


# ── Main ──────────────────────────────────────────────────────────────────────

LABEL_GENERATORS = {
    "create_task":   make_create_task,
    "complete_task": make_complete_task,
    "update_status": make_update_status,
    "assign_task":   make_assign_task,
    "add_comment":   make_add_comment,
}


def generate_dataset(output_path: str):
    rows = []

    for label, generator in LABEL_GENERATORS.items():
        print(f"  Generating {label}...")
        sentences = generator()
        for sentence in sentences:
            rows.append({"sentence": sentence, "label": label})

    # Shuffle so classes are interleaved
    random.shuffle(rows)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sentence", "label"])
        writer.writeheader()
        writer.writerows(rows)

    # Print class distribution
    from collections import Counter
    counts = Counter(r["label"] for r in rows)
    print(f"\nDataset saved to: {output_path}")
    print(f"Total rows: {len(rows)}")
    print("\nClass distribution:")
    for label, count in sorted(counts.items()):
        print(f"  {label:<20} {count}")


if __name__ == "__main__":
    here = os.path.dirname(__file__)
    output = os.path.join(here, "synthetic_scrum_data.csv")
    print("Generating synthetic Scrum dataset...\n")
    generate_dataset(output)
    print("\nDone. Upload synthetic_scrum_data.csv to Google Colab.")