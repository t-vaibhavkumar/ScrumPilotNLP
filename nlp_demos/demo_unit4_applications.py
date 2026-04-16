"""
=============================================================
Demo — Unit 4: NLP Applications (Summarization, QA, Generation)
=============================================================
Run: python nlp_demos/demo_unit4_applications.py
=============================================================
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.nlp.unit4_applications.summarizer     import extractive_summarize
from backend.nlp.unit4_applications.qa_system      import _load_qa_pipeline, answer_question, build_backlog_context
from backend.nlp.unit4_applications.text_generator import _load_generator, generate_epic_description, generate_user_story


TRANSCRIPT = """
Sarah: Good morning everyone. Let's review the Q2 product priorities.
The main requirement from stakeholders is the payment gateway. It has
a business value of nine out of ten and is critical for revenue.
Without it we cannot process any online transactions.

Tech Lead: I agree. The authentication system is also urgent.
We have security vulnerabilities that must be addressed before the audit.
Business value is eight and time criticality is nine.

Product Manager: The analytics dashboard is lower priority — business value
of six. We can defer it to Q3. The notification system is medium priority,
business value seven, useful for user retention.

Sarah: So our priorities are: first the payment gateway, second authentication,
third notifications, and the dashboard goes to Q3. Agreed?

Team: Yes, agreed. The payment feature is the most critical for Q2.
"""

BACKLOG_CONTEXT = """
The highest WSJF score belongs to the Payment Gateway Integration with a score of 8.5.
Epic 1: Payment Gateway Integration. WSJF Score: 8.5. Priority Rank: 1.
Business Value: 9. Effort: 5 story points. Critical for Q2 revenue.
The payment feature must be delivered before June launch.

Epic 2: User Authentication System. WSJF Score: 7.2. Priority Rank: 2.
Business Value: 8. Time Criticality: 9. Effort: 4 story points.
Required before security audit on June 1st.

Epic 3: Analytics Dashboard. WSJF Score: 4.1. Priority Rank: 3.
Business Value: 6. Effort: 3 story points. Deferred to Q3.

Epic 4: Notification System. WSJF Score: 3.8. Priority Rank: 4.
Business Value: 7. Effort: 5 story points. Planned for Q3 Sprint 2.

Sarah handles the payment work. Mike handles the authentication work.
Team capacity: 80 hours. Sprint end: April 28, 2026.
"""


def section(title: str) -> None:
    print(f"\n{'─' * 65}")
    print(f"  {title}")
    print("─" * 65)


if __name__ == "__main__":
    os.makedirs("nlp_demos/output", exist_ok=True)

    print("╔══════════════════════════════════════════════════════════╗")
    print("║     UNIT 4 — NLP APPLICATIONS DEMO                      ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # ── Extractive Summarization ──────────────────────────
    section("1. EXTRACTIVE SUMMARIZATION  (TF-IDF sentence scoring)")
    print(f"  Input: {len(TRANSCRIPT.split())} words of PM meeting transcript")
    ext_summary = extractive_summarize(TRANSCRIPT, n_sentences=3)
    print(f"\n  Extractive Summary (3 sentences):")
    print(f"  {ext_summary}")
    print("\n  + Verbatim sentences from original")
    print("  + Fast, no neural model needed")
    print("  − Choppy, may miss cross-sentence context")

    # ── Abstractive Summarization (BART) ──────────────────
    section("2. ABSTRACTIVE SUMMARIZATION  (DistilBART)")
    print("  Loading DistilBART …")
    try:
        from backend.nlp.unit4_applications.summarizer import _load_bart, abstractive_summarize as abs_sum
        bart = _load_bart()
        abs_summary = abs_sum(TRANSCRIPT.strip(), bart)
        print(f"\n  Abstractive Summary:")
        print(f"  {abs_summary}")
        print("\n  + Fluent, human-like paragraph")
        print("  + Combines information from multiple sentences")
    except Exception as e:
        print(f"  BART not available: {e}")

    # ── Extractive QA ─────────────────────────────────────
    section("3. QUESTION ANSWERING  (DistilBERT SQuAD)")
    print("  Loading QA model …")
    qa = _load_qa_pipeline()

    questions = [
        "Which epic has the highest WSJF score?",
        "What is the priority rank of the Authentication System?",
        "Who handles the payment work?",
        "When does the sprint end?",
        "What is the team capacity?",
    ]

    print(f"\n  Context: backlog text ({len(BACKLOG_CONTEXT.split())} words)")
    print("  Q&A Results:")
    for q in questions:
        result = answer_question(q, BACKLOG_CONTEXT, qa)
        conf   = result["score"]
        marker = "✓" if conf > 0.4 else "?"
        print(f"\n    {marker} Q: {q}")
        print(f"       A: {result['answer']!r}  (conf={conf:.4f})")

    # ── Language Generation ───────────────────────────────
    section("4. LANGUAGE GENERATION  (Flan-T5-small)")
    print("  Loading Flan-T5 …")
    generator = _load_generator()

    epic_titles = [
        "User Authentication System",
        "Payment Gateway Integration",
    ]
    print("\n  Epic Description Generation:")
    for title in epic_titles:
        desc = generate_epic_description(title, generator)
        print(f"\n    Title : {title}")
        print(f"    Output: {desc}")

    print("\n  User Story Generation:")
    feature = "users can log in using their Google accounts via OAuth2"
    story = generate_user_story(feature, generator)
    print(f"    Feature: {feature}")
    print(f"    Story  : {story}")

    print("\n✓ Unit 4 Applications demo complete!\n")
