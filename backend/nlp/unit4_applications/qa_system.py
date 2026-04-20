"""
=============================================================
Unit 4 — Extractive Question Answering (QA)
=============================================================
Uses a BERT-based QA model to answer questions about the backlog.

Models compared:
  distilbert-base-cased-distilled-squad (small, fast, CPU-friendly)
  deepset/roberta-base-squad2           (more accurate, slightly larger)

How it works:
  Input: (question, context_passage)
  Model: reads both, highlights the answer SPAN in the context
  Output: extracted text span + confidence score

Application:
  Context = serialized backlog / sprint data
  User asks: "Which Epic has the highest WSJF score?"
  Model finds and returns the answer from the backlog text

Syllabus: Unit 4 — Question Answering (QA) Systems:
          Closed-domain QA (backlog as context)
Run     : python backend/nlp/unit4_applications/qa_system.py
=============================================================
"""

from typing import List, Dict, Tuple


# ── Load QA model ─────────────────────────────────────────

QA_MODEL = "distilbert-base-cased-distilled-squad"


def _load_qa_pipeline():
    """
    Load QA model directly (transformers v5 removed 'question-answering' pipeline).
    Returns (tokenizer, model) tuple.
    """
    import torch
    from transformers import DistilBertTokenizerFast, DistilBertForQuestionAnswering
    print(f"  Loading QA model: {QA_MODEL} …")
    tokenizer = DistilBertTokenizerFast.from_pretrained(QA_MODEL)
    model     = DistilBertForQuestionAnswering.from_pretrained(QA_MODEL)
    model.eval()
    return tokenizer, model


# ── QA functions ──────────────────────────────────────────

def answer_question(
    question: str,
    context: str,
    qa_pipeline,
) -> Dict:
    """
    Find the answer to a question within a given context paragraph.

    The model identifies the start/end token positions of the answer
    span within the context — extractive QA (not generation).

    Args:
        question  : Natural language question
        context   : Text document containing the answer
        qa_pipeline: (tokenizer, model) tuple from _load_qa_pipeline()

    Returns:
        {question, answer, score, start, end}
    """
    import torch
    import torch.nn.functional as F
    tokenizer, model = qa_pipeline

    inputs = tokenizer(
        question,
        context,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        stride=128,
        return_offsets_mapping=False,
    )
    with torch.no_grad():
        outputs = model(**inputs)

    start_idx = int(outputs.start_logits.argmax())
    end_idx   = int(outputs.end_logits.argmax())
    # Ensure end >= start
    if end_idx < start_idx:
        end_idx = start_idx

    input_ids     = inputs["input_ids"][0]
    answer_tokens = input_ids[start_idx : end_idx + 1]
    answer        = tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()

    # Approximate confidence: mean of softmax scores at chosen positions
    start_score = float(F.softmax(outputs.start_logits, dim=1)[0][start_idx])
    end_score   = float(F.softmax(outputs.end_logits,   dim=1)[0][end_idx])
    score       = (start_score + end_score) / 2.0

    return {
        "question": question,
        "answer":   answer,
        "score":    round(score, 4),
        "start":    start_idx,
        "end":      end_idx,
    }


def batch_qa(
    questions: List[str],
    context: str,
    qa_pipeline,
) -> List[Dict]:
    """Answer multiple questions against the same context."""
    return [answer_question(q, context, qa_pipeline) for q in questions]


def build_backlog_context(epics: List[Dict]) -> str:
    """
    Serialize a list of Epic dicts into a readable context paragraph
    for the QA model.

    Args:
        epics: List of Epic dicts with keys: title, wsjf_score, priority_rank, description

    Returns:
        Formatted text context
    """
    lines = ["Product Backlog Summary:\n"]
    for i, epic in enumerate(epics, 1):
        lines.append(
            f"Epic {i}: {epic.get('title', 'Unknown')}. "
            f"WSJF Score: {epic.get('wsjf_score', 'N/A')}. "
            f"Priority Rank: {epic.get('priority_rank', 'N/A')}. "
            f"Description: {epic.get('description', '')}. "
            f"Business Value: {epic.get('business_value', 'N/A')}."
        )
    return "\n".join(lines)


# ── Standalone Demo ───────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("      UNIT 4 — QUESTION ANSWERING SYSTEM DEMO")
    print("=" * 65)
    print(f"\nModel  : {QA_MODEL} (Extractive QA)")
    print("Task   : Find answer spans in the backlog context")
    print("Type   : Closed-domain QA (context = product backlog)\n")

    # ── Simulated backlog context ─────────────────────────
    BACKLOG_CONTEXT = """
    Product Backlog Summary:

    Epic 1: Payment Gateway Integration. WSJF Score: 8.5. Priority Rank: 1.
    Description: Integrate Stripe payment for online checkout. Business Value: 9.
    This epic is critical for Q2 revenue generation and must be completed before launch.
    Estimated effort: 5 story points.

    Epic 2: User Authentication System. WSJF Score: 7.2. Priority Rank: 2.
    Description: Implement JWT-based secure login with MFA and SSO support.
    Business Value: 8. Time criticality: 9. Effort: 4 story points.
    The security audit requires this to be completed before June 1st.

    Epic 3: Analytics Dashboard. WSJF Score: 4.1. Priority Rank: 3.
    Description: Real-time sprint velocity and burndown charts for the team.
    Business Value: 6. Effort: 3 story points. Deferred to Q3.

    Epic 4: Notification System. WSJF Score: 3.8. Priority Rank: 4.
    Description: Email and SMS push notifications for task assignments.
    Business Value: 7. Effort: 5 story points. Planned for Q3 sprint 2.

    Current Sprint: Sprint 12. Team capacity: 80 hours. Velocity: 34 story points.
    Sprint goal: Complete the Payment Gateway Integration and start Authentication.
    Sprint start: April 14, 2026. Sprint end: April 28, 2026.
    Lead developer: Sarah handles the payment work. Mike handles the authentication.
    """

    qa = _load_qa_pipeline()

    # ── Questions about the backlog ───────────────────────
    questions = [
        "Which epic has the highest WSJF score?",
        "What is the priority rank of the Authentication System?",
        "Who is the lead developer for the payment work?",
        "When does the current sprint end?",
        "What is the sprint goal?",
        "What is the business value of the notification system?",
        "How many story points is the analytics dashboard estimated at?",
        "What is the team capacity for this sprint?",
    ]

    print("CONTEXT: Backlog text (serialized from JSON)")
    print("-" * 55)
    print(BACKLOG_CONTEXT.strip()[:300] + "…\n")

    print("Q&A RESULTS:")
    print("-" * 55)
    for q in questions:
        result = answer_question(q, BACKLOG_CONTEXT, qa)
        conf   = result["score"]
        status = "✓" if conf > 0.5 else "?"
        print(f"\n  {status} Q: {result['question']}")
        print(f"    A: {result['answer']!r}  (confidence={conf:.4f})")

    # ── Programmatic backlog context ──────────────────────
    print("\n[2] STRUCTURED BACKLOG as QA Context")
    print("-" * 55)
    epics = [
        {"title": "Payment Gateway", "wsjf_score": 8.5, "priority_rank": 1,
         "description": "Stripe checkout integration", "business_value": 9},
        {"title": "User Auth System", "wsjf_score": 7.2, "priority_rank": 2,
         "description": "JWT login with MFA",          "business_value": 8},
    ]
    ctx = build_backlog_context(epics)
    print(f"  Generated context:\n  {ctx}\n")
    q = "What is the WSJF score of the Payment Gateway?"
    r = answer_question(q, ctx, qa)
    print(f"  Q: {r['question']}")
    print(f"  A: {r['answer']}  (confidence={r['score']:.4f})")

    print("\nQA Model Notes:")
    print("  Extractive QA: answers are SPANS from the context (not generated)")
    print("  Model reads question + context → predicts start & end token positions")
    print("  For generative answers → use T5/GPT-2 (text_generator.py)")
    print()
