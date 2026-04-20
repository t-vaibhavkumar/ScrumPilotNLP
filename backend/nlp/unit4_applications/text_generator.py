"""
=============================================================
Unit 4 — Language Generation (T5 / Flan-T5)
=============================================================
Uses google/flan-t5-small to generate text given an instruction prompt.

T5 (Text-to-Text Transfer Transformer):
  • Encoder-Decoder architecture (like BART, unlike GPT which is decoder-only)
  • Every task is framed as text-to-text:
      Summarize: "summarize: {text}"
      Translate: "translate English to French: {text}"
      Generate : "generate epic description for: {title}"
  • Flan-T5: fine-tuned on 1800+ instruction tasks → better instruction following

Applications:
  • Generate Epic descriptions from just a title
  • Generate User Story acceptance criteria
  • Generate sprint retrospective action items

Syllabus: Unit 4 — Language Generation
          Unit 3 — Generative models: GPT-2/3, T5; prompt-based learning
Run     : python backend/nlp/unit4_applications/text_generator.py
=============================================================
"""

from typing import List, Dict


# ── Load model ────────────────────────────────────────────

MODEL_NAME = "google/flan-t5-small"   # ~80M params, runs on CPU


def _load_generator():
    """
    Load Flan-T5 directly (transformers v5 removed 'text2text-generation' pipeline).
    Returns (tokenizer, model) tuple.
    """
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    print(f"  Loading {MODEL_NAME} …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model


# ── Generation functions ──────────────────────────────────

def _generate(prompt: str, generator, max_new_tokens: int = 120) -> str:
    """Internal helper: run Flan-T5 generation on a single prompt."""
    import torch
    tokenizer, model = generator
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        output_ids = model.generate(
            inputs["input_ids"],
            max_new_tokens=max_new_tokens,
            num_beams=4,
            early_stopping=True,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


def generate_epic_description(title: str, generator) -> str:
    """
    Generate a software Epic description from just a title.

    Flan-T5-small works best with short fill-in-the-blank prompts.
    We use QA-style: question + partial answer that the model completes.

    Args:
        title    : Epic title string
        generator: (tokenizer, model) tuple from _load_generator()

    Returns:
        Generated description sentence
    """
    # T5 works well with Q&A style + constrained completion
    prompt = (
        f"Question: What is the goal of the {title} software feature?\n"
        f"Answer:"
    )
    result = _generate(prompt, generator, max_new_tokens=80)
    # Fallback: if result is empty/bad, use template
    if not result or len(result) < 5:
        result = f"Enable {title.lower()} capabilities for users."
    return result


def generate_user_story(feature: str, generator) -> str:
    """
    Generate a User Story in standard format from a feature description.

    User Story format:
      As a [user type], I want to [action] so that [benefit]

    T5-small works best with fill-in-the-blank style.

    Args:
        feature  : Feature description
        generator: (tokenizer, model) tuple from _load_generator()

    Returns:
        User Story string
    """
    # Feed the template start, let T5 complete the benefit clause
    prompt = (
        f"Complete the user story: "
        f"As a software user, I want to {feature}. The benefit is:"
    )
    benefit = _generate(prompt, generator, max_new_tokens=60)
    return f"As a software user, I want to {feature} so that {benefit}"


def generate_acceptance_criteria(story: str, generator) -> str:
    """
    Generate acceptance criteria for a User Story.

    Uses Q&A format: better suited for Flan-T5-small.

    Args:
        story    : User Story description
        generator: (tokenizer, model) tuple from _load_generator()

    Returns:
        Acceptance criteria string
    """
    prompt = (
        f"What are the acceptance criteria for this user story: {story}?"
    )
    return _generate(prompt, generator, max_new_tokens=120)


def batch_generate(prompts: List[str], generator) -> List[str]:
    """Generate text for a list of prompts."""
    return [_generate(p, generator, max_new_tokens=100) for p in prompts]


# ── Standalone Demo ───────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("       UNIT 4 — LANGUAGE GENERATION DEMO")
    print("=" * 65)
    print(f"\nModel      : {MODEL_NAME}")
    print("Type       : Text-to-Text (Encoder-Decoder)")
    print("Task       : Generate Scrum artifacts from titles\n")

    generator = _load_generator()

    # ── Epic description generation ───────────────────────
    print("[1] EPIC DESCRIPTION GENERATION")
    print("-" * 55)
    epic_titles = [
        "User Authentication System",
        "Payment Gateway Integration",
        "Real-Time Analytics Dashboard",
    ]
    for title in epic_titles:
        print(f"\n  Input  (title): {title}")
        desc = generate_epic_description(title, generator)
        print(f"  Output (desc) : {desc}")

    # ── User Story generation ─────────────────────────────
    print("\n[2] USER STORY GENERATION")
    print("-" * 55)
    features = [
        "users can log in using their Google account",
        "the payment page shows a secure checkout with Stripe",
    ]
    for feature in features:
        print(f"\n  Feature : {feature}")
        story = generate_user_story(feature, generator)
        print(f"  Story   : {story}")

    # ── Acceptance criteria generation ────────────────────
    print("\n[3] ACCEPTANCE CRITERIA GENERATION")
    print("-" * 55)
    story = "As a developer, I want to set up the CI/CD pipeline so that code is automatically tested"
    print(f"  User Story: {story}")
    criteria = generate_acceptance_criteria(story, generator)
    print(f"  Criteria  : {criteria}")

    # ── T5 vs GPT-2 architecture comparison ───────────────
    print("\n[4] T5 vs GPT-2 ARCHITECTURE")
    print("-" * 55)
    print(f"  {'Model':<10} {'Architecture':<20} {'Input':<20} {'Output'}")
    print("  " + "-" * 65)
    print(f"  {'T5':<10} {'Encoder-Decoder':<20} {'Full context':<20} 'text-to-text generation'")
    print(f"  {'GPT-2':<10} {'Decoder-only':<20} {'Left context':<20} 'autoregressive generation'")
    print(f"  {'BART':<10} {'Encoder-Decoder':<20} {'Full context':<20} 'summarization/translation'")
    print(f"  {'BERT':<10} {'Encoder-only':<20} {'Full context':<20} 'classification/QA'")
    print()
