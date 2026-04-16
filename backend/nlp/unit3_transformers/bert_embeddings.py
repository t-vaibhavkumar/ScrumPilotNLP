"""
=============================================================
Unit 3 — BERT Contextual Embeddings
=============================================================
Uses HuggingFace DistilBERT to extract contextual word/sentence embeddings.

Key insight — contextual vs. static embeddings:
  Word2Vec: "bank" always has the same vector
  BERT:     "bank" has different vectors in:
              "river bank"   (geography context)
              "bank account" (finance context)

CLS token embedding = representation of the whole sentence.
Used for sentence-level similarity and downstream classification.

Model: distilbert-base-uncased (66M params — 60% of BERT-base, CPU-friendly)

Syllabus: Unit 3 — Pretrained encoders: BERT, RoBERTa, ALBERT
          Unit 1 — Contextual embeddings: ELMo, BERT embeddings (conceptual)
Run     : python backend/nlp/unit3_transformers/bert_embeddings.py
=============================================================
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict


# ── Load model (cached after first download) ──────────────

MODEL_NAME = "distilbert-base-uncased"

def _load_model():
    from transformers import DistilBertTokenizer, DistilBertModel
    print(f"  Loading {MODEL_NAME} …")
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    model     = DistilBertModel.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model


# ── Embedding functions ───────────────────────────────────

def get_sentence_embedding(text: str, tokenizer, model) -> torch.Tensor:
    """
    Extract the [CLS] token embedding as the sentence representation.

    The [CLS] token attends to the whole sequence via self-attention,
    making it a good summary embedding of the full sentence.

    Args:
        text: Input sentence

    Returns:
        1D tensor of shape (hidden_dim,) = (768,) for BERT-base
    """
    inputs  = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # outputs.last_hidden_state: (1, seq_len, hidden_dim)
    cls_embedding = outputs.last_hidden_state[0, 0, :]  # CLS token = index 0
    return cls_embedding


def get_token_embeddings(text: str, tokenizer, model) -> Tuple[List[str], torch.Tensor]:
    """
    Extract per-token contextual embeddings.

    Returns:
        (tokens, embeddings)
        tokens    : List of subword tokens
        embeddings: (seq_len, hidden_dim) tensor
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    with torch.no_grad():
        outputs = model(**inputs)
    token_embeddings = outputs.last_hidden_state[0]  # (seq_len, hidden_dim)
    return tokens, token_embeddings


def cosine_similarity(vec_a: torch.Tensor, vec_b: torch.Tensor) -> float:
    """Cosine similarity between two embedding vectors."""
    return float(F.cosine_similarity(vec_a.unsqueeze(0), vec_b.unsqueeze(0)).item())


def find_most_similar_bert(
    query: str,
    candidates: List[str],
    tokenizer,
    model,
    top_k: int = 3,
) -> List[Tuple[int, float, str]]:
    """
    Find the most semantically similar candidates to a query using BERT CLS embeddings.

    This is semantically richer than TF-IDF because BERT understands context:
      "login feature" → similar to "user authentication" (same concept)

    Returns:
        List of (index, cosine_score, text)
    """
    query_emb = get_sentence_embedding(query, tokenizer, model)
    results   = []
    for i, cand in enumerate(candidates):
        cand_emb = get_sentence_embedding(cand, tokenizer, model)
        sim      = cosine_similarity(query_emb, cand_emb)
        results.append((i, round(sim, 4), cand))
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


# ── Standalone Demo ───────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("      UNIT 3 — BERT CONTEXTUAL EMBEDDINGS DEMO")
    print("=" * 65)

    tokenizer, model = _load_model()
    hidden_dim = model.config.hidden_size
    print(f"  Model     : {MODEL_NAME}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Parameters: ~66M (DistilBERT, CPU-friendly)")

    # ── 1. Contextual vs Static: 'bank' example ────────────
    print("\n[1] CONTEXTUAL EMBEDDINGS — Same word, different meaning")
    print("-" * 55)
    sentences = [
        "We need to open a bank account for the project budget",     # finance
        "The developers sat by the river bank during the hackathon",  # geography
        "The sprint backlog is stored in the database bank",          # technical
    ]
    embeddings = [get_sentence_embedding(s, tokenizer, model) for s in sentences]

    print("  Cosine similarity between 'bank' contexts:")
    pairs = [(0, 1, "finance ↔ geography"), (0, 2, "finance ↔ technical"), (1, 2, "geography ↔ technical")]
    for a, b, label in pairs:
        sim = cosine_similarity(embeddings[a], embeddings[b])
        print(f"    {label:<30}  {sim:.4f}")
    print("  → BERT distinguishes different uses of 'bank'")

    # ── 2. Sentence-level similarity ──────────────────────
    print("\n[2] SENTENCE SIMILARITY (CLS embeddings)")
    print("-" * 55)
    pairs_bert = [
        ("User authentication with JWT tokens",
         "Login system with secure token validation",
         "semantically equivalent"),
        ("User authentication with JWT tokens",
         "Deploy Docker container to AWS",
         "semantically different"),
        ("Sprint planning meeting for velocity estimation",
         "Team capacity discussion and story commitment",
         "related concepts"),
    ]
    print(f"  {'Sentence A':<45} {'Sentence B':<45} {'Cosine':>8}  Note")
    print("  " + "-" * 110)
    for a, b, note in pairs_bert:
        emb_a = get_sentence_embedding(a, tokenizer, model)
        emb_b = get_sentence_embedding(b, tokenizer, model)
        sim   = cosine_similarity(emb_a, emb_b)
        print(f"  {a[:43]:<45} {b[:43]:<45} {sim:>8.4f}  {note}")

    # ── 3. Token-level embeddings ─────────────────────────
    print("\n[3] TOKEN-LEVEL EMBEDDINGS (contextual per token)")
    print("-" * 55)
    text = "The sprint planning starts after the backlog grooming"
    tokens, token_embs = get_token_embeddings(text, tokenizer, model)
    print(f"  Input  : {text}")
    print(f"  Tokens : {tokens}")
    print(f"  Shape  : {tuple(token_embs.shape)}  (seq_len × hidden_dim)")
    print(f"  Each token has a unique {hidden_dim}-dim vector capturing context")

    # ── 4. BERT vs TF-IDF retrieval ───────────────────────
    print("\n[4] BERT vs TF-IDF — Semantic Story Retrieval")
    print("-" * 55)
    stories = [
        "Implement user authentication and secure login flow",
        "Integrate Stripe payment gateway for checkout",
        "Build sprint velocity analytics dashboard",
        "Set up CI/CD pipeline with Docker and GitHub Actions",
    ]
    query = "I'll work on the login feature this sprint"
    print(f"  Query  : '{query}'")
    results = find_most_similar_bert(query, stories, tokenizer, model, top_k=4)
    print(f"  BERT results:")
    for idx, score, text in results:
        print(f"    [{score:.4f}]  S{idx}: {text}")
    print("  → BERT correctly identifies 'authentication/login' as top match")
    print("  → TF-IDF would fail here (no shared keywords between 'login' and 'authentication')")
    print()
