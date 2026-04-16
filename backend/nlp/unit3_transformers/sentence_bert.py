"""
=============================================================
Unit 3 — Sentence-BERT (Semantic Similarity)
=============================================================
Uses sentence-transformers to produce sentence-level embeddings
optimised for semantic similarity comparison.

Why Sentence-BERT over plain BERT?
  BERT [CLS] embeddings are NOT optimised for similarity tasks.
  Sentence-BERT (SBERT) is fine-tuned with a siamese network on
  NLI + STS datasets → embeddings are directly comparable.

Application in ScrumPilot:
  Sprint Planning: "I'll handle the login this sprint"
  → SBERT cosine similarity → finds "User Authentication Story" (SP-001)
  This replaces the LLM API call for natural language → story ID mapping.

Model: all-MiniLM-L6-v2 (22M params, ~5× faster than BERT, CPU-friendly)

Syllabus: Unit 3 — HuggingFace Transformers Library
          Unit 3 — Modern embedding applications
Run     : python backend/nlp/unit3_transformers/sentence_bert.py
=============================================================
"""

from typing import List, Tuple, Dict
import torch
import torch.nn.functional as F


# ── Load Sentence-Transformers model ─────────────────────

MODEL_NAME = "all-MiniLM-L6-v2"

def _patch_torchcodec():
    """
    Monkey-patch stub torchcodec modules so sentence_transformers >= 3.x
    does not crash on Windows when FFmpeg DLLs are missing.
    We only use SBERT for text embeddings — video/audio codecs not needed.
    """
    import sys
    import types
    from importlib.machinery import ModuleSpec

    # Try the real package first — only stub if it explodes at import time
    try:
        import torchcodec  # noqa: F401
        return                # FFmpeg present & working — nothing to do
    except (RuntimeError, OSError, ImportError, Exception):
        pass

    def _stub(name, attrs=None):
        """Create a lightweight stub and register it in sys.modules."""
        if name in sys.modules:
            return sys.modules[name]
        mod = types.ModuleType(name)
        # ← key fix: give the stub a real __spec__ so Python doesn't warn
        mod.__spec__    = ModuleSpec(name=name, loader=None)
        mod.__package__ = name.rsplit(".", 1)[0] if "." in name else name
        mod.__path__    = []           # marks it as a package
        if attrs:
            for k, v in attrs.items():
                setattr(mod, k, v)
        sys.modules[name] = mod
        return mod

    # Minimal stubs that satisfy sentence_transformers' import chain
    _stub("torchcodec")
    _stub("torchcodec._core")
    _stub("torchcodec._core.ops")
    _stub("torchcodec._core._metadata")
    _stub("torchcodec.encoders")
    _stub("torchcodec.samplers")
    _stub("torchcodec.transforms")
    _stub("torchcodec.decoders", {
        "AudioDecoder": type("AudioDecoder", (), {}),
        "VideoDecoder": type("VideoDecoder", (), {}),
    })
    _stub("torchcodec.decoders._core", {
        "AudioStreamMetadata": type("AudioStreamMetadata", (), {}),
        "VideoStreamMetadata": type("VideoStreamMetadata", (), {}),
    })


def _load_sbert():
    _patch_torchcodec()          # ← must run before sentence_transformers import
    from sentence_transformers import SentenceTransformer
    print(f"  Loading {MODEL_NAME} …")
    return SentenceTransformer(MODEL_NAME)


# ── Core functions ────────────────────────────────────────

def encode_sentences(sentences: List[str], model) -> "torch.Tensor":
    """
    Encode a list of sentences into dense embedding vectors.

    Args:
        sentences: List of text strings
        model    : Loaded SentenceTransformer model

    Returns:
        Tensor of shape (n_sentences, embedding_dim)
        embedding_dim = 384 for all-MiniLM-L6-v2
    """
    embeddings = model.encode(sentences, convert_to_tensor=True, show_progress_bar=False)
    return embeddings


def semantic_similarity(text_a: str, text_b: str, model) -> float:
    """
    Compute the semantic similarity between two texts.

    Returns:
        Cosine similarity score in [-1, 1] (usually 0–1 for natural text)
    """
    embs = encode_sentences([text_a, text_b], model)
    sim  = F.cosine_similarity(embs[0].unsqueeze(0), embs[1].unsqueeze(0))
    return round(float(sim.item()), 4)


def find_matching_story(
    utterance: str,
    stories: List[str],
    story_ids: List[str],
    model,
    top_k: int = 3,
    threshold: float = 0.3,
) -> List[Dict]:
    """
    Find the most semantically matching User Stories for a spoken utterance.

    Sprint Planning use case:
      Input : "I'll work on the login feature"
      Output: [{"story_id": "SP-001", "title": "User Authentication", "score": 0.72}]

    Args:
        utterance : Natural language phrase from the meeting
        stories   : User Story titles/descriptions
        story_ids : Corresponding Jira story IDs (e.g. SP-001)
        model     : Loaded SentenceTransformer
        top_k     : Return this many results
        threshold : Minimum similarity score to include

    Returns:
        List of {story_id, title, score} dicts sorted by score
    """
    utt_emb    = encode_sentences([utterance], model)[0]
    story_embs = encode_sentences(stories, model)

    results = []
    for i, (story_id, story) in enumerate(zip(story_ids, stories)):
        sim = float(F.cosine_similarity(utt_emb.unsqueeze(0), story_embs[i].unsqueeze(0)).item())
        if sim >= threshold:
            results.append({
                "story_id": story_id,
                "title":    story,
                "score":    round(sim, 4),
            })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def semantic_clustering(sentences: List[str], model, n_clusters: int = 3) -> Dict:
    """
    Group semantically similar sentences using K-Means on SBERT embeddings.

    Useful for grouping Epics / User Stories into themes.

    Returns:
        Dict mapping cluster_id → list of sentences
    """
    from sklearn.cluster import KMeans
    import numpy as np

    embeddings = encode_sentences(sentences, model).cpu().numpy()
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(embeddings)

    clusters: Dict[int, List[str]] = {i: [] for i in range(n_clusters)}
    for sent, label in zip(sentences, labels):
        clusters[int(label)].append(sent)
    return clusters


# ── Standalone Demo ───────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("      UNIT 3 — SENTENCE-BERT DEMO")
    print("=" * 65)

    sbert = _load_sbert()
    print(f"  Model         : {MODEL_NAME}")
    print(f"  Embedding dim : 384")
    print(f"  Parameters    : ~22M  (very fast on CPU)")

    # ── User Story database ───────────────────────────────
    STORY_IDS = ["SP-001", "SP-002", "SP-003", "SP-004", "SP-005", "SP-006", "SP-007", "SP-008"]
    STORIES   = [
        "Implement user authentication and secure login with JWT tokens",
        "Integrate Stripe payment gateway for online checkout processing",
        "Build real-time sprint analytics dashboard with burndown charts",
        "Set up CI/CD pipeline with GitHub Actions and Docker containers",
        "Create push notification system for email and SMS delivery",
        "Design REST API for user profile management and settings",
        "Add OAuth2 social login with Google and GitHub providers",
        "Build automated test suite with unit and integration tests",
    ]

    # ── 1. Sprint planning utterance mapping ──────────────
    print("\n[1] SPRINT PLANNING — Natural Language → Story ID")
    print("    (Replaces LLM API for story mapping)")
    print("-" * 55)

    utterances = [
        "I'll work on the login feature this sprint",
        "Let's pull in the payment work for Q2",
        "I can handle the authentication this week",
        "Who's doing the CI pipeline setup?",
        "Sarah will take the testing stories",
    ]

    for utt in utterances:
        matches = find_matching_story(utt, STORIES, STORY_IDS, sbert, top_k=2)
        print(f"\n  Speaker: '{utt}'")
        for m in matches:
            print(f"    → {m['story_id']}  [{m['score']:.4f}]  {m['title']}")

    # ── 2. Sentence similarity pairs ──────────────────────
    print("\n[2] SENTENCE SIMILARITY PAIRS")
    print("-" * 55)
    pairs = [
        ("User authentication with JWT tokens",
         "Login system with secure token-based access",
         "semantically equivalent"),
        ("Stripe payment gateway integration",
         "Online checkout processing with credit cards",
         "same concept, different words"),
        ("Sprint planning meeting",
         "User authentication with JWT",
         "unrelated concepts"),
        ("Build CI/CD pipeline",
         "Set up automated deployment workflow",
         "near-synonym"),
    ]
    print(f"  {'Pair':<5} {'Score':>7}  {'Note'}")
    print("  " + "-" * 60)
    for i, (a, b, note) in enumerate(pairs):
        sim = semantic_similarity(a, b, sbert)
        print(f"  #{i+1}     {sim:>7.4f}  {note}")
        print(f"         A: {a[:60]}")
        print(f"         B: {b[:60]}")

    # ── 3. Semantic clustering of Epics ───────────────────
    print("\n[3] SEMANTIC CLUSTERING — Group Epics by Theme")
    print("-" * 55)
    epics = [
        "Build user authentication with SSO and JWT",
        "Implement social login with OAuth2 for Google",
        "Integrate Stripe payment gateway for checkout",
        "Add PayPal payment method and invoice generation",
        "Build sprint analytics dashboard with charts",
        "Create burndown chart for sprint tracking",
        "Set up CI/CD pipeline with GitHub Actions",
        "Automate Docker container deployment to AWS",
    ]
    clusters = semantic_clustering(epics, sbert, n_clusters=3)
    print("  Semantic clusters (auto-discovered themes):")
    for cluster_id, texts in clusters.items():
        print(f"\n  Cluster {cluster_id}:")
        for text in texts:
            print(f"    • {text}")

    # ── 4. SBERT vs TF-IDF comparison ─────────────────────
    print("\n[4] SBERT vs TF-IDF COMPARISON")
    print("-" * 55)
    query = "user login and signup with secure access"
    print(f"  Query: '{query}'")
    print(f"  Expected match: '{STORIES[0]}' (authentication/login)")

    # SBERT
    matches = find_matching_story(query, STORIES, STORY_IDS, sbert, top_k=1, threshold=0.0)
    print(f"\n  SBERT top match : {matches[0]['story_id']} [{matches[0]['score']:.4f}] {matches[0]['title']}")

    # TF-IDF
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
    import numpy as np
    tv = TfidfVectorizer(stop_words="english")
    tfidf_mat = tv.fit_transform(STORIES)
    q_vec = tv.transform([query])
    tfidf_sims = sk_cosine(q_vec, tfidf_mat)[0]
    best_idx = int(np.argmax(tfidf_sims))
    print(f"  TF-IDF top match: {STORY_IDS[best_idx]} [{tfidf_sims[best_idx]:.4f}] {STORIES[best_idx]}")
    print("\n  → SBERT captures semantic meaning; TF-IDF relies on word overlap")
    print()
