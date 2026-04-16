"""
=============================================================
ScrumPilotNLP — Full End-to-End NLP Pipeline
=============================================================
Chains ALL 5 syllabus units on a single meeting transcript:

  Unit 1 → Preprocessing + Representations
  Unit 2 → LSTM classification + GRU action extraction
  Unit 3 → Sentence-BERT story mapping
  Unit 4 → Extractive + Abstractive summarization + QA
  Unit 5 → TTS meeting summary

Run: python nlp_demos/demo_full_pipeline.py
=============================================================
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── ANSI colour helpers (Windows-friendly) ────────────────
def banner(text: str) -> None:
    print(f"\n{'═' * 65}")
    print(f"  {text}")
    print("═" * 65)

def stage(n: int, title: str) -> None:
    print(f"\n{'─' * 65}")
    print(f"  STAGE {n} ▶  {title}")
    print("─" * 65)

def ok(msg: str) -> None:
    print(f"  ✓  {msg}")

def info(label: str, value) -> None:
    print(f"  {label:<28} {value}")


# ════════════════════════════════════════════════════════════
# INPUT: Simulated Whisper ASR transcript
# (In production this comes from backend/speech/whisperai/)
# ════════════════════════════════════════════════════════════

TRANSCRIPT = """
Sarah: Good morning team. Let's kick off today's standup.

Mike: Yesterday I completed the Stripe payment gateway integration
and pushed it to the staging environment. Today I will start writing
unit tests for the payment module. No blockers from my side.

Alice: I am still working on the CI pipeline setup with GitHub Actions.
It should be done by tomorrow. I am blocked waiting for DevOps to
provide the AWS credentials. That is my only blocker right now.

Tom: Yesterday I reviewed Mike's pull request for the payment feature.
Today I will start implementing the user authentication REST API endpoints.
I estimated eight story points for the authentication epic. No blockers.

Sarah: Great. So Mike completed payment, Alice is blocked on DevOps credentials,
Tom starts authentication today. Sprint ends this Friday April 25th.
Team capacity is 80 hours and we have 34 story points committed.
Sarah will organise the sprint review meeting for Friday afternoon.
"""

# ── Product backlog (story ID → title) ───────────────────
STORIES = [
    "Implement user authentication and secure login with JWT tokens",
    "Integrate Stripe payment gateway for online checkout processing",
    "Build real-time sprint analytics dashboard with burndown charts",
    "Set up CI/CD pipeline with GitHub Actions and Docker containers",
    "Create push notification system for email and SMS delivery",
    "Design REST API for user profile management and settings",
    "Add OAuth2 social login with Google and GitHub providers",
    "Build automated test suite with unit and integration tests",
]
STORY_IDS = [f"SP-{i+1:03d}" for i in range(len(STORIES))]

# Training data for meeting classifiers
from backend.nlp.unit2_models.lstm_classifier import TRAINING_DATA as LSTM_TRAIN
from backend.nlp.unit2_models.gru_classifier  import TRAINING_DATA as GRU_TRAIN, CLASS_NAMES as ACTION_CLASSES


# ════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    t0 = time.time()

    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║     SCRUMPILOT NLP — FULL END-TO-END PIPELINE            ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"\n  INPUT  : Raw standup meeting transcript")
    print(f"  WORDS  : {len(TRANSCRIPT.split())} words")
    print(f"\n  Pipeline: Unit1 → Unit2 → Unit3 → Unit4 → Unit5\n")
    print(TRANSCRIPT.strip())

    # ════════════════════════════════════════════════════════
    # STAGE 1 — PREPROCESSING  (Unit 1)
    # ════════════════════════════════════════════════════════
    stage(1, "PREPROCESSING  (Unit 1 — spaCy / NLTK / BERT)")

    from backend.nlp.unit1_preprocessing.normalizer  import normalize
    from backend.nlp.unit1_preprocessing.tokenizer   import word_tok, sentence_tok
    from backend.nlp.unit1_preprocessing.lemmatizer  import extract_lemmas
    from backend.nlp.unit1_preprocessing.ner         import extract_entities, extract_assignees, extract_estimates, extract_dates

    raw_tokens  = word_tok(TRANSCRIPT)
    sentences   = sentence_tok(TRANSCRIPT)
    normalized  = normalize(TRANSCRIPT, remove_stops=True)
    norm_tokens = word_tok(normalized)
    key_lemmas  = extract_lemmas(TRANSCRIPT, pos_filter=["NOUN", "VERB"])[:12]
    entities    = extract_entities(TRANSCRIPT)
    assignees   = list(dict.fromkeys(extract_assignees(TRANSCRIPT)))  # deduplicated
    estimates   = extract_estimates(TRANSCRIPT)
    dates       = extract_dates(TRANSCRIPT)

    info("Raw token count:", len(raw_tokens))
    info("After normalisation:", len(norm_tokens))
    info("Sentences detected:", len(sentences))
    info("Key lemmas:", key_lemmas)
    info("People (NER):", assignees)
    info("Estimates (NER):", estimates)
    info("Dates (NER):", dates[:4])
    ok("Preprocessing complete")

    # ════════════════════════════════════════════════════════
    # STAGE 2 — MEETING TYPE CLASSIFICATION  (Unit 2 — LSTM)
    # ════════════════════════════════════════════════════════
    stage(2, "MEETING TYPE  (Unit 2 — LSTM Classifier)")

    from backend.nlp.unit2_models.lstm_classifier import train as lstm_train, predict as lstm_predict
    LSTM_CLASS_NAMES = ["PM_MEETING", "SPRINT_PLANNING", "STANDUP"]

    print("  Training LSTM on meeting corpus …")
    lstm_model, lstm_vocab = lstm_train(LSTM_TRAIN, epochs=40, lr=1e-3)

    # Classify the whole transcript using the opening lines
    sample = " ".join(sentences[:3])
    result = lstm_predict(lstm_model, lstm_vocab, sample)
    meeting_type = result["prediction"]

    info("Meeting type:", f"{meeting_type}  (confidence={result['confidence']:.3f})")
    ok(f"Classified as: {meeting_type}")

    # ════════════════════════════════════════════════════════
    # STAGE 3 — ACTION EXTRACTION  (Unit 2 — GRU per sentence)
    # ════════════════════════════════════════════════════════
    stage(3, "ACTION EXTRACTION  (Unit 2 — Bidirectional GRU)")

    from backend.nlp.unit2_models.gru_classifier import train as gru_train, predict as gru_predict

    print("  Training GRU action classifier …")
    gru_model, gru_vocab = gru_train(GRU_TRAIN, epochs=50, lr=1e-3)

    # Run NER and GRU on each sentence together
    print("\n  Sentence-level action analysis:")
    print(f"  {'Sentence':<52} {'Action':<16} {'Conf':>6}")
    print("  " + "─" * 77)

    from backend.nlp.unit1_preprocessing.ner import extract_assignees as sent_assignees

    actions_found = []
    for sent in sentences:
        sent = sent.strip()
        if len(sent.split()) < 4:
            continue
        gru_res = gru_predict(gru_model, gru_vocab, sent)
        action  = gru_res["prediction"]
        conf    = gru_res["confidence"]
        people  = sent_assignees(sent)

        if action != "no_action":
            actor = people[0] if people else "?"
            actions_found.append({
                "sentence": sent,
                "action":   action,
                "actor":    actor,
                "conf":     conf,
            })
            short = sent[:50] + "…" if len(sent) > 50 else sent
            print(f"  {short:<52} [{actor}] {action:<14} {conf:>6.3f}")

    ok(f"Extracted {len(actions_found)} actionable sentences")

    # ════════════════════════════════════════════════════════
    # STAGE 4 — STORY MAPPING  (Unit 3 — Sentence-BERT)
    # ════════════════════════════════════════════════════════
    stage(4, "STORY MAPPING  (Unit 3 — Sentence-BERT)")

    from backend.nlp.unit3_transformers.sentence_bert import _load_sbert, find_matching_story

    print(f"  Loading Sentence-BERT …")
    sbert = _load_sbert()

    print("\n  Mapping actions → Story IDs:")
    print(f"  {'Actor':<8} {'Action':<16} {'Top Story':<10} {'Score':>6}  Title")
    print("  " + "─" * 80)

    for a in actions_found:
        matches = find_matching_story(a["sentence"], STORIES, STORY_IDS, sbert, top_k=1, threshold=0.0)
        if matches:
            m = matches[0]
            a["story_id"]    = m["story_id"]
            a["story_title"] = m["title"]
            a["story_score"] = m["score"]
            print(f"  {a['actor']:<8} {a['action']:<16} {m['story_id']:<10} {m['score']:>6.3f}  {m['title'][:40]}")

    ok("Story mapping complete")

    # ════════════════════════════════════════════════════════
    # STAGE 5 — SUMMARIZATION  (Unit 4 — Extractive + BART)
    # ════════════════════════════════════════════════════════
    stage(5, "SUMMARIZATION  (Unit 4 — TF-IDF Extractive + DistilBART)")

    from backend.nlp.unit4_applications.summarizer import extractive_summarize, _load_bart, abstractive_summarize

    ext_summary = extractive_summarize(TRANSCRIPT, n_sentences=3)
    print(f"\n  [EXTRACTIVE — TF-IDF sentence scoring]")
    print(f"  {ext_summary}\n")

    print(f"  [ABSTRACTIVE — DistilBART generation]")
    print(f"  Loading DistilBART …")
    try:
        bart        = _load_bart()
        abs_summary = abstractive_summarize(TRANSCRIPT.strip(), bart, max_length=80, min_length=20)
        print(f"  {abs_summary}")
    except Exception as e:
        abs_summary = ext_summary
        print(f"  (BART unavailable: {e} — using extractive summary)")

    ok("Summarization complete")

    # ════════════════════════════════════════════════════════
    # STAGE 6 — QA OVER TRANSCRIPT  (Unit 4 — DistilBERT-SQuAD)
    # ════════════════════════════════════════════════════════
    stage(6, "QUESTION ANSWERING  (Unit 4 — DistilBERT-SQuAD)")

    from backend.nlp.unit4_applications.qa_system import _load_qa_pipeline, answer_question

    print("  Loading QA model …")
    qa = _load_qa_pipeline()

    qa_questions = [
        "Who is blocked?",
        "What did Mike complete?",
        "When does the sprint end?",
        "Who will start authentication?",
        "What is the team capacity?",
    ]

    print(f"\n  {'Question':<40} {'Answer':<30} {'Conf':>6}")
    print("  " + "─" * 80)
    for q in qa_questions:
        r    = answer_question(q, TRANSCRIPT, qa)
        conf = r["score"]
        mark = "✓" if conf > 0.3 else "?"
        print(f"  {mark} {q:<38} {r['answer']!r:<30} {conf:>6.4f}")

    ok("QA complete")

    # ════════════════════════════════════════════════════════
    # STAGE 7 — STRUCTURED MEETING REPORT
    # ════════════════════════════════════════════════════════
    stage(7, "MEETING REPORT GENERATION")

    print(f"""
  ╔══════════════════════════════════════════════════════╗
  ║           STANDUP MEETING REPORT                     ║
  ╚══════════════════════════════════════════════════════╝

  Meeting Type  : {meeting_type}
  Participants  : {', '.join(assignees)}
  Estimates     : {estimates}
  Dates         : {dates[:2]}

  ACTION ITEMS:""")

    for a in actions_found:
        story = a.get("story_id", "?")
        print(f"    [{a['action']:<15}]  {a['actor']:<8}  → {story}  {a.get('story_title','')[:40]}")

    print(f"""
  BLOCKERS:
    Alice — Waiting for DevOps AWS credentials

  SUMMARY:
    {abs_summary[:200]}

  Key Lemmas: {key_lemmas[:8]}
    """)

    ok("Report generated")

    # ════════════════════════════════════════════════════════
    # STAGE 8 — TEXT-TO-SPEECH  (Unit 5)
    # ════════════════════════════════════════════════════════
    stage(8, "TEXT-TO-SPEECH  (Unit 5 — pyttsx3 + gTTS)")

    from backend.nlp.unit5_speech.tts import speak_offline, speak_gtts

    tts_text = (
        f"Standup meeting report for {', '.join(assignees[:3])} and team. "
        f"Meeting type: {meeting_type}. "
        f"{len(actions_found)} action items identified. "
        + abs_summary[:150]
    )

    print(f"\n  Speaking: '{tts_text[:100]}…'")

    try:
        speak_offline(tts_text, rate=150)
        ok("Spoken via pyttsx3 (system TTS)")
    except Exception as e:
        print(f"  pyttsx3 error: {e}")

    os.makedirs("nlp_demos/output", exist_ok=True)
    try:
        mp3_path = speak_gtts(tts_text, save_path="nlp_demos/output/meeting_report.mp3")
        ok(f"Saved MP3 → {mp3_path}")
    except Exception as e:
        print(f"  gTTS: {e}")

    # ════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ════════════════════════════════════════════════════════
    elapsed = time.time() - t0
    banner(f"PIPELINE COMPLETE  ({elapsed:.1f}s)")

    print(f"""
  ┌─────────────────────────────────────────────────────┐
  │  UNIT 1  Preprocessing + NER + TF-IDF retrieval     │
  │  UNIT 2  LSTM meeting type + GRU action extraction  │
  │  UNIT 3  Sentence-BERT story ID mapping             │
  │  UNIT 4  DistilBART summary + DistilBERT QA         │
  │  UNIT 5  pyttsx3 spoken report + gTTS MP3           │
  └─────────────────────────────────────────────────────┘

  Input    : {len(TRANSCRIPT.split())} word standup transcript
  Output   : Structured meeting report + spoken summary
  API calls: ZERO (100% offline, 100% CPU)
    """)
