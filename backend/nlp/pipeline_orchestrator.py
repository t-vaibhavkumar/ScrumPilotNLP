"""
=============================================================
NLP Pipeline Orchestrator
=============================================================
Central bridge that takes raw transcript text and runs it
through ALL NLP units, returning a structured dict that
feeds into the existing ApprovalService + JiraAgent.

Usage:
    from backend.nlp.pipeline_orchestrator import NLPOrchestrator

    orch   = NLPOrchestrator()
    result = orch.run(transcript_text)

    # result["meeting_type"]    → "STANDUP" | "PM_MEETING" | ...
    # result["approval_payload"] → ready for ApprovalService
    # result["summary"]          → BART abstractive summary
    # result["blockers"]         → list of blocker sentences

All models are loaded lazily and cached between calls.
=============================================================
"""

import logging
import time
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Story backlog (loaded from DB/config in production) ───
# Using demo stories here; replace with real Jira query
DEFAULT_STORIES = [
    "Implement user authentication and secure login with JWT tokens",
    "Integrate Stripe payment gateway for online checkout processing",
    "Build real-time sprint analytics dashboard with burndown charts",
    "Set up CI/CD pipeline with GitHub Actions and Docker containers",
    "Create push notification system for email and SMS delivery",
    "Design REST API for user profile management and settings",
    "Add OAuth2 social login with Google and GitHub providers",
    "Build automated test suite with unit and integration tests",
]
DEFAULT_STORY_IDS = [f"SP-{i+1:03d}" for i in range(len(DEFAULT_STORIES))]

# Training data — imported directly from the classifier modules
# (they use integer labels: PM_MEETING=0, SPRINT_PLANNING=1, STANDUP=2)
from backend.nlp.unit2_models.lstm_classifier import TRAINING_DATA as LSTM_TRAINING_DATA
from backend.nlp.unit2_models.gru_classifier  import TRAINING_DATA as GRU_TRAINING_DATA



class NLPOrchestrator:
    """
    Runs the full NLP pipeline on a meeting transcript.

    Models are loaded lazily and cached as instance attributes.
    Re-use the same NLPOrchestrator instance across Telegram
    message handling to avoid repeated model loading.
    """

    def __init__(self, stories: List[str] = None, story_ids: List[str] = None):
        # Try to load real Jira stories; fall back to defaults
        jira_stories, jira_ids = self._load_jira_stories()
        self.stories   = stories   or jira_stories or DEFAULT_STORIES
        self.story_ids = story_ids or jira_ids     or DEFAULT_STORY_IDS

        # Lazy-loaded model caches
        self._sbert     = None
        self._bart      = None
        self._qa        = None
        self._lstm      = None
        self._lstm_vocab= None
        self._gru       = None
        self._gru_vocab = None

    @staticmethod
    def _load_jira_stories() -> tuple:
        """
        Fetch open stories/tasks from the real Jira project.
        Returns (story_titles, story_keys) or ([], []) on failure.
        This makes SBERT map NLP actions to REAL Jira ticket keys.
        """
        try:
            from backend.tools.jira_client import JiraManager
            jira = JiraManager()
            result = jira.search_tickets(
                status=None,          # all statuses
                max_results=50,
            )
            if result.get('success') and result.get('issues'):
                titles = [i['summary'] for i in result['issues']]
                keys   = [i['key']     for i in result['issues']]
                logger.info(f"Loaded {len(keys)} real Jira stories: {keys[:5]}...")
                return titles, keys
        except Exception as e:
            logger.warning(f"Could not load Jira stories ({e}); using defaults")
        return [], []

    # ── Lazy model loaders ────────────────────────────────

    def _get_sbert(self):
        if self._sbert is None:
            from backend.nlp.unit3_transformers.sentence_bert import _load_sbert
            logger.info("Loading Sentence-BERT …")
            self._sbert = _load_sbert()
        return self._sbert

    def _get_bart(self):
        if self._bart is None:
            from backend.nlp.unit4_applications.summarizer import _load_bart
            logger.info("Loading DistilBART …")
            self._bart = _load_bart()
        return self._bart

    def _get_lstm(self):
        if self._lstm is None:
            from backend.nlp.unit2_models.lstm_classifier import train as lstm_train
            logger.info("Training LSTM classifier …")
            self._lstm, self._lstm_vocab = lstm_train(
                LSTM_TRAINING_DATA, epochs=40, lr=1e-3
            )
        return self._lstm, self._lstm_vocab

    def _get_gru(self):
        if self._gru is None:
            from backend.nlp.unit2_models.gru_classifier import train as gru_train
            logger.info("Training GRU classifier …")
            self._gru, self._gru_vocab = gru_train(
                GRU_TRAINING_DATA, epochs=50, lr=1e-3
            )
        return self._gru, self._gru_vocab

    # ── Internal helpers ──────────────────────────────────

    def _detect_blockers(self, sentences: List[str]) -> List[str]:
        """Heuristic: sentences containing blocker-related keywords."""
        keywords = ["block", "wait", "stuck", "depend", "credentials", "access",
                    "missing", "need help", "can't", "cannot", "delayed"]
        blockers = []
        for s in sentences:
            if any(k in s.lower() for k in keywords):
                blockers.append(s.strip())
        return blockers

    def _extract_epics_from_pm_meeting(self, sentences: List[str], entities: Dict) -> List[Dict]:
        """
        Lightweight epic extractor for PM meetings.
        Uses N-gram LM + TF-IDF to find high-value sentences
        and wraps them as Epic candidates.
        """
        from backend.nlp.unit1_representations.tfidf import build_tfidf, retrieve_top_k

        # Find sentences mentioning business value / priority / epic-related terms
        epic_keywords = ["priority", "business value", "epic", "feature", "integrate",
                         "build", "implement", "dashboard", "gateway", "authentication"]
        epic_sentences = [
            s for s in sentences
            if any(k in s.lower() for k in epic_keywords) and len(s.split()) > 5
        ][:6]  # top 6 candidates

        epics = []
        for i, s in enumerate(epic_sentences):
            epics.append({
                "title":            s[:80].strip().capitalize(),
                "description":      s.strip(),
                "business_value":   8,   # default; override with NER number extraction
                "time_criticality": 7,
                "risk_reduction":   5,
                "effort":           5,
                "wsjf": {
                    "wsjf_score":    (8 + 7 + 5) / 5,
                    "cost_of_delay": 8 + 7 + 5,
                    "job_size":      5,
                },
                "rank": i + 1,
            })
        return epics

    # ════════════════════════════════════════════════════════
    # MAIN ENTRY POINT
    # ════════════════════════════════════════════════════════

    def run(self, transcript: str) -> Dict:
        """
        Run the complete NLP pipeline on a meeting transcript.

        Args:
            transcript: Raw text from Telegram messages / Whisper ASR

        Returns:
            {
                "meeting_type":     str,
                "meeting_conf":     float,
                "assignees":        List[str],
                "estimates":        List[str],
                "dates":            List[str],
                "sentences":        List[str],
                "actions":          List[Dict],   (GRU + NER + SBERT)
                "blockers":         List[str],
                "summary_extract":  str,           (TF-IDF extractive)
                "summary_abstract": str,           (DistilBART)
                "approval_payload": Dict,          (→ ApprovalService)
                "elapsed_s":        float,
            }
        """
        t0 = time.time()
        logger.info("NLP Pipeline starting …")

        # ── UNIT 1: Preprocessing ─────────────────────────
        from backend.nlp.unit1_preprocessing.tokenizer   import sentence_tok
        from backend.nlp.unit1_preprocessing.normalizer  import normalize
        from backend.nlp.unit1_preprocessing.ner         import (
            extract_assignees, extract_estimates, extract_dates, extract_entities
        )
        from backend.nlp.unit4_applications.summarizer   import extractive_summarize

        sentences  = sentence_tok(transcript)
        normalized = normalize(transcript, remove_stops=False)
        assignees  = list(dict.fromkeys(extract_assignees(transcript)))
        estimates  = extract_estimates(transcript)
        dates      = extract_dates(transcript)
        entities   = {
            "assignees": assignees,
            "estimates": estimates,
            "dates":     dates,
        }
        logger.info(f"  Unit 1 done: {len(sentences)} sentences, {len(assignees)} people")

        # ── UNIT 1: Extractive summary ────────────────────
        summary_extract = extractive_summarize(transcript, n_sentences=3)

        # ── UNIT 2: Meeting type (LSTM) ───────────────────
        from backend.nlp.unit2_models.lstm_classifier import predict as lstm_predict
        lstm_model, lstm_vocab = self._get_lstm()
        # Use first 3 sentences as meeting context
        context = " ".join(sentences[:3])
        lstm_res = lstm_predict(lstm_model, lstm_vocab, context)
        meeting_type = lstm_res["prediction"]
        meeting_conf = lstm_res["confidence"]
        logger.info(f"  Unit 2 LSTM: {meeting_type} ({meeting_conf:.3f})")

        # ── UNIT 2: Action extraction (GRU per sentence) ──
        from backend.nlp.unit2_models.gru_classifier import predict as gru_predict
        from backend.nlp.unit1_preprocessing.ner     import extract_assignees as sent_people
        gru_model, gru_vocab = self._get_gru()

        raw_actions = []
        for sent in sentences:
            if len(sent.split()) < 4:
                continue
            gru_res = gru_predict(gru_model, gru_vocab, sent)
            action  = gru_res["prediction"]
            conf    = gru_res["confidence"]
            people  = sent_people(sent)
            raw_actions.append({
                "sentence": sent.strip(),
                "action":   action,
                "actor":    people[0] if people else (assignees[0] if assignees else "?"),
                "conf":     conf,
            })

        logger.info(f"  Unit 2 GRU: {len(raw_actions)} sentences classified")

        # ── UNIT 3: Story mapping (SBERT) ─────────────────
        from backend.nlp.unit3_transformers.sentence_bert import find_matching_story
        sbert = self._get_sbert()

        actions_with_stories = []
        for a in raw_actions:
            if a["action"] == "no_action":
                continue
            matches = find_matching_story(
                a["sentence"], self.stories, self.story_ids,
                sbert, top_k=1, threshold=0.0
            )
            if matches:
                a["story_id"]    = matches[0]["story_id"]
                a["story_title"] = matches[0]["title"]
                a["story_score"] = matches[0]["score"]
            else:
                a["story_id"]    = None
                a["story_title"] = None
                a["story_score"] = 0.0
            actions_with_stories.append(a)

        logger.info(f"  Unit 3 SBERT: {len(actions_with_stories)} actions mapped")

        # ── UNIT 4: Abstractive summary (BART) ────────────
        from backend.nlp.unit4_applications.summarizer import abstractive_summarize
        try:
            bart           = self._get_bart()
            summary_abstract = abstractive_summarize(
                transcript.strip(), bart, max_length=80, min_length=20
            )
        except Exception as e:
            logger.warning(f"BART unavailable: {e} — using extractive fallback")
            summary_abstract = summary_extract

        logger.info("  Unit 4 BART: summary generated")

        # ── Detect blockers ───────────────────────────────
        blockers = self._detect_blockers(sentences)

        # ── Build approval payload (→ ApprovalService) ────
        from backend.nlp.jira_action_mapper import (
            map_standup_approval_payload,
            map_pm_meeting_epics,
            map_sprint_planning,
        )

        if meeting_type == "PM_MEETING":
            epics           = self._extract_epics_from_pm_meeting(sentences, entities)
            approval_payload = map_pm_meeting_epics(summary_abstract, entities, epics)
            approval_type    = "epic_creation"

        elif meeting_type == "SPRINT_PLANNING":
            approval_payload = map_sprint_planning(
                summary_abstract, entities, actions_with_stories
            )
            approval_type = "sprint_planning"

        else:  # STANDUP (default)
            approval_payload = map_standup_approval_payload(
                actions_with_stories, summary_abstract, entities, blockers
            )
            approval_type = "standup_update"

        elapsed = round(time.time() - t0, 2)
        logger.info(f"NLP Pipeline complete in {elapsed}s")

        return {
            "meeting_type":      meeting_type,
            "meeting_conf":      meeting_conf,
            "approval_type":     approval_type,
            "assignees":         assignees,
            "estimates":         estimates,
            "dates":             dates,
            "sentences":         sentences,
            "actions":           actions_with_stories,
            "blockers":          blockers,
            "summary_extract":   summary_extract,
            "summary_abstract":  summary_abstract,
            "approval_payload":  approval_payload,
            "elapsed_s":         elapsed,
        }


# ── Singleton (reuse across requests) ────────────────────
_orchestrator: Optional[NLPOrchestrator] = None

def get_orchestrator() -> NLPOrchestrator:
    """
    Get (or create) the shared NLPOrchestrator instance.
    On first call, queries live Jira for real story keys.
    """
    global _orchestrator
    if _orchestrator is None:
        logger.info("Initialising NLPOrchestrator (loading Jira stories…)")
        _orchestrator = NLPOrchestrator()
        logger.info(
            f"Story corpus: {len(_orchestrator.story_ids)} tickets "
            f"({_orchestrator.story_ids[:3]}…)"
        )
    return _orchestrator


def refresh_orchestrator() -> NLPOrchestrator:
    """Force a fresh NLPOrchestrator (re-queries Jira stories)."""
    global _orchestrator
    _orchestrator = None
    return get_orchestrator()

