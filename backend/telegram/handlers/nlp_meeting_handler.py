"""
=============================================================
NLP Meeting Handler — Telegram Commands
=============================================================
Adds two new commands to the existing ScrumPilot bot:

  /meeting  — Start collecting meeting transcript
  /done     — Stop collection, run NLP pipeline, send HITL

Full flow:
  /meeting
     → bot: "Send messages. /done when finished."
  [team sends standup updates]
  /done
     → NLP pipeline runs (Units 1–4)
     → ApprovalService creates approval request
     → PM/SM gets Telegram HITL buttons:
         [✅ Approve] [❌ Reject] [✏️ Edit] [👁️ View]
     → On Approve → JiraAgent.execute_actions()
     → Confirmation sent to group

Supports all 3 meeting types (auto-detected by LSTM):
  • STANDUP       → standup_update approval
  • PM_MEETING    → epic_creation approval
  • SPRINT_PLANNING → sprint_planning approval

Also handles voice messages if Whisper ASR is configured.
=============================================================
"""

import logging
from typing import Dict, Optional
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes

logger = logging.getLogger(__name__)

# Telegram hard limit is 4096 chars; leave buffer for formatting
_TG_MAX = 3800

TELEGRAM_CHAR_LIMIT = _TG_MAX

# Key used in bot context.chat_data to buffer transcript
_BUFFER_KEY  = "nlp_transcript_buffer"
_ACTIVE_KEY  = "nlp_session_active"
_RESULT_KEY  = "nlp_last_result"


# ════════════════════════════════════════════════════════════
# /meeting — Start transcript collection
# ════════════════════════════════════════════════════════════

async def handle_meeting(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /meeting — Open a transcript collection window.

    Team members send their standup/PM/planning updates as
    normal chat messages. The bot buffers them by chat ID.
    `/done` triggers the NLP pipeline.
    """
    chat_id = update.effective_chat.id

    # Initialise buffer for this chat
    context.chat_data[_BUFFER_KEY]  = []
    context.chat_data[_ACTIVE_KEY]  = True

    await update.message.reply_text(
        "🎙️ *Meeting recording started!*\n\n"
        "Send your updates as messages in this chat.\n"
        "Each person: one or more messages with your:\n"
        "  • ✅ What you *completed*\n"
        "  • 🔄 What you're *working on today*\n"
        "  • ⚠️ Any *blockers*\n\n"
        "When everyone is done, type `/done`\n\n"
        "_🤖 Powered by NLP: GRU · SBERT · DistilBART_",
        parse_mode="Markdown",
    )
    logger.info(f"[{chat_id}] Meeting collection started")


# ════════════════════════════════════════════════════════════
# Message collector — called by existing message_handler
# ════════════════════════════════════════════════════════════

async def collect_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """
    If a meeting session is active, buffer the incoming message.

    Returns True if message was buffered (so message_handler
    knows not to process it further).
    """
    if not context.chat_data.get(_ACTIVE_KEY, False):
        return False

    user = update.effective_user
    text = update.message.text or ""
    if text.startswith("/"):
        return False  # let command handlers deal with it

    # Prefix with speaker name for NER
    speaker = user.first_name or user.username or "Unknown"
    context.chat_data[_BUFFER_KEY].append(f"{speaker}: {text}")

    # Acknowledge with a quick ✓ reaction (Telegram supports this in groups)
    try:
        await update.message.set_reaction("👍")
    except Exception:
        pass  # not all clients / groups support reactions

    return True


# ════════════════════════════════════════════════════════════
# /done — Run NLP pipeline + create approval request
# ════════════════════════════════════════════════════════════

async def handle_done(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /done — Stop collection and run the full NLP pipeline.

    Steps:
      1. Validate buffer not empty
      2. Run NLPOrchestrator.run(transcript)
      3. Post interim "analysing…" message
      4. Call ApprovalService.create_*_approval()
      5. HITL message sent automatically by ApprovalService
      6. Post summary to group
    """
    chat_id = update.effective_chat.id

    if not context.chat_data.get(_ACTIVE_KEY, False):
        await update.message.reply_text(
            "❌ No active meeting session.\n"
            "Use /meeting to start one."
        )
        return

    buffer: list = context.chat_data.get(_BUFFER_KEY, [])
    if not buffer:
        await update.message.reply_text(
            "⚠️ No messages collected yet. "
            "Ask the team to send their updates first."
        )
        return

    # Close collection window
    context.chat_data[_ACTIVE_KEY] = False
    transcript = "\n\n".join(buffer)

    # ── Interim status ────────────────────────────────────
    status_msg = await update.message.reply_text(
        f"⚙️ *Analysing {len(buffer)} messages…*\n\n"
        f"Running NLP pipeline:\n"
        f"  🔤 Preprocessing (spaCy NER)\n"
        f"  🧠 Classifying meeting type (LSTM)\n"
        f"  ⚡ Extracting actions (GRU)\n"
        f"  🔍 Mapping to stories (Sentence-BERT)\n"
        f"  📝 Summarising (DistilBART)\n\n"
        f"_This takes ~30s on first run (model loading)…_",
        parse_mode="Markdown",
    )

    # ── Run NLP pipeline ──────────────────────────────────
    try:
        from backend.nlp.pipeline_orchestrator import get_orchestrator
        orch   = get_orchestrator()
        result = orch.run(transcript)
        context.chat_data[_RESULT_KEY] = result
    except Exception as e:
        logger.exception("NLP pipeline error")
        short_err = str(e)[:300]  # Never send full traceback to Telegram
        try:
            await status_msg.edit_text(
                f"❌ NLP pipeline failed:\n`{short_err}`\n\nCheck bot logs.",
                parse_mode="Markdown",
            )
        except Exception:
            await update.message.reply_text(
                f"❌ NLP pipeline failed: {short_err[:200]}"
            )
        return

    # ── Edit status → NLP report ──────────────────────────
    await status_msg.edit_text(
        _format_pipeline_report(result),
        parse_mode="Markdown",
    )

    # ── Route to ApprovalService ──────────────────────────
    approval_sent = await _create_approval(update, context, result)

    if approval_sent:
        await update.message.reply_text(
            "📬 *Approval request sent to Scrum Master / PM.*\n\n"
            "Nothing will be updated in Jira until they approve.\n"
            "_Use /approvals to see pending requests._",
            parse_mode="Markdown",
        )
    else:
        # Fallback inline approval in the group chat itself
        await _send_inline_approval(update, context, result)


# ════════════════════════════════════════════════════════════
# Approval routing
# ════════════════════════════════════════════════════════════

async def _create_approval(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    result: Dict,
) -> bool:
    """
    Try to create a DB-persisted approval via ApprovalService.
    Falls back to inline Telegram buttons if DB is unavailable.

    Returns True if DB approval was created.
    """
    try:
        from backend.telegram.services.approval_service import ApprovalService
        from backend.db.connection import get_session
        from backend.db.models import User

        # Get the requesting user
        user_id = update.effective_user.id
        with get_session() as session:
            db_user = session.query(User).filter(
                User.telegram_user_id == user_id
            ).first()
            requesting_user_id = db_user.id if db_user else 1

        # Get PM / Scrum Master user for assignment
        pm_user_id = ApprovalService.get_pm_user_id() or requesting_user_id

        approval_type   = result.get("approval_type", "standup_update")
        payload         = result["approval_payload"]

        if approval_type == "epic_creation":
            ApprovalService.create_epic_approval(
                payload, requesting_user_id, pm_user_id, priority="high"
            )
        elif approval_type == "sprint_planning":
            ApprovalService.create_sprint_approval(
                payload, requesting_user_id, pm_user_id, priority="high"
            )
        else:  # standup_update
            # ApprovalService doesn't have a create_standup_approval method yet;
            # we use the generic pattern (stored as 'standup_update' request_type)
            from backend.db.models import ApprovalRequest
            from datetime import datetime, timezone, timedelta

            with get_session() as session:
                approval = ApprovalRequest(
                    request_type  = "standup_update",
                    entity_type   = "task",
                    entity_id     = 0,
                    requested_by  = requesting_user_id,
                    assigned_to   = pm_user_id,
                    status        = "pending",
                    priority      = "normal",
                    request_data  = payload,
                    original_data = payload,
                    created_at    = datetime.now(timezone.utc),
                    expires_at    = datetime.now(timezone.utc) + timedelta(hours=24),
                )
                session.add(approval)
                session.commit()
                session.refresh(approval)
                approval_id = approval.approval_id

            # Send Telegram notification to PM
            ApprovalService._send_telegram_notification(
                telegram_user_id = pm_user_id,
                telegram_chat_id = pm_user_id,
                approval_id      = approval_id,
            )

        return True

    except Exception as e:
        logger.warning(f"DB approval unavailable ({e}) — using inline fallback")
        return False


async def _send_inline_approval(
    update:  Update,
    context: ContextTypes.DEFAULT_TYPE,
    result:  Dict,
):
    """
    Fallback: send HITL approval directly in the group chat
    when the DB / ApprovalService is unavailable.
    """
    meeting_type = result["meeting_type"]
    actions      = result["actions"]
    blockers     = result["blockers"]
    summary      = result["summary_abstract"]

    # Build preview text
    lines = [f"🤖 *NLP Meeting Report — {meeting_type}*\n"]

    if actions:
        lines.append("*Proposed Jira Actions:*")
        for a in actions[:5]:
            emoji = {"complete_task": "✅", "create_task": "🆕",
                     "update_status": "🔄", "assign_task": "👤"}.get(a["action"], "•")
            story = a.get("story_id", "?")
            lines.append(f"  {emoji} [{a['action']}] {a['actor']} → {story}")
        if len(actions) > 5:
            lines.append(f"  … and {len(actions)-5} more")

    if blockers:
        lines.append(f"\n⚠️ *Blockers*: {'; '.join(blockers[:2])}")

    lines.append(f"\n📝 *Summary*: {summary[:120]}…")
    lines.append("\n_Approve to push to Jira:_")

    keyboard = [
        [
            InlineKeyboardButton("✅ Approve All", callback_data="nlp_approve"),
            InlineKeyboardButton("❌ Reject",       callback_data="nlp_reject"),
        ],
        [
            InlineKeyboardButton("✏️ Edit Actions", callback_data="nlp_edit"),
            InlineKeyboardButton("👁️ Full Report",  callback_data="nlp_view"),
        ],
    ]

    await update.message.reply_text(
        "\n".join(lines),
        parse_mode   = "Markdown",
        reply_markup = InlineKeyboardMarkup(keyboard),
    )


# ════════════════════════════════════════════════════════════
# Inline callback handlers (approve/reject/edit/view)
# ════════════════════════════════════════════════════════════

async def handle_nlp_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handle inline button callbacks from _send_inline_approval.
    Registered in callback_handler.py under "nlp_" prefix.
    """
    query = update.callback_query
    await query.answer()

    data   = query.data
    result = context.chat_data.get(_RESULT_KEY)

    if data == "nlp_approve":
        await _execute_jira_actions(query, context, result)

    elif data == "nlp_reject":
        await query.edit_message_text(
            "❌ *Actions rejected.* No Jira updates were made.\n\n"
            "Run /meeting again to restart.",
            parse_mode="Markdown",
        )

    elif data == "nlp_view":
        if result:
            full = _format_full_report(result)
            await query.message.reply_text(full, parse_mode="Markdown")
        else:
            await query.answer("No result data available.")

    elif data == "nlp_edit":
        await query.edit_message_text(
            "✏️ *Edit mode not yet available in inline fallback.*\n\n"
            "Use /approvals if the approval was saved to the database.",
            parse_mode="Markdown",
        )


async def _execute_jira_actions(query, context, result: Optional[Dict]):
    """
    Execute approved NLP actions via JiraAgent.

    Falls back to a dry-run report if:
      - 'jira' package is not installed
      - Jira credentials are not in .env
      - Jira API is unreachable
    """
    if not result:
        await query.edit_message_text("❌ No pipeline result to execute.")
        return

    from backend.nlp.jira_action_mapper import map_standup_actions
    jira_actions = map_standup_actions(result.get("actions", []))

    if not jira_actions:
        await query.edit_message_text(
            "ℹ️ No actionable items found in the transcript.\n"
            "Nothing was pushed to Jira."
        )
        return

    await query.edit_message_text("⚙️ *Pushing to Jira…*", parse_mode="Markdown")

    try:
        # ── Attempt real Jira push ────────────────────────
        from backend.agents.jira_agent import JiraAgent
        jira_agent = JiraAgent()
        report     = jira_agent.execute_actions(jira_actions)

        lines = [f"✅ *Jira Updated!* ({len(jira_actions)} action(s))\n"]
        for a in jira_actions:
            emoji = {"complete_task": "✅", "create_task": "🆕",
                     "update_status": "🔄", "assign_task": "👤"}.get(a["action"], "•")
            lines.append(f"  {emoji} `{a['action']}` → {a['summary'][:50]}")
        lines.append("\n_Jira board updated. ✓_")

        await query.edit_message_text("\n".join(lines), parse_mode="Markdown")
        logger.info(f"Jira execution complete: {report[:200]}")

    except (ImportError, ModuleNotFoundError):
        # ── jira package not installed → dry run ──────────
        await _send_dry_run_report(query, jira_actions,
            "⚠️ `jira` package not installed.\nInstall with: `pip install jira`")

    except ValueError as e:
        # ── Missing credentials → dry run ─────────────────
        await _send_dry_run_report(query, jira_actions,
            f"⚠️ Jira credentials not configured in `.env`:\n`{e}`")

    except Exception as e:
        logger.exception("Jira execution failed")
        await _send_dry_run_report(query, jira_actions,
            f"⚠️ Jira API error: `{str(e)[:120]}`")


async def _send_dry_run_report(query, jira_actions: list, reason: str):
    """Show what WOULD have been pushed to Jira (dry-run mode)."""
    lines = [
        "📋 *DRY RUN — Jira Actions (not pushed)*\n",
        reason[:200],   # cap the reason to avoid length issues
        "",
        f"*{len(jira_actions)} action(s) extracted by NLP:*",
    ]
    action_emoji = {"complete_task": "✅", "create_task": "🆕",
                    "update_status": "🔄", "assign_task": "👤"}
    for i, a in enumerate(jira_actions[:8], 1):   # cap at 8 items
        em  = action_emoji.get(a["action"], "•")
        act = a["action"].replace("_", " ").title()
        lines.append(f"  {i}. {em} *{act}*")
        lines.append(f"      Ticket: `{a['summary'][:55]}`")
        lines.append(f"      Actor : {a.get('assignee','?')}")
        if a.get("story_id"):
            lines.append(f"      Story : {a['story_id']}")

    if len(jira_actions) > 8:
        lines.append(f"  … and {len(jira_actions)-8} more")

    lines.append(
        "\n_Add Jira credentials to `.env` to enable live push._"
    )
    text = "\n".join(lines)[:_TG_MAX]
    try:
        await query.edit_message_text(text, parse_mode="Markdown")
    except Exception:
        await query.message.reply_text(text, parse_mode="Markdown")


# ════════════════════════════════════════════════════════════
# Formatting helpers
# ════════════════════════════════════════════════════════════

def _format_pipeline_report(result: Dict) -> str:
    """Format the NLP pipeline result as a Telegram Markdown message."""
    mt   = result["meeting_type"]
    conf = result["meeting_conf"]
    acts = result["actions"]
    blk  = result["blockers"]
    summ = result["summary_abstract"]
    secs = result["elapsed_s"]

    emoji = {"STANDUP": "📊", "PM_MEETING": "📋", "SPRINT_PLANNING": "🏃"}.get(mt, "📌")

    lines = [
        f"{emoji} *NLP Pipeline Complete* ({secs}s)\n",
        f"*Meeting Type*: {mt}  _(confidence: {conf:.2f})_",
        f"*Participants*: {', '.join(result['assignees']) or '—'}",
        f"*Estimates*:    {', '.join(result['estimates']) or '—'}",
        f"*Dates*:        {', '.join(result['dates'][:2]) or '—'}",
        "",
        f"*NLP Models Used*:",
        f"  🔤 Unit 1: spaCy NER · NLTK tokenizer",
        f"  🧠 Unit 2: LSTM → {mt} · GRU → {len(acts)} actions",
        f"  🔍 Unit 3: Sentence-BERT story mapping",
        f"  📝 Unit 4: DistilBART summary",
        "",
    ]

    if acts:
        lines.append("*Extracted Actions:*")
        action_emoji = {
            "complete_task": "✅", "create_task": "🆕",
            "update_status": "🔄", "assign_task":  "👤",
        }
        for a in acts[:6]:
            em    = action_emoji.get(a["action"], "•")
            story = a.get("story_id", "?")
            score = a.get("story_score", 0)
            lines.append(
                f"  {em} *{a['actor']}* [{a['action']}]"
                f" → {story} _{a.get('story_title','')[:35]}_ (sim={score:.2f})"
            )
        if len(acts) > 6:
            lines.append(f"  … and {len(acts)-6} more")

    if blk:
        lines.append(f"\n⚠️ *Blockers*:")
        for b in blk[:2]:
            lines.append(f"  • {b[:80]}")

    lines.append(f"\n📝 *Summary*:")
    lines.append(f"  _{summ[:200]}_")

    return "\n".join(lines)


def _format_full_report(result: Dict) -> str:
    """Longer full report for the 👁️ View Details button."""
    lines = [_format_pipeline_report(result), ""]
    lines.append("*All Actions (detailed):*")
    for a in result.get("actions", []):
        lines.append(
            f"  • [{a['action']}] {a['actor']} → "
            f"{a.get('story_id','?')} {a.get('story_title','')[:50]}"
        )
        lines.append(f"    _{a['sentence'][:100]}_")
    return "\n".join(lines)[:_TG_MAX]
