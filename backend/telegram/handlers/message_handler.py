"""
Message Handler

Handles text messages for conversation flows:
- NLP meeting transcript collection (/meeting → /done)
- Email linking
- Rejection reasons
- Edit flows
"""
from telegram import Update
from telegram.ext import ContextTypes

from backend.telegram.handlers.start_handler    import handle_email_linking
from backend.telegram.handlers.callback_handler import handle_rejection_reason


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handle text messages based on conversation state.

    Priority order:
      1. NLP meeting buffer (if /meeting session is active)
      2. Email linking flow
      3. Rejection reason flow
      4. Default help message
    """
    # ── 1. NLP meeting transcript collection ──────────────
    # If /meeting is active, buffer every message for the NLP pipeline
    try:
        from backend.telegram.handlers.nlp_meeting_handler import collect_message
        if await collect_message(update, context):
            return   # message buffered — no further handling needed
    except Exception:
        pass  # never block normal flow

    # ── 2. Existing conversation flows ────────────────────
    if context.user_data.get('awaiting_email'):
        await handle_email_linking(update, context)

    elif context.user_data.get('awaiting_rejection_reason'):
        await handle_rejection_reason(update, context)

    else:
        await update.message.reply_text(
            "ℹ️ I didn't understand that.\n\n"
            "Use /help to see available commands.\n"
            "Use /meeting to start a standup session."
        )
