"""
chat.py
-------
Main question-answering handler.

Flow
~~~~
1. Guard: check allow-list (if configured) and ignore non-text input.
2. Send an animated "Searching…" status message.
3. Keep the Telegram *typing* indicator alive in the background (it
   expires after 5 s; we re-send it every 4 s).
4. Invoke the RAG chain in a thread-pool executor (it's blocking I/O +
   model inference — never run on the event loop directly).
5. Edit the status message in place with the formatted answer + inline
   source keyboard.
6. Handle errors gracefully: edit the status message with a user-
   friendly error rather than leaving a "searching…" ghost message.
"""

from __future__ import annotations

import asyncio
import functools
import logging

from aiogram import Bot, F, Router
from aiogram.enums import ChatAction
from aiogram.types import Message

from bot import services
from bot.config import settings
from bot.keyboards import format_error, format_rag_response, sources_keyboard

log = logging.getLogger(__name__)

router = Router(name="chat")

# Telegram's typing action expires after 5 s; refresh every 4 s.
_TYPING_INTERVAL = 4


# ---------------------------------------------------------------------------
# Guard filter
# ---------------------------------------------------------------------------

def _is_allowed(user_id: int) -> bool:
    """True if the user is permitted to query the bot."""
    return settings.is_public or user_id in settings.allowed_user_ids


# ---------------------------------------------------------------------------
# Typing-indicator background task
# ---------------------------------------------------------------------------

async def _keep_typing(chat_id: int, bot: Bot, stop: asyncio.Event) -> None:
    """Re-send the typing chat action every 4 s until *stop* is set."""
    while not stop.is_set():
        try:
            await bot.send_chat_action(chat_id, ChatAction.TYPING)
        except Exception:
            pass  # best-effort; don't crash the whole flow
        await asyncio.sleep(_TYPING_INTERVAL)


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

@router.message(F.text)
async def handle_question(message: Message, bot: Bot) -> None:
    # ── Access control ────────────────────────────────────────────────────
    if not _is_allowed(message.from_user.id):
        await message.reply("🔒 This bot is private. You are not authorised.")
        return

    question = message.text.strip()
    if not question:
        return

    log.info(
        "[Handler] Question from user=%d | text=%r",
        message.from_user.id, question[:120],
    )

    # ── Send placeholder while we work ────────────────────────────────────
    status_msg = await message.reply(
        "🔍 <b>Searching PyTorch docs…</b>",
        parse_mode="HTML",
    )

    # ── Keep typing indicator alive in background ─────────────────────────
    stop_typing = asyncio.Event()
    typing_task = asyncio.create_task(
        _keep_typing(message.chat.id, bot, stop_typing)
    )

    try:
        # ── Ensure pipeline is initialised ────────────────────────────────
        chain = await services.get_chain()

        # ── Run chain in executor (blocking) ──────────────────────────────
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            functools.partial(chain.invoke, question),
        )

        log.info(
            "[Handler] Answer ready | answer_chars=%d | sources=%d",
            len(result.answer), len(result.sources),
        )

        # ── Format and send ───────────────────────────────────────────────
        formatted = format_rag_response(result)
        keyboard  = sources_keyboard(result.sources)

        await status_msg.edit_text(
            formatted,
            parse_mode="HTML",
            reply_markup=keyboard,
            disable_web_page_preview=True,
        )

    except Exception as exc:
        log.exception("[Handler] Chain error for user=%d", message.from_user.id)
        try:
            await status_msg.edit_text(format_error(exc), parse_mode="HTML")
        except Exception:
            pass  # message might have been deleted

    finally:
        stop_typing.set()
        typing_task.cancel()


# ---------------------------------------------------------------------------
# Catch-all for non-text messages
# ---------------------------------------------------------------------------

@router.message()
async def handle_unsupported(message: Message) -> None:
    await message.reply(
        "💬 Please send a text question. "
        "I can only process plain text messages."
    )
