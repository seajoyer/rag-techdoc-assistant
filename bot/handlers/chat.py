"""
chat.py
-------
Main question-answering handler.

Flow — standard (STREAMING_ENABLED=false)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
7. Forward the query to the logging group (if LOG_GROUP_ID is set),
   best-effort, after the reply has been delivered.

Flow — streaming (STREAMING_ENABLED=true, private chats only)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Steps 1–3 are identical.  Step 4 is replaced by:

4a. Open an async generator over ``services.astream_answer()``.
4b. As each token arrives, append it to an accumulation buffer and
    call ``bot.send_message_draft()`` (Telegram Bot API 9.5) at most
    once every ``STREAMING_DRAFT_INTERVAL`` seconds.  The draft shows
    a live, animated preview of the growing answer to the user.
4c. The last item from the generator is a ``RAGResult``; capture it.
4d. Edit the status message with the fully formatted, citation-linked
    final answer (same as step 5 in the standard flow).

Streaming falls back to the standard path automatically when:
  - The chat is not a private chat (sendMessageDraft is private-only).
  - The ``aiogram.methods.SendMessageDraft`` symbol is not importable
    (older aiogram build without Bot API 9.5 support).
  - Any error occurs during the draft-update phase.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import time

from aiogram import Bot, F, Router
from aiogram.enums import ChatAction
from aiogram.types import Message

from bot import services
from bot.config import settings
from bot.keyboards import format_error, format_rag_response, sources_keyboard
from bot.query_log import log_query

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
# Standard (non-streaming) answer path
# ---------------------------------------------------------------------------

async def _standard_answer(
    message: Message,
    status_msg: Message,
    bot: Bot,
    question: str,
) -> None:
    """Invoke the RAG chain synchronously in a thread pool and edit the result."""
    chain = await services.get_chain()

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None,
        functools.partial(chain.invoke, question),
    )

    log.info(
        "[Handler] Standard answer ready | answer_chars=%d | sources=%d",
        len(result.answer), len(result.sources),
    )

    formatted = format_rag_response(result)
    keyboard  = sources_keyboard(result.sources)

    await status_msg.edit_text(
        formatted,
        parse_mode="HTML",
        reply_markup=keyboard,
        disable_web_page_preview=True,
    )


# ---------------------------------------------------------------------------
# Streaming answer path  (Bot API 9.5 sendMessageDraft)
# ---------------------------------------------------------------------------

async def _stream_answer(
    message: Message,
    status_msg: Message,
    bot: Bot,
    question: str,
) -> None:
    """
    Stream tokens via ``sendMessageDraft``, then edit the final message.

    The draft preview is updated at most once per
    ``settings.streaming_draft_interval`` seconds to stay well within
    Telegram rate limits.  The final ``edit_text`` call delivers the
    fully-formatted, citation-linked answer exactly as the standard path.

    Raises
    ------
    ImportError
        If ``aiogram.methods.SendMessageDraft`` is not available (caller
        should fall back to ``_standard_answer``).
    Any exception from ``services.astream_answer`` is re-raised so the
    outer handler can display a user-friendly error.
    """
    from aiogram.methods import SendMessageDraft  # raises ImportError if unavailable
    from src.rag.chain import RAGResult

    # Use the status_msg's message_id as a stable, unique draft identifier
    # for this streaming session within the chat.
    draft_id: int = status_msg.message_id

    accumulated: str = ""
    last_draft_ts: float = 0.0
    final_result: RAGResult | None = None
    draft_interval: float = settings.streaming_draft_interval

    log.info(
        "[Handler] Streaming answer | draft_id=%d | draft_interval=%.2fs",
        draft_id, draft_interval,
    )

    async for item in services.astream_answer(question):
        if isinstance(item, RAGResult):
            # Final item — capture and stop iterating.
            final_result = item
            break

        # Token chunk — accumulate and maybe push a draft update.
        accumulated += item

        now = time.monotonic()
        if accumulated and (now - last_draft_ts) >= draft_interval:
            try:
                await bot(
                    SendMessageDraft(
                        chat_id=message.chat.id,
                        draft_id=draft_id,
                        text=accumulated,
                        # No parse_mode here: partial Markdown can produce
                        # malformed HTML/MarkdownV2.  The draft is plain-text
                        # preview only; the final edit_text uses HTML.
                    )
                )
                last_draft_ts = now
            except Exception as draft_exc:
                # A failed draft update is not fatal — log and continue.
                log.debug("[Handler] sendMessageDraft failed: %s", draft_exc)

    if final_result is None:
        raise RuntimeError(
            "Streaming ended without producing a RAGResult — "
            "check stream_rag() for early exit."
        )

    log.info(
        "[Handler] Streaming complete | answer_chars=%d | sources=%d",
        len(final_result.answer), len(final_result.sources),
    )

    # Edit the placeholder with the final formatted, citation-linked answer.
    formatted = format_rag_response(final_result)
    keyboard  = sources_keyboard(final_result.sources)

    await status_msg.edit_text(
        formatted,
        parse_mode="HTML",
        reply_markup=keyboard,
        disable_web_page_preview=True,
    )


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
        "[Handler] Question from user=%d | streaming=%s | text=%r",
        message.from_user.id, settings.streaming_enabled, question[:120],
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

    # ── Choose answer path ────────────────────────────────────────────────
    # Streaming is only available for private chats (sendMessageDraft is
    # defined for private chats in Bot API 9.5).
    use_streaming = (
        settings.streaming_enabled
        and message.chat.type == "private"
    )

    try:
        if use_streaming:
            try:
                await _stream_answer(message, status_msg, bot, question)
            except ImportError:
                # aiogram build predates Bot API 9.5 — degrade gracefully.
                log.warning(
                    "[Handler] SendMessageDraft not available in this aiogram "
                    "build — falling back to standard path."
                )
                await _standard_answer(message, status_msg, bot, question)
        else:
            await _standard_answer(message, status_msg, bot, question)

    except Exception as exc:
        log.exception("[Handler] Chain error for user=%d", message.from_user.id)
        try:
            await status_msg.edit_text(format_error(exc), parse_mode="HTML")
        except Exception:
            pass  # message might have been deleted

    finally:
        stop_typing.set()
        typing_task.cancel()

    # ── Forward query to logging group (best-effort, after reply) ─────────
    if settings.log_group_id is not None:
        await log_query(
            bot=bot,
            user=message.from_user,
            query=question,
            log_group_id=settings.log_group_id,
        )


# ---------------------------------------------------------------------------
# Catch-all for non-text messages
# ---------------------------------------------------------------------------

@router.message()
async def handle_unsupported(message: Message) -> None:
    await message.reply(
        "💬 Please send a text question. "
        "I can only process plain text messages."
    )
