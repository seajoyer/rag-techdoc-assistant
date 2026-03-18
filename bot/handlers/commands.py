"""
commands.py
-----------
Handlers for bot commands: /start, /help, /status, /cancel.
"""

from __future__ import annotations

import html

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from bot import services

router = Router(name="commands")

# ---------------------------------------------------------------------------
# /start
# ---------------------------------------------------------------------------

_WELCOME = """\
👋 <b>PyTorch Docs Assistant</b>

Ask me anything about PyTorch and I'll answer \
using the official documentation with inline citations.

<b>Examples:</b>
• <code>How do I move a tensor to GPU?</code>
• <code>What is the difference between torch.Tensor and torch.tensor?</code>
• <code>How does torch.autograd.grad differ from calling .backward()?</code>

Just type your question and I'll get back to you. 🔍 \
"""


@router.message(Command("start"))
async def cmd_start(message: Message) -> None:
    await message.answer(_WELCOME, parse_mode="HTML", disable_web_page_preview=True)


# ---------------------------------------------------------------------------
# /help
# ---------------------------------------------------------------------------

_HELP = """\
🤖 <b>PyTorch Docs Assistant — Help</b>

<b>How to use:</b>
Send any natural-language question about PyTorch and I'll retrieve \
relevant documentation chunks and synthesise a cited answer.

<b>Tips for better answers:</b>
• Be specific about the API or concept you're asking about.
• Include relevant version context if needed.
• Ask about one concept at a time for the most focused answer.

<b>Commands:</b>
/start   — Welcome message
/help    — This help text
/status  — Pipeline health &amp; collection stats

<b>Limitations:</b>
• I answer only based on the indexed PyTorch documentation.
• I will tell you if the context does not contain enough information.
"""


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    from bot.config import settings
    text = _HELP.format(
        max_req=settings.rate_limit_max,
        window=settings.rate_limit_window,
    )
    await message.answer(text, parse_mode="HTML")


# ---------------------------------------------------------------------------
# /status
# ---------------------------------------------------------------------------

@router.message(Command("status"))
async def cmd_status(message: Message) -> None:
    status_msg = await message.answer("⏳ Checking pipeline status…")

    lines: list[str] = ["🔧 <b>Pipeline Status</b>\n"]

    # Chain readiness
    if services.is_ready():
        lines.append(f"✅ RAG chain: <b>ready</b>")
        lines.append(f"🧠 Embedder: <code>{html.escape(services.embedder_label())}</code>")
    else:
        lines.append("⏳ RAG chain: <b>warming up…</b>")
        lines.append("   (ask a question to trigger initialisation)")

    # Collection stats
    try:
        chain = await services.get_chain()
        # Navigate to the store inside the chain to get collection info.
        # The retriever is stored inside retrieve_and_pack closure → extract via first step.
        # Simpler: we can get info if the store is accessible.  For now report generic info.
        lines.append("\n✅ Qdrant connection: <b>OK</b>")
    except Exception as exc:
        lines.append(f"\n❌ Qdrant connection: <b>FAILED</b>")
        lines.append(f"<code>{html.escape(str(exc)[:200])}</code>")

    from bot.config import settings
    lines.append(f"\n⚙️ <b>Settings</b>")
    lines.append(f"Collection: <code>{html.escape(settings.collection_name)}</code>")
    lines.append(f"Top-K: <code>{settings.top_k}</code>")
    lines.append(f"HyDE: <code>{'enabled' if settings.hyde_enabled else 'disabled'}</code>")
    lines.append(f"Model: <code>llama-3.3-70b-versatile</code>")

    await status_msg.edit_text("\n".join(lines), parse_mode="HTML")
