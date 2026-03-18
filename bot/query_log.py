"""
query_log.py
------------
Forwards user queries to a private logging group after the bot responds.

Each notification looks like:

    👤 First Last (@username)  |  id: 123456789
    https://t.me/username  (or tg://user?id=... for users without a username)

    Query:

    How does torch.autograd.grad differ from .backward()?

The message is sent best-effort: any Telegram API error is logged and
silently swallowed so a logging failure never affects the user-facing reply.
"""

from __future__ import annotations

import html
import logging

from aiogram import Bot
from aiogram.types import User

log = logging.getLogger(__name__)


async def log_query(bot: Bot, user: User, query: str, log_group_id: int) -> None:
    """
    Send a query-log notification to *log_group_id*.

    Parameters
    ----------
    bot:
        The running ``aiogram.Bot`` instance.
    user:
        The Telegram ``User`` who sent the query.
    query:
        The raw text of their message.
    log_group_id:
        Chat ID of the private logging group.
    """
    # Build a human-readable user line and a deep-link.
    name_parts = [user.first_name]
    if user.last_name:
        name_parts.append(user.last_name)
    full_name = " ".join(name_parts)

    if user.username:
        user_line = f"{full_name} (@{user.username})  |  id: {user.id}"
        user_link = f"https://t.me/{user.username}"
    else:
        user_line = f"{full_name}  |  id: {user.id}"
        user_link = f"tg://user?id={user.id}"

    text = (
        f"👤 {html.escape(user_line)}\n"
        f'<a href="{user_link}">{html.escape(user_link)}</a>\n\n'
        f"<b>Query:</b>\n\n"
        f"{html.escape(query)}"
    )

    try:
        await bot.send_message(
            chat_id=log_group_id,
            text=text,
            parse_mode="HTML",
            disable_web_page_preview=True,
        )
    except Exception as exc:
        log.warning(
            "[QueryLog] Failed to forward query to group %d: %s",
            log_group_id, exc,
        )
