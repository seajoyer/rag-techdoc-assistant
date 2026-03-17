"""
throttle.py
-----------
Per-user rate-limiting middleware for aiogram 3.

Each user is allowed at most ``max_requests`` messages within a rolling
``window_seconds`` window.  When the limit is exceeded the middleware
replies with a friendly wait message and swallows the update so no
handler runs.

Usage
~~~~~
    from bot.middleware.throttle import ThrottleMiddleware
    dp.message.middleware(ThrottleMiddleware(max_requests=5, window_seconds=60))
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any, Awaitable, Callable

from aiogram import BaseMiddleware
from aiogram.types import Message, TelegramObject


class ThrottleMiddleware(BaseMiddleware):
    """
    Sliding-window rate limiter applied to Message updates.

    Parameters
    ----------
    max_requests:
        Maximum number of messages allowed per user per window.
    window_seconds:
        Length of the sliding window in seconds.
    """

    def __init__(self, max_requests: int = 5, window_seconds: int = 60) -> None:
        self._max  = max_requests
        self._win  = window_seconds
        # user_id → list of UNIX timestamps for requests in the current window
        self._history: dict[int, list[float]] = defaultdict(list)
        super().__init__()

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        # Only throttle actual messages (not callbacks, inline queries, …)
        if not isinstance(event, Message) or event.from_user is None:
            return await handler(event, data)

        user_id = event.from_user.id
        now     = time.monotonic()
        cutoff  = now - self._win

        # Evict timestamps outside the window
        history = self._history[user_id]
        self._history[user_id] = [t for t in history if t > cutoff]

        if len(self._history[user_id]) >= self._max:
            # Calculate how many seconds until the oldest entry expires
            oldest  = min(self._history[user_id])
            wait    = int(self._win - (now - oldest)) + 1
            await event.answer(
                f"⏳ You're sending messages too quickly.\n"
                f"Please wait <b>{wait}s</b> before asking again.",
                parse_mode="HTML",
            )
            return  # swallow the update — no handler runs

        self._history[user_id].append(now)
        return await handler(event, data)
