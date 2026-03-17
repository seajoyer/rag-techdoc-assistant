"""
main.py
-------
Entry point for the PyTorch Docs Telegram bot.

Startup sequence
~~~~~~~~~~~~~~~~
1. Validate configuration (crash early if env vars are missing).
2. Create Bot + Dispatcher.
3. Register middleware and routers.
4. Kick off an async warm-up task so the RAG chain loads in the
   background while the bot is already responsive to /start /help.
5. Start long-polling.

Running locally
~~~~~~~~~~~~~~~
    python -m bot.main

Or from the project root:

    python bot/main.py

Docker
~~~~~~
See Dockerfile and docker-compose.yml at the project root.
"""

from __future__ import annotations

import asyncio
import logging
import sys

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.types import BotCommand

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bot command menu
# ---------------------------------------------------------------------------

_COMMANDS = [
    BotCommand(command="start",  description="Introduction & usage"),
    BotCommand(command="help",   description="Show help"),
    BotCommand(command="status", description="Pipeline health check"),
]


# ---------------------------------------------------------------------------
# Lifecycle hooks
# ---------------------------------------------------------------------------

async def _on_startup(bot: Bot) -> None:
    """Called once after polling begins."""
    # Register the command menu visible in the Telegram UI
    await bot.set_my_commands(_COMMANDS)
    log.info("[Startup] Bot commands registered.")

    # Warm up the RAG chain in the background so the first query is fast.
    # get_chain() is idempotent — calling it here and in a handler is safe.
    from bot import services
    log.info("[Startup] Starting background chain warm-up …")
    asyncio.create_task(_warm_up_chain())


async def _warm_up_chain() -> None:
    from bot import services
    try:
        await services.get_chain()
        log.info("[Startup] Chain warm-up complete.")
    except Exception as exc:
        log.error(
            "[Startup] Chain warm-up FAILED: %s\n"
            "The bot will retry on the first incoming question.",
            exc,
        )


async def _on_shutdown(bot: Bot) -> None:
    """Called once before the bot stops."""
    log.info("[Shutdown] Bot is shutting down.")
    await bot.session.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    # ── Logging ───────────────────────────────────────────────────────────
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-30s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    # Quiet down noisy third-party loggers
    for noisy in ("httpx", "httpcore", "hpack", "aiogram.event"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # ── Config ────────────────────────────────────────────────────────────
    from bot.config import settings  # validates env vars; raises on missing

    # ── Bot + Dispatcher ──────────────────────────────────────────────────
    bot = Bot(
        token=settings.telegram_bot_token,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    dp = Dispatcher()

    # ── Middleware ────────────────────────────────────────────────────────
    from bot.middleware import ThrottleMiddleware
    dp.message.middleware(
        ThrottleMiddleware(
            max_requests=settings.rate_limit_max,
            window_seconds=settings.rate_limit_window,
        )
    )

    # ── Routers ───────────────────────────────────────────────────────────
    from bot.handlers import register_all
    register_all(dp)

    # ── Lifecycle ────────────────────────────────────────────────────────
    dp.startup.register(_on_startup)
    dp.shutdown.register(_on_shutdown)

    # ── Polling ───────────────────────────────────────────────────────────
    me = await bot.get_me()
    log.info("[Startup] Bot: @%s (id=%d)", me.username, me.id)
    log.info("[Startup] Collection: %s | Top-K: %d | HyDE: %s",
             settings.collection_name, settings.top_k, settings.hyde_enabled)
    log.info("[Startup] Embedder mode: %s", settings.embedder_mode)
    if not settings.is_public:
        log.info("[Startup] Access restricted to %d user(s).", len(settings.allowed_user_ids))
    else:
        log.info("[Startup] Bot is PUBLIC (no ALLOWED_USER_IDS set).")

    await dp.start_polling(bot, allowed_updates=["message"])


if __name__ == "__main__":
    # Allow running as:  python bot/main.py
    # from the project root (adds project root to sys.path automatically).
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    asyncio.run(main())
