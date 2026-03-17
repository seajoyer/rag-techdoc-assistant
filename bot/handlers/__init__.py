"""
handlers/__init__.py
--------------------
Registers all routers onto the dispatcher.

Import order matters: command router must come first so /start /help /status
are matched before the catch-all text handler in chat.py.
"""

from aiogram import Dispatcher

from .commands import router as commands_router
from .chat import router as chat_router


def register_all(dp: Dispatcher) -> None:
    """Include all routers into *dp* in priority order."""
    dp.include_router(commands_router)
    dp.include_router(chat_router)
