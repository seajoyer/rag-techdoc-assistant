"""
config.py
---------
All runtime configuration for the Telegram bot, loaded from environment
variables (and optionally a .env file).

Embedder modes
~~~~~~~~~~~~~~
``EMBEDDER_MODE`` controls which backend computes dense vectors:

    local   → BGEM3Embedder   — runs BAAI/bge-m3 on the local GPU/CPU.
                                 Requires torch + FlagEmbedding.
    hf      → HFInferenceEmbedder — calls the HuggingFace Inference API.
                                 Requires HUGGINGFACEHUB_API_TOKEN.
    auto    → Try local first; fall back to HF if torch/FlagEmbedding
              is unavailable or CUDA isn't present.  (default)
"""

from __future__ import annotations

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",          # ignore unknown env vars silently
    )

    # ── Telegram ──────────────────────────────────────────────────────────
    telegram_bot_token: str = Field(..., validation_alias="TELEGRAM_BOT_TOKEN")

    # ── Qdrant ────────────────────────────────────────────────────────────
    qdrant_url: str = Field(..., validation_alias="QDRANT_URL")
    qdrant_api_key: str = Field(..., validation_alias="QDRANT_API_KEY")
    collection_name: str = Field("pytorch_docs", validation_alias="COLLECTION_NAME")

    # ── Groq ──────────────────────────────────────────────────────────────
    groq_api_key: str = Field(..., validation_alias="GROQ_API_KEY")

    # ── Embedder ──────────────────────────────────────────────────────────
    embedder_mode: str = Field("auto", validation_alias="EMBEDDER_MODE")
    hf_api_token: str = Field("", validation_alias="HUGGINGFACEHUB_API_TOKEN")

    # ── RAG pipeline ──────────────────────────────────────────────────────
    top_k: int   = Field(6,     validation_alias="TOP_K")
    max_tokens: int   = Field(1024,  validation_alias="MAX_TOKENS")
    temperature: float = Field(0.0,   validation_alias="TEMPERATURE")
    hyde_enabled: bool  = Field(True,  validation_alias="HYDE_ENABLED")

    # ── Rate limiting (per user) ───────────────────────────────────────────
    rate_limit_window: int = Field(60, validation_alias="RATE_LIMIT_WINDOW")
    rate_limit_max: int    = Field(5,  validation_alias="RATE_LIMIT_MAX")

    # ── Query logging ─────────────────────────────────────────────────────
    # Telegram chat ID of the private group where queries are forwarded.
    # Leave empty (or unset) to disable query logging.
    log_group_id: int | None = Field(None, validation_alias="LOG_GROUP_ID")

    # ── Misc ──────────────────────────────────────────────────────────────
    # Telegram user IDs allowed to use the bot.
    # Empty list = no restriction (public bot).
    allowed_user_ids: list[int] = Field(
        default_factory=list, validation_alias="ALLOWED_USER_IDS"
    )

    @field_validator("allowed_user_ids", mode="before")
    @classmethod
    def _parse_user_ids(cls, v: object) -> object:
        if isinstance(v, str):
            return [int(x.strip()) for x in v.split(",") if x.strip()]
        return v

    @property
    def is_public(self) -> bool:
        return not self.allowed_user_ids


# Module-level singleton — import and use anywhere:
#   from bot.config import settings
settings = Settings()
