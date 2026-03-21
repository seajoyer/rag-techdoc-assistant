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

Streaming mode
~~~~~~~~~~~~~~
``STREAMING_ENABLED`` enables token-by-token answer previews via the
Telegram Bot API ``sendMessageDraft`` method (Bot API 9.5+).  Only works
in private chats; group chats fall back to the standard path automatically.

When enabled the bot sends live draft updates while the LLM generates the
answer, then edits the final message with full citation formatting once
generation is complete.  Requires aiogram with ``SendMessageDraft`` support.
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
    top_k: int    = Field(6,     validation_alias="TOP_K")
    max_tokens: int    = Field(1024,  validation_alias="MAX_TOKENS")
    temperature: float = Field(0.0,   validation_alias="TEMPERATURE")
    hyde_enabled: bool  = Field(True,  validation_alias="HYDE_ENABLED")

    # ── Reranker ──────────────────────────────────────────────────────────────
    reranker_enabled: bool  = Field(False, validation_alias="RERANKER_ENABLED")
    reranker_alpha:   float = Field(0.7,   validation_alias="RERANKER_ALPHA")
    # Candidates fetched from Qdrant before reranking; final_top_k = top_k.
    # Set higher than top_k so the CE has more material to reorder.
    reranker_top_k:   int   = Field(12,    validation_alias="RERANKER_TOP_K")

    # ── Streaming (Telegram Bot API 9.5+ sendMessageDraft) ────────────────
    # When True the bot streams token-by-token previews in private chats.
    # Falls back to the standard path automatically for group chats or when
    # the aiogram SendMessageDraft method is unavailable.
    streaming_enabled: bool = Field(False, validation_alias="STREAMING_ENABLED")
    # Minimum seconds between successive sendMessageDraft calls.
    # Lower values feel more responsive but burn more API quota.
    streaming_draft_interval: float = Field(0.5, validation_alias="STREAMING_DRAFT_INTERVAL")

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
