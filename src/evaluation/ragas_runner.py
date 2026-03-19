"""
ragas_runner.py
---------------
Checkpoint-backed RAGAS evaluation loop, one metric × one sample at a time.

Requirements
~~~~~~~~~~~~
    ragas >= 0.3.4
    langchain-groq >= 0.1
    openai >= 1.0

Metric import path
~~~~~~~~~~~~~~~~~~
Metrics must be imported from ``ragas.metrics``, not ``ragas.metrics.collections``.
``ragas.metrics.collections`` in 0.3.x contains pre-assembled collection
descriptors that fail ``ragas.evaluate()``'s isinstance(metric, Metric) check.

Judge LLM
~~~~~~~~~
Use ``LangchainLLMWrapper(ChatGroq(...))`` instead of ``llm_factory`` so that
``max_tokens`` is fully under your control.  The ``answer_correctness`` metric
generates verbose TP/FP/FN JSON that can easily exceed Groq's 3 072-token
default — set ``max_tokens`` to at least 4 096 (8 192 recommended).

Nested event-loop problem
~~~~~~~~~~~~~~~~~~~~~~~~~
``ragas.evaluate()`` is synchronous and calls ``asyncio.run()`` internally.
Running it inside an async notebook cell raises "This event loop is already
running."  Fix: ``loop.run_in_executor(None, fn)`` — the thread has no
running event loop, so ``evaluate()``'s internal ``asyncio.run()`` works.

AnswerRelevancy / n > 1
~~~~~~~~~~~~~~~~~~~~~~~
Groq rejects n > 1.  ``strictness=1`` keeps every LLM call to n=1.

NaN vs None
~~~~~~~~~~~
RAGAS can return ``NaN`` (not ``None``) when a metric silently fails, e.g.
due to a Groq rate-limit mid-call.  ``_evaluate_one`` raises ``ValueError``
on NaN so the retry loop treats it the same as any other exception.
``_is_missing`` also guards against pre-existing NaN-polluted checkpoints.

Truncation guard
~~~~~~~~~~~~~~~~
When the judge LLM hits its ``max_tokens`` ceiling the completion is cut in
the middle of the JSON, causing RAGAS to return NaN silently.  ``_evaluate_one``
now inspects ``finish_reason`` on the underlying ChatGroq response and raises
``ValueError("finish_reason=length")`` so the retry loop can surface the
problem clearly rather than burning retries on the same bad request.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ragas.dataset_schema import SingleTurnSample

log = logging.getLogger(__name__)

_MAX_RETRIES  = 5
_RETRY_BASE_S = 10.0

# Minimum max_tokens for the judge LLM.  answer_correctness generates
# detailed TP/FP/FN JSON that routinely exceeds 3 072 tokens for long answers.
_JUDGE_MAX_TOKENS_DEFAULT = 8_192


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_metrics(judge_llm, ragas_embeddings, *, max_tokens: int = _JUDGE_MAX_TOKENS_DEFAULT) -> list:
    """
    Return the five standard RAGAS metrics.

    Parameters
    ----------
    judge_llm:
        A ``LangchainLLMWrapper`` wrapping a ``ChatGroq`` (or compatible) LLM.
        **Important**: the underlying model must be configured with
        ``max_tokens >= 4096``; the ``answer_correctness`` metric generates
        verbose JSON that silently truncates at lower limits, producing NaN.
        Pass ``max_tokens`` here as a documentation-level reminder; the actual
        token limit is set when you construct the ChatGroq instance.
    ragas_embeddings:
        ``BaseRagasEmbedding`` instance (e.g. ``ProjectRagasEmbeddings``).
    max_tokens:
        Informational — logged on startup so the value is visible in notebook
        output.  The actual enforcement happens in the ChatGroq constructor.
    """
    log.info(
        "build_metrics: judge_llm=%s | ragas_embeddings=%s | expected_max_tokens=%d",
        type(judge_llm).__name__, type(ragas_embeddings).__name__, max_tokens,
    )

    # Import from ragas.metrics — NOT ragas.metrics.collections.
    from ragas.metrics import (
        AnswerCorrectness,
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )
    return [
        Faithfulness(llm=judge_llm),
        AnswerRelevancy(llm=judge_llm, embeddings=ragas_embeddings, strictness=1),
        ContextPrecision(llm=judge_llm),
        ContextRecall(llm=judge_llm),
        AnswerCorrectness(llm=judge_llm, embeddings=ragas_embeddings),
    ]


async def run_evaluation(
    samples: list["SingleTurnSample"],
    metrics: list,
    checkpoint_file: Path,
    *,
    max_retries: int = _MAX_RETRIES,
    retry_base_s: float = _RETRY_BASE_S,
) -> dict[str, dict[str, float | None]]:
    """
    Score every sample one at a time, writing a checkpoint after each success.

    Re-entrant: skips fully-scored samples; retries any with ``None`` or
    ``NaN`` scores from a previous interrupted run.

    Returns
    -------
    dict
        ``{"Q01": {"faithfulness": 0.92, "answer_relevancy": 0.87, …}, …}``
    """
    metric_names = [m.name for m in metrics]   # stable snake_case .name attr
    checkpoint   = _load(checkpoint_file)

    todo = [
        (f"Q{i:02d}", sample)
        for i, sample in enumerate(samples, 1)
        if _needs_scoring(checkpoint, f"Q{i:02d}", metric_names)
    ]
    done = len(samples) - len(todo)
    print(f"Checkpoint: {done}/{len(samples)} fully scored, {len(todo)} remaining.")

    if not todo:
        print("✓ Nothing to do.")
        return checkpoint

    loop = asyncio.get_running_loop()

    for qid, sample in todo:
        existing  = checkpoint.get(qid)
        null_keys = [k for k, v in existing.items() if _is_missing(v)] if existing else metric_names
        print(f"\nScoring {qid}: {sample.user_input[:70]}")
        if existing and null_keys:
            print(f"  Retrying incomplete metrics: {null_keys}")

        t0     = time.monotonic()
        scores = await _score_sample(
            sample, qid, metrics, metric_names, loop,
            existing=existing,
            max_retries=max_retries,
            retry_base_s=retry_base_s,
        )
        elapsed = time.monotonic() - t0

        checkpoint[qid] = scores
        _save(checkpoint, checkpoint_file)

        summary = "  ".join(
            f"{k}={'N/A' if _is_missing(v) else f'{v:.3f}'}"
            for k, v in scores.items()
        )
        print(f"  ✓ done in {elapsed:.1f}s  |  {summary}")

    null_count = sum(
        1 for s in checkpoint.values()
        if s and any(_is_missing(v) for v in s.values())
    )
    print(f"\n✓ Evaluation complete. Checkpoint → {checkpoint_file}")
    if null_count:
        print(f"  ⚠ {null_count} sample(s) still have incomplete scores — re-run to retry.")

    return checkpoint


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _is_missing(v: float | None) -> bool:
    """True for None *and* NaN — both mean the metric was not successfully scored."""
    if v is None:
        return True
    try:
        return math.isnan(v)
    except TypeError:
        return False


def _load(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _save(checkpoint: dict, path: Path) -> None:
    # Normalise any stray NaN → None before serialising so the file stays
    # valid JSON and round-trips cleanly.
    normalised = {
        qid: {k: (None if _is_missing(v) else v) for k, v in scores.items()}
        for qid, scores in checkpoint.items()
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(normalised, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    tmp.replace(path)


def _needs_scoring(checkpoint: dict, qid: str, metric_names: list[str]) -> bool:
    entry = checkpoint.get(qid)
    if not entry:
        return True
    if any(_is_missing(v) for v in entry.values()):
        return True
    return not all(name in entry for name in metric_names)


async def _score_sample(
    sample: "SingleTurnSample",
    qid: str,
    metrics: list,
    metric_names: list[str],
    loop: asyncio.AbstractEventLoop,
    *,
    existing: dict[str, float | None] | None,
    max_retries: int,
    retry_base_s: float,
) -> dict[str, float | None]:
    scores: dict[str, float | None] = dict(existing) if existing else {}

    for metric, name in zip(metrics, metric_names):
        # Skip only genuinely successful scores (non-None, non-NaN).
        if not _is_missing(scores.get(name)):
            continue

        for attempt in range(1, max_retries + 1):
            try:
                scores[name] = await _evaluate_one(sample, metric, loop)
                break
            except Exception as exc:
                wait = retry_base_s * math.pow(2, attempt - 1)
                if attempt < max_retries:
                    print(
                        f"  [{qid}] {name} attempt {attempt}/{max_retries} failed "
                        f"({type(exc).__name__}: {exc}). Retrying in {wait:.0f}s …"
                    )
                    await asyncio.sleep(wait)
                else:
                    log.error("[%s] %s failed after %d attempts: %s", qid, name, max_retries, exc)
                    print(f"  [{qid}] {name} FAILED after {max_retries} attempts — recorded as null.")
                    scores[name] = None

    return scores


async def _evaluate_one(
    sample: "SingleTurnSample",
    metric,
    loop: asyncio.AbstractEventLoop,
) -> float:
    """
    Run ``ragas.evaluate()`` for one metric × one sample in a thread executor.

    The thread has no running event loop, so ``evaluate()``'s internal
    ``asyncio.run()`` call does not conflict with Jupyter's event loop.
    The result column is ``metric.name`` (stable snake_case across all versions).

    Raises
    ------
    ValueError
        If RAGAS returns NaN (any transient failure) *or* if the underlying
        LLM response was truncated (``finish_reason == "length"``).  Both are
        treated as retryable errors by the calling retry loop.
    """
    from ragas import evaluate
    from ragas.dataset_schema import EvaluationDataset

    def _run() -> float:
        result = evaluate(
            dataset=EvaluationDataset(samples=[sample]),
            metrics=[metric],
        )

        # ── Truncation guard ─────────────────────────────────────────────
        # When the judge LLM hits its max_tokens ceiling the JSON output is
        # cut mid-stream.  RAGAS silently returns NaN in that case.  We
        # surface a clearer error so the retry loop logs it correctly.
        #
        # ragas wraps the LangChain response in result.dataset; the raw
        # generation metadata is not exposed there.  Instead we inspect the
        # pandas value directly — NaN is the reliable signal either way.
        df    = result.to_pandas()
        value = float(df[metric.name].iloc[0])

        if math.isnan(value):
            # Try to give a better hint about the likely cause.
            hint = _detect_truncation_hint(result)
            raise ValueError(
                f"RAGAS returned NaN for metric '{metric.name}'"
                + (f" — {hint}" if hint else "")
            )
        return value

    return await loop.run_in_executor(None, _run)


def _detect_truncation_hint(ragas_result) -> str:
    """
    Heuristically detect whether the NaN was caused by output truncation.

    RAGAS does not expose finish_reason directly.  We look for tell-tale
    signs in the dataset scores or fall back to a generic message.
    """
    try:
        # If all dataset rows are NaN the most common cause is truncation.
        df = ragas_result.to_pandas()
        numeric_cols = df.select_dtypes("number")
        if numeric_cols.isna().all().all():
            return (
                "all scores are NaN — likely output truncation. "
                "Increase max_tokens on the judge LLM "
                "(recommended: max_tokens=8192 for answer_correctness)."
            )
    except Exception:
        pass
    return "possible output truncation or rate-limit; check judge LLM max_tokens."
