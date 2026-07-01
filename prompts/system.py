"""
prompts/system.py
==================
DeepSeek LLM initialisation — Phase 4.1.

Provides a ``get_llm()`` factory that returns a configured ``ChatDeepSeek``
instance.  All tools in Phase 4 import from here so LLM config is a single
source of truth.

Retry behaviour (Phase 4.6) is implemented here via a thin wrapper that
re-calls the LLM with exponential back-off on rate-limit / server errors.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ── LLM config ─────────────────────────────────────────────────────────────────
DEEPSEEK_MODEL: str = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
DEEPSEEK_TEMPERATURE: float = float(os.getenv("DEEPSEEK_TEMPERATURE", "0.0"))
DEEPSEEK_MAX_TOKENS: int = int(os.getenv("DEEPSEEK_MAX_TOKENS", "4096"))

# ── Retry config ────────────────────────────────────────────────────────────────
MAX_RETRIES: int = 6
RETRY_BASE_DELAY: float = 2.0   # seconds; doubles each attempt


def get_llm():
    """Return a configured ``ChatDeepSeek`` instance.

    Uses ``deepseek-chat`` by default.  Temperature is set to 0 for
    deterministic structured output; ``max_tokens`` is generous enough for long
    candidate reports.

    Returns
    -------
    ChatDeepSeek
        A ready-to-use LangChain chat model instance.

    Raises
    ------
    EnvironmentError
        If ``DEEPSEEK_API_KEY`` is not set in the environment.
    """
    try:
        from langchain_deepseek import ChatDeepSeek
    except ImportError:
        raise ImportError("Please run: pip install langchain-deepseek")

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "DEEPSEEK_API_KEY is not set. Add it to your .env file."
        )

    return ChatDeepSeek(
        model=DEEPSEEK_MODEL,
        temperature=DEEPSEEK_TEMPERATURE,
        max_tokens=DEEPSEEK_MAX_TOKENS,
        api_key=api_key,
        max_retries=0,  # We handle retries natively
    )


def call_llm_with_retry(
    llm,
    messages: list,
    max_retries: int = MAX_RETRIES,
    base_delay: float = RETRY_BASE_DELAY,
) -> Any:
    """Invoke ``llm`` with exponential back-off retry on transient errors.

    Retries on:
    - DeepSeek rate-limit errors (HTTP 429)
    - DeepSeek server errors (HTTP 5xx)
    - Network timeouts / connection errors

    Parameters
    ----------
    llm:
        A LangChain chat model (``ChatDeepSeek`` or any compatible model).
    messages:
        List of LangChain message objects to send.
    max_retries:
        Maximum number of attempts (default 3).
    base_delay:
        Initial wait in seconds; doubles each retry (1s → 2s → 4s).

    Returns
    -------
    AIMessage
        The model's response.

    Raises
    ------
    RuntimeError
        After all retries are exhausted.
    """
    last_exc: Exception | None = None
    delay = base_delay

    for attempt in range(1, max_retries + 1):
        try:
            logger.info("call_llm_with_retry: Sending request to LLM (Attempt %d/%d) with %d messages...", attempt, max_retries, len(messages))
            if messages:
                last_msg = messages[-1].content if hasattr(messages[-1], "content") else str(messages[-1])
                logger.info("call_llm_with_retry: Last message snippet: %r", str(last_msg)[:200] + "..." if len(str(last_msg)) > 200 else last_msg)
            
            response = llm.invoke(messages)
            
            logger.info("call_llm_with_retry: SUCCESS! Received response of length %d", len(response.content if hasattr(response, "content") else str(response)))
            return response
        except Exception as exc:
            last_exc = exc
            err_str = str(exc).lower()

            # Determine if this is a retryable error
            retryable = any(
                kw in err_str
                for kw in ("rate limit", "429", "503", "502", "500",
                            "timeout", "connection", "server error")
            )

            if not retryable or attempt == max_retries:
                logger.error(
                    "call_llm_with_retry: non-retryable error or max retries "
                    "reached (attempt %d/%d): %s",
                    attempt, max_retries, exc,
                )
                break

            if "429" in err_str or "rate limit" in err_str:
                logger.warning("call_llm_with_retry: Rate limit hit. Sleeping for 10 seconds to allow limits to reset...")
                time.sleep(10)
            else:
                logger.warning(
                    "call_llm_with_retry: transient error on attempt %d/%d — "
                    "retrying in %.1fs. Error: %s",
                    attempt, max_retries, delay, exc,
                )
                time.sleep(delay)
                delay *= 2

    raise RuntimeError(
        f"DeepSeek LLM call failed after {max_retries} attempt(s): {last_exc}"
    ) from last_exc
