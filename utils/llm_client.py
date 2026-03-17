"""
Unified LLM client for GenArtist.

Provides a single generate_reply() entrypoint that wraps OpenAI-compatible chat
models (including proxies or local gateways) via the official openai library.
"""
from __future__ import annotations

import logging
from typing import Optional, List, Dict, Any

from openai import OpenAI
from openai import APIError, RateLimitError, APITimeoutError

from config import OPENAI_API_KEY, OPENAI_BASE_URL


logger = logging.getLogger(__name__)


class LLMClientError(RuntimeError):
    """Generic error raised by llm_client on failure."""


def _build_client() -> OpenAI:
    """
    Instantiate an OpenAI client using config.

    OPENAI_BASE_URL allows pointing to proxies or local OpenAI-compatible gateways.
    """
    if not OPENAI_API_KEY:
        raise LLMClientError("OPENAI_API_KEY is not set; cannot contact LLM backend.")
    if not OPENAI_BASE_URL:
        raise LLMClientError("OPENAI_BASE_URL is not set; set it in .env (e.g. DashScope compatible-mode URL).")

    return OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)


# Model routing: vision vs text-only to avoid 400 modality mismatch
DEFAULT_VISION_MODEL = "qwen-vl-max"
DEFAULT_TEXT_MODEL = "qwen-max"


def generate_reply(
    system_prompt: str,
    user_prompt: str,
    image_b64: Optional[str] = None,
    model: str = "qwen-max",
) -> str:
    """
    High-level helper for chat completion.

    - system_prompt: Instructions for the assistant role
    - user_prompt: Main user message (may already include templates)
    - image_b64: Optional base64-encoded image for multimodal input
    - model: Chat model name (overridden by vision/text routing when image present)

    When image_b64 is present, request is routed to vision model (e.g. qwen-vl-max);
    otherwise to text-only model (e.g. qwen-max) to prevent 400 modality mismatch.

    Returns the assistant's message content as a plain string.
    Raises LLMClientError on failures.
    """
    # Pre-dispatch: force vision model when image input exists
    effective_model = DEFAULT_VISION_MODEL if image_b64 else (model or DEFAULT_TEXT_MODEL)

    client = _build_client()

    messages: List[Dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    if image_b64:
        # multimodal content
        content = [
            {"type": "text", "text": user_prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
        ]
    else:
        content = user_prompt

    messages.append({"role": "user", "content": content})

    try:
        logger.info("LLM request to %s model=%s", OPENAI_BASE_URL, effective_model)
        resp = client.chat.completions.create(model=effective_model, messages=messages)
        return resp.choices[0].message.content
    except (RateLimitError, APITimeoutError) as e:
        msg = f"LLM rate limit/timeout error: {e}"
        logger.error(msg)
        raise LLMClientError(msg) from e
    except APIError as e:
        msg = f"LLM API error: {e}"
        logger.error(msg)
        raise LLMClientError(msg) from e
    except Exception as e:
        msg = f"LLM unknown error: {e}"
        logger.error(msg)
        raise LLMClientError(msg) from e

