"""Gemini models served through Vertex AI (google-genai SDK).

Authentication uses Application Default Credentials
(``gcloud auth application-default login``), which the SDK refreshes
automatically, so long runs do not hit the one-hour access-token expiry of the
OpenAI-compatible Vertex endpoint.
"""

import logging
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from .base import BaseLLM


class VertexGeminiLLM(BaseLLM):
    """Gemini on Vertex AI with greedy decoding and thread-pooled batch generation.

    Providing ``batch_generate`` makes BatchEvaluationCycle process all prompts
    of a cycle concurrently instead of one by one.
    """

    DEFAULT_SYSTEM_PROMPT = "You are a helpful AI assistant."

    def __init__(self, model_config: Dict[str, Any]):
        super().__init__(model_config)
        self.model_id = model_config.get("model_id", "gemini-3.5-flash-lite")
        self.project = model_config.get("project") or os.environ.get("GOOGLE_CLOUD_PROJECT")
        # Gemini 3.x models are served from the global endpoint
        self.location = model_config.get("location") or os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
        self.max_tokens = model_config.get("max_tokens", 4096)
        self.temperature = model_config.get("temperature", 0.0)  # 0.0 = greedy decoding
        self.top_p = model_config.get("top_p", 1.0)
        self.seed = model_config.get("seed")
        # None or "" sends the prompt without a system instruction.
        self.system_prompt = model_config.get("system_prompt", self.DEFAULT_SYSTEM_PROMPT)
        # 0 disables thinking (Gemini Flash/Flash-Lite); None keeps the model default.
        self.thinking_budget = model_config.get("thinking_budget", 0)
        self.max_workers = model_config.get("max_workers", 8)

        # Rate limits (429) are common on shared Gemini quota; retry long enough (~30 min)
        # that a busy quota slows the run down instead of turning cycles into failures.
        self.max_retries = 30
        self.max_empty_retries = 3
        self.retry_delay = 3  # seconds, doubled after each failed attempt (capped at 60, plus jitter)
        self.logger = logging.getLogger(__name__)

        # Billed token totals over all calls (the evaluator stores them in the results file)
        self.usage = {"requests": 0, "input_tokens": 0, "output_tokens": 0, "thinking_tokens": 0}
        self._usage_lock = threading.Lock()

        if not self.project:
            raise ValueError(
                "Vertex AI needs a GCP project: pass --gcp_project or set GOOGLE_CLOUD_PROJECT"
            )

        # Imported here so the rest of the framework works without google-genai installed.
        from google import genai
        from google.genai import types

        self._types = types
        self.client = genai.Client(vertexai=True, project=self.project, location=self.location)

    def _generation_config(self):
        types = self._types
        config = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_output_tokens": self.max_tokens,
            "candidate_count": 1,
        }
        if self.seed is not None:
            config["seed"] = self.seed
        if self.system_prompt:
            config["system_instruction"] = self.system_prompt
        if self.thinking_budget is not None:
            config["thinking_config"] = types.ThinkingConfig(thinking_budget=self.thinking_budget)
        return types.GenerateContentConfig(**config)

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text, retrying rate limits, server errors and (briefly) empty responses.

        Setup errors (401/403/404: bad credentials, project or model id) are raised so a
        misconfigured run stops immediately. Other client errors are not retried and are
        returned as an "Error: ..." string, like the OpenAI backend does.
        """
        delay = self.retry_delay
        last_error: Optional[Exception] = None
        empty_responses = 0

        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model_id,
                    contents=prompt,
                    config=self._generation_config(),
                )
                self._record_usage(response)
                text = response.text
                if text and text.strip():
                    return text
                # Greedy decoding usually repeats an empty answer (safety block,
                # token limit), so give up after a few attempts.
                empty_responses += 1
                reason = response.candidates[0].finish_reason if response.candidates else "no candidates"
                self.logger.warning(f"Empty response (finish_reason={reason}), attempt {empty_responses}")
                if empty_responses >= self.max_empty_retries:
                    return f"Error: Empty response (finish_reason={reason})."
            except Exception as e:  # google.genai raises APIError subclasses carrying .code
                code = getattr(e, "code", None)
                if code in (401, 403, 404):
                    raise
                if isinstance(code, int) and 400 <= code < 500 and code != 429:
                    self.logger.error(f"Vertex AI client error, not retrying: {e}")
                    return f"Error: {e}"
                last_error = e
                self.logger.warning(f"Vertex AI error, retry {attempt + 1}/{self.max_retries}: {code}")
            # Jitter keeps the worker threads from retrying in lockstep
            time.sleep(delay + random.uniform(0, delay / 2))
            delay = min(delay * 2, 60)

        return f"Error: {last_error}"

    def _record_usage(self, response) -> None:
        meta = getattr(response, "usage_metadata", None)
        with self._usage_lock:
            self.usage["requests"] += 1
            if meta is not None:
                self.usage["input_tokens"] += meta.prompt_token_count or 0
                self.usage["output_tokens"] += meta.candidates_token_count or 0
                self.usage["thinking_tokens"] += meta.thoughts_token_count or 0

    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate responses for many prompts concurrently, preserving order."""
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            return list(pool.map(lambda p: self.generate(p, **kwargs), prompts))
