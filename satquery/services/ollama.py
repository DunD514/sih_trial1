"""
Centralized Ollama Service for SatQuery AI.
Provides robust connection management, JSON/schema extraction, error handling,
and structured interaction with qwen3:8b via Ollama.
"""

import os
import re
import json
import logging
import time
from typing import Dict, Any, List, Optional, Tuple
import httpx

logger = logging.getLogger("satquery.ollama")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")


class OllamaService:
    """
    Centralized communication and validation layer for Ollama qwen3:8b.
    Optimized for minimum latency via persistent connection pooling,
    keep_alive to prevent model eviction, and tuned context limits.
    """

    def __init__(self, base_url: str = OLLAMA_BASE_URL, model: str = OLLAMA_MODEL):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._last_health_check_time = 0.0
        self._cached_health_result: Optional[Tuple[bool, str]] = None
        self._client: Optional[httpx.AsyncClient] = None

    def _get_client(self) -> httpx.AsyncClient:
        """Returns a reusable, pooled httpx.AsyncClient to eliminate connection handshake latency."""
        if self._client is None or self._client.is_closed:
            limits = httpx.Limits(max_keepalive_connections=10, max_connections=20, keepalive_expiry=60.0)
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(120.0, connect=5.0),
                limits=limits
            )
        return self._client

    async def check_health(self) -> Tuple[bool, str]:
        """
        Verifies that Ollama is reachable and that the required model is installed.
        Uses a 10-second cache to avoid connection overhead on every turn.
        """
        now = time.time()
        if self._cached_health_result is not None and (now - self._last_health_check_time) < 10.0:
            return self._cached_health_result

        try:
            client = self._get_client()
            resp = await client.get(f"{self.base_url}/api/tags")
            resp.raise_for_status()
            models = [m.get("name", "") for m in resp.json().get("models", [])]
            if any(self.model in m for m in models):
                res = (True, "ONLINE")
            else:
                short_list = ", ".join(models[:5]) or "none"
                res = (False, f"Model '{self.model}' not found in Ollama. Available: {short_list}")
        except httpx.ConnectError:
            res = (False, f"Ollama server not reachable at {self.base_url}")
        except Exception as exc:
            res = (False, f"Ollama health check error: {str(exc)}")

        self._cached_health_result = res
        self._last_health_check_time = now
        return res

    def extract_json(self, text: str) -> Dict[str, Any]:
        """
        Strips <think>...</think> reasoning blocks, markdown fences, and extracts a valid JSON object.
        """
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.MULTILINE).strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        for pattern in [r"\{[\s\S]*\}", r"\{.*?\}"]:
            m = re.search(pattern, cleaned, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group())
                except json.JSONDecodeError:
                    continue

        raise ValueError(f"Failed to extract valid JSON from LLM output: {text[:250]}")

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        num_predict: int = 160,
        stop: Optional[List[str]] = None,
        timeout: float = 60.0,
        think: bool = False
    ) -> str:
        """
        Sends chat completion request to Ollama and returns assistant content string.
        Optimized with keep_alive='30m' to prevent cold eviction and num_ctx=2048 to save memory.
        Passes think=False to disable thinking generation on reasoning models (e.g. qwen3:8b),
        and prevents internal chain-of-thought from leaking into final answers.
        """
        patched_messages = list(messages)
        if "qwen3" in self.model.lower() and patched_messages:
            last = patched_messages[-1]
            if last.get("role") == "user" and "/no_think" not in last.get("content", ""):
                patched_messages[-1] = {
                    **last,
                    "content": last["content"] + " /no_think"
                }

        options: Dict[str, Any] = {
            "temperature": temperature,
            "num_predict": num_predict,
            "num_ctx": 2048,
            "top_p": 0.9,
            "num_thread": min(8, max(2, (os.cpu_count() or 4) // 2))
        }
        if stop:
            options["stop"] = stop

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": patched_messages,
            "stream": False,
            "keep_alive": "30m",
            "think": think,
            "options": options
        }

        client = self._get_client()
        resp = await client.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=timeout
        )

        resp.raise_for_status()
        data = resp.json()
        msg = data.get("message", {})
        content = msg.get("content", "")
        if not content:
            if think and msg.get("thinking"):
                content = msg.get("thinking", "")
            else:
                logger.warning(
                    "[OllamaService] Model returned empty content (had thinking: %s).",
                    bool(msg.get("thinking"))
                )
                return ""
        return content

    async def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        num_predict: int = 250,
        timeout: float = 60.0
    ) -> Dict[str, Any]:
        """
        Calls Ollama and extracts a structured JSON object.
        """
        raw_text = await self.chat(
            messages=messages,
            temperature=temperature,
            num_predict=num_predict,
            timeout=timeout
        )
        return self.extract_json(raw_text)


# Singleton instance
ollama_service = OllamaService()
