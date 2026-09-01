"""LLM provider implementations.

Contains the local (Ollama) and API (OpenAI/Anthropic) adapters as
signature-complete stubs, plus a fully-working :class:`EchoLLM` default so the
package imports and tests run without any model backend.

The real adapters are builder-owned (their bodies raise ``NotImplementedError``);
``EchoLLM`` is the offline stub.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Iterator, Optional, Sequence

from deepvision.providers.base import LLMProvider, Message
from deepvision.utils.logger import get_logger

__all__ = ["EchoLLM", "LocalLLM", "APILLM"]

_log = get_logger(__name__)

#: Anthropic model ids that reject a sampling ``temperature`` param with an
#: HTTP 400 (older claude-3-x models and claude-haiku-4-5 still accept it).
_ANTHROPIC_NO_SAMPLING = (
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-fable-5",
)


#: Anthropic model ids where *adaptive thinking is on by default* when the
#: request omits the ``thinking`` field. This matters because ``max_tokens`` is
#: a hard cap on thinking tokens **plus** visible text: leaving thinking
#: implicit lets it eat the budget the summarizer sized for prose, so a
#: Detailed section (16-22 sentences) can come back truncated — or, at the
#: small budgets used elsewhere, empty. We therefore ask for thinking off
#: explicitly. Both ids accept ``{"type": "disabled"}`` (on claude-opus-5 only
#: at effort <= high, which is what we get since we never send
#: ``output_config.effort``). Do NOT add claude-fable-5/claude-mythos-5 here —
#: their thinking cannot be disabled and the field 400s.
_ANTHROPIC_THINKING_ON_BY_DEFAULT = (
    "claude-opus-5",
    "claude-sonnet-5",
)


def _anthropic_accepts_temperature(model: str) -> bool:
    """False for Anthropic model ids that 400 on a ``temperature`` field."""
    return model not in _ANTHROPIC_NO_SAMPLING


def _anthropic_thinking_off(model: str) -> Optional[dict]:
    """The ``thinking`` field to send, or ``None`` to omit it entirely."""
    if model in _ANTHROPIC_THINKING_ON_BY_DEFAULT:
        return {"type": "disabled"}
    return None


class EchoLLM(LLMProvider):
    """A trivial, dependency-free LLM used as the safe import/test default.

    It echoes the last user message back (optionally prefixed), so any code path
    that needs *an* ``LLMProvider`` works before real backends are wired.
    """

    def __init__(self, model: str = "echo", prefix: str = "") -> None:
        self.model = model
        self._prefix = prefix

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ) -> str:
        last_user = next(
            (m.content for m in reversed(messages) if m.role == "user"), ""
        )
        out = f"{self._prefix}{last_user}"
        return out[:max_tokens] if max_tokens else out

    def stream(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ) -> Iterator[str]:
        for token in self.complete(
            messages, temperature=temperature, max_tokens=max_tokens
        ).split(" "):
            yield token + " "

    def health(self) -> bool:
        return True


class LocalLLM(LLMProvider):
    """Local LLM via Ollama (default: llama3.1:8b).

    Talks to the Ollama server's REST API (``/api/chat``) directly over
    :mod:`urllib` — no third-party ``ollama`` client package required, so this
    module has zero optional dependencies to guard. Ollama itself is a local
    server process, not a pip package; when it is not running (or the model
    tag is not pulled) every method here logs a warning and degrades
    gracefully instead of raising, per the local-first contract.
    """

    #: Network timeout (seconds) for calls to the local Ollama server.
    #:
    #: Measured on this machine: a Standard report on llama3.1:8b made 12 LLM
    #: calls averaging ~105s each, and a Detailed call is materially longer
    #: (``_DETAIL_PARAMS`` in ``agents/summarizer_agent.py`` raises max_tokens
    #: from 400 -> 1100 -> 2600). The previous 120s ceiling therefore sat right
    #: at the edge of a *normal* call: six of those twelve calls tripped it and
    #: returned "" , and ``agents.base.complete_with_fallback`` silently
    #: substituted the extractive draft — so asking for more detail produced
    #: *less* model-written prose. 600s gives real headroom, so the fallback
    #: fires only on a genuinely stuck Ollama server rather than on slow but
    #: healthy generation. Overridable per deployment via
    #: ``Config.local_llm_timeout`` / ``DEEPVISION_LOCAL_LLM_TIMEOUT``, which
    #: the provider factory passes in; this constant is the fallback when
    #: ``LocalLLM`` is constructed directly.
    DEFAULT_TIMEOUT = 600.0

    def __init__(
        self,
        model: str = "llama3.1:8b",
        *,
        host: str = "http://localhost:11434",
        timeout: Optional[float] = None,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = self.DEFAULT_TIMEOUT if timeout is None else float(timeout)
        self._log = _log.bind(provider="LocalLLM", model=model)

    def _payload(
        self,
        messages: Sequence[Message],
        *,
        temperature: float,
        max_tokens: Optional[int],
        stream: bool,
    ) -> dict:
        options: dict = {"temperature": temperature}
        if max_tokens:
            options["num_predict"] = max_tokens
        return {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": stream,
            "options": options,
        }

    def _request(self, path: str, payload: dict):
        url = f"{self.host}{path}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        return urllib.request.urlopen(req, timeout=self.timeout)

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ) -> str:
        payload = self._payload(
            messages, temperature=temperature, max_tokens=max_tokens, stream=False
        )
        try:
            with self._request("/api/chat", payload) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            self._log.warning(
                "Ollama unreachable; returning empty completion", error=str(exc)
            )
            return ""
        except (json.JSONDecodeError, KeyError) as exc:
            self._log.warning("Ollama returned an unparseable response", error=str(exc))
            return ""
        return (body.get("message") or {}).get("content", "") or ""

    def stream(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ) -> Iterator[str]:
        payload = self._payload(
            messages, temperature=temperature, max_tokens=max_tokens, stream=True
        )
        try:
            resp = self._request("/api/chat", payload)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            self._log.warning("Ollama unreachable; no stream produced", error=str(exc))
            return
        try:
            with resp:
                for raw_line in resp:
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line.decode("utf-8"))
                    except json.JSONDecodeError:
                        continue
                    delta = (chunk.get("message") or {}).get("content", "")
                    if delta:
                        yield delta
                    if chunk.get("done"):
                        break
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            self._log.warning("Ollama stream interrupted", error=str(exc))
            return

    def health(self) -> bool:
        try:
            req = urllib.request.Request(f"{self.host}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return 200 <= resp.status < 300
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            self._log.info("Ollama health check failed", error=str(exc))
            return False


class APILLM(LLMProvider):
    """API LLM adapter (OpenAI/Anthropic-compatible).

    ``provider`` selects the wire protocol ('openai' | 'anthropic'). The
    factory (``providers/factory.py``) never sets ``provider`` explicitly, so
    when it is left at its default of ``"openai"`` this adapter auto-detects
    Anthropic-style model tags (``claude-*``) from ``model`` and switches
    protocol + default base URL accordingly — see the class docstring in the
    report handed back to the integrator for why.

    Implemented over :mod:`urllib` (stdlib) rather than the ``openai`` /
    ``anthropic`` SDKs so this module has no third-party import to guard;
    the HTTP contracts of both APIs are simple JSON POSTs.
    """

    DEFAULT_TIMEOUT = 60.0
    _ANTHROPIC_BASE = "https://api.anthropic.com"
    _OPENAI_BASE = "https://api.openai.com/v1"
    _ANTHROPIC_VERSION = "2023-06-01"
    #: Default max_tokens for Anthropic requests when the caller passes none.
    #: 1024 truncates a Detailed report section mid-sentence; 4096 gives room.
    _ANTHROPIC_DEFAULT_MAX_TOKENS = 4096

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        provider: str = "openai",
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.provider = provider
        self.timeout = self.DEFAULT_TIMEOUT
        self._log = _log.bind(provider="APILLM", model=model)

    # -- protocol resolution -------------------------------------------------

    def _protocol(self) -> str:
        if self.provider and self.provider != "openai":
            return self.provider
        m = self.model.lower()
        if m.startswith("claude") or "anthropic" in m:
            return "anthropic"
        return "openai"

    def _resolved_base_url(self, protocol: str) -> str:
        if self.base_url and self.base_url not in (self._OPENAI_BASE, ""):
            return self.base_url.rstrip("/")
        return self._ANTHROPIC_BASE if protocol == "anthropic" else self._OPENAI_BASE

    def _require_key(self) -> str:
        if not self.api_key:
            raise RuntimeError(
                "APILLM requires an api_key; configure llm_api_key in Settings "
                "(llm_mode='api')."
            )
        return self.api_key

    # -- request helpers -------------------------------------------------

    def _post(self, url: str, payload: dict, headers: dict):
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST", headers=headers)
        try:
            return urllib.request.urlopen(req, timeout=self.timeout)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"APILLM request to {url} failed ({exc.code}): {detail}"
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise RuntimeError(f"APILLM request to {url} failed: {exc}") from exc

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ) -> str:
        self._require_key()
        protocol = self._protocol()
        if protocol == "anthropic":
            return self._complete_anthropic(messages, temperature, max_tokens)
        return self._complete_openai(messages, temperature, max_tokens)

    def stream(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ) -> Iterator[str]:
        self._require_key()
        protocol = self._protocol()
        if protocol == "anthropic":
            yield from self._stream_anthropic(messages, temperature, max_tokens)
        else:
            yield from self._stream_openai(messages, temperature, max_tokens)

    def health(self) -> bool:
        return bool(self.api_key)

    # -- OpenAI-compatible protocol -------------------------------------

    def _openai_messages(self, messages: Sequence[Message]) -> list[dict]:
        return [{"role": m.role, "content": m.content} for m in messages]

    def _complete_openai(
        self,
        messages: Sequence[Message],
        temperature: float,
        max_tokens: Optional[int],
    ) -> str:
        url = f"{self._resolved_base_url('openai')}/chat/completions"
        payload: dict = {
            "model": self.model,
            "messages": self._openai_messages(messages),
            "temperature": temperature,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        with self._post(url, payload, headers) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        choices = body.get("choices") or []
        if not choices:
            return ""
        return (choices[0].get("message") or {}).get("content", "") or ""

    def _stream_openai(
        self,
        messages: Sequence[Message],
        temperature: float,
        max_tokens: Optional[int],
    ) -> Iterator[str]:
        url = f"{self._resolved_base_url('openai')}/chat/completions"
        payload: dict = {
            "model": self.model,
            "messages": self._openai_messages(messages),
            "temperature": temperature,
            "stream": True,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        with self._post(url, payload, headers) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = (choices[0].get("delta") or {}).get("content", "")
                if delta:
                    yield delta

    # -- Anthropic protocol -----------------------------------------------

    def _anthropic_payload(
        self,
        messages: Sequence[Message],
        temperature: float,
        max_tokens: Optional[int],
        *,
        stream: bool,
    ) -> dict:
        system_parts = [m.content for m in messages if m.role == "system"]
        turns = [
            {"role": m.role, "content": m.content}
            for m in messages
            if m.role in ("user", "assistant")
        ]
        payload: dict = {
            "model": self.model,
            "messages": turns,
            "max_tokens": max_tokens or self._ANTHROPIC_DEFAULT_MAX_TOKENS,
            "stream": stream,
        }
        if _anthropic_accepts_temperature(self.model):
            payload["temperature"] = temperature
        thinking = _anthropic_thinking_off(self.model)
        if thinking is not None:
            # max_tokens caps thinking + text together; the callers here size it
            # for prose only, so keep thinking out of the budget.
            payload["thinking"] = thinking
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)
        return payload

    def _anthropic_headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "x-api-key": self.api_key or "",
            "anthropic-version": self._ANTHROPIC_VERSION,
        }

    def _complete_anthropic(
        self,
        messages: Sequence[Message],
        temperature: float,
        max_tokens: Optional[int],
    ) -> str:
        url = f"{self._resolved_base_url('anthropic')}/v1/messages"
        payload = self._anthropic_payload(
            messages, temperature, max_tokens, stream=False
        )
        with self._post(url, payload, self._anthropic_headers()) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        parts = body.get("content") or []
        return "".join(p.get("text", "") for p in parts if p.get("type") == "text")

    def _stream_anthropic(
        self,
        messages: Sequence[Message],
        temperature: float,
        max_tokens: Optional[int],
    ) -> Iterator[str]:
        url = f"{self._resolved_base_url('anthropic')}/v1/messages"
        payload = self._anthropic_payload(
            messages, temperature, max_tokens, stream=True
        )
        with self._post(url, payload, self._anthropic_headers()) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if not data:
                    continue
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if event.get("type") == "content_block_delta":
                    delta = (event.get("delta") or {}).get("text", "")
                    if delta:
                        yield delta
                elif event.get("type") == "message_stop":
                    break
