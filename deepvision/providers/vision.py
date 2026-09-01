"""Vision provider implementations.

Local (Qwen2-VL / Florence-2 via transformers) and API vision adapters as
signature stubs, plus a functional :class:`NullVision` default so imports and
tests work without a model.

Real adapters are builder-owned (bodies raise ``NotImplementedError``);
``NullVision`` is the offline stub.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

from deepvision.providers.base import VisionAvailability, VisionProvider
from deepvision.utils.logger import get_logger

__all__ = ["NullVision", "LocalVision", "APIVision"]

_log = get_logger(__name__)

#: Output budget for one figure/table description, both wire protocols.
_DESCRIBE_MAX_TOKENS = 512

#: Anthropic model ids where adaptive thinking runs unless the request opts out
#: (mirrors ``providers.llm._ANTHROPIC_THINKING_ON_BY_DEFAULT``). Because
#: ``max_tokens`` bounds thinking *and* visible text, omitting the field at the
#: budget above would spend most of it on reasoning we never read and return a
#: truncated — often empty — description. Never list claude-fable-5 /
#: claude-mythos-5 here: their thinking cannot be disabled and the field 400s.
_ANTHROPIC_THINKING_ON_BY_DEFAULT = (
    "claude-opus-5",
    "claude-sonnet-5",
)


def _hf_cache_dir() -> Path:
    """Best-effort resolution of the local Hugging Face hub cache directory.

    Honors ``HF_HOME`` / ``HUGGINGFACE_HUB_CACHE`` if set, otherwise falls back
    to the standard ``~/.cache/huggingface/hub`` location. Implemented without
    importing ``huggingface_hub`` so the availability check works even when
    that package (a transitive dependency of ``transformers``) is absent.
    """
    hf_home = os.environ.get("HF_HOME")
    hub_cache = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if hub_cache:
        return Path(hub_cache)
    if hf_home:
        return Path(hf_home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _hf_model_dir(model_id: str) -> Path:
    return _hf_cache_dir() / ("models--" + model_id.replace("/", "--"))


# Cache of computed model-snapshot sizes, keyed by the snapshots dir path, so
# repeated availability()/status polls (e.g. GET /settings, GET /health hit
# this on every call) don't re-walk a multi-GB HuggingFace cache directory
# each time. Invalidated only when the snapshots dir's own mtime changes
# (i.e. files were added/removed directly under it — snapshot contents are
# otherwise immutable once downloaded), not on a wall-clock TTL, so a fresh
# download is picked up without a stale reading lingering.
_SIZE_HINT_CACHE: dict[str, tuple[float, Optional[str]]] = {}
_SIZE_HINT_LOCK = threading.Lock()


def _cached_size_hint(model_dir: Path) -> Optional[str]:
    """Human-readable total size of ``model_dir``, memoized by snapshot mtime."""
    snapshots = model_dir / "snapshots"
    try:
        mtime = snapshots.stat().st_mtime
    except OSError:
        return None

    key = str(snapshots)
    with _SIZE_HINT_LOCK:
        cached = _SIZE_HINT_CACHE.get(key)
        if cached is not None and cached[0] == mtime:
            return cached[1]

    try:
        total_bytes = sum(
            f.stat().st_size for f in model_dir.rglob("*") if f.is_file()
        )
        size_hint = f"{total_bytes / (1024 ** 3):.1f} GB"
    except OSError:
        size_hint = None

    with _SIZE_HINT_LOCK:
        _SIZE_HINT_CACHE[key] = (mtime, size_hint)
    return size_hint


class NullVision(VisionProvider):
    """A no-op vision provider used as the safe import/test default.

    Returns the caption hint (or a placeholder) and reports itself unavailable so
    callers can detect that no real vision backend is wired.
    """

    def __init__(self, model: str = "null-vision") -> None:
        self.model = model

    def describe_image(
        self,
        image_path: str,
        *,
        prompt: Optional[str] = None,
        caption_hint: Optional[str] = None,
    ) -> str:
        return caption_hint or ""

    def availability(self) -> VisionAvailability:
        return VisionAvailability(
            available=False,
            model=self.model,
            detail="No vision backend configured.",
        )


class LocalVision(VisionProvider):
    """Local vision model (Qwen2-VL-7B / Florence-2).

    Uses ``transformers`` + ``torch`` + ``Pillow`` via the generic
    ``AutoModelForImageTextToText`` / ``AutoProcessor`` chat-template path,
    which covers Qwen2-VL (the configured default). All three imports are
    lazy and guarded: the module — and this class — import fine with none of
    them installed; only calling :meth:`describe_image` or
    :meth:`availability` touches them, and both degrade gracefully (log +
    fall back) rather than raising, per the local-first contract.

    Note: Florence-2 uses a different (task-prompt, ``trust_remote_code``)
    calling convention than the chat-template VLM path used here. If a
    Florence-2 model id is selected via Settings this adapter will still try
    the chat-template path, fail, log, and fall back to ``caption_hint`` —
    see the integration report for this known limitation.
    """

    #: Generic instruction used when no prompt/caption hint is supplied.
    _DEFAULT_PROMPT = (
        "Describe this figure or table from a research paper in detail, "
        "including axes, trends, structure, and the key takeaway."
    )
    _MAX_NEW_TOKENS = 256

    #: Florence-2 task prompt that yields a rich free-text description.
    _FLORENCE_TASK = "<MORE_DETAILED_CAPTION>"

    def __init__(self, model: str = "Qwen/Qwen2-VL-7B-Instruct") -> None:
        self.model = model
        self._is_florence = "florence" in model.lower()
        self._loaded = None  # (processor, model, device) once lazily loaded
        self._log = _log.bind(provider="LocalVision", model=model)

    # -- lazy model loading -----------------------------------------------

    def _pick_device(self, torch):
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"

    def _ensure_loaded(self):
        if self._loaded is not None:
            return self._loaded
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "transformers/torch are not installed; local vision is unavailable"
            ) from exc

        device = self._pick_device(torch)
        # float16 only helps on CUDA; Florence-2 / CPU / MPS want float32.
        dtype = torch.float16 if device == "cuda" else torch.float32

        if self._is_florence:
            # Florence-2: causal-LM head + custom code, task-prompt calling
            # convention (distinct from the chat-template VLM path).
            try:
                from transformers import AutoModelForCausalLM, AutoProcessor
            except ImportError as exc:
                raise RuntimeError(
                    "transformers/torch are not installed; local vision is unavailable"
                ) from exc
            processor = AutoProcessor.from_pretrained(
                self.model, trust_remote_code=True
            )
            hf_model = AutoModelForCausalLM.from_pretrained(
                self.model, torch_dtype=dtype, trust_remote_code=True
            )
        else:
            try:
                from transformers import AutoModelForImageTextToText, AutoProcessor
            except ImportError as exc:
                raise RuntimeError(
                    "transformers/torch are not installed; local vision is unavailable"
                ) from exc
            processor = AutoProcessor.from_pretrained(self.model)
            hf_model = AutoModelForImageTextToText.from_pretrained(
                self.model, torch_dtype=dtype
            )

        hf_model.to(device)
        hf_model.eval()
        self._loaded = (processor, hf_model, device)
        return self._loaded

    def describe_image(
        self,
        image_path: str,
        *,
        prompt: Optional[str] = None,
        caption_hint: Optional[str] = None,
    ) -> str:
        try:
            processor, hf_model, device = self._ensure_loaded()
        except Exception as exc:  # noqa: BLE001 - load failures must degrade, not crash
            self._log.warning(
                "local vision unavailable, falling back to caption hint",
                error=str(exc),
            )
            return caption_hint or ""

        try:
            from PIL import Image
            import torch

            image = Image.open(image_path).convert("RGB")
            text_prompt = prompt or (
                f"{self._DEFAULT_PROMPT} Nearby caption for context: {caption_hint!r}"
                if caption_hint
                else self._DEFAULT_PROMPT
            )

            if self._is_florence:
                # Florence-2: task-token prompt in, structured parse out.
                inputs = processor(
                    text=self._FLORENCE_TASK, images=image, return_tensors="pt"
                ).to(device)
                with torch.no_grad():
                    generated_ids = hf_model.generate(
                        input_ids=inputs["input_ids"],
                        pixel_values=inputs["pixel_values"],
                        max_new_tokens=self._MAX_NEW_TOKENS,
                        num_beams=3,
                        do_sample=False,
                    )
                raw = processor.batch_decode(
                    generated_ids, skip_special_tokens=False
                )[0]
                parsed = processor.post_process_generation(
                    raw,
                    task=self._FLORENCE_TASK,
                    image_size=(image.width, image.height),
                )
                text = (parsed.get(self._FLORENCE_TASK) or "").strip()
                return text or (caption_hint or "")

            chat = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": text_prompt},
                    ],
                }
            ]
            chat_prompt = processor.apply_chat_template(
                chat, add_generation_prompt=True
            )
            inputs = processor(
                text=[chat_prompt], images=[image], return_tensors="pt"
            ).to(device)
            with torch.no_grad():
                output_ids = hf_model.generate(
                    **inputs, max_new_tokens=self._MAX_NEW_TOKENS
                )
            input_len = inputs["input_ids"].shape[1]
            generated = output_ids[:, input_len:]
            text = processor.batch_decode(generated, skip_special_tokens=True)[0]
            text = text.strip()
            return text or (caption_hint or "")
        except Exception as exc:  # noqa: BLE001 - never crash the pipeline
            self._log.warning(
                "local vision inference failed, falling back to caption hint",
                error=str(exc),
            )
            return caption_hint or ""

    def availability(self) -> VisionAvailability:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError:
            return VisionAvailability(
                available=False,
                model=self.model,
                detail="transformers/torch are not installed.",
            )

        model_dir = _hf_model_dir(self.model)
        snapshots = model_dir / "snapshots"
        has_snapshot = snapshots.is_dir() and any(snapshots.iterdir())
        if not has_snapshot:
            return VisionAvailability(
                available=False,
                model=self.model,
                detail=f"Model weights for {self.model} were not found in the "
                "local Hugging Face cache. Pull them once (e.g. via "
                "`huggingface-cli download`) to enable local vision.",
            )

        size_hint = _cached_size_hint(model_dir)

        return VisionAvailability(
            available=True,
            model=self.model,
            detail="Local vision model installed and available.",
            size_hint=size_hint,
        )


class APIVision(VisionProvider):
    """API vision adapter (multimodal endpoint).

    Implemented over :mod:`urllib` against the OpenAI- or Anthropic-style
    multimodal chat endpoints (mirrors :class:`deepvision.providers.llm.APILLM`'s
    protocol auto-detection from the model tag).
    """

    DEFAULT_TIMEOUT = 60.0
    _ANTHROPIC_BASE = "https://api.anthropic.com"
    _OPENAI_BASE = "https://api.openai.com/v1"
    _ANTHROPIC_VERSION = "2023-06-01"
    _DEFAULT_PROMPT = (
        "Describe this figure or table from a research paper in detail, "
        "including axes, trends, structure, and the key takeaway."
    )

    def __init__(
        self,
        model: str = "gpt-4o",
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
        self._log = _log.bind(provider="APIVision", model=model)

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

    @staticmethod
    def _encode_image(image_path: str) -> tuple[str, str]:
        media_type = mimetypes.guess_type(image_path)[0] or "image/png"
        with open(image_path, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode("ascii")
        return b64, media_type

    def describe_image(
        self,
        image_path: str,
        *,
        prompt: Optional[str] = None,
        caption_hint: Optional[str] = None,
    ) -> str:
        if not self.api_key:
            raise RuntimeError(
                "APIVision requires an api_key; configure vision_api_key in "
                "Settings (vision_mode='api')."
            )
        text_prompt = prompt or (
            f"{self._DEFAULT_PROMPT} Nearby caption for context: {caption_hint!r}"
            if caption_hint
            else self._DEFAULT_PROMPT
        )
        b64, media_type = self._encode_image(image_path)
        protocol = self._protocol()
        try:
            if protocol == "anthropic":
                return self._describe_anthropic(text_prompt, b64, media_type)
            return self._describe_openai(text_prompt, b64, media_type)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"APIVision request failed ({exc.code}): {detail}"
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise RuntimeError(f"APIVision request failed: {exc}") from exc

    def _describe_openai(self, text_prompt: str, b64: str, media_type: str) -> str:
        url = f"{self._resolved_base_url('openai')}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{b64}"
                            },
                        },
                    ],
                }
            ],
            "max_tokens": _DESCRIBE_MAX_TOKENS,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST", headers=headers)
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        choices = body.get("choices") or []
        if not choices:
            return ""
        return (choices[0].get("message") or {}).get("content", "") or ""

    def _describe_anthropic(self, text_prompt: str, b64: str, media_type: str) -> str:
        url = f"{self._resolved_base_url('anthropic')}/v1/messages"
        payload: dict = {
            "model": self.model,
            "max_tokens": _DESCRIBE_MAX_TOKENS,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": b64,
                            },
                        },
                        {"type": "text", "text": text_prompt},
                    ],
                }
            ],
        }
        if self.model in _ANTHROPIC_THINKING_ON_BY_DEFAULT:
            # On these ids adaptive thinking runs unless asked not to, and
            # max_tokens caps thinking + text together — at this budget the
            # figure description would come back truncated or empty.
            payload["thinking"] = {"type": "disabled"}
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key or "",
            "anthropic-version": self._ANTHROPIC_VERSION,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST", headers=headers)
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        parts = body.get("content") or []
        return "".join(p.get("text", "") for p in parts if p.get("type") == "text")

    def availability(self) -> VisionAvailability:
        protocol = self._protocol()
        vendor_label = "Claude" if protocol == "anthropic" else "OpenAI"
        if self.api_key:
            return VisionAvailability(
                available=True,
                model=self.model,
                detail=f"{vendor_label} vision configured ({self.model}).",
            )
        return VisionAvailability(
            available=False,
            model=self.model,
            detail=f"No API key configured for {vendor_label} vision.",
        )
