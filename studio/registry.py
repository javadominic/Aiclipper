"""Centralized, self-validating, provider-agnostic AI model registry and deterministic router.

Maintains normalized model records across all providers, performs dynamic discovery
(Ollama, Gemini, OpenAI-compatible, Anthropic), enforces exact-ID routing without
fuzzy substitutions, manages cached discovery with TTL, and implements strict
vs opt-in fallback policies with complete execution verification logging.
"""
import json
import logging
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

from . import config

log = logging.getLogger("studio.registry")

# Model availability status
STATUS_AVAILABLE = "available"        # Verified working via test call
STATUS_UNVERIFIED = "unverified"      # Discovered from provider API, not yet tested
STATUS_UNAVAILABLE = "unavailable"    # 404 / Model not found on provider
STATUS_UNAUTHORIZED = "unauthorized"  # 401/403 Invalid API key or quota exceeded
STATUS_UNSUPPORTED = "unsupported"    # Lacks required capabilities (e.g. generateContent/chat)
STATUS_UNREACHABLE = "unreachable"    # Network timeout or connection refused

DEFAULT_CACHE_TTL_SEC = 3600          # 1 hour discovery cache TTL


@dataclass
class ModelRecord:
    """Normalized record for a model from any provider."""
    provider: str                     # e.g. "gemini", "ollama", "groq", "openai", "anthropic"
    id: str                           # Exact API model ID (e.g. "gemini-2.0-flash", "llama3.1:latest")
    name: str                         # Human-readable display name
    capabilities: Dict[str, bool] = field(default_factory=lambda: {"chat": True, "json": True})
    status: str = STATUS_UNVERIFIED
    last_verified: Optional[float] = None
    last_discovered: float = field(default_factory=time.time)
    error_message: Optional[str] = None
    is_alias: bool = False
    target_id: Optional[str] = None   # If is_alias is True, the exact model ID it maps to
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelRecord":
        return cls(
            provider=str(d.get("provider", "")),
            id=str(d.get("id", "")),
            name=str(d.get("name", d.get("id", ""))),
            capabilities=dict(d.get("capabilities", {"chat": True, "json": True})),
            status=str(d.get("status", STATUS_UNVERIFIED)),
            last_verified=d.get("last_verified"),
            last_discovered=float(d.get("last_discovered", time.time())),
            error_message=d.get("error_message"),
            is_alias=bool(d.get("is_alias", False)),
            target_id=d.get("target_id"),
            metadata=dict(d.get("metadata", {})),
        )


@dataclass
class ExecutionLog:
    """Audit log record for outgoing model requests."""
    timestamp: float
    requested_provider: str
    requested_model: str
    dispatched_provider: str
    dispatched_model: str
    status: str                       # "success", "error", "fallback"
    tokens_in: int = 0
    tokens_out: int = 0
    duration_ms: float = 0.0
    error_message: Optional[str] = None
    fallback_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Explicit, validated alias table: provider -> { alias: exact_model_id }
EXPLICIT_MODEL_ALIASES: Dict[str, Dict[str, str]] = {
    "gemini": {
        "gemini-flash": "gemini-2.0-flash",
        "gemini-pro": "gemini-1.5-pro",
    },
    "openai": {
        "gpt-4o-mini": "gpt-4o-mini",
        "gpt-4o": "gpt-4o",
    },
    "groq": {
        "llama-instant": "llama-3.1-8b-instant",
    },
}


# ==================================================================== Provider Adapters

class BaseProviderAdapter:
    id: str = ""
    name: str = ""
    default_base_url: str = ""
    needs_key: bool = True
    key_url: str = ""

    def discover(self, base_url: str, key: str) -> List[ModelRecord]:
        raise NotImplementedError

    def call(self, model_id: str, base_url: str, key: str, prompt: str, schema: dict) -> Tuple[Any, int, int]:
        from . import ai
        url = (base_url or self.default_base_url).rstrip("/")
        return ai._call_as(self.id, model_id, url, key, prompt, schema)

    def validate(self, model_id: str, base_url: str, key: str) -> Tuple[bool, str]:
        """Validate model existence and capability with a minimal test probe."""
        probe_prompt = 'Answer with a JSON array that holds the single string "ok".'
        probe_schema = {"type": "ARRAY", "items": {"type": "STRING"}}
        try:
            out, sent, wrote = self.call(model_id, base_url, key, probe_prompt, probe_schema)
            if isinstance(out, list) and "ok" in out:
                return True, "Model verified successfully"
            return True, f"Model answered: {str(out)[:60]}"
        except Exception as e:
            return False, str(e)


class OllamaAdapter(BaseProviderAdapter):
    id = "ollama"
    name = "Ollama (on this machine)"
    default_base_url = "http://127.0.0.1:11434"
    needs_key = False
    key_url = "https://ollama.com"

    def discover(self, base_url: str, key: str) -> List[ModelRecord]:
        url = (base_url or self.default_base_url).rstrip("/")
        # Call GET /api/tags to list locally installed Ollama models
        r = httpx.get(f"{url}/api/tags", timeout=10)
        if r.status_code != 200:
            raise RuntimeError(f"Ollama returned HTTP {r.status_code}: {r.text[:200]}")
        data = r.json()
        records = []
        for m in data.get("models", []):
            tag = str(m.get("name") or m.get("model") or "").strip()
            if not tag:
                continue
            records.append(ModelRecord(
                provider=self.id,
                id=tag,  # Preserve exact tag (e.g. llama3.1:latest)
                name=f"Ollama: {tag}",
                capabilities={"chat": True, "json": True},
                status=STATUS_UNVERIFIED,
                metadata={"size": m.get("size"), "digest": m.get("digest")},
            ))
        return records


class GeminiAdapter(BaseProviderAdapter):
    id = "gemini"
    name = "Gemini"
    default_base_url = "https://generativelanguage.googleapis.com/v1beta/models"
    needs_key = True
    key_url = "https://aistudio.google.com/apikey"

    def discover(self, base_url: str, key: str) -> List[ModelRecord]:
        url = (base_url or self.default_base_url).rstrip("/")
        if not key:
            raise ValueError("Gemini API key is required for discovery")

        records = []
        page_token = None
        while True:
            params: Dict[str, Any] = {"pageSize": 100}
            if page_token:
                params["pageToken"] = page_token

            r = httpx.get(url, params=params, headers={"x-goog-api-key": key}, timeout=20)
            if r.status_code != 200:
                _raise_api_error("Gemini", r, "discovery", key)
            data = r.json()
            for m in data.get("models", []):
                methods = m.get("supportedGenerationMethods") or []
                if "generateContent" not in methods:
                    continue
                raw_name = m.get("name", "")
                exact_id = raw_name.split("/")[-1] if "/" in raw_name else raw_name
                if not exact_id:
                    continue
                display = m.get("displayName") or exact_id
                records.append(ModelRecord(
                    provider=self.id,
                    id=exact_id,
                    name=display,
                    capabilities={"chat": True, "json": True, "vision": "image" in str(methods).lower()},
                    status=STATUS_UNVERIFIED,
                    metadata={"description": m.get("description", ""), "supported_methods": methods},
                ))
            page_token = data.get("nextPageToken")
            if not page_token:
                break
        return records


class OpenAICompatibleAdapter(BaseProviderAdapter):
    def __init__(self, prov_id: str, name: str, default_url: str, key_url: str = "", needs_key: bool = True):
        self.id = prov_id
        self.name = name
        self.default_base_url = default_url
        self.needs_key = needs_key
        self.key_url = key_url

    def discover(self, base_url: str, key: str) -> List[ModelRecord]:
        url = (base_url or self.default_base_url).rstrip("/")
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        r = httpx.get(f"{url}/models", headers=headers, timeout=20)
        if r.status_code != 200:
            _raise_api_error(self.name, r, "discovery", key)
        data = r.json()
        models_data = data.get("data", []) if isinstance(data, dict) else []
        records = []
        for m in models_data:
            if not isinstance(m, dict):
                continue
            mid = str(m.get("id", "")).strip()
            if not mid:
                continue
            mid_low = mid.lower()
            # Filter non-chat models
            if any(k in mid_low for k in ("whisper", "embedding", "dall-e", "tts", "moderation", "audio")):
                continue
            records.append(ModelRecord(
                provider=self.id,
                id=mid,
                name=mid,
                capabilities={"chat": True, "json": True},
                status=STATUS_UNVERIFIED,
                metadata=m,
            ))
        return records


class AnthropicAdapter(BaseProviderAdapter):
    id = "anthropic"
    name = "Claude"
    default_base_url = "https://api.anthropic.com/v1"
    needs_key = True
    key_url = "https://console.anthropic.com"

    def discover(self, base_url: str, key: str) -> List[ModelRecord]:
        url = (base_url or self.default_base_url).rstrip("/")
        headers = {"x-api-key": key, "anthropic-version": "2023-06-01"}
        r = httpx.get(f"{url}/models", params={"limit": 100}, headers=headers, timeout=20)
        if r.status_code != 200:
            _raise_api_error("Claude", r, "discovery", key)
        data = r.json()
        records = []
        for m in data.get("data", []):
            mid = str(m.get("id", "")).strip()
            if not mid:
                continue
            records.append(ModelRecord(
                provider=self.id,
                id=mid,
                name=m.get("display_name") or mid,
                capabilities={"chat": True, "json": True},
                status=STATUS_UNVERIFIED,
                metadata=m,
            ))
        return records


# Adapter registry
ADAPTERS: Dict[str, BaseProviderAdapter] = {
    "gemini": GeminiAdapter(),
    "ollama": OllamaAdapter(),
    "openai": OpenAICompatibleAdapter("openai", "OpenAI", "https://api.openai.com/v1", "https://platform.openai.com/api-keys"),
    "xai": OpenAICompatibleAdapter("xai", "Grok", "https://api.x.ai/v1", "https://console.x.ai"),
    "groq": OpenAICompatibleAdapter("groq", "Groq", "https://api.groq.com/openai/v1", "https://console.groq.com/keys"),
    "openrouter": OpenAICompatibleAdapter("openrouter", "OpenRouter", "https://openrouter.ai/api/v1", "https://openrouter.ai/keys"),
    "anthropic": AnthropicAdapter(),
    "custom": OpenAICompatibleAdapter("custom", "Custom (OpenAI-compatible)", "", "", needs_key=False),
}


# ==================================================================== Model Registry

class ModelRegistry:
    """Centralized, cached registry of models for all providers."""

    def __init__(self, cache_file: Optional[Path] = None, ttl_sec: int = DEFAULT_CACHE_TTL_SEC):
        self._lock = threading.Lock()
        self._cache_file = cache_file or (config.DATA_DIR / "models_cache.json")
        self._ttl_sec = ttl_sec
        # provider -> { model_id: ModelRecord }
        self._records: Dict[str, Dict[str, ModelRecord]] = {}
        self._provider_status: Dict[str, Dict[str, Any]] = {}
        self._load_cache()

    def _load_cache(self):
        try:
            if self._cache_file.exists():
                data = json.loads(self._cache_file.read_text(encoding="utf-8"))
                for prov, p_data in data.get("providers", {}).items():
                    self._records[prov] = {}
                    for m_data in p_data.get("models", []):
                        rec = ModelRecord.from_dict(m_data)
                        self._records[prov][rec.id] = rec
                    self._provider_status[prov] = {
                        "last_discovered": p_data.get("last_discovered", 0.0),
                        "status": p_data.get("status", STATUS_UNVERIFIED),
                        "error": p_data.get("error"),
                    }
        except Exception as e:
            log.warning("Could not read model cache from %s: %s", self._cache_file, e)

    def _save_cache(self):
        try:
            out: Dict[str, Any] = {"updated_at": time.time(), "providers": {}}
            for prov, models in self._records.items():
                p_stat = self._provider_status.get(prov, {})
                out["providers"][prov] = {
                    "last_discovered": p_stat.get("last_discovered", 0.0),
                    "status": p_stat.get("status", STATUS_AVAILABLE),
                    "error": p_stat.get("error"),
                    "models": [m.to_dict() for m in models.values() if not m.is_alias],
                }
            tmp = self._cache_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(out, indent=2), encoding="utf-8")
            tmp.replace(self._cache_file)
        except Exception as e:
            log.warning("Could not save model cache to %s: %s", self._cache_file, e)

    def discover(self, provider_id: str, base_url: str = "", key: str = "", force: bool = False) -> List[ModelRecord]:
        """Dynamically discover models for a provider, using cache when fresh."""
        prov = provider_id.lower().strip()
        adapter = ADAPTERS.get(prov)
        if not adapter:
            raise ValueError(f"Unknown provider: {provider_id}")

        now = time.time()
        with self._lock:
            p_stat = self._provider_status.get(prov, {})
            last_disc = p_stat.get("last_discovered", 0.0)
            if not force and (now - last_disc < self._ttl_sec) and prov in self._records and self._records[prov]:
                return list(self._records[prov].values())

        # Execute dynamic discovery via adapter
        try:
            discovered = adapter.discover(base_url, key)
            with self._lock:
                if prov not in self._records:
                    self._records[prov] = {}
                # Update existing records or insert new ones (preserving previous verified timestamps)
                for rec in discovered:
                    existing = self._records[prov].get(rec.id)
                    if existing:
                        rec.last_verified = existing.last_verified
                        rec.status = existing.status if existing.status == STATUS_AVAILABLE else rec.status
                    self._records[prov][rec.id] = rec
                self._provider_status[prov] = {
                    "last_discovered": now,
                    "status": STATUS_AVAILABLE,
                    "error": None,
                }
                self._save_cache()
                return list(self._records[prov].values())
        except Exception as e:
            err_low = str(e).lower()
            if "401" in err_low or "403" in err_low or "unauthorized" in err_low or "api key" in err_low:
                status = STATUS_UNAUTHORIZED
            elif "timed out" in err_low or "timeout" in err_low or "connection" in err_low or "getaddrinfo" in err_low:
                status = STATUS_UNREACHABLE
            else:
                status = STATUS_UNAVAILABLE

            with self._lock:
                # Do NOT delete previously discovered records on API failure!
                self._provider_status[prov] = {
                    "last_discovered": now,
                    "status": status,
                    "error": str(e),
                }
                # Mark cached models with unreachable status if not previously verified
                if prov in self._records:
                    for rec in self._records[prov].values():
                        if rec.status != STATUS_AVAILABLE:
                            rec.status = status
                            rec.error_message = str(e)
                self._save_cache()
            raise

    def register_manual(self, provider_id: str, model_id: str, display_name: str = "") -> ModelRecord:
        """Register a manually specified model ID if not yet discovered."""
        prov = provider_id.lower().strip()
        mid = str(model_id).strip()
        rec = ModelRecord(
            provider=prov,
            id=mid,
            name=display_name or mid,
            status=STATUS_UNVERIFIED,
        )
        with self._lock:
            if prov not in self._records:
                self._records[prov] = {}
            if mid not in self._records[prov]:
                self._records[prov][mid] = rec
                self._save_cache()
            return self._records[prov][mid]

    def validate_model(self, provider_id: str, model_id: str, base_url: str = "", key: str = "") -> Tuple[bool, str]:
        """Validate an exact model ID with a live probe and update its record status."""
        prov = provider_id.lower().strip()
        adapter = ADAPTERS.get(prov)
        if not adapter:
            raise ValueError(f"Unknown provider: {provider_id}")

        resolved_id, is_alias = self.resolve_model_id(prov, model_id)
        ok, msg = adapter.validate(resolved_id, base_url, key)
        now = time.time()

        with self._lock:
            if prov not in self._records:
                self._records[prov] = {}
            rec = self._records[prov].get(resolved_id)
            if not rec:
                rec = ModelRecord(provider=prov, id=resolved_id, name=resolved_id)
                self._records[prov][resolved_id] = rec

            if ok:
                rec.status = STATUS_AVAILABLE
                rec.last_verified = now
                rec.error_message = None
            else:
                err_low = msg.lower()
                if "404" in err_low or "not found" in err_low:
                    rec.status = STATUS_UNAVAILABLE
                elif "401" in err_low or "403" in err_low:
                    rec.status = STATUS_UNAUTHORIZED
                elif "timed out" in err_low or "connect" in err_low:
                    rec.status = STATUS_UNREACHABLE
                else:
                    rec.status = STATUS_UNSUPPORTED
                rec.error_message = msg
            self._save_cache()
        return ok, msg

    def resolve_model_id(self, provider_id: str, requested_id: str) -> Tuple[str, bool]:
        """Resolve model ID strictly: exact ID or via explicit alias table. Never fuzzy matches."""
        prov = provider_id.lower().strip()
        req = str(requested_id or "").strip()
        aliases = EXPLICIT_MODEL_ALIASES.get(prov, {})
        if req in aliases:
            return aliases[req], True
        return req, False

    def get_model(self, provider_id: str, model_id: str) -> Optional[ModelRecord]:
        prov = provider_id.lower().strip()
        resolved_id, _ = self.resolve_model_id(prov, model_id)
        with self._lock:
            return self._records.get(prov, {}).get(resolved_id)

    def list_models(self, provider_id: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._lock:
            out = []
            provs = [provider_id.lower()] if provider_id else list(ADAPTERS.keys())
            for p in provs:
                records = list(self._records.get(p, {}).values())
                # Add explicit aliases
                for alias, target in EXPLICIT_MODEL_ALIASES.get(p, {}).items():
                    out.append({
                        "provider": p,
                        "id": alias,
                        "name": f"{alias} (alias -> {target})",
                        "status": STATUS_AVAILABLE if any(r.id == target and r.status == STATUS_AVAILABLE for r in records) else STATUS_UNVERIFIED,
                        "is_alias": True,
                        "target_id": target,
                        "capabilities": {"chat": True, "json": True},
                    })
                for r in records:
                    out.append(r.to_dict())
            return out


# ==================================================================== Model Router

class ModelRouter:
    """Deterministic, provider-agnostic model router with execution verification logging."""

    def __init__(self, registry: ModelRegistry, max_log_entries: int = 100):
        self._registry = registry
        self._logs: List[ExecutionLog] = []
        self._log_lock = threading.Lock()
        self._max_log_entries = max_log_entries

    def _record_log(self, entry: ExecutionLog):
        with self._log_lock:
            self._logs.insert(0, entry)
            if len(self._logs) > self._max_log_entries:
                self._logs.pop()

    def get_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._log_lock:
            return [l.to_dict() for l in self._logs[:limit]]

    def dispatch(
        self,
        provider_id: str,
        model_id: str,
        base_url: str,
        key: str,
        prompt: str,
        schema: dict,
        allow_fallback: bool = False,
        fallback_config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, str, int, int]:
        """Dispatch a prompt to the exact validated model ID.
        If strict mode (allow_fallback=False), fails immediately on error without substitution.
        If opt-in fallback is enabled, transparently switches to the configured fallback model on 429/outage.
        """
        prov = provider_id.lower().strip()
        adapter = ADAPTERS.get(prov)
        if not adapter:
            raise ValueError(f"Unknown provider: {provider_id}")

        exact_model_id, is_alias = self._registry.resolve_model_id(prov, model_id)
        start_t = time.time()
        last_err: Optional[Exception] = None

        # 1. Primary dispatch: exact ID routed directly to adapter
        try:
            out, sent, wrote = adapter.call(exact_model_id, base_url, key, prompt, schema)
            duration_ms = (time.time() - start_t) * 1000.0
            self._record_log(ExecutionLog(
                timestamp=start_t,
                requested_provider=prov,
                requested_model=model_id,
                dispatched_provider=prov,
                dispatched_model=exact_model_id,
                status="success",
                tokens_in=sent,
                tokens_out=wrote,
                duration_ms=duration_ms,
            ))
            return out, exact_model_id, sent, wrote
        except Exception as e:
            last_err = e
            duration_ms = (time.time() - start_t) * 1000.0

        # Check if fallback is permitted and applicable
        is_rate_limit_or_outage = _is_rate_limit_or_outage(last_err)
        if not allow_fallback or not fallback_config or not is_rate_limit_or_outage:
            self._record_log(ExecutionLog(
                timestamp=start_t,
                requested_provider=prov,
                requested_model=model_id,
                dispatched_provider=prov,
                dispatched_model=exact_model_id,
                status="error",
                duration_ms=duration_ms,
                error_message=str(last_err),
            ))
            raise last_err

        # 2. Opt-in fallback dispatch
        fb_prov = str(fallback_config.get("provider", "")).lower().strip()
        fb_adapter = ADAPTERS.get(fb_prov)
        fb_model = str((fallback_config.get("models") or [""])[0]).strip()
        fb_url = str(fallback_config.get("base_url") or "")
        fb_key = str(fallback_config.get("key") or "")
        if not fb_adapter or not fb_model:
            self._record_log(ExecutionLog(
                timestamp=start_t,
                requested_provider=prov,
                requested_model=model_id,
                dispatched_provider=prov,
                dispatched_model=exact_model_id,
                status="error",
                duration_ms=duration_ms,
                error_message=f"Primary failed ({last_err}), fallback invalid",
            ))
            raise last_err

        fb_exact_model, _ = self._registry.resolve_model_id(fb_prov, fb_model)
        try:
            fb_start = time.time()
            out, sent, wrote = fb_adapter.call(fb_exact_model, fb_url, fb_key, prompt, schema)
            fb_duration_ms = (time.time() - fb_start) * 1000.0
            self._record_log(ExecutionLog(
                timestamp=start_t,
                requested_provider=prov,
                requested_model=model_id,
                dispatched_provider=fb_prov,
                dispatched_model=fb_exact_model,
                status="fallback",
                tokens_in=sent,
                tokens_out=wrote,
                duration_ms=fb_duration_ms,
                fallback_reason=f"{prov} error: {last_err}",
            ))
            return out, f"{fb_exact_model} ({fb_prov})", sent, wrote
        except Exception as fb_err:
            self._record_log(ExecutionLog(
                timestamp=start_t,
                requested_provider=prov,
                requested_model=model_id,
                dispatched_provider=fb_prov,
                dispatched_model=fb_exact_model,
                status="error",
                duration_ms=duration_ms,
                error_message=f"Primary failed ({last_err}); Fallback failed ({fb_err})",
            ))
            raise RuntimeError(f"Both primary ({exact_model_id}) and fallback ({fb_exact_model}) failed: {fb_err}") from fb_err


# ==================================================================== Global Registry and Helpers

global_registry = ModelRegistry()
global_router = ModelRouter(global_registry)


def _is_rate_limit_or_outage(err: Exception) -> bool:
    s = str(err).lower()
    return any(k in s for k in ("429", "quota", "rate limit", "rate_limit", "exhausted", "credit", "503", "timed out", "timeout"))


def _raise_api_error(prov_name: str, r: Optional[httpx.Response], model_id: str, key: str = ""):
    if r is None:
        raise RuntimeError(f"{prov_name} request timed out for model {model_id}")
    sanitized_text = r.text[:300].replace(key or "\0", "<key>") if key else r.text[:300]
    raise RuntimeError(f"{prov_name} HTTP {r.status_code} on model {model_id}: {sanitized_text}")


def _json_schema(s: Any) -> Any:
    if isinstance(s, dict):
        return {k: (v.lower() if k == "type" and isinstance(v, str) else _json_schema(v)) for k, v in s.items()}
    if isinstance(s, list):
        return [_json_schema(x) for x in s]
    return s


def _wrapped_prompt(prompt: str, schema: dict) -> str:
    return (prompt + "\n\nAnswer with JSON only, no other text and no code fences: one object with a single key "
            "\"items\", whose value follows this JSON Schema:\n" + json.dumps(_json_schema(schema)))


def _parse_json(text: Optional[str]) -> Any:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        out = json.loads(text)
    except ValueError:
        m = re.search(r"[\[{].*[\]}]", text, re.S)
        if not m:
            raise RuntimeError("The AI did not answer with JSON.")
        out = json.loads(m.group(0))
    if isinstance(out, dict) and "items" in out:
        out = out["items"]
    return out
