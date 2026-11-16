"""Tests for the centralized, self-validating model registry and deterministic router.
Covers dynamic discovery, pagination, exact-ID routing, regression protections,
explicit alias resolution, cache retention on failure, strict vs opt-in fallback,
and execution verification logging.
"""
import json
import os
import sys
import tempfile
import time

os.environ["AUTOCLIPPER_DATA"] = tempfile.mkdtemp(prefix="pc-registry-test-")
os.environ["DATABASE_URL"] = ""
sys.path.insert(0, ".")

from studio import config
from studio.registry import (
    ADAPTERS,
    EXPLICIT_MODEL_ALIASES,
    GeminiAdapter,
    ModelRecord,
    ModelRegistry,
    ModelRouter,
    OllamaAdapter,
    OpenAICompatibleAdapter,
    STATUS_AVAILABLE,
    STATUS_UNAVAILABLE,
    STATUS_UNAUTHORIZED,
    STATUS_UNREACHABLE,
    STATUS_UNVERIFIED,
)

fails = []


def check(name: str, ok: bool, extra: str = ""):
    print(("PASS " if ok else "FAIL ") + name + (f"  {extra}" if extra and not ok else ""))
    if not ok:
        fails.append(name)


class MockResponse:
    def __init__(self, status_code: int, data: dict, text: str = ""):
        self.status_code = status_code
        self._data = data
        self.text = text or json.dumps(data)

    def json(self):
        return self._data


# ==================================================================== 1. Dynamic Discovery Tests

# Gemini Discovery with pagination and generateContent filtering
def test_gemini_discovery():
    calls = []

    def mock_gemini_get(url, params=None, headers=None, timeout=None):
        calls.append((url, params or {}, headers or {}))
        if params.get("pageToken") == "page2":
            return MockResponse(200, {
                "models": [
                    {"name": "models/gemini-2.0-flash", "displayName": "Gemini 2.0 Flash", "supportedGenerationMethods": ["generateContent"]},
                    {"name": "models/text-embedding-004", "displayName": "Embedding", "supportedGenerationMethods": ["embedContent"]},
                ],
            })
        return MockResponse(200, {
            "models": [
                {"name": "models/gemini-1.5-pro", "displayName": "Gemini 1.5 Pro", "supportedGenerationMethods": ["generateContent"]},
                {"name": "models/gemini-1.5-flash", "displayName": "Gemini 1.5 Flash", "supportedGenerationMethods": ["generateContent"]},
            ],
            "nextPageToken": "page2",
        })

    from studio import registry
    orig_get = registry.httpx.get
    registry.httpx.get = mock_gemini_get
    try:
        adapter = GeminiAdapter()
        records = adapter.discover("https://test.gemini/v1beta/models", "key-123")
        check("gemini: pagination called multiple pages", len(calls) == 2 and calls[1][1].get("pageToken") == "page2")
        ids = [r.id for r in records]
        check("gemini: exact model IDs stripped of models/ prefix", ids == ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-2.0-flash"], str(ids))
        check("gemini: non-generateContent embedding models filtered out", "text-embedding-004" not in ids)
        check("gemini: headers pass x-goog-api-key", calls[0][2].get("x-goog-api-key") == "key-123")
    finally:
        registry.httpx.get = orig_get


test_gemini_discovery()


# Ollama Discovery using GET /api/tags and exact tag preservation
def test_ollama_discovery():
    calls = []

    def mock_ollama_get(url, headers=None, timeout=None, **_k):
        calls.append(url)
        return MockResponse(200, {
            "models": [
                {"name": "llama3.1:latest", "model": "llama3.1:latest", "size": 4700000000},
                {"name": "deepseek-r1:8b", "model": "deepseek-r1:8b", "size": 4900000000},
                {"name": "mistral:7b-instruct-q4_K_M", "model": "mistral:7b-instruct-q4_K_M", "size": 4100000000},
            ],
        })

    from studio import registry
    orig_get = registry.httpx.get
    registry.httpx.get = mock_ollama_get
    try:
        adapter = OllamaAdapter()
        records = adapter.discover("http://127.0.0.1:11434", "")
        check("ollama: calls /api/tags endpoint", calls == ["http://127.0.0.1:11434/api/tags"])
        tags = [r.id for r in records]
        check("ollama: preserves exact model tags and variants", tags == ["llama3.1:latest", "deepseek-r1:8b", "mistral:7b-instruct-q4_K_M"], str(tags))
    finally:
        registry.httpx.get = orig_get


test_ollama_discovery()


# OpenAI-compatible discovery with non-chat filtering
def test_openai_compatible_discovery():
    def mock_openai_get(url, headers=None, timeout=None, **_k):
        return MockResponse(200, {
            "data": [
                {"id": "gpt-4o-mini"},
                {"id": "text-embedding-3-small"},
                {"id": "whisper-1"},
                {"id": "dall-e-3"},
                {"id": "gpt-4o"},
            ],
        })

    from studio import registry
    orig_get = registry.httpx.get
    registry.httpx.get = mock_openai_get
    try:
        adapter = OpenAICompatibleAdapter("openai", "OpenAI", "https://api.openai.com/v1")
        records = adapter.discover("https://api.openai.com/v1", "key-456")
        ids = [r.id for r in records]
        check("openai: filters out whisper, embedding, and dall-e models", ids == ["gpt-4o-mini", "gpt-4o"], str(ids))
    finally:
        registry.httpx.get = orig_get


test_openai_compatible_discovery()


# ==================================================================== 2. Deterministic Routing & Regression Protection

def test_deterministic_routing():
    reg = ModelRegistry()
    router = ModelRouter(reg)

    dispatched = []

    def mock_call_as(prov, mdl, url, k, prompt, schema):
        dispatched.append((prov, mdl))
        return ({"items": [{"score": 90}]}, 100, 20)

    from studio import ai
    orig_call_as = ai._call_as
    ai._call_as = mock_call_as
    try:
        # Regression 1: Selecting an Ollama model CANNOT route to a different Ollama model
        dispatched.clear()
        out, exact_id, sent, wrote = router.dispatch("ollama", "deepseek-r1:8b", "http://127.0.0.1:11434", "", "prompt", {})
        check("regression: ollama routes to exact requested tag", dispatched == [("ollama", "deepseek-r1:8b")])

        # Regression 2: Selecting a Gemini model CANNOT substitute an older model
        dispatched.clear()
        out, exact_id, sent, wrote = router.dispatch("gemini", "gemini-2.0-flash", "https://gemini", "key", "prompt", {})
        check("regression: gemini routes to exact requested model without legacy downgrade", dispatched == [("gemini", "gemini-2.0-flash")])

        # Explicit alias resolution
        dispatched.clear()
        out, exact_id, sent, wrote = router.dispatch("gemini", "gemini-flash", "https://gemini", "key", "prompt", {})
        check("alias: gemini-flash resolves strictly via explicit alias table to gemini-2.0-flash", dispatched == [("gemini", "gemini-2.0-flash")])

        # Unknown alias/fuzzy model is NOT guessed
        resolved, is_alias = reg.resolve_model_id("gemini", "gemini-something-random")
        check("strictness: random string is never fuzzy matched to existing models", resolved == "gemini-something-random" and not is_alias)
    finally:
        ai._call_as = orig_call_as


test_deterministic_routing()


# ==================================================================== 3. Cache TTL and Failure Retention

def test_cache_and_failure_retention():
    cache_path = config.DATA_DIR / "test_models_cache.json"
    if cache_path.exists():
        cache_path.unlink()

    reg = ModelRegistry(cache_file=cache_path, ttl_sec=2)

    # Initial successful discovery
    def mock_good_discovery(base_url, key):
        return [
            ModelRecord(provider="groq", id="llama-3.1-8b-instant", name="Llama 3.1 8B"),
            ModelRecord(provider="groq", id="openai/gpt-oss-120b", name="GPT OSS 120B"),
        ]

    from studio.registry import ADAPTERS
    orig_groq_disc = ADAPTERS["groq"].discover
    ADAPTERS["groq"].discover = mock_good_discovery
    try:
        records1 = reg.discover("groq", force=True)
        check("cache: initial discovery succeeds", len(records1) == 2)

        # Subsequent call before TTL expiry uses cache
        ADAPTERS["groq"].discover = lambda b, k: []
        records2 = reg.discover("groq", force=False)
        check("cache: fresh cache returned without hitting adapter", len(records2) == 2 and records2[0].id == "llama-3.1-8b-instant")

        # Now simulate remote provider network outage during forced refresh
        def mock_failing_discovery(base_url, key):
            raise RuntimeError("Connection timed out (getaddrinfo failed)")

        ADAPTERS["groq"].discover = mock_failing_discovery
        failed = False
        try:
            reg.discover("groq", force=True)
        except Exception:
            failed = True
        check("cache failure: exception raised on network outage", failed)

        # CRITICAL requirement: Previously discovered records MUST NOT be deleted on network failure
        cached_models = reg.list_models("groq")
        cached_ids = [m["id"] for m in cached_models if not m.get("is_alias")]
        check("cache retention: previous records retained after provider failure", cached_ids == ["llama-3.1-8b-instant", "openai/gpt-oss-120b"], str(cached_ids))
    finally:
        ADAPTERS["groq"].discover = orig_groq_disc
        if cache_path.exists():
            cache_path.unlink()


test_cache_and_failure_retention()


# ==================================================================== 4. Strict Mode vs Opt-in Fallback

def test_fallback_policies():
    reg = ModelRegistry()
    router = ModelRouter(reg)

    calls = []

    def mock_call_as(prov, mdl, url, k, prompt, schema):
        calls.append((prov, mdl))
        if prov == "gemini":
            raise RuntimeError("HTTP 429 Resource has been exhausted (quota exceeded)")
        return ({"items": [{"result": "fallback_ok"}]}, 80, 15)

    from studio import ai
    orig_call_as = ai._call_as
    ai._call_as = mock_call_as
    try:
        # Strict mode: allow_fallback=False (Default) -> Must NOT substitute model, must raise error
        calls.clear()
        strict_failed = False
        try:
            router.dispatch("gemini", "gemini-2.0-flash", "https://gemini", "k", "test", {}, allow_fallback=False)
        except Exception as e:
            strict_failed = True
            check("strict mode: raises primary error without substitution", "429" in str(e))
        check("strict mode: no secondary calls made", calls == [("gemini", "gemini-2.0-flash")])

        # Opt-in fallback mode: allow_fallback=True -> Transparently switches to fallback and logs reason
        calls.clear()
        fb_config = {"provider": "groq", "models": ["llama-3.1-8b-instant"], "base_url": "", "key": "groq-key"}
        out, dispatched_model, sent, wrote = router.dispatch(
            "gemini", "gemini-2.0-flash", "https://gemini", "k", "test", {},
            allow_fallback=True, fallback_config=fb_config
        )
        check("opt-in fallback: dispatched to configured fallback model", "llama-3.1-8b-instant" in dispatched_model and calls == [("gemini", "gemini-2.0-flash"), ("groq", "llama-3.1-8b-instant")])

        # Check execution log verification
        logs = router.get_logs(limit=5)
        last_log = logs[0]
        check("audit log: records fallback event with exact models", last_log["status"] == "fallback" and last_log["requested_model"] == "gemini-2.0-flash" and last_log["dispatched_model"] == "llama-3.1-8b-instant")
        check("audit log: records fallback reason", "429" in (last_log.get("fallback_reason") or ""))
    finally:
        ai._call_as = orig_call_as


test_fallback_policies()


# ==================================================================== Final Summary
print("\n" + ("ALL REGISTRY AND ROUTER TESTS PASSED" if not fails else f"{len(fails)} TESTS FAILED"))
import shutil
shutil.rmtree(os.environ["AUTOCLIPPER_DATA"], ignore_errors=True)
sys.exit(1 if fails else 0)
