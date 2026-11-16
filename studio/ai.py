"""Talking to the AI service. You bring your own key, and choose the service and model in the dashboard
(Settings, AI) or in .env. What is saved from the dashboard is kept in ai.json in the data folder and wins.

    AUTOCLIPPER_AI_PROVIDER   gemini | openai | xai | groq | openrouter | anthropic | ollama | custom
    AUTOCLIPPER_AI_MODELS     one or more model ids, comma separated, tried in this order
    AUTOCLIPPER_AI_KEY        your key (or AUTOCLIPPER_AI_KEY_FILE, a file holding it)
    AUTOCLIPPER_AI_BASE_URL   only for "custom" (any service that speaks the OpenAI chat format), or to override

A fallback service can be set up in the same dashboard. When the primary service hits its rate limit or
runs out of credit, Autoclipper switches to the fallback automatically and keeps going.

Every service is asked for the same thing: JSON that follows a small schema. Gemini takes the schema directly.
The others are told the schema in the prompt and asked for a JSON object, which works with any chat model."""
import json
import os
import re
import threading

import httpx

from . import config

# name -> (label shown on the pages, address, how it is spoken to, where to get a key, an example model id)
PROVIDERS = {
    "gemini": ("Gemini", "https://generativelanguage.googleapis.com/v1beta/models", "gemini", "https://aistudio.google.com/apikey", "gemini-2.5-flash"),
    "openai": ("OpenAI", "https://api.openai.com/v1", "openai", "https://platform.openai.com/api-keys", "gpt-4o-mini"),
    "xai": ("Grok", "https://api.x.ai/v1", "openai", "https://console.x.ai", "grok-4"),
    "groq": ("Groq", "https://api.groq.com/openai/v1", "openai", "https://console.groq.com/keys", "llama-3.3-70b-versatile"),
    "openrouter": ("OpenRouter", "https://openrouter.ai/api/v1", "openai", "https://openrouter.ai/keys", "google/gemini-2.5-flash"),
    "anthropic": ("Claude", "https://api.anthropic.com/v1", "anthropic", "https://console.anthropic.com", ""),
    "ollama": ("Ollama (on this machine)", "http://127.0.0.1:11434/v1", "openai", "https://ollama.com", "llama3.1"),
    "custom": ("Custom (OpenAI-compatible)", "", "openai", "", ""),
}
ALIASES = {"google": "gemini", "grok": "xai", "claude": "anthropic", "local": "ollama"}
NO_KEY = {"ollama", "custom"}       # a model running on your own machine needs no key

# ---------------------------------------------------------------- the settings in use
_lock = threading.Lock()
_saved = {"at": None, "data": {}}


def _file():
    return config.DATA_DIR / "ai.json"


def _from_dashboard() -> dict:
    """What was saved from the dashboard, read again only when the file changes."""
    f = _file()
    try:
        at = f.stat().st_mtime_ns
    except OSError:
        at = 0
    if _saved["at"] != at:
        try:
            data = json.loads(f.read_text(encoding="utf-8")) if at else {}
        except (OSError, ValueError):
            data = {}
        _saved.update(at=at, data=data if isinstance(data, dict) else {})
    return _saved["data"]


def current() -> dict:
    """Provider, models, address and key: what the dashboard saved, or else what .env says."""
    d = _from_dashboard()
    p = str(d.get("provider") or config.AI_PROVIDER or "").strip().lower()
    return {"provider": ALIASES.get(p, p),
            "models": list(d["models"]) if d.get("provider") else list(config.AI_MODELS),
            "base_url": (str(d.get("base_url") or "") if d.get("provider") else config.AI_BASE_URL).rstrip("/"),
            "key": str(d["key"]) if d.get("provider") and "key" in d else config.ai_key()}


def provider() -> str:
    return current()["provider"]


def models() -> list:
    return current()["models"]


def key() -> str:
    return current()["key"]


def label() -> str:
    return PROVIDERS.get(provider(), ("",))[0]


def base_url() -> str:
    c = current()
    return c["base_url"] or PROVIDERS.get(c["provider"], ("", ""))[1]


def ready() -> bool:
    c = current()
    return bool(c["provider"] in PROVIDERS and c["models"] and base_url() and (c["key"] or c["provider"] in NO_KEY))


def public() -> dict:
    """What the settings page may see. The key itself never leaves the server: only whether there is one, and
    its last four characters so the owner can tell which key it is."""
    c = current()
    fb = fallback()
    fb_info = {}
    if fb:
        fb_info = {"fallback_provider": fb["provider"], "fallback_models": fb["models"],
                   "fallback_base_url": fb.get("base_url", ""),
                   "fallback_has_key": bool(fb.get("key")),
                   "fallback_key_hint": fb["key"][-4:] if len(fb.get("key", "")) >= 12 else "",
                   "fallback_label": PROVIDERS.get(fb["provider"], ("",))[0],
                   "fallback_ready": bool(fb["provider"] in PROVIDERS and fb["models"]
                                         and (fb.get("base_url") or PROVIDERS.get(fb["provider"], ("", ""))[1])
                                         and (fb.get("key") or fb["provider"] in NO_KEY))}
    return {"provider": c["provider"], "models": c["models"], "base_url": c["base_url"], "has_key": bool(c["key"]),
            "key_hint": c["key"][-4:] if len(c["key"]) >= 12 else "", "ready": ready(), "label": label(),
            "saved_here": bool(_from_dashboard().get("provider")),
            "providers": [{"id": k, "label": v[0], "address": v[1], "needs_key": k not in NO_KEY, "key_url": v[3], "example": v[4]}
                          for k, v in PROVIDERS.items()],
            **fb_info}


def save(provider_id: str, model_ids, address: str = "", new_key=None) -> dict:
    """Save the choice made in the dashboard. `new_key` None keeps the key that is already there (the page never
    has it to send back); an empty string removes it."""
    p = ALIASES.get(str(provider_id or "").strip().lower(), str(provider_id or "").strip().lower())
    if p not in PROVIDERS:
        raise ValueError("Choose one of the AI services in the list.")
    if isinstance(model_ids, str):
        model_ids = model_ids.split(",")
    ms = [str(m).strip()[:120] for m in (model_ids or []) if str(m).strip()][:5]
    if not ms:
        raise ValueError("Enter the model to use.")
    if any(not re.fullmatch(r"[\w.\-:/@]+", m) for m in ms):
        raise ValueError("A model id has letters, digits and . - : / only.")
    address = str(address or "").strip().rstrip("/")[:300]
    if address and not re.fullmatch(r"https?://[^\s]+", address):
        raise ValueError("The address has to start with http:// or https://")
    if p == "custom" and not address:
        raise ValueError("A custom service needs its address, for example http://127.0.0.1:1234/v1")
    old = current()
    k = old["key"] if new_key is None and old["provider"] == p else str(new_key or "").strip()[:400]
    if not k and p not in NO_KEY:
        raise ValueError("Paste the key for this service.")
    with _lock:
        f = _file()
        # keep the fallback section when saving primary settings
        existing = _from_dashboard()
        data = {"provider": p, "models": ms, "base_url": address, "key": k}
        if existing.get("fallback"):
            data["fallback"] = existing["fallback"]
        tmp = f.with_suffix(".tmp")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        try:
            os.chmod(tmp, 0o600)        # the key is only for whoever runs the server
        except OSError:
            pass
        tmp.replace(f)
        _saved["at"] = None
    return public()


# ---------------------------------------------------------------- fallback: a second service when the first hits its limit

def fallback() -> dict:
    """The fallback provider settings, or an empty dict if none is set."""
    d = _from_dashboard()
    fb = d.get("fallback")
    if fb and isinstance(fb, dict):
        p = ALIASES.get(str(fb.get("provider", "")).lower(), str(fb.get("provider", "")).lower())
        if p in PROVIDERS:
            return {"provider": p,
                    "models": list(fb.get("models") or []),
                    "base_url": str(fb.get("base_url") or "").rstrip("/"),
                    "key": str(fb.get("key") or "")}
    # Check config fallback
    p = ALIASES.get(str(config.AI_FALLBACK_PROVIDER or "").lower(), str(config.AI_FALLBACK_PROVIDER or "").lower())
    if p in PROVIDERS:
        models_list = list(config.AI_FALLBACK_MODELS or ([PROVIDERS[p][4]] if PROVIDERS[p][4] else []))
        return {"provider": p,
                "models": models_list,
                "base_url": config.AI_FALLBACK_BASE_URL.rstrip("/"),
                "key": config.ai_fallback_key()}
    return {}


def fallback_ready() -> bool:
    fb = fallback()
    if not fb:
        return False
    return bool(fb["provider"] in PROVIDERS and fb["models"]
                and (fb.get("base_url") or PROVIDERS.get(fb["provider"], ("", ""))[1])
                and (fb.get("key") or fb["provider"] in NO_KEY))


def fallback_label() -> str:
    fb = fallback()
    return PROVIDERS.get(fb.get("provider", ""), ("",))[0]


def fallback_models() -> list:
    fb = fallback()
    return fb.get("models", [])


def save_fallback(provider_id: str, model_ids, address: str = "", new_key=None) -> dict:
    """Save the fallback service. Same rules as save() but stored under a 'fallback' key."""
    p = ALIASES.get(str(provider_id or "").strip().lower(), str(provider_id or "").strip().lower())
    if p not in PROVIDERS:
        raise ValueError("Choose one of the AI services in the list.")
    if isinstance(model_ids, str):
        model_ids = model_ids.split(",")
    ms = [str(m).strip()[:120] for m in (model_ids or []) if str(m).strip()][:5]
    if not ms:
        raise ValueError("Enter the model to use.")
    if any(not re.fullmatch(r"[\w.\-:/@]+", m) for m in ms):
        raise ValueError("A model id has letters, digits and . - : / only.")
    address = str(address or "").strip().rstrip("/")[:300]
    if address and not re.fullmatch(r"https?://[^\s]+", address):
        raise ValueError("The address has to start with http:// or https://")
    if p == "custom" and not address:
        raise ValueError("A custom service needs its address, for example http://127.0.0.1:1234/v1")
    old_fb = fallback()
    k = old_fb.get("key", "") if new_key is None and old_fb.get("provider") == p else str(new_key or "").strip()[:400]
    if not k and p not in NO_KEY:
        raise ValueError("Paste the key for this service.")
    with _lock:
        f = _file()
        existing = _from_dashboard()
        existing["fallback"] = {"provider": p, "models": ms, "base_url": address, "key": k}
        tmp = f.with_suffix(".tmp")
        tmp.write_text(json.dumps(existing), encoding="utf-8")
        try:
            os.chmod(tmp, 0o600)
        except OSError:
            pass
        tmp.replace(f)
        _saved["at"] = None
    return public()


def remove_fallback() -> dict:
    """Remove the fallback service."""
    with _lock:
        f = _file()
        existing = _from_dashboard()
        existing.pop("fallback", None)
        tmp = f.with_suffix(".tmp")
        tmp.write_text(json.dumps(existing), encoding="utf-8")
        try:
            os.chmod(tmp, 0o600)
        except OSError:
            pass
        tmp.replace(f)
        _saved["at"] = None
    return public()


def _call_as(prov: str, mdl: str, url: str, k: str, prompt: str, schema: dict):
    """Make one AI call using the given provider settings, not the current ones."""
    how = PROVIDERS[prov][2]
    if how == "gemini":
        body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.4, "responseMimeType": "application/json", "responseSchema": schema}}
        r = httpx.post(f"{url}/{mdl}:generateContent", json=body, headers={"x-goog-api-key": k}, timeout=120)
        if r.status_code != 200:
            _fail(r, mdl, k)
        data = r.json()
        text = "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
        u = data.get("usageMetadata", {})
        sent = int(u.get("promptTokenCount") or 0)
        return json.loads(text), sent, max(0, int(u.get("totalTokenCount") or 0) - sent)
    elif how == "anthropic":
        body = {"model": mdl, "max_tokens": 8192, "messages": [{"role": "user", "content": _wrapped(prompt, schema)}]}
        r = httpx.post(f"{url}/messages", json=body, timeout=180,
                       headers={"x-api-key": k, "anthropic-version": "2023-06-01"})
        if r.status_code != 200:
            _fail(r, mdl, k)
        data = r.json()
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        u = data.get("usage") or {}
        return _parse(text), int(u.get("input_tokens") or 0), int(u.get("output_tokens") or 0)
    else:
        headers = {"Authorization": f"Bearer {k}"} if k else {}
        body = {"model": mdl, "messages": [{"role": "user", "content": _wrapped(prompt, schema)}]}
        extras = [{"temperature": 0.4, "response_format": {"type": "json_object"}}, {"response_format": {"type": "json_object"}}, {}]
        r = None
        for extra in extras:
            r = httpx.post(f"{url}/chat/completions", json={**body, **extra}, headers=headers, timeout=180)
            if r.status_code not in (400, 422):
                break
        if r.status_code != 200:
            _fail(r, mdl, k)
        data = r.json()
        u = data.get("usage") or {}
        return _parse(data["choices"][0]["message"].get("content")), int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0)


def call_fallback(model: str, prompt: str, schema: dict):
    """Ask the fallback model. Returns (parsed JSON, tokens sent, tokens written back)."""
    fb = fallback()
    if not fallback_ready():
        raise AIError("Fallback AI service is not configured or ready.")
    url = fb.get("base_url") or PROVIDERS.get(fb["provider"], ("", ""))[1]
    return _call_as(fb["provider"], model, url, fb.get("key", ""), prompt, schema)


def test_fallback() -> dict:
    """One tiny request with the fallback settings, to verify fallback works."""
    if not fallback_ready():
        raise ValueError("Save a fallback service, model and key first.")
    fb = fallback()
    m = (fb.get("models") or [""])[0]
    out, sent, wrote = call_fallback(m, 'Answer with a JSON array that holds the single string "ok".', {"type": "ARRAY", "items": {"type": "STRING"}})
    return {"ok": True, "provider": fb["provider"], "model": m, "tokens": sent + wrote, "answer": str(out)[:80]}


def list_models(provider_id=None, address=None, with_key=None) -> list:
    """The model ids a service offers, so the settings page can show a list to pick from."""
    c = current()
    p = ALIASES.get(str(provider_id or c["provider"]).lower(), str(provider_id or c["provider"]).lower())
    if p not in PROVIDERS:
        raise ValueError("Choose one of the AI services in the list.")
    k = str(with_key or "").strip() or (c["key"] if c["provider"] == p else "")
    base = str(address or "").strip().rstrip("/") or (c["base_url"] if c["provider"] == p else "") or PROVIDERS[p][1]
    if not base:
        raise ValueError("Enter the service's address first.")
    how = PROVIDERS[p][2]
    if how == "gemini":
        r = httpx.get(base, params={"pageSize": 200}, headers={"x-goog-api-key": k}, timeout=20)
    elif how == "anthropic":
        r = httpx.get(f"{base}/models", params={"limit": 200}, headers={"x-api-key": k, "anthropic-version": "2023-06-01"}, timeout=20)
    else:
        r = httpx.get(f"{base}/models", headers={"Authorization": f"Bearer {k}"} if k else {}, timeout=20)
    if r.status_code != 200:
        raise AIError(f"HTTP {r.status_code} {r.text[:200].replace(k or chr(0), '<key>')}")
    data = r.json()
    if how == "gemini":
        found = [m.get("name", "").split("/")[-1] for m in data.get("models", [])
                 if "generateContent" in (m.get("supportedGenerationMethods") or [])]
    else:
        found = [m.get("id", "") for m in data.get("data", []) if isinstance(m, dict) and "whisper" not in m.get("id", "").lower()]
    models_list = sorted({m for m in found if m})
    preferred = PROVIDERS[p][4]
    if preferred in models_list:
        models_list.remove(preferred)
        return [preferred] + models_list[:399]
    return models_list[:400]


def test() -> dict:
    """One tiny request with the settings in use, to prove the key, the address and the model all work."""
    ms = models()
    if not ready():
        raise ValueError("Save a service, a model and a key first.")
    out, sent, wrote = call(ms[0], 'Answer with a JSON array that holds the single string "ok".', {"type": "ARRAY", "items": {"type": "STRING"}})
    clear_problem()
    return {"ok": True, "model": ms[0], "tokens": sent + wrote, "answer": str(out)[:80]}


class AIError(RuntimeError):
    pass


problem = {"text": "", "at": 0.0, "is_fallback": False, "fallback_provider": ""}       # why the AI last failed, for the dashboard; cleared when it answers again


def explain(e, prov=None, mdl=None) -> str:
    """What went wrong with an AI call, in words the owner can act on."""
    text, low = str(e), str(e).lower()
    p_id = prov or provider()
    p_label = PROVIDERS.get(p_id, ("",))[0] or p_id.title()
    name = mdl or ((fallback_models() if prov == (fallback().get("provider")) else models()) or ["?"])[0]
    if "404" in text or "not found" in low or "does not exist" in low or "unknown model" in low or "not_found" in low:
        return f"{p_label} doesn't know the model \"{name}\". Check the model name in Settings, AI."
    if "401" in text or "403" in text or "api key" in low or "api_key" in low or "unauthorized" in low or "permission" in low:
        return f"{p_label} refused the key. Check your {p_label} API key under Settings, AI."
    if "429" in text or "quota" in low or "rate limit" in low or "exhausted" in low:
        return f"{p_label} says the key has hit its limit or has no credit left. Check the account with {p_label}, or try again later."
    if "connect" in low or "timed out" in low or "timeout" in low or "getaddrinfo" in low or "refused" in low:
        return f"Autoclipper couldn't reach {p_label}. Check the internet connection" + (" and that Ollama is running." if p_id == "ollama" else ".")
    return f"{p_label}: {text[:160]}"


def note_problem(e, prov=None, mdl=None):
    import time
    why = explain(e, prov=prov, mdl=mdl)
    if why:
        problem.update(text=why, at=time.time(), is_fallback=False, fallback_provider="")


def note_fallback(primary_name: str, fallback_name: str, reason: str = ""):
    import time
    problem.update(
        text=f"{primary_name} hit its rate limit or quota. Switched automatically to {fallback_name}.",
        at=time.time(),
        is_fallback=True,
        fallback_provider=fallback_name
    )


def clear_problem():
    problem.update(text="", at=0.0, is_fallback=False, fallback_provider="")


def _fail(r, model, k=None):
    text = r.text[:300].replace((k or key()) or "\0", "<key>")
    raise AIError(f"HTTP {r.status_code} {text}")


# ---------------------------------------------------------------- the schema, said in words for chat models
def _json_schema(s):
    """Gemini's schema spelling (ARRAY, OBJECT...) as ordinary JSON Schema."""
    if isinstance(s, dict):
        return {k: (v.lower() if k == "type" and isinstance(v, str) else _json_schema(v)) for k, v in s.items()}
    if isinstance(s, list):
        return [_json_schema(x) for x in s]
    return s


def _wrapped(prompt, schema):
    return (prompt + "\n\nAnswer with JSON only, no other text and no code fences: one object with a single key "
            "\"items\", whose value follows this JSON Schema:\n" + json.dumps(_json_schema(schema)))


def _parse(text):
    """The JSON in an answer, even if the model wrapped it in a code fence or a sentence."""
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        out = json.loads(text)
    except ValueError:
        m = re.search(r"[\[{].*[\]}]", text, re.S)
        if not m:
            raise AIError("The AI did not answer with JSON.")
        out = json.loads(m.group(0))
    if isinstance(out, dict) and "items" in out:
        out = out["items"]
    return out


# ---------------------------------------------------------------- one call
def call(model: str, prompt: str, schema: dict):
    """Ask one model using primary AI settings. Returns (parsed JSON, tokens sent, tokens written back)."""
    return _call_as(provider(), model, base_url(), key(), prompt, schema)
