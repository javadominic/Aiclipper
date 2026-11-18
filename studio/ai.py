"""Talking to the AI service. You bring your own key, and choose the service and model in the dashboard
(Settings, AI) or in .env. What is saved from the dashboard is kept in ai.json in the data folder and wins.

    AUTOCLIPPER_AI_PROVIDER   gemini | openai | xai | groq | openrouter | anthropic | ollama | custom
    AUTOCLIPPER_AI_MODELS     one or more model ids, comma separated, tried in this order
    AUTOCLIPPER_AI_KEY        your key (or AUTOCLIPPER_AI_KEY_FILE, a file holding it)
    AUTOCLIPPER_AI_BASE_URL   only for "custom" (any service that speaks the OpenAI chat format), or to override

Two things keep the clips coming when a service says "no more", and both are set up in the dashboard:

  * A POOL OF KEYS for one service. Free keys run out one at a time, so several can be saved and Autoclipper
    moves to the next the moment one hits its limit. Each key rests only as long as its own limit lasts.
  * A BACKUP SERVICE, tried once every key and model of the first one has had its turn.

Model discovery, exact-id routing and the audit log of what was dispatched live in registry.py. This file owns
the settings, the pool of keys, and the order the keys and models are tried in.

Every service is asked for the same thing: JSON that follows a small schema. Gemini takes the schema directly.
The others are told the schema in the prompt and asked for a JSON object, which works with any chat model."""
import hashlib
import json
import os
import re
import threading
import time

import httpx

from . import config
from . import registry
from .registry import (
    ADAPTERS,
    EXPLICIT_MODEL_ALIASES,
    STATUS_AVAILABLE,
    STATUS_UNAVAILABLE,
    STATUS_UNAUTHORIZED,
    STATUS_UNREACHABLE,
    STATUS_UNSUPPORTED,
    STATUS_UNVERIFIED,
    global_registry,
    global_router,
)

# name -> (label shown on the pages, address, how it is spoken to, where to get a key, an example model id)
PROVIDERS = {
    "gemini": ("Gemini", "https://generativelanguage.googleapis.com/v1beta/models", "gemini", "https://aistudio.google.com/apikey", "gemini-2.0-flash"),
    "openai": ("OpenAI", "https://api.openai.com/v1", "openai", "https://platform.openai.com/api-keys", "gpt-4o-mini"),
    "xai": ("Grok", "https://api.x.ai/v1", "openai", "https://console.x.ai", "grok-4"),
    "groq": ("Groq", "https://api.groq.com/openai/v1", "openai", "https://console.groq.com/keys", "openai/gpt-oss-120b"),
    "openrouter": ("OpenRouter", "https://openrouter.ai/api/v1", "openai", "https://openrouter.ai/keys", "google/gemini-2.0-flash-001"),
    "anthropic": ("Claude", "https://api.anthropic.com/v1", "anthropic", "https://console.anthropic.com", "claude-3-5-haiku-20241022"),
    "ollama": ("Ollama (on this machine)", "http://127.0.0.1:11434/v1", "openai", "https://ollama.com", "llama3.1"),
    "custom": ("Custom (OpenAI-compatible)", "", "openai", "", ""),
}
ALIASES = {"google": "gemini", "grok": "xai", "claude": "anthropic", "local": "ollama"}
NO_KEY = {"ollama", "custom"}       # a model running on your own machine needs no key

# Known working models in priority order for each provider, tried automatically if needed
PROVIDER_CANDIDATES = {
    "gemini": ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-2.0-flash-lite", "gemini-1.5-pro"],
    "groq": ["llama-3.1-8b-instant", "openai/gpt-oss-120b", "deepseek-r1-distill-llama-70b", "llama-3.3-70b-versatile"],
    "openai": ["gpt-4o-mini", "gpt-4o"],
    "xai": ["grok-2", "grok-beta"],
    "anthropic": ["claude-3-5-haiku-20241022", "claude-3-5-sonnet-20241022"],
    "ollama": ["llama3.1", "mistral"],
    "openrouter": ["google/gemini-2.0-flash-001", "meta-llama/llama-3.1-8b-instruct"],
}

# Never offered for clipping: these answer, but not with the JSON we ask for, or they are not language models.
SKIP = ("embed", "imagen", "tts", "whisper", "dall", "realtime", "gecko", "bison", "aqa", "transcribe",
        "moderation", "computer-use", "guard", "-audio", "search-grounding", "rerank", "ocr")

MAX_KEYS = 8            # how many keys one service may hold
MAX_MODELS = 12         # how many models are written down and tried, in order
AUTO_PICK = 6           # how many "Find models" ticks by itself

# How long a refusal is remembered for. A per-minute limit lifts in a minute; a daily quota does not, so it is
# worth six hours before Autoclipper asks again. A key the service will not accept, and a model name it does not
# know, are left alone until the settings are saved again.
REST_MINUTE, REST_DAY, REST_DEAD = 75, 6 * 3600, 24 * 3600

TINY_PROMPT = 'Answer with a JSON array that holds the single string "ok".'
TINY_SCHEMA = {"type": "ARRAY", "items": {"type": "STRING"}}


def _norm(p) -> str:
    """A service name as this file spells it: lower case, and any of its other names turned into the main one."""
    p = str(p or "").strip().lower()
    return ALIASES.get(p, p)


def _kid(k: str) -> str:
    """A short stand-in name for a key. Limits are remembered against this, and the page is shown this, so the
    key itself is never written into a dictionary key or sent to a browser."""
    return hashlib.sha256((k or "").encode("utf-8")).hexdigest()[:12]

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


def _keys_of(d: dict) -> list:
    """Every key in one saved section, as [{'key','label'}]. Files written before there was a pool hold a single
    key under "key", and that is still read."""
    out = []
    raw = d.get("keys")
    if isinstance(raw, list):
        for entry in raw:
            if isinstance(entry, dict):
                text, lab = str(entry.get("key") or ""), str(entry.get("label") or "")
            else:
                text, lab = str(entry or ""), ""
            text = text.strip()
            if text and text not in [o["key"] for o in out]:
                out.append({"key": text[:400], "label": lab.strip()[:40]})
    if not out:
        one = str(d.get("key") or "").strip()
        if one:
            out.append({"key": one[:400], "label": ""})
    return out[:MAX_KEYS]


def current() -> dict:
    """Provider, models, address and keys: what the dashboard saved, or else what .env says.

    "key" is the first key of the pool, kept because the registry and the older settings ask for one. "keys" is
    the whole pool, and is what calls are actually made with."""
    d = _from_dashboard()
    saved_here = bool(d.get("provider"))
    p = _norm(d.get("provider") or config.AI_PROVIDER)
    env_key = config.ai_key()
    pool = _keys_of(d) if saved_here else ([{"key": env_key, "label": ""}] if env_key else [])
    return {"provider": p,
            "models": list(d["models"]) if saved_here else list(config.AI_MODELS),
            "base_url": (str(d.get("base_url") or "") if saved_here else config.AI_BASE_URL).rstrip("/"),
            "key": pool[0]["key"] if pool else "",
            "keys": pool}


def provider() -> str:
    return current()["provider"]


def models() -> list:
    return current()["models"]


def key() -> str:
    return current()["key"]


def keys() -> list:
    """The pool of keys for the service in use."""
    return current()["keys"]


def all_keys() -> list:
    """Every key text anywhere in the settings, primary and backup. Used only to scrub keys out of messages."""
    got = [k["key"] for k in current()["keys"]]
    fb = fallback()
    if fb:
        got += [k["key"] for k in _keys_of(fb)]
    fb2 = fallback2()
    if fb2:
        got += [k["key"] for k in _keys_of(fb2)]
    return [k for k in got if k]


def label() -> str:
    return PROVIDERS.get(provider(), ("",))[0]


def base_url() -> str:
    c = current()
    return c["base_url"] or PROVIDERS.get(c["provider"], ("", ""))[1]


def ready() -> bool:
    c = current()
    return bool(c["provider"] in PROVIDERS and c["models"] and base_url()
                and (c["keys"] or c["provider"] in NO_KEY))


def _public_keys(prov: str, pool: list) -> list:
    """The pool as the page may see it: a name for each key, its last four characters, and whether it is resting.
    The keys themselves never leave the server."""
    now = time.time()
    if prov == provider():
        ms = models()
    elif prov == (fallback().get("provider") or ""):
        ms = fallback_models()
    elif prov == (fallback2().get("provider") or ""):
        ms = fallback2_models()
    else:
        ms = []
    resting = {}
    with _state_lock:
        for (p, kid), (until, why, _lab) in list(_key_out.items()):
            if p == prov and until > now:
                resting[kid] = (int(until - now), why)
        # A key the service turned away is not the only kind of resting. A free key usually hits its quota one
        # model at a time, so once every model on that key is over quota the key is spent too, and showing it as
        # "Ready" would have the owner wondering why nothing is using it.
        for k in pool:
            kid = _kid(k["key"])
            if kid in resting or not ms:
                continue
            waits = []
            for m in ms:
                got = _pair_out.get((prov, kid, m))
                if not got or got[0] <= now:
                    waits = None
                    break
                waits.append((got[0] - now, got[1]))
            if waits:
                resting[kid] = (int(min(w for w, _ in waits)), waits[0][1])
    out = []
    for i, k in enumerate(pool):
        kid = _kid(k["key"])
        wait = resting.get(kid)
        out.append({"id": kid, "index": i, "label": k["label"] or f"Key {i + 1}",
                    "hint": k["key"][-4:] if len(k["key"]) >= 12 else "", "has_key": bool(k["key"]),
                    "resting": wait[0] if wait else 0, "why": wait[1] if wait else ""})
    return out


def public() -> dict:
    """What the settings page may see. The key itself never leaves the server: only whether there is one, and
    its last four characters so the owner can tell which key it is."""
    c = current()
    fb = fallback()
    fb_pool = _keys_of(fb) if fb else []
    fb_info = {}
    if fb:
        fb_info = {"fallback_provider": fb["provider"], "fallback_models": fb["models"],
                   "fallback_base_url": fb.get("base_url", ""),
                   "fallback_has_key": bool(fb.get("key")),
                   "fallback_key_hint": fb["key"][-4:] if len(fb.get("key", "")) >= 12 else "",
                   "fallback_label": PROVIDERS.get(fb["provider"], ("",))[0],
                   "fallback_keys": _public_keys(fb["provider"], fb_pool),
                   "fallback_ready": bool(fb["provider"] in PROVIDERS and fb["models"]
                                         and (fb.get("base_url") or PROVIDERS.get(fb["provider"], ("", ""))[1])
                                         and (fb_pool or fb["provider"] in NO_KEY))}
    fb2 = fallback2()
    fb2_pool = _keys_of(fb2) if fb2 else []
    fb2_info = {}
    if fb2:
        fb2_info = {"fallback2_provider": fb2["provider"], "fallback2_models": fb2["models"],
                    "fallback2_base_url": fb2.get("base_url", ""),
                    "fallback2_has_key": bool(fb2.get("key")),
                    "fallback2_key_hint": fb2["key"][-4:] if len(fb2.get("key", "")) >= 12 else "",
                    "fallback2_label": PROVIDERS.get(fb2["provider"], ("",))[0],
                    "fallback2_keys": _public_keys(fb2["provider"], fb2_pool),
                    "fallback2_ready": bool(fb2["provider"] in PROVIDERS and fb2["models"]
                                           and (fb2.get("base_url") or PROVIDERS.get(fb2["provider"], ("", ""))[1])
                                           and (fb2_pool or fb2["provider"] in NO_KEY))}
    return {"provider": c["provider"], "models": c["models"], "base_url": c["base_url"], "has_key": bool(c["key"]),
            "key_hint": c["key"][-4:] if len(c["key"]) >= 12 else "", "ready": ready(), "label": label(),
            "saved_here": bool(_from_dashboard().get("provider")),
            "keys": _public_keys(c["provider"], c["keys"]),
            "candidates": list(PROVIDER_CANDIDATES.get(c["provider"], ())),
            "max_keys": MAX_KEYS, "max_models": MAX_MODELS, "auto_pick": AUTO_PICK,
            "using": describe(last_used) if last_used.get("model") else "",
            "providers": [{"id": k, "label": v[0], "address": v[1], "needs_key": k not in NO_KEY, "key_url": v[3],
                           "example": v[4], "candidates": list(PROVIDER_CANDIDATES.get(k, ()))}
                          for k, v in PROVIDERS.items()],
            "registry": global_registry.list_models(),
            "aliases": EXPLICIT_MODEL_ALIASES,
            "execution_logs": global_router.get_logs(limit=20),
            **pool_status(), **fb_info, **fb2_info}


def _clean_models(model_ids, p: str) -> list:
    """The model list from the page, checked, and kept in the order it was given: that order is the order they
    are tried in. An empty list, or the word "auto", means "use what this service is known to offer", so nobody
    has to know a model name to get started."""
    if isinstance(model_ids, str):
        model_ids = model_ids.split(",")
    ms = []
    for m in (model_ids or []):
        m = str(m).strip()[:120]
        if m and m not in ms:
            ms.append(m)
    if not ms or ms == ["auto"] or ms == [""]:
        ms = list(PROVIDER_CANDIDATES.get(p) or ([PROVIDERS[p][4]] if PROVIDERS[p][4] else []))
    if not ms:
        raise ValueError("Enter the model to use.")
    if any(not re.fullmatch(r"[\w.\-:/@+]+", m) for m in ms):
        raise ValueError("A model id has letters, digits and . - : / + only.")
    return ms[:MAX_MODELS]


def _clean_address(address, p: str) -> str:
    address = str(address or "").strip().rstrip("/")[:300]
    if address and not re.fullmatch(r"https?://[^\s]+", address):
        raise ValueError("The address has to start with http:// or https://")
    if p == "custom" and not address:
        raise ValueError("A custom service needs its address, for example http://127.0.0.1:1234/v1")
    return address


def _merge_keys(keys, new_key, kept: list) -> list:
    """Work out the pool of keys to save.

    The page sends the pool back as a list of {id, key, label}. It never has the key texts, only their stand-in
    ids, so an entry with an id and no key text means "keep that one"; an entry with no id and some text is a new
    key; a saved key the page leaves out has been removed. `new_key` is the older single-key argument and still
    works: it replaces the first key. `kept` is the pool already saved for this same service."""
    by_id = {_kid(k["key"]): k for k in kept}
    if isinstance(keys, str):
        keys = [{"key": keys}]
    if keys is None:
        if new_key is None:
            return list(kept)                       # nothing said about keys: leave the pool alone
        keys = [{"key": str(new_key or "").strip()}]
    out = []
    for entry in keys:
        if not isinstance(entry, dict):
            entry = {"key": str(entry or "")}
        kid = str(entry.get("id") or "")
        text = str(entry.get("key") or "").strip()
        lab = str(entry.get("label") or "").strip()[:40]
        have = by_id.get(kid) if kid else None
        if have:
            text = text or have["key"]              # a blank box keeps the key already saved there
            lab = lab or have.get("label", "")
        if text and text not in [o["key"] for o in out]:
            out.append({"key": text[:400], "label": lab})
    return out[:MAX_KEYS]


def _write(data: dict):
    """Put the settings on disk whole, so a half-written file is never read, and only this server's owner can."""
    with _lock:
        f = _file()
        tmp = f.with_suffix(".tmp")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        try:
            os.chmod(tmp, 0o600)        # the keys are only for whoever runs the server
        except OSError:
            pass
        tmp.replace(f)
        _saved["at"] = None
    reset_state()                                   # whatever was resting is given another chance


def save(provider_id: str, model_ids, address: str = "", new_key=None, keys=None) -> dict:
    """Save the choice made in the dashboard: the service, the models in the order to try them, and the pool of
    keys. `new_key` None keeps the keys that are already there (the page never has them to send back); an empty
    string removes them. `keys` is the whole pool, and wins over `new_key` when both are given."""
    p = _norm(provider_id)
    if p not in PROVIDERS:
        raise ValueError("Choose one of the AI services in the list.")
    ms = _clean_models(model_ids, p)
    address = _clean_address(address, p)
    old = current()
    same = old["provider"] == p and bool(_from_dashboard().get("provider"))
    pool = _merge_keys(keys, new_key, old["keys"] if same else [])
    if not pool and p not in NO_KEY:
        raise ValueError("Paste the key for this service.")
    existing = dict(_from_dashboard())
    data = {"provider": p, "models": ms, "base_url": address,
            "key": pool[0]["key"] if pool else "", "keys": pool}
    if existing.get("fallback"):
        data["fallback"] = existing["fallback"]     # saving the primary settings leaves the backup alone
    _write(data)
    return public()


# ---------------------------------------------------------------- fallback: a second service when the first hits its limit

def fallback() -> dict:
    """The backup service settings, or an empty dict if none is set. "keys" is its own pool, and "key" the first
    of them, because the router takes one key at a time."""
    d = _from_dashboard()
    fb = d.get("fallback")
    if fb and isinstance(fb, dict):
        p = _norm(fb.get("provider"))
        if p in PROVIDERS:
            pool = _keys_of(fb)
            return {"provider": p,
                    "models": [str(m) for m in (fb.get("models") or [])][:MAX_MODELS],
                    "base_url": str(fb.get("base_url") or "").rstrip("/"),
                    "key": pool[0]["key"] if pool else "",
                    "keys": pool}
    # nothing saved from the dashboard: a backup may still be set in .env
    p = _norm(config.AI_FALLBACK_PROVIDER)
    if p in PROVIDERS:
        ms = list(config.AI_FALLBACK_MODELS or ([PROVIDERS[p][4]] if PROVIDERS[p][4] else []))
        one = config.ai_fallback_key()
        return {"provider": p, "models": ms, "base_url": config.AI_FALLBACK_BASE_URL.rstrip("/"),
                "key": one, "keys": ([{"key": one, "label": ""}] if one else [])}
    return {}


def fallback_ready() -> bool:
    fb = fallback()
    if not fb:
        return False
    return bool(fb["provider"] in PROVIDERS and fb["models"]
                and (fb.get("base_url") or PROVIDERS.get(fb["provider"], ("", ""))[1])
                and (fb.get("keys") or fb["provider"] in NO_KEY))


def fallback_label() -> str:
    fb = fallback()
    return PROVIDERS.get(fb.get("provider", ""), ("",))[0]


def fallback_models() -> list:
    fb = fallback()
    return fb.get("models", [])


def save_fallback(provider_id: str, model_ids, address: str = "", new_key=None, keys=None) -> dict:
    """Save the backup service. The same rules as save(), stored under a "fallback" key, with its own pool."""
    p = _norm(provider_id)
    if p not in PROVIDERS:
        raise ValueError("Choose one of the AI services in the list.")
    ms = _clean_models(model_ids, p)
    address = _clean_address(address, p)
    old_fb = fallback()
    same = old_fb.get("provider") == p and bool(_from_dashboard().get("fallback"))
    pool = _merge_keys(keys, new_key, old_fb.get("keys", []) if same else [])
    if not pool and p not in NO_KEY:
        raise ValueError("Paste the key for this service.")
    existing = dict(_from_dashboard())
    existing["fallback"] = {"provider": p, "models": ms, "base_url": address,
                            "key": pool[0]["key"] if pool else "", "keys": pool}
    _write(existing)
    return public()


def remove_fallback() -> dict:
    """Remove the backup service."""
    existing = dict(_from_dashboard())
    existing.pop("fallback", None)
    _write(existing)
    return public()


# ---------------------------------------------------------------- fallback2: a third service when both primary and fallback are exhausted

def fallback2() -> dict:
    """The second backup service settings, or an empty dict if none is set."""
    d = _from_dashboard()
    fb2 = d.get("fallback2")
    if fb2 and isinstance(fb2, dict):
        p = _norm(fb2.get("provider"))
        if p in PROVIDERS:
            pool = _keys_of(fb2)
            return {"provider": p,
                    "models": [str(m) for m in (fb2.get("models") or [])][:MAX_MODELS],
                    "base_url": str(fb2.get("base_url") or "").rstrip("/"),
                    "key": pool[0]["key"] if pool else "",
                    "keys": pool}
    # nothing saved from the dashboard: check .env
    p = _norm(config.AI_FALLBACK2_PROVIDER)
    if p in PROVIDERS:
        ms = list(config.AI_FALLBACK2_MODELS or ([PROVIDERS[p][4]] if PROVIDERS[p][4] else []))
        one = config.ai_fallback2_key()
        return {"provider": p, "models": ms, "base_url": config.AI_FALLBACK2_BASE_URL.rstrip("/"),
                "key": one, "keys": ([{"key": one, "label": ""}] if one else [])}
    return {}


def fallback2_ready() -> bool:
    fb2 = fallback2()
    if not fb2:
        return False
    return bool(fb2["provider"] in PROVIDERS and fb2["models"]
                and (fb2.get("base_url") or PROVIDERS.get(fb2["provider"], ("", ""))[1])
                and (fb2.get("keys") or fb2["provider"] in NO_KEY))


def fallback2_label() -> str:
    fb2 = fallback2()
    return PROVIDERS.get(fb2.get("provider", ""), ("",))[0]


def fallback2_models() -> list:
    fb2 = fallback2()
    return fb2.get("models", [])


def save_fallback2(provider_id: str, model_ids, address: str = "", new_key=None, keys=None) -> dict:
    """Save the second backup service. Same rules as save_fallback(), stored under a 'fallback2' key."""
    p = _norm(provider_id)
    if p not in PROVIDERS:
        raise ValueError("Choose one of the AI services in the list.")
    ms = _clean_models(model_ids, p)
    address = _clean_address(address, p)
    old_fb2 = fallback2()
    same = old_fb2.get("provider") == p and bool(_from_dashboard().get("fallback2"))
    pool = _merge_keys(keys, new_key, old_fb2.get("keys", []) if same else [])
    if not pool and p not in NO_KEY:
        raise ValueError("Paste the key for this service.")
    existing = dict(_from_dashboard())
    existing["fallback2"] = {"provider": p, "models": ms, "base_url": address,
                             "key": pool[0]["key"] if pool else "", "keys": pool}
    _write(existing)
    return public()


def remove_fallback2() -> dict:
    """Remove the second backup service."""
    existing = dict(_from_dashboard())
    existing.pop("fallback2", None)
    _write(existing)
    return public()


# Words that mean "this 400 is about a setting we added", so it is worth asking again without it. Any other 400
# is about the request itself, and asking twice more would only waste what is left of the key's quota.
_SETTING_WORDS = ("response_format", "temperature", "json_object", "json mode", "json_schema",
                  "unsupported parameter", "unknown parameter", "not supported", "unsupported value")


def _about_settings(text: str) -> bool:
    low = str(text or "").lower()
    return any(w in low for w in _SETTING_WORDS)


def _call_as(prov: str, mdl: str, url: str, k: str, prompt: str, schema: dict):
    """Make one AI call using the given provider settings, not the current ones.
    Returns (parsed JSON, tokens sent, tokens written back). Anything that goes wrong is raised as an AIError
    whose text says which service and model it was, so the caller can work out what to try next.

    The registry's adapters call this by name, and their tests replace it, so the signature and the three-part
    answer are a contract: do not change them."""
    how = PROVIDERS[prov][2]
    if how == "gemini":
        body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.4, "responseMimeType": "application/json", "responseSchema": schema}}
        r = httpx.post(f"{url}/{mdl}:generateContent", json=body, headers={"x-goog-api-key": k}, timeout=120)
        if r.status_code != 200:
            _fail(r, mdl, k)
        data = r.json()
        # A blocked or truncated answer comes back with no candidates at all. Saying so plainly matters: a bare
        # KeyError here would be blamed on the key, and the next key in the pool would be tried for nothing.
        cands = data.get("candidates") or []
        if not cands:
            why = (data.get("promptFeedback") or {}).get("blockReason") or "the service sent back no answer"
            raise AIError(f"HTTP 200 from {prov}/{mdl} but nothing to read: {why}")
        parts = (cands[0].get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts if isinstance(p, dict))
        if not text.strip():
            raise AIError(f"HTTP 200 from {prov}/{mdl} but the answer was empty "
                          f"({cands[0].get('finishReason') or 'no reason given'})")
        u = data.get("usageMetadata") or {}
        sent = int(u.get("promptTokenCount") or 0)
        return _parse(text), sent, max(0, int(u.get("totalTokenCount") or 0) - sent)
    elif how == "anthropic":
        body = {"model": mdl, "max_tokens": 8192, "messages": [{"role": "user", "content": _wrapped(prompt, schema)}]}
        r = httpx.post(f"{url}/messages", json=body, timeout=180,
                       headers={"x-api-key": k, "anthropic-version": "2023-06-01"})
        if r.status_code != 200:
            _fail(r, mdl, k)
        data = r.json()
        text = "".join(b.get("text", "") for b in (data.get("content") or [])
                       if isinstance(b, dict) and b.get("type") == "text")
        if not text.strip():
            raise AIError(f"HTTP 200 from {prov}/{mdl} but the answer was empty "
                          f"({data.get('stop_reason') or 'no reason given'})")
        u = data.get("usage") or {}
        return _parse(text), int(u.get("input_tokens") or 0), int(u.get("output_tokens") or 0)
    else:
        headers = {"Authorization": f"Bearer {k}"} if k else {}
        body = {"model": mdl, "messages": [{"role": "user", "content": _wrapped(prompt, schema)}]}
        # Some services refuse the extra settings, so they are dropped one at a time and the call made again.
        extras = [{"temperature": 0.4, "response_format": {"type": "json_object"}}, {"response_format": {"type": "json_object"}}, {}]
        r = None
        for extra in extras:
            r = httpx.post(f"{url}/chat/completions", json={**body, **extra}, headers=headers, timeout=180)
            if r.status_code not in (400, 422) or not _about_settings(r.text):
                break
        if r.status_code != 200:
            _fail(r, mdl, k)
        data = r.json()
        choices = data.get("choices") or []
        if not choices:
            raise AIError(f"HTTP 200 from {prov}/{mdl} but the service sent back no answer")
        u = data.get("usage") or {}
        return (_parse((choices[0].get("message") or {}).get("content")),
                int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0))


def call_fallback(model: str, prompt: str, schema: dict):
    """Ask the fallback model in strict mode. Returns (parsed JSON, tokens sent, tokens written back)."""
    fb = fallback()
    if not fallback_ready():
        raise AIError("Fallback AI service is not configured or ready.")
    url = fb.get("base_url") or PROVIDERS.get(fb["provider"], ("", ""))[1]
    out, dispatched_model, sent, wrote = global_router.dispatch(
        provider_id=fb["provider"],
        model_id=model,
        base_url=url,
        key=fb.get("key", ""),
        prompt=prompt,
        schema=schema,
        allow_fallback=False,
    )
    return out, sent, wrote


def test_fallback() -> dict:
    """Ask the backup service with a tiny request, trying each of its keys, to prove it really will take over."""
    if not fallback_ready():
        raise ValueError("Save a backup service, model and key first.")
    fb = fallback()
    worked, refused, tokens, last, out = [], [], 0, None, None
    for cand in candidates(limit=MAX_KEYS * 3):
        if cand.get("tier") != "fallback":
            continue
        try:
            out, sent, wrote = call_candidate(cand, TINY_PROMPT, TINY_SCHEMA)
            note_success(cand)
            worked.append(describe(cand))
            tokens += sent + wrote
            rec = global_registry.register_manual(cand["provider"], cand["model"])
            rec.status = STATUS_AVAILABLE
            rec.last_verified = time.time()
        except Exception as e:  # noqa: BLE001
            last = e
            note_failure(cand, e)
            refused.append({"model": describe(cand),
                            "why": explain(e, prov=cand["provider"], mdl=cand["model"]) or str(e)[:160]})
    if not worked:
        raise AIError((refused or [{}])[0].get("why") or explain(last, prov=fb["provider"]) or str(last))
    global_registry._save_cache()
    return {"ok": True, "provider": fb["provider"], "model": worked[0], "models": worked,
            "tokens": tokens, "refused": refused, "answer": str(out)[:80]}


def test_fallback2() -> dict:
    """Ask the second backup service with a tiny request, trying each of its keys, to prove it really will take over."""
    if not fallback2_ready():
        raise ValueError("Save a second backup service, model and key first.")
    fb2 = fallback2()
    worked, refused, tokens, last, out = [], [], 0, None, None
    for cand in candidates(limit=MAX_KEYS * 3):
        if cand.get("tier") != "fallback2":
            continue
        try:
            out, sent, wrote = call_candidate(cand, TINY_PROMPT, TINY_SCHEMA)
            note_success(cand)
            worked.append(describe(cand))
            tokens += sent + wrote
            rec = global_registry.register_manual(cand["provider"], cand["model"])
            rec.status = STATUS_AVAILABLE
            rec.last_verified = time.time()
        except Exception as e:  # noqa: BLE001
            last = e
            note_failure(cand, e)
            refused.append({"model": describe(cand),
                            "why": explain(e, prov=cand["provider"], mdl=cand["model"]) or str(e)[:160]})
    if not worked:
        raise AIError((refused or [{}])[0].get("why") or explain(last, prov=fb2["provider"]) or str(last))
    global_registry._save_cache()
    return {"ok": True, "provider": fb2["provider"], "model": worked[0], "models": worked,
            "tokens": tokens, "refused": refused, "answer": str(out)[:80]}


def _ranked(p: str, ids) -> list:
    """A service's model ids in the order worth trying them: the ones we know are good for clipping first, then
    everything else the key may use. Models that cannot answer our question at all are left out."""
    pref = PROVIDER_CANDIDATES.get(p, ())
    kept = {m for m in ids if m and not any(s in m.lower() for s in SKIP)}

    def rank(m):
        low, tail_low = m.lower(), m.lower().split("/")[-1]
        # An exact match anywhere in the list beats a partial match on an earlier entry. Without this first pass
        # "gemini-2.0-flash-lite" reads as a near-miss of "gemini-2.0-flash" and sorts below the dearer models,
        # which is the opposite of what "quick and cheap first" means.
        for i, want in enumerate(pref):
            if low == want or tail_low == want.split("/")[-1]:
                return (0, i, m)
        for i, want in enumerate(pref):
            if want.split("/")[-1] in low:
                return (1, i, m)
        return (2, 0, m)

    return sorted(kept, key=rank)[:400]


def _saved_key_for(p: str) -> str:
    """A key already saved for this service, primary or backup. The page never has the key texts, so this is what
    lets "Find models" work without the owner pasting the same key a second time."""
    c = current()
    if c["provider"] == p and c["keys"]:
        return c["keys"][0]["key"]
    fb = fallback()
    if fb.get("provider") == p and fb.get("keys"):
        return fb["keys"][0]["key"]
    return ""


def list_models(provider_id=None, address=None, with_key=None, force=False) -> list:
    """The model ids this key may use, best first. Discovered through the registry, which caches the answer, so
    reopening the settings page does not ask the service all over again."""
    c = current()
    p = _norm(provider_id or c["provider"])
    if p not in PROVIDERS:
        raise ValueError("Choose one of the AI services in the list.")
    typed = str(with_key or "").strip()
    k = typed or _saved_key_for(p)
    if not k and p not in NO_KEY:
        # Asking with no key only earns a confusing "401" from the service. Say what to do instead.
        raise ValueError(f"Paste your {PROVIDERS[p][0]} key first, then press Find models. "
                         "The key is used only to ask the service what it offers, and is kept on this server.")
    base = str(address or "").strip().rstrip("/") or (c["base_url"] if c["provider"] == p else "") or PROVIDERS[p][1]
    if not base and p not in NO_KEY:
        raise ValueError("Enter the service's address first.")
    try:
        # A key typed into the box is being checked, not browsed, so it must not be answered from a cache that
        # was filled using a different key: that is how a wrong key comes to look like a working one.
        records = global_registry.discover(p, base, k, force=force or bool(typed))
        found = [r.id for r in records if not r.is_alias]
        if found:
            return _ranked(p, found)
    except Exception as e:  # noqa: BLE001
        cached = [r["id"] for r in global_registry.list_models(p) if not r.get("is_alias")]
        if cached and not typed:
            return _ranked(p, cached)
        raise AIError(f"Could not discover models from {PROVIDERS[p][0]}: {explain(e, prov=p)}")
    return []


def auto_pick(provider_id=None, address=None, with_key=None, want: int = AUTO_PICK) -> list:
    """The models to use with this key, chosen by Autoclipper: list_models already puts the quick inexpensive ones
    that are good for clipping at the top, so taking the first few is the whole job."""
    return list_models(provider_id, address, with_key)[:max(1, min(MAX_MODELS, int(want or AUTO_PICK)))]


def test() -> dict:
    """One tiny request with the settings in use, to prove the keys, the address and the models all work.

    With a pool of keys every one of them is asked, so the owner finds out now that the third key is bad rather
    than halfway through a job. The answer names what worked and what did not. Models are never substituted:
    each one is asked under its own exact id, and the registry records which of them verified."""
    if not ready():
        raise ValueError("Save a service, a model and a key first.")
    p = provider()
    reset_state()                       # a test should try everything, even what hit a limit a minute ago
    worked, refused, tokens, last, out = [], [], 0, None, None
    for cand in candidates(limit=max(3, MAX_KEYS)):
        try:
            out, sent, wrote = call_candidate(cand, TINY_PROMPT, TINY_SCHEMA)
            note_success(cand)
            note_working(cand, tried=len(refused))
            worked.append(describe(cand))
            tokens += sent + wrote
            rec = global_registry.register_manual(cand["provider"], cand["model"])
            rec.status = STATUS_AVAILABLE
            rec.last_verified = time.time()
        except Exception as e:  # noqa: BLE001
            last = e
            note_failure(cand, e)
            refused.append({"model": describe(cand),
                            "why": explain(e, prov=cand["provider"], mdl=cand["model"]) or str(e)[:160]})
    if not worked:
        note_problem(last, prov=None, mdl=None)
        raise AIError((refused or [{}])[0].get("why") or explain(last) or str(last))
    global_registry._save_cache()
    return {"ok": True, "provider": p, "model": worked[0], "models": worked, "tokens": tokens,
            "refused": refused, "answer": str(out)[:80]}


class AIError(RuntimeError):
    pass


problem = {"text": "", "at": 0.0, "is_fallback": False, "fallback_provider": ""}       # why the AI last failed, for the dashboard; cleared when it answers again
last_used = {"model": "", "provider": "", "label": "", "backup": False, "key_label": "", "pool_size": 1, "at": 0.0}


# ---------------------------------------------------------------- what is out of action right now
# A service that says "no more" is not asked again until its limit has had time to lift. Kept in memory only:
# limits lift by themselves, and a restart should start hopeful rather than remember yesterday's quota.
_state_lock = threading.Lock()
_pair_out: dict = {}      # (service, key id, model) -> (until, why, key label): that key hit its limit on that model
_key_out: dict = {}       # (service, key id)        -> (until, why, key label): the service will not take that key
_model_out: dict = {}     # (service, model)         -> (until, why, ""): no key will help, there is no such model


def reset_state():
    """Forget every limit and every "what answered last". Called when settings are saved, so a corrected model
    name or a topped-up key is tried again at once instead of waiting out the old rest."""
    with _state_lock:
        _pair_out.clear()
        _key_out.clear()
        _model_out.clear()
    last_used.update(model="", provider="", label="", backup=False, key_label="", pool_size=1, at=0.0)


def _limit_kind(e) -> str:
    """What sort of "no" a service said.

    "minute" and "day" are limits that lift by themselves, "dead" is a key the service will not take, "model" is
    a name it does not know, and "" is anything else: a network fault or a strange answer, which is worth another
    try rather than a rest."""
    s = str(e).lower()
    if not s.strip():
        s = type(e).__name__.lower()      # an httpx timeout often carries no message, only its class name
    if ("401" in s or "403" in s or "api key not valid" in s or "invalid api key" in s or "invalid_api_key" in s
            or "unauthorized" in s or "permission_denied" in s or "authentication_error" in s):
        return "dead"
    if ("404" in s or "not found" in s or "does not exist" in s or "unknown model" in s
            or "not_found" in s or "model_not_found" in s or "no such model" in s):
        return "model"
    if ("429" in s or "quota" in s or "rate limit" in s or "rate_limit" in s or "exhausted" in s
            or "too many requests" in s or "insufficient" in s or "credit" in s or "billing" in s):
        if any(w in s for w in ("per day", "daily", "rpd", "quota", "resource_exhausted",
                                "insufficient", "credit", "billing")):
            return "day"
        return "minute"
    return ""


def _resting(prov: str, kid: str, mdl: str):
    """How many seconds this key and model must still be left alone, and why. (0, "") when it may be tried now."""
    now = time.time()
    with _state_lock:
        for table, k in ((_model_out, (prov, mdl)), (_key_out, (prov, kid)), (_pair_out, (prov, kid, mdl))):
            got = table.get(k)
            if not got:
                continue
            if got[0] > now:
                return got[0] - now, got[1]
            table.pop(k, None)            # its time is up: forget it
    return 0.0, ""


def note_failure(cand, e) -> str:
    """Remember that this key and model said no, and what sort of no it was, so the next call goes straight to
    one that still has quota left instead of asking the same tired key again. Returns the sort of no."""
    kind = _limit_kind(e)
    if not kind or not cand:
        return kind
    prov, kid, mdl = cand.get("provider", ""), _kid(cand.get("key", "")), cand.get("model", "")
    lab = cand.get("key_label", "")
    until = {"minute": REST_MINUTE, "day": REST_DAY, "dead": REST_DEAD, "model": REST_DEAD}[kind]
    why = {"minute": "Hit its per-minute limit.", "day": "Out of quota for today.",
           "dead": "The service refused this key.", "model": f'The service has no model "{mdl}".'}[kind]
    with _state_lock:
        if kind == "dead":
            _key_out[(prov, kid)] = (time.time() + until, why, lab)
        elif kind == "model":
            _model_out[(prov, mdl)] = (time.time() + until, why, "")
        else:
            _pair_out[(prov, kid, mdl)] = (time.time() + until, why, lab)
    return kind


def note_success(cand):
    """This key and model answered, so any limit remembered against them is out of date: forget it."""
    if not cand:
        return
    prov, kid = cand.get("provider", ""), _kid(cand.get("key", ""))
    with _state_lock:
        _pair_out.pop((prov, kid, cand.get("model", "")), None)
        _key_out.pop((prov, kid), None)


def pool_status() -> dict:
    """Which keys and models are resting and for how long, so the settings page can say why one is greyed out."""
    now = time.time()
    resting = []
    with _state_lock:
        for (prov, kid, mdl), (until, why, lab) in sorted(_pair_out.items()):
            if until > now:
                resting.append({"provider": prov, "key": kid, "key_label": lab, "model": mdl,
                                "wait": int(until - now), "why": why})
        for (prov, kid), (until, why, lab) in sorted(_key_out.items()):
            if until > now:
                resting.append({"provider": prov, "key": kid, "key_label": lab, "model": "",
                                "wait": int(until - now), "why": why})
        for (prov, mdl), (until, why, _l) in sorted(_model_out.items()):
            if until > now:
                resting.append({"provider": prov, "key": "", "key_label": "", "model": mdl,
                                "wait": int(until - now), "why": why})
    return {"resting": resting}


# ---------------------------------------------------------------- the order keys and models are tried in

def _primary_section() -> dict:
    """The service in use, as one entry of the ladder, or {} when it is not usable."""
    c = current()
    if c["provider"] not in PROVIDERS or not c["models"]:
        return {}
    pool = c["keys"] or ([{"key": "", "label": ""}] if c["provider"] in NO_KEY else [])
    base = c["base_url"] or PROVIDERS.get(c["provider"], ("", ""))[1]
    if not pool or not base:
        return {}
    return {"provider": c["provider"], "label": PROVIDERS.get(c["provider"], ("",))[0] or c["provider"],
            "models": c["models"], "base_url": base, "keys": pool, "backup": False, "tier": "primary"}


def _backup_section() -> dict:
    """The backup service, as one entry of the ladder, or {} when there is none."""
    fb = fallback()
    if not fb or not fallback_ready():
        return {}
    pool = fb.get("keys") or ([{"key": "", "label": ""}] if fb["provider"] in NO_KEY else [])
    base = fb.get("base_url") or PROVIDERS.get(fb["provider"], ("", ""))[1]
    if not pool or not base:
        return {}
    return {"provider": fb["provider"], "label": PROVIDERS.get(fb["provider"], ("",))[0] or fb["provider"],
            "models": fb["models"], "base_url": base, "keys": pool, "backup": True, "tier": "fallback"}


def _backup2_section() -> dict:
    """The second backup service, as one entry of the ladder, or {} when there is none."""
    fb2 = fallback2()
    if not fb2 or not fallback2_ready():
        return {}
    pool = fb2.get("keys") or ([{"key": "", "label": ""}] if fb2["provider"] in NO_KEY else [])
    base = fb2.get("base_url") or PROVIDERS.get(fb2["provider"], ("", ""))[1]
    if not pool or not base:
        return {}
    return {"provider": fb2["provider"], "label": PROVIDERS.get(fb2["provider"], ("",))[0] or fb2["provider"],
            "models": fb2["models"], "base_url": base, "keys": pool, "backup": True, "tier": "fallback2"}


def candidates(limit: int = 24) -> list:
    """Every key and model to try, best first: each model across all the primary keys, then the backup service
    the same way. Anything resting after a limit is left out until its time is up, so a call goes straight to
    something that still has quota.

    Model-first on purpose. The limits a free key has are per model, so trying the best model on every key
    before dropping to a weaker one keeps the clips as good as they can be. Model ids are passed on exactly as
    configured: the router may resolve a known alias, but nothing is ever fuzzy-matched."""
    out = []
    for sec in (_primary_section(), _backup_section(), _backup2_section()):
        if not sec:
            continue
        for mdl in sec["models"]:
            for i, k in enumerate(sec["keys"]):
                if _resting(sec["provider"], _kid(k["key"]), mdl)[0]:
                    continue
                out.append({"provider": sec["provider"], "label": sec["label"], "base_url": sec["base_url"],
                            "key": k["key"], "key_index": i, "key_label": k["label"] or f"Key {i + 1}",
                            "model": mdl, "pool_size": len(sec["keys"]), "backup": sec["backup"], "tier": sec["tier"]})
                if len(out) >= limit:
                    return out
    return out


def all_resting() -> float:
    """How many seconds until something may be tried again, or 0 when something is available now. Lets the
    dashboard say "every key is resting, back in 4 minutes" instead of "the AI is not set up"."""
    soonest = 0.0
    for sec in (_primary_section(), _backup_section(), _backup2_section()):
        if not sec:
            continue
        for mdl in sec["models"]:
            for k in sec["keys"]:
                wait = _resting(sec["provider"], _kid(k["key"]), mdl)[0]
                if not wait:
                    return 0.0
                soonest = wait if not soonest else min(soonest, wait)
    return soonest


def describe(cand) -> str:
    """How to name a key and model on the dashboard: the model, plus the service or the key when more than one
    is in play, so it is clear which of them answered."""
    if not cand:
        return ""
    if cand.get("backup"):
        extra = cand.get("label") or ""
    elif cand.get("pool_size", 1) > 1:
        extra = cand.get("key_label") or ""
    else:
        extra = ""
    return f"{cand.get('model', '')} ({extra})" if extra else str(cand.get("model", ""))


def call_candidate(cand, prompt: str, schema: dict):
    """Ask one key and model from the ladder. This goes through the router rather than straight to _call_as, so
    every attempt still lands in the execution log the dashboard shows.
    Returns (parsed JSON, tokens sent, tokens written back)."""
    out, _dispatched, sent, wrote = global_router.dispatch(
        provider_id=cand["provider"], model_id=cand["model"], base_url=cand["base_url"],
        key=cand.get("key", ""), prompt=prompt, schema=schema, allow_fallback=False)
    return out, sent, wrote


# ---------------------------------------------------------------- what went wrong, in plain words

_MODEL_IN_ERROR = re.compile(r"(?:models?/|models?[\s:=]+[\"']?|[\"'])([A-Za-z][A-Za-z0-9][\w.\-:/@+]{1,120})")
_NOT_A_MODEL = {"not", "the", "is", "for", "and", "was", "are", "with", "from", "that", "this", "you", "your",
                "has", "have", "been", "only", "does", "found", "exist", "valid", "error", "message", "code"}


def _model_in(text: str) -> str:
    """The model name a service complained about, when it named one. Worth more than guessing from the settings:
    with a pool of models the wrong one would otherwise be blamed."""
    for m in _MODEL_IN_ERROR.finditer(text or ""):
        got = m.group(1).strip(" .,;:'\"()")
        if len(got) >= 3 and got.lower() not in _NOT_A_MODEL:
            return got
    return ""


def _first_model(p_id: str) -> str:
    """The first model written down for a service: the primary's, or the backup's when that is what was asked."""
    if p_id and p_id != provider():
        fb = fallback()
        if fb.get("provider") == p_id:
            return (fb.get("models") or [""])[0]
    return (models() or [""])[0]


def explain(e, prov=None, mdl=None) -> str:
    """What went wrong with an AI call, in words the owner can act on. Empty when the fault is not one we
    understand: a confident guess is worse than the service's own message, which the caller then shows."""
    text, low = str(e), str(e).lower()
    p_id = prov or provider()
    p_label = PROVIDERS.get(p_id, ("",))[0] or str(p_id).title()
    if "404" in text or "not found" in low or "does not exist" in low or "unknown model" in low or "not_found" in low:
        name = _model_in(text) or mdl or _first_model(p_id)
        return (f'{p_label} doesn\'t know the model "{name}". Press "Find models" in Settings, AI '
                "and pick one from the list.")
    if ("401" in text or "403" in text or "api key" in low or "api_key" in low
            or "unauthorized" in low or "permission" in low):
        return f"{p_label} refused the key. Check your {p_label} API key under Settings, AI."
    if "429" in text or "quota" in low or "rate limit" in low or "exhausted" in low:
        return (f"{p_label} says this key has hit its limit or has no credit left. "
                f"Add another {p_label} key under Settings, AI, or try again later.")
    if "connect" in low or "timed out" in low or "timeout" in low or "getaddrinfo" in low or "refused" in low:
        return (f"Autoclipper couldn't reach {p_label}. Check the internet connection"
                + (" and that Ollama is running." if p_id == "ollama" else "."))
    return ""


def note_problem(e, prov=None, mdl=None):
    """Keep why the AI last failed, for the dashboard. `prov` says which service to blame: without it the primary
    is named, which reads wrong when it was the backup that gave up."""
    p_id = prov or provider()
    p_label = PROVIDERS.get(p_id, ("",))[0] or str(p_id).title()
    why = explain(e, prov=prov, mdl=mdl) or (f"{p_label}: {str(e)[:160]}" if e is not None and str(e) else "")
    if why:
        problem.update(text=why, at=time.time(), is_fallback=False, fallback_provider="")


def note_working(cand, tried: int = 0):
    """The AI answered. Remember what answered, and say so on the dashboard when it was not the first choice:
    the owner should be able to see that a key ran out even though the clips kept coming."""
    last_used.update(model=cand.get("model", ""), provider=cand.get("provider", ""), label=cand.get("label", ""),
                     backup=bool(cand.get("backup")), key_label=cand.get("key_label", ""),
                     pool_size=cand.get("pool_size", 1), at=time.time())
    if not tried and not cand.get("backup"):
        clear_problem()
        return
    if cand.get("backup"):
        problem.update(text=f"{label()} hit its limit, so {cand.get('label') or 'the backup service'} took over. "
                            "Clips are still being made.",
                       at=time.time(), is_fallback=True, fallback_provider=cand.get("label", ""))
    else:
        problem.update(text=f"Moved to {describe(cand)}: the key or model before it hit its limit.",
                       at=time.time(), is_fallback=True, fallback_provider="")


def note_fallback(primary_name: str, fallback_name: str, reason: str = ""):
    """Kept for older callers: the backup service took over from the primary one."""
    problem.update(text=f"{primary_name} hit its rate limit or quota. Switched automatically to {fallback_name}.",
                   at=time.time(), is_fallback=True, fallback_provider=fallback_name)


def clear_problem():
    problem.update(text="", at=0.0, is_fallback=False, fallback_provider="")


def _fail(r, model, k=None):
    """Turn a refused request into an AIError, with every key in the pool scrubbed out of the service's words:
    the key that failed is not always the first one, and none of them may reach a log or a browser."""
    text = r.text[:300]
    for secret in [k or "", *all_keys()]:
        if secret:
            text = text.replace(secret, "<key>")
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


# ---------------------------------------------------------------- deterministic model calls
def call(model: str, prompt: str, schema: dict):
    """Ask one model using primary AI settings in strict mode (no silent model substitution).
    Dispatches to the exact model ID and logs execution. Returns (parsed JSON, tokens sent, tokens written back)."""
    p = provider()
    out, dispatched_model, sent, wrote = global_router.dispatch(
        provider_id=p,
        model_id=model,
        base_url=base_url(),
        key=key(),
        prompt=prompt,
        schema=schema,
        allow_fallback=False,
    )
    return out, sent, wrote


def call_with_fallback(model: str, prompt: str, schema: dict):
    """Ask primary model, with opt-in fallback if configured and rate-limited.
    Returns (parsed JSON, dispatched_model, sent, wrote)."""
    p = provider()
    fb = fallback() if fallback_ready() else None
    return global_router.dispatch(
        provider_id=p,
        model_id=model,
        base_url=base_url(),
        key=key(),
        prompt=prompt,
        schema=schema,
        allow_fallback=bool(fb),
        fallback_config=fb,
    )


def get_registry(provider_id: str = None) -> list:
    """Return normalized model records from the registry."""
    return global_registry.list_models(provider_id)


def get_logs(limit: int = 50) -> list:
    """Return recent execution logs."""
    return global_router.get_logs(limit)


def refresh_registry(provider_id: str = None) -> list:
    """Force-refresh dynamic model discovery for a provider or current provider."""
    p = ALIASES.get(str(provider_id or provider()).lower(), str(provider_id or provider()).lower())
    c = current()
    base = c["base_url"] if c["provider"] == p else PROVIDERS.get(p, ("", ""))[1]
    k = c["key"] if c["provider"] == p else ""
    return global_registry.discover(p, base, k, force=True)
