"""The AI layer, with the network replaced: every kind of service is asked correctly and its answer is read
correctly. Nothing leaves this machine and no key is needed."""
import json
import os
import sys
import tempfile

os.environ["AUTOCLIPPER_DATA"] = tempfile.mkdtemp(prefix="pc-ai-")
os.environ["DATABASE_URL"] = ""
os.environ.update(AUTOCLIPPER_AI_PROVIDER="gemini", AUTOCLIPPER_AI_MODELS="model-a, model-b", AUTOCLIPPER_AI_KEY="test-key-123456", AUTOCLIPPER_AI_BASE_URL="",
                  AUTOCLIPPER_AI_FALLBACK_PROVIDER="", AUTOCLIPPER_AI_FALLBACK_MODELS="", AUTOCLIPPER_AI_FALLBACK_KEY="")
sys.path.insert(0, ".")
from studio import ai, auth, brain, config, ctx, db, usage  # noqa: E402

db.init()
fails = []


def check(name, ok, extra=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  {extra}" if extra and not ok else ""))
    ok or fails.append(name)


class Reply:
    def __init__(self, status, data):
        self.status_code, self._data, self.text = status, data, json.dumps(data)

    def json(self):
        return self._data


CLIP = {"start": 1, "end": 20, "score": 80, "hook": "WOW", "title": "t", "description": "d", "hashtags": ["a"], "reason": "r"}
seen = []


def fake_post(url, json=None, headers=None, timeout=None, **_k):  # noqa: A002
    seen.append((url, json, headers or {}))
    if url.endswith(":generateContent"):
        return Reply(200, {"candidates": [{"content": {"parts": [{"text": globals()["json_text"]([CLIP])}]}}],
                           "usageMetadata": {"promptTokenCount": 100, "totalTokenCount": 130}})
    if url.endswith("/chat/completions"):
        if "temperature" in json:                      # a model that refuses an extra setting: asked again without it
            return Reply(400, {"error": {"message": "temperature is not supported"}})
        return Reply(200, {"choices": [{"message": {"content": "```json\n" + globals()["json_text"]({"items": [CLIP]}) + "\n```"}}],
                           "usage": {"prompt_tokens": 90, "completion_tokens": 20}})
    if url.endswith("/messages"):
        return Reply(200, {"content": [{"type": "text", "text": "Here you go: " + globals()["json_text"]({"items": [CLIP]})}],
                           "usage": {"input_tokens": 70, "output_tokens": 10}})
    return Reply(404, {})


def json_text(x):
    return json.dumps(x)


ai.httpx.post = fake_post


def use(provider, base=""):
    config.AI_PROVIDER, config.AI_BASE_URL = provider, base
    seen.clear()


use("gemini")
out, sent, wrote = ai.call("model-a", "hello", brain.SCHEMA)
check("gemini: answer and tokens", out == [CLIP] and (sent, wrote) == (100, 30), str((out, sent, wrote)))
check("gemini: key in the header, schema passed as is", seen[0][2].get("x-goog-api-key") == "test-key-123456"
      and seen[0][1]["generationConfig"]["responseSchema"] is brain.SCHEMA and seen[0][0].endswith("/model-a:generateContent"))

for name, host in (("openai", "api.openai.com"), ("xai", "api.x.ai"), ("grok", "api.x.ai"), ("groq", "api.groq.com"), ("openrouter", "openrouter.ai")):
    use(name)
    out, sent, wrote = ai.call("model-a", "hello", brain.SCHEMA)
    check(f"{name}: answer unwrapped from a code fence, tokens read", out == [CLIP] and (sent, wrote) == (90, 20), str((out, sent, wrote)))
    check(f"{name}: right address, key as a bearer token", host in seen[0][0] and seen[0][2].get("Authorization") == "Bearer test-key-123456")
check("a refused extra setting is dropped and asked again", len(seen) == 2 and "temperature" not in seen[1][1])
check("the schema is told in the prompt for chat models", '"type": "array"' in seen[1][1]["messages"][0]["content"])

use("anthropic")
out, sent, wrote = ai.call("model-a", "hello", brain.SCHEMA)
check("anthropic: answer found inside a sentence, tokens read", out == [CLIP] and (sent, wrote) == (70, 10), str((out, sent, wrote)))
check("anthropic: key and version headers", seen[0][2].get("x-api-key") == "test-key-123456" and "anthropic-version" in seen[0][2])

use("custom", "http://127.0.0.1:1234/v1")
out, _s, _w = ai.call("model-a", "hello", brain.SCHEMA)
check("custom: uses the address from settings", seen[0][0] == "http://127.0.0.1:1234/v1/chat/completions" and out == [CLIP])

use("custom")
check("custom without an address is not ready", not ai.ready())
use("nonsense")
check("an unknown provider is not ready", not ai.ready())
use("ollama")
os.environ["AUTOCLIPPER_AI_KEY"] = ""
config._kept.clear()
check("ollama needs no key", ai.ready() and ai.label().startswith("Ollama"))
out, _s, _w = ai.call("model-a", "hello", brain.SCHEMA)
check("ollama: no key header is sent", "Authorization" not in seen[0][2] and out == [CLIP])
use("gemini")
check("gemini without a key is not ready", not ai.ready())
os.environ["AUTOCLIPPER_AI_KEY"] = "test-key-123456"
config._kept.clear()

# ---- the whole path: a call is made for a user and shows up in their usage, and only theirs
u1, _ = auth._create("ai-one@example.com", "One", password_hash="x")
u2, _ = auth._create("ai-two@example.com", "Two", password_hash="x")
words = [{"w": f"word{i}", "s": i * 0.5, "e": i * 0.5 + 0.4} for i in range(120)]
with ctx.as_user(u1.id):
    usage.set_context(None, "A test video")
    import numpy as np
    clips, info = brain.find_clips(words, np.zeros(60), 60.0, title="T", uploader="U")
    check("clips come back through the whole path", len(clips) == 1 and clips[0]["hook"] == "WOW" and info["model"] == "model-a", str((clips, info)))
    s = usage.summary({})
    check("the call is in this user's usage", s["calls_total"] == 1 and s["tokens_total"] == 130 and s["recent"][0]["where"] == "A test video", str(s))
    check("usage says which AI is set", s["ready"] and s["provider"] == "Gemini" and s["model"] == "model-a")
with ctx.as_user(u2.id):
    check("another user sees none of it", usage.summary({})["calls_total"] == 0)

# ---- a pool of keys: every key and model is tried before Autoclipper gives up
K1, K2 = "rotate-key-one-1111", "rotate-key-two-2222"
hit, out_of_quota = [], set()


def rotate_post(url, json=None, headers=None, timeout=None, **_k):  # noqa: A002
    k = (headers or {}).get("x-goog-api-key", "")
    mdl = url.split("/models/")[-1].split(":")[0]
    hit.append((k, mdl))
    if (k, mdl) in out_of_quota or (k, "*") in out_of_quota:
        return Reply(429, {"error": {"message": "You exceeded your current quota for this model."}})
    return Reply(200, {"candidates": [{"content": {"parts": [{"text": json_text([CLIP])}]}}],
                       "usageMetadata": {"promptTokenCount": 100, "totalTokenCount": 130}})


ai.httpx.post = rotate_post
config.AI_PROVIDER, config.AI_BASE_URL = "gemini", ""
config.AI_FALLBACK_PROVIDER, config.AI_FALLBACK2_PROVIDER = "", ""
ai.remove_fallback()
ai.remove_fallback2()
ai.save("gemini", "model-a, model-b", "", keys=[{"key": K1, "label": "one"}, {"key": K2, "label": "two"}])

# note: check() in this file takes a true/false, not a pair to compare, so every assertion below has to say
# "got == want" itself or it would pass without checking anything.
ladder = [(c["model"], c["key_label"]) for c in ai.candidates()]
check("the ladder is model-first: the best model on every key, then the next model",
      ladder == [("model-a", "one"), ("model-a", "two"), ("model-b", "one"), ("model-b", "two")], str(ladder))

out_of_quota = {(K1, "*")}
hit.clear()
with ctx.as_user(u1.id):
    _o, used, _t = brain.ask_ai("hello", purpose="rotate")
check("a key out of quota hands the SAME model to the next key",
      (used, [m for _k, m in hit]) == ("model-a (two)", ["model-a", "model-a"]), str((used, hit)))

hit.clear()
with ctx.as_user(u1.id):
    _o, used, _t = brain.ask_ai("hello", purpose="rotate")
check("the spent key is not asked again on the next call", [k for k, _m in hit] == [K2], str(hit))

out_of_quota = {(K1, "model-a"), (K2, "model-a")}
ai.reset_state()
hit.clear()
with ctx.as_user(u1.id):
    _o, used, _t = brain.ask_ai("hello", purpose="rotate")
check("with every key spent on the best model it drops to the next one",
      (used, [m for _k, m in hit]) == ("model-b (one)", ["model-a", "model-a", "model-b"]), str((used, hit)))

out_of_quota = {(K1, "*"), (K2, "*")}
ai.reset_state()
try:
    with ctx.as_user(u1.id):
        brain.ask_ai("hello", purpose="rotate")
    check("it gives up loudly", False)
except RuntimeError as e:
    check("it says the keys hit their limit, not that the AI is missing", "limit" in str(e), str(e))
check("nothing is left to try while every key rests", ai.candidates() == [], str(ai.candidates()))
check("...and it knows how long to wait before trying again", ai.all_resting() > 0, str(ai.all_resting()))
ai.reset_state()

print("ALL PASSED" if not fails else f"{len(fails)} failed")
import shutil  # noqa: E402
db.engine.dispose()
shutil.rmtree(os.environ["AUTOCLIPPER_DATA"], ignore_errors=True)
sys.exit(1 if fails else 0)
