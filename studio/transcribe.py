"""Speech to text with word timestamps, using Parakeet 0.6b through sherpa-onnx.

It runs in one of two places. On this machine's processors (the default), or, when AUTOCLIPPER_ASR_URL is set, on a
separate machine with a graphics card (studio/asr_worker.py), which does the same work many times faster. If that
machine cannot be reached, the pieces are done here instead, so transcription never stops because of it."""
import os
import queue
import threading
import time

import numpy as np

from . import config

_lock = threading.Lock()

# Several copies of the speech model, each on a few cores, instead of one copy on all of them: one copy cannot keep
# a big machine busy, and several videos transcribing at once would queue behind each other inside it. On a small
# machine (under 8 cores) this is still a single copy, as before.
_CPUS = os.cpu_count() or 4
PROVIDER = os.environ.get("AUTOCLIPPER_ASR_PROVIDER", "cpu").strip().lower() or "cpu"      # "cuda" on a graphics card
THREADS = int(os.environ.get("AUTOCLIPPER_ASR_THREADS") or (4 if _CPUS >= 8 else max(2, _CPUS - 2)))
COPIES = int(os.environ.get("AUTOCLIPPER_ASR_COPIES") or (1 if PROVIDER != "cpu" else max(1, _CPUS // THREADS) if _CPUS >= 8 else 1))
_free: "queue.LifoQueue" = queue.LifoQueue()
_made = 0

# The machine with the graphics card, if there is one.
REMOTE = os.environ.get("AUTOCLIPPER_ASR_URL", "").strip().rstrip("/")
REMOTE_TOKEN = os.environ.get("AUTOCLIPPER_ASR_TOKEN", "").strip()
REMOTE_BATCH = int(os.environ.get("AUTOCLIPPER_ASR_REMOTE_BATCH", "8"))         # pieces sent together: the card does them at once
REMOTE_PARALLEL = int(os.environ.get("AUTOCLIPPER_ASR_REMOTE_PARALLEL", "4"))   # pieces of one video worked on at once
# The card has a ceiling: beyond a few requests at once it gets no faster. So only this many are sent to it at a
# time (all videos together), and the rest is done on this machine's own processors at the same moment: both are used.
REMOTE_SLOTS = threading.BoundedSemaphore(max(1, int(os.environ.get("AUTOCLIPPER_ASR_REMOTE_INFLIGHT", "6"))))
_remote_down_until = 0.0


def model_ready() -> bool:
    from . import fetch_model
    return fetch_model.ready()


def multi_model_ready() -> bool:
    """Is the multilingual (Whisper) speech model present, for Hindi, Urdu and other non-English videos?"""
    from . import fetch_model
    return fetch_model.multi_ready()


def detect_lang(audio: np.ndarray) -> str:
    """A best-effort spoken-language guess (e.g. "hi") from a short clip of audio, using Whisper's own
    language detector. Returns "" if the multilingual model is not set up or detection is unsupported,
    so callers fall back to whatever they already know (or to the English model)."""
    if len(audio) < 16000:
        return ""
    try:
        rec = _multi_recognizer()
        st = rec.create_stream()
        st.accept_waveform(16000, audio.astype(np.float32) / 32768.0)
        rec.decode_stream(st)
        lang = getattr(st.result, "lang", "") or ""
        return lang.strip().lower()
    except Exception:  # noqa: BLE001
        return ""


def _model_files(d) -> dict:
    """The model's three parts. A graphics card wants the full-precision files; processors the compact ones."""
    kinds = ("fp16", "", "int8") if PROVIDER != "cpu" else ("int8", "", "fp16")
    for k in kinds:
        names = {p: d / (f"{p}.{k}.onnx" if k else f"{p}.onnx") for p in ("encoder", "decoder", "joiner")}
        if all(f.is_file() for f in names.values()):
            return {p: str(f) for p, f in names.items()}
    raise FileNotFoundError(f"No speech model files in {d}")


def _whisper_files(d) -> dict:
    """Whisper's two parts (encoder, decoder), compact build first."""
    for enc in sorted(d.glob("*encoder*.int8.onnx")) + sorted(d.glob("*encoder*.onnx")):
        tag = enc.name.split("encoder")[0]
        dec = next(iter(sorted(d.glob(f"{tag}decoder*.int8.onnx")) or sorted(d.glob(f"{tag}decoder*.onnx"))), None)
        if dec:
            return {"encoder": str(enc), "decoder": str(dec)}
    raise FileNotFoundError(f"No Whisper model files in {d}")


def _new():
    import sherpa_onnx
    from . import fetch_model
    fetch_model.ensure()             # downloaded here, once, if it was never fetched
    d = config.ASR_MODEL_DIR
    return sherpa_onnx.OfflineRecognizer.from_transducer(
        **_model_files(d),
        tokens=str(d / "tokens.txt"),
        num_threads=THREADS,
        sample_rate=16000,
        feature_dim=80,
        decoding_method="greedy_search",
        model_type="nemo_transducer",
        provider=PROVIDER,
    )


_multi_lock = threading.Lock()
_multi_rec = None


def _new_multi():
    """A multilingual (Whisper) recognizer: Hindi, Urdu and 90+ other languages. Built once and reused: it is
    only asked for when a video's language is not English, which is uncommon enough that one copy is plenty."""
    import sherpa_onnx
    from . import fetch_model
    fetch_model.ensure_multi()
    d = config.ASR_MULTI_MODEL_DIR
    return sherpa_onnx.OfflineRecognizer.from_whisper(
        **_whisper_files(d),
        tokens=str(d / "tokens.txt"),
        num_threads=THREADS,
        decoding_method="greedy_search",
        provider=PROVIDER,
    )


def _multi_recognizer():
    """One shared copy of the multilingual model, made the first time it is needed."""
    global _multi_rec
    if _multi_rec is None:
        with _multi_lock:
            if _multi_rec is None:
                _multi_rec = _new_multi()
    return _multi_rec


def _take():
    """A free copy of the model: one that is idle, a new one while there is room for more, else wait for one."""
    global _made
    try:
        return _free.get_nowait()
    except queue.Empty:
        pass
    with _lock:
        make = _made < COPIES
        if make:
            _made += 1
    if make:
        try:
            return _new()
        except Exception:
            with _lock:
                _made -= 1
            raise
    return _free.get()


def _take_now():
    """A copy of the model if one is free (or can still be made) this instant, else None."""
    global _made
    try:
        return _free.get_nowait()
    except queue.Empty:
        pass
    with _lock:
        make = _made < COPIES
        if make:
            _made += 1
    if not make:
        return None
    try:
        return _new()
    except Exception:
        with _lock:
            _made -= 1
        raise


def _recognizer():
    """Kept for callers that only need the model loaded (warm-up)."""
    rec = _take()
    _free.put(rec)
    return rec


def _chunks(audio: np.ndarray, lo=12.0, hi=24.0):
    """Split at the quietest point between lo and hi seconds so words are not cut in half."""
    sr, hop = 16000, 800  # 50 ms frames
    n = len(audio)
    frames = n // hop
    if frames == 0:
        return
    x = audio[: frames * hop].reshape(frames, hop).astype(np.float32)
    rms = np.sqrt((x ** 2).mean(axis=1))
    pos = 0
    while pos < n:
        if n - pos <= hi * sr:
            yield pos, n
            return
        a, b = (pos + int(lo * sr)) // hop, (pos + int(hi * sr)) // hop
        cut = (a + int(np.argmin(rms[a:b]))) * hop + hop // 2
        yield pos, cut
        pos = cut


BATCH = int(os.environ.get("AUTOCLIPPER_ASR_BATCH", "1"))   # pieces decoded together. Measured: no faster on CPU, since
# each piece already uses every core, so one at a time (which also keeps the words exactly as before).


def _words(tokens, timestamps, a: int, b: int) -> list:
    out = []
    off, seg_end = a / 16000.0, b / 16000.0
    cur, gap = None, False
    for tok, t in zip(tokens, timestamps):
        new_word = tok.startswith((" ", "▁")) or gap
        piece = tok.replace("▁", " ").strip()
        if not piece:
            gap = True      # a space on its own: whatever comes next starts a new word ("in" "120", not "in120")
            continue
        gap = False
        is_punct = all(not c.isalnum() for c in piece)
        if cur is None or (new_word and not is_punct):
            if cur:
                out.append(cur)
            cur = {"w": piece, "s": round(off + t, 2), "e": round(off + t + 0.12, 2)}
        else:
            cur["w"] += piece
            cur["e"] = round(off + t + 0.12, 2)
    if cur:
        cur["e"] = min(cur["e"], round(seg_end, 2))
        out.append(cur)
    return out


def decode_pieces(pieces: list, rec=None) -> list:
    """[(tokens, timestamps), ...] for float32 pieces of audio, done on this machine with one copy of the model."""
    streams = []
    rec = rec or _take()       # held only for these pieces, so videos transcribing at once take turns fairly
    try:
        for seg in pieces:
            st = rec.create_stream()
            st.accept_waveform(16000, seg)
            streams.append(st)
        rec.decode_streams(streams)
    finally:
        _free.put(rec)
    return [(list(st.result.tokens), [float(t) for t in st.result.timestamps]) for st in streams]


def _remote_ok() -> bool:
    return bool(REMOTE) and time.time() >= _remote_down_until


def _decode_remote(pieces: list) -> list:
    """The same, on the machine with the graphics card. Audio goes as 16-bit sound, the answer comes back as text."""
    import httpx
    raw = [np.clip(seg * 32768.0, -32768, 32767).astype("<i2") for seg in pieces]
    r = httpx.post(REMOTE + "/decode", content=b"".join(x.tobytes() for x in raw), timeout=httpx.Timeout(120, connect=4),
                   headers={"Authorization": "Bearer " + REMOTE_TOKEN, "X-Lens": ",".join(str(len(x)) for x in raw),
                            "Content-Type": "application/octet-stream"})
    r.raise_for_status()
    got = r.json()["pieces"]
    if len(got) != len(pieces):
        raise ValueError("wrong number of pieces came back")
    return [(p["tokens"], p["timestamps"]) for p in got]


def _decode_multi(pieces: list) -> list:
    """[(tokens, timestamps), ...] for float32 pieces of audio, using the shared multilingual model.
    Whisper in sherpa-onnx does not hand back per-word timestamps the way the transducer does, so each
    piece's words are spread evenly across the piece: good enough for caption snapping on non-English audio."""
    rec = _multi_recognizer()
    out = []
    for seg in pieces:
        st = rec.create_stream()
        st.accept_waveform(16000, seg)
        rec.decode_stream(st)
        text = (st.result.text or "").strip()
        toks = text.split()
        n = len(seg) / 16000.0
        step = n / max(1, len(toks))
        out.append((toks, [round(i * step, 2) for i in range(len(toks))]))
    return out


def transcribe(audio: np.ndarray, on_progress=None, lang: str = "") -> list:
    """Returns [{'w': word, 's': start_sec, 'e': end_sec}, ...] for the whole audio.
    Speech is cut at quiet moments into 12-24 s pieces; silent pieces are skipped.
    `lang` is the spoken language if already known (e.g. from YouTube metadata): anything other than English
    is sent to the multilingual model instead of the English-only one."""
    global _remote_down_until
    multi = bool(lang) and lang.lower() not in config.ASR_ENGLISH_LANGS
    words = []
    todo = []
    for a, b in _chunks(audio):
        seg = audio[a:b].astype(np.float32) / 32768.0
        if len(seg) < 4800 or float(np.sqrt((seg ** 2).mean())) < 0.003:
            continue  # silence: skip, saves time and avoids made-up words
        todo.append((a, b, seg))
    remote = _remote_ok()
    size = max(1, REMOTE_BATCH if remote else BATCH)
    groups = [todo[i:i + size] for i in range(0, len(todo), size)]
    done = [0]

    def decode(group):
        global _remote_down_until
        segs = [seg for _, _, seg in group]
        if multi:                      # non-English: always the multilingual model, no remote card involved
            got = _decode_multi(segs)
            out = [_words(tok, ts, a, b) for (a, b, _), (tok, ts) in zip(group, got)]
            done[0] += 1
            if on_progress:
                on_progress(done[0] / len(groups))
            return out
        got = None
        while got is None:
            # whichever is free first: a place on the card's machine, or a copy of the model on this one
            if remote and _remote_ok() and REMOTE_SLOTS.acquire(blocking=False):
                try:
                    got = _decode_remote(segs)
                except Exception:  # noqa: BLE001 - the card's machine is away or busy: these pieces are done here
                    _remote_down_until = time.time() + 60
                finally:
                    REMOTE_SLOTS.release()
                continue
            rec = _take_now()
            if rec is None and not (remote and _remote_ok()):
                rec = _take()                                # nothing else to wait for: wait for a copy here
            if rec is not None:
                got = []
                step = max(1, BATCH)
                try:
                    for i in range(0, len(segs), step):
                        got += decode_pieces(segs[i:i + step], rec)
                        rec = None if i + step >= len(segs) else _take()
                except Exception:
                    got = None
                    raise
                continue
            time.sleep(0.02)
        out = [_words(tok, ts, a, b) for (a, b, _), (tok, ts) in zip(group, got)]
        done[0] += 1
        if on_progress:
            on_progress(done[0] / len(groups))
        return out

    # Pieces are independent, so they go to every copy of the model that is free: a video alone on the server
    # uses all of it, and several videos share it piece by piece. The words come back in order.
    workers = REMOTE_PARALLEL if remote else COPIES
    if workers > 1 and len(groups) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(min(workers, len(groups)), thread_name_prefix="asr") as ex:
            results = list(ex.map(decode, groups))
    else:
        results = [decode(g) for g in groups]
    for out in results:
        for piece in out:
            for w in piece:
                if words and words[-1]["w"] == w["w"] and words[-1]["s"] == w["s"]:
                    continue  # the model sometimes says the same word twice at the same moment
                words.append(w)
    # A word lasts until the next one starts, capped so pauses stay pauses.
    for i, w in enumerate(words):
        nxt = words[i + 1]["s"] if i + 1 < len(words) else w["e"] + 0.4
        w["e"] = round(max(w["s"] + 0.08, min(nxt, max(w["e"], w["s"] + 0.6))), 2)
    return words
