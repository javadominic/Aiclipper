"""The speech-to-text model: downloaded once, by itself, the first time it is needed.

Nothing has to be run by hand. (python -m studio.fetch_model still works, for getting it ahead of time.) It is
about 480 MB. It is unpacked next to where it will live and moved into place only when it is whole, so a download
that was cut off can never leave half a model behind."""
import shutil
import tarfile
import threading
import urllib.request

from . import config

_lock = threading.Lock()
state = {"doing": False, "percent": 0, "error": ""}
# Each model is checked and fetched the same way, just a different name, folder and release asset.
_MULTI_LOCK = threading.Lock()
multi_state = {"doing": False, "percent": 0, "error": ""}


def _url(name: str) -> str:
    return f"https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/{name}.tar.bz2"


def ready() -> bool:
    """Is the whole model there: the word list and all three parts?"""
    return _model_ready(config.ASR_MODEL_DIR)


def multi_ready() -> bool:
    """Is the multilingual (Whisper) model there: its encoder, decoder and token list?"""
    d = config.ASR_MULTI_MODEL_DIR
    if not (d / "tokens.txt").is_file():
        return False
    return any(d.glob("*encoder*.onnx")) and any(d.glob("*decoder*.onnx"))


def _model_ready(d) -> bool:
    if not (d / "tokens.txt").is_file():
        return False
    return all(any(d.glob(f"{part}*.onnx")) for part in ("encoder", "decoder", "joiner"))


def _progress(blocks, size, total, into: dict):
    if total > 0:
        into["percent"] = min(99, int(blocks * size * 100 / total))


def _fetch(name: str, dest, lock, into: dict, label: str) -> None:
    """Download and unpack one model by its release name into `dest`, the same way for any of them."""
    with lock:
        into.update(doing=True, percent=0, error="")
        print(f"\n  Downloading the {label} (one time)...\n", flush=True)
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        tmp = config.MODELS_DIR / f"{name}.tar.bz2"
        part = config.MODELS_DIR / f"{name}.unpacking"
        try:
            urllib.request.urlretrieve(_url(name), tmp, lambda b, s, t: _progress(b, s, t, into))
            shutil.rmtree(part, ignore_errors=True)
            part.mkdir()
            with tarfile.open(tmp, "r:bz2") as t:
                for m in t.getmembers():  # only plain files and folders that stay inside the unpacking folder
                    if (m.isfile() or m.isdir()) and not m.name.startswith(("/", "..")) and ".." not in m.name.split("/"):
                        t.extract(m, part)
            got = part / name
            if not got.is_dir():
                raise OSError("the download did not contain the model")
            shutil.rmtree(dest, ignore_errors=True)        # a half-finished one from before
            got.rename(dest)
            print(f"  {label.capitalize()} ready.\n", flush=True)
        except Exception as e:  # noqa: BLE001
            into["error"] = str(e)[-200:]
            raise RuntimeError(f"Autoclipper couldn't download its {label}. Check the internet connection and "
                               "free disk space, then try again.") from e
        finally:
            into["doing"] = False
            tmp.unlink(missing_ok=True)
            shutil.rmtree(part, ignore_errors=True)


def ensure() -> None:
    """Make sure the (English) speech model is on this computer, downloading it if it is not."""
    if ready():
        return
    with _lock:
        if ready():
            return
    _fetch(config.ASR_MODEL_NAME, config.ASR_MODEL_DIR, _lock, state, "speech model (about 480 MB)")


def ensure_multi() -> None:
    """Make sure the multilingual speech model is on this computer. Only downloaded the first time a
    non-English video needs it, so an English-only setup never pays for it."""
    if multi_ready():
        return
    with _MULTI_LOCK:
        if multi_ready():
            return
    _fetch(config.ASR_MULTI_MODEL_NAME, config.ASR_MULTI_MODEL_DIR, _MULTI_LOCK, multi_state,
          "multilingual speech model (for Hindi, Urdu and other languages)")


def main():
    if ready():
        print("Speech model already present.")
    else:
        ensure()
    if multi_ready():
        print("Multilingual speech model already present.")
    else:
        ensure_multi()


if __name__ == "__main__":
    main()
