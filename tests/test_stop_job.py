"""Tests for stopping video generation and keeping existing clips intact."""
import os, sys, tempfile
os.environ["AUTOCLIPPER_DATA"] = tempfile.mkdtemp(prefix="pc-stop-")
os.environ["DATABASE_URL"] = ""
sys.path.insert(0, ".")
from pathlib import Path
from studio import auth, config, ctx, db, jobs

db.init()
fails = []

def check(name, ok, extra=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  {extra}" if extra else ""))
    if not ok:
        fails.append(name)

u1, _ = auth._create("u1@example.com", "U1", password_hash="x")
u2, _ = auth._create("u2@example.com", "U2", password_hash="x")

with ctx.as_user(u1.id):
    # 1. Stopping a running video sets stop_requested and marks stage
    j = jobs.create({"type": "file", "path": "", "name": "test.mp4"}, {}, enqueue=False)
    j["status"] = "running"
    j["clips"].append({"id": "clip01", "status": "done", "title": "First clip", "start": 0, "end": 10})
    jobs.save(j)

    ok = jobs.stop(j["id"])
    saved = jobs.get(j["id"])
    check("stop() returns True on running job", ok)
    check("stop_requested is recorded on job", saved.get("stop_requested") is True)
    check("existing clips remain intact", len(saved["clips"]) == 1 and saved["clips"][0]["title"] == "First clip")

    # 2. Stopping an already finished job returns False
    j_done = jobs.create({"type": "file", "path": "", "name": "done.mp4"}, {}, enqueue=False)
    j_done["status"] = "done"
    jobs.save(j_done)
    check("stop() returns False on finished job", not jobs.stop(j_done["id"]))

    # 3. Stopping a queued job immediately marks it done
    j_queued = jobs.create({"type": "file", "path": "", "name": "queued.mp4"}, {}, enqueue=False)
    j_queued["status"] = "queued"
    jobs.save(j_queued)
    check("stop() on queued job succeeds", jobs.stop(j_queued["id"]))
    saved_q = jobs.get(j_queued["id"])
    check("queued job marked done when stopped", saved_q["status"] == "done" and saved_q.get("progress") == 1.0)

with ctx.as_user(u2.id):
    # 4. Another user cannot stop my video
    check("another user cannot stop my video", not jobs.stop(j["id"]))

print("ALL PASSED" if not fails else f"{len(fails)} failed")
import shutil
shutil.rmtree(os.environ["AUTOCLIPPER_DATA"], ignore_errors=True)
sys.exit(1 if fails else 0)
