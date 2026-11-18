"""Tests for Creator Campaign Brief and AI guidelines."""
import os, sys, tempfile
os.environ["AUTOCLIPPER_DATA"] = tempfile.mkdtemp(prefix="pc-brief-")
os.environ["DATABASE_URL"] = ""
sys.path.insert(0, ".")
from studio import auth, config, ctx, db, creators, posting, brain

db.init()
fails = []

def check(name, ok, extra=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  {extra}" if extra else ""))
    if not ok:
        fails.append(name)

u1, _ = auth._create("brief_user@example.com", "BriefUser", password_hash="x")

with ctx.as_user(u1.id):
    # 1. Creator retains campaign_brief
    sample_brief = """Tomo GTA VI Clipping Campaign
CTA Requirement: every clip must mention Tomo and say 'subscribe to Tomo on YouTube'.
Hashtags: #tomo #tomovids #gta6 #gtavi #gaming
Priority format: funny moments and first reactions to GTA VI."""

    c = creators.create({"name": "Tomo", "campaign_brief": sample_brief})
    saved = creators.get(c["id"])
    check("creator saves campaign_brief", bool(saved.get("campaign_brief")), "brief present")
    check("campaign_brief content matches", "Tomo GTA VI" in saved["campaign_brief"])

    # 2. _tags_list in posting extracts hashtags from campaign_brief
    clip = {"hashtags": ["funnymoments", "viral"]}
    tags = posting._tags_list(clip, saved)
    check("hashtags extracted from campaign_brief", "#tomovids" in tags and "#gta6" in tags and "#gtavi" in tags, str(tags))
    check("clip tags still merged", "#viral" in tags and "#funnymoments" in tags, str(tags))

    # 3. Clearing campaign_brief works
    c_updated = creators.update(c["id"], {"campaign_brief": ""})
    check("clearing campaign_brief succeeds", c_updated.get("campaign_brief") == "")

print("ALL PASSED" if not fails else f"{len(fails)} failed")
import shutil
shutil.rmtree(os.environ["AUTOCLIPPER_DATA"], ignore_errors=True)
sys.exit(1 if fails else 0)
