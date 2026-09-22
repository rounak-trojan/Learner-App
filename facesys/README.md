# Face Recognition Attendance System

Single-command demo/reference implementation. Run `python3 app.py`, open `http://localhost:5000`.

## What's actually inside

- **Detection**: MTCNN (`facenet-pytorch`)
- **Embedding**: InceptionResnetV1 pretrained on VGGFace2 — 512-d vector per face (`facenet-pytorch`)
- **Matching**: cosine similarity, in-memory index rebuilt **per center**, so a scan at one center
  can only ever be compared against that center's own learners — never the whole repository.
- **Storage**: SQLite (`db/attendance.db`, created automatically on first run). Embeddings are
  stored as raw float32 blobs — no images or video are kept after enrollment processing.
- Nothing here is "pixel-based" matching on purpose — see the note in the original design
  discussion for why raw pixel comparison doesn't hold up in production.

This was built and tested end-to-end in a sandboxed environment (enrollment → embedding →
center-scoped scan → dedupe → attendance log → CSV export → center-isolation check all verified
against a real face photo, not a mock).

## Setup

```bash
pip install -r requirements.txt
python3 app.py
```

First run downloads the pretrained VGGFace2 weights (~107MB, cached by torch under
`~/.cache/torch` afterward — only happens once).

Default admin login: `admin` / `admin123` (override with `ADMIN_USERNAME` / `ADMIN_PASSWORD`
environment variables before running — don't ship the default in production).

## Flow

1. **Admin** (`/admin/login`) adds a **Center** (city, centre name, centre UID, Syncup email —
   the email becomes that center's login).
2. **Admin** adds a **Learner** (name, LUID, roll no, center, video file). On submit, the video is
   processed immediately: frames are sampled, faces detected, blurry/low-confidence frames
   discarded, up to 6 diverse embeddings extracted (farthest-point sampling across the accepted
   frames, so you get pose/lighting variety instead of 6 near-duplicate frames), and stored. The
   video itself is deleted right after — set `RETAIN_ENROLLMENT_VIDEO = True` in `config.py` if
   you need to keep it (mind biometric-data retention law if you do).
3. **Center** (`/center/login`) logs in with its Syncup email. This scopes everything that
   follows to that center only.
4. **Scanner** (`/center/scanner`) opens the webcam, auto-captures a frame every ~2.5s, posts it
   to `/api/scan`. Match found + not already marked today → attendance logged. Already marked
   today → told so, not double-logged. No match → told so, nothing written except an audit row in
   `scan_attempts` (kept for debugging false rejections/accepts later).
5. **Attendance log** (`/center/attendance`) — filter by date range / name / roll no, export CSV.

## Where this deliberately cuts scope vs. a real deployment

- **Google Drive ingestion**: the brief said learner videos live in Drive. Wiring that in cleanly
  needs a service-account credential and OAuth setup that can't just run out of the box, so this
  build takes a direct video upload from the admin's machine instead. To add Drive: pull the file
  with the Drive API into `config.VIDEO_TMP_DIR` before calling
  `face_engine.process_enrollment_video()` in the `add_learner` route — everything downstream is
  already decoupled from where the file came from.
- **Synchronous processing**: video processing runs in the request thread. Fine for onboarding a
  handful of learners at a time; for bulk enrollment, move `process_enrollment_video()` behind a
  Celery/RQ queue so the admin isn't stuck on a spinner for a long upload.
- **No real liveness/anti-spoof model**: there's a blur/quality gate (Laplacian variance) at both
  enrollment and scan time, which blocks garbage frames and very low-effort spoofing, but it's not
  a substitute for an actual anti-spoofing model (e.g. Silent-Face-Anti-Spoofing) if this needs to
  resist printed photos or phone-screen replays.
- **Single dev server**: `app.run(debug=True)` is Flask's dev server. For anything beyond a demo,
  run it behind gunicorn/uwsgi + nginx, and move SQLite to Postgres once you're past a
  few-thousand-learner scale (SQLite's fine well past that for this workload, but Postgres gives
  you proper concurrent writes across centers).
- **No consent/retention workflow**: biometric data — the embeddings — is regulated (DPDP Act in
  India, GDPR-equivalent elsewhere). This build has no consent capture or deletion-request flow;
  add one before this touches real learners.

## File map

```
app.py            Flask routes / entry point — python3 app.py starts everything
config.py         Tunables: match threshold, blur threshold, admin creds, paths
db.py             SQLite schema + all queries
face_engine.py    Detection, embedding, enrollment video processing, per-center matcher
templates/        Jinja2 pages
static/css,js/    Styling + scanner webcam capture logic
db/               attendance.db lives here (created on first run)
static/uploads/   Enrollment videos land here briefly during processing
```

## Tuning

`config.MATCH_THRESHOLD` (default 0.62, cosine similarity) is the main knob. Lower it and you'll
get more matches but more false accepts; raise it and you'll get fewer false accepts but more
learners rejected on a bad-lighting day. Tune it against your actual enrollment videos, not a
guess — pull a week of `scan_attempts` rows and look at the confidence distribution of true
matches vs. mismatches before moving the number.
