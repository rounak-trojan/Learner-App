"""
Face Recognition Attendance System
Run with:  python3 app.py
Then open: http://localhost:5000
"""
import base64
import csv
import datetime
import io
import json
import os
import re
import threading
import uuid
from collections import defaultdict, OrderedDict
from functools import wraps

import cv2
import numpy as np
import requests
from flask import (
    Flask, render_template, request, redirect, url_for, session, jsonify, flash, Response
)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

import config
import db
import face_engine
import mailer

app = Flask(__name__)
app.secret_key = config.SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH
app.config["MAX_FORM_MEMORY_SIZE"] = config.MAX_FORM_MEMORY_SIZE

db.init_db()

# In-memory bulk-import job tracker (single-process dev server; jobs don't survive a
# restart). Keyed by job_id -> {"total", "processed", "results": [...], "done": bool}
BULK_JOBS = {}
BULK_JOBS_LOCK = threading.Lock()


# ---------------------------------------------------------------- helpers --

def admin_required(view):
    """Full-access admin only (the primary config-based login, or an admin_users
    row with role='full'). Restricted admins are bounced to their own overview
    dashboards rather than the full admin panel."""
    @wraps(view)
    def wrapped(*a, **kw):
        if not session.get("is_admin") or session.get("admin_role") != "full":
            return redirect(url_for("admin_login"))
        return view(*a, **kw)
    return wrapped


def admin_dashboard_required(view):
    """Any logged-in admin - full or restricted - can reach these (the read-only
    cross-center overview dashboards)."""
    @wraps(view)
    def wrapped(*a, **kw):
        if not session.get("is_admin"):
            return redirect(url_for("admin_login"))
        return view(*a, **kw)
    return wrapped


def center_required(view):
    @wraps(view)
    def wrapped(*a, **kw):
        if not session.get("center_uid"):
            return redirect(url_for("center_login"))
        return view(*a, **kw)
    return wrapped


def learner_required(view):
    @wraps(view)
    def wrapped(*a, **kw):
        if not session.get("learner_luid"):
            return redirect(url_for("index"))
        return view(*a, **kw)
    return wrapped


def allowed_video(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in config.ALLOWED_VIDEO_EXT


def decode_base64_image(data_url):
    """data_url like 'data:image/jpeg;base64,....' -> BGR numpy array"""
    header, encoded = data_url.split(",", 1)
    binary = base64.b64decode(encoded)
    arr = np.frombuffer(binary, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


_DRIVE_ID_PATTERNS = [
    re.compile(r"/file/d/([a-zA-Z0-9_-]+)"),
    re.compile(r"[?&]id=([a-zA-Z0-9_-]+)"),
    re.compile(r"/d/([a-zA-Z0-9_-]+)"),
]


def extract_drive_file_id(url_or_id):
    """Accepts a raw Drive file ID or any of the common shareable-link shapes
    ('.../file/d/<id>/view', '...?id=<id>', '.../d/<id>'). Returns the file ID or None."""
    url_or_id = (url_or_id or "").strip()
    if not url_or_id:
        return None
    if "/" not in url_or_id and "?" not in url_or_id:
        return url_or_id  # already looks like a bare file ID
    for pattern in _DRIVE_ID_PATTERNS:
        m = pattern.search(url_or_id)
        if m:
            return m.group(1)
    return None


def download_drive_video(url_or_id, dest_dir):
    """
    Fetch a video from Google Drive by shareable link or file ID and save it to dest_dir.
    Handles the 'file too big for an automatic virus scan' confirmation redirect Drive
    issues for larger files. Raises ValueError with a human-readable reason on failure.
    Only works for files shared as 'Anyone with the link' - Drive has no way to authenticate
    a service-less anonymous fetch against a private file.
    """
    file_id = extract_drive_file_id(url_or_id)
    if not file_id:
        raise ValueError(f"Could not parse a Google Drive file ID from: {url_or_id}")

    session_ = requests.Session()
    base = "https://drive.google.com/uc?export=download"
    resp = session_.get(base, params={"id": file_id}, stream=True, timeout=config.GDRIVE_DOWNLOAD_TIMEOUT)

    token = None
    for key, value in resp.cookies.items():
        if key.startswith("download_warning"):
            token = value
            break
    if token is None and "text/html" in resp.headers.get("Content-Type", ""):
        m = re.search(r"confirm=([0-9A-Za-z_-]+)", resp.text)
        if m:
            token = m.group(1)

    if token:
        resp = session_.get(
            base, params={"id": file_id, "confirm": token}, stream=True,
            timeout=config.GDRIVE_DOWNLOAD_TIMEOUT,
        )

    if resp.status_code != 200:
        raise ValueError(f"Drive returned HTTP {resp.status_code} for file ID {file_id}")

    content_type = resp.headers.get("Content-Type", "")
    if "text/html" in content_type:
        raise ValueError(
            f"File {file_id} did not return a video (likely not shared as "
            "'Anyone with the link', or the ID/link is wrong)"
        )

    ext = "mp4"
    disp = resp.headers.get("Content-Disposition", "")
    m = re.search(r'filename="?([^";]+)"?', disp)
    if m and "." in m.group(1):
        candidate_ext = m.group(1).rsplit(".", 1)[1].lower()
        if candidate_ext in config.ALLOWED_VIDEO_EXT:
            ext = candidate_ext

    dest_path = os.path.join(dest_dir, f"{uuid.uuid4().hex}_drive_{file_id}.{ext}")
    total = 0
    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if not chunk:
                continue
            total += len(chunk)
            if total > config.GDRIVE_MAX_BYTES:
                f.close()
                os.remove(dest_path)
                raise ValueError(f"Video for file ID {file_id} exceeds the {config.GDRIVE_MAX_BYTES // (1024*1024)}MB limit")
            f.write(chunk)

    if total == 0:
        os.remove(dest_path)
        raise ValueError(f"Downloaded 0 bytes for file ID {file_id} - link may be invalid or private")

    return dest_path


def process_face_source(video_file, drive_link, capture_frames_file, tmp_dir):
    """
    Resolve one of three enrollment data sources into a list of face embeddings:
      1. an uploaded video file
      2. a Google Drive link
      3. a JSON file of base64 JPEG frames from the live in-browser registration
         scanner (front/left/right guided capture) - sent as a real file part
         (not a form field) specifically so it isn't subject to Werkzeug's
         much smaller max_form_memory_size cap on plain text fields.
    Priority when more than one is supplied: file > drive link > live capture.

    Returns (embeddings, tmp_video_path_or_None, error_message_or_None). The caller
    is responsible for removing tmp_video_path once it's done with it (respecting
    config.RETAIN_ENROLLMENT_VIDEO) - live capture never creates a tmp file.
    """
    has_file = bool(video_file and video_file.filename)
    has_drive = bool(drive_link)
    has_capture = bool(capture_frames_file and capture_frames_file.filename)

    if not has_file and not has_drive and not has_capture:
        return None, None, "Provide a learner video, a Google Drive link, or use the live camera scanner"

    if has_file:
        if not allowed_video(video_file.filename):
            return None, None, f"Unsupported video format. Allowed: {', '.join(config.ALLOWED_VIDEO_EXT)}"
        tmp_name = f"{uuid.uuid4().hex}_{secure_filename(video_file.filename)}"
        tmp_path = os.path.join(tmp_dir, tmp_name)
        video_file.save(tmp_path)
        try:
            embeddings = face_engine.process_enrollment_video(tmp_path)
        except Exception as e:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            return None, None, f"Video processing failed: {e}"
        return embeddings, tmp_path, None

    if has_drive:
        try:
            tmp_path = download_drive_video(drive_link, tmp_dir)
        except ValueError as e:
            return None, None, f"Could not fetch the Drive video: {e}"
        try:
            embeddings = face_engine.process_enrollment_video(tmp_path)
        except Exception as e:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            return None, None, f"Video processing failed: {e}"
        return embeddings, tmp_path, None

    # Live camera registration: capture_frames_file is a small JSON file of
    # 'data:image/jpeg;base64,...' strings produced by the guided front/left/right scanner.
    try:
        raw = capture_frames_file.read()
        frames_data = json.loads(raw.decode("utf-8"))
    except Exception:
        return None, None, "Could not read the captured face data - run the scan again"
    if not isinstance(frames_data, list) or not frames_data:
        return None, None, "No frames were captured - run the face scan again"

    frames = []
    for data_url in frames_data[: config.CAPTURE_MAX_FRAMES]:
        try:
            frames.append(decode_base64_image(data_url))
        except Exception:
            continue
    if not frames:
        return None, None, "Could not decode the captured frames - run the face scan again"

    try:
        embeddings = face_engine.process_enrollment_frames(frames)
    except Exception as e:
        return None, None, f"Face processing failed: {e}"
    return embeddings, None, None


def center_smtp_kwargs(center_uid):
    """If this center has its own Syncup email + app password configured, route its
    attendance mail through that mailbox. Otherwise return {} so mailer falls back
    to the global SMTP account in config.py."""
    center = db.get_center(center_uid)
    if center and center["syncup_email"] and center["app_password"]:
        return {
            "smtp_username": center["syncup_email"],
            "smtp_password": center["app_password"],
            "from_email": center["syncup_email"],
            "from_name": center["centre_name"],
        }
    return {}


def decode_csv_bytes(raw_bytes):
    """Decode uploaded CSV bytes, detecting the encoding rather than assuming UTF-8.
    Windows/Excel 'Save As CSV' frequently writes UTF-16 (with a BOM) instead of
    UTF-8 - if that's fed straight through as UTF-8 every field ends up full of
    embedded NUL bytes, which silently corrupts the header names (so a perfectly
    valid file ends up with zero rows matching 'learner_uid', with no visible
    error). Falls back to cp1252 (the common Windows default) if UTF-8 fails."""
    if raw_bytes.startswith(b"\xff\xfe") or raw_bytes.startswith(b"\xfe\xff"):
        return raw_bytes.decode("utf-16")
    try:
        return raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw_bytes.decode("cp1252", errors="replace")


def sniff_delimiter(raw):
    """Detect comma vs semicolon vs tab delimiting (many non-US Excel locales export
    ';'-delimited 'CSV' files) rather than assuming a comma. Falls back to comma if
    detection is inconclusive."""
    sample = raw[:4096]
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t").delimiter
    except csv.Error:
        return ","


def read_csv_rows(file_storage, max_rows=None):
    """Parse an uploaded CSV FileStorage into a list of dicts, header-normalized
    (lowercase, stripped, spaces -> underscores), and a count of rows that had to be
    skipped because they were unparseable. max_rows=None (the default) means no cap -
    used for the Atlas/Test/NPS dumps, which can legitimately be very large. Pass
    max_rows=config.BULK_MAX_ROWS for the bulk center/learner enrollment importers,
    which still cap since each row triggers real enrollment work (video download +
    face processing), not just a database insert.

    Tolerant of the defects real-world exports commonly have, rather than aborting
    the whole upload (or silently returning zero rows) over them:
      - wrong text encoding (UTF-16 from Windows/Excel) and non-comma delimiters
        (semicolon from non-US Excel locales) - detected rather than assumed.
      - mixed/legacy line endings (bare \\r, \\r\\n, \\n mixed in one file) - normalized
        up front, which is what usually produces csv's "new-line character seen in
        unquoted field" on an otherwise valid file.
      - a stray unescaped comma inside a free-text column (review text, names, etc.)
        producing a row with more fields than the header - the overflow is folded
        back into the last column instead of csv.DictReader's default behavior of
        stashing it in a list under a None key, which crashes the caller.
    A row csv itself can't parse at all (unbalanced quotes, etc.) is skipped and
    counted rather than failing every row after it.
    """
    raw = decode_csv_bytes(file_storage.read())
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    delimiter = sniff_delimiter(raw)

    reader = csv.reader(io.StringIO(raw), delimiter=delimiter)
    try:
        header = next(reader)
    except StopIteration:
        return [], 0
    except csv.Error as e:
        raise ValueError(f"Could not read the header row: {e}")

    fieldnames = [(fn or "").strip().lower().replace(" ", "_") for fn in header]

    rows = []
    skipped = 0
    while True:
        try:
            raw_row = next(reader)
        except StopIteration:
            break
        except csv.Error:
            skipped += 1
            continue

        if not raw_row or all(not (c or "").strip() for c in raw_row):
            continue  # blank line

        clean = {}
        for i, key in enumerate(fieldnames):
            if not key:
                continue
            value = raw_row[i] if i < len(raw_row) else ""
            clean[key] = (value or "").strip()

        if len(raw_row) > len(fieldnames) and fieldnames:
            overflow = ",".join(c for c in raw_row[len(fieldnames):] if c)
            if overflow:
                last_key = fieldnames[-1]
                clean[last_key] = f"{clean.get(last_key, '')},{overflow}".strip(",").strip()

        rows.append(clean)
        if max_rows is not None and len(rows) > max_rows:
            raise ValueError(f"CSV has more than {max_rows} rows - split it into batches")

    return rows, skipped


# ---- Atlas / Test / NPS dump parsing (admin uploads) ----

ATLAS_FIELDS = ["batch_name", "centre_city_name", "centre_name", "goal_name",
                "learner_name", "learner_uid", "roll_number"]

TEST_FIELDS = ["learner_id", "learner_uid", "learner_name", "roll_no", "test_uid", "test_title",
               "test_start_date", "course_name", "course_uid", "test_score", "attempt_type",
               "rank_on_score", "learner_test_start_at", "learner_test_end_at", "learner_batch_uid",
               "learner_batch_name", "centre_name", "learner_center_uid", "center_city",
               "section_name", "total_ques_attempted", "section_score"]

# normalized-source-header -> db field name (source headers normalized the same way
# read_csv_rows does: lowercased, spaces -> underscores, everything else left as-is)
NPS_FIELD_MAP = {
    "date_of_feedback": "feedback_date",
    "city_name": "city_name",
    "centre_name": "centre_name",
    "category": "category",
    "version": "version",
    "learner_uid": "learner_uid",
    "learner_id": "learner_id",
    "learner_name": "learner_name",
    "learner_user_name": "learner_username",
    "state_name": "state_name",
    "first_feedback_flag_(in_goal)": "first_feedback_flag",
    "feedback_platform": "feedback_platform",
    "feedback_review": "feedback_review",
    "rating": "rating",
    "app_and_technology-_flag": "flag_app_tech",
    "class_recording_-_flag": "flag_class_recording",
    "staff_co-operation_-_flag": "flag_staff_cooperation",
    "study_material_-_flag": "flag_study_material",
    "tests_-_flag": "flag_tests",
    "quality_of_educators_-_flag": "flag_quality_educators",
    "syllabus_progress_-_flag": "flag_syllabus_progress",
    "centre_facilities_cleanliness_-_flag": "flag_centre_facilities",
}


def parse_atlas_rows(raw_rows):
    """Extract only the required Atlas columns, then de-duplicate per learner_uid:
    a learner_uid appearing more than once keeps only the row(s) with a non-blank
    batch_name (first one); a learner_uid appearing exactly once is kept as-is even
    if batch_name is blank."""
    extracted = []
    for row in raw_rows:
        learner_uid = row.get("learner_uid", "").strip()
        if not learner_uid:
            continue
        extracted.append({f: row.get(f, "").strip() for f in ATLAS_FIELDS} | {"learner_uid": learner_uid})

    groups = defaultdict(list)
    for r in extracted:
        groups[r["learner_uid"]].append(r)

    result = []
    for group in groups.values():
        if len(group) == 1:
            result.append(group[0])
        else:
            non_blank = [g for g in group if g["batch_name"]]
            result.append(non_blank[0] if non_blank else group[0])
    return result


def parse_test_rows(raw_rows):
    out = []
    for row in raw_rows:
        learner_uid = row.get("learner_uid", "").strip()
        if not learner_uid:
            continue
        out.append({f: row.get(f, "").strip() for f in TEST_FIELDS})
    return out


def parse_nps_rows(raw_rows):
    out = []
    for row in raw_rows:
        rec = {dest: row.get(src, "").strip() for src, dest in NPS_FIELD_MAP.items()}
        if not rec.get("learner_uid"):
            continue
        out.append(rec)
    return out


def parse_attendance_dump_rows(raw_rows):
    """Extract (learner_uid, date, log_in, log_out) from an Attendance Dump CSV
    (city_name, centre_name, centre_uid, learner_uid, roll_number, Day of dates,
    log_in_time, log_out_time). A row needs a learner_uid and a parseable date to be
    usable - log_in/log_out can each be blank (a learner who only punched one side)."""
    out = []
    unparseable_dates = 0
    for row in raw_rows:
        learner_uid = row.get("learner_uid", "").strip()
        if not learner_uid:
            continue
        date_obj = db.parse_dump_date(row.get("day_of_dates", ""))
        if not date_obj:
            unparseable_dates += 1
            continue
        out.append({
            "learner_uid": learner_uid,
            "date": date_obj,
            "log_in": db.parse_dump_datetime(row.get("log_in_time", "")),
            "log_out": db.parse_dump_datetime(row.get("log_out_time", "")),
        })
    return out, unparseable_dates


def pivot_test_records(rows):
    """Group flat test_records rows by test_uid into one row per test, laying each
    row's Section Name/Section Score out as its own column (horizontally) with a
    total across all of that test's sections."""
    grouped = OrderedDict()
    for r in rows:
        key = r["test_uid"] or f"row{r['id']}"
        if key not in grouped:
            grouped[key] = {
                "learner_name": r["learner_name"], "learner_uid": r["learner_uid"],
                "roll_no": r["roll_no"], "batch_name": r["learner_batch_name"],
                "test_title": r["test_title"], "test_start_date": r["test_start_date"],
                "attempt_type": r["attempt_type"], "sections": [], "total": 0.0,
            }
        section_name = (r["section_name"] or "").strip() or "Overall"
        score_raw = (r["section_score"] or "").strip()
        grouped[key]["sections"].append({"name": section_name, "score": score_raw or "-"})
        try:
            grouped[key]["total"] += float(score_raw)
        except (TypeError, ValueError):
            pass
    return list(grouped.values())


def center_test_ranking(center_uid):
    """Rank a center's learners by average Test Score (numeric, attempted tests only)
    and bucket them into Top / Medium / Low tertiles - 'who is doing good to bad'."""
    rows = db.get_test_records_for_center(center_uid)
    scores = defaultdict(list)
    names = {}
    for r in rows:
        try:
            v = float(r["test_score"])
        except (TypeError, ValueError):
            continue
        scores[r["learner_uid"]].append(v)
        names[r["learner_uid"]] = r["learner_name"]

    ranking = []
    for luid, vals in scores.items():
        ranking.append({"learner_uid": luid, "name": names.get(luid, luid),
                        "avg_score": round(sum(vals) / len(vals), 2), "attempts": len(vals)})
    ranking.sort(key=lambda x: x["avg_score"], reverse=True)

    n = len(ranking)
    third = n / 3
    for i, r in enumerate(ranking):
        r["rank"] = i + 1
        r["tier"] = "Top" if (n <= 2 or i < third) else ("Medium" if i < 2 * third else "Low")
    return ranking


def no_rows_message(raw_rows, column_label):
    """A precise reason the parsed row-count came back empty, instead of a flat 'no
    rows found' that looks identical whether the file was empty, unreadable, or just
    used a different header spelling than expected."""
    if not raw_rows:
        return "The CSV had no data rows - check the file isn't empty."
    sample_keys = ", ".join(sorted(raw_rows[0].keys())) if raw_rows[0] else "(none)"
    return (
        f"Read {len(raw_rows)} row(s) from the file, but none had a value in the "
        f"'{column_label}' column. Columns found in the file: {sample_keys}"
    )


CENTER_TEMPLATE_HEADERS = ["center_uid", "city", "centre_name", "syncup_email", "password", "app_password"]
LEARNER_TEMPLATE_HEADERS = ["city", "centre_name", "center_uid", "name", "luid", "roll_no", "email", "google_drive_video_link"]

CENTER_TEMPLATE_SAMPLE = [
    ["CTR-DEL-001", "Delhi", "Delhi Central Hub", "delhicentral@syncup.org", "ChangeMe123!", "abcd efgh ijkl mnop"],
    ["CTR-MUM-002", "Mumbai", "Mumbai West Hub", "mumbaiwest@syncup.org", "ChangeMe456!", ""],
]
LEARNER_TEMPLATE_SAMPLE = [
    ["Delhi", "Delhi Central Hub", "CTR-DEL-001", "Aisha Khan", "LUID-1001", "R-101", "aisha.parent@example.com",
     "https://drive.google.com/file/d/1AbCdEfGhIjKlMnOpQrStUvWxYz/view?usp=sharing"],
    ["Mumbai", "Mumbai West Hub", "CTR-MUM-002", "Rohan Mehta", "LUID-1002", "R-102", "rohan.parent@example.com",
     "https://drive.google.com/file/d/1ZyXwVuTsRqPoNmLkJiHgFeDcBa/view?usp=sharing"],
]


def csv_response(headers, sample_rows, filename):
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(headers)
    writer.writerows(sample_rows)
    return Response(
        buf.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


# --------------------------------------------------------------- routing --

@app.route("/")
def index():
    if session.get("is_admin"):
        if session.get("admin_role") == "full":
            return redirect(url_for("admin_dashboard"))
        return redirect(url_for("admin_overview_attendance"))
    if session.get("center_uid"):
        return redirect(url_for("center_dashboard"))
    if session.get("learner_luid"):
        return redirect(url_for("learner_dashboard"))
    return render_template("index.html")


@app.route("/learner/login", methods=["POST"])
def learner_login():
    email = request.form.get("email", "").strip()
    roll_no = request.form.get("roll_no", "").strip()
    if not email or not roll_no:
        flash("Enter both your email and roll number", "error")
        return render_template("index.html")

    learner = db.get_learner_by_login(email, roll_no)
    if not learner:
        flash("No learner found with that email and roll number combination", "error")
        return render_template("index.html")

    session.clear()
    session["learner_luid"] = learner["luid"]
    session["learner_name"] = learner["name"]
    return redirect(url_for("learner_dashboard"))


@app.route("/learner/logout")
def learner_logout():
    session.clear()
    return redirect(url_for("index"))


# ---- Admin auth ----

# ---- Learner portal ----

@app.route("/learner")
@learner_required
def learner_dashboard():
    learner = db.get_learner(session["learner_luid"])
    if not learner:
        session.clear()
        return redirect(url_for("index"))
    atlas = db.get_latest_atlas_for_learner(learner["luid"])
    return render_template("learner_dashboard.html", learner=learner, atlas=atlas)


@app.route("/learner/attendance")
@learner_required
def learner_attendance():
    luid = session["learner_luid"]
    today = datetime.date.today()
    try:
        year = int(request.args.get("year", today.year))
        month = int(request.args.get("month", today.month))
    except ValueError:
        year, month = today.year, today.month
    month = min(12, max(1, month))

    by_day = db.get_attendance_calendar(luid, year, month)
    import calendar as _calendar
    cal = _calendar.Calendar(firstweekday=0)
    weeks = cal.monthdayscalendar(year, month)  # 0 = day outside this month

    prev_month, prev_year = (12, year - 1) if month == 1 else (month - 1, year)
    next_month, next_year = (1, year + 1) if month == 12 else (month + 1, year)

    return render_template(
        "learner_attendance.html",
        weeks=weeks, by_day=by_day, year=year, month=month,
        month_name=_calendar.month_name[month],
        prev_month=prev_month, prev_year=prev_year,
        next_month=next_month, next_year=next_year,
        today_iso=today.isoformat(),
    )


@app.route("/learner/tests")
@learner_required
def learner_tests():
    luid = session["learner_luid"]
    rows = db.get_test_records_for_learner(luid)
    tests = pivot_test_records(rows)
    return render_template("learner_tests.html", tests=tests)


@app.route("/learner/nps")
@learner_required
def learner_nps():
    luid = session["learner_luid"]
    rows = db.get_nps_for_learner(luid)
    return render_template("learner_nps.html", rows=rows)


@app.route("/learner/feedback", methods=["GET", "POST"])
@learner_required
def learner_feedback():
    luid = session["learner_luid"]
    last = db.get_last_feedback(luid)
    can_submit = True
    next_allowed = None
    if last:
        last_dt = datetime.datetime.fromisoformat(last["submitted_at"])
        elapsed = datetime.datetime.now() - last_dt
        if elapsed.days < config.FEEDBACK_COOLDOWN_DAYS:
            can_submit = False
            next_allowed = (last_dt + datetime.timedelta(days=config.FEEDBACK_COOLDOWN_DAYS)).date().isoformat()

    if request.method == "POST":
        if not can_submit:
            flash(f"You can submit feedback again on {next_allowed}", "error")
            return redirect(url_for("learner_feedback"))

        try:
            rating = int(request.form.get("rating", "0"))
        except ValueError:
            rating = 0
        if rating < 1 or rating > 5:
            flash("Select a star rating from 1 to 5", "error")
            return redirect(url_for("learner_feedback"))

        flag_keys = ["app_tech", "class_recording", "staff_cooperation", "study_material",
                     "tests", "quality_educators", "syllabus_progress", "centre_facilities"]
        flags = {}
        for k in flag_keys:
            v = request.form.get(f"flag_{k}", "").strip().lower()
            flags[k] = v if v in ("like", "dislike") else None

        remark = request.form.get("remark", "").strip()
        db.add_feedback(luid, rating, flags, remark)
        flash("Thanks for your feedback!", "success")
        return redirect(url_for("learner_feedback"))

    return render_template(
        "learner_feedback.html", can_submit=can_submit, next_allowed=next_allowed,
        cooldown_days=config.FEEDBACK_COOLDOWN_DAYS,
    )


@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        u = request.form.get("username", "")
        p = request.form.get("password", "")
        if u == config.ADMIN_USERNAME and p == config.ADMIN_PASSWORD:
            session.clear()
            session["is_admin"] = True
            session["admin_role"] = "full"
            return redirect(url_for("admin_dashboard"))

        admin_user = db.get_admin_user(u)
        if admin_user and check_password_hash(admin_user["password_hash"], p):
            session.clear()
            session["is_admin"] = True
            session["admin_role"] = admin_user["role"]
            session["admin_username"] = admin_user["username"]
            if admin_user["role"] == "full":
                return redirect(url_for("admin_dashboard"))
            return redirect(url_for("admin_overview_attendance"))

        flash("Invalid admin credentials", "error")
    return render_template("admin_login.html")


@app.route("/admin/logout")
def admin_logout():
    session.clear()
    return redirect(url_for("index"))


# ---- Admin dashboard ----

@app.route("/admin")
@admin_required
def admin_dashboard():
    centers = db.list_centers()
    learners = db.list_learners()
    return render_template("admin_dashboard.html", centers=centers, learners=learners)


# ---- Atlas dump (learner directory/subscription data) ----

@app.route("/admin/atlas", methods=["GET", "POST"])
@admin_required
def admin_atlas():
    if request.method == "POST":
        csv_file = request.files.get("csv_file")
        if not csv_file or csv_file.filename == "":
            flash("Choose a CSV file first", "error")
            return render_template("admin_atlas.html", uploads=db.list_atlas_uploads())
        try:
            raw_rows, skipped = read_csv_rows(csv_file)
            rows = parse_atlas_rows(raw_rows)
        except Exception as e:
            flash(f"Could not read CSV: {e}", "error")
            return render_template("admin_atlas.html", uploads=db.list_atlas_uploads())
        if not rows:
            flash(no_rows_message(raw_rows, "learner_uid"), "error")
            return render_template("admin_atlas.html", uploads=db.list_atlas_uploads())

        upload_id, count, evicted = db.save_atlas_upload(rows)
        msg = f"Atlas dump uploaded: {count} learners."
        if skipped:
            msg += f" {skipped} row(s) could not be parsed and were skipped."
        if evicted:
            msg += f" Removed {evicted} older upload(s) beyond the 3 most recent (rolling retention)."
        flash(msg, "success")
        return redirect(url_for("admin_atlas"))

    return render_template("admin_atlas.html", uploads=db.list_atlas_uploads())


# ---- Test dump ----

@app.route("/admin/test-dump", methods=["GET", "POST"])
@admin_required
def admin_test_dump():
    if request.method == "POST":
        csv_file = request.files.get("csv_file")
        if not csv_file or csv_file.filename == "":
            flash("Choose a CSV file first", "error")
            return render_template("admin_test_dump.html", uploads=db.list_test_uploads())
        try:
            raw_rows, skipped = read_csv_rows(csv_file)
            rows = parse_test_rows(raw_rows)
        except Exception as e:
            flash(f"Could not read CSV: {e}", "error")
            return render_template("admin_test_dump.html", uploads=db.list_test_uploads())
        if not rows:
            flash(no_rows_message(raw_rows, "learner_uid"), "error")
            return render_template("admin_test_dump.html", uploads=db.list_test_uploads())

        upload_id, count, evicted = db.save_test_upload(rows)
        msg = f"Test dump uploaded: {count} rows."
        if skipped:
            msg += f" {skipped} row(s) could not be parsed and were skipped."
        if evicted:
            msg += f" Removed {evicted} older upload(s) beyond the 3 most recent (rolling retention)."
        flash(msg, "success")
        return redirect(url_for("admin_test_dump"))

    return render_template("admin_test_dump.html", uploads=db.list_test_uploads())


# ---- NPS dump ----

@app.route("/admin/nps-dump", methods=["GET", "POST"])
@admin_required
def admin_nps_dump():
    if request.method == "POST":
        csv_file = request.files.get("csv_file")
        if not csv_file or csv_file.filename == "":
            flash("Choose a CSV file first", "error")
            return render_template("admin_nps_dump.html", uploads=db.list_nps_uploads())
        try:
            raw_rows, skipped = read_csv_rows(csv_file)
            rows = parse_nps_rows(raw_rows)
        except Exception as e:
            flash(f"Could not read CSV: {e}", "error")
            return render_template("admin_nps_dump.html", uploads=db.list_nps_uploads())
        if not rows:
            flash(no_rows_message(raw_rows, "learner_uid"), "error")
            return render_template("admin_nps_dump.html", uploads=db.list_nps_uploads())

        upload_id, count = db.save_nps_upload(rows)
        msg = f"NPS dump uploaded: {count} responses. Full history is kept for date-range reporting."
        if skipped:
            msg += f" {skipped} row(s) could not be parsed and were skipped."
        flash(msg, "success")
        return redirect(url_for("admin_nps_dump"))

    return render_template("admin_nps_dump.html", uploads=db.list_nps_uploads())


# ---- Attendance dump (login/logout override for a specific date) ----

@app.route("/admin/attendance-dump", methods=["GET", "POST"])
@admin_required
def admin_attendance_dump():
    if request.method == "POST":
        csv_file = request.files.get("csv_file")
        if not csv_file or csv_file.filename == "":
            flash("Choose a CSV file first", "error")
            return render_template("admin_attendance_dump.html", uploads=db.list_attendance_dump_uploads())
        try:
            raw_rows, csv_skipped = read_csv_rows(csv_file)
            rows, bad_dates = parse_attendance_dump_rows(raw_rows)
        except Exception as e:
            flash(f"Could not read CSV: {e}", "error")
            return render_template("admin_attendance_dump.html", uploads=db.list_attendance_dump_uploads())
        if not rows:
            flash(no_rows_message(raw_rows, "learner_uid / Day of dates"), "error")
            return render_template("admin_attendance_dump.html", uploads=db.list_attendance_dump_uploads())

        updated, skipped_unknown, skipped_uids = db.apply_attendance_dump(rows)
        msg = f"Attendance dump applied: {updated} learner-day(s) updated."
        if skipped_unknown:
            sample = ", ".join(skipped_uids[:10])
            more = f" (+{skipped_unknown - min(10, len(skipped_uids))} more)" if skipped_unknown > 10 else ""
            msg += f" {skipped_unknown} row(s) skipped - learner_uid not enrolled here: {sample}{more}."
        if bad_dates:
            msg += f" {bad_dates} row(s) had an unparseable date and were skipped."
        if csv_skipped:
            msg += f" {csv_skipped} row(s) could not be parsed from the CSV itself."
        flash(msg, "success")
        return redirect(url_for("admin_attendance_dump"))

    return render_template("admin_attendance_dump.html", uploads=db.list_attendance_dump_uploads())


# ---- Feedback responses (learner-submitted, from the app's own Feedback tab) ----

@app.route("/admin/feedback-responses")
@admin_required
def admin_feedback_responses():
    date_from = request.args.get("from") or None
    date_to = request.args.get("to") or None
    rows = db.get_feedback_responses(date_from, date_to)
    center_averages = db.get_feedback_center_averages(date_from, date_to)
    overall_avg = (
        sum(r["rating"] for r in rows) / len(rows) if rows else None
    )
    return render_template(
        "admin_feedback_responses.html", rows=rows, center_averages=center_averages,
        overall_avg=overall_avg, date_from=date_from or "", date_to=date_to or "",
    )


# ---- Cross-center overview dashboards (both full and restricted admins) ----

@app.route("/admin/overview/attendance")
@admin_dashboard_required
def admin_overview_attendance():
    city = request.args.get("city") or None
    center_uid = request.args.get("center_uid") or None
    date_from = request.args.get("from") or None
    date_to = request.args.get("to") or None
    rows = db.admin_attendance_overview(city, center_uid, date_from, date_to)
    return render_template(
        "admin_overview_attendance.html", rows=rows, cities=db.list_cities(), centers=db.list_centers(),
        city=city or "", center_uid=center_uid or "", date_from=date_from or "", date_to=date_to or "",
    )


@app.route("/admin/overview/nps")
@admin_dashboard_required
def admin_overview_nps():
    city = request.args.get("city") or None
    center_uid = request.args.get("center_uid") or None
    date_from = request.args.get("from") or None
    date_to = request.args.get("to") or None
    rows = db.admin_nps_overview(city, center_uid, date_from, date_to)
    return render_template(
        "admin_overview_nps.html", rows=rows, cities=db.list_cities(), centers=db.list_centers(),
        city=city or "", center_uid=center_uid or "", date_from=date_from or "", date_to=date_to or "",
    )


@app.route("/admin/overview/learners")
@admin_dashboard_required
def admin_overview_learners():
    city = request.args.get("city") or None
    center_uid = request.args.get("center_uid") or None
    rows = db.admin_learner_overview(city, center_uid)
    return render_template(
        "admin_overview_learners.html", rows=rows, cities=db.list_cities(), centers=db.list_centers(),
        city=city or "", center_uid=center_uid or "",
    )


@app.route("/admin/centers/add", methods=["GET", "POST"])
@admin_required
def add_center():
    if request.method == "POST":
        center_uid = request.form.get("center_uid", "").strip()
        city = request.form.get("city", "").strip()
        centre_name = request.form.get("centre_name", "").strip()
        syncup_email = request.form.get("syncup_email", "").strip()
        password = request.form.get("password", "")
        app_password = request.form.get("app_password", "").strip()

        if not all([center_uid, city, centre_name, syncup_email, password]):
            flash("All fields are required", "error")
            return render_template("add_center.html")

        if len(password) < 6:
            flash("Password must be at least 6 characters", "error")
            return render_template("add_center.html")

        if db.get_center(center_uid):
            flash(f"Center UID '{center_uid}' already exists", "error")
            return render_template("add_center.html")

        if db.get_center_by_email(syncup_email):
            flash("That Syncup email is already registered to a center", "error")
            return render_template("add_center.html")

        db.add_center(center_uid, city, centre_name, syncup_email, generate_password_hash(password), app_password)
        flash(f"Center '{centre_name}' added successfully", "success")
        return redirect(url_for("admin_dashboard"))

    return render_template("add_center.html")


@app.route("/admin/centers/<center_uid>/edit", methods=["GET", "POST"])
@admin_required
def edit_center(center_uid):
    center = db.get_center(center_uid)
    if not center:
        flash(f"Center '{center_uid}' not found", "error")
        return redirect(url_for("admin_dashboard"))

    if request.method == "POST":
        city = request.form.get("city", "").strip()
        centre_name = request.form.get("centre_name", "").strip()
        syncup_email = request.form.get("syncup_email", "").strip()
        app_password = request.form.get("app_password", "").strip()
        new_password = request.form.get("new_password", "").strip()

        if not all([city, centre_name, syncup_email]):
            flash("City, Centre Name and Syncup Email are required", "error")
            return render_template("edit_center.html", center=center)

        existing = db.get_center_by_email(syncup_email)
        if existing and existing["center_uid"] != center_uid:
            flash("That Syncup email is already registered to another center", "error")
            return render_template("edit_center.html", center=center)

        if new_password and len(new_password) < 6:
            flash("New password must be at least 6 characters (leave blank to keep the current one)", "error")
            return render_template("edit_center.html", center=center)

        password_hash = generate_password_hash(new_password) if new_password else None
        db.update_center(center_uid, city, centre_name, syncup_email, app_password, password_hash)
        flash(f"Center '{centre_name}' updated", "success")
        return redirect(url_for("admin_dashboard"))

    return render_template("edit_center.html", center=center)


@app.route("/admin/learners/add", methods=["GET", "POST"])
@admin_required
def add_learner():
    centers = db.list_centers()

    if request.method == "POST":
        city = request.form.get("city", "").strip()
        centre_name = request.form.get("centre_name", "").strip()
        center_uid = request.form.get("center_uid", "").strip()
        name = request.form.get("name", "").strip()
        luid = request.form.get("luid", "").strip()
        roll_no = request.form.get("roll_no", "").strip()
        email = request.form.get("email", "").strip()
        video_file = request.files.get("video")
        drive_link = request.form.get("drive_link", "").strip()
        capture_frames_file = request.files.get("capture_frames_file")

        if not all([city, centre_name, center_uid, name, luid, roll_no, email]):
            flash("All fields are required", "error")
            return render_template("add_learner.html", centers=centers)

        if not mailer.is_valid_email(email):
            flash("Enter a valid learner email address", "error")
            return render_template("add_learner.html", centers=centers)

        if not db.get_center(center_uid):
            flash("Selected center does not exist", "error")
            return render_template("add_learner.html", centers=centers)

        if db.learner_exists(luid):
            flash(f"LUID '{luid}' is already enrolled", "error")
            return render_template("add_learner.html", centers=centers)

        embeddings, tmp_path, err = process_face_source(
            video_file, drive_link, capture_frames_file, config.VIDEO_TMP_DIR
        )
        if err:
            flash(err, "error")
            return render_template("add_learner.html", centers=centers)

        if not embeddings:
            flash(
                "No usable face detected. Ensure the learner's face is "
                "clearly visible, well-lit, and not blurry, then try again.",
                "error",
            )
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
            return render_template("add_learner.html", centers=centers)

        db.add_learner(luid, name, roll_no, center_uid, email)
        db.add_embeddings(luid, embeddings)
        face_engine.center_index.invalidate(center_uid)  # force reload with new learner

        if tmp_path and not config.RETAIN_ENROLLMENT_VIDEO:
            os.remove(tmp_path)

        flash(
            f"Learner '{name}' enrolled with {len(embeddings)} face embeddings.", "success"
        )
        return redirect(url_for("admin_dashboard"))

    return render_template("add_learner.html", centers=centers)


@app.route("/admin/learners")
@admin_required
def list_learners_admin():
    learners = db.list_learners()
    centers = {c["center_uid"]: c for c in db.list_centers()}
    return render_template("admin_learners.html", learners=learners, centers=centers)


@app.route("/admin/learners/<luid>/edit", methods=["GET", "POST"])
@admin_required
def edit_learner_admin(luid):
    learner = db.get_learner(luid)
    if not learner:
        flash(f"Learner '{luid}' not found", "error")
        return redirect(url_for("list_learners_admin"))
    centers = db.list_centers()

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        roll_no = request.form.get("roll_no", "").strip()
        email = request.form.get("email", "").strip()
        center_uid = request.form.get("center_uid", "").strip()

        if not all([name, roll_no, center_uid]):
            flash("Name, Roll No and Centre are required", "error")
            return render_template("edit_learner.html", learner=learner, centers=centers, is_admin=True)
        if email and not mailer.is_valid_email(email):
            flash("Enter a valid learner email address", "error")
            return render_template("edit_learner.html", learner=learner, centers=centers, is_admin=True)
        if not db.get_center(center_uid):
            flash("Selected center does not exist", "error")
            return render_template("edit_learner.html", learner=learner, centers=centers, is_admin=True)

        old_center_uid = learner["center_uid"]
        db.update_learner(luid, name, roll_no, email, center_uid)
        if center_uid != old_center_uid:
            face_engine.center_index.invalidate(old_center_uid)
            face_engine.center_index.invalidate(center_uid)
        flash(f"Learner '{name}' updated", "success")
        return redirect(url_for("list_learners_admin"))

    return render_template("edit_learner.html", learner=learner, centers=centers, is_admin=True)


# ---- Bulk import (CSV) ----

@app.route("/admin/centers/template.csv")
@admin_required
def center_csv_template():
    return csv_response(CENTER_TEMPLATE_HEADERS, CENTER_TEMPLATE_SAMPLE, "center_bulk_template.csv")


@app.route("/admin/learners/template.csv")
@admin_required
def learner_csv_template():
    return csv_response(LEARNER_TEMPLATE_HEADERS, LEARNER_TEMPLATE_SAMPLE, "learner_bulk_template.csv")


@app.route("/admin/centers/bulk", methods=["GET", "POST"])
@admin_required
def bulk_add_centers():
    if request.method != "POST":
        return render_template("bulk_centers.html", results=None)

    csv_file = request.files.get("csv_file")
    if not csv_file or csv_file.filename == "":
        flash("Choose a CSV file first", "error")
        return render_template("bulk_centers.html", results=None)

    try:
        rows, csv_skipped = read_csv_rows(csv_file, max_rows=config.BULK_MAX_ROWS)
    except Exception as e:
        flash(f"Could not read CSV: {e}", "error")
        return render_template("bulk_centers.html", results=None)

    results = []
    for i, row in enumerate(rows, start=2):  # row 1 is the header
        center_uid = row.get("center_uid", "")
        city = row.get("city", "")
        centre_name = row.get("centre_name", "")
        syncup_email = row.get("syncup_email", "")
        password = row.get("password", "")
        app_password = row.get("app_password", "")

        label = center_uid or centre_name or f"row {i}"
        if not all([center_uid, city, centre_name, syncup_email, password]):
            results.append({"row": i, "label": label, "ok": False, "message": "Missing required field(s)"})
            continue
        if len(password) < 6:
            results.append({"row": i, "label": label, "ok": False, "message": "Password must be at least 6 characters"})
            continue
        if db.get_center(center_uid):
            results.append({"row": i, "label": label, "ok": False, "message": f"Centre UID '{center_uid}' already exists"})
            continue
        if db.get_center_by_email(syncup_email):
            results.append({"row": i, "label": label, "ok": False, "message": "Syncup email already registered"})
            continue

        try:
            db.add_center(center_uid, city, centre_name, syncup_email, generate_password_hash(password), app_password)
            results.append({"row": i, "label": label, "ok": True, "message": "Added"})
        except Exception as e:
            results.append({"row": i, "label": label, "ok": False, "message": str(e)})

    ok_count = sum(1 for r in results if r["ok"])
    msg = f"Bulk import finished: {ok_count}/{len(results)} centers added"
    if csv_skipped:
        msg += f" ({csv_skipped} CSV row(s) could not be parsed and were skipped)"
    flash(msg, "success" if ok_count else "error")
    return render_template("bulk_centers.html", results=results)


@app.route("/admin/learners/bulk", methods=["GET"])
@admin_required
def bulk_add_learners():
    return render_template("bulk_learners.html")


def _process_bulk_learner_row(i, row):
    name = row.get("name", "")
    luid = row.get("luid", "")
    roll_no = row.get("roll_no", "")
    center_uid = row.get("center_uid", "")
    email = row.get("email", "") or row.get("learner_email", "")
    drive_link = row.get("google_drive_video_link", "") or row.get("google_drive_link", "")

    label = name or luid or f"row {i}"

    if not all([name, luid, roll_no, center_uid, email, drive_link]):
        return {"row": i, "label": label, "ok": False, "message": "Missing required field(s)"}
    if not mailer.is_valid_email(email):
        return {"row": i, "label": label, "ok": False, "message": f"Invalid email '{email}'"}
    if not db.get_center(center_uid):
        return {"row": i, "label": label, "ok": False, "message": f"Centre '{center_uid}' does not exist"}
    if db.learner_exists(luid):
        return {"row": i, "label": label, "ok": False, "message": f"LUID '{luid}' already enrolled"}

    tmp_path = None
    try:
        tmp_path = download_drive_video(drive_link, config.VIDEO_TMP_DIR)
        embeddings = face_engine.process_enrollment_video(tmp_path)
        if not embeddings:
            return {"row": i, "label": label, "ok": False, "message": "No usable face detected in the fetched video"}

        db.add_learner(luid, name, roll_no, center_uid, email)
        db.add_embeddings(luid, embeddings)
        face_engine.center_index.invalidate(center_uid)
        return {"row": i, "label": label, "ok": True, "message": f"Added with {len(embeddings)} embeddings"}
    except Exception as e:
        return {"row": i, "label": label, "ok": False, "message": str(e)}
    finally:
        if tmp_path and os.path.exists(tmp_path) and not config.RETAIN_ENROLLMENT_VIDEO:
            os.remove(tmp_path)


def _run_bulk_learners_job(job_id, rows):
    job = BULK_JOBS[job_id]
    for i, row in enumerate(rows, start=2):
        result = _process_bulk_learner_row(i, row)
        with BULK_JOBS_LOCK:
            job["results"].append(result)
            job["processed"] += 1
    with BULK_JOBS_LOCK:
        job["done"] = True


@app.route("/admin/learners/bulk/start", methods=["POST"])
@admin_required
def bulk_add_learners_start():
    csv_file = request.files.get("csv_file")
    if not csv_file or csv_file.filename == "":
        return jsonify({"error": "Choose a CSV file first"}), 400
    try:
        rows, csv_skipped = read_csv_rows(csv_file, max_rows=config.BULK_MAX_ROWS)
    except Exception as e:
        return jsonify({"error": f"Could not read CSV: {e}"}), 400
    if not rows:
        return jsonify({"error": "CSV has no data rows"}), 400

    job_id = uuid.uuid4().hex
    with BULK_JOBS_LOCK:
        BULK_JOBS[job_id] = {"total": len(rows), "processed": 0, "results": [], "done": False}
    threading.Thread(target=_run_bulk_learners_job, args=(job_id, rows), daemon=True).start()
    return jsonify({"job_id": job_id, "total": len(rows), "csv_skipped": csv_skipped})


@app.route("/admin/learners/bulk/progress/<job_id>")
@admin_required
def bulk_add_learners_progress(job_id):
    job = BULK_JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Unknown or expired job"}), 404
    ok_count = sum(1 for r in job["results"] if r["ok"])
    error_count = sum(1 for r in job["results"] if not r["ok"])
    return jsonify({
        "total": job["total"],
        "processed": job["processed"],
        "done": job["done"],
        "ok_count": ok_count,
        "error_count": error_count,
        "results": job["results"],
    })


@app.route("/admin/learners/bulk/download/<job_id>")
@admin_required
def bulk_add_learners_download(job_id):
    job = BULK_JOBS.get(job_id)
    if not job:
        flash("Unknown or expired job — results are only kept until the server restarts", "error")
        return redirect(url_for("bulk_add_learners"))

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["Row", "Learner", "Status", "Detail"])
    for r in job["results"]:
        writer.writerow([r["row"], r["label"], "OK" if r["ok"] else "FAILED", r["message"]])

    return Response(
        buf.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename=bulk_learners_result_{job_id[:8]}.csv"},
    )


# ---- Reports (city / center wise) ----

@app.route("/admin/reports")
@admin_required
def admin_reports():
    city = request.args.get("city") or None
    center_uid = request.args.get("center_uid") or None
    date_from = request.args.get("from") or None
    date_to = request.args.get("to") or None
    q = request.args.get("q") or None

    by_city = db.report_summary_by_city(date_from, date_to)
    by_center = db.report_summary_by_center(city, date_from, date_to)
    rows = db.report_rows(city, center_uid, date_from, date_to, q)
    cities = db.list_cities()
    centers = db.list_centers()

    return render_template(
        "admin_reports.html",
        by_city=by_city, by_center=by_center, rows=rows[:300], total_rows=len(rows),
        cities=cities, centers=centers,
        city=city or "", center_uid=center_uid or "", date_from=date_from or "", date_to=date_to or "", q=q or "",
    )


@app.route("/admin/reports/export")
@admin_required
def admin_reports_export():
    city = request.args.get("city") or None
    center_uid = request.args.get("center_uid") or None
    date_from = request.args.get("from") or None
    date_to = request.args.get("to") or None
    q = request.args.get("q") or None
    rows = db.report_rows(city, center_uid, date_from, date_to, q)

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["City", "Centre Name", "Centre UID", "Name", "Roll No", "LUID", "Timestamp", "Confidence", "Status", "Punch Type"])
    for row in rows:
        writer.writerow([
            row["city"], row["centre_name"], row["center_uid"], row["name"], row["roll_no"],
            row["luid"], row["ts"], row["confidence"], row["status"], row["punch_type"],
        ])

    return Response(
        buf.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=admin_attendance_report.csv"},
    )


# ---- Center auth ----

@app.route("/center/login", methods=["GET", "POST"])
def center_login():
    if request.method == "POST":
        email = request.form.get("syncup_email", "").strip()
        password = request.form.get("password", "")
        center = db.get_center_by_email(email)
        if not center:
            flash("No center registered with that Syncup email", "error")
            return render_template("center_login.html")
        if not center["password_hash"] or not check_password_hash(center["password_hash"], password):
            flash("Incorrect password", "error")
            return render_template("center_login.html")
        session.clear()
        session["center_uid"] = center["center_uid"]
        session["center_name"] = center["centre_name"]
        return redirect(url_for("center_dashboard"))
    return render_template("center_login.html")


@app.route("/center/logout")
def center_logout():
    session.clear()
    return redirect(url_for("index"))


# ---- Center dashboard ----

@app.route("/center")
@center_required
def center_dashboard():
    center_uid = session["center_uid"]
    stats = db.attendance_stats(center_uid)
    today_logs = db.today_attendance_detail(center_uid)
    presence = db.learner_presence_today(center_uid)
    return render_template(
        "center_dashboard.html", stats=stats, today_logs=today_logs, presence=presence
    )


@app.route("/center/scanner")
@center_required
def scanner():
    return render_template("scanner.html")


@app.route("/center/learners")
@center_required
def center_learners():
    center_uid = session["center_uid"]
    q = request.args.get("q", "").strip()
    batch_filter = request.args.get("batch", "").strip()
    active_filter = request.args.get("active", "")  # "", "1", "0"

    learners = db.list_learners(center_uid=center_uid)
    atlas_map = db.get_latest_atlas_map()

    rows = []
    batches = set()
    for l in learners:
        atlas = atlas_map.get(l["luid"])
        batch_name = (atlas["batch_name"] if atlas and atlas["batch_name"] else "")
        if batch_name:
            batches.add(batch_name)
        rows.append({"luid": l["luid"], "name": l["name"], "roll_no": l["roll_no"],
                     "batch_name": batch_name, "active": l["active"]})

    if q:
        ql = q.lower()
        rows = [r for r in rows if ql in r["name"].lower() or ql in r["luid"].lower() or ql in r["roll_no"].lower()]
    if batch_filter:
        rows = [r for r in rows if r["batch_name"] == batch_filter]
    if active_filter in ("0", "1"):
        rows = [r for r in rows if str(r["active"]) == active_filter]

    return render_template(
        "center_learners.html", rows=rows, batches=sorted(batches),
        q=q, batch_filter=batch_filter, active_filter=active_filter,
    )


@app.route("/center/learners/<luid>/toggle-active", methods=["POST"])
@center_required
def center_toggle_active(luid):
    learner = db.get_learner(luid)
    if not learner or learner["center_uid"] != session["center_uid"]:
        return jsonify({"error": "Learner not found in your center"}), 404
    new_active = 0 if learner["active"] else 1
    db.set_learner_active(luid, new_active)
    return jsonify({"luid": luid, "active": bool(new_active)})


@app.route("/center/learners/<luid>/full")
@center_required
def center_learner_full(luid):
    center_uid = session["center_uid"]
    learner = db.get_learner(luid)
    if not learner or learner["center_uid"] != center_uid:
        return jsonify({"error": "Learner not found in your center"}), 404

    atlas = db.get_latest_atlas_for_learner(luid)
    attendance_rows = [r for r in db.get_attendance(center_uid) if r["luid"] == luid][:25]
    tests = pivot_test_records(db.get_test_records_for_learner(luid))

    return jsonify({
        "learner": {"luid": learner["luid"], "name": learner["name"], "roll_no": learner["roll_no"],
                    "email": learner["email"], "active": bool(learner["active"])},
        "atlas": dict(atlas) if atlas else None,
        "attendance": [{"ts": r["ts"], "punch_type": r["punch_type"], "status": r["status"]} for r in attendance_rows],
        "tests": tests,
    })


@app.route("/center/learners/<luid>/attendance")
@center_required
def center_learner_attendance(luid):
    center_uid = session["center_uid"]
    learner = db.get_learner(luid)
    if not learner or learner["center_uid"] != center_uid:
        flash("Learner not found in your center", "error")
        return redirect(url_for("center_learners"))

    today = datetime.date.today()
    try:
        year = int(request.args.get("year", today.year))
        month = int(request.args.get("month", today.month))
    except ValueError:
        year, month = today.year, today.month
    month = min(12, max(1, month))

    by_day = db.get_attendance_calendar(luid, year, month)
    import calendar as _calendar
    cal = _calendar.Calendar(firstweekday=0)
    weeks = cal.monthdayscalendar(year, month)

    prev_month, prev_year = (12, year - 1) if month == 1 else (month - 1, year)
    next_month, next_year = (1, year + 1) if month == 12 else (month + 1, year)

    return render_template(
        "center_learner_attendance.html",
        learner=learner, weeks=weeks, by_day=by_day, year=year, month=month,
        month_name=_calendar.month_name[month],
        prev_month=prev_month, prev_year=prev_year,
        next_month=next_month, next_year=next_year,
        today_iso=today.isoformat(),
    )


@app.route("/center/learners/export/details")
@center_required
def export_learner_details():
    center_uid = session["center_uid"]
    learners = db.list_learners(center_uid=center_uid)
    atlas_map = db.get_latest_atlas_map()

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["Name", "LUID", "Roll No", "Batch Name", "Active"])
    for l in learners:
        atlas = atlas_map.get(l["luid"])
        writer.writerow([l["name"], l["luid"], l["roll_no"],
                         (atlas["batch_name"] if atlas else "") or "", "Yes" if l["active"] else "No"])
    return Response(
        buf.getvalue(), mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=learner_details.csv"},
    )


@app.route("/center/learners/export/tests")
@center_required
def export_learner_tests():
    center_uid = session["center_uid"]
    rows = db.get_test_records_for_center(center_uid)

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["Learner Name", "Learner UID", "Roll No", "Batch Name", "Test Title",
                     "Test Start Date", "Attempt Type", "Section Name", "Section Score"])
    for r in rows:
        writer.writerow([r["learner_name"], r["learner_uid"], r["roll_no"], r["learner_batch_name"],
                         r["test_title"], r["test_start_date"], r["attempt_type"],
                         r["section_name"], r["section_score"]])
    return Response(
        buf.getvalue(), mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=learner_test_details.csv"},
    )


@app.route("/center/learners/add", methods=["GET", "POST"])
@center_required
def center_add_learner():
    center_uid = session["center_uid"]
    center = db.get_center(center_uid)

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        luid = request.form.get("luid", "").strip()
        roll_no = request.form.get("roll_no", "").strip()
        email = request.form.get("email", "").strip()
        video_file = request.files.get("video")
        drive_link = request.form.get("drive_link", "").strip()
        capture_frames_file = request.files.get("capture_frames_file")

        if not all([name, luid, roll_no, email]):
            flash("Name, LUID, Roll No and Email are required", "error")
            return render_template("center_add_learner.html", center=center)

        if not mailer.is_valid_email(email):
            flash("Enter a valid learner email address", "error")
            return render_template("center_add_learner.html", center=center)

        if db.learner_exists(luid):
            flash(f"LUID '{luid}' is already enrolled", "error")
            return render_template("center_add_learner.html", center=center)

        embeddings, tmp_path, err = process_face_source(
            video_file, drive_link, capture_frames_file, config.VIDEO_TMP_DIR
        )
        if err:
            flash(err, "error")
            return render_template("center_add_learner.html", center=center)

        if not embeddings:
            flash(
                "No usable face detected. Ensure the learner's face is "
                "clearly visible, well-lit, and not blurry, then try again.",
                "error",
            )
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
            return render_template("center_add_learner.html", center=center)

        # center_uid is always the logged-in center's own - a center can only enroll
        # learners into itself, never into another center.
        db.add_learner(luid, name, roll_no, center_uid, email)
        db.add_embeddings(luid, embeddings)
        face_engine.center_index.invalidate(center_uid)

        if tmp_path and not config.RETAIN_ENROLLMENT_VIDEO:
            os.remove(tmp_path)

        flash(f"Learner '{name}' enrolled with {len(embeddings)} face embeddings.", "success")
        return redirect(url_for("center_learners"))

    return render_template("center_add_learner.html", center=center)


@app.route("/center/learners/<luid>/edit", methods=["GET", "POST"])
@center_required
def edit_learner_center(luid):
    center_uid = session["center_uid"]
    learner = db.get_learner(luid)
    if not learner or learner["center_uid"] != center_uid:
        flash("Learner not found in your center", "error")
        return redirect(url_for("center_learners"))

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        roll_no = request.form.get("roll_no", "").strip()
        email = request.form.get("email", "").strip()

        if not all([name, roll_no]):
            flash("Name and Roll No are required", "error")
            return render_template("edit_learner.html", learner=learner, centers=None, is_admin=False)
        if email and not mailer.is_valid_email(email):
            flash("Enter a valid learner email address", "error")
            return render_template("edit_learner.html", learner=learner, centers=None, is_admin=False)

        db.update_learner(luid, name, roll_no, email)  # center_uid untouched by a center user
        flash(f"Learner '{name}' updated", "success")
        return redirect(url_for("center_learners"))

    return render_template("edit_learner.html", learner=learner, centers=None, is_admin=False)


@app.route("/api/scan", methods=["POST"])
@center_required
def api_scan():
    center_uid = session["center_uid"]
    data = request.get_json(silent=True) or {}
    image_data = data.get("image")
    mode = data.get("mode", "login")
    if mode not in ("login", "logout"):
        mode = "login"
    if not image_data:
        return jsonify({"status": "error", "message": "No image received"}), 400

    try:
        frame = decode_base64_image(image_data)
    except Exception:
        return jsonify({"status": "error", "message": "Could not decode image"}), 400

    if frame is None:
        return jsonify({"status": "error", "message": "Could not decode image"}), 400

    face_engine.center_index.ensure_loaded(center_uid, db.get_embeddings_for_center)

    # detect_faces_for_scan: enhanced (contrast/sharpen/upscale) pass, looser thresholds
    # than enrollment so faces farther from the camera or mid-crowd still register, and
    # keep_all returns every face in the frame in one pass (group scans, not just one).
    dets = face_engine.detect_faces_for_scan(frame)
    if not dets:
        return jsonify({"status": "no_face", "message": "No face detected. Center in frame and hold still."})

    # Guard against low quality frames (basic spoof/quality gate, not full liveness)
    dets = [d for d in dets if d["blur"] >= config.SCAN_BLUR_THRESHOLD]
    if not dets:
        return jsonify({"status": "low_quality", "message": "Image too blurry. Hold steady and try again."})

    results = []
    for d in dets:
        match = face_engine.center_index.search(center_uid, d["embedding"])
        if match is None:
            continue
        luid, score = match
        if luid is None:
            db.log_scan_attempt(center_uid, None, score, matched=False)
            results.append({"status": "no_match", "confidence": round(score, 3)})
            continue

        learner = db.get_learner(luid)
        db.log_scan_attempt(center_uid, luid, score, matched=True)

        if mode == "logout":
            login_rec = db.last_attendance_today(luid, punch_type="login")
            if not login_rec:
                results.append(
                    {
                        "status": "not_logged_in",
                        "mode": mode,
                        "name": learner["name"],
                        "roll_no": learner["roll_no"],
                        "confidence": round(score, 3),
                        "message": f"{learner['name']} is not logged in today — cannot log out.",
                    }
                )
                continue

        already = db.last_attendance_today(luid, punch_type=mode)
        if already:
            results.append(
                {
                    "status": "already_marked",
                    "mode": mode,
                    "name": learner["name"],
                    "roll_no": learner["roll_no"],
                    "confidence": round(score, 3),
                    "marked_at": already["ts"],
                }
            )
        else:
            db.mark_attendance(luid, center_uid, score, punch_type=mode)
            mailer.send_attendance_email_async(
                learner["email"], learner["name"], action=mode, **center_smtp_kwargs(center_uid)
            )
            results.append(
                {
                    "status": "marked",
                    "mode": mode,
                    "name": learner["name"],
                    "roll_no": learner["roll_no"],
                    "confidence": round(score, 3),
                }
            )

    if not results:
        return jsonify({"status": "no_match", "message": "Face not recognized in this center's records."})

    # report the most confident result first
    results.sort(key=lambda r: r.get("confidence", 0), reverse=True)
    return jsonify({"status": "ok", "results": results})


@app.route("/api/scan/recent")
@center_required
def api_scan_recent():
    mode = request.args.get("mode", "login")
    if mode not in ("login", "logout"):
        mode = "login"
    rows = db.recent_punches(session["center_uid"], mode, limit=10)
    return jsonify({"mode": mode, "names": [{"name": r["name"], "roll_no": r["roll_no"], "ts": r["ts"]} for r in rows]})


@app.route("/center/attendance")
@center_required
def attendance():
    center_uid = session["center_uid"]
    date_from = request.args.get("from") or None
    date_to = request.args.get("to") or None
    q = request.args.get("q") or None
    logs = db.get_attendance(center_uid, date_from, date_to, q)
    return render_template("attendance.html", logs=logs, date_from=date_from or "", date_to=date_to or "", q=q or "")


@app.route("/center/attendance/export")
@center_required
def export_attendance():
    center_uid = session["center_uid"]
    date_from = request.args.get("from") or None
    date_to = request.args.get("to") or None
    q = request.args.get("q") or None
    logs = db.get_attendance(center_uid, date_from, date_to, q)

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["Name", "Roll No", "LUID", "Timestamp", "Confidence", "Status", "Punch Type"])
    for row in logs:
        writer.writerow([row["name"], row["roll_no"], row["luid"], row["ts"], row["confidence"], row["status"], row["punch_type"]])

    return Response(
        buf.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=attendance_export.csv"},
    )


@app.route("/center/reports")
@center_required
def center_reports():
    center_uid = session["center_uid"]

    # 1. Learner details report
    learners = db.list_learners(center_uid=center_uid)
    atlas_map = db.get_latest_atlas_map()
    batch_counts = defaultdict(int)
    active_count = 0
    for l in learners:
        atlas = atlas_map.get(l["luid"])
        bn = atlas["batch_name"] if (atlas and atlas["batch_name"]) else "Unassigned"
        batch_counts[bn] += 1
        if l["active"]:
            active_count += 1
    learner_report = {
        "total": len(learners), "active": active_count, "inactive": len(learners) - active_count,
        "batches": sorted(batch_counts.items(), key=lambda x: -x[1]),
    }

    # 2. Attendance report - from/to date range, learners present per day in range
    att_from = request.args.get("att_from") or None
    att_to = request.args.get("att_to") or None
    att_logs = db.get_attendance(center_uid, att_from, att_to)
    present_by_day = defaultdict(set)
    for r in att_logs:
        present_by_day[r["ts"][:10]].add(r["luid"])
    today = datetime.date.today().isoformat()
    attendance_report = {
        "total_learners": len(learners),
        "present_today": len(present_by_day.get(today, set())),
        "days": sorted(({"date": d, "present": len(s)} for d, s in present_by_day.items()),
                       key=lambda x: x["date"], reverse=True),
        "from": att_from or "", "to": att_to or "",
    }

    # 3. Test report - ranked learners
    test_ranking = center_test_ranking(center_uid)
    test_report = {
        "ranking": test_ranking,
        "top_count": sum(1 for r in test_ranking if r["tier"] == "Top"),
        "medium_count": sum(1 for r in test_ranking if r["tier"] == "Medium"),
        "low_count": sum(1 for r in test_ranking if r["tier"] == "Low"),
    }

    return render_template(
        "center_reports.html", learner_report=learner_report,
        attendance_report=attendance_report, test_report=test_report,
    )


@app.route("/center/nps")
@center_required
def center_nps():
    center_uid = session["center_uid"]
    date_from = request.args.get("from") or None
    date_to = request.args.get("to") or None
    rating_filter = request.args.get("rating", "").strip()

    rows = db.get_nps_for_center(center_uid, date_from, date_to)
    summary = db.summarize_nps(rows)

    # NPS Rating: sum of every rating in this date range, divided by the total number
    # of feedback responses given (not unique learners - a learner who gave feedback
    # twice contributes two ratings to the sum, so the denominator has to be response
    # count, not headcount, or the average comes out inflated above 5).
    total_feedback = summary["total"]
    rating_sum = sum(int(r["rating"]) for r in rows if (r["rating"] or "").strip().isdigit())
    nps_rating = round(rating_sum / total_feedback, 2) if total_feedback else 0

    display_rows = rows
    if rating_filter.isdigit():
        display_rows = [r for r in rows if (r["rating"] or "").strip() == rating_filter]

    return render_template(
        "center_nps.html", rows=display_rows, summary=summary, date_from=date_from or "",
        date_to=date_to or "", rating_filter=rating_filter, nps_rating=nps_rating,
        total_feedback=total_feedback,
    )


if __name__ == "__main__":
    print("=" * 60)
    print(" Face Recognition Attendance System")
    print(f" Admin login   : http://localhost:5000/admin/login  ({config.ADMIN_USERNAME}/{config.ADMIN_PASSWORD})")
    print(" Center login  : http://localhost:5000/center/login  (use a Syncup email you added via admin)")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
