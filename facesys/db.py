import sqlite3
import datetime
import numpy as np
import config
from werkzeug.security import generate_password_hash

SCHEMA = """
-- ---- Restricted-access admin accounts (view-only cross-center dashboards).
-- The single primary admin (config.ADMIN_USERNAME/PASSWORD) always has full access
-- and isn't stored here - this table is only for additional admin logins, each with
-- an explicit role.
CREATE TABLE IF NOT EXISTS admin_users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'restricted',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS centers (
    center_uid TEXT PRIMARY KEY,
    city TEXT NOT NULL,
    centre_name TEXT NOT NULL,
    syncup_email TEXT UNIQUE NOT NULL,
    password_hash TEXT,
    app_password TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS learners (
    luid TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    roll_no TEXT NOT NULL,
    center_uid TEXT NOT NULL,
    email TEXT,
    enrolled_at TEXT NOT NULL,
    FOREIGN KEY (center_uid) REFERENCES centers(center_uid)
);

CREATE INDEX IF NOT EXISTS idx_learners_center ON learners(center_uid);
CREATE INDEX IF NOT EXISTS idx_learners_email_lower ON learners(lower(email));
CREATE INDEX IF NOT EXISTS idx_learners_luid_lower ON learners(lower(luid));

CREATE TABLE IF NOT EXISTS embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    luid TEXT NOT NULL,
    vector BLOB NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (luid) REFERENCES learners(luid)
);
CREATE INDEX IF NOT EXISTS idx_embeddings_luid ON embeddings(luid);

CREATE TABLE IF NOT EXISTS attendance_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    luid TEXT NOT NULL,
    center_uid TEXT NOT NULL,
    ts TEXT NOT NULL,
    confidence REAL NOT NULL,
    status TEXT NOT NULL,
    punch_type TEXT NOT NULL DEFAULT 'login'
);
CREATE INDEX IF NOT EXISTS idx_attendance_luid_ts ON attendance_logs(luid, ts);
CREATE INDEX IF NOT EXISTS idx_attendance_center_ts ON attendance_logs(center_uid, ts);

CREATE TABLE IF NOT EXISTS scan_attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    center_uid TEXT NOT NULL,
    luid TEXT,
    ts TEXT NOT NULL,
    confidence REAL,
    matched INTEGER NOT NULL
);

-- ---- Atlas dump (learner directory/subscription data, admin-uploaded, rolling 3-upload retention) ----
CREATE TABLE IF NOT EXISTS atlas_uploads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    uploaded_at TEXT NOT NULL,
    row_count INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS atlas_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    upload_id INTEGER NOT NULL,
    batch_name TEXT,
    centre_city_name TEXT,
    centre_name TEXT,
    goal_name TEXT,
    learner_name TEXT,
    learner_uid TEXT NOT NULL,
    roll_number TEXT,
    FOREIGN KEY (upload_id) REFERENCES atlas_uploads(id)
);
CREATE INDEX IF NOT EXISTS idx_atlas_learner ON atlas_records(learner_uid);
CREATE INDEX IF NOT EXISTS idx_atlas_upload ON atlas_records(upload_id);

-- Materialized "latest atlas row per learner", rebuilt once per admin upload rather
-- than recomputed by a GROUP BY/JOIN aggregate on every dashboard page view - that
-- recomputation was the main cause of slow page loads across the learner dashboard,
-- center learner list, and center reports once Atlas dumps grew large.
CREATE TABLE IF NOT EXISTS atlas_latest (
    learner_uid TEXT PRIMARY KEY,
    batch_name TEXT,
    centre_city_name TEXT,
    centre_name TEXT,
    goal_name TEXT,
    learner_name TEXT,
    roll_number TEXT,
    upload_id INTEGER
);

-- ---- Test dump (admin-uploaded, rolling 3-upload retention) ----
CREATE TABLE IF NOT EXISTS test_uploads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    uploaded_at TEXT NOT NULL,
    row_count INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS test_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    upload_id INTEGER NOT NULL,
    learner_id TEXT,
    learner_uid TEXT NOT NULL,
    learner_name TEXT,
    roll_no TEXT,
    test_uid TEXT,
    test_title TEXT,
    test_start_date TEXT,
    course_name TEXT,
    course_uid TEXT,
    test_score TEXT,
    attempt_type TEXT,
    rank_on_score TEXT,
    learner_test_start_at TEXT,
    learner_test_end_at TEXT,
    learner_batch_uid TEXT,
    learner_batch_name TEXT,
    centre_name TEXT,
    learner_center_uid TEXT,
    center_city TEXT,
    section_name TEXT,
    total_ques_attempted TEXT,
    section_score TEXT,
    FOREIGN KEY (upload_id) REFERENCES test_uploads(id)
);
CREATE INDEX IF NOT EXISTS idx_test_learner ON test_records(learner_uid);
CREATE INDEX IF NOT EXISTS idx_test_upload ON test_records(upload_id);

-- ---- NPS dump (admin-uploaded; kept in full, not rotated - reports filter by date range) ----
CREATE TABLE IF NOT EXISTS nps_uploads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    uploaded_at TEXT NOT NULL,
    row_count INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS nps_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    upload_id INTEGER NOT NULL,
    feedback_date TEXT,
    feedback_date_norm TEXT,
    city_name TEXT,
    centre_name TEXT,
    category TEXT,
    version TEXT,
    learner_uid TEXT,
    learner_id TEXT,
    learner_name TEXT,
    learner_username TEXT,
    state_name TEXT,
    first_feedback_flag TEXT,
    feedback_platform TEXT,
    feedback_review TEXT,
    rating TEXT,
    flag_app_tech TEXT,
    flag_class_recording TEXT,
    flag_staff_cooperation TEXT,
    flag_study_material TEXT,
    flag_tests TEXT,
    flag_quality_educators TEXT,
    flag_syllabus_progress TEXT,
    flag_centre_facilities TEXT,
    FOREIGN KEY (upload_id) REFERENCES nps_uploads(id)
);
CREATE INDEX IF NOT EXISTS idx_nps_learner ON nps_records(learner_uid);
CREATE INDEX IF NOT EXISTS idx_nps_date ON nps_records(feedback_date_norm);
CREATE INDEX IF NOT EXISTS idx_nps_centre_name_lower ON nps_records(lower(trim(centre_name)));

-- ---- Attendance dump (admin-uploaded login/logout override for a specific date) ----
CREATE TABLE IF NOT EXISTS attendance_dump_uploads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    uploaded_at TEXT NOT NULL,
    row_count INTEGER NOT NULL,
    days_updated INTEGER NOT NULL,
    skipped INTEGER NOT NULL
);

-- ---- In-app learner feedback (submitted by the learner themself, admin-only to view) ----
CREATE TABLE IF NOT EXISTS learner_feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    luid TEXT NOT NULL,
    submitted_at TEXT NOT NULL,
    rating INTEGER NOT NULL,
    flag_app_tech TEXT,
    flag_class_recording TEXT,
    flag_staff_cooperation TEXT,
    flag_study_material TEXT,
    flag_tests TEXT,
    flag_quality_educators TEXT,
    flag_syllabus_progress TEXT,
    flag_centre_facilities TEXT,
    remark TEXT,
    FOREIGN KEY (luid) REFERENCES learners(luid)
);
CREATE INDEX IF NOT EXISTS idx_feedback_luid ON learner_feedback(luid);
"""


def get_conn():
    # timeout + WAL mode + busy_timeout together are what stop "database is locked"
    # errors under concurrent access (e.g. a CSV upload writing rows at the same
    # moment a face scan is reading/writing attendance) - WAL lets readers proceed
    # while a write is in progress instead of blocking on a single file lock, and the
    # busy timeout makes a connection retry for a while instead of failing instantly
    # if it still has to wait a moment for the writer to finish.
    conn = sqlite3.connect(config.DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 30000")
    return conn


def init_db():
    conn = get_conn()
    conn.executescript(SCHEMA)
    _migrate(conn)
    conn.commit()
    conn.close()


def _migrate(conn):
    """Add columns introduced after initial release, for DBs created before them."""
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(centers)").fetchall()}
    if "password_hash" not in cols:
        conn.execute("ALTER TABLE centers ADD COLUMN password_hash TEXT")

    if "app_password" not in cols:
        conn.execute("ALTER TABLE centers ADD COLUMN app_password TEXT")

    learner_cols = {r["name"] for r in conn.execute("PRAGMA table_info(learners)").fetchall()}
    if "email" not in learner_cols:
        conn.execute("ALTER TABLE learners ADD COLUMN email TEXT")

    att_cols = {r["name"] for r in conn.execute("PRAGMA table_info(attendance_logs)").fetchall()}
    if "punch_type" not in att_cols:
        conn.execute("ALTER TABLE attendance_logs ADD COLUMN punch_type TEXT NOT NULL DEFAULT 'login'")

    if "active" not in learner_cols:
        conn.execute("ALTER TABLE learners ADD COLUMN active INTEGER NOT NULL DEFAULT 1")

    # One-time backfill for DBs that already had Atlas uploads before atlas_latest
    # existed - without this, "My Details" / the center learner list would go blank
    # for everyone until the next Atlas upload happened to repopulate it.
    has_atlas_records = conn.execute(
        "SELECT COUNT(*) c FROM atlas_records"
    ).fetchone()["c"] > 0
    atlas_latest_empty = conn.execute(
        "SELECT COUNT(*) c FROM atlas_latest"
    ).fetchone()["c"] == 0
    if has_atlas_records and atlas_latest_empty:
        conn.execute(
            """INSERT INTO atlas_latest (learner_uid, batch_name, centre_city_name, centre_name,
               goal_name, learner_name, roll_number, upload_id)
               SELECT learner_uid, batch_name, centre_city_name, centre_name, goal_name, learner_name,
                      roll_number, upload_id
               FROM (
                   SELECT learner_uid, batch_name, centre_city_name, centre_name, goal_name, learner_name,
                          roll_number, upload_id,
                          ROW_NUMBER() OVER (PARTITION BY learner_uid ORDER BY upload_id DESC) AS rn
                   FROM atlas_records
               ) WHERE rn = 1"""
        )

    # Seed the requested restricted-access admin account (view-only cross-center
    # overview dashboards) if it doesn't already exist. Safe to run on every startup -
    # only inserts once, and never touches the account again after that (so changing
    # the password later, once there's a way to, won't get stomped on restart).
    _seed_admin_user(
        conn, "mahajan.sumit@unacademy.com", "MahajanS@123", "restricted"
    )


def _seed_admin_user(conn, username, password, role):
    existing = conn.execute(
        "SELECT id FROM admin_users WHERE lower(username) = lower(?)", (username,)
    ).fetchone()
    if not existing:
        conn.execute(
            "INSERT INTO admin_users (username, password_hash, role, created_at) VALUES (?,?,?,?)",
            (username, generate_password_hash(password), role, now()),
        )


def now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def parse_date_flexible(s):
    """Best-effort parse of a date string in one of several common export formats into
    an ISO 'YYYY-MM-DD' string, for date-range filtering. Returns None if unparseable -
    the raw string is always kept alongside for display regardless."""
    if not s:
        return None
    s = s.strip()
    for fmt in ("%d %B %Y", "%d %b %Y", "%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y",
                "%d/%m/%Y %H:%M:%S", "%m/%d/%Y", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            continue
    return None


# ---------- Centers ----------

def add_center(center_uid, city, centre_name, syncup_email, password_hash, app_password=None):
    conn = get_conn()
    conn.execute(
        "INSERT INTO centers (center_uid, city, centre_name, syncup_email, password_hash, app_password, created_at) "
        "VALUES (?,?,?,?,?,?,?)",
        (center_uid, city, centre_name, syncup_email.lower().strip(), password_hash,
         (app_password or "").strip() or None, now()),
    )
    conn.commit()
    conn.close()


def update_center(center_uid, city, centre_name, syncup_email, app_password, password_hash=None):
    """password_hash=None keeps the existing login password unchanged."""
    conn = get_conn()
    if password_hash:
        conn.execute(
            "UPDATE centers SET city=?, centre_name=?, syncup_email=?, app_password=?, password_hash=? "
            "WHERE center_uid=?",
            (city, centre_name, syncup_email.lower().strip(), (app_password or "").strip() or None,
             password_hash, center_uid),
        )
    else:
        conn.execute(
            "UPDATE centers SET city=?, centre_name=?, syncup_email=?, app_password=? WHERE center_uid=?",
            (city, centre_name, syncup_email.lower().strip(), (app_password or "").strip() or None, center_uid),
        )
    conn.commit()
    conn.close()


def get_center_by_email(email):
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM centers WHERE syncup_email = ?", (email.lower().strip(),)
    ).fetchone()
    conn.close()
    return row


def get_center(center_uid):
    conn = get_conn()
    row = conn.execute("SELECT * FROM centers WHERE center_uid = ?", (center_uid,)).fetchone()
    conn.close()
    return row


def list_centers():
    conn = get_conn()
    rows = conn.execute("SELECT * FROM centers ORDER BY created_at DESC").fetchall()
    conn.close()
    return rows


def list_cities():
    conn = get_conn()
    rows = conn.execute("SELECT DISTINCT city FROM centers ORDER BY city").fetchall()
    conn.close()
    return [r["city"] for r in rows]


def get_admin_user(username):
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM admin_users WHERE lower(username) = lower(?)", (username,)
    ).fetchone()
    conn.close()
    return row


def _filtered_centers(conn, city, center_uid):
    sql = "SELECT center_uid, city, centre_name FROM centers WHERE 1=1"
    params = []
    if city:
        sql += " AND lower(city) = lower(?)"
        params.append(city)
    if center_uid:
        sql += " AND center_uid = ?"
        params.append(center_uid)
    sql += " ORDER BY city, centre_name"
    return conn.execute(sql, params).fetchall()


def admin_attendance_overview(city=None, center_uid=None, date_from=None, date_to=None):
    """Per-center attendance summary for the cross-center admin dashboard: how many
    of a center's ACTIVE learners have punched attendance at least once in the
    selected date range, and what share of the center's active roster that is.
    Inactive learners are excluded from both sides of the percentage - they aren't
    expected to be attending, so counting them would understate every center."""
    conn = get_conn()
    centers = _filtered_centers(conn, city, center_uid)

    rows = []
    for c in centers:
        total = conn.execute(
            "SELECT COUNT(*) n FROM learners WHERE center_uid = ? AND active = 1", (c["center_uid"],)
        ).fetchone()["n"]

        att_sql = """SELECT COUNT(DISTINCT al.luid) n FROM attendance_logs al
                     JOIN learners l ON l.luid = al.luid AND l.active = 1
                     WHERE al.center_uid = ?"""
        att_params = [c["center_uid"]]
        if date_from:
            att_sql += " AND al.ts >= ?"
            att_params.append(date_from)
        if date_to:
            att_sql += " AND al.ts <= ?"
            att_params.append(date_to + "T23:59:59")
        attended = conn.execute(att_sql, att_params).fetchone()["n"]

        pct = round(attended / total * 100, 1) if total else 0.0
        rows.append({
            "city": c["city"], "centre_name": c["centre_name"], "center_uid": c["center_uid"],
            "total_learners": total, "attended_learners": attended, "pct": pct,
        })
    conn.close()
    return rows


def admin_nps_overview(city=None, center_uid=None, date_from=None, date_to=None):
    """Per-center NPS summary: the same NPS Rating (sum of ratings / total feedback
    given) as the center's own NPS Report, plus each of the 8 flag columns expressed
    as like% / dislike% of (like + dislike) responses to that category - 'none'
    responses aren't counted in either side of the percentage, matching the formula
    as specified (like / (like+dislike), dislike / (like+dislike))."""
    conn = get_conn()
    centers = _filtered_centers(conn, city, center_uid)
    conn.close()

    rows = []
    for c in centers:
        nps_rows = get_nps_for_center(c["center_uid"], date_from, date_to)
        summary = summarize_nps(nps_rows)
        total_feedback = summary["total"]
        rating_sum = sum(int(r["rating"]) for r in nps_rows if (r["rating"] or "").strip().isdigit())
        nps_rating = round(rating_sum / total_feedback, 2) if total_feedback else 0

        flag_pcts = []
        for f in summary["flags"]:
            denom = f["like"] + f["dislike"]
            like_pct = round(f["like"] / denom * 100, 1) if denom else 0.0
            dislike_pct = round(f["dislike"] / denom * 100, 1) if denom else 0.0
            flag_pcts.append({"label": f["label"], "like_pct": like_pct, "dislike_pct": dislike_pct})

        rows.append({
            "city": c["city"], "centre_name": c["centre_name"], "center_uid": c["center_uid"],
            "nps_rating": nps_rating, "total_feedback": total_feedback, "flags": flag_pcts,
        })
    return rows


def admin_learner_overview(city=None, center_uid=None):
    """Per-center learner summary: active learner headcount, and how many distinct
    batches (from the latest Atlas data) are represented among that center's
    learners."""
    conn = get_conn()
    centers = _filtered_centers(conn, city, center_uid)

    rows = []
    for c in centers:
        active = conn.execute(
            "SELECT COUNT(*) n FROM learners WHERE center_uid = ? AND active = 1", (c["center_uid"],)
        ).fetchone()["n"]
        batches = conn.execute(
            """SELECT COUNT(DISTINCT al.batch_name) n FROM atlas_latest al
               JOIN learners l ON l.luid = al.learner_uid
               WHERE l.center_uid = ? AND al.batch_name IS NOT NULL AND al.batch_name != ''""",
            (c["center_uid"],),
        ).fetchone()["n"]
        rows.append({
            "city": c["city"], "centre_name": c["centre_name"], "center_uid": c["center_uid"],
            "active_learners": active, "total_batches": batches,
        })
    conn.close()
    return rows


# ---------- Learners ----------

def add_learner(luid, name, roll_no, center_uid, email=None):
    conn = get_conn()
    conn.execute(
        "INSERT INTO learners (luid, name, roll_no, center_uid, email, enrolled_at) VALUES (?,?,?,?,?,?)",
        (luid, name, roll_no, center_uid, (email or "").strip() or None, now()),
    )
    conn.commit()
    conn.close()


def update_learner_email(luid, email):
    conn = get_conn()
    conn.execute("UPDATE learners SET email = ? WHERE luid = ?", ((email or "").strip() or None, luid))
    conn.commit()
    conn.close()


def update_learner(luid, name, roll_no, email, center_uid=None):
    """center_uid=None leaves the learner's center assignment unchanged."""
    conn = get_conn()
    if center_uid:
        conn.execute(
            "UPDATE learners SET name=?, roll_no=?, email=?, center_uid=? WHERE luid=?",
            (name, roll_no, (email or "").strip() or None, center_uid, luid),
        )
    else:
        conn.execute(
            "UPDATE learners SET name=?, roll_no=?, email=? WHERE luid=?",
            (name, roll_no, (email or "").strip() or None, luid),
        )
    conn.commit()
    conn.close()


def learner_exists(luid):
    conn = get_conn()
    row = conn.execute("SELECT 1 FROM learners WHERE luid = ?", (luid,)).fetchone()
    conn.close()
    return row is not None


def list_learners(center_uid=None):
    conn = get_conn()
    if center_uid:
        rows = conn.execute(
            "SELECT * FROM learners WHERE center_uid = ? ORDER BY enrolled_at DESC", (center_uid,)
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM learners ORDER BY enrolled_at DESC").fetchall()
    conn.close()
    return rows


def count_learners(center_uid):
    conn = get_conn()
    n = conn.execute("SELECT COUNT(*) c FROM learners WHERE center_uid = ?", (center_uid,)).fetchone()["c"]
    conn.close()
    return n


def get_learner(luid):
    conn = get_conn()
    row = conn.execute("SELECT * FROM learners WHERE luid = ?", (luid,)).fetchone()
    conn.close()
    return row


def get_learner_by_login(email, roll_no):
    """Learner portal auth: email as ID, roll number as 'password' (plaintext match -
    roll numbers aren't secrets, matching the spec as given)."""
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM learners WHERE lower(email) = ? AND roll_no = ?",
        ((email or "").strip().lower(), (roll_no or "").strip()),
    ).fetchone()
    conn.close()
    return row


def set_learner_active(luid, active):
    conn = get_conn()
    conn.execute("UPDATE learners SET active = ? WHERE luid = ?", (1 if active else 0, luid))
    conn.commit()
    conn.close()


# ---------- Embeddings ----------

def add_embeddings(luid, vectors):
    conn = get_conn()
    ts = now()
    for v in vectors:
        blob = np.asarray(v, dtype=np.float32).tobytes()
        conn.execute(
            "INSERT INTO embeddings (luid, vector, created_at) VALUES (?,?,?)",
            (luid, blob, ts),
        )
    conn.commit()
    conn.close()


def get_embeddings_for_center(center_uid):
    """Returns list of (luid, vector_bytes) for every embedding belonging to learners in this center."""
    conn = get_conn()
    rows = conn.execute(
        """SELECT e.luid, e.vector FROM embeddings e
           JOIN learners l ON l.luid = e.luid
           WHERE l.center_uid = ?""",
        (center_uid,),
    ).fetchall()
    conn.close()
    return [(r["luid"], r["vector"]) for r in rows]


def count_embeddings(luid):
    conn = get_conn()
    row = conn.execute("SELECT COUNT(*) c FROM embeddings WHERE luid = ?", (luid,)).fetchone()
    conn.close()
    return row["c"]


# ---------- Attendance ----------

def last_attendance_today(luid, punch_type=None):
    conn = get_conn()
    today = datetime.date.today().isoformat()
    if punch_type:
        row = conn.execute(
            "SELECT * FROM attendance_logs WHERE luid = ? AND punch_type = ? AND ts LIKE ? ORDER BY ts DESC LIMIT 1",
            (luid, punch_type, f"{today}%"),
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT * FROM attendance_logs WHERE luid = ? AND ts LIKE ? ORDER BY ts DESC LIMIT 1",
            (luid, f"{today}%"),
        ).fetchone()
    conn.close()
    return row


def mark_attendance(luid, center_uid, confidence, punch_type="login"):
    conn = get_conn()
    conn.execute(
        "INSERT INTO attendance_logs (luid, center_uid, ts, confidence, status, punch_type) VALUES (?,?,?,?,?,?)",
        (luid, center_uid, now(), confidence, "present", punch_type),
    )
    conn.commit()
    conn.close()


def recent_punches(center_uid, punch_type, limit=10):
    conn = get_conn()
    rows = conn.execute(
        """SELECT a.ts, a.punch_type, l.name, l.roll_no
           FROM attendance_logs a JOIN learners l ON l.luid = a.luid
           WHERE a.center_uid = ? AND a.punch_type = ?
           ORDER BY a.ts DESC LIMIT ?""",
        (center_uid, punch_type, limit),
    ).fetchall()
    conn.close()
    return rows


def today_attendance_detail(center_uid):
    conn = get_conn()
    today = datetime.date.today().isoformat()
    rows = conn.execute(
        """SELECT a.ts, a.punch_type, a.confidence, l.name, l.roll_no, l.luid
           FROM attendance_logs a JOIN learners l ON l.luid = a.luid
           WHERE a.center_uid = ? AND a.ts LIKE ?
           ORDER BY a.ts DESC""",
        (center_uid, f"{today}%"),
    ).fetchall()
    conn.close()
    return rows


def learner_presence_today(center_uid):
    """Per-learner present/absent status for today, with login/logout timestamps."""
    conn = get_conn()
    today = datetime.date.today().isoformat()
    learners = conn.execute(
        "SELECT luid, name, roll_no FROM learners WHERE center_uid = ? ORDER BY name", (center_uid,)
    ).fetchall()
    login_rows = conn.execute(
        "SELECT luid, MIN(ts) ts FROM attendance_logs WHERE center_uid = ? AND punch_type = 'login' AND ts LIKE ? GROUP BY luid",
        (center_uid, f"{today}%"),
    ).fetchall()
    logout_rows = conn.execute(
        "SELECT luid, MAX(ts) ts FROM attendance_logs WHERE center_uid = ? AND punch_type = 'logout' AND ts LIKE ? GROUP BY luid",
        (center_uid, f"{today}%"),
    ).fetchall()
    conn.close()

    login_map = {r["luid"]: r["ts"] for r in login_rows}
    logout_map = {r["luid"]: r["ts"] for r in logout_rows}

    result = []
    for l in learners:
        login_ts = login_map.get(l["luid"])
        result.append({
            "luid": l["luid"],
            "name": l["name"],
            "roll_no": l["roll_no"],
            "present": login_ts is not None,
            "login_ts": login_ts,
            "logout_ts": logout_map.get(l["luid"]),
        })
    return result


def log_scan_attempt(center_uid, luid, confidence, matched):
    conn = get_conn()
    conn.execute(
        "INSERT INTO scan_attempts (center_uid, luid, ts, confidence, matched) VALUES (?,?,?,?,?)",
        (center_uid, luid, now(), confidence, 1 if matched else 0),
    )
    conn.commit()
    conn.close()


def get_attendance(center_uid, date_from=None, date_to=None, query=None):
    conn = get_conn()
    sql = """SELECT a.id, a.ts, a.confidence, a.status, a.punch_type, l.name, l.roll_no, l.luid
             FROM attendance_logs a JOIN learners l ON l.luid = a.luid
             WHERE a.center_uid = ?"""
    params = [center_uid]
    if date_from:
        sql += " AND a.ts >= ?"
        params.append(date_from)
    if date_to:
        sql += " AND a.ts <= ?"
        params.append(date_to + "T23:59:59")
    if query:
        sql += " AND (l.name LIKE ? OR l.roll_no LIKE ?)"
        params.extend([f"%{query}%", f"%{query}%"])
    sql += " ORDER BY a.ts DESC"
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return rows


def attendance_stats(center_uid):
    conn = get_conn()
    today = datetime.date.today().isoformat()
    total_learners = conn.execute(
        "SELECT COUNT(*) c FROM learners WHERE center_uid = ?", (center_uid,)
    ).fetchone()["c"]
    present_today = conn.execute(
        "SELECT COUNT(DISTINCT luid) c FROM attendance_logs WHERE center_uid = ? AND ts LIKE ?",
        (center_uid, f"{today}%"),
    ).fetchone()["c"]
    conn.close()
    return {"total_learners": total_learners, "present_today": present_today}


def get_attendance_calendar(luid, year, month):
    """Per-day login/logout times (HH:MM:SS) for one learner in one calendar month,
    for the learner-portal attendance calendar. Login = earliest login punch that day,
    logout = latest logout punch that day."""
    conn = get_conn()
    prefix = f"{year:04d}-{month:02d}"
    rows = conn.execute(
        "SELECT ts, punch_type FROM attendance_logs WHERE luid = ? AND ts LIKE ? ORDER BY ts",
        (luid, f"{prefix}%"),
    ).fetchall()
    conn.close()
    by_day = {}
    for r in rows:
        day = r["ts"][8:10]
        entry = by_day.setdefault(day, {"login": None, "logout": None})
        t = r["ts"][11:19]
        if r["punch_type"] == "login":
            if entry["login"] is None or t < entry["login"]:
                entry["login"] = t
        elif r["punch_type"] == "logout":
            if entry["logout"] is None or t > entry["logout"]:
                entry["logout"] = t
    return by_day


def parse_dump_date(s):
    """Parse an Attendance Dump's 'Day of dates' column ('September 1, 2026' etc.)
    into a date. Returns None if unparseable."""
    if not s:
        return None
    s = s.strip()
    for fmt in ("%B %d, %Y", "%b %d, %Y", "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"):
        try:
            return datetime.datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def parse_dump_datetime(s):
    """Parse an Attendance Dump's log_in_time/log_out_time column
    ('9/1/2026 6:46:25 AM' etc.) into a datetime. Returns None if blank or
    unparseable (a learner who only punched one side of the day, e.g.)."""
    if not s or not s.strip():
        return None
    s = s.strip()
    for fmt in ("%m/%d/%Y %I:%M:%S %p", "%m/%d/%Y %H:%M:%S", "%d/%m/%Y %I:%M:%S %p",
                "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def apply_attendance_dump(rows):
    """rows: list of dicts with learner_uid, date (a date object), log_in (datetime or
    None), log_out (datetime or None) - already parsed by the caller.

    Every row is first grouped by (learner, date) and reduced to a single earliest
    login / latest logout for that day, exactly like the calendar itself aggregates
    live scanner punches - so a CSV with more than one row for the same learner on
    the same day (repeat taps, a login-only row and a separate logout-only row, plain
    duplicates) is merged correctly instead of each row deleting and replacing
    whatever the row before it just wrote. Only after that merge does each (learner,
    date) get written: one full override - every existing attendance_logs entry for
    that learner on that day, whether from the face scanner or a previous dump, is
    replaced by the merged result.

    The stored timestamp's date always comes from 'Day of dates', not whatever date
    happens to be embedded in log_in_time/log_out_time - only the time-of-day from
    those columns is used - so the row can never end up filed under a different day
    than the one it was just deleted from, even if the source export's two date
    fields ever disagree.

    Matching is case-insensitive (a dump's learner_uid casing doesn't always match how
    the LUID was typed in at enrollment), and the row is always filed under the
    learner's own canonical luid so it lines up with what the attendance calendar
    looks up later. A learner not found at all is skipped (there's no center_uid to
    file the row under), and its learner_uid is returned so admin can see exactly
    which ones didn't match rather than just a bare count.

    Returns (days_updated, skipped_count, skipped_learner_uids)."""
    conn = get_conn()

    # ---- Pass 1: resolve each row's learner and merge same (learner, date) rows ----
    merged = {}  # (canonical_luid, date_iso) -> {"center_uid", "login_dt", "logout_dt"}
    skipped = 0
    skipped_uids = []
    luid_cache = {}

    for r in rows:
        key_uid = r["learner_uid"].lower()
        if key_uid not in luid_cache:
            luid_cache[key_uid] = conn.execute(
                "SELECT luid, center_uid FROM learners WHERE lower(luid) = ?", (key_uid,)
            ).fetchone()
        learner = luid_cache[key_uid]
        if not learner:
            skipped += 1
            if len(skipped_uids) < 50:
                skipped_uids.append(r["learner_uid"])
            continue

        canonical_luid = learner["luid"]
        date_obj = r["date"]
        group_key = (canonical_luid, date_obj.isoformat())
        entry = merged.setdefault(group_key, {
            "center_uid": learner["center_uid"], "date": date_obj, "login_dt": None, "logout_dt": None,
        })

        if r["log_in"]:
            login_dt = datetime.datetime.combine(date_obj, r["log_in"].time())
            if entry["login_dt"] is None or login_dt < entry["login_dt"]:
                entry["login_dt"] = login_dt
        if r["log_out"]:
            logout_dt = datetime.datetime.combine(date_obj, r["log_out"].time())
            if entry["logout_dt"] is None or logout_dt > entry["logout_dt"]:
                entry["logout_dt"] = logout_dt

    # ---- Pass 2: one delete + write per (learner, date) group ----
    updated = 0
    for (canonical_luid, date_iso), entry in merged.items():
        conn.execute(
            "DELETE FROM attendance_logs WHERE luid = ? AND ts LIKE ?",
            (canonical_luid, f"{date_iso}%"),
        )
        if entry["login_dt"]:
            conn.execute(
                "INSERT INTO attendance_logs (luid, center_uid, ts, confidence, status, punch_type) VALUES (?,?,?,?,?,?)",
                (canonical_luid, entry["center_uid"], entry["login_dt"].isoformat(timespec="seconds"), 1.0, "present", "login"),
            )
        if entry["logout_dt"]:
            conn.execute(
                "INSERT INTO attendance_logs (luid, center_uid, ts, confidence, status, punch_type) VALUES (?,?,?,?,?,?)",
                (canonical_luid, entry["center_uid"], entry["logout_dt"].isoformat(timespec="seconds"), 1.0, "present", "logout"),
            )
        updated += 1

    conn.execute(
        "INSERT INTO attendance_dump_uploads (uploaded_at, row_count, days_updated, skipped) VALUES (?,?,?,?)",
        (now(), len(rows), updated, skipped),
    )
    conn.commit()
    conn.close()
    return updated, skipped, skipped_uids


def list_attendance_dump_uploads():
    conn = get_conn()
    rows = conn.execute("SELECT * FROM attendance_dump_uploads ORDER BY id DESC").fetchall()
    conn.close()
    return rows


# ---------- Atlas dump (learner directory, rolling 3-upload retention) ----------

def save_atlas_upload(rows):
    """rows: list of dicts (already de-duplicated by the caller) with keys batch_name,
    centre_city_name, centre_name, goal_name, learner_name, learner_uid, roll_number.
    Inserts a new upload batch, then keeps only the 3 most recent uploads, permanently
    deleting anything older (4th upload evicts the oldest of the previous 3), then
    rebuilds the atlas_latest materialized table so reads never have to recompute the
    'latest per learner' aggregate themselves."""
    conn = get_conn()
    cur = conn.execute("INSERT INTO atlas_uploads (uploaded_at, row_count) VALUES (?,?)", (now(), len(rows)))
    upload_id = cur.lastrowid
    conn.executemany(
        """INSERT INTO atlas_records (upload_id, batch_name, centre_city_name, centre_name,
           goal_name, learner_name, learner_uid, roll_number) VALUES (?,?,?,?,?,?,?,?)""",
        [(upload_id, r.get("batch_name") or None, r.get("centre_city_name") or None,
          r.get("centre_name") or None, r.get("goal_name") or None, r.get("learner_name") or None,
          r["learner_uid"], r.get("roll_number") or None) for r in rows],
    )
    evicted = [row["id"] for row in conn.execute(
        "SELECT id FROM atlas_uploads ORDER BY id DESC LIMIT -1 OFFSET 3"
    ).fetchall()]
    for oid in evicted:
        conn.execute("DELETE FROM atlas_records WHERE upload_id = ?", (oid,))
        conn.execute("DELETE FROM atlas_uploads WHERE id = ?", (oid,))

    # Rebuild the materialized "latest per learner" table once here (an admin action,
    # rare) instead of every dashboard/report read (frequent) recomputing the same
    # aggregate from scratch - this was the main cause of slow page loads. Uses
    # ROW_NUMBER() rather than a GROUP BY self-join back onto atlas_records: the join
    # form measured 1000x+ slower here at real-world data volumes for reasons that
    # didn't reproduce on the (fast) GROUP BY or window-function scan in isolation -
    # something about this SQLite build's plan for that specific join shape.
    conn.execute("DELETE FROM atlas_latest")
    conn.execute(
        """INSERT INTO atlas_latest (learner_uid, batch_name, centre_city_name, centre_name,
           goal_name, learner_name, roll_number, upload_id)
           SELECT learner_uid, batch_name, centre_city_name, centre_name, goal_name, learner_name,
                  roll_number, upload_id
           FROM (
               SELECT learner_uid, batch_name, centre_city_name, centre_name, goal_name, learner_name,
                      roll_number, upload_id,
                      ROW_NUMBER() OVER (PARTITION BY learner_uid ORDER BY upload_id DESC) AS rn
               FROM atlas_records
           ) WHERE rn = 1"""
    )
    conn.commit()
    conn.close()
    return upload_id, len(rows), len(evicted)


def list_atlas_uploads():
    conn = get_conn()
    rows = conn.execute("SELECT * FROM atlas_uploads ORDER BY id DESC").fetchall()
    conn.close()
    return rows


def get_latest_atlas_map():
    """learner_uid -> that learner's most recent atlas record. Reads the atlas_latest
    materialized table (rebuilt once per upload in save_atlas_upload) instead of
    recomputing the aggregate on every call - this is on the hot path of the learner
    dashboard, the center learner list, and center reports, so it needs to be a flat,
    fast scan rather than a GROUP BY/JOIN done fresh every page view."""
    conn = get_conn()
    rows = conn.execute("SELECT * FROM atlas_latest").fetchall()
    conn.close()
    return {r["learner_uid"]: r for r in rows}


def get_latest_atlas_for_learner(learner_uid):
    """Single indexed PK lookup (learner_uid is atlas_latest's primary key) rather than
    building the whole map just to read one entry - this is what the learner
    dashboard and the center 'show more' modal call on every page view."""
    conn = get_conn()
    row = conn.execute("SELECT * FROM atlas_latest WHERE learner_uid = ?", (learner_uid,)).fetchone()
    conn.close()
    return row


# ---------- Test dump (rolling 3-upload retention) ----------

def save_test_upload(rows):
    conn = get_conn()
    cur = conn.execute("INSERT INTO test_uploads (uploaded_at, row_count) VALUES (?,?)", (now(), len(rows)))
    upload_id = cur.lastrowid
    conn.executemany(
        """INSERT INTO test_records (upload_id, learner_id, learner_uid, learner_name, roll_no,
           test_uid, test_title, test_start_date, course_name, course_uid, test_score, attempt_type,
           rank_on_score, learner_test_start_at, learner_test_end_at, learner_batch_uid,
           learner_batch_name, centre_name, learner_center_uid, center_city, section_name,
           total_ques_attempted, section_score)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        [(upload_id, r.get("learner_id"), r.get("learner_uid"), r.get("learner_name"), r.get("roll_no"),
          r.get("test_uid"), r.get("test_title"), r.get("test_start_date"), r.get("course_name"),
          r.get("course_uid"), r.get("test_score"), r.get("attempt_type"), r.get("rank_on_score"),
          r.get("learner_test_start_at"), r.get("learner_test_end_at"), r.get("learner_batch_uid"),
          r.get("learner_batch_name"), r.get("centre_name"), r.get("learner_center_uid"),
          r.get("center_city"), r.get("section_name"), r.get("total_ques_attempted"),
          r.get("section_score")) for r in rows],
    )
    evicted = [row["id"] for row in conn.execute(
        "SELECT id FROM test_uploads ORDER BY id DESC LIMIT -1 OFFSET 3"
    ).fetchall()]
    for oid in evicted:
        conn.execute("DELETE FROM test_records WHERE upload_id = ?", (oid,))
        conn.execute("DELETE FROM test_uploads WHERE id = ?", (oid,))
    conn.commit()
    conn.close()
    return upload_id, len(rows), len(evicted)


def list_test_uploads():
    conn = get_conn()
    rows = conn.execute("SELECT * FROM test_uploads ORDER BY id DESC").fetchall()
    conn.close()
    return rows


def get_test_records_for_learner(learner_uid):
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM test_records WHERE learner_uid = ? ORDER BY id DESC", (learner_uid,)
    ).fetchall()
    conn.close()
    return rows


def get_test_records_for_center(center_uid):
    """Matched via learners.center_uid (our own registry), not the free-text
    'Learner Center UID' column in the dump, which comes from an external system and
    isn't guaranteed to line up with our center_uid values."""
    conn = get_conn()
    rows = conn.execute(
        """SELECT tr.* FROM test_records tr
           JOIN learners l ON l.luid = tr.learner_uid
           WHERE l.center_uid = ?
           ORDER BY tr.id DESC""",
        (center_uid,),
    ).fetchall()
    conn.close()
    return rows


# ---------- NPS dump (kept in full; reports filter by date range) ----------

def save_nps_upload(rows):
    conn = get_conn()
    cur = conn.execute("INSERT INTO nps_uploads (uploaded_at, row_count) VALUES (?,?)", (now(), len(rows)))
    upload_id = cur.lastrowid
    conn.executemany(
        """INSERT INTO nps_records (upload_id, feedback_date, feedback_date_norm, city_name, centre_name,
           category, version, learner_uid, learner_id, learner_name, learner_username, state_name,
           first_feedback_flag, feedback_platform, feedback_review, rating, flag_app_tech,
           flag_class_recording, flag_staff_cooperation, flag_study_material, flag_tests,
           flag_quality_educators, flag_syllabus_progress, flag_centre_facilities)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        [(upload_id, r.get("feedback_date"), parse_date_flexible(r.get("feedback_date")),
          r.get("city_name"), r.get("centre_name"), r.get("category"), r.get("version"),
          r.get("learner_uid"), r.get("learner_id"), r.get("learner_name"), r.get("learner_username"),
          r.get("state_name"), r.get("first_feedback_flag"), r.get("feedback_platform"),
          r.get("feedback_review"), r.get("rating"), r.get("flag_app_tech"),
          r.get("flag_class_recording"), r.get("flag_staff_cooperation"), r.get("flag_study_material"),
          r.get("flag_tests"), r.get("flag_quality_educators"), r.get("flag_syllabus_progress"),
          r.get("flag_centre_facilities")) for r in rows],
    )
    conn.commit()
    conn.close()
    return upload_id, len(rows)


def list_nps_uploads():
    conn = get_conn()
    rows = conn.execute("SELECT * FROM nps_uploads ORDER BY id DESC").fetchall()
    conn.close()
    return rows


def get_nps_for_learner(learner_uid, date_from=None, date_to=None):
    conn = get_conn()
    sql = "SELECT * FROM nps_records WHERE learner_uid = ?"
    params = [learner_uid]
    if date_from:
        sql += " AND (feedback_date_norm >= ? OR feedback_date_norm IS NULL)"
        params.append(date_from)
    if date_to:
        sql += " AND (feedback_date_norm <= ? OR feedback_date_norm IS NULL)"
        params.append(date_to)
    sql += " ORDER BY feedback_date_norm DESC, id DESC"
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return rows


def get_nps_for_center(center_uid, date_from=None, date_to=None):
    """The admin-uploaded NPS dump, visible to a center by matching the dump's own
    centre_name field against this center's registered name (case/whitespace
    insensitive) - NPS learner_uids come from an external export and won't reliably
    line up with our own learners table, so name matching is the primary path, with a
    learners-table join as a fallback for anything the name match misses.
    This never includes the in-app learner Feedback tab (learner_feedback table) -
    that stays admin-only, regardless of this matching."""
    conn = get_conn()
    center = conn.execute("SELECT centre_name FROM centers WHERE center_uid = ?", (center_uid,)).fetchone()
    if not center:
        conn.close()
        return []
    sql = """SELECT DISTINCT nr.* FROM nps_records nr
             LEFT JOIN learners l ON l.luid = nr.learner_uid
             WHERE (lower(trim(nr.centre_name)) = lower(trim(?))
                    OR (l.luid IS NOT NULL AND l.center_uid = ?))"""
    params = [center["centre_name"], center_uid]
    if date_from:
        sql += " AND (nr.feedback_date_norm >= ? OR nr.feedback_date_norm IS NULL)"
        params.append(date_from)
    if date_to:
        sql += " AND (nr.feedback_date_norm <= ? OR nr.feedback_date_norm IS NULL)"
        params.append(date_to)
    sql += " ORDER BY nr.feedback_date_norm DESC, nr.id DESC"
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return rows


NPS_FLAG_COLUMNS = [
    ("flag_app_tech", "App and Technology"),
    ("flag_class_recording", "Class Recording"),
    ("flag_staff_cooperation", "Staff Co-operation"),
    ("flag_study_material", "Study Material"),
    ("flag_tests", "Tests"),
    ("flag_quality_educators", "Quality of Educators"),
    ("flag_syllabus_progress", "Syllabus Progress"),
    ("flag_centre_facilities", "Centre facilities cleanliness"),
]


def summarize_nps(rows):
    """Aggregate a set of nps_records rows: total responses, unique learners, a 1-5
    rating histogram, and a like/dislike/none count for each of the 8 flag columns.
    Flag values are read heuristically ('dislike' anywhere in the text wins over
    'like', since it's a substring of it; anything else counts as 'none') since the
    exact encoding used by the source export wasn't available to pin down exactly."""
    total = len(rows)
    unique_learners = len({r["learner_uid"] for r in rows if r["learner_uid"]})
    rating_counts = {str(i): 0 for i in range(1, 6)}
    for r in rows:
        rt = (r["rating"] or "").strip()
        if rt in rating_counts:
            rating_counts[rt] += 1
        else:
            try:
                v = str(int(float(rt)))
                if v in rating_counts:
                    rating_counts[v] += 1
            except (TypeError, ValueError):
                pass
    flags = []
    for col, label in NPS_FLAG_COLUMNS:
        like = dislike = none = 0
        for r in rows:
            v = (r[col] or "").strip().lower()
            if "dislike" in v:
                dislike += 1
            elif "like" in v:
                like += 1
            else:
                none += 1
        flags.append({"label": label, "like": like, "dislike": dislike, "none": none})
    return {"total": total, "unique_learners": unique_learners, "rating_counts": rating_counts, "flags": flags}


# ---------- In-app learner feedback ----------

def get_last_feedback(luid):
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM learner_feedback WHERE luid = ? ORDER BY submitted_at DESC LIMIT 1", (luid,)
    ).fetchone()
    conn.close()
    return row


def add_feedback(luid, rating, flags, remark):
    conn = get_conn()
    conn.execute(
        """INSERT INTO learner_feedback (luid, submitted_at, rating, flag_app_tech, flag_class_recording,
           flag_staff_cooperation, flag_study_material, flag_tests, flag_quality_educators,
           flag_syllabus_progress, flag_centre_facilities, remark) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        (luid, now(), rating, flags.get("app_tech"), flags.get("class_recording"),
         flags.get("staff_cooperation"), flags.get("study_material"), flags.get("tests"),
         flags.get("quality_educators"), flags.get("syllabus_progress"), flags.get("centre_facilities"),
         (remark or "").strip() or None),
    )
    conn.commit()
    conn.close()


def get_feedback_responses(date_from=None, date_to=None):
    conn = get_conn()
    sql = """SELECT lf.*, l.name AS learner_name, l.roll_no, l.center_uid, c.centre_name
             FROM learner_feedback lf
             JOIN learners l ON l.luid = lf.luid
             LEFT JOIN centers c ON c.center_uid = l.center_uid
             WHERE 1=1"""
    params = []
    if date_from:
        sql += " AND lf.submitted_at >= ?"
        params.append(date_from)
    if date_to:
        sql += " AND lf.submitted_at <= ?"
        params.append(date_to + "T23:59:59")
    sql += " ORDER BY lf.submitted_at DESC"
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return rows


def get_feedback_center_averages(date_from=None, date_to=None):
    """Average learner-given rating per center, from the app's own Feedback tab
    (learner_feedback), not the admin-uploaded NPS dump."""
    conn = get_conn()
    sql = """SELECT c.centre_name AS centre_name, AVG(lf.rating) AS avg_rating, COUNT(*) AS n
             FROM learner_feedback lf
             JOIN learners l ON l.luid = lf.luid
             LEFT JOIN centers c ON c.center_uid = l.center_uid
             WHERE 1=1"""
    params = []
    if date_from:
        sql += " AND lf.submitted_at >= ?"
        params.append(date_from)
    if date_to:
        sql += " AND lf.submitted_at <= ?"
        params.append(date_to + "T23:59:59")
    sql += " GROUP BY c.centre_name ORDER BY avg_rating DESC"
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return rows


# ---------- Admin-wide reporting (city / center wise) ----------

def report_rows(city=None, center_uid=None, date_from=None, date_to=None, query=None):
    """Detailed attendance rows across every center, joined with city/centre name."""
    conn = get_conn()
    sql = """SELECT a.id, a.ts, a.confidence, a.status, a.punch_type, l.name, l.roll_no, l.luid,
                     c.center_uid, c.centre_name, c.city
              FROM attendance_logs a
              JOIN learners l ON l.luid = a.luid
              JOIN centers c ON c.center_uid = a.center_uid
              WHERE 1=1"""
    params = []
    if city:
        sql += " AND c.city = ?"
        params.append(city)
    if center_uid:
        sql += " AND c.center_uid = ?"
        params.append(center_uid)
    if date_from:
        sql += " AND a.ts >= ?"
        params.append(date_from)
    if date_to:
        sql += " AND a.ts <= ?"
        params.append(date_to + "T23:59:59")
    if query:
        sql += " AND (l.name LIKE ? OR l.roll_no LIKE ?)"
        params.extend([f"%{query}%", f"%{query}%"])
    sql += " ORDER BY a.ts DESC"
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return rows


def report_summary_by_city(date_from=None, date_to=None):
    conn = get_conn()
    # learner totals per city (independent of date filter)
    learner_totals = {r["city"]: r["total_learners"] for r in conn.execute(
        "SELECT c.city AS city, COUNT(l.luid) AS total_learners "
        "FROM centers c LEFT JOIN learners l ON l.center_uid = c.center_uid GROUP BY c.city"
    ).fetchall()}

    att_sql = """SELECT c.city AS city, COUNT(DISTINCT a.luid || '|' || substr(a.ts,1,10)) AS present_count
                 FROM attendance_logs a JOIN centers c ON c.center_uid = a.center_uid WHERE 1=1"""
    params = []
    if date_from:
        att_sql += " AND a.ts >= ?"
        params.append(date_from)
    if date_to:
        att_sql += " AND a.ts <= ?"
        params.append(date_to + "T23:59:59")
    att_sql += " GROUP BY c.city"
    present_by_city = {r["city"]: r["present_count"] for r in conn.execute(att_sql, params).fetchall()}
    conn.close()

    cities = sorted(set(learner_totals) | set(present_by_city))
    return [
        {
            "city": city,
            "total_learners": learner_totals.get(city, 0),
            "present_count": present_by_city.get(city, 0),
        }
        for city in cities
    ]


def report_summary_by_center(city=None, date_from=None, date_to=None):
    conn = get_conn()
    center_sql = "SELECT center_uid, centre_name, city FROM centers"
    center_params = []
    if city:
        center_sql += " WHERE city = ?"
        center_params.append(city)
    centers = conn.execute(center_sql, center_params).fetchall()

    learner_counts = {r["center_uid"]: r["c"] for r in conn.execute(
        "SELECT center_uid, COUNT(*) c FROM learners GROUP BY center_uid"
    ).fetchall()}

    att_sql = "SELECT center_uid, COUNT(DISTINCT luid || '|' || substr(ts,1,10)) c FROM attendance_logs WHERE 1=1"
    att_params = []
    if date_from:
        att_sql += " AND ts >= ?"
        att_params.append(date_from)
    if date_to:
        att_sql += " AND ts <= ?"
        att_params.append(date_to + "T23:59:59")
    att_sql += " GROUP BY center_uid"
    present_counts = {r["center_uid"]: r["c"] for r in conn.execute(att_sql, att_params).fetchall()}
    conn.close()

    return [
        {
            "center_uid": c["center_uid"],
            "centre_name": c["centre_name"],
            "city": c["city"],
            "total_learners": learner_counts.get(c["center_uid"], 0),
            "present_count": present_counts.get(c["center_uid"], 0),
        }
        for c in centers
    ]


def list_cities():
    conn = get_conn()
    rows = conn.execute("SELECT DISTINCT city FROM centers ORDER BY city").fetchall()
    conn.close()
    return [r["city"] for r in rows]
