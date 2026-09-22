import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "db", "attendance.db")
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
VIDEO_TMP_DIR = os.path.join(BASE_DIR, "static", "uploads", "videos")

os.makedirs(os.path.join(BASE_DIR, "db"), exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VIDEO_TMP_DIR, exist_ok=True)

# --- Recognition tuning ---
MATCH_THRESHOLD = float(os.environ.get("MATCH_THRESHOLD", 0.62))   # cosine similarity, 0-1, higher = stricter
MIN_DETECT_PROB = 0.90            # MTCNN detection confidence to accept a face
BLUR_THRESHOLD = 60.0             # Laplacian variance; frames below this are rejected as too blurry
MAX_EMBEDDINGS_PER_LEARNER = 6    # diverse embeddings stored per learner (pose/lighting coverage)
ENROLL_SAMPLE_SECONDS = 1.0       # sample one frame per this many seconds of enrollment video
ATTENDANCE_DEDUPE_HOURS = 12      # don't log a second "present" for the same learner within this window

# --- Admin auth (env override recommended for real deployment) ---
ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin123")

SECRET_KEY = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me-in-production")

ALLOWED_VIDEO_EXT = {"mp4", "mov", "avi", "mkv", "webm"}

# Keep raw enrollment video after processing? (biometric data retention has compliance implications)
RETAIN_ENROLLMENT_VIDEO = False

# --- Bulk import (Google Drive fetch) ---
GDRIVE_DOWNLOAD_TIMEOUT = 120         # seconds, per video
GDRIVE_MAX_BYTES = 500 * 1024 * 1024  # 500MB hard cap per fetched video
BULK_MAX_ROWS = 1000                  # refuse CSVs bigger than this in one go

# --- Live camera face registration (guided front/left/right capture) ---
CAPTURE_MAX_FRAMES = 60               # hard cap on frames accepted from one registration session

# --- Learner feedback (in-app NPS-style feedback form) ---
FEEDBACK_COOLDOWN_DAYS = 15   # a learner can submit feedback at most once per this many days

# --- Upload / request size limits ---
# Werkzeug 3.x caps non-file multipart form fields at 500KB by default (max_form_memory_size),
# which the live-capture JSON blows past if it's ever sent as a plain field again - keep this
# generous as a backstop even though the capture payload now travels as a file part.
MAX_CONTENT_LENGTH = 1024 * 1024 * 1024     # 1GB hard cap per request - not a row-count limit;
                                             # just a basic memory-exhaustion backstop, well above
                                             # what any real Atlas/Test/NPS CSV should ever reach
MAX_FORM_MEMORY_SIZE = 8 * 1024 * 1024      # 8MB headroom for any non-file form field

# --- Far-field / multi-face scan tuning (attendance scanner, not enrollment) ---
SCAN_MIN_DETECT_PROB = 0.80    # looser than enrollment: small/far faces score lower
SCAN_MIN_FACE_PX = 40          # ignore boxes smaller than this (noise, not a real far face)
SCAN_BLUR_THRESHOLD = 35.0     # looser than BLUR_THRESHOLD after enhancement is applied
MTCNN_MIN_FACE_SIZE = 25       # smaller than facenet-pytorch default (20->25 tradeoff tuned for group shots)
ENHANCE_UPSCALE_MAX_WIDTH = 1920  # enhancement upscales small frames up to this width before detection

# --- Attendance email notifications ---
EMAIL_ENABLED = os.environ.get("EMAIL_ENABLED", "true").lower() == "true"
SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", 587))
SMTP_USE_TLS = os.environ.get("SMTP_USE_TLS", "true").lower() == "true"
SMTP_USERNAME = os.environ.get("SMTP_USERNAME", "")          # your sending email address
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "")          # app password / SMTP secret, never hardcode
EMAIL_FROM = os.environ.get("EMAIL_FROM", SMTP_USERNAME)
EMAIL_FROM_NAME = os.environ.get("EMAIL_FROM_NAME", "Unacademy Team")
EMAIL_SUBJECT = os.environ.get("EMAIL_SUBJECT", "Attendance Confirmation")
EMAIL_TIMEOUT = int(os.environ.get("EMAIL_TIMEOUT", 15))     # seconds, per SMTP connection attempt
