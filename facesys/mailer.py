"""
Attendance email notifications.

Sends the parent/learner notification email on a background thread so the
/api/scan request never blocks on SMTP round-trip time. Configure the
sending account via environment variables (see config.py):
    SMTP_HOST, SMTP_PORT, SMTP_USE_TLS, SMTP_USERNAME, SMTP_PASSWORD,
    EMAIL_FROM, EMAIL_FROM_NAME, EMAIL_ENABLED
"""
import datetime
import logging
import re
import smtplib
import threading
from email.message import EmailMessage

import config

logger = logging.getLogger("mailer")

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

BODY_TEMPLATE = (
    "Hi {name} Parent,\n\n"
    "Your ward has {action} at {timestamp}.\n\n"
    "Thanks & regards\n"
    "{sender_name}"
)

ACTION_TEXT = {
    "login": "arrived at center",
    "logout": "left the center",
}


def is_valid_email(email):
    return bool(email) and bool(EMAIL_RE.match(email.strip()))


def format_timestamp(ts=None):
    """dd/mm/yyyy HH:MM:SS — ts accepts a datetime, an ISO string, or None (=now)."""
    if ts is None:
        dt = datetime.datetime.now()
    elif isinstance(ts, str):
        dt = datetime.datetime.fromisoformat(ts)
    else:
        dt = ts
    return dt.strftime("%d/%m/%Y %H:%M:%S")


def _send_now(to_email, learner_name, timestamp_str, action, smtp_username, smtp_password, from_email, from_name):
    if not smtp_username or not smtp_password:
        logger.error("Attendance email skipped for %s: SMTP credentials not configured", to_email)
        return

    msg = EmailMessage()
    msg["Subject"] = config.EMAIL_SUBJECT
    msg["From"] = f"{from_name} <{from_email}>"
    msg["To"] = to_email
    msg.set_content(BODY_TEMPLATE.format(
        name=learner_name, action=ACTION_TEXT.get(action, action), timestamp=timestamp_str, sender_name=from_name
    ))

    try:
        with smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT, timeout=config.EMAIL_TIMEOUT) as smtp:
            if config.SMTP_USE_TLS:
                smtp.starttls()
            smtp.login(smtp_username, smtp_password)
            smtp.send_message(msg)
        logger.info("Attendance email sent to %s from %s", to_email, from_email)
    except Exception:
        logger.exception("Attendance email failed for %s", to_email)


def send_attendance_email_async(
    to_email, learner_name, timestamp=None, action="login",
    smtp_username=None, smtp_password=None, from_email=None, from_name=None,
):
    """Fire-and-forget: queues the send on a daemon thread, returns immediately.
    No-op (logged) if EMAIL_ENABLED is false or the address is missing/invalid.

    smtp_username/smtp_password/from_email let a caller route the mail through a
    center-specific mailbox (e.g. the center's own Syncup email + its app password)
    instead of the global SMTP account in config.py. If any of these are omitted,
    the global config values are used instead.
    """
    if not config.EMAIL_ENABLED:
        return
    if not is_valid_email(to_email):
        logger.warning("Attendance email skipped: no valid email for learner '%s'", learner_name)
        return

    smtp_username = smtp_username or config.SMTP_USERNAME
    smtp_password = smtp_password or config.SMTP_PASSWORD
    from_email = from_email or config.EMAIL_FROM or smtp_username
    from_name = from_name or config.EMAIL_FROM_NAME

    ts_str = format_timestamp(timestamp)
    threading.Thread(
        target=_send_now,
        args=(to_email.strip(), learner_name, ts_str, action, smtp_username, smtp_password, from_email, from_name),
        daemon=True,
    ).start()
