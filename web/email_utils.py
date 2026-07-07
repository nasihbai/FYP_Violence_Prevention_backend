"""
Auth email helpers — verification-link emails for POST /auth/register and
POST /auth/resend-verification. Standalone module (not alerts.alert_system.EmailAlert,
which is shaped around AlertManager's recipient-list/alert-formatted use case).
"""
import logging
import secrets
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from config import EmailConfig

logger = logging.getLogger(__name__)


def generate_verification_token() -> str:
    return secrets.token_urlsafe(32)


def build_verification_link(token: str) -> str:
    return f"{EmailConfig.FRONTEND_URL}/verify-email?token={token}"


def send_verification_email(to_email: str, username: str, token: str) -> bool:
    """
    Send the verification email, or (dev mode / EMAIL_ENABLED=false) log the
    link instead. Never raises — a failed send must not break registration,
    since the token is already persisted and /auth/resend-verification exists
    as a recovery path. Returns True if actually sent via SMTP, False if only logged.
    """
    link = build_verification_link(token)

    if not EmailConfig.ENABLED:
        # .warning (not .info) so this is visible via Python's default handler
        # even on entry points that never call logging.basicConfig().
        logger.warning(f"[DEV-EMAIL] Verification link for {to_email}: {link}")
        return False

    try:
        msg = MIMEMultipart()
        msg["From"] = EmailConfig.SENDER
        msg["To"] = to_email
        msg["Subject"] = "Verify your EYES of Soteria account"
        body = (
            f"Hi {username},\n\n"
            f"Please verify your email address by clicking the link below:\n\n"
            f"{link}\n\n"
            f"This link expires in {EmailConfig.VERIFICATION_TOKEN_TTL_HOURS} hours.\n"
            f"If you did not create this account, you can ignore this email.\n"
        )
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP(EmailConfig.SMTP_SERVER, EmailConfig.SMTP_PORT) as server:
            server.starttls()
            server.login(EmailConfig.SENDER, EmailConfig.PASSWORD)
            server.send_message(msg)

        logger.info(f"Verification email sent to {to_email}")
        return True
    except Exception as exc:
        logger.error(f"Failed to send verification email to {to_email}: {exc}")
        return False
