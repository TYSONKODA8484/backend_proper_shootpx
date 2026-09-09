import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from app.core.config import settings

logger = logging.getLogger(__name__)


def _html_to_text(html: str) -> str:
    import re

    text = re.sub(r"<[^>]+>", " ", html)
    return re.sub(r"\s+", " ", text).strip()


def send_email(to: str, subject: str, html: str, text: str | None = None) -> None:
    """Send a multipart (text + HTML) email via SMTP. Raises on failure."""
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = f"{settings.email_from_name} <{settings.email_from_address}>"
    msg["To"] = to

    msg.attach(MIMEText(text or _html_to_text(html), "plain"))
    msg.attach(MIMEText(html, "html"))

    with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=20) as server:
        server.starttls()
        server.login(settings.smtp_user, settings.smtp_password)
        server.send_message(msg)

    logger.info("email accepted by SMTP  to=%s  subject=%r", to, subject)
