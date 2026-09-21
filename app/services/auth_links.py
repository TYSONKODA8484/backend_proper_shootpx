from firebase_admin import auth as firebase_auth

from app.core.cache import acquire_cooldown, clear_cached
from app.core.email import send_email

# Only per-IP rate limiting (slowapi, in the route) previously guarded this
# endpoint -- trivial to rotate past. This is a second, per-TARGET-email
# throttle so one inbox can't be email-bombed by an attacker spreading
# requests across IPs.
SIGN_IN_LINK_COOLDOWN_SECONDS = 60


class SignInLinkRateLimitedError(Exception):
    """A sign-in link was already requested for this email too recently."""


def send_sign_in_link(email: str, continue_url: str) -> None:
    """Generate a Firebase email sign-in link and email it to the user."""
    email = email.strip().lower()
    cooldown_key = f"authmail:cooldown:{email}"
    if not acquire_cooldown(cooldown_key, SIGN_IN_LINK_COOLDOWN_SECONDS):
        raise SignInLinkRateLimitedError(
            "A sign-in link was already sent to this email recently. Please wait a moment and try again."
        )

    try:
        action_code_settings = firebase_auth.ActionCodeSettings(
            url=continue_url,
            handle_code_in_app=True,
        )

        link = firebase_auth.generate_sign_in_with_email_link(email, action_code_settings)

        send_email(
            to=email,
            subject="Your ShootPX sign-in link",
            html=(
                "<p>Click below to sign in:</p>"
                f"<p><a href='{link}'>Sign in to ShootPX</a></p>"
            ),
        )
    except Exception:
        # No email actually went out -- don't let the failed attempt burn the
        # user's cooldown slot and lock them out of a real retry.
        clear_cached(cooldown_key)
        raise
