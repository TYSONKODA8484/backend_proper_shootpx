from firebase_admin import auth as firebase_auth

from app.core.email import send_email


def send_sign_in_link(email: str, continue_url: str) -> None:
    """Generate a Firebase email sign-in link and email it to the user."""
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
