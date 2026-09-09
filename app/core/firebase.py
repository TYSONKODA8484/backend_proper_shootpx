import firebase_admin
from firebase_admin import credentials, auth

from app.core.config import settings

# Guard against "default app already exists" — this module can be imported more
# than once (uvicorn --reload, test collection, multiple routers).
if not firebase_admin._apps:
    cred = credentials.Certificate(settings.firebase_credentials_path)
    firebase_admin.initialize_app(cred)


def verify_token(id_token: str) -> dict:
    """Verify a Firebase ID token. Raises on invalid/expired/revoked tokens."""
    return auth.verify_id_token(id_token)
