import base64
import hashlib
import hmac
import json
import secrets
import time
from dataclasses import dataclass

from fastapi import Request

from core.config import settings


AUTH_QUERY_PARAM = "access_token"


@dataclass(frozen=True)
class OwnerSession:
    username: str
    expires_at: int
    authenticated_via: str


def is_private_access_enabled() -> bool:
    return settings.private_access_enabled and bool(settings.owner_password)


def _b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("utf-8").rstrip("=")


def _b64decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(f"{value}{padding}".encode("utf-8"))


def _get_signing_secret() -> str:
    return settings.session_secret or settings.admin_api_token or settings.owner_password


def _get_request_token(request: Request) -> str:
    auth_header = request.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        return auth_header[7:].strip()

    for header_name in ("x-solollm-admin-token", "x-admin-token"):
        header_value = request.headers.get(header_name, "").strip()
        if header_value:
            return header_value

    return request.query_params.get(AUTH_QUERY_PARAM, "").strip()


def verify_owner_credentials(username: str, password: str) -> bool:
    if not is_private_access_enabled():
        return False
    return secrets.compare_digest(username.strip(), settings.owner_username) and secrets.compare_digest(password, settings.owner_password)


def issue_owner_session_token() -> tuple[str, int]:
    now = int(time.time())
    expires_at = now + max(settings.session_ttl_hours, 1) * 3600
    payload = {
        "sub": settings.owner_username,
        "iat": now,
        "exp": expires_at,
    }
    payload_part = _b64encode(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8"))
    signature = hmac.new(
        _get_signing_secret().encode("utf-8"),
        payload_part.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return f"{payload_part}.{signature}", expires_at


def get_request_session(request: Request) -> OwnerSession | None:
    token = _get_request_token(request)
    if not token:
        return None

    if settings.admin_api_token and secrets.compare_digest(token, settings.admin_api_token):
        return OwnerSession(
            username=settings.owner_username,
            expires_at=0,
            authenticated_via="admin_token",
        )

    if is_private_access_enabled() and secrets.compare_digest(token, settings.owner_password):
        return OwnerSession(
            username=settings.owner_username,
            expires_at=0,
            authenticated_via="owner_password",
        )

    if "." not in token:
        return None

    payload_part, signature = token.split(".", 1)
    expected_signature = hmac.new(
        _get_signing_secret().encode("utf-8"),
        payload_part.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    if not secrets.compare_digest(signature, expected_signature):
        return None

    try:
        payload = json.loads(_b64decode(payload_part).decode("utf-8"))
    except (json.JSONDecodeError, ValueError, UnicodeDecodeError):
        return None

    username = str(payload.get("sub", ""))
    expires_at = int(payload.get("exp", 0))
    if not username or not secrets.compare_digest(username, settings.owner_username):
        return None
    if expires_at <= int(time.time()):
        return None

    return OwnerSession(
        username=username,
        expires_at=expires_at,
        authenticated_via="session",
    )


def is_request_authenticated(request: Request) -> bool:
    if not is_private_access_enabled():
        return True
    return get_request_session(request) is not None


def has_admin_access(request: Request) -> bool:
    if get_request_session(request) is not None:
        return True
    if not settings.admin_api_token:
        return True

    token = _get_request_token(request)
    return bool(token) and secrets.compare_digest(token, settings.admin_api_token)