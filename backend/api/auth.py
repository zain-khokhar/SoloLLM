from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from core.auth import get_request_session, is_private_access_enabled, issue_owner_session_token, verify_owner_credentials
from core.config import settings


router = APIRouter(prefix="/api/auth")


class LoginRequest(BaseModel):
    username: str
    password: str


def _format_timestamp(timestamp: int) -> str | None:
    if timestamp <= 0:
        return None
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat().replace("+00:00", "Z")


@router.get("/config")
async def get_auth_config():
    return {
        "private_access_enabled": is_private_access_enabled(),
        "owner_username": settings.owner_username,
        "session_ttl_hours": settings.session_ttl_hours,
        "admin_token_enabled": bool(settings.admin_api_token),
        "public_base_url": settings.public_base_url,
    }


@router.get("/session")
async def get_auth_session(request: Request):
    session = get_request_session(request)
    return {
        "authenticated": session is not None or not is_private_access_enabled(),
        "private_access_enabled": is_private_access_enabled(),
        "owner_username": settings.owner_username,
        "expires_at": _format_timestamp(session.expires_at) if session else None,
        "authenticated_via": session.authenticated_via if session else None,
        "admin_token_enabled": bool(settings.admin_api_token),
    }


@router.post("/login")
async def login_owner(request: LoginRequest):
    if not is_private_access_enabled():
        return {
            "authenticated": True,
            "private_access_enabled": False,
            "owner_username": settings.owner_username,
            "token": "",
            "expires_at": None,
            "authenticated_via": "disabled",
        }

    if not verify_owner_credentials(request.username, request.password):
        raise HTTPException(
            status_code=401,
            detail="Invalid owner credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token, expires_at = issue_owner_session_token()
    return {
        "authenticated": True,
        "private_access_enabled": True,
        "owner_username": settings.owner_username,
        "token": token,
        "expires_at": _format_timestamp(expires_at),
        "authenticated_via": "session",
    }


@router.post("/logout")
async def logout_owner():
    return {"success": True}