"""Password hashing and JWT issuance/verification.

Password hashing uses the `bcrypt` library directly rather than passlib's
CryptContext wrapper: passlib is no longer actively maintained and its
backend auto-detection breaks against bcrypt>=4.1's stricter input
validation (raises on its own internal self-test before hashing anything).
Calling bcrypt directly is fewer moving parts and avoids that entirely.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from enum import StrEnum

import bcrypt
from jose import JWTError, jwt

from app.core.config import get_settings

_BCRYPT_MAX_BYTES = 72  # bcrypt silently ignores bytes beyond this; truncate explicitly rather than surprise a user


class TokenType(StrEnum):
    ACCESS = "access"
    REFRESH = "refresh"


def hash_password(plain_password: str) -> str:
    pw_bytes = plain_password.encode("utf-8")[:_BCRYPT_MAX_BYTES]
    return bcrypt.hashpw(pw_bytes, bcrypt.gensalt()).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    pw_bytes = plain_password.encode("utf-8")[:_BCRYPT_MAX_BYTES]
    try:
        return bcrypt.checkpw(pw_bytes, hashed_password.encode("utf-8"))
    except ValueError:
        return False  # malformed stored hash -- fail closed, not with a 500


def create_token(subject: str, token_type: TokenType, expires_delta: timedelta | None = None) -> str:
    settings = get_settings()
    if expires_delta is None:
        expires_delta = (
            timedelta(minutes=settings.access_token_expire_minutes)
            if token_type == TokenType.ACCESS
            else timedelta(days=settings.refresh_token_expire_days)
        )
    now = datetime.now(UTC)
    payload = {"sub": subject, "type": token_type.value, "iat": now, "exp": now + expires_delta}
    return jwt.encode(payload, settings.secret_key, algorithm=settings.jwt_algorithm)


def create_access_token(subject: str) -> str:
    return create_token(subject, TokenType.ACCESS)


def create_refresh_token(subject: str) -> str:
    return create_token(subject, TokenType.REFRESH)


class InvalidTokenError(Exception):
    pass


def decode_token(token: str, expected_type: TokenType) -> str:
    """Returns the token's subject (user id) or raises InvalidTokenError."""
    settings = get_settings()
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.jwt_algorithm])
    except JWTError as exc:
        raise InvalidTokenError(str(exc)) from exc

    if payload.get("type") != expected_type.value:
        raise InvalidTokenError(f"Expected a {expected_type.value} token, got {payload.get('type')}")

    subject = payload.get("sub")
    if subject is None:
        raise InvalidTokenError("Token has no subject")
    return subject
