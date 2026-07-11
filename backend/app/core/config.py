"""Application settings, loaded from environment variables (or a .env file
in local development). Nothing secret has a real default -- SECRET_KEY in
particular must be overridden in any non-local environment, enforced in
main.py's startup check.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="BP_", extra="ignore")

    environment: str = "development"  # "development" | "staging" | "production"
    debug: bool = True

    # --- Auth ---
    secret_key: str = "dev-only-insecure-secret-change-me"
    access_token_expire_minutes: int = 60
    refresh_token_expire_days: int = 7
    jwt_algorithm: str = "HS256"

    # --- Database ---
    database_url: str = "sqlite+aiosqlite:///./bloodprint.db"

    # --- Redis cache ---
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 3600
    cache_enabled: bool = True

    # --- ML model ---
    model_bundle_path: str = "../ml_artifacts/model_bundle/bundle.json"
    inference_device: str = "cpu"
    default_use_tta: bool = True
    default_tta_augments: int = 6
    default_explain: bool = True

    # --- Uploads ---
    upload_dir: str = "uploads"
    max_upload_size_bytes: int = 16 * 1024 * 1024
    allowed_extensions: set[str] = {"png", "jpg", "jpeg", "bmp"}

    # --- CORS ---
    cors_allow_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173", "http://localhost:3000"])

    # --- Rate limiting ---
    rate_limit_per_minute: int = 30

    # --- Observability ---
    log_level: str = "INFO"
    log_json: bool = False

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


@lru_cache
def get_settings() -> Settings:
    return Settings()
