from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import google.auth


@dataclass(frozen=True)
class Settings:
    endpoint_api_key: str | None
    planner_mode: str
    allow_beta_endpoints: bool
    gemini_api_key: str | None
    google_cloud_project: str | None
    google_cloud_location: str
    gemini_model: str
    gemini_max_attachment_bytes: int
    gemini_attachment_text_chars: int
    runs_dir: Path
    tripletex_timeout_seconds: float


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    runs_dir = Path(os.getenv("TRIPLETEX_AGENT_RUNS_DIR", ".work/runs")).resolve()
    runs_dir.mkdir(parents=True, exist_ok=True)
    return Settings(
        endpoint_api_key=os.getenv("TRIPLETEX_AGENT_API_KEY"),
        planner_mode=os.getenv("TRIPLETEX_AGENT_PLANNER_MODE", "stub").strip().lower(),
        allow_beta_endpoints=parse_bool_env("TRIPLETEX_AGENT_ALLOW_BETA_ENDPOINTS", default=False),
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        google_cloud_project=resolve_google_cloud_project(),
        google_cloud_location=os.getenv("GOOGLE_CLOUD_LOCATION", "global").strip(),
        gemini_model=os.getenv("TRIPLETEX_AGENT_GEMINI_MODEL", "gemini-2.5-flash").strip(),
        gemini_max_attachment_bytes=int(
            os.getenv("TRIPLETEX_AGENT_GEMINI_MAX_ATTACHMENT_BYTES", "8000000")
        ),
        gemini_attachment_text_chars=int(
            os.getenv("TRIPLETEX_AGENT_GEMINI_ATTACHMENT_TEXT_CHARS", "12000")
        ),
        tripletex_timeout_seconds=float(os.getenv("TRIPLETEX_AGENT_TIMEOUT_SECONDS", "30")),
        runs_dir=runs_dir,
    )


def resolve_google_cloud_project() -> str | None:
    explicit = (
        os.getenv("GOOGLE_CLOUD_PROJECT")
        or os.getenv("GCP_PROJECT")
        or os.getenv("GCLOUD_PROJECT")
    )
    if explicit:
        return explicit.strip()

    try:
        _, discovered_project = google.auth.default()
    except Exception:
        return None

    return discovered_project


def parse_bool_env(name: str, *, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default
