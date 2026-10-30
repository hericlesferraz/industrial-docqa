"""Process settings read once from the environment, with fail-fast validation."""

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
EMBEDDING_PROVIDERS = ("local", "gemini")
BLOB_STORES = ("local",)


class ConfigError(Exception):
    """A required variable is missing or a variable has an invalid value; the message names the variable."""


@dataclass(frozen=True)
class Settings:
    """Typed view of every environment variable the services read.

    Attributes:
        gemini_api_key: Provider key; never logged.
        admin_api_key: Key of the admin routes; never logged.
        database_url: Postgres connection URL; empty in processes that only talk HTTP.
        gemini_answer_model: Model of the answer path.
        gemini_ocr_model: Default OCR model.
        gemini_ocr_fallback_model: OCR model used after a refusal or on the last attempt.
        embedding_provider: `local` or `gemini`.
        blob_store: Blob store implementation; only `local`.
        blob_store_path: Root folder of the local blob store.
        max_upload_mb: Upload size limit in MiB.
        max_pdf_pages: Page limit of one PDF.
        extract_timeout_s: Wall-clock limit of the extraction child: a slow 500-page catalog (1.5 s per page
            measured on the slowest manual) passes with a 3x margin; kept below `eval_ingest_timeout_s`.
        extract_memory_mb: Resident memory limit of the extraction child.
        ocr_max_concurrency: Global cap of running `ocr_page` jobs.
        ocr_min_interval_s: Minimum gap between two OCR calls per model.
        embed_batch_size: Chunks per embedding call.
        log_level: Logging level name.
        eval_api_url: Base URL of the API for the eval and the CLI.
        eval_ingest_timeout_s: How long the eval waits for ingestion.
        eval_judge_model: Model of the eval judge.
    """

    gemini_api_key: str = field(repr=False)
    admin_api_key: str = field(repr=False)
    database_url: str = field(repr=False)
    gemini_answer_model: str = "gemini-3.8-flash"
    gemini_ocr_model: str = "gemini-3.5-flash-lite"
    gemini_ocr_fallback_model: str = "gemini-3.8-flash"
    embedding_provider: str = "local"
    blob_store: str = "local"
    blob_store_path: Path = Path("/data/blobs")
    max_upload_mb: int = 200
    max_pdf_pages: int = 2000
    extract_timeout_s: int = 2400
    extract_memory_mb: int = 2048
    ocr_max_concurrency: int = 3
    ocr_min_interval_s: float = 1.0
    embed_batch_size: int = 1
    log_level: str = "INFO"
    eval_api_url: str = "http://api:8000"
    eval_ingest_timeout_s: int = 3600
    eval_judge_model: str = "gemini-3.5-flash-lite"


def _required(env: Mapping[str, str], name: str) -> str:
    """Read a variable that must be set and non-empty.

    Args:
        env: Environment mapping.
        name: Variable name.

    Returns:
        The value, stripped.

    Raises:
        ConfigError: The variable is missing or empty.
    """
    value = env.get(name, "").strip()
    if not value:
        raise ConfigError(f"{name} is not set; copy .env.example to .env and set it")
    return value


def _text(env: Mapping[str, str], name: str, default: str) -> str:
    """Read an optional text variable.

    Args:
        env: Environment mapping.
        name: Variable name.
        default: Value when the variable is missing or empty.

    Returns:
        The value, stripped, or the default.
    """
    value = env.get(name, "").strip()
    return value or default


def _choice(env: Mapping[str, str], name: str, default: str, allowed: tuple[str, ...]) -> str:
    """Read an optional variable restricted to a set of values.

    Args:
        env: Environment mapping.
        name: Variable name.
        default: Value when the variable is missing or empty.
        allowed: Accepted values.

    Returns:
        The value.

    Raises:
        ConfigError: The value is not one of `allowed`.
    """
    value = _text(env, name, default)
    if value not in allowed:
        raise ConfigError(f"{name} must be one of {', '.join(allowed)}; got {value!r}")
    return value


def _positive_int(env: Mapping[str, str], name: str, default: int) -> int:
    """Read an optional positive integer variable.

    Args:
        env: Environment mapping.
        name: Variable name.
        default: Value when the variable is missing or empty.

    Returns:
        The value.

    Raises:
        ConfigError: The value is not a positive integer.
    """
    raw = env.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ConfigError(f"{name} must be a positive integer; got {raw!r}") from None
    if value <= 0:
        raise ConfigError(f"{name} must be a positive integer; got {raw!r}")
    return value


def _non_negative_float(env: Mapping[str, str], name: str, default: float) -> float:
    """Read an optional non-negative number variable.

    Args:
        env: Environment mapping.
        name: Variable name.
        default: Value when the variable is missing or empty.

    Returns:
        The value.

    Raises:
        ConfigError: The value is not a non-negative number.
    """
    raw = env.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        raise ConfigError(f"{name} must be a non-negative number; got {raw!r}") from None
    if value < 0:
        raise ConfigError(f"{name} must be a non-negative number; got {raw!r}")
    return value


def load_settings(env: Mapping[str, str] = os.environ, *, require_database: bool = True) -> Settings:
    """Build the settings from an environment mapping.

    Args:
        env: Environment mapping; the process environment by default.
        require_database: False for processes that only talk HTTP (the eval, the CLI), which get no
            `DATABASE_URL`; their `database_url` is then empty.

    Returns:
        The validated settings.

    Raises:
        ConfigError: A required variable is missing or a value is invalid; the message names the variable.
    """
    defaults = Settings(gemini_api_key="", admin_api_key="", database_url="")
    return Settings(
        gemini_api_key=_required(env, "GEMINI_API_KEY"),
        admin_api_key=_required(env, "ADMIN_API_KEY"),
        database_url=_required(env, "DATABASE_URL") if require_database else _text(env, "DATABASE_URL", ""),
        gemini_answer_model=_text(env, "GEMINI_ANSWER_MODEL", defaults.gemini_answer_model),
        gemini_ocr_model=_text(env, "GEMINI_OCR_MODEL", defaults.gemini_ocr_model),
        gemini_ocr_fallback_model=_text(env, "GEMINI_OCR_FALLBACK_MODEL", defaults.gemini_ocr_fallback_model),
        embedding_provider=_choice(env, "EMBEDDING_PROVIDER", defaults.embedding_provider, EMBEDDING_PROVIDERS),
        blob_store=_choice(env, "BLOB_STORE", defaults.blob_store, BLOB_STORES),
        blob_store_path=Path(_text(env, "BLOB_STORE_PATH", str(defaults.blob_store_path))),
        max_upload_mb=_positive_int(env, "MAX_UPLOAD_MB", defaults.max_upload_mb),
        max_pdf_pages=_positive_int(env, "MAX_PDF_PAGES", defaults.max_pdf_pages),
        extract_timeout_s=_positive_int(env, "EXTRACT_TIMEOUT_S", defaults.extract_timeout_s),
        extract_memory_mb=_positive_int(env, "EXTRACT_MEMORY_MB", defaults.extract_memory_mb),
        ocr_max_concurrency=_positive_int(env, "OCR_MAX_CONCURRENCY", defaults.ocr_max_concurrency),
        ocr_min_interval_s=_non_negative_float(env, "OCR_MIN_INTERVAL_S", defaults.ocr_min_interval_s),
        embed_batch_size=_positive_int(env, "EMBED_BATCH_SIZE", defaults.embed_batch_size),
        log_level=_choice(env, "LOG_LEVEL", defaults.log_level, LOG_LEVELS),
        eval_api_url=_text(env, "EVAL_API_URL", defaults.eval_api_url).rstrip("/"),
        eval_ingest_timeout_s=_positive_int(env, "EVAL_INGEST_TIMEOUT_S", defaults.eval_ingest_timeout_s),
        eval_judge_model=_text(env, "EVAL_JUDGE_MODEL", defaults.eval_judge_model),
    )
