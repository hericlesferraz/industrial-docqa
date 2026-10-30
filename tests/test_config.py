"""Settings: required variables, validation, defaults and the fail-fast start."""

import subprocess
import sys
from pathlib import Path

import pytest

from docqa.config import ConfigError, load_settings
from docqa.evaluation.harness import RunConfig
from docqa.ingest.jobs import EXTRACT_STALL_S

BASE = {"GEMINI_API_KEY": "secret-gemini", "ADMIN_API_KEY": "secret-admin", "DATABASE_URL": "postgresql://x/y"}
ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("name", ["GEMINI_API_KEY", "ADMIN_API_KEY", "DATABASE_URL"])
def test_missing_required_variable_names_it(name: str) -> None:
    """A missing or blank required variable raises `ConfigError` naming it.

    Args:
        name: The variable left out.
    """
    env = {key: value for key, value in BASE.items() if key != name}
    with pytest.raises(ConfigError, match=name):
        load_settings(env)
    with pytest.raises(ConfigError, match=name):
        load_settings({**env, name: "   "})


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("EMBEDDING_PROVIDER", "openai"),
        ("BLOB_STORE", "s3"),
        ("MAX_UPLOAD_MB", "lots"),
        ("MAX_PDF_PAGES", "0"),
        ("OCR_MIN_INTERVAL_S", "-1"),
        ("LOG_LEVEL", "LOUD"),
    ],
)
def test_invalid_values_fail_fast(name: str, value: str) -> None:
    """An invalid value raises `ConfigError` naming the variable.

    Args:
        name: Variable.
        value: Invalid value.
    """
    with pytest.raises(ConfigError, match=name):
        load_settings({**BASE, name: value})


def test_defaults_and_overrides() -> None:
    """Optional variables take their documented defaults and accept overrides."""
    settings = load_settings(BASE)
    assert settings.gemini_answer_model == "gemini-3.8-flash"
    assert settings.gemini_ocr_model == "gemini-3.5-flash-lite"
    assert settings.embedding_provider == "local"
    assert settings.max_upload_mb == 200
    assert settings.max_pdf_pages == 2000
    assert str(settings.blob_store_path) == "/data/blobs"
    custom = load_settings(
        {**BASE, "EMBEDDING_PROVIDER": "gemini", "OCR_MIN_INTERVAL_S": "4.5", "EMBED_BATCH_SIZE": "2"}
    )
    assert custom.embedding_provider == "gemini"
    assert custom.ocr_min_interval_s == 4.5
    assert custom.embed_batch_size == 2


def test_extraction_limits_fit_a_slow_catalog_inside_the_eval_wait() -> None:
    """A slow 500-page catalog extracts within the limits, which stay below the eval's wait for the batch.

    The slowest manual measured extracts at 1.5 s per page, and its slowest page takes 21.5 s.
    """
    settings = load_settings(BASE)
    assert settings.extract_timeout_s >= 3 * 500 * 1.5
    assert EXTRACT_STALL_S >= 10 * 21.5
    assert EXTRACT_STALL_S < settings.extract_timeout_s < settings.eval_ingest_timeout_s
    assert RunConfig(Path("."), [], Path("."), "http://api").ingest_timeout_s == settings.eval_ingest_timeout_s


def test_repr_hides_keys() -> None:
    """Neither key nor the database URL appears in `repr(settings)`."""
    text = repr(load_settings(BASE))
    assert "secret-gemini" not in text
    assert "secret-admin" not in text
    assert "postgresql://" not in text


@pytest.mark.parametrize("module", ["docqa.worker", "docqa.cli"])
def test_process_exits_2_without_gemini_key(module: str) -> None:
    """The worker and the CLI exit with code 2 and one clear line when `GEMINI_API_KEY` is missing.

    Args:
        module: Module run with `python -m`.
    """
    args = [sys.executable, "-m", module] + (["create-tenant", "--name", "x"] if module == "docqa.cli" else [])
    env = {"PATH": "/usr/bin:/bin", "PYTHONPATH": str(ROOT), "ADMIN_API_KEY": "a", "DATABASE_URL": "postgresql://x/y"}
    result = subprocess.run(args, env=env, capture_output=True, text=True, timeout=60, check=False)
    assert result.returncode == 2
    assert result.stderr.strip() == "config error: GEMINI_API_KEY is not set; copy .env.example to .env and set it"


def test_api_factory_exits_2_without_gemini_key(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The API factory exits with code 2 and the same message.

    Args:
        monkeypatch: Pytest monkeypatch.
        capsys: Pytest output capture.
    """
    from docqa.api.app import create_app_from_env

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("ADMIN_API_KEY", "a")
    monkeypatch.setenv("DATABASE_URL", "postgresql://x/y")
    with pytest.raises(SystemExit) as exited:
        create_app_from_env()
    assert exited.value.code == 2
    assert "config error: GEMINI_API_KEY is not set" in capsys.readouterr().err
