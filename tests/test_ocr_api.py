"""What the API shows about OCR: the document's OCR block, provider pauses, and the start-up model checks."""

from dataclasses import replace
from uuid import UUID

from fastapi.testclient import TestClient

from docqa.api.app import check_models, create_app
from docqa.api.deps import Deps
from docqa.config import Settings
from docqa.providers.fake import FakeLLM, result
from docqa.providers.llm import ProviderError
from docqa.telemetry import MemoryTelemetry, PostgresTelemetry
from docqa.worker import WorkerDeps, run_one
from tests.helpers import TenantInfo, make_runnable
from tests.ingest_helpers import drain, jobs_of
from tests.pdfs import rasterized_pdf

LITE = "gemini-3.5-flash-lite"
FLASH = "gemini-3.8-flash"
ALL_KINDS = ("extract", "ocr_page", "index")
TEXT = "# Lubrication\n\n| Speed | Hours |\n|---|---|\n| 3600 | 5500Hrs. |"


def _upload(client: TestClient, tenant: TenantInfo, data: bytes, batch_id: str) -> str:
    """Upload a file into a batch.

    Args:
        client: HTTP client.
        tenant: Uploading tenant.
        data: File bytes.
        batch_id: Batch id.

    Returns:
        The document id.
    """
    response = client.post(
        "/v1/documents",
        params={"batch_id": batch_id},
        files={"file": ("scan.pdf", data, "application/pdf")},
        headers={"Authorization": f"Bearer {tenant.key}"},
    )
    assert response.status_code == 202, response.text
    return str(response.json()["document_id"])


def test_a_pause_shows_on_the_document_and_the_batch(
    worker_deps: WorkerDeps, client: TestClient, tenant: TenantInfo
) -> None:
    """A paused model is named, with the reason and the end, on the document and on its batch item.

    Args:
        worker_deps: Worker dependencies.
        client: HTTP client.
        tenant: A tenant.
    """
    fake = FakeLLM([ProviderError("quota_exhausted", "429 per day", status=429)])
    deps = replace(worker_deps, llm=fake)
    headers = {"Authorization": f"Bearer {tenant.key}"}
    batch_id = client.post("/v1/batches", headers=headers).json()["batch_id"]
    document_id = _upload(client, tenant, rasterized_pdf(), batch_id)
    assert run_one(deps, "extract")
    assert run_one(deps, "ocr_page")
    [job] = [job for job in jobs_of(deps.pool, UUID(document_id)) if job["kind"] == "ocr_page"]
    make_runnable(deps.pool, job["id"])
    assert not run_one(deps, "ocr_page")
    detail = client.get(f"/v1/documents/{document_id}", headers=headers).json()
    paused = detail["ocr"]["paused"]
    assert (paused["model"], paused["reason"]) == (LITE, "quota_exhausted")
    view = client.get(f"/v1/batches/{batch_id}", headers=headers).json()
    assert view["items"][0]["document"]["ocr_paused"] == paused
    assert view["complete"] is False


def test_document_view_reports_ocr_calls_and_tokens(
    worker_deps: WorkerDeps, client: TestClient, tenant: TenantInfo
) -> None:
    """The OCR block sums the document's model calls from the spans and the pages' evidence.

    Args:
        worker_deps: Worker dependencies.
        client: HTTP client.
        tenant: A tenant.
    """
    telemetry = PostgresTelemetry(worker_deps.pool)
    fake = FakeLLM([result(None, "RECITATION", output_tokens=0), result(TEXT, thinking_tokens=10)], telemetry=telemetry)
    deps = replace(worker_deps, llm=fake, telemetry=telemetry)
    headers = {"Authorization": f"Bearer {tenant.key}"}
    batch_id = client.post("/v1/batches", headers=headers).json()["batch_id"]
    document_id = _upload(client, tenant, rasterized_pdf(), batch_id)
    drain(deps, ALL_KINDS)
    detail = client.get(f"/v1/documents/{document_id}", headers=headers).json()
    assert detail["status"] == "ready"
    assert detail["ocr"] == {
        "pages_flagged": 1,
        "pages_transcribed": 1,
        "pages_from_cache": 0,
        "recitations": 1,
        "fallback_used": 0,
        "calls": 2,
        "input_tokens": 2600,
        "output_tokens": 410,
        "by_model": {LITE: {"calls": 2, "input_tokens": 2600, "output_tokens": 410}},
        "paused": None,
    }
    assert detail["pages_by_source"] == {"layer": 0, "ocr": 1}


def test_model_checks_mark_the_service_degraded(settings: Settings, deps: Deps) -> None:
    """A rejected model leaves `/healthz` at 200 but `degraded`, with the model and the reason.

    Args:
        settings: Test settings.
        deps: API dependencies.
    """
    fake = FakeLLM(model_errors={FLASH: ProviderError("misconfigured", "404 NOT_FOUND", status=404)})
    health = check_models(fake, settings, MemoryTelemetry())
    assert fake.checked == [FLASH, LITE]
    assert health.models == {FLASH: "misconfigured", LITE: "ok"}
    assert health.reasons == [f"{FLASH}: provider rejected the model"]
    with TestClient(create_app(settings, replace(deps, health=health))) as test_client:
        response = test_client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert response.json()["checks"]["models"] == {FLASH: "misconfigured", LITE: "ok"}
