"""`POST /v1/questions` and `GET /v1/answers/{id}` end to end with a scripted model."""

import json
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from psycopg_pool import ConnectionPool

from docqa.answer import service
from docqa.answer.prompt import PENDING_NOTE
from docqa.api.app import create_app
from docqa.api.deps import Deps
from docqa.config import Settings
from docqa.db import tx
from docqa.providers.fake import FakeCall, FakeLLM, Outcome, result
from docqa.providers.llm import ProviderError
from docqa.repo import documents
from docqa.retrieval.search import SEARCH_VERSION
from docqa.storage import LocalBlobStore
from docqa.telemetry import PostgresTelemetry
from docqa.worker import WorkerDeps
from tests.helpers import TenantInfo, make_tenant, new_document
from tests.index_helpers import FakeEmbedder, index_all, make_document, with_embedder

PAGES = json.loads((Path(__file__).parent / "fixtures" / "verifier_pages.json").read_text(encoding="utf-8"))
LB5001 = ["# **Baldor motors**\n\nGeneral information about the motor.", PAGES["lb5001_p2"]["text"]]
GEAR = ["# **Gear unit**\n\nChange the gear oil every 10000 hours.\n\nThe warranty covers 12 months."]
QUESTION = "What is the relubrication interval for frame 256 at 1800 RPM?"
Step = Callable[[FakeCall], Outcome]


@dataclass
class Qa:
    """An app over an indexed tenant, with a scripted model.

    Attributes:
        client: HTTP client.
        tenant: The tenant that owns the documents.
        llm: The scripted model; tests set its script.
        pool: Database pool.
        lb5001: The LB5001 document id.
        gear: The gear-unit document id.
    """

    client: TestClient
    tenant: TenantInfo
    llm: FakeLLM
    pool: ConnectionPool
    lb5001: UUID
    gear: UUID

    def ask(self, question: str = QUESTION, key: str | None = None, **extra: object) -> dict[str, object]:
        """Ask a question.

        Args:
            question: The question.
            key: API key; the tenant's by default.
            **extra: More body fields.

        Returns:
            Status code, headers and JSON body under `status`, `headers` and `body`.
        """
        response = self.client.post(
            "/v1/questions",
            json={"question": question, **extra},
            headers={"Authorization": f"Bearer {key or self.tenant.key}"},
        )
        return {"status": response.status_code, "headers": response.headers, "body": response.json()}

    def record(self, answer_id: str, key: str | None = None) -> dict[str, object]:
        """Read an answer record.

        Args:
            answer_id: Answer id.
            key: API key; the tenant's by default.

        Returns:
            Status code and JSON body under `status` and `body`.
        """
        response = self.client.get(
            f"/v1/answers/{answer_id}", headers={"Authorization": f"Bearer {key or self.tenant.key}"}
        )
        return {"status": response.status_code, "body": response.json()}


@pytest.fixture
def qa(
    db: ConnectionPool,
    settings: Settings,
    blobs: LocalBlobStore,
    worker_deps: WorkerDeps,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[Qa]:
    """Index LB5001 and a gear-unit manual in one tenant and serve the API with Postgres telemetry and a fake model.

    Args:
        db: Pool on an empty database.
        settings: Test settings.
        blobs: Blob store.
        worker_deps: Worker dependencies, for the index jobs.
        monkeypatch: Removes the backoff between provider retries.

    Yields:
        The test context.
    """
    monkeypatch.setattr(service, "PROVIDER_RETRY_BACKOFF_S", 0.0)
    telemetry = PostgresTelemetry(db)
    llm = FakeLLM(telemetry=telemetry)
    deps = Deps(pool=db, settings=settings, telemetry=telemetry, blobs=blobs, llm=llm, embedder=FakeEmbedder())
    with TestClient(create_app(settings, deps)) as client:
        tenant = make_tenant(client)
        lb5001 = make_document(db, tenant.id, LB5001, filename="LB5001.pdf")
        gear = make_document(db, tenant.id, GEAR, filename="gear.pdf")
        index_all(with_embedder(worker_deps))
        yield Qa(client=client, tenant=tenant, llm=llm, pool=db, lb5001=lb5001, gear=gear)


def passages_of(call: FakeCall) -> dict[str, str]:
    """Read the labelled lines of the passages the model received.

    Args:
        call: A model call.

    Returns:
        Label to rendered unit.
    """
    text = str(call.contents[0].parts[0])
    return {match.group(1): match.group(2) for match in re.finditer(r"^(S\d+): (.*)$", text, re.MULTILINE)}


def label(call: FakeCall, needle: str) -> str:
    """Find the label of the passage line that contains a text.

    Args:
        call: A model call.
        needle: Text of the unit.

    Returns:
        The label.
    """
    return next(name for name, line in passages_of(call).items() if needle in line)


def script(*steps: Step) -> Callable[[FakeCall], Outcome]:
    """Chain steps, one per model call.

    Args:
        *steps: Functions of the call that return its outcome.

    Returns:
        A script for `FakeLLM`.
    """
    queue = list(steps)

    def next_step(call: FakeCall) -> Outcome:
        """Run the next step.

        Args:
            call: The model call.

        Returns:
            The step's outcome.
        """
        assert queue, "the model was called more times than scripted"
        return queue.pop(0)(call)

    return next_step


def answered(needles: list[str], text: str) -> Step:
    """Answer citing the passages that contain some texts.

    Args:
        needles: One text per cited unit.
        text: Answer text.

    Returns:
        The step.
    """
    return lambda call: result(
        json.dumps({"status": "answered", "citations": [label(call, n) for n in needles], "answer": text})
    )


def fixed(outcome: Outcome) -> Step:
    """Return the same outcome whatever the call.

    Args:
        outcome: A result or a provider error.

    Returns:
        The step.
    """
    return lambda call: outcome


def test_answered_with_expanded_citation_and_record(qa: Qa) -> None:
    """A verified answer cites one page with the header rows attached; its record holds the whole run.

    Args:
        qa: Test context.
    """
    qa.llm.script = script(answered(["9500Hrs."], "9,500 hours at 1800 RPM."))
    reply = qa.ask()
    body = reply["body"]
    assert reply["status"] == 200, body
    assert body["answer_id"] == reply["headers"]["X-Request-ID"]
    assert body["status"] == "answered" and body["answer"] == "9,500 hours at 1800 RPM."
    assert set(body) == {"answer_id", "status", "answer", "citations", "coverage"}
    (citation,) = body["citations"]
    assert (citation["filename"], citation["page"], citation["page_label"], citation["transcribed"]) == (
        "LB5001.pdf",
        2,
        None,
        False,
    )
    assert citation["quote"].splitlines() == [
        "Relubricatio | n Intervals",
        "NEMA (IEC) Frame Size | Rated Speed | (RPM)",
        "3600 | 1800 | 1200 | 900",
        "Over 210 to 280 incl.(180) | 3600Hrs. | 9500Hrs. | 15000Hrs. | 18000Hrs.",
    ]
    assert body["coverage"] == {
        "complete": True,
        "documents": 2,
        "documents_incomplete": 0,
        "pages_indexed": 3,
        "pages_pending": 0,
        "pages_failed": 0,
        "incomplete": [],
    }
    (call,) = qa.llm.calls
    assert call.stage == "answer" and call.config.thinking_level == "low"
    assert call.config.max_output_tokens == 4096 and call.timeout_s <= service.ANSWER_CALL_TIMEOUT_S
    assert "anyOf" in (call.config.response_json_schema or {})
    record = qa.record(body["answer_id"])
    assert record["status"] == 200
    rec = record["body"]
    assert rec["outcome"] == {"http_status": 200, "status": "answered", "reason": None, "error_code": None}
    assert rec["versions"]["prompt"] == "g5-v3" and rec["versions"]["assembly"] == "b2"
    assert rec["versions"]["index"] == "win250-head4|fake-hash@1"
    assert rec["versions"]["search"] == SEARCH_VERSION == "rrf60-df0.3pct-stop"
    assert rec["question"] == rec["search_query"] == QUESTION
    assert rec["response"] == body
    assert rec["candidates"] and all(c["sent_to_llm"] == (c["rank"] <= 8) for c in rec["candidates"])
    assert set(rec["labels"]) == set(passages_of(call))
    (entry,) = rec["calls"]
    assert entry["purpose"] == "answer" and entry["verifier"]["passed"] and entry["finish_reason"] == "STOP"
    assert (entry["input_tokens"], entry["output_tokens"]) == (1300, 400)
    names = [stage["name"] for stage in rec["stages"]]
    assert {"answer.coverage", "answer.embed_query", "answer.search", "answer.prompt", "gen_ai.call"} <= set(names)
    assert {"answer.verify", "answer.question", "search.vector"} <= set(names)
    assert rec["totals"]["latency_ms"] is not None and rec["totals"]["model_calls"] == 1
    assert QUESTION not in json.dumps([stage["attributes"] for stage in rec["stages"]])


def test_verifier_retry_then_pass(qa: Qa) -> None:
    """A rejected answer gets one retry with the verifier's message; the second answer is returned.

    Args:
        qa: Test context.
    """
    qa.llm.script = script(
        answered(["Upto 210 incl.(132)"], "9500 hours."),
        answered(["9500Hrs."], "9500 hours."),
    )
    reply = qa.ask()
    assert reply["status"] == 200 and reply["body"]["status"] == "answered"
    first, second = qa.llm.calls
    assert (first.stage, second.stage) == ("answer", "answer_retry")
    assert [turn.role for turn in second.contents] == ["user", "model", "user"]
    assert str(second.contents[2].parts[0]).startswith("Verification failed: answer: the numbers 9500 have no source")
    calls = qa.record(reply["body"]["answer_id"])["body"]["calls"]
    assert second.contents[1].parts == [calls[0]["raw_output"]]
    assert [(c["purpose"], c["verifier"]["passed"]) for c in calls] == [("answer", False), ("verifier_retry", True)]


def test_two_failed_verifications_decline_with_passages(qa: Qa) -> None:
    """After a second rejection the answer is `not_found` + `unverified`, with passages; the text never leaves.

    Args:
        qa: Test context.
    """
    wrong = answered(["Upto 210 incl.(132)"], "48.8 hours.")
    qa.llm.script = script(wrong, wrong)
    reply = qa.ask()
    body = reply["body"]
    assert reply["status"] == 200
    assert (body["status"], body["reason"], body["answer"]) == ("not_found", "unverified", service.UNVERIFIED_ANSWER)
    assert body["citations"] == [] and "options" not in body
    assert 1 <= len(body["passages"]) <= 3 and all(p["text"] for p in body["passages"])
    assert "48.8" not in json.dumps(body)
    outcome = qa.record(body["answer_id"])["body"]["outcome"]
    assert outcome == {"http_status": 200, "status": "not_found", "reason": "unverified", "error_code": None}


def test_ambiguous_options_and_merged_citations(qa: Qa) -> None:
    """Each option carries its own citations; the top level merges them per page and `answer` is `depends_on`.

    Args:
        qa: Test context.
    """

    def ambiguous(call: FakeCall) -> Outcome:
        """Answer with two frame-size options.

        Args:
            call: The model call.

        Returns:
            The output.
        """
        options = [
            {
                "condition": "Frame up to 210",
                "citations": [label(call, "Upto 210 incl.(132) | 5500")],
                "answer": "5500 h",
            },
            {"condition": "Frame 210 to 280", "citations": [label(call, "9500Hrs.")], "answer": "9500 h at 1800 RPM"},
        ]
        return result(json.dumps({"status": "ambiguous", "depends_on": "The frame size.", "options": options}))

    qa.llm.script = script(ambiguous)
    body = qa.ask("How often should I regrease a Baldor motor?")["body"]
    assert body["status"] == "ambiguous" and body["answer"] == "The frame size."
    assert [option["condition"] for option in body["options"]] == ["Frame up to 210", "Frame 210 to 280"]
    (merged,) = body["citations"]
    assert "Upto 210 incl.(132)" in merged["quote"] and "9500Hrs." in merged["quote"]
    assert "9500Hrs." not in body["options"][0]["citations"][0]["quote"]


def test_not_found_from_the_model(qa: Qa) -> None:
    """A `not_found` output has no citations, options, reason or passages.

    Args:
        qa: Test context.
    """
    qa.llm.script = script(fixed(result('{"status": "not_found", "answer": "The manual gives no IP rating."}')))
    body = qa.ask("What is the IP rating?")["body"]
    assert set(body) == {"answer_id", "status", "answer", "citations", "coverage"}
    assert (body["status"], body["citations"]) == ("not_found", [])


@pytest.mark.parametrize(
    ("errors", "status", "code", "retryable", "retry_after", "calls"),
    [
        ([ProviderError("transient", "503")] * 2, 503, "llm_unavailable", True, None, 2),
        ([ProviderError("rate_limited", "429", retry_after_s=120)], 503, "llm_rate_limited", True, "120", 1),
        ([ProviderError("quota_exhausted", "429 per day")], 503, "llm_rate_limited", True, "3600", 1),
        ([ProviderError("billing", "402")], 503, "llm_billing", False, None, 1),
        ([ProviderError("misconfigured", "404 model")], 503, "llm_misconfigured", False, None, 1),
        ([ProviderError("timeout", "read timeout")] * 2, 504, "llm_timeout", True, None, 2),
        ([ProviderError("invalid_request", "400")], 502, "llm_invalid_request", False, None, 1),
    ],
)
def test_provider_failures(
    qa: Qa,
    errors: list[ProviderError],
    status: int,
    code: str,
    retryable: bool,
    retry_after: str | None,
    calls: int,
) -> None:
    """Each provider failure maps to its HTTP error; 503s carry passages; only transient kinds are retried; recorded.

    Args:
        qa: Test context.
        errors: Errors raised by the successive calls.
        status: Expected HTTP status.
        code: Expected error code.
        retryable: Expected `retryable`.
        retry_after: Expected `Retry-After` header.
        calls: Expected number of model calls.
    """
    qa.llm.script = script(*[fixed(error) for error in errors])
    reply = qa.ask()
    error = reply["body"]["error"]
    assert (reply["status"], error["code"], error["retryable"]) == (status, code, retryable)
    assert reply["headers"].get("Retry-After") == retry_after
    assert len(qa.llm.calls) == calls
    if status == 503:
        passages = error["details"]["passages"]
        assert passages and passages[0]["filename"] in {"LB5001.pdf", "gear.pdf"}
        assert "not an answer" in error["message"]
    else:
        assert "details" not in error
    rec = qa.record(error["request_id"])["body"]
    assert rec["outcome"] == {"http_status": status, "status": None, "reason": None, "error_code": code}
    assert rec["response"] == reply["body"] and len(rec["calls"]) == calls


def test_transient_failure_then_success(qa: Qa) -> None:
    """One retry after a transient failure is enough.

    Args:
        qa: Test context.
    """
    qa.llm.script = script(fixed(ProviderError("transient", "503")), answered(["9500Hrs."], "9500 hours."))
    reply = qa.ask()
    assert reply["status"] == 200 and [call.attempt for call in qa.llm.calls] == [1, 2]


@pytest.mark.parametrize(
    ("outcome", "code"),
    [
        (result(None, "SAFETY"), "llm_refused"),
        (result('{"status": "answered"', "MAX_TOKENS"), "llm_invalid_output"),
        (result("not json"), "llm_invalid_output"),
    ],
)
def test_invalid_or_refused_output(qa: Qa, outcome: Outcome, code: str) -> None:
    """A refusal is `llm_refused`; a cut or unparsable output is `llm_invalid_output`, with the reason recorded.

    Args:
        qa: Test context.
        outcome: The model's result.
        code: Expected error code.
    """
    qa.llm.script = script(fixed(outcome))
    reply = qa.ask()
    assert (reply["status"], reply["body"]["error"]["code"]) == (502, code)
    (entry,) = qa.record(reply["body"]["error"]["request_id"])["body"]["calls"]
    assert entry["error_code"] is not None


def test_deadline_passed_is_a_504(qa: Qa, monkeypatch: pytest.MonkeyPatch) -> None:
    """With no time left for a call, the answer is a 504 and no call is made.

    Args:
        qa: Test context.
        monkeypatch: Shrinks the deadline.
    """
    monkeypatch.setattr(service, "ANSWER_DEADLINE_S", 1.0)
    reply = qa.ask()
    assert (reply["status"], reply["body"]["error"]["code"]) == (504, "llm_timeout")
    assert qa.llm.calls == []


def test_coverage_pending_and_failed_documents(qa: Qa) -> None:
    """A queued document makes coverage incomplete and adds the prompt note; a failed one never adds the note.

    Args:
        qa: Test context.
    """
    not_found = fixed(result('{"status": "not_found", "answer": "No."}'))
    failed = new_document(qa.pool, qa.tenant.id, sha256="f" * 64)
    with tx(qa.pool) as conn:
        documents.set_document_failed(
            conn, tenant_id=qa.tenant.id, document_id=failed, code="pdf_encrypted", message="x"
        )
    qa.llm.script = script(not_found, not_found)
    body = qa.ask("Torque?")["body"]
    assert body["coverage"]["complete"] is False
    assert body["coverage"]["incomplete"] == [
        {
            "document_id": str(failed),
            "filename": "manual.pdf",
            "status": "failed",
            "pages_pending": None,
            "pages_failed": None,
            "error_code": "pdf_encrypted",
        }
    ]
    assert PENDING_NOTE not in str(qa.llm.calls[0].contents[0].parts[0])
    queued = new_document(qa.pool, qa.tenant.id, sha256="e" * 64)
    body = qa.ask("Torque?")["body"]
    assert [item["status"] for item in body["coverage"]["incomplete"]] == ["queued", "failed"]
    assert body["coverage"]["incomplete"][0]["document_id"] == str(queued)
    assert str(qa.llm.calls[1].contents[0].parts[0]).endswith(PENDING_NOTE)


def test_scope_errors_and_filter(qa: Qa) -> None:
    """Unknown and foreign ids are the same 404; an empty scope is a 422; a scope limits the passages.

    Args:
        qa: Test context.
    """
    other = make_tenant(qa.client, "other")
    foreign = new_document(qa.pool, other.id, sha256="d" * 64)
    unknown = uuid4()
    reply = qa.ask(document_ids=[str(foreign), str(qa.gear), str(unknown)])
    assert reply["status"] == 404
    assert reply["body"]["error"]["code"] == "document_not_found"
    assert reply["body"]["error"]["details"]["document_ids"] == [str(foreign), str(unknown)]
    assert qa.record(reply["body"]["error"]["request_id"])["status"] == 404
    assert qa.ask(document_ids=[])["status"] == 422
    assert qa.ask("   ")["status"] == 422
    qa.llm.script = script(fixed(result('{"status": "not_found", "answer": "No."}')))
    body = qa.ask(document_ids=[str(qa.gear), str(qa.gear)])["body"]
    assert body["coverage"]["documents"] == 1
    prompt = str(qa.llm.calls[0].contents[0].parts[0])
    assert "[gear.pdf, page 1]" in prompt and "LB5001.pdf" not in prompt


def test_tenant_isolation(qa: Qa) -> None:
    """Another tenant's question sees none of these documents, and can't read this tenant's records.

    Args:
        qa: Test context.
    """
    qa.llm.script = script(
        answered(["9500Hrs."], "9500 hours."), fixed(result('{"status": "not_found", "answer": "No."}'))
    )
    mine = qa.ask()["body"]["answer_id"]
    other = make_tenant(qa.client, "other")
    reply = qa.ask(key=other.key)
    assert reply["body"]["status"] == "not_found" and reply["body"]["coverage"]["documents"] == 0
    schema = qa.llm.calls[1].config.response_json_schema or {}
    assert "anyOf" not in schema and schema["properties"]["status"]["enum"] == ["not_found"]
    assert passages_of(qa.llm.calls[1]) == {}
    missing = qa.record(mine, key=other.key)
    assert (missing["status"], missing["body"]["error"]["code"]) == (404, "answer_not_found")
    assert qa.record(str(uuid4()))["status"] == 404
