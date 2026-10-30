"""The reference-guided judge: verdict by code, cache by hash, numeric cross-check, failures."""

import json
from pathlib import Path

import pytest

from docqa.evaluation.judge import (
    JUDGE_SCHEMA,
    JUDGE_STAGE,
    Fact,
    Judge,
    JudgeCache,
    JudgeInput,
    JudgeOutcome,
    cache_key,
    disagreement,
    numeric_check,
    response_text,
    verdict_from_facts,
)
from docqa.providers.fake import FakeLLM, result
from docqa.providers.llm import ProviderError

ITEM = JudgeInput(
    question="What is the oil change interval at 90 °C with CLP PG oil?",
    expected_status="answered",
    expected_answer="13,000 hours.",
    response_status="answered",
    response_text="13.000 horas.",
)


def _output(*facts: tuple[str, bool, str]) -> str:
    """Build the judge's JSON output.

    Args:
        *facts: `(fact, essential, verdict)` triples.

    Returns:
        The JSON text.
    """
    return json.dumps({"facts": [{"fact": f, "essential": e, "verdict": v, "note": "n"} for f, e, v in facts]})


@pytest.mark.parametrize(
    ("facts", "verdict", "contradicted"),
    [
        ([("a", True, "stated"), ("b", False, "missing")], "correct", False),
        ([("a", True, "stated"), ("b", True, "missing")], "partial", False),
        ([("a", True, "stated"), ("b", False, "contradicted")], "wrong", True),
        ([("a", True, "missing"), ("b", False, "stated")], "wrong", False),
        ([("a", False, "stated"), ("b", False, "missing")], "partial", False),
    ],
)
def test_the_verdict_is_computed_from_the_facts(
    facts: list[tuple[str, bool, str]], verdict: str, contradicted: bool
) -> None:
    """Correct, partial and wrong follow the rule; with no essential fact, every fact counts as essential.

    Args:
        facts: The judge's facts.
        verdict: Expected verdict.
        contradicted: Expected contradiction flag.
    """
    assert verdict_from_facts([Fact(f, e, v, "") for f, e, v in facts]) == (verdict, contradicted)


def test_one_call_then_the_cache_answers(tmp_path: Path) -> None:
    """The first judgement calls the model; the same input later comes from the cache file, even in a new process.

    The call uses temperature 0 and the schema; a different answer or judge model is a new cache key.

    Args:
        tmp_path: Pytest temp folder.
    """
    llm = FakeLLM([result(_output(("13,000 hours", True, "stated")), input_tokens=500, output_tokens=60)])
    judge = Judge(llm, "judge-model", JudgeCache(tmp_path))
    first = judge.judge(ITEM)
    assert (first.verdict, first.cached, first.input_tokens, first.output_tokens) == ("correct", False, 500, 60)
    [call] = llm.calls
    assert (call.model, call.stage, call.config.temperature) == ("judge-model", JUDGE_STAGE, 0.0)
    assert call.config.response_json_schema == JUDGE_SCHEMA
    assert "Reference answer: 13,000 hours." in str(call.contents[0].parts[0])
    again = Judge(FakeLLM([]), "judge-model", JudgeCache(tmp_path)).judge(ITEM)
    assert (again.verdict, again.cached, again.input_tokens) == ("correct", True, 0)
    other = JudgeInput(**{**ITEM.__dict__, "response_text": "7,500 hours."})
    assert cache_key("judge-model", other) != cache_key("judge-model", ITEM)
    assert cache_key("other-model", ITEM) != cache_key("judge-model", ITEM)


def test_failures_leave_the_verdict_unknown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A provider error after one retry, a refusal or an invalid output: no verdict, the reason kept, nothing cached.

    Args:
        tmp_path: Pytest temp folder.
        monkeypatch: Removes the wait before the retry.
    """
    monkeypatch.setattr("docqa.evaluation.judge.JUDGE_RETRY_WAIT_S", 0.0)
    transient = ProviderError("transient", "503")
    llm = FakeLLM([transient, ProviderError("billing", "402")])
    outcome = Judge(llm, "m", JudgeCache(None)).judge(ITEM)
    assert (outcome.verdict, outcome.error, len(llm.calls)) == (None, "provider: billing", 2)
    refused = Judge(FakeLLM([result(None, "SAFETY")]), "m", JudgeCache(tmp_path)).judge(ITEM)
    assert refused.error == "finish: SAFETY"
    broken = Judge(FakeLLM([result('{"facts": []}')]), "m", JudgeCache(tmp_path)).judge(ITEM)
    assert broken.verdict is None and broken.error is not None and "invalid output" in broken.error
    assert not (tmp_path / "judge_cache.jsonl").exists()


def test_numeric_cross_check_flags_disagreements() -> None:
    """Numbers of the reference that the question doesn't hold are looked up with every locale reading."""
    check = numeric_check(ITEM.question, "13,000 hours at 90 °C.", "13.000 horas")
    assert (check.expected, check.found, check.recall) == (["13,000"], ["13,000"], 1.0)
    assert numeric_check("q", "No numbers here.", "anything").recall is None
    missing = numeric_check("q", "9,500 hours.", "12,000 hours")
    assert missing.recall == 0
    assert disagreement(JudgeOutcome("correct", False, [], "m"), missing) is not None
    assert disagreement(JudgeOutcome("wrong", True, [], "m"), check) is not None
    assert disagreement(JudgeOutcome("correct", False, [], "m"), check) is None
    assert disagreement(JudgeOutcome("partial", False, [], "m"), check) is not None
    assert disagreement(JudgeOutcome(None, False, [], "m"), missing) is None


def test_response_text_lists_the_options() -> None:
    """An ambiguous answer is judged with its summary and every option."""
    body = {
        "answer": "Depends on the oil.",
        "options": [{"condition": "CLP HC", "answer": "7,500 h"}, {"condition": "CLP PG", "answer": "13,000 h"}],
    }
    assert response_text(body) == "Depends on the oil.\nOptions:\n- CLP HC: 7,500 h\n- CLP PG: 13,000 h"
    assert response_text({"answer": "13,000 h"}) == "13,000 h"
