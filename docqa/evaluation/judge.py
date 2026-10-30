"""Reference-guided judge of answer content, with the verdict computed by code.

The model only lists the facts of the reference answer, marks each one essential or an example, and says whether the
response states, misses or contradicts it. The verdict is a rule over those facts, so it can be audited and changed
without asking the model again. Verdicts are cached by a hash of everything that shapes them; a numeric cross-check
flags verdicts that disagree with the numbers of the reference, for a person to review.
"""

import hashlib
import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from docqa.providers.llm import LLM, GenConfig, ProviderError, Turn
from docqa.text.numbers import numbers, pool

JUDGE_PROMPT_VERSION = "j2"
JUDGE_STAGE = "eval_judge"
JUDGE_TIMEOUT_S = 90.0
JUDGE_MAX_OUTPUT_TOKENS = 4096
JUDGE_RETRY_KINDS = {"transient", "rate_limited", "timeout"}
JUDGE_RETRY_WAIT_S = 2.0
VERDICTS = ("stated", "missing", "contradicted")
CACHE_FILE = "judge_cache.jsonl"

JUDGE_SYSTEM = """You check an answer from a document question-answering system against a reference answer written \
by a person who read the documents.

Steps:
1. Split the reference answer into short, atomic facts: each value with its unit and the condition it applies to, \
each named item, and each required behaviour (for example "the answer depends on the frame size" or "asks which oil \
type is used"). Don't add facts that are not in the reference.
2. Mark each fact essential or not. A fact is not essential (essential = false) when the reference gives it as an \
example or illustration (after "e.g.", "for example", "such as") or marks it as optional ("may be stated", "not \
required"). Every other fact is essential.
3. Compare each fact with the response:
   - "stated": the response says it, in any language or wording. Numbers are equal across formats: 9,500 = 9500 = \
9.500 hours; 12 meses = 12 months; 49 m = 49m.
   - "missing": the response doesn't say it.
   - "contradicted": the response gives a different value, a different condition, or the opposite for the same thing.
4. A condition that the question itself states (a speed, a frame size, a temperature, an oil type) counts as stated \
even when the response doesn't repeat it: the reference often restates the question's conditions.
5. Judge only against the reference; don't use your own knowledge. Extra details in the response earn nothing and \
cost nothing, unless they contradict a fact of the reference.
6. In "note", quote the words of the response that state or contradict the fact, or write "absent".

Return only the JSON object."""

JUDGE_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "minItems": 1,
            "maxItems": 12,
            "items": {
                "type": "object",
                "properties": {
                    "fact": {"type": "string"},
                    "essential": {"type": "boolean"},
                    "verdict": {"type": "string", "enum": list(VERDICTS)},
                    "note": {"type": "string"},
                },
                "required": ["fact", "essential", "verdict", "note"],
            },
        }
    },
    "required": ["facts"],
}


@dataclass(frozen=True)
class JudgeInput:
    """What the judge sees.

    Attributes:
        question: The question.
        expected_status: Status the reference expects.
        expected_answer: Reference answer.
        response_status: Status the system returned.
        response_text: The system's answer as text (see `response_text`).
    """

    question: str
    expected_status: str
    expected_answer: str
    response_status: str
    response_text: str


@dataclass(frozen=True)
class Fact:
    """A fact of the reference and how the response treats it.

    Attributes:
        fact: The fact, in the judge's words.
        essential: False for an example or an optional detail.
        verdict: `stated`, `missing` or `contradicted`.
        note: Words of the response that state or contradict it, or `absent`.
    """

    fact: str
    essential: bool
    verdict: str
    note: str


@dataclass
class JudgeOutcome:
    """The judge's result for one response.

    Attributes:
        verdict: `correct`, `partial` or `wrong`; None when the judge failed.
        contradicted: Some fact of the reference is contradicted (a wrong value or condition).
        facts: The facts as the judge listed them.
        model: Judge model.
        prompt_version: Judge prompt version.
        cached: The facts came from the cache; no call was made.
        input_tokens: Tokens of the call (0 when cached).
        output_tokens: Output tokens of the call, thinking included (0 when cached).
        error: Why the judge failed, if it did.
    """

    verdict: str | None
    contradicted: bool
    facts: list[Fact]
    model: str
    prompt_version: str = JUDGE_PROMPT_VERSION
    cached: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None

    def to_json(self) -> dict[str, object]:
        """Render for `results.jsonl`.

        Returns:
            A JSON-ready dict.
        """
        return asdict(self)


@dataclass(frozen=True)
class NumericCheck:
    """Numbers of the reference found in the response, the judge's cross-check.

    Attributes:
        expected: Numeric tokens of the reference, without those the question already holds.
        found: The ones with a matching reading in the response.
        recall: `len(found) / len(expected)`, or None when the reference has no numbers of its own.
    """

    expected: list[str] = field(default_factory=list)
    found: list[str] = field(default_factory=list)
    recall: float | None = None


def response_text(body: dict[str, object]) -> str:
    """Render an answer body as the text the judge grades: the answer and, for `ambiguous`, every option.

    Args:
        body: A 200 body of `POST /v1/questions`.

    Returns:
        The answer, then one line per option (`- <condition>: <answer>`).
    """
    lines = [str(body.get("answer") or "")]
    options = body.get("options")
    if isinstance(options, list) and options:
        lines.append("Options:")
        lines.extend(
            f"- {option.get('condition', '')}: {option.get('answer', '')}"
            for option in options
            if isinstance(option, dict)
        )
    return "\n".join(lines)


def verdict_from_facts(facts: list[Fact]) -> tuple[str, bool]:
    """Compute the verdict from the judge's facts.

    Correct: every essential fact stated and nothing contradicted. Partial: nothing contradicted, some essential fact
    missing. Wrong: something contradicted, or no essential fact stated. If the judge marked no fact essential, every
    fact counts as essential.

    Args:
        facts: The judge's facts.

    Returns:
        The verdict and whether a fact is contradicted.
    """
    contradicted = any(fact.verdict == "contradicted" for fact in facts)
    essentials = [fact for fact in facts if fact.essential] or facts
    stated = sum(fact.verdict == "stated" for fact in essentials)
    if contradicted or stated == 0:
        return "wrong", contradicted
    if stated == len(essentials):
        return "correct", False
    return "partial", False


def numeric_check(question: str, expected_answer: str, answer_text: str) -> NumericCheck:
    """Look for the numbers of the reference in the response, with every locale reading (DECISIONS #7).

    Numbers the question already holds (a frame size, a speed, a temperature) are conditions, not answers, and are
    left out.

    Args:
        question: The question.
        expected_answer: The reference answer.
        answer_text: The response text.

    Returns:
        The check.
    """
    asked = pool([question])
    expected = [(tok, found) for tok, found in numbers(expected_answer) if not (found & asked)]
    if not expected:
        return NumericCheck()
    available = pool([answer_text])
    found = [tok for tok, readings in expected if readings & available]
    return NumericCheck(expected=[tok for tok, _ in expected], found=found, recall=len(found) / len(expected))


def disagreement(outcome: JudgeOutcome, check: NumericCheck) -> str | None:
    """Flag a verdict that the numeric cross-check contradicts, for a person to review.

    Args:
        outcome: The judge's result.
        check: The numeric check of the same response.

    Returns:
        Why the verdict needs review, or None.
    """
    if outcome.verdict is None or check.recall is None:
        return None
    if outcome.verdict == "correct" and check.recall == 0:
        return "judged correct, but no number of the reference appears in the answer"
    if outcome.contradicted and check.recall == 1:
        return "judged contradicted, but every number of the reference appears in the answer"
    if outcome.verdict == "partial" and check.recall == 1:
        return "judged partial, but every number of the reference appears in the answer"
    return None


def cache_key(model: str, item: JudgeInput) -> str:
    """Hash everything that shapes a verdict.

    Args:
        model: Judge model.
        item: The judged input.

    Returns:
        The sha256 hex digest of the model, the prompt version and the input.
    """
    blob = json.dumps({"model": model, "prompt": JUDGE_PROMPT_VERSION, **asdict(item)}, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class JudgeCache:
    """Verdict facts by hash, in an append-only JSON-lines file shared by runs."""

    def __init__(self, folder: Path | None) -> None:
        """Load the cache file, if any.

        Args:
            folder: Folder of `judge_cache.jsonl`; None keeps the cache in memory only.
        """
        self._path = folder / CACHE_FILE if folder is not None else None
        self._lock = threading.Lock()
        self._rows: dict[str, list[Fact]] = {}
        if self._path is not None and self._path.is_file():
            for line in self._path.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                    self._rows[row["key"]] = [Fact(**fact) for fact in row["facts"]]
                except (ValueError, KeyError, TypeError):
                    continue

    def get(self, key: str) -> list[Fact] | None:
        """Return cached facts.

        Args:
            key: Cache key.

        Returns:
            The facts, or None.
        """
        with self._lock:
            return self._rows.get(key)

    def put(self, key: str, facts: list[Fact]) -> None:
        """Store facts and append them to the file.

        Args:
            key: Cache key.
            facts: The judge's facts.
        """
        with self._lock:
            self._rows[key] = facts
            if self._path is not None:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                with self._path.open("a", encoding="utf-8") as handle:
                    row = {"key": key, "prompt": JUDGE_PROMPT_VERSION, "facts": [asdict(f) for f in facts]}
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_facts(text: str | None) -> list[Fact]:
    """Parse the judge's output.

    Args:
        text: The model's JSON text.

    Returns:
        The facts.

    Raises:
        ValueError: The output is not the expected JSON.
    """
    data = json.loads(text or "")
    raw = data.get("facts") if isinstance(data, dict) else None
    if not isinstance(raw, list) or not raw:
        raise ValueError("no facts in the judge output")
    facts = []
    for item in raw:
        if not isinstance(item, dict) or item.get("verdict") not in VERDICTS:
            raise ValueError("a fact has no valid verdict")
        facts.append(
            Fact(
                fact=str(item.get("fact", "")),
                essential=bool(item.get("essential", True)),
                verdict=str(item["verdict"]),
                note=str(item.get("note", "")),
            )
        )
    return facts


def user_content(item: JudgeInput) -> str:
    """Build the judge's user message.

    Args:
        item: The judged input.

    Returns:
        The message.
    """
    return (
        f"Question: {item.question}\n"
        f"Expected status: {item.expected_status}\n"
        f"Reference answer: {item.expected_answer}\n\n"
        f"Response to check (status {item.response_status}):\n{item.response_text}"
    )


class Judge:
    """Calls the judge model once per uncached response."""

    def __init__(self, llm: LLM, model: str, cache: JudgeCache) -> None:
        """Keep the gateway, the model and the cache.

        Args:
            llm: Provider gateway.
            model: Judge model (`EVAL_JUDGE_MODEL`).
            cache: Verdict cache.
        """
        self.model = model
        self._llm = llm
        self._cache = cache

    def judge(self, item: JudgeInput) -> JudgeOutcome:
        """Judge one response: from the cache, or with one call (one more after a transient failure).

        Args:
            item: The judged input.

        Returns:
            The outcome; `verdict` is None and `error` says why when the judge failed.
        """
        key = cache_key(self.model, item)
        cached = self._cache.get(key)
        if cached is not None:
            verdict, contradicted = verdict_from_facts(cached)
            return JudgeOutcome(verdict, contradicted, cached, self.model, cached=True)
        config = GenConfig(
            system_instruction=JUDGE_SYSTEM,
            response_json_schema=JUDGE_SCHEMA,
            thinking_level="low",
            max_output_tokens=JUDGE_MAX_OUTPUT_TOKENS,
            temperature=0.0,
        )
        contents = [Turn(role="user", parts=[user_content(item)])]
        input_tokens = output_tokens = 0
        for attempt in (1, 2):
            try:
                result = self._llm.generate(
                    model=self.model,
                    contents=contents,
                    config=config,
                    timeout_s=JUDGE_TIMEOUT_S,
                    stage=JUDGE_STAGE,
                    attempt=attempt,
                )
            except ProviderError as exc:
                if exc.kind in JUDGE_RETRY_KINDS and attempt == 1:
                    time.sleep(JUDGE_RETRY_WAIT_S)
                    continue
                return JudgeOutcome(None, False, [], self.model, error=f"provider: {exc.kind}")
            input_tokens += result.input_tokens or 0
            output_tokens += (result.output_tokens or 0) + (result.thinking_tokens or 0)
            if result.finish_reason != "STOP":
                error = f"finish: {result.finish_reason}"
                return JudgeOutcome(
                    None, False, [], self.model, input_tokens=input_tokens, output_tokens=output_tokens, error=error
                )
            try:
                facts = parse_facts(result.text)
            except ValueError as exc:
                error = f"invalid output: {exc}"
                return JudgeOutcome(
                    None, False, [], self.model, input_tokens=input_tokens, output_tokens=output_tokens, error=error
                )
            self._cache.put(key, facts)
            verdict, contradicted = verdict_from_facts(facts)
            return JudgeOutcome(
                verdict, contradicted, facts, self.model, input_tokens=input_tokens, output_tokens=output_tokens
            )
        return JudgeOutcome(None, False, [], self.model, error="provider: retries exhausted")
