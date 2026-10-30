"""The answer schema (DECISIONS #12): one shape per status, sent to Gemini as an `anyOf` and validated again here.

In each shape `status` is a single-value enum, and citation items are an enum of the request's labels, so the model
can't produce a label that doesn't exist. Evidence comes before the text inside each shape. With no passages at all, the
schema is the `not_found` shape alone, because an empty enum is invalid.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

MAX_CITATIONS = 6
OPTIONS_MIN = 2
OPTIONS_MAX = 5

Citations = Annotated[list[str], Field(min_length=1, max_length=MAX_CITATIONS)]


class InvalidOutput(Exception):
    """The model output is not JSON, doesn't fit the schema, or cites a label the request doesn't have."""


class Answered(BaseModel):
    """The passages answer the question.

    Attributes:
        status: `answered`.
        citations: 1 to 6 labels with the evidence.
        answer: The answer, in the question's language.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["answered"]
    citations: Citations
    answer: str


class Option(BaseModel):
    """One case of an ambiguous answer.

    Attributes:
        condition: The case.
        citations: 1 to 6 labels with its evidence.
        answer: The answer for that case.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    condition: str
    citations: Citations
    answer: str


class Ambiguous(BaseModel):
    """The answer depends on something the question doesn't settle.

    Attributes:
        status: `ambiguous`.
        depends_on: One sentence: what the answer depends on.
        options: 2 to 5 cases.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["ambiguous"]
    depends_on: str
    options: Annotated[list[Option], Field(min_length=OPTIONS_MIN, max_length=OPTIONS_MAX)]


class NotFound(BaseModel):
    """The passages don't answer the question.

    Attributes:
        status: `not_found`.
        answer: What the passages don't state.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["not_found"]
    answer: str


ModelOutput = Answered | Ambiguous | NotFound
_OUTPUT = TypeAdapter(Annotated[ModelOutput, Field(discriminator="status")])


def _label_number(label: str) -> int:
    """Order labels numerically, so `S10` comes after `S9`.

    Args:
        label: A request label such as `S12`.

    Returns:
        Its number.
    """
    return int(label[1:])


def _not_found_shape() -> dict[str, object]:
    """Build the `not_found` shape.

    Returns:
        The JSON schema object.
    """
    return {
        "type": "object",
        "properties": {"status": {"type": "string", "enum": ["not_found"]}, "answer": {"type": "string"}},
        "required": ["status", "answer"],
    }


def response_schema(labels: list[str]) -> dict[str, object]:
    """Build the JSON schema of the answer call (DECISIONS #12): one shape per status.

    Args:
        labels: The request's labels.

    Returns:
        The `anyOf` of the three shapes, with citations as an enum of the labels sorted numerically; the `not_found`
        shape alone when there are no labels.
    """
    if not labels:
        return _not_found_shape()
    cite = {
        "type": "array",
        "items": {"type": "string", "enum": sorted(labels, key=_label_number)},
        "minItems": 1,
        "maxItems": MAX_CITATIONS,
    }
    answered = {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["answered"]},
            "citations": cite,
            "answer": {"type": "string"},
        },
        "required": ["status", "citations", "answer"],
    }
    option = {
        "type": "object",
        "properties": {"condition": {"type": "string"}, "citations": cite, "answer": {"type": "string"}},
        "required": ["condition", "citations", "answer"],
    }
    ambiguous = {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["ambiguous"]},
            "depends_on": {"type": "string"},
            "options": {"type": "array", "minItems": OPTIONS_MIN, "maxItems": OPTIONS_MAX, "items": option},
        },
        "required": ["status", "depends_on", "options"],
    }
    return {"anyOf": [answered, ambiguous, _not_found_shape()]}


def cited_labels(output: ModelOutput) -> list[str]:
    """List every label an output cites, options included, in order.

    Args:
        output: A parsed output.

    Returns:
        The labels, repeated when cited twice.
    """
    if isinstance(output, Answered):
        return list(output.citations)
    if isinstance(output, Ambiguous):
        return [label for option in output.options for label in option.citations]
    return []


def parse_output(text: str | None, labels: set[str]) -> ModelOutput:
    """Parse and validate the model's JSON on our side.

    Args:
        text: The model's output text.
        labels: The request's labels.

    Returns:
        The output, one of the three shapes.

    Raises:
        InvalidOutput: Empty output, invalid JSON, a shape the schema doesn't allow, or an unknown label.
    """
    if not text or not text.strip():
        raise InvalidOutput("empty output")
    try:
        output = _OUTPUT.validate_json(text)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in error['loc']) or 'output'}: {error['msg']}" for error in exc.errors()
        )
        raise InvalidOutput(problems[:500]) from None
    unknown = [label for label in cited_labels(output) if label not in labels]
    if unknown:
        raise InvalidOutput(f"unknown labels {sorted(set(unknown))}")
    return output
