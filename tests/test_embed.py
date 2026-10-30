"""Embedders: the int8 ONNX model against the prototype's vectors, the Gemini batching, and the gateway call."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from google.genai import errors as genai_errors
from google.genai import types

from docqa.config import Settings, load_settings
from docqa.index.embed import (
    GEMINI_BATCH_TEXTS,
    GEMINI_EMBED_MODEL,
    MODEL_DIR,
    GeminiEmbedder,
    LocalOnnxEmbedder,
    gemini_batches,
    get_embedder,
    index_version,
)
from docqa.providers.fake import FakeLLM
from docqa.providers.gemini import GeminiGateway
from docqa.providers.llm import ProviderError
from docqa.telemetry import MemoryTelemetry
from tests.helpers import ADMIN_KEY

FIXTURE = Path(__file__).parent / "fixtures" / "embed_queries.npz"


@pytest.fixture(scope="module")
def onnx_embedder() -> LocalOnnxEmbedder:
    """Load the image's int8 model once per module.

    Returns:
        The embedder.
    """
    return LocalOnnxEmbedder(MODEL_DIR)


def test_query_vectors_match_the_prototype(onnx_embedder: LocalOnnxEmbedder) -> None:
    """Question vectors are within cosine 0.999 of the measured prototype's, each embedded alone, on one int8 file.

    Args:
        onnx_embedder: The loaded model.
    """
    fixture = np.load(FIXTURE)
    for query, expected in zip(fixture["queries"], fixture["vectors"], strict=True):
        got = onnx_embedder.embed_query(str(query))
        assert got.shape == (1024,)
        assert got.dtype == np.float32
        assert abs(float(np.linalg.norm(got)) - 1.0) < 1e-4
        assert float(got @ expected) > 0.999, query


def test_documents_come_back_in_input_order(onnx_embedder: LocalOnnxEmbedder) -> None:
    """One text per call gives each text its own vector; a larger batch keeps the input order and is named apart.

    Args:
        onnx_embedder: The loaded model.
    """
    texts = ["A long passage about regreasing ball bearings every 9500 hours at 1800 RPM. " * 6, "W3Seal", "Óleo"]
    alone = np.array([onnx_embedder.embed_query(text) for text in texts])
    assert np.allclose(onnx_embedder.embed_documents(texts), alone, atol=1e-6)
    assert onnx_embedder.embed_documents([]).shape == (0, 1024)
    onnx_embedder.batch_size = 3
    try:
        batched = onnx_embedder.embed_documents(texts)
        assert onnx_embedder.model_id == "bge-m3-int8@5617a9f6/b3"
    finally:
        onnx_embedder.batch_size = 1
    cosines = (batched * alone).sum(axis=1)
    assert all(cosine > 0.95 for cosine in cosines)
    assert onnx_embedder.model_id == "bge-m3-int8@5617a9f6"


def test_index_version_names_chunker_and_model(onnx_embedder: LocalOnnxEmbedder) -> None:
    """The index version joins the chunker version and the model revision.

    Args:
        onnx_embedder: The loaded model.
    """
    assert index_version(onnx_embedder) == "win250-head4|bge-m3-int8@5617a9f6"
    assert index_version(GeminiEmbedder(FakeLLM())) == "win250-head4|gemini-embedding-2@1024"


def _settings(provider: str) -> Settings:
    """Build settings with an embedding provider.

    Args:
        provider: `local` or `gemini`.

    Returns:
        The settings.
    """
    return load_settings(
        {"GEMINI_API_KEY": "k", "ADMIN_API_KEY": ADMIN_KEY, "DATABASE_URL": "x", "EMBEDDING_PROVIDER": provider}
    )


def test_get_embedder_picks_the_provider() -> None:
    """`EMBEDDING_PROVIDER=gemini` gives the Gemini embedder, which needs the gateway."""
    assert isinstance(get_embedder(_settings("gemini"), FakeLLM()), GeminiEmbedder)
    with pytest.raises(ValueError, match="gateway"):
        get_embedder(_settings("gemini"), None)


def test_gemini_batches_respect_count_and_tokens() -> None:
    """Batches hold at most 50 texts and about 8,000 estimated tokens; a huge text goes alone."""
    small = ["word " * 20] * 120
    assert [len(batch) for batch in gemini_batches(small)] == [GEMINI_BATCH_TEXTS, GEMINI_BATCH_TEXTS, 20]
    big = ["x" * 4 * 5000, "x" * 4 * 5000, "short"]
    assert gemini_batches(big) == [[0], [1, 2]]
    huge = ["x" * 4 * 9000, "y"]
    assert gemini_batches(huge) == [[0], [1]]


class _EmbedLLM(FakeLLM):
    """A fake gateway whose `embed` records its calls and returns scaled vectors.

    Attributes:
        embed_calls: `(texts, task, dim, stage)` per call.
    """

    def __init__(self) -> None:
        """Start with no calls."""
        super().__init__()
        self.embed_calls: list[tuple[list[str], str, int, str]] = []

    def embed(self, *, model: str, texts: list[str], task: str, dim: int, stage: str) -> np.ndarray:
        """Record the call and return non-normalized vectors.

        Args:
            model: Embedding model name.
            texts: Texts.
            task: Task type.
            dim: Dimensions.
            stage: `docqa.stage` value.

        Returns:
            One vector per text, of norm 3.
        """
        assert model == GEMINI_EMBED_MODEL
        self.embed_calls.append((texts, task, dim, stage))
        vectors = np.zeros((len(texts), dim), dtype=np.float32)
        for row, text in enumerate(texts):
            vectors[row, len(text) % dim] = 3.0
        return vectors


def test_gemini_embedder_batches_and_normalizes() -> None:
    """Chunks go as `RETRIEVAL_DOCUMENT` in batches, questions as `RETRIEVAL_QUERY`; vectors come back normalized."""
    llm = _EmbedLLM()
    embedder = GeminiEmbedder(llm)
    texts = [f"chunk {n}" + "x" * n for n in range(60)]
    vectors = embedder.embed_documents(texts)
    assert vectors.shape == (60, 1024)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)
    assert vectors[5, len(texts[5]) % 1024] == pytest.approx(1.0)
    assert [(len(t), task, dim, stage) for t, task, dim, stage in llm.embed_calls] == [
        (50, "RETRIEVAL_DOCUMENT", 1024, "embed_chunks"),
        (10, "RETRIEVAL_DOCUMENT", 1024, "embed_chunks"),
    ]
    query = embedder.embed_query("How often?")
    assert query.shape == (1024,)
    assert llm.embed_calls[-1][1:] == ("RETRIEVAL_QUERY", 1024, "embed_query")


class _Models:
    """Stand-in for `client.models` that records `embed_content` calls.

    Attributes:
        calls: Keyword arguments of each call.
        response: What `embed_content` returns, or the exception it raises.
    """

    def __init__(self, response: object) -> None:
        """Keep the scripted response.

        Args:
            response: Returned or raised by `embed_content`.
        """
        self.calls: list[dict[str, object]] = []
        self.response = response

    def embed_content(self, **kwargs: object) -> object:
        """Record the call.

        Args:
            **kwargs: The call's arguments.

        Returns:
            The scripted response.

        Raises:
            Exception: The scripted exception.
        """
        self.calls.append(kwargs)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def _gateway(response: object, telemetry: MemoryTelemetry) -> tuple[GeminiGateway, _Models]:
    """Build a gateway whose client is replaced by the stand-in.

    Args:
        response: Scripted response of `embed_content`.
        telemetry: Where the call spans go.

    Returns:
        The gateway and the stand-in.
    """
    gateway = GeminiGateway("secret-key", telemetry)
    models = _Models(response)
    gateway._client = SimpleNamespace(models=models)
    return gateway, models


def _response(n: int, tokens: int | None = None) -> SimpleNamespace:
    """Build an embedding response.

    Args:
        n: Number of vectors.
        tokens: Token count per vector, when the provider reports it.

    Returns:
        An object shaped like `EmbedContentResponse`.
    """
    stats = SimpleNamespace(token_count=tokens) if tokens else None
    return SimpleNamespace(embeddings=[SimpleNamespace(values=[float(i + 1)] * 4, statistics=stats) for i in range(n)])


def test_gateway_sends_one_content_per_text() -> None:
    """Each text is its own `types.Content` (a list of strings would be one aggregated vector), with task and size."""
    telemetry = MemoryTelemetry()
    gateway, models = _gateway(_response(2, tokens=7), telemetry)
    vectors = gateway.embed(model=GEMINI_EMBED_MODEL, texts=["a b", "c"], task="RETRIEVAL_DOCUMENT", dim=4, stage="s")
    assert vectors.shape == (2, 4)
    call = models.calls[0]
    contents = call["contents"]
    assert isinstance(contents, list) and len(contents) == 2
    assert all(isinstance(content, types.Content) and len(content.parts or []) == 1 for content in contents)
    assert [content.parts[0].text for content in contents] == ["a b", "c"]
    config = call["config"]
    assert isinstance(config, types.EmbedContentConfig)
    assert (config.task_type, config.output_dimensionality) == ("RETRIEVAL_DOCUMENT", 4)
    telemetry.flush()
    [span] = telemetry.named("gen_ai.call")
    assert span.attributes["gen_ai.operation.name"] == "embeddings"
    assert span.attributes["docqa.stage"] == "s"
    assert span.attributes["gen_ai.usage.input_tokens"] == 14
    assert span.attributes["docqa.embed.count"] == 2


def test_gateway_rejects_a_vector_count_mismatch() -> None:
    """Fewer vectors than texts is a provider failure, never silently misaligned rows."""
    gateway, _ = _gateway(_response(1), MemoryTelemetry())
    with pytest.raises(ProviderError) as caught:
        gateway.embed(model=GEMINI_EMBED_MODEL, texts=["a", "b"], task="RETRIEVAL_QUERY", dim=4, stage="s")
    assert caught.value.kind == "transient"


def test_gateway_classifies_embedding_errors() -> None:
    """A 429 on the embedding call is classified like any other call, and the key never reaches the message."""
    body = {"error": {"code": 429, "message": "secret-key Resource exhausted", "status": "RESOURCE_EXHAUSTED"}}
    error = genai_errors.ClientError(429, body)
    gateway, _ = _gateway(error, MemoryTelemetry())
    with pytest.raises(ProviderError) as caught:
        gateway.embed(model=GEMINI_EMBED_MODEL, texts=["a"], task="RETRIEVAL_QUERY", dim=4, stage="s")
    assert caught.value.kind == "rate_limited"
    assert "secret-key" not in caught.value.message
