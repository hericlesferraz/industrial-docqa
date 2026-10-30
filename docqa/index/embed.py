"""Embedders (DECISIONS #10): BGE-M3 int8 on CPU through ONNX Runtime by default, Gemini Embedding 2 by flag.

Both give L2-normalized 1024-dimension vectors in the same column. One provider per deployment: the embedder's
`model_id` enters the index version, so switching providers reindexes the pages and never mixes vectors.
"""

import json
from pathlib import Path
from typing import Protocol

import numpy as np
import onnxruntime
from tokenizers import Tokenizer

from docqa.config import Settings
from docqa.index.chunking import CHUNKER_VERSION, ntok
from docqa.providers.llm import LLM

MODEL_DIR = Path("/models/bge-m3-int8")
TOKENIZER_MAX_LEN = 2048
EMBED_DIM = 1024
LOCAL_MODEL_NAME = "bge-m3-int8"
GEMINI_EMBED_MODEL = "gemini-embedding-2"
GEMINI_BATCH_TEXTS = 50
GEMINI_BATCH_TOKENS = 8000
PAD_TOKEN = "<pad>"
SENTENCE_OUTPUT = "sentence_embedding"


class Embedder(Protocol):
    """Turns texts into L2-normalized 1024-dimension vectors.

    Attributes:
        model_id: Identifier that enters the index version.
        dim: Vector size, 1024.
        provider: `local` or `gemini`, for telemetry.
    """

    model_id: str
    dim: int
    provider: str

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed chunk texts.

        Args:
            texts: Chunk texts, context line included.

        Returns:
            Array of shape (len(texts), 1024), float32, L2-normalized.
        """
        ...

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a question.

        Args:
            text: The question as typed.

        Returns:
            Array of shape (1024,), float32, L2-normalized.
        """
        ...


def normalize(vectors: np.ndarray) -> np.ndarray:
    """Scale each row to unit length.

    Args:
        vectors: Array of shape (n, dim).

    Returns:
        Float32 array of the same shape with L2-normalized rows; an all-zero row stays zero.
    """
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return (vectors / np.where(norms == 0, 1.0, norms)).astype(np.float32)


def index_version(embedder: Embedder) -> str:
    """Return the index version of a deployment: chunker and embedding model.

    Args:
        embedder: The deployment's embedder.

    Returns:
        For example `win250-head4|bge-m3-int8@5617a9f6`.
    """
    return f"{CHUNKER_VERSION}|{embedder.model_id}"


def local_model_id(model_dir: Path) -> str:
    """Name the local model by the upstream revision recorded at build time.

    Args:
        model_dir: Folder with `model.onnx`, `tokenizer.json` and `MODEL_INFO.json`.

    Returns:
        `bge-m3-int8@<first 8 characters of the revision>`.

    Raises:
        FileNotFoundError: `MODEL_INFO.json` is missing.
    """
    info = json.loads((model_dir / "MODEL_INFO.json").read_text(encoding="utf-8"))
    return f"{LOCAL_MODEL_NAME}@{str(info['revision'])[:8]}"


class LocalOnnxEmbedder:
    """BGE-M3 int8 through ONNX Runtime and the `tokenizers` tokenizer; no torch.

    The int8 weights are quantized dynamically: each matrix product quantizes its input with one range for the whole
    batch, padding included, so a text's vector depends slightly on the texts embedded with it (cosine about 0.98
    between batch sizes 8 and 1). One text per call (the default) gives every text the same vector it gets alone, as a
    question does, and on CPU it is also faster. A larger batch is recorded in the model id, so its vectors never mix
    with those of another batch size.

    Attributes:
        model_id: `bge-m3-int8@<revision>`, plus `/b<batch size>` when the batch size is not 1.
        dim: 1024.
        provider: `local`.
        batch_size: Texts per ONNX call; larger batches are sorted by length to cut padding.
    """

    dim = EMBED_DIM
    provider = "local"

    def __init__(self, model_dir: Path = MODEL_DIR, batch_size: int = 1) -> None:
        """Load the model and the tokenizer, and run one tiny inference so a broken file fails now.

        Args:
            model_dir: Folder with `model.onnx`, `tokenizer.json` and `MODEL_INFO.json`.
            batch_size: Texts per ONNX call (`EMBED_BATCH_SIZE`).
        """
        self._base_id = local_model_id(model_dir)
        self.batch_size = batch_size
        self._session = onnxruntime.InferenceSession(str(model_dir / "model.onnx"), providers=["CPUExecutionProvider"])
        self._tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self._tokenizer.enable_truncation(TOKENIZER_MAX_LEN)
        self._pad = self._tokenizer.token_to_id(PAD_TOKEN)
        self._inputs = {item.name for item in self._session.get_inputs()}
        self._sentence = SENTENCE_OUTPUT in {item.name for item in self._session.get_outputs()}
        self._encode(["warm-up"])

    @property
    def model_id(self) -> str:
        """Name the model and, when it is not 1, the batch size, which changes the vectors.

        Returns:
            For example `bge-m3-int8@5617a9f6` or `bge-m3-int8@5617a9f6/b8`.
        """
        return self._base_id if self.batch_size == 1 else f"{self._base_id}/b{self.batch_size}"

    def _encode(self, texts: list[str]) -> np.ndarray:
        """Run one batch through the model.

        Args:
            texts: The batch.

        Returns:
            Normalized vectors, one row per text.
        """
        encodings = self._tokenizer.encode_batch(texts)
        width = max(len(encoding.ids) for encoding in encodings)
        ids = np.full((len(encodings), width), self._pad, dtype=np.int64)
        mask = np.zeros((len(encodings), width), dtype=np.int64)
        for row, encoding in enumerate(encodings):
            ids[row, : len(encoding.ids)] = encoding.ids
            mask[row, : len(encoding.ids)] = 1
        feed = {name: value for name, value in (("input_ids", ids), ("attention_mask", mask)) if name in self._inputs}
        if self._sentence:
            vectors = self._session.run([SENTENCE_OUTPUT], feed)[0]
        else:
            vectors = self._session.run(None, feed)[0][:, 0]
        return normalize(vectors)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed chunk texts, `batch_size` per call, shortest first.

        Args:
            texts: Chunk texts.

        Returns:
            Array of shape (len(texts), 1024) in input order.
        """
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        order = sorted(range(len(texts)), key=lambda index: len(texts[index]))
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for start in range(0, len(order), self.batch_size):
            batch = order[start : start + self.batch_size]
            out[batch] = self._encode([texts[index] for index in batch])
        return out

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a question, with no prefix (BGE-M3 uses none).

        Args:
            text: The question.

        Returns:
            Array of shape (1024,).
        """
        return self._encode([text])[0]


def gemini_batches(texts: list[str]) -> list[list[int]]:
    """Group texts for Gemini calls: at most `GEMINI_BATCH_TEXTS` texts and about `GEMINI_BATCH_TOKENS` tokens each.

    Args:
        texts: Texts to embed.

    Returns:
        Lists of text indexes, in order; a text larger than the token budget gets a call of its own.
    """
    batches: list[list[int]] = []
    current: list[int] = []
    budget = 0
    for index, text in enumerate(texts):
        size = ntok(text)
        if current and (len(current) >= GEMINI_BATCH_TEXTS or budget + size > GEMINI_BATCH_TOKENS):
            batches.append(current)
            current, budget = [], 0
        current.append(index)
        budget += size
    if current:
        batches.append(current)
    return batches


class GeminiEmbedder:
    """Gemini Embedding 2 at 1024 dimensions through the provider gateway, normalized here.

    Attributes:
        model_id: `gemini-embedding-2@1024`.
        dim: 1024.
        provider: `gemini`.
    """

    dim = EMBED_DIM
    provider = "gemini"
    model_id = f"{GEMINI_EMBED_MODEL}@{EMBED_DIM}"

    def __init__(self, llm: LLM) -> None:
        """Keep the gateway; its calls record the `gen_ai.call` spans and classify errors.

        Args:
            llm: Provider gateway.
        """
        self._llm = llm

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed chunk texts as `RETRIEVAL_DOCUMENT`, in batches.

        Args:
            texts: Chunk texts.

        Returns:
            Array of shape (len(texts), 1024) in input order.

        Raises:
            ProviderError: A call failed.
        """
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for batch in gemini_batches(texts):
            vectors = self._llm.embed(
                model=GEMINI_EMBED_MODEL,
                texts=[texts[index] for index in batch],
                task="RETRIEVAL_DOCUMENT",
                dim=self.dim,
                stage="embed_chunks",
            )
            out[batch] = normalize(vectors)
        return out

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a question as `RETRIEVAL_QUERY`.

        Args:
            text: The question.

        Returns:
            Array of shape (1024,).

        Raises:
            ProviderError: The call failed.
        """
        vectors = self._llm.embed(
            model=GEMINI_EMBED_MODEL, texts=[text], task="RETRIEVAL_QUERY", dim=self.dim, stage="embed_query"
        )
        return normalize(vectors)[0]


def get_embedder(settings: Settings, llm: LLM | None) -> Embedder:
    """Build the embedder selected by `EMBEDDING_PROVIDER`.

    Args:
        settings: Process settings.
        llm: Provider gateway, required for `gemini`.

    Returns:
        A loaded embedder.

    Raises:
        ValueError: `gemini` was selected without a gateway.
    """
    if settings.embedding_provider == "gemini":
        if llm is None:
            raise ValueError("EMBEDDING_PROVIDER=gemini needs the provider gateway")
        return GeminiEmbedder(llm)
    return LocalOnnxEmbedder(MODEL_DIR, settings.embed_batch_size)
