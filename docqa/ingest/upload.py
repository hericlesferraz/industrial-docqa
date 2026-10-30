"""Streaming reader of a one-file multipart upload: the file goes to a temp file while its sha256 is computed.

The API never opens the PDF. While the bytes stream in, it checks only what is cheap: the part is named `file` and
has a filename, the first 1 KiB contains `%PDF-`, the file is not empty and not over the size limit.
"""

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from python_multipart.exceptions import FormParserError
from python_multipart.multipart import MultipartParser, parse_options_header

from docqa.errors import EMPTY_FILE, FILE_TOO_LARGE, INVALID_UPLOAD, NOT_A_PDF

HEADER_LIMIT = 64 * 1024
MAGIC = b"%PDF-"
MAGIC_WINDOW = 1024
FILENAME_MAX = 255
CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f-\x9f]")


class UploadRejected(Exception):
    """The upload can't become a document.

    Attributes:
        status: HTTP status: 413 or 422.
        code: Error code: `file_too_large`, `empty_file`, `not_a_pdf` or `invalid_upload`.
        message: Explanation for the client.
        filename: Cleaned filename of the rejected file, when its part headers were read.
    """

    def __init__(self, status: int, code: str, message: str) -> None:
        """Store the rejection.

        Args:
            status: HTTP status.
            code: Error code.
            message: Explanation for the client.
        """
        super().__init__(code, message)
        self.status = status
        self.code = code
        self.message = message
        self.filename: str | None = None


def too_large(max_bytes: int) -> UploadRejected:
    """Build the rejection of a file over the size limit.

    Args:
        max_bytes: The limit in bytes.

    Returns:
        A 413 `file_too_large` rejection.
    """
    return UploadRejected(413, FILE_TOO_LARGE, f"The file is larger than the {max_bytes // 2**20} MiB limit.")


def invalid(message: str) -> UploadRejected:
    """Build the rejection of a malformed multipart body.

    Args:
        message: What is wrong.

    Returns:
        A 422 `invalid_upload` rejection.
    """
    return UploadRejected(422, INVALID_UPLOAD, message)


def clean_filename(raw: bytes) -> str:
    """Reduce a client filename to a safe display name.

    Args:
        raw: The `filename` parameter of the part, as sent.

    Returns:
        The basename without control characters, at most 255 characters; empty when nothing is left.
    """
    try:
        name = raw.decode("utf-8")
    except UnicodeDecodeError:
        name = raw.decode("latin-1")
    name = re.split(r"[/\\]", name)[-1]
    name = CONTROL_CHARS.sub("", name).strip()
    return "" if name in (".", "..") else name[:FILENAME_MAX]


def boundary_of(content_type: str | None) -> bytes:
    """Read the multipart boundary of a request.

    Args:
        content_type: The request's `Content-Type` header.

    Returns:
        The boundary.

    Raises:
        UploadRejected: The request is not `multipart/form-data` with a boundary.
    """
    kind, options = parse_options_header(content_type or "")
    boundary = options.get(b"boundary")
    if kind != b"multipart/form-data" or not boundary:
        raise invalid("The request must be multipart/form-data with one part named 'file'.")
    return boundary


@dataclass(frozen=True)
class UploadedFile:
    """A file fully received into a temp file.

    Attributes:
        filename: Cleaned client filename.
        sha256: Hex digest of the bytes.
        size: Size in bytes.
    """

    filename: str
    sha256: str
    size: int


class UploadReader:
    """Feeds multipart bytes to a streaming parser and writes the `file` part to a temp file.

    Attributes:
        filename: Cleaned filename of the part, once its headers are read.
        headers_done: Whether the part's headers have been read.
    """

    def __init__(self, boundary: bytes, tmp: Path, max_bytes: int) -> None:
        """Start reading an upload.

        Args:
            boundary: Multipart boundary.
            tmp: Temp file that receives the part's bytes.
            max_bytes: File size limit.

        Raises:
            UploadRejected: The boundary is unusable.
        """
        self.filename: str | None = None
        self.headers_done = False
        self._max_bytes = max_bytes
        self._parts = 0
        self._ended = False
        self._header_field = bytearray()
        self._header_value = bytearray()
        self._headers: dict[bytes, bytes] = {}
        self._size = 0
        self._head = bytearray()
        self._digest = hashlib.sha256()
        try:
            self._parser = MultipartParser(
                boundary,
                {
                    "on_part_begin": self._on_part_begin,
                    "on_header_field": self._on_header_field,
                    "on_header_value": self._on_header_value,
                    "on_header_end": self._on_header_end,
                    "on_headers_finished": self._on_headers_finished,
                    "on_part_data": self._on_part_data,
                    "on_end": self._on_end,
                },
            )
        except FormParserError as exc:
            raise invalid("The multipart boundary is invalid.") from exc
        self._out: BinaryIO = tmp.open("wb")

    def _on_part_begin(self) -> None:
        """Start a part; only one is allowed.

        Raises:
            UploadRejected: A second part.
        """
        self._parts += 1
        if self._parts > 1:
            raise invalid("Send exactly one part, named 'file'.")

    def _on_header_field(self, data: bytes, start: int, end: int) -> None:
        """Collect a header name.

        Args:
            data: Buffer.
            start: Start of the name piece.
            end: End of the name piece.
        """
        self._header_field += data[start:end]

    def _on_header_value(self, data: bytes, start: int, end: int) -> None:
        """Collect a header value.

        Args:
            data: Buffer.
            start: Start of the value piece.
            end: End of the value piece.
        """
        self._header_value += data[start:end]

    def _on_header_end(self) -> None:
        """Store a finished header."""
        self._headers[bytes(self._header_field).strip().lower()] = bytes(self._header_value).strip()
        self._header_field.clear()
        self._header_value.clear()

    def _on_headers_finished(self) -> None:
        """Check that the part is a file named `file`, and keep its filename.

        Raises:
            UploadRejected: The part is not named `file` or has no usable filename.
        """
        disposition, options = parse_options_header(self._headers.get(b"content-disposition", b""))
        if disposition != b"form-data" or options.get(b"name") != b"file":
            raise invalid("The part must be named 'file'.")
        filename = clean_filename(options.get(b"filename", b""))
        if not filename:
            raise invalid("The 'file' part must carry a filename.")
        self.filename = filename
        self.headers_done = True

    def _on_part_data(self, data: bytes, start: int, end: int) -> None:
        """Write a piece of the file, hashing it and checking the size and the PDF header.

        Args:
            data: Buffer.
            start: Start of the piece.
            end: End of the piece.

        Raises:
            UploadRejected: Over the size limit, or no `%PDF-` in the first 1 KiB.
        """
        piece = data[start:end]
        self._size += len(piece)
        if self._size > self._max_bytes:
            raise too_large(self._max_bytes)
        if len(self._head) < MAGIC_WINDOW:
            self._head += piece[: MAGIC_WINDOW - len(self._head)]
            if len(self._head) >= MAGIC_WINDOW:
                self._check_magic()
        self._digest.update(piece)
        self._out.write(piece)

    def _on_end(self) -> None:
        """Note the end of the multipart body."""
        self._ended = True

    def _check_magic(self) -> None:
        """Check that the start of the file looks like a PDF.

        Raises:
            UploadRejected: `%PDF-` is not in the first 1 KiB.
        """
        if MAGIC not in self._head:
            raise UploadRejected(422, NOT_A_PDF, "The file is not a PDF.")

    def feed(self, chunk: bytes) -> None:
        """Parse one chunk of the request body.

        Args:
            chunk: Bytes received.

        Raises:
            UploadRejected: The body is malformed, or the file breaks a rule.
        """
        try:
            self._parser.write(chunk)
        except FormParserError as exc:
            raise invalid("The multipart body is malformed.") from exc

    def finish(self) -> UploadedFile:
        """Close the temp file after the whole body was read and run the final checks.

        Returns:
            The received file.

        Raises:
            UploadRejected: The body ended early, no file part was sent, the file is empty or not a PDF.
        """
        self.close()
        if not self.headers_done or not self._ended:
            raise invalid("The multipart body ended before a complete 'file' part.")
        if self._size == 0:
            raise UploadRejected(422, EMPTY_FILE, "The file is empty.")
        self._check_magic()
        assert self.filename is not None
        return UploadedFile(filename=self.filename, sha256=self._digest.hexdigest(), size=self._size)

    def close(self) -> None:
        """Close the temp file; safe to call twice."""
        if not self._out.closed:
            self._out.close()
