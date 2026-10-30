"""Question sets in the format of `eval/public_questions.jsonl`, read leniently against the folder of files.

The harness relies only on the public fields: `id`, `question`, `expected_status`, `expected_answer` and `sources`
(`[{"file", "pages"}]`). `lang` is optional and only feeds the per-language breakdown; any other field is ignored.

A line the run can't score (not JSON, a required field missing, an unknown status, a repeated id) is skipped, and a
part of a line that is unusable (a malformed source, a source file that isn't in the folder) is dropped or marked
while the question is still asked. Each case becomes an `InputProblem`, listed at the top of the report, so one bad
line never costs the whole run. Source files match the folder by base name, ignoring case and Unicode normalization.
"""

import codecs
import json
import os
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

STATUSES = ("answered", "ambiguous", "not_found")
PATH_SEPARATORS = re.compile(r"[\\/]")


class DatasetError(Exception):
    """The run can't start: the folder is missing or has no readable file, or no question file has a usable line."""


@dataclass(frozen=True)
class InputProblem:
    """A line of a question file that was skipped, or asked with part of it ignored.

    Attributes:
        file: Question file name.
        line: 1-based line number, or None for a problem with the whole file.
        skipped: The line's question is not asked.
        reason: What was wrong, and what the harness did about it.
    """

    file: str
    line: int | None
    skipped: bool
    reason: str

    def to_json(self) -> dict[str, object]:
        """Render for `results.jsonl`.

        Returns:
            A JSON-ready dict.
        """
        return {"file": self.file, "line": self.line, "skipped": self.skipped, "reason": self.reason}


@dataclass(frozen=True)
class Source:
    """A page set that answers a question.

    Attributes:
        file: Name of the matching file in the folder, or the name as written when no file matches.
        pages: 1-based pages of that file; any one of them counts.
        in_folder: A file of the folder matches the name.
    """

    file: str
    pages: tuple[int, ...]
    in_folder: bool = True


@dataclass(frozen=True)
class Question:
    """One line of a question file.

    Attributes:
        set_name: Name of the set, the file name without its extension.
        id: Question id, unique within the set.
        question: The question as asked.
        expected_status: `answered`, `ambiguous` or `not_found`.
        expected_answer: Reference answer written by a person.
        sources: Where the answer is; empty for `not_found`.
        lang: Language code, when the file gives one.
    """

    set_name: str
    id: str
    question: str
    expected_status: str
    expected_answer: str
    sources: tuple[Source, ...]
    lang: str | None

    def expected_pages(self) -> set[tuple[str, int]]:
        """Return every `(file, page)` that counts as the source.

        Returns:
            The pairs; empty for `not_found`.
        """
        return {(source.file, page) for source in self.sources for page in source.pages}

    def missing_files(self) -> list[str]:
        """Return the source files that no file of the folder matches.

        Returns:
            The names as written in the question file, sorted.
        """
        return sorted({source.file for source in self.sources if not source.in_folder})


def file_key(name: str) -> str:
    """Reduce a file name to the key used to match it: base name only, NFC, case folded.

    Args:
        name: A file name, possibly with a path prefix (either separator).

    Returns:
        The key.
    """
    base = PATH_SEPARATORS.split(name.strip())[-1]
    return unicodedata.normalize("NFC", base).casefold()


class FolderFiles:
    """The names of the files of the folder, matched tolerantly against the names a question file writes."""

    def __init__(self, names: list[str]) -> None:
        """Index the names by exact name and by key.

        Args:
            names: Names of the files the harness uploads.
        """
        self.names = set(names)
        self._by_key: dict[str, list[str]] = defaultdict(list)
        for name in sorted(names):
            self._by_key[file_key(name)].append(name)

    def resolve(self, written: str) -> list[str]:
        """Find the folder files a written name refers to.

        Args:
            written: The `file` of a source.

        Returns:
            The exact name when it exists; otherwise every name with the same key (usually one, possibly none).
        """
        if written in self.names:
            return [written]
        return list(self._by_key.get(file_key(written), []))


def _identifier(value: object) -> str | None:
    """Read an id: a non-empty string, or an integer written without quotes.

    Args:
        value: The raw `id`.

    Returns:
        The id as text, or None when unusable.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _pages(value: object) -> tuple[int, ...] | None:
    """Read the `pages` of a source: page numbers from 1, as a list or a single number (digit strings accepted).

    Args:
        value: The raw `pages`.

    Returns:
        The pages, or None when unusable.
    """
    items = value if isinstance(value, list) else [value]
    pages: list[int] = []
    for item in items:
        page = int(item.strip()) if isinstance(item, str) and item.strip().isdigit() else item
        if isinstance(page, bool) or not isinstance(page, int) or page < 1:
            return None
        pages.append(page)
    return tuple(pages) or None


def _sources(raw: object, folder: FolderFiles, notes: list[str]) -> tuple[Source, ...]:
    """Read the `sources` field; an unusable source is dropped, a file missing from the folder is kept and marked.

    Args:
        raw: The raw `sources` (a list, or one object).
        folder: The folder's file names.
        notes: Where the caveats are appended.

    Returns:
        The usable sources.
    """
    if raw is None:
        return ()
    items = [raw] if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        notes.append("`sources` is not a list; source and retrieval are not checked")
        return ()
    found: list[Source] = []
    for n, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            notes.append(f"source {n} is not an object with `file` and `pages`; ignored")
            continue
        file, pages = item.get("file"), _pages(item.get("pages"))
        if not isinstance(file, str) or not file.strip():
            notes.append(f"source {n} has no `file`; ignored")
            continue
        if pages is None:
            notes.append(f"source {n} ({file!r}): `pages` must be page numbers from 1; ignored")
            continue
        matches = folder.resolve(file)
        if not matches:
            notes.append(f"source {n}: file {file!r} is not in the folder; asked anyway, its source reported missing")
            found.append(Source(file=file, pages=pages, in_folder=False))
            continue
        if len(matches) > 1:
            notes.append(f"source {n}: file {file!r} matches {', '.join(matches)}; using {matches[0]!r}")
        found.append(Source(file=matches[0], pages=pages))
    return tuple(found)


def parse_line(line: str, *, set_name: str, folder: FolderFiles) -> tuple[Question | None, list[str]]:
    """Parse one line of a question file.

    Required: `id`, `question`, `expected_status` and, unless the status is `not_found`, `expected_answer`. A missing
    `sources` means none; a malformed source or `lang` is dropped with a note.

    Args:
        line: The JSON line.
        set_name: Name of the set.
        folder: The folder's file names.

    Returns:
        The question and its caveats, or None and the reasons the line is skipped.
    """
    try:
        row = json.loads(line)
    except json.JSONDecodeError as exc:
        return None, [f"not valid JSON ({exc.msg})"]
    if not isinstance(row, dict):
        return None, ["not a JSON object"]
    reasons: list[str] = []
    qid = _identifier(row.get("id"))
    if qid is None:
        reasons.append("`id` is missing or empty")
    question = row.get("question")
    if not isinstance(question, str) or not question.strip():
        reasons.append("`question` is missing or empty")
    raw_status = row.get("expected_status")
    status = raw_status.strip().lower() if isinstance(raw_status, str) else None
    if raw_status is None:
        reasons.append("`expected_status` is missing")
    elif status not in STATUSES:
        reasons.append(f"unknown `expected_status` {raw_status!r} (expected one of {', '.join(STATUSES)})")
    expected_answer = row.get("expected_answer")
    if isinstance(expected_answer, int | float) and not isinstance(expected_answer, bool):
        expected_answer = str(expected_answer)
    if expected_answer is None and status == "not_found":
        expected_answer = ""
    if not isinstance(expected_answer, str):
        reasons.append("`expected_answer` is missing or not text")
    if reasons:
        return None, reasons
    notes: list[str] = []
    sources = _sources(row.get("sources"), folder, notes)
    if not sources and status != "not_found":
        notes.append("no usable `sources`: source and retrieval are not checked")
    lang = row.get("lang")
    if lang is not None and not isinstance(lang, str):
        notes.append("`lang` is not text; ignored")
        lang = None
    return (
        Question(
            set_name=set_name,
            id=str(qid),
            question=str(question),
            expected_status=str(status),
            expected_answer=str(expected_answer),
            sources=sources,
            lang=lang or None,
        ),
        notes,
    )


def pdf_folder_files(pdf_dir: Path) -> list[Path]:
    """List the files the harness uploads: every regular, non-hidden file of the folder, PDFs or not.

    Args:
        pdf_dir: The folder.

    Returns:
        The files, sorted by name.

    Raises:
        DatasetError: The folder doesn't exist.
    """
    if not pdf_dir.is_dir():
        raise DatasetError(f"{pdf_dir}: the PDF folder does not exist")
    return sorted(p for p in pdf_dir.iterdir() if p.is_file() and not p.name.startswith("."))


def _lines(path: Path) -> list[tuple[int, str | None]]:
    """Split a JSONL file into lines, tolerating a byte-order mark and CRLF line ends.

    Args:
        path: The file.

    Returns:
        `(line number, text)`; text is None for a line that is not UTF-8.
    """
    data = path.read_bytes().removeprefix(codecs.BOM_UTF8)
    lines: list[tuple[int, str | None]] = []
    for number, raw in enumerate(data.split(b"\n"), start=1):
        try:
            lines.append((number, raw.rstrip(b"\r").decode("utf-8")))
        except UnicodeDecodeError:
            lines.append((number, None))
    return lines


def load_questions(
    path: Path, folder: FolderFiles, *, set_name: str | None = None
) -> tuple[list[Question], list[InputProblem]]:
    """Load one question file, skipping the lines the run can't score.

    Args:
        path: The JSONL file.
        folder: The folder's file names.
        set_name: Name of the set; the file name without its extension by default.

    Returns:
        The questions in file order, and the problems found.
    """
    name = path.name
    try:
        lines = _lines(path)
    except OSError as exc:
        return [], [InputProblem(name, None, True, f"the question file can't be read ({exc.strerror or exc})")]
    set_name = set_name or path.stem
    questions: list[Question] = []
    problems: list[InputProblem] = []
    first_line: dict[str, int] = {}
    for number, line in lines:
        if line is None:
            problems.append(InputProblem(name, number, True, "not UTF-8 text"))
            continue
        if not line.strip():
            continue
        question, notes = parse_line(line, set_name=set_name, folder=folder)
        if question is None:
            problems.append(InputProblem(name, number, True, "; ".join(notes)))
            continue
        if question.id in first_line:
            reason = f"duplicate id {question.id!r} (first on line {first_line[question.id]})"
            problems.append(InputProblem(name, number, True, reason))
            continue
        first_line[question.id] = number
        questions.append(question)
        problems.extend(InputProblem(name, number, False, note) for note in notes)
    if not questions and not problems:
        problems.append(InputProblem(name, None, True, "no questions in the file"))
    return questions, problems


def readable_files(pdf_dir: Path) -> list[Path]:
    """List the folder's files and check that the run has something to upload.

    Args:
        pdf_dir: The folder.

    Returns:
        The files the harness uploads.

    Raises:
        DatasetError: The folder doesn't exist, or none of its files can be read.
    """
    files = pdf_folder_files(pdf_dir)
    if not any(os.access(p, os.R_OK) for p in files):
        raise DatasetError(f"{pdf_dir}: no readable file to upload")
    return files


def load_sets(paths: list[Path], pdf_dir: Path) -> tuple[list[Question], list[InputProblem]]:
    """Load several question files against the folder; a set is named after its file (`-2`, `-3` on a clash).

    Args:
        paths: The JSONL files.
        pdf_dir: The folder of files to upload.

    Returns:
        Every question, set by set, and every problem found.

    Raises:
        DatasetError: The folder is missing or has no readable file, or no file has a usable question.
    """
    folder = FolderFiles([p.name for p in readable_files(pdf_dir)])
    questions: list[Question] = []
    problems: list[InputProblem] = []
    used: set[str] = set()
    for path in paths:
        set_name, n = path.stem, 1
        while set_name in used:
            n += 1
            set_name = f"{path.stem}-{n}"
        used.add(set_name)
        found, issues = load_questions(path, folder, set_name=set_name)
        questions.extend(found)
        problems.extend(issues)
    if not questions:
        detail = "\n".join(f"{p.file}:{p.line or '-'}: {p.reason}" for p in problems)
        raise DatasetError("no question to ask" + (f":\n{detail}" if detail else ""))
    return questions, problems
