"""Question files of the eval: the public format, read leniently against the folder of files."""

import json
from pathlib import Path

import pytest

from docqa.evaluation.dataset import DatasetError, FolderFiles, file_key, load_questions, load_sets, pdf_folder_files

ROOT = Path(__file__).resolve().parent.parent


def _write(path: Path, rows: list[object]) -> Path:
    """Write rows as JSON lines (strings are written as they are).

    Args:
        path: Output file.
        rows: Objects, or raw strings for broken lines.

    Returns:
        The path.
    """
    lines = [row if isinstance(row, str) else json.dumps(row) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def pdf_dir(tmp_path: Path) -> Path:
    """A folder with one file and one hidden file.

    Args:
        tmp_path: Pytest temp folder.

    Returns:
        The folder.
    """
    folder = tmp_path / "pdfs"
    folder.mkdir()
    (folder / "manual.pdf").write_bytes(b"%PDF-1.7\n")
    (folder / ".hidden.pdf").write_bytes(b"%PDF-1.7\n")
    return folder


def _row(**changes: object) -> dict[str, object]:
    """Build a valid question with some fields changed.

    Args:
        **changes: Fields to replace; a value of None removes the field.

    Returns:
        The row.
    """
    row: dict[str, object] = {
        "id": "q-1",
        "question": "How long?",
        "expected_status": "answered",
        "expected_answer": "12 months.",
        "sources": [{"file": "manual.pdf", "pages": [2]}],
    }
    for key, value in changes.items():
        if value is None:
            row.pop(key, None)
        else:
            row[key] = value
    return row


def _folder(pdf_dir: Path) -> FolderFiles:
    """Index the names of the folder's files as the harness does.

    Args:
        pdf_dir: The folder.

    Returns:
        The index.
    """
    return FolderFiles([p.name for p in pdf_folder_files(pdf_dir)])


def test_the_public_questions_load_against_the_case_files() -> None:
    """The shipped public file loads against `case_files/` with no problem."""
    questions, problems = load_questions(ROOT / "eval" / "public_questions.jsonl", _folder(ROOT / "case_files"))
    assert problems == []
    assert [q.id for q in questions] == [f"pub-0{n}" for n in range(1, 8)]
    assert questions[0].set_name == "public_questions"
    assert questions[0].expected_pages() == {("LB5001.pdf", 2)}
    assert questions[0].lang is None


def test_the_committed_sets_are_valid() -> None:
    """The design and holdout sets load together with the public one, with a language on each question."""
    files = [ROOT / "eval" / name for name in ("public_questions.jsonl", "questions_design.jsonl")]
    files.append(ROOT / "eval" / "questions_holdout.jsonl")
    questions, problems = load_sets(files, ROOT / "case_files")
    assert problems == []
    assert len(questions) == 7 + 18 + 48
    assert all(q.lang for q in questions if q.set_name != "public_questions")


def test_optional_language_and_extra_fields(tmp_path: Path, pdf_dir: Path) -> None:
    """`lang` is read when present; unknown fields are ignored.

    Args:
        tmp_path: Pytest temp folder.
        pdf_dir: The PDF folder.
    """
    path = _write(tmp_path / "set.jsonl", [_row(lang="pt", extra="ignored"), _row(id="q-2")])
    (first, second), problems = load_questions(path, _folder(pdf_dir))
    assert (first.lang, second.lang, problems) == ("pt", None, [])


@pytest.mark.parametrize(
    ("row", "reason"),
    [
        ("not json", "not valid JSON"),
        ("[1, 2]", "not a JSON object"),
        (_row(id=None), "`id` is missing"),
        (_row(question="  "), "`question` is missing or empty"),
        (_row(expected_status=None), "`expected_status` is missing"),
        (_row(expected_status="needs_clarification"), "unknown `expected_status` 'needs_clarification'"),
        (_row(expected_answer=None), "`expected_answer` is missing"),
        (_row(id="ok"), "duplicate id 'ok' (first on line 1)"),
    ],
)
def test_a_bad_line_is_skipped_and_the_rest_still_load(tmp_path: Path, pdf_dir: Path, row: object, reason: str) -> None:
    """Each kind of unusable line is skipped with its file, line and reason; the lines around it still load.

    Args:
        tmp_path: Pytest temp folder.
        pdf_dir: The PDF folder.
        row: The bad line.
        reason: Expected part of the reason.
    """
    path = _write(tmp_path / "set.jsonl", [_row(id="ok"), row, _row(id="after")])
    questions, problems = load_questions(path, _folder(pdf_dir))
    assert [q.id for q in questions] == ["ok", "after"]
    [problem] = problems
    assert (problem.file, problem.line, problem.skipped) == ("set.jsonl", 2, True)
    assert reason in problem.reason


@pytest.mark.parametrize(
    ("row", "note", "sources"),
    [
        (_row(sources="manual.pdf"), "`sources` is not a list", ()),
        (_row(sources=[{"file": "manual.pdf", "pages": []}]), "`pages` must be page numbers", ()),
        (_row(sources=[{"file": "manual.pdf", "pages": [0]}]), "`pages` must be page numbers", ()),
        (_row(sources=[3]), "is not an object", ()),
        (_row(sources=None), "no usable `sources`", ()),
        (_row(lang=3), "`lang` is not text", (("manual.pdf", (2,), True),)),
        (_row(sources=[{"file": "other.pdf", "pages": [1]}]), "'other.pdf' is not in", (("other.pdf", (1,), False),)),
        (_row(sources=[{"file": ".hidden.pdf", "pages": [1]}]), "is not in", ((".hidden.pdf", (1,), False),)),
    ],
)
def test_an_unusable_part_is_dropped_and_the_question_is_still_asked(
    tmp_path: Path, pdf_dir: Path, row: object, note: str, sources: tuple[tuple[str, tuple[int, ...], bool], ...]
) -> None:
    """A malformed source or `lang` is dropped and a missing file is marked, with a caveat; the question stays.

    Args:
        tmp_path: Pytest temp folder.
        pdf_dir: The PDF folder.
        row: The line.
        note: Expected part of the caveat.
        sources: Expected `(file, pages, in_folder)` of the kept sources.
    """
    path = _write(tmp_path / "set.jsonl", [row])
    [question], problems = load_questions(path, _folder(pdf_dir))
    assert [(s.file, s.pages, s.in_folder) for s in question.sources] == list(sources)
    assert {(p.line, p.skipped) for p in problems} == {(1, False)}
    assert note in problems[0].reason


def test_a_missing_source_file_is_reported_by_the_question(tmp_path: Path, pdf_dir: Path) -> None:
    """The question keeps the missing source, names it, and keeps the sources that are in the folder.

    Args:
        tmp_path: Pytest temp folder.
        pdf_dir: The PDF folder.
    """
    sources = [{"file": "gone.pdf", "pages": [3]}, {"file": "manual.pdf", "pages": [2]}]
    [question], _ = load_questions(_write(tmp_path / "set.jsonl", [_row(sources=sources)]), _folder(pdf_dir))
    assert question.missing_files() == ["gone.pdf"]
    assert question.expected_pages() == {("gone.pdf", 3), ("manual.pdf", 2)}


def test_file_names_match_by_base_name_case_and_unicode_form(tmp_path: Path, pdf_dir: Path) -> None:
    """A path prefix, another case or the decomposed Unicode form still find the folder's file, by its own name.

    Args:
        tmp_path: Pytest temp folder.
        pdf_dir: The PDF folder.
    """
    (pdf_dir / "Manual Técnico.PDF").write_bytes(b"%PDF-1.7\n")
    written = ["docs/manual.pdf", "C:\\in\\MANUAL.PDF", "manual técnico.pdf", "x/Manual Te\u0301cnico.pdf"]
    rows = [_row(id=f"q-{n}", sources=[{"file": name, "pages": [1]}]) for n, name in enumerate(written)]
    questions, problems = load_questions(_write(tmp_path / "set.jsonl", rows), _folder(pdf_dir))
    assert problems == []
    assert [q.sources[0].file for q in questions] == ["manual.pdf"] * 2 + ["Manual Técnico.PDF"] * 2
    assert file_key("a/B\\Ç.pdf") == file_key("ç.PDF")


def test_byte_order_mark_crlf_blank_lines_and_lenient_values(tmp_path: Path, pdf_dir: Path) -> None:
    """A BOM, CRLF line ends, blank lines, a numeric id, a capitalized status and a single page number all load.

    Args:
        tmp_path: Pytest temp folder.
        pdf_dir: The PDF folder.
    """
    rows = [
        json.dumps(_row(id=7, expected_status=" Answered ", sources=[{"file": "manual.pdf", "pages": 2}])),
        "",
        "   ",
        json.dumps(_row(id="q-2", expected_status="not_found", expected_answer=None, sources=None)),
    ]
    path = tmp_path / "bom.jsonl"
    path.write_bytes(b"\xef\xbb\xbf" + "\r\n".join(rows).encode() + b"\r\n\xff\xfe\r\n")
    questions, problems = load_questions(path, _folder(pdf_dir))
    assert [(q.id, q.expected_status, q.expected_answer) for q in questions] == [
        ("7", "answered", "12 months."),
        ("q-2", "not_found", ""),
    ]
    assert questions[0].sources[0].pages == (2,)
    assert [(p.line, p.skipped, p.reason) for p in problems] == [(5, True, "not UTF-8 text")]


def test_only_a_run_with_nothing_to_do_is_refused(tmp_path: Path, pdf_dir: Path) -> None:
    """No usable question, a missing folder or one without files stop the run; a missing file alone doesn't.

    Args:
        tmp_path: Pytest temp folder.
        pdf_dir: The PDF folder.
    """
    good = _write(tmp_path / "good.jsonl", [_row()])
    questions, problems = load_sets([good, tmp_path / "missing.jsonl"], pdf_dir)
    assert len(questions) == 1
    assert [(p.file, p.line, p.skipped) for p in problems] == [("missing.jsonl", None, True)]
    with pytest.raises(DatasetError, match="no question to ask"):
        load_sets([_write(tmp_path / "bad.jsonl", ["oops", ""])], pdf_dir)
    with pytest.raises(DatasetError, match="no question to ask"):
        load_sets([_write(tmp_path / "empty.jsonl", [""])], pdf_dir)
    with pytest.raises(DatasetError, match="PDF folder does not exist"):
        load_sets([good], tmp_path / "nowhere")
    (tmp_path / "bare").mkdir()
    (tmp_path / "bare" / ".hidden.pdf").write_bytes(b"%PDF-1.7\n")
    with pytest.raises(DatasetError, match="no readable file"):
        load_sets([good], tmp_path / "bare")


def test_sets_with_the_same_name_are_renamed_and_problems_of_every_file_are_kept(tmp_path: Path, pdf_dir: Path) -> None:
    """Two files with one name become two sets; the problems of several files are all listed.

    Args:
        tmp_path: Pytest temp folder.
        pdf_dir: The PDF folder.
    """
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    one = _write(tmp_path / "a" / "set.jsonl", [_row(), "oops"])
    two = _write(tmp_path / "b" / "set.jsonl", [_row(expected_status="no"), _row()])
    questions, problems = load_sets([one, two], pdf_dir)
    assert [q.set_name for q in questions] == ["set", "set-2"]
    assert [(p.file, p.line) for p in problems] == [("set.jsonl", 2), ("set.jsonl", 1)]


def test_the_folder_listing_skips_hidden_files_and_folders(pdf_dir: Path) -> None:
    """Every regular, non-hidden file is uploaded, PDF or not, whatever its extension's case; folders are skipped.

    Args:
        pdf_dir: The PDF folder.
    """
    (pdf_dir / "notes.txt").write_text("not a pdf")
    (pdf_dir / "UPPER.PDF").write_bytes(b"%PDF-1.7\n")
    (pdf_dir / "sub").mkdir()
    (pdf_dir / "sub" / "inner.pdf").write_bytes(b"%PDF-1.7\n")
    assert [p.name for p in pdf_folder_files(pdf_dir)] == ["UPPER.PDF", "manual.pdf", "notes.txt"]
