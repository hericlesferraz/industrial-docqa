"""Command line of the eval harness (run by `make eval` in the compose `eval` service).

`python -m docqa.evaluation run --pdfs DIR --questions FILE [--questions FILE ...] --out DIR [--repeat N]
[--concurrency K] [--no-judge] [--judge-model M] [--judge-cache DIR] [--ingest-timeout S]` and
`python -m docqa.evaluation validate --pdfs DIR --questions FILE [...]`.

Exit codes: 0 when the run finished, whatever the scores; 2 when the run can't start (configuration, a missing folder
or one with no readable file, no usable question); 1 when the harness itself failed (API unreachable, a call the run
depends on refused). A bad line of a question file never stops `run`: it is skipped and listed in the report.
`validate` prints every problem and exits 2 when a line would be skipped or a source file is missing.
"""

import argparse
import sys
from pathlib import Path

from docqa.config import ConfigError, load_settings
from docqa.evaluation.client import EvalClient, HarnessError
from docqa.evaluation.dataset import DatasetError, InputProblem, load_sets
from docqa.evaluation.harness import RunConfig, run
from docqa.evaluation.judge import Judge, JudgeCache
from docqa.evaluation.report import render
from docqa.providers.gemini import GeminiGateway
from docqa.telemetry import MemoryTelemetry


def log(message: str) -> None:
    """Print progress on stderr, so stdout carries only the report.

    Args:
        message: The line.
    """
    print(message, file=sys.stderr, flush=True)


def describe(problem: InputProblem) -> str:
    """Render an input problem for the console.

    Args:
        problem: The problem.

    Returns:
        `file:line: skipped|kept: reason`.
    """
    where = f"{problem.file}:{problem.line}" if problem.line else problem.file
    return f"{where}: {'skipped' if problem.skipped else 'kept'}: {problem.reason}"


def parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Returns:
        The parser.
    """
    root = argparse.ArgumentParser(prog="python -m docqa.evaluation")
    commands = root.add_subparsers(dest="command", required=True)
    for name in ("run", "validate"):
        command = commands.add_parser(name)
        command.add_argument("--pdfs", type=Path, required=True, help="folder of files to upload")
        command.add_argument(
            "--questions", type=Path, action="append", required=True, help="question file (repeat for several sets)"
        )
    run_command = commands.choices["run"]
    run_command.add_argument("--out", type=Path, required=True, help="folder for results.jsonl and report.md")
    run_command.add_argument("--repeat", type=int, default=1, help="times each question is asked")
    run_command.add_argument("--concurrency", type=int, default=1, help="questions in flight at once")
    run_command.add_argument("--no-judge", action="store_true", help="skip the content judge (no judge cost)")
    run_command.add_argument("--judge-model", help="judge model (default: EVAL_JUDGE_MODEL)")
    run_command.add_argument("--judge-cache", type=Path, help="folder of the judge's verdict cache")
    run_command.add_argument("--ingest-timeout", type=float, help="seconds to wait (default: EVAL_INGEST_TIMEOUT_S)")
    return root


def main(argv: list[str] | None = None) -> int:
    """Run a command.

    Args:
        argv: Arguments; `sys.argv[1:]` when None.

    Returns:
        The exit code.
    """
    args = parser().parse_args(argv)
    try:
        settings = load_settings(require_database=False)
        questions, problems = load_sets(args.questions, args.pdfs)
    except (ConfigError, DatasetError) as exc:
        print(f"invalid input: {exc}", file=sys.stderr)
        return 2
    for problem in problems:
        log(describe(problem))
    if args.command == "validate":
        blocking = any(p.skipped for p in problems) or any(q.missing_files() for q in questions)
        print(f"{len(questions)} questions in {len(args.questions)} file(s); {len(problems)} problem(s)")
        return 2 if blocking else 0
    if args.repeat < 1 or args.concurrency < 1:
        print("invalid input: --repeat and --concurrency must be at least 1", file=sys.stderr)
        return 2
    judge_model = None if args.no_judge else (args.judge_model or settings.eval_judge_model)
    config = RunConfig(
        pdf_dir=args.pdfs,
        question_files=args.questions,
        out_dir=args.out,
        api_url=settings.eval_api_url,
        repeat=args.repeat,
        concurrency=args.concurrency,
        judge_model=judge_model,
        ingest_timeout_s=args.ingest_timeout or float(settings.eval_ingest_timeout_s),
    )
    judge = None
    if judge_model:
        judge = Judge(
            GeminiGateway(settings.gemini_api_key, MemoryTelemetry()), judge_model, JudgeCache(args.judge_cache)
        )
    client = EvalClient(settings.eval_api_url, settings.admin_api_key)
    try:
        outcome = run(config, questions, client, judge, problems=problems, log=log)
    except HarnessError as exc:
        print(f"harness error: {exc}", file=sys.stderr)
        return 1
    finally:
        client.close()
    report = render(outcome)
    (config.out_dir / "report.md").write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
