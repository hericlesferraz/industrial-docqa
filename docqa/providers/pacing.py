"""Per-model call slots and pauses, shared by every worker through the `provider_state` table.

A slot spaces two OCR calls to the same model by at least `OCR_MIN_INTERVAL_S`, across threads and workers. A pause
stops calls to a model, or to the whole account (`'*'`), for `PROVIDER_PAUSE_S` when retrying can't help: the daily
quota is spent, the prepaid balance or the spend cap ran out, the key or the model was rejected. No job attempt is
spent while paused; the first call after the pause ends is the probe.
"""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from psycopg import Connection

from docqa.providers.llm import ProviderError

PROVIDER_PAUSE_S = 900
ACCOUNT = "*"
PAUSE_KINDS = ("quota_exhausted", "billing", "misconfigured")


@dataclass(frozen=True)
class Pause:
    """An active pause.

    Attributes:
        model: The paused model, or `'*'` for the whole account.
        until: When the pause ends.
        reason: `quota_exhausted`, `billing` or `misconfigured`.
        detail: Sanitized provider message.
    """

    model: str
    until: datetime
    reason: str
    detail: str | None


def acquire_slot(conn: Connection, model: str, min_interval_s: float) -> datetime:
    """Reserve the next call slot for a model, shared by every worker.

    Args:
        conn: Connection in autocommit mode or inside a short transaction.
        model: Model about to be called.
        min_interval_s: Minimum gap between two calls to this model.

    Returns:
        When this call may start; the caller sleeps until then, outside any transaction.
    """
    row = conn.execute(
        "INSERT INTO provider_state AS p (model, next_call_at) VALUES (%(model)s, now() + make_interval(secs => %(gap)s))"
        " ON CONFLICT (model) DO UPDATE SET next_call_at = greatest(p.next_call_at, now()) + make_interval(secs => %(gap)s)"
        " RETURNING next_call_at - make_interval(secs => %(gap)s) AS slot",
        {"model": model, "gap": min_interval_s},
    ).fetchone()
    assert row is not None
    return row["slot"]


def seconds_until(moment: datetime) -> float:
    """Return how long to wait until a moment.

    Args:
        moment: A timezone-aware time.

    Returns:
        Seconds from now, 0 when the moment has passed.
    """
    return max(0.0, (moment - datetime.now(UTC)).total_seconds())


def hold(conn: Connection, model: str, seconds: float) -> None:
    """Push a model's next free slot at least `seconds` into the future, after a per-minute rate limit.

    Args:
        conn: Any connection.
        model: The rate-limited model.
        seconds: The wait the provider asked for.
    """
    conn.execute(
        "INSERT INTO provider_state AS p (model, next_call_at) VALUES (%(model)s, now() + make_interval(secs => %(s)s))"
        " ON CONFLICT (model) DO UPDATE SET next_call_at = greatest(p.next_call_at, now() + make_interval(secs => %(s)s))",
        {"model": model, "s": seconds},
    )


def pause(conn: Connection, model: str, reason: str, detail: str, until: datetime) -> None:
    """Pause calls to a model, or to the whole account with `model='*'`.

    Args:
        conn: Connection inside a transaction.
        model: Model name, or `'*'`.
        reason: `quota_exhausted`, `billing` or `misconfigured`.
        detail: Sanitized provider message.
        until: End of the pause; the first claim after it is the probe.

    Raises:
        ValueError: The reason is unknown.
    """
    if reason not in PAUSE_KINDS:
        raise ValueError(f"unknown pause reason {reason!r}")
    conn.execute(
        "INSERT INTO provider_state AS p (model, paused_until, pause_reason, pause_detail) VALUES (%s, %s, %s, %s)"
        " ON CONFLICT (model) DO UPDATE SET paused_until = excluded.paused_until,"
        " pause_reason = excluded.pause_reason, pause_detail = excluded.pause_detail",
        (model, until, reason, detail),
    )


def active_pause(conn: Connection, model: str) -> Pause | None:
    """Return the pause that blocks a model, checking its own row and `'*'`.

    Args:
        conn: Any connection.
        model: Model about to be called.

    Returns:
        The latest-ending active pause, or None.
    """
    return first_pause(conn, [model])


def first_pause(conn: Connection, models: list[str]) -> Pause | None:
    """Return the active pause that blocks the first of several models, `'*'` included.

    Args:
        conn: Any connection.
        models: Models in order of importance (the default OCR model first).

    Returns:
        The account pause, else the pause of the earliest model in `models` that is paused, or None.
    """
    rows = conn.execute(
        "SELECT model, paused_until, pause_reason, pause_detail FROM provider_state"
        " WHERE model = ANY(%s) AND paused_until > now()",
        ([ACCOUNT, *models],),
    ).fetchall()
    order = {name: rank for rank, name in enumerate([ACCOUNT, *models])}
    rows.sort(key=lambda row: (order[row["model"]], -row["paused_until"].timestamp()))
    if not rows:
        return None
    row = rows[0]
    return Pause(model=row["model"], until=row["paused_until"], reason=row["pause_reason"], detail=row["pause_detail"])


def pause_for(conn: Connection, error: ProviderError, *, now: datetime | None = None) -> Pause:
    """Pause what a provider error blocks: the whole account for billing or a rejected key, else its model.

    Args:
        conn: Connection inside a transaction.
        error: A `quota_exhausted`, `billing` or `misconfigured` error.
        now: Current time; tests pass a fixed one.

    Returns:
        The pause written.
    """
    target = ACCOUNT if error.account_wide or error.kind == "billing" or not error.model else error.model
    until = (now or datetime.now(UTC)) + timedelta(seconds=PROVIDER_PAUSE_S)
    pause(conn, target, error.kind, error.message, until)
    return Pause(model=target, until=until, reason=error.kind, detail=error.message)
