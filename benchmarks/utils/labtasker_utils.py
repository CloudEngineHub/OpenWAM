"""Small, benchmark-agnostic helpers for Labtasker entry points."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import sys
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from labtasker import Client, Task


def new_submission_id() -> str:
    """Return a readable UTC timestamp with a short collision suffix."""

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"sub-{timestamp}-{secrets.token_hex(2)}"


def print_submission_id(submission_id: str) -> None:
    """Highlight the copyable ID in terminals, without escape codes in logs."""
    label = f"SUBMISSION_ID={submission_id}"
    if sys.stdout.isatty() and "NO_COLOR" not in os.environ:
        label = f"\033[1;36m{label}\033[0m"
    print(f"\n{label}\n", flush=True)


def print_summary_table(title: str, headers: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    """Print a compact aligned table without adding a formatting dependency."""

    rendered = [[str(value) for value in row] for row in rows]
    if any(len(row) != len(headers) for row in rendered):
        raise ValueError("summary table rows must match the headers")
    widths = [max(len(header), *(len(row[index]) for row in rendered)) for index, header in enumerate(headers)]

    def line(values: Sequence[str]) -> str:
        return "  ".join(value.ljust(width) for value, width in zip(values, widths, strict=True)).rstrip()

    print(title)
    print(line(headers))
    print(line(["-" * width for width in widths]))
    for row in rendered:
        print(line(row))


def derive_task_id(submission_id: str, task_index: int) -> str:
    """Derive a Labtasker ID so resubmitting one submission is idempotent."""

    digest = hashlib.sha256(f"{submission_id}\0{task_index}".encode()).digest()[:9]
    return "t_" + base64.urlsafe_b64encode(digest).decode()


def build_submission_filter(benchmark: str, submission_id: str, operation: str | None = None) -> str:
    """Build an exact filter without duplicating benchmark args in metadata."""

    clauses = [
        f"metadata.benchmark == {json.dumps(benchmark)}",
        f"metadata.submission_id == {json.dumps(submission_id)}",
    ]
    if operation is not None:
        clauses.append(f"args.operation == {json.dumps(operation)}")
    return " and ".join(clauses)


def list_submission_tasks(
    client: Client, benchmark: str, submission_id: str, operation: str | None = None
) -> list[Task]:
    """Read all pages so summaries and resubmission checks include every batch.

    Query the Server rather than local attempt files, which may contain retries
    or results whose completion was never accepted.
    """

    expression = build_submission_filter(benchmark, submission_id, operation)
    tasks: list[Task] = []
    cursor: str | None = None
    while True:
        page = client.list_tasks(filter=expression, limit=100, cursor=cursor)
        tasks.extend(page.items)
        cursor = page.next_cursor
        if cursor is None:
            return tasks


def submit_tasks(
    client: Client,
    inputs: Iterable[dict[str, Any]],
    *,
    benchmark: str,
    submission_id: str,
    route: str,
    names: Sequence[str],
    max_attempts: int,
    priority: int,
) -> list[str]:
    """Resume partial submission using Labtasker's native idempotent Task creation.

    Resubmit the same definitions and IDs; existing Tasks keep their state and
    only missing Tasks are created. Reusing a submission ID with another
    definition is unsupported.
    """

    inputs = list(inputs)
    task_names = list(names)
    if len(task_names) != len(inputs):
        raise ValueError("Task names and inputs must have the same length")
    metadata = {"benchmark": benchmark, "submission_id": submission_id}

    return [
        client.submit_task(
            args,
            task_id=derive_task_id(submission_id, task_index),
            name=task_name,
            routes=[route],
            metadata=metadata,
            max_attempts=max_attempts,
            priority=priority,
        ).id
        for task_index, (args, task_name) in enumerate(zip(inputs, task_names, strict=True))
    ]


def count_task_statuses(tasks: Sequence[Task]) -> dict[str, int]:
    """Collapse cancelled into failed and keep the requested compact view."""

    counts = {"succeeded": 0, "pending": 0, "running": 0, "failed": 0}
    for task in tasks:
        status = "failed" if task.status == "cancelled" else task.status
        if status not in counts:
            raise ValueError(f"unsupported Labtasker status: {task.status}")
        counts[status] += 1
    return counts | {"expected": len(tasks)}


@contextmanager
def project_context() -> Iterator[None]:
    """Make local submissions and Workers select the same repository project.

    Entry scripts may be launched from different directories. Temporarily using
    the repository root avoids accidentally selecting separate local Servers.
    """
    previous = Path.cwd()
    os.chdir(Path(__file__).resolve().parents[2])
    try:
        yield
    finally:
        os.chdir(previous)
