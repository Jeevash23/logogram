"""Portable CSV exports of a saved run: one row per site and prompt, or one row per site."""

from __future__ import annotations

import csv
import io
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

# The prompt's text, added to each row when asked for.
TEXT_COLUMNS = ("clean", "corrupt", "answer", "distractor")


def _cell(value: Any) -> Any:
    # Spreadsheet programs interpret these prefixes even in quoted CSV cells.
    if isinstance(value, str) and value.startswith(("=", "+", "-", "@", "\t", "\r", "\n")):
        return "'" + value
    return value


def _answer(value: Any) -> str:
    return " | ".join(value) if isinstance(value, list) else str(value)


def results_csv(path: Path, records: list[Any] | None = None) -> Iterator[str]:
    """results.parquet as CSV, one row per site and prompt; with ``records`` (the run's prompts,
    in order), each row also carries its prompt's text."""
    table = pq.ParquetFile(path)
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    names = table.schema_arrow.names
    writer.writerow([*names, *(TEXT_COLUMNS if records is not None else ())])
    yield buffer.getvalue()
    for batch in table.iter_batches(batch_size=4096):
        buffer.seek(0)
        buffer.truncate(0)
        for row in batch.to_pylist():
            cells = [_cell(v) for v in row.values()]
            if records is not None:
                i = row.get("prompt")
                record = records[i] if isinstance(i, int) and 0 <= i < len(records) else None
                cells += (
                    [
                        _cell(record.clean),
                        _cell(record.corrupt),
                        _cell(_answer(record.answer)),
                        _cell(_answer(record.distractor)),
                    ]
                    if record is not None
                    else ["", "", "", ""]
                )
            writer.writerow(cells)
        yield buffer.getvalue()


def sites_csv(rows: list[dict[str, Any]]) -> str:
    """One row per site (report.sites_table) as CSV."""
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    if rows:
        writer.writerow(list(rows[0]))
        for row in rows:
            writer.writerow([_cell(v) for v in row.values()])
    return buffer.getvalue()


def sweep_csv(rows: list[dict[str, Any]]) -> str:
    """A sweep's table, one row per site and value (sweeps.table_rows), as CSV. A value without
    results has empty cells, never zeros."""
    from logogram.sweeps import TABLE_COLUMNS

    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(TABLE_COLUMNS)
    for row in rows:
        writer.writerow([_cell(row.get(name)) for name in TABLE_COLUMNS])
    return buffer.getvalue()
