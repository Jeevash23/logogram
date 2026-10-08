"""Portable per-prompt CSV exports of a saved result table."""

from __future__ import annotations

import csv
import io
from collections.abc import Iterator
from pathlib import Path

import pyarrow.parquet as pq


def results_csv(path: Path) -> Iterator[str]:
    table = pq.ParquetFile(path)
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(table.schema_arrow.names)
    yield buffer.getvalue()
    for batch in table.iter_batches(batch_size=4096):
        buffer.seek(0)
        buffer.truncate(0)
        for row in batch.to_pylist():
            # Spreadsheet programs interpret these prefixes even in quoted CSV cells.
            writer.writerow(
                [
                    "'" + value
                    if isinstance(value, str)
                    and value.startswith(("=", "+", "-", "@", "\t", "\r", "\n"))
                    else value
                    for value in row.values()
                ]
            )
        yield buffer.getvalue()
