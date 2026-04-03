"""Metadata filtering for XTR-WARP indexes using DuckDB."""

from __future__ import annotations

import os

import duckdb
import pyarrow as pa

# Column name for passage IDs in the metadata table
_PID_COL = "_passage_id"


def _to_arrow(metadata: list[dict], start_pid: int) -> pa.Table:
    """Convert metadata dicts to a PyArrow table with passage IDs.

    PyArrow infers all types natively — scalars, lists, nested structs,
    list-of-structs, etc. — so no manual type mapping is needed.
    """
    table = pa.Table.from_pylist(metadata)
    pids = pa.array(range(start_pid, start_pid + len(metadata)), type=pa.int64())
    return table.append_column(_PID_COL, pids)


class MetadataStore:
    """DuckDB-backed metadata store for document-level metadata.

    Stores metadata alongside a WARP index as ``metadata.duckdb`` in the
    index directory.  Each row is keyed by ``_passage_id`` (stable integer
    assigned during index creation / add).

    Column types are inferred by PyArrow from the Python objects:
    ``list`` becomes ``LIST``, ``dict`` becomes ``STRUCT``, nested
    combinations are supported to arbitrary depth.
    """

    def __init__(self, index_path: str) -> None:
        self.index_path = index_path
        self.db_path = os.path.join(index_path, "metadata.duckdb")
        self._conn: duckdb.DuckDBPyConnection | None = None

    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        if self._conn is None:
            self._conn = duckdb.connect(self.db_path)
        return self._conn

    @property
    def exists(self) -> bool:
        return os.path.exists(self.db_path)

    def _table_exists(self) -> bool:
        result = self.conn.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'metadata'"
        ).fetchone()
        return result is not None and result[0] > 0

    def create(self, metadata: list[dict], start_pid: int = 0) -> None:
        """Create the metadata table from a list of dicts.

        Types are inferred by PyArrow directly from the Python objects.

        Args:
        ----
        metadata:
            One dict per document.  Keys become columns; values determine types.
        start_pid:
            Passage ID for the first document (subsequent docs are sequential).

        """
        if not metadata:
            return

        _input = _to_arrow(metadata, start_pid)
        self.conn.execute("DROP TABLE IF EXISTS metadata")
        self.conn.execute("CREATE TABLE metadata AS SELECT * FROM _input")

    def add(self, metadata: list[dict], start_pid: int) -> None:
        """Append metadata rows for newly added documents.

        Args:
        ----
        metadata:
            One dict per new document.
        start_pid:
            Passage ID for the first new document.

        """
        if not metadata:
            return

        if not self._table_exists():
            self.create(metadata, start_pid)
            return

        _input = _to_arrow(metadata, start_pid)
        self.conn.execute(
            "CREATE OR REPLACE TEMP TABLE _staging AS SELECT * FROM _input"
        )

        # Add any new columns from staging to the metadata table
        existing = {
            r[0]
            for r in self.conn.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'metadata'"
            ).fetchall()
        }
        for name, dtype in self.conn.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_name = '_staging'"
        ).fetchall():
            if name not in existing:
                self.conn.execute(f'ALTER TABLE metadata ADD COLUMN "{name}" {dtype}')

        # BY NAME matches columns and NULL-fills missing ones automatically
        self.conn.execute("INSERT INTO metadata BY NAME SELECT * FROM _staging")
        self.conn.execute("DROP TABLE IF EXISTS _staging")

    def delete(self, passage_ids: list[int]) -> None:
        """Remove metadata for the given passage IDs."""
        if not passage_ids or not self._table_exists():
            return
        placeholders = ", ".join(["?"] * len(passage_ids))
        self.conn.execute(
            f"DELETE FROM metadata WHERE {_PID_COL} IN ({placeholders})",
            passage_ids,
        )

    def filter(
        self,
        condition: str,
        parameters: list | tuple | None = None,
    ) -> list[int]:
        """Return passage IDs matching a SQL WHERE condition.

        Args:
        ----
        condition:
            SQL WHERE clause fragment, e.g. ``"category = ? AND age > ?"``.
            DuckDB native functions are supported (``list_contains``, struct
            dot-notation, lambdas, etc.).
        parameters:
            Values for ``?`` placeholders in *condition*.

        Returns
        -------
        List of matching passage IDs.

        """
        if not self._table_exists():
            return []

        sql = f"SELECT {_PID_COL} FROM metadata WHERE {condition}"
        params = list(parameters) if parameters is not None else []
        result = self.conn.execute(sql, params).fetchall()
        return [row[0] for row in result]

    def get(self, passage_ids: list[int] | None = None) -> list[dict]:
        """Retrieve metadata rows as dicts.

        Args:
        ----
        passage_ids:
            If provided, only return these rows.  Otherwise return all.

        """
        if not self._table_exists():
            return []

        if passage_ids is not None:
            placeholders = ", ".join(["?"] * len(passage_ids))
            result = self.conn.execute(
                f"SELECT * FROM metadata WHERE {_PID_COL} IN ({placeholders})",
                passage_ids,
            )
        else:
            result = self.conn.execute("SELECT * FROM metadata")

        columns = [desc[0] for desc in result.description]
        rows = result.fetchall()
        return [dict(zip(columns, row)) for row in rows]

    def close(self) -> None:
        """Close the DuckDB connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
