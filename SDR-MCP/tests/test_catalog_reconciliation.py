from __future__ import annotations

import os
from pathlib import Path

from rf_mcp.catalog import Catalog


def _artifact_file(root: Path, name: str, content: bytes = b"data") -> Path:
    path = root / "results" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_first_reconciliation_indexes_existing_file(tmp_path: Path) -> None:
    path = _artifact_file(tmp_path, "first.json")
    with Catalog(tmp_path) as catalog:
        assert catalog.list_artifacts() == []
        assert catalog.index_existing_artifacts() == 1
        artifact = catalog.list_artifacts()[0]
        assert artifact["path"] == str(path.resolve())
        assert artifact["size_bytes"] == 4


def test_unchanged_reconciliation_performs_no_writes(tmp_path: Path) -> None:
    _artifact_file(tmp_path, "stable.json")
    with Catalog(tmp_path) as catalog:
        assert catalog.index_existing_artifacts() == 1
        connection = catalog._connect()
        before = connection.total_changes
        statements: list[str] = []
        connection.set_trace_callback(statements.append)
        assert catalog.index_existing_artifacts() == 0
        connection.set_trace_callback(None)
        assert connection.total_changes == before
        assert not any(statement.lstrip().upper().startswith(
            ("INSERT", "UPDATE", "DELETE")) for statement in statements)


def test_changed_file_refreshes_metadata(tmp_path: Path) -> None:
    path = _artifact_file(tmp_path, "changed.json")
    with Catalog(tmp_path) as catalog:
        catalog.index_existing_artifacts()
        original = catalog.list_artifacts()[0]
        path.write_bytes(b"changed contents")
        os.utime(path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns + 1_000_000))
        assert catalog.index_existing_artifacts() == 1
        changed = catalog.list_artifacts()[0]
        assert changed["artifact_id"] == original["artifact_id"]
        assert changed["size_bytes"] == len(b"changed contents")


def test_missing_file_is_removed_from_catalog(tmp_path: Path) -> None:
    path = _artifact_file(tmp_path, "missing.json")
    with Catalog(tmp_path) as catalog:
        catalog.index_existing_artifacts()
        path.unlink()
        assert catalog.index_existing_artifacts() == 1
        assert catalog.list_artifacts() == []


def test_large_batch_is_reconciled_and_then_unchanged(tmp_path: Path) -> None:
    for number in range(500):
        _artifact_file(tmp_path, f"artifact-{number:04d}.json", str(number).encode())
    with Catalog(tmp_path) as catalog:
        assert catalog.index_existing_artifacts() == 500
        assert catalog._connect().execute(
            "SELECT COUNT(*) FROM artifacts"
        ).fetchone()[0] == 500
        assert catalog.index_existing_artifacts() == 0
