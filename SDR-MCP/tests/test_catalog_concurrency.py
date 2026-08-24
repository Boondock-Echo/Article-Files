from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from rf_mcp.catalog import Catalog


@pytest.fixture
def catalog(tmp_path):
    value = Catalog(tmp_path, index_existing=False)
    yield value
    value.close()


def _job(catalog: Catalog, number: int) -> None:
    catalog.upsert_job(f"job-{number}", "test", "complete", summary={"number": number})


def test_concurrent_readers_and_writer(catalog: Catalog) -> None:
    for number in range(10):
        _job(catalog, number)

    def read_repeatedly() -> None:
        for _ in range(50):
            assert catalog.list_jobs(limit=100)
            assert catalog.get_job("job-0")["job_id"] == "job-0"

    def write_repeatedly() -> None:
        for number in range(10, 60):
            _job(catalog, number)

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(read_repeatedly) for _ in range(4)]
        futures.append(executor.submit(write_repeatedly))
        for future in futures:
            future.result()

    assert len(catalog.list_jobs(limit=100)) == 60
    assert len(catalog._connections) >= 2


def test_connection_context_rolls_back(catalog: Catalog) -> None:
    with pytest.raises(RuntimeError, match="abort"):
        with catalog._connect() as connection:
            connection.execute(
                "INSERT INTO jobs (job_id, job_type, state, created_at) VALUES (?, ?, ?, ?)",
                ("rolled-back", "test", "queued", "2026-01-01T00:00:00+00:00"),
            )
            raise RuntimeError("abort")

    with pytest.raises(ValueError, match="Unknown persisted RF job_id"):
        catalog.get_job("rolled-back")


def test_close_closes_worker_connections(catalog: Catalog) -> None:
    with ThreadPoolExecutor(max_workers=2) as executor:
        connections = list(executor.map(lambda _: catalog._connect(), range(2)))

    catalog.close()

    assert not catalog._connections
    with pytest.raises(RuntimeError, match="closed"):
        catalog.list_jobs()
    for connection in connections:
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            connection.execute("SELECT 1")
