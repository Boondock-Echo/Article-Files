"""Micro-benchmark for the catalog's common job operations.

Run with ``python benchmarks/catalog_benchmark.py --iterations 1000``.
"""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

from rf_mcp.catalog import Catalog


def run(iterations: int) -> dict[str, float]:
    with tempfile.TemporaryDirectory() as temporary, Catalog(
        Path(temporary), index_existing=False
    ) as catalog:
        catalog.upsert_job("bench", "benchmark", "running")
        timings = {}
        for name, operation in (
            ("list_jobs", lambda: catalog.list_jobs(limit=10)),
            ("get_job", lambda: catalog.get_job("bench")),
            (
                "upsert_job",
                lambda: catalog.upsert_job("bench", "benchmark", "running"),
            ),
        ):
            started = time.perf_counter()
            for _ in range(iterations):
                operation()
            timings[name] = time.perf_counter() - started
        return timings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument(
        "--max-seconds", type=float, default=10.0,
        help="fail if the complete benchmark exceeds this acceptance threshold",
    )
    args = parser.parse_args()
    timings = run(args.iterations)
    total = sum(timings.values())
    for name, elapsed in timings.items():
        print(f"{name:12} {elapsed:8.4f}s ({args.iterations / elapsed:,.0f} ops/s)")
    print(f"{'total':12} {total:8.4f}s")
    if total > args.max_seconds:
        raise SystemExit(
            f"catalog benchmark took {total:.3f}s; limit is {args.max_seconds:.3f}s"
        )


if __name__ == "__main__":
    main()
