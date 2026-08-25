"""Interruptible reads for receiver subprocess streams."""
from __future__ import annotations

import os
import select
import threading
import time
from typing import Iterator, BinaryIO


def read_chunks(stream: BinaryIO, chunk_bytes: int, stop_event: threading.Event,
                deadline: float, *, poll_seconds: float = 0.05) -> Iterator[bytes]:
    """Accumulate short reads, while checking stop independently of chunk size.

    A final aligned partial chunk is the caller's responsibility. Stop deliberately
    discards a partial chunk; EOF and the time deadline return it to the caller.
    """
    pending = bytearray()
    try:
        fd = stream.fileno()
    except (AttributeError, OSError):
        # In-memory streams used by adapters/tests have no selectable descriptor.
        # Real subprocess pipes always take the interruptible path below.
        while not stop_event.is_set():
            block = stream.read(chunk_bytes - len(pending))
            if not block:
                if pending:
                    yield bytes(pending)
                return
            pending.extend(block)
            if len(pending) == chunk_bytes:
                yield bytes(pending)
                pending.clear()
        return
    while True:
        if stop_event.is_set():
            return
        remaining_time = deadline - time.monotonic()
        if remaining_time <= 0:
            if pending:
                yield bytes(pending)
            return
        readable, _, _ = select.select([fd], [], [], min(poll_seconds, remaining_time))
        if not readable:
            continue
        # A bounded read avoids waiting for the rest of a requested chunk. os.read
        # returns immediately with whatever the pipe currently has available.
        block = os.read(fd, min(chunk_bytes - len(pending), 65_536))
        if not block:  # EOF, rather than an operator-requested stop.
            if pending:
                yield bytes(pending)
            return
        pending.extend(block)
        if len(pending) == chunk_bytes:
            yield bytes(pending)
            pending.clear()
