"""Static benchmark for realistic 100 ms live-IQ demodulator chunks.

Run with ``PYTHONPATH=src python benchmarks/live_audio_demodulator.py``.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from rf_mcp.live_audio import LiveAudioConfig, StreamingDemodulator


MODES = (("broadcast_fm", 200_000), ("nfm", 12_500), ("am", 10_000),
         ("usb", 3_000), ("lsb", 3_000), ("cw", 500))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--input-rate", type=int, default=768_000)
    args = parser.parse_args()
    chunk_samples = args.input_rate // 10
    phase = 2*np.pi*1_000*np.arange(chunk_samples)/args.input_rate
    iq = np.exp(1j*phase).astype(np.complex64)
    for mode, bandwidth in MODES:
        settings = LiveAudioConfig(100_000_000, mode, bandwidth)
        demodulator = StreamingDemodulator(settings, args.input_rate)
        started = time.perf_counter()
        for _ in range(args.iterations): demodulator.process(iq.copy())
        elapsed = time.perf_counter() - started
        realtime = args.iterations*.1/elapsed
        print(f"{mode:12} {elapsed/args.iterations*1000:8.2f} ms/chunk  {realtime:6.1f}x realtime")


if __name__ == "__main__":
    main()
