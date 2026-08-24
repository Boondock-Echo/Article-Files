"""Shared cached loaders for optional, feature-specific scientific dependencies."""
from __future__ import annotations

from functools import lru_cache, wraps
from importlib import import_module


@lru_cache(maxsize=None)
def _module(name: str):
    return import_module(name)


def _lazy_callable(module_name: str, attribute: str):
    @wraps(_lazy_callable)
    def call(*args, **kwargs):
        return getattr(_module(module_name), attribute)(*args, **kwargs)

    call.__name__ = attribute
    return call


class _LazyModule:
    def __init__(self, module_name: str) -> None:
        self._module_name = module_name

    def __getattr__(self, name: str):
        return getattr(_module(self._module_name), name)


butter = _lazy_callable("scipy.signal", "butter")
find_peaks = _lazy_callable("scipy.signal", "find_peaks")
firwin = _lazy_callable("scipy.signal", "firwin")
hilbert = _lazy_callable("scipy.signal", "hilbert")
lfilter = _lazy_callable("scipy.signal", "lfilter")
resample_poly = _lazy_callable("scipy.signal", "resample_poly")
sosfilt = _lazy_callable("scipy.signal", "sosfilt")
spectrogram = _lazy_callable("scipy.signal", "spectrogram")
wavfile = _LazyModule("scipy.io.wavfile")
