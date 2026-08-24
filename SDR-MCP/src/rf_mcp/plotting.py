"""Cached, headless Matplotlib loading for optional plotting code."""
from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def pyplot():
    """Load pyplot once, after forcing the non-interactive backend."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt

class _LazyPyplot:
    """Compatibility proxy for modules with many independent plot functions."""

    def __getattr__(self, name: str):
        return getattr(pyplot(), name)


lazy_pyplot = _LazyPyplot()
