# Licensed under a 3-clause BSD style license - see LICENSE.rst

try:
    from xrayvision.version import __version__  # type: ignore
except ImportError:
    __version__ = "unknown"
__all__: list[str] = []
