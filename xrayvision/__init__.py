# Licensed under a 3-clause BSD style license - see LICENSE.rst

try:
    from xrayvision.version import __version__ # type: ignore
except ImportError:
    __version__ = "unknown"
__all__ = []

from pkg_resources import resource_filename

SAMPLE_RHESSI_VISIBILITIES = resource_filename('xrayvision', 'data/hsi_visibili_20131028_0156_20131028_0200_6_12.fits')
