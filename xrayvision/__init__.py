# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
This is an SunPy affiliated package.
"""


# Affiliated packages may add whatever they like to this file, but
# should keep this content at the top.
# ----------------------------------------------------------------------------
from ._sunpy_init import *
# ----------------------------------------------------------------------------

#if not _ASTROPY_SETUP_:
#    # For egg_info test builds to pass, put package imports here.
#    from xrayvision import *


from pkg_resources import resource_filename

SAMPLE_RHESSI_VISIBILITIES = resource_filename('xrayvision', 'data/hsi_visibili_20131028_0156_20131028_0200_6_12.fits')