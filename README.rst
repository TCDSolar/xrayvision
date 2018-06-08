XRAYVISION  - X-RAY VIsibility Synthesis ImagiNg
================================================

|Powered By| |Build Status| |Doc Status| |Python Versions|

XRAYVISION is an open-source Python library for Fourier or synthesis imaging of X-Rays. The most
common usage of this technique is radio interferometry however there have been a number of solar
X-ray missions which also use this technique but obtain the visibilities via a different method.

Installation
------------

Requirements: Python3.6+, SunPy0.8+

As XRAYVISION is still a work in progress it has not been release to PyPI yet. The recommended way
to install XRAYVISION is via pip from git.

.. code:: bash

    pip install git+https://github.com/sunpy/xrayvision.git

Usage
-----

.. code:: python

    from astropy import units as u

    from xrayvision.visibility import RHESSIVisibility
    from xrayvision import SAMPLE_RHESSI_VISIBILITIES

    rhessi_vis = RHESSIVisibility.from_fits_file(SAMPLE_RHESSI_VISIBILITIES)
    rhessi_map = rhessi_vis.to_map(shape=(65, 65), pixel_size=[4., 4.] * u.arcsec)
    rhessi_map.peek()


Getting Help
------------



Contributing
~~~~~~~~~~~~
When you are interacting with the SunPy community you are asked to
follow our `Code of Conduct`_.

.. |Powered By| image:: http://img.shields.io/badge/powered%20by-SunPy-orange.svg?style=flat
    :target: http://www.sunpy.org
    :alt: Powered by SunPy Badge

.. |Build Status| image:: https://travis-ci.org/sunpy/xrayvision.svg?branch=master
    :target: https://travis-ci.org/sunpy/xrayvision
    :alt: Travis-CI build status

.. |Doc Status| image:: https://readthedocs.org/projects/xrayvision/badge/?version=latest
    :target: http://xrayvision.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. |Python Versions| image:: https://img.shields.io/badge/python-3.6-blue.svg
    :target: https://www.python.org/downloads/release/python-360/
    :alt: Python Versions

.. _Code of Conduct: http://docs.sunpy.org/en/stable/coc.html
