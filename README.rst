XRAYVISION  - X-RAY VIsibility Synthesis ImagiNg
================================================

|Powered By| |Build Status| |Doc Status|

XRAYVISION is an open-source Python library for Fourier or synthesis imaging of X-Rays. The most
common usage of this technique is radio interferometry however there have been a number of solar
X-ray mission which use also use this technique but obtain the visibilities in a very different
manner.

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
    from xrayvision.visibilty import RHESSIVisibilty
    rhessi_vis = RHESSIVisibilty.from_fits_file('<>')
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

.. |Build Status| image:: https://travis-ci.org/samaloney/xrayvision.svg?branch=master
    :target: https://travis-ci.org/sunpy/xrayvision
    :alt: Travis-CI build status

.. |Doc Status|  image:: https://readthedocs.org/projects/xrayvision/badge/?version=latest
    :target: http://xrayvision.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. _Code of Conduct: http://docs.sunpy.org/en/stable/coc.html
