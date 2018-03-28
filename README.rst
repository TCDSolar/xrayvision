XRAYVISION  - X-RAY VIsibility Synthesis ImagiNg
================================================

|Powered By| |Build Status|

XRAYVISION is an open-source Python library for Fourier based imaging.

Installation
------------

Requirements: >Python3.6, >SunPy0.8

The recommended way to install XRAYVISION is via pip

.. code:: bash
    pip install git

Usage
-----

.. code:: python
    from astropy import units as u
    from xrayvision.visibilty import RHESSIVisibilty
    rhessi_vis = RHESSIVisibilty.from_fits_file('<>')
    rhessi_map = rhessi_vis.to_map(shape=(65, 65), pixel_size=[4., 4.]* u.arcsec)
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

.. _Code of Conduct: http://docs.sunpy.org/en/stable/coc.html
