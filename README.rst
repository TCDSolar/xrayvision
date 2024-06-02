XRAYVISION - X-RAY VIsibility Synthesis ImagiNg
===============================================

|Powered By| |Build Status| |Doc Status| |Python Versions|

.. |Powered By| image:: http://img.shields.io/badge/powered%20by-SunPy-orange.svg?style=flat
    :target: https://www.sunpy.org
    :alt: Powered by SunPy Badge

.. |Build Status| image:: https://github.com/TCDSolar/xrayvision/actions/workflows/ci.yml/badge.svg
    :target: https://github.com/TCDSolar/xrayvision/actions/workflows/ci.yaml
    :alt: Build Status

.. |Doc Status| image:: https://readthedocs.org/projects/xrayvision/badge/?version=stable
    :target: https://xrayvision.readthedocs.io/en/latest/?badge=stable
    :alt: Documentation Status

.. |Python Versions| image:: https://img.shields.io/pypi/pyversions/xrayvisim
    :target: https://pypi.python.org/pypi/xrayvisim/
    :alt: Python Versions

XRAYVISION is an open-source Python library for Fourier or synthesis imaging of X-Rays. The most
common usage of this technique is radio interferometry however there have been a number of solar
X-ray missions which also use this technique but obtain the visibilities via a different method.

Installation
------------

It is strongly advised that you use and isolated environment through python's venv, virturalenv, anaconda or similar.

.. note::

The name 'xrayvision' was already taken on PyPi so the package name is 'xrayvisim' e.g.

.. code-block::

    pip install xrayvisim

Usage
-----

See the `example gallery`_.

Getting Help
------------

See the `issue tracker`_.

Contributing
------------
When you are interacting with the community you are asked to
follow the `Code of Conduct`_.

.. _Code of Conduct: http://docs.sunpy.org/en/stable/coc.html
.. _example gallery: https://xrayvision.readthedocs.io/en/latest/generated/gallery/index.html
.. _issue tracker: https://github.com/TCDSolar/xrayvision/issues
