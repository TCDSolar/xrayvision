"""
Modules contains visibility related classes.

This contains classes to hold general visibilities and specialised classes hold visibilities from
certain spacecraft or instruments
"""
from datetime import datetime

import astropy.units as apu
import numpy as np


__all__ = ['Visibility']


class BaseVisibility:
    r"""
    Base visibility containing bare essential fields, u, v, and complex vis
    """
    @apu.quantity_input(u=1/apu.arcsec, v=1/apu.arcsec, center=apu.arcsec)
    def __int__(self, u, v, vis, center=(0, 0)*apu.arcsec):
        self.u = u
        self.v = v
        self.vis = vis
        self.center = center


class Visibility:
    r"""
    Hold a set of related visibilities and information.

    Attributes
    ----------
    vis : `numpy.ndarray`
        Array of N complex visibilities at coordinates in `uv`
    u : `numpy.ndarray`
        Array of `u` coordinates where visibilities will be evaluated
    v : `numpy.ndarray`
        Array of `v` coordinates where visibilities will be evaluated
    center : `float` (x, y), optional
        The x, y offset of phase center

    """
    @apu.quantity_input(uv=1/apu.arcsec, center=apu.arcsec, pixel_size=apu.arcsec)
    def __init__(self, vis, *, u, v, center=(0., 0.) * apu.arcsec):
        r"""
        Initialise a new Visibility object.

        Parameters
        ----------
        vis : `numpy.ndarray`
            Array of N complex visibilities at coordinates in `uv`
        u : `numpy.ndarray`
            Array of `u` coordinates where visibilities will be evaluated
        v : `numpy.ndarray`
            Array of `v` coordinates where visibilities will be evaluated

        """
        self.u = u
        self.v = v
        self.vis = vis
        self.center = center

    def __repr__(self):
        r"""
        Return a printable representation of the visibility.

        Returns
        -------
        `str`

        """
        return f"{self.__class__.__name__}< {self.u.size}, {self.vis}>"

    def __eq__(self, other):
        r"""
        Equality for Visibility class

        Parameters
        ----------
        other : `Visibility`
            The other visibility to compare

        Returns
        -------
        `boolean`

        """
        props_equal = []
        for key in self.__dict__.keys():
                props_equal.append(np.array_equal(self.__dict__[key], other.__dict__[key]))

        if all(props_equal):
            return True
        else:
            return False
