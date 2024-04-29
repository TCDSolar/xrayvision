"""
Modules contains visibility related classes.

This contains classes to hold general visibilities and specialised classes hold visibilities from
certain spacecraft or instruments
"""

import abc
import astropy.units as apu
import numpy as np
import astropy.coordinates
import astropy.units as u


__all__ = ['Visibility']

class VisibilityABC:
    @property
    @abc.abstractmethod
    def amplitude(self) -> np.ndarray:
        """
        Amplitudes of the visibilities.
        """
    
    @property
    @abc.abstractmethod   
    def phase(self) -> np.ndarray:
        """
        Phases of the visibilities.
        """
    
    @property
    @abc.abstractmethod   
    def center(self) -> astropy.coordinates.SkyCoord:
        """
        Center of the image described by the visibilities.
        """

    @property
    @abc.abstractmethod   
    def observer_coordinate(self) -> astropy.coordinates.SkyCoord:
        """
        Location of the observer.
        """

    @property
    @abc.abstractmethod   
    def energy_range(self) -> Iterable[u.Quantity]:
        """
        Energy range over which the visibilities are computed.
        """

    @property
    @abc.abstractmethod   
    def date(self) -> astropy.time.Time:
        """
        Centre time over which the visibilities are computed.
        """

    @property
    @abc.abstractmethod   
    def integration_time(self) -> u.Quantity:
        """
        Time duration over which the visibilities are computed.
        """

    @property
    @abc.abstractmethod   
    def u(self) -> np.ndarray:
        """
        u-coordinate on the complex plane of the visibilities.
        """

    @property
    @abc.abstractmethod   
    def v(self) -> np.ndarray:
        """
        v-coordinate on the complex plane of the visibilities.
        """

    @property
    @abc.abstractmethod   
    def vis(self) -> np.ndarray:
        """
        Complex numbers representing the visibilities.
        """

    @property
    @abc.abstractmethod   
    def keys(self) -> Iterable[str]:
        """
        Names for each visibility. 

        Must be same length as self.vis.
        """

    @property
    @abc.abstractmethod   
    def uncertaintly(self) -> np.ndarray:
        """
        uncertainties on visibilities values. 
        """


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
    @apu.quantity_input(uv=1/apu.arcsec, offset=apu.arcsec, center=apu.arcsec, pixel_size=apu.arcsec)
    def __init__(self, vis, *, u, v, offset=(0., 0.) * apu.arcsec, center=(0., 0.) * apu.arcsec):
        r"""
        Initialise a new Visibility object.

        Parameters
        ----------
        vis : `numpy.ndarray`
            Array of N complex visibilities at coordinates in `uv`.
        u : `numpy.ndarray`
            Array of `u` coordinates where visibilities will be evaluated.
        v : `numpy.ndarray`
            Array of `v` coordinates where visibilities will be evaluated.
        center :
            Phase centre
        """
        self.u = u
        self.v = v
        self.vis = vis
        self.center = center
        self.offset = offset

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
