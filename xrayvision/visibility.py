"""
Modules contains visibility related classes.

This contains classes to hold general visibilities and specialised classes hold visibilities from
certain spacecraft or instruments
"""

import abc
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from typing import Any


__all__ = ['Visibilities']

class VisibilitiesBaseABC:
    @property
    @abc.abstractmethod   
    def visibilities(self) -> Iterable[u.Quantity]:
        """
        Complex numbers representing the visibilities.
        """

    @property
    @abc.abstractmethod   
    def u(self) -> Iterable[u.Quantity]:
        """
        u-coordinate on the complex plane of the visibilities.
        """

    @property
    @abc.abstractmethod   
    def v(self) -> Iterable[u.Quantity]:
        """
        v-coordinate on the complex plane of the visibilities.
        """

    @property
    @abc.abstractmethod   
    def names(self) -> Iterable[str]:
        """
        Names for each visibility. 

        Must be same length as self.vis.
        """

    @property
    @abc.abstractmethod   
    def uncertainty(self) -> Any:
        """
        Uncertainties on visibilities values.
        """

    @property
    @abc.abstractmethod   
    def meta(self) -> Any:
        """
        Meta data.
        """

class VisibilitiesABC(VisibilitiesBaseABC):
    @property
    @abc.abstractmethod   
    def uncertainty(self) -> Iterable[u.Quantity]:
        """
        Uncertainties on visibilities values.
        """

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
    def amplitude_uncertainty(self) -> np.ndarray:
        """
        Amplitude uncertainty of the visibilities.
        """
    
    @property
    @abc.abstractmethod   
    def phase_uncertainty(self) -> np.ndarray:
        """
        Phase uncertainty of the visibilities.
        """

    @property
    @abc.abstractmethod   
    def meta(self) -> VisMetaABC:
        """
        Meta data.
        """

class VisMetaABC: 
    @property
    @abc.abstractmethod   
    def energy_range(self) -> Iterable[u.Quantity]:
        """
        Energy range over which the visibilities are computed.
        """

    @property
    @abc.abstractmethod   
    def date_range(self) -> Iterable[astropy.time.Time]:
        """
        Centre time over which the visibilities are computed.
        """ 
         
    @property
    @abc.abstractmethod   
    def center(self) -> SkyCoord:
        """
        Center of the image described by the visibilities.
        """

    @property
    @abc.abstractmethod   
    def observer_coordinate(self) -> SkyCoord:
        """
        Location of the observer.
        """


class Visibilities:
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
    @apu.quantity_input(u=1/u.arcsec, v=1/u.arcsec, energy_range=u.keV, center=apu.arcsec)
    def __init__(self, visibilities, *, u, v, names, energy_range, date_range=None, center=None, observer_coordinate=None, uncertainty=None):
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
        self._visibilities = visibilities
        nvis = len(self._visibilities)
        if len(u) != nvis: 
            raise ValueError('u must be the same length as visibilities.')
        self._u = u
        if len(v) != nvis: 
            raise ValueError('v must be the same length as visibilities.')
        self._v = v
        if len(names) != nvis: 
            raise ValueError('names must be the same length as visibilities.')
        if not (isinstance(name, str) for name in names).all(): 
            raise TypeError('names must all be strings.')
        self._names = names
        if len(energy_range) != 2: 
            raise ValueError('energy range must be length 2.')
        self._energy_range = energy_range
        if center is not None and (not isinstance(center, SkyCoord) or not name.isscalar): 
            raise ValueError('center must be a scalar SkyCoord.')           
        self._center = center
        if date_range is not None and (not isinstance(date_range, astropy.time.Time) or len(date_range) != 2): 
            raise ValueError('date_range must be a length 2 astropy time object.')   
        self._date_range = date_range
        if observer_coordinate is not None and (not isinstance(observer_coordinate, SkyCoord) or not observer_coordinate.isscalar): 
            raise ValueError('observer_coordinate must be a scalar SkyCoord.')  
        self._observer_coordinate = observer_coordinate
        if uncertainty is not None and (len(uncertainty) != nvis): 
            raise ValueError('uncertainty must be the same length as visibilities.')
        self._uncertainty = uncertainty

    @property
    def visibilities(self): 
        return self._visibilities
    
    @property
    def u(self): 
        return self._u
    
    @property
    def v(self): 
        return self._v
    
    @property
    def names(self): 
        return self._names
     
    @property
    def energy_range(self): 
        return self._energy_range 
       
    @property
    def center(self): 
        return self._center 
             
    @property
    def date_range(self): 
        return self._date_range 
             
    @property
    def observer_coordinate(self): 
        return self._observer_coordinate  
             
    @property
    def uncertainty(self): 
        return self._uncertainty      



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
