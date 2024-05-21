"""
Modules contains visibility related classes.

This contains classes to hold general visibilities and specialised classes hold visibilities from
certain spacecraft or instruments
"""

import abc
from types import SimpleNamespace
from typing import Any, Union
from collections.abc import Iterable

import astropy.units as apu
import numpy as np
import xarray
from astropy.coordinates import SkyCoord
from astropy.time import Time

__all__ = ['Visibility', 'Visibilities', 'VisibilitiesBase', 'VisMeta']


class VisMetaABC(abc.ABC):
    @property
    @abc.abstractmethod
    def energy_range(self) -> Iterable[apu.Quantity]:
        """
        Energy range over which the visibilities are computed.
        """

    @property
    @abc.abstractmethod
    def time_range(self) -> Iterable[Time]:
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


class VisibilitiesBaseABC(abc.ABC):
    @property
    @abc.abstractmethod
    def visibilities(self) -> Iterable[apu.Quantity]:
        """
        Complex numbers representing the visibilities.
        """

    @property
    @abc.abstractmethod
    def u(self) -> Iterable[apu.Quantity]:
        """
        u-coordinate on the complex plane of the visibilities.
        """

    @property
    @abc.abstractmethod
    def v(self) -> Iterable[apu.Quantity]:
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
    def uncertainty(self) -> Iterable[apu.Quantity]:
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


class VisibilitiesBase(VisibilitiesBaseABC):
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
    @apu.quantity_input()
    def __init__(self,
                 visibilities: apu.Quantity,
                 u: apu.Quantity[1/apu.arcsec],
                 v: apu.Quantity[1/apu.arcsec],
                 names: Iterable[str],
                 uncertainty: Union[apu.Quantity, None],
                 meta: Any = None,
                 dims: Iterable[str] = ("uv",),
                 coords: dict = {}):

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
        if not isinstance(visibilities, apu.Quantity) or visibilities.isscalar:
            raise TypeError('visibilities must all be a non scalar Astropy quantity.')
        nvis = len(visibilities)
        if len(u) != nvis:
            raise ValueError('u must be the same length as visibilities.')
        if len(v) != nvis:
            raise ValueError('v must be the same length as visibilities.')
        if len(names) != nvis:
            raise ValueError('names must be the same length as visibilities.')
        if not np.array(isinstance(name, str) for name in names).all():
            raise TypeError('names must all be strings.')
        if uncertainty is not None and not isinstance(uncertainty, apu.Quantity):
            raise TypeError('uncertainty must be None or same type as visibilities.')
        uv_name = "uv"
        if uv_name not in dims:
            raise ValueError(f"dims must contain '{uv_name}'.")
        data = {"data": (dims, visibilities.value)}
        if uncertainty is not None:
            data["uncertainty"] = (dims, uncertainty.to_value(visibilities.unit))
        for key, value in coords.items():
            print(key, value)
        units = dict([(key, value[1].unit) if hasattr(value[1], "unit") else (key, None)
                      for key, value in coords.items()])
        units["data"] = visibilities.unit
        units["u"] = u.unit
        units["v"] = v.unit
        coords["u"] = ([uv_name], u)
        coords["v"] = ([uv_name], v)
        coords["names"] = ([uv_name], names)
        attrs = {"units": units, "meta": meta}
        self._data = xarray.Dataset(data, coords=coords, attrs=attrs)

    @property
    def visibilities(self):
        return apu.Quantity(self._data["data"], unit=self._data.attrs["units"]["data"])

    @property
    def u(self):
        return apu.Quantity(self._data.coords["u"].values, unit=self._data.attrs["units"]["u"])

    @property
    def v(self):
        return apu.Quantity(self._data.coords["v"].values, unit=self._data.attrs["units"]["v"])

    @property
    def names(self):
        return self._data.coords["names"]

    @property
    def uncertainty(self):
        unc_name = "uncertainty"
        return apu.Quantity(self._data[unc_name], unit=self._data.attrs["units"]["data"]) if unc_name in self._data.keys() else None

    @property
    def meta(self):
        return self._data.attrs["meta"]

    ################# Everything above is required by ABC, adding extra functionality below #################

    def _build_quantity(self, label, unit_label=None):
        if unit_label is None:
            unit_label = label
        return apu.Quantity(self._data[label], unit=self._data.attrs["units"][unit_label])

    @property
    def amplitude(self):
        label = "amplitude"
        unit_label = label if label in self._data.attrs["units"].keys() else "data"
        if label in self._data.keys():
            return self._build_quantity(label, unit_label=unit_label)
        else:
            return np.sqrt(np.real(self.visibilities) ** 2 + np.imag(self.visibilities) ** 2)

    @property
    def amplitude_uncertainty(self):
        label = "amplitude_uncertainty"
        unit_label = label if label in self._data.attrs["units"].keys() else "data"
        if label in self._data.keys():
            return self._build_quantity(label, unit_label=unit_label)
        else:
            amplitude = self.amplitude
            return np.sqrt((np.real(self.visibilities) / amplitude * np.real(self.uncertainty)) ** 2
                           + (np.imag(self.visibilities) / amplitude * np.imag(self.uncertainty)) ** 2 )

    @property
    def phase(self):
        label = "phase"
        if label in self._data.keys():
            return self._build_quantity(label)
        else:
            return np.arctan2(np.imag(self.visibilities), np.real(self.visibilities)).to(apu.deg)

    @property
    def phase_uncertainty(self):
        label = "phase_uncertainty"
        unit_label = label if label in self._data.attrs["units"].keys() else "phase"
        if label in self._data.keys():
            return self._build_quantity(label, unit_label=unit_label)
        else:
            amplitude = self.amplitude
            return (np.sqrt(np.imag(self.visibilities) ** 2 / amplitude ** 4 * np.real(self.uncertainty) ** 2
                            + np.real(self.visibilities) ** 2 / amplitude ** 4 * np.imag(self.uncertainty) ** 2 ) * apu.rad).to(apu.deg)

    def __repr__(self):
        r"""
        Return a printable representation of the visibility.

        Returns
        -------
        `str`

        """
        return f"{self.__class__.__name__}< {self.u.size}, {self.visibilities}>"

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


class Visibilities(VisibilitiesBase,VisibilitiesABC):
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
    #@apu.quantity_input(u=1/apu.arcsec, v=1/apu.arcsec)
    def __init__(self, visibilities, u, v, names, uncertainty=None, meta=None):
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
        nvis = len(visibilities)
        if uncertainty.isscalar or len(uncertainty) != nvis:
            raise TypeError('uncertainty must be the same length as visibilities.')

        if not isinstance(meta, VisMetaABC):
            raise TypeError('Meta must be an instance of VisMetaABC.')

        super().__init__(visibilities, u, v, names, uncertainty=uncertainty, meta=meta)


class VisMeta(VisMetaABC, dict):
    """
    A class for holding Visibility-specific metadata.

    Parameters
    ----------
    meta: `dict`
        A dictionary of the metadata
    """
    def __init__(self, meta):
        energy_range = meta.get("energy_range", None)
        time_range = meta.get("time_range", None)
        center_range = meta.get("center", None)
        observer_coordinate = meta.get("observer_coordinate", None)
        if not isinstance(energy_range, apu.Quantity) or len(energy_range) != 2:
            raise ValueError("Input must contain the key 'energy_range' "
                             "which gives a length-2 astropy Quantity.")
        if not isinstance(time_range, Time) or len(time_range) != 2:
            raise ValueError("Input must contain the key 'time_range' "
                             "which gives a length 2 astropy time object.")
        if not isinstance(center, SkyCoord) or not center.isscalar:
            raise ValueError("Input must contain the key 'center' which gives a scalar SkyCoord.")
        if not isinstance(observer_coordinate, SkyCoord) or not observer_coordinate.isscalar:
            raise ValueError("Input must contain the key 'observer_coordinate' "
                             "which gives a scalar SkyCoord.")
        super().__init__(meta)

    @property
    def energy_range(self):
        return self["energy_range"]

    @property
    def time_range(self):
        return self["energy_range"]

    @property
    def center(self):
        return self["center"]

    @property
    def observer_coordinate(self):
        return self["observer_coordinate"]


class BaseVisibility:
    r"""
    Base visibility containing bare essential fields, u, v, and complex vis
    """

    @apu.quantity_input(u=1 / apu.arcsec, v=1 / apu.arcsec, center=apu.arcsec)
    def __int__(self, u, v, vis, center=(0, 0) * apu.arcsec):
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

    @apu.quantity_input(uv=1 / apu.arcsec, offset=apu.arcsec, center=apu.arcsec, pixel_size=apu.arcsec)
    def __init__(self, vis, *, u, v, offset=(0.0, 0.0) * apu.arcsec, center=(0.0, 0.0) * apu.arcsec):
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
