"""
Modules contains visibility related classes.

This contains classes to hold general visibilities and specialised classes hold visibilities from
certain spacecraft or instruments
"""

import abc
from typing import Any, Union
from collections.abc import Iterable

import astropy.units as apu
import numpy as np
import xarray
from astropy.coordinates import SkyCoord
from astropy.time import Time

__all__ = ["Visibility", "Visibilities", "VisMeta"]

_E_RANGE_KEY = "energy_range"
_T_RANGE_KEY = "time_range"
_OBS_COORD_KEY = "observer_coordinate"
_VIS_LABELS_KEY = "vis_labels"


class VisMetaABC(abc.ABC):
    @property
    @abc.abstractmethod
    def energy_range(self) -> Union[Iterable[apu.Quantity], None]:
        """
        Energy range over which the visibilities are computed.
        """

    @property
    @abc.abstractmethod
    def time_range(self) -> Union[Iterable[Time], None]:
        """
        Time range over which the visibilities are computed.
        """

    @property
    @abc.abstractmethod
    def observer_coordinate(self) -> Union[SkyCoord, None]:
        """
        Location of the observer.
        """

    @property
    @abc.abstractmethod
    def vis_labels(self) -> Union[Iterable[str], None]:
        """
        Labels of each visibility.
        """


class VisibilitiesABC(abc.ABC):
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
    def phase_center(self) -> apu.Quantity[apu.deg]:
        """
        The position of the phase center of the visibilities.
        """

    @property
    @abc.abstractmethod
    def meta(self) -> VisMetaABC:
        """
        Metadata.
        """

    @property
    @abc.abstractmethod
    def uncertainty(self) -> Union[Iterable[apu.Quantity], None]:
        """
        Uncertainties on visibilities values.
        """

    @property
    @abc.abstractmethod
    def amplitude(self) -> Union[Iterable[apu.Quantity], None]:
        """
        Amplitudes of the visibilities.
        """

    @property
    @abc.abstractmethod
    def amplitude_uncertainty(self) -> Union[Iterable[apu.Quantity], None]:
        """
        Amplitude uncertainty of the visibilities.
        """

    @property
    @abc.abstractmethod
    def phase(self) -> Union[Iterable[apu.Quantity[apu.deg]], None]:
        """
        Phases of the visibilities.
        """

    @property
    @abc.abstractmethod
    def phase_uncertainty(self) -> Union[Iterable[apu.Quantity[apu.deg]], None]:
        """
        Phase uncertainty of the visibilities.
        """


class Visibilities(VisibilitiesABC):
    @apu.quantity_input()
    def __init__(
        self,
        visibilities: apu.Quantity,
        u: apu.Quantity[1 / apu.arcsec],
        v: apu.Quantity[1 / apu.arcsec],
        phase_center: apu.Quantity[apu.deg],
        uncertainty: Union[apu.Quantity, None] = None,
        vis_labels: Union[Iterable[str], None] = None,
        observer_coordinate: Union[SkyCoord, None] = None,
        energy_range: Union[apu.Quantity[apu.keV], None] = None,
        time_range: Union[Time, None] = None,
        meta: Any = dict(),
        amplitude: Union[apu.Quantity, None] = None,
        amplitude_uncertainty: Union[apu.Quantity, None] = None,
        phase: Union[apu.Quantity[apu.arcsec], None] = None,
        phase_uncertainty: Union[apu.Quantity[apu.arcsec], None] = None,
    ):
        r"""
        A class for holding visibilities.

        Parameters
        ----------
        visibilities : `astropy.units.Quantity`
            Array of N complex visibilities at coordinates in `uv`.
        u : `numpy.ndarray`
            Array of `u` coordinates where visibilities will be evaluated.
        v : `numpy.ndarray`
            Array of `v` coordinates where visibilities will be evaluated.
        phase_center : `astropy.units.Quantity` with angular unit.
            The location of the phase center of the visibilities.
        uncertainty: `astropy.units.Quantity`, optional
            The uncertainty of the visibilities.
            Must be same shape and unit as visibilities.
        vis_labels : iterable of `str`, optional
            The label of each visibility, e.g. the name of the detector that
            measured it. Must be same length as visibilities.
        observer_coordinate : `astropy.coordinates.SkyCoord`, optional
            The position of the observer that measured the visibilities.
        energy_range : `astropy.units.Quantity` with a spectral unit, optional
            The energy range over which the visibilities were calculated.
        time_range : `astropy.time.Time`, optional
            The time range over which the visibilities were calculated.
        meta : `VisMetaABC` or dict-like, optional
            Metadata associated with the visibilities.
        amplitude : `astropy.units.Quantity`, optional
            The amplitude of the visibilities.  If not given, amplitudes
            be calculated directly from the visibilities.
            Must be same shape and unit as visibilities.
        amplitude_uncertainty : `astropy.units.Quantity`, optional
            The uncertainty of the visibility amplitudes. If not provided,
            amplitude uncertainties will be calculated from visibilities,
            visibility uncertainties, and amplitudes.
            Must be same shape and unit as visibilities.
        phase : `astropy.units.Quantity` with an angle unit, optional
            The phase of the visibilities.  If not given, phases will
            be calculated directly from the visibilities.
            Must be same shape as visibilities.
        phase_uncertainty : `astropy.units.Quantity` with an angle unit, optional
            The uncertainty of the visibility phases. If not provided,
            phase uncertainties will be calculated from visibilities,
            visibility uncertainties, and amplitudes.
            Must be same shape and unit as visibilities.
        """
        # Saitize inputs.
        if not isinstance(visibilities, apu.Quantity) or visibilities.isscalar:
            raise TypeError("visibilities must all be a non scalar Astropy quantity.")
        nvis = visibilities.shape[-1]
        if len(u) != nvis:
            raise ValueError("u must be the same length as visibilities.")
        if len(v) != nvis:
            raise ValueError("v must be the same length as visibilities.")
        if len(vis_labels) != nvis:
            raise ValueError("names must be the same length as visibilities.")
        if uncertainty is not None and uncertainty.shape != visibilities.shape:
            raise TypeError("uncertainty must be same shape as visibilities.")
        if amplitude is not None and amplitude.shape != visibilities.shape:
            raise TypeError("amplitude must be same shape as visibilities.")
        if amplitude_uncertainty is not None and amplitude_uncertainty.shape != visibilities.shape:
            raise TypeError("amplitude_uncertainty must be same shape as visibilities.")
        if phase is not None and phase.shape != visibilities.shape:
            raise TypeError("phase must be same shape as visibilities.")
        if phase_uncertainty is not None and phase_uncertainty.shape != visibilities.shape:
            raise TypeError("phase_uncertainty must be same shape as visibilities.")

        # Define names used internally for different pieces of data.
        self._vis_key = "data"
        self._uncert_key = "uncertainty"
        self._amplitude_key = "amplitude"
        self._amplitude_uncert_key = "amplitude_uncertainty"
        self._phase_key = "phase"
        self._phase_uncert_key = "phase_uncertainty"
        self._u_key = "u"
        self._v_key = "v"
        self._vis_labels_key = _VIS_LABELS_KEY
        self._phase_center_key = "phase_center"
        self._obs_coord_key = _OBS_COORD_KEY
        self._e_range_key = _E_RANGE_KEY
        self.t_range_key = _T_RANGE_KEY
        self._meta_key = "meta"
        self._uv_key = "uv"
        self._units_key = "units"

        # Build meta.
        if not isinstance(meta, VisMetaABC):
            meta = VisMeta(meta)
        meta[self._phase_center_key] = phase_center
        if observer_coordinate is not None:
            meta[self._obs_coord_key] = observer_coordinate
        if energy_range is not None:
            meta[self._e_range_key] = energy_range
        if time_range is not None:
            meta[self._t_range_key] = time_range

        # Construct underlying data object.
        # In case visibilities is multi-dimensional, assume last axis is the uv-axis.
        # and give other axes arbitrary names.
        dims = [f"dim{i}" for i in range(0, len(visibilities.shape) - 1)] + [self._uv_key]
        data = {self._vis_key: (dims, visibilities.value)}
        coords = {self._u_key: ([self._uv_key], u.value), self._v_key: ([self._uv_key], v.to_value(u.unit))}
        units = {self._vis_key: visibilities.unit, self._uv_key: u.unit}
        if uncertainty is not None:
            data[self._uncert_key] = (dims, uncertainty.to_value(visibilities.unit))
        if amplitude is not None:
            data[self._amplitude_key] = (dims, amplitude.to_value(visibilities.unit))
        if amplitude_uncertainty is not None:
            data[self._amplitude_uncert_key] = (dims, amplitude_uncertainty.to_value(visibilities.unit))
        if phase is not None:
            data[self._phase_key] = (dims, phase.value.to_value(visibilities.unit))
            units[self._phase_key] = phase.unit
        if phase_uncertainty is not None:
            data[self._phase_uncert_key] = (dims, phase_uncertainty.to_value(phase.unit))
        if vis_labels is not None:
            coords[self._vis_labels_key] = ([self._uv_key], vis_labels)
        attrs = {self._units_key: units, self._meta_key: meta}
        self._data = xarray.Dataset(data, coords=coords, attrs=attrs)

    @property
    def visibilities(self):
        return self._build_quantity(self._vis_key)

    @property
    def u(self):
        return self._build_quantity(self._u_key, self._uv_key)

    @property
    def v(self):
        return self._build_quantity(self._v_key, self._uv_key)

    @property
    def phase_center(self):
        return self._data.meta[self._phase_center_key]

    @phase_center.setter
    def phase_center(self, value: apu.Quantity[apu.deg]):
        self._data.attrs[self._meta_key][self._phase_center_key] = value

    @property
    def uncertainty(self):
        return self._build_quantity(self._uncert_key, self._vis_key) if self._uncert_key in self._data.keys() else None

    @property
    def meta(self):
        meta = self._data.attrs[self._meta_key]
        meta[self._vis_labels_key] = self._data.coords[self._vis_labels_key][1]
        return meta

    @property
    def amplitude(self):
        if self._amplitude_key in self._data:
            return self._build_quantity(self._amplitude_key, self._vis_key)
        else:
            return np.sqrt(np.real(self.visibilities) ** 2 + np.imag(self.visibilities) ** 2)

    @property
    def amplitude_uncertainty(self):
        if self._amplitude_uncert_key in self._data:
            return self._build_quantity(self._amplitude_uncert_key, self._vis_key)
        else:
            vis = self.visibilities
            uncert = self.uncertainty
            amplitude = self.amplitude
            return np.sqrt(
                (np.real(vis) / amplitude * np.real(uncert)) ** 2 + (np.imag(vis) / amplitude * np.imag(uncert)) ** 2
            )

    @property
    def phase(self):
        if self._phase_key in self._data:
            return self._build_quantity(self._phase_key)
        else:
            vis = self.visibilities
            return np.arctan2(np.imag(vis), np.real(vis)).to(apu.deg)

    @property
    def phase_uncertainty(self):
        if self._phase_uncert_key in self._data:
            return self._build_quantity(self._phase_uncert_key, self._phase_key)
        else:
            vis = self.visibilities
            uncert = self.uncertainty
            amplitude = self.amplitude
            return (
                np.sqrt(
                    np.imag(vis) ** 2 / amplitude**4 * np.real(uncert) ** 2
                    + np.real(vis) ** 2 / amplitude**4 * np.imag(uncert) ** 2
                )
                * apu.rad
            ).to(apu.deg)

    def _build_quantity(self, label, unit_label=None):
        if unit_label is None:
            unit_label = label
        return apu.Quantity(self._data[label], unit=self._data.attrs[self._units_key][unit_label])

    def __repr__(self):
        r"""
        Return a printable representation of the visibility.

        Returns
        -------
        `str`

        """
        return f"{self.__class__.__name__}< {self.u.size}, {self.visibilities}>"


class VisMeta(VisMetaABC, dict):
    """
    A class for holding Visibility-specific metadata.

    Parameters
    ----------
    meta: `dict`
        A dictionary of the metadata
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._e_range_key = _E_RANGE_KEY
        self._t_range_key = _T_RANGE_KEY
        self._obs_coord_key = _OBS_COORD_KEY
        self._vis_labels_key = _VIS_LABELS_KEY

    @property
    def energy_range(self):
        return self.get(self._e_range_key, None)

    @property
    def time_range(self):
        return self.get(self._t_range_key, None)

    @property
    def observer_coordinate(self):
        return self.get(self._obs_coord_key, None)

    @property
    def vis_labels(self):
        return self.get(self._vis_labels_key, None)


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
