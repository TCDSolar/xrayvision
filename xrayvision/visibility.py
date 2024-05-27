"""
Modules contains visibility related classes.

This contains classes to hold general visibilities and specialised classes hold visibilities from
certain spacecraft or instruments
"""

import abc
from typing import Union, Optional
from collections.abc import Iterable

import astropy.units as apu
import numpy as np
import xarray
from astropy.coordinates import SkyCoord
from astropy.time import Time

__all__ = ["Visibility", "Visibilities", "VisMeta", "VisibilitiesABC", "VisMetaABC"]

_E_RANGE_KEY = "spectral_range"
_T_RANGE_KEY = "time_range"
_OBS_COORD_KEY = "observer_coordinate"
_VIS_LABELS_KEY = "vis_labels"
_INSTR_KEYS = ["instrument", "INSTRUME"]


class VisMetaABC(abc.ABC):
    @property
    @abc.abstractmethod
    def observer_coordinate(self) -> Union[SkyCoord, None]:
        """
        Location of the observer.
        """

    @property
    @abc.abstractmethod
    def spectral_range(self) -> Union[Iterable[apu.Quantity], None]:
        """
        Spectral range over which the visibilities are computed.
        """

    @property
    @abc.abstractmethod
    def time_range(self) -> Union[Iterable[Time], None]:
        """
        Time range over which the visibilities are computed.
        """

    @property
    @abc.abstractmethod
    def vis_labels(self) -> Union[Iterable[str], None]:
        """
        Labels of each visibility.
        """

    @property
    @abc.abstractmethod
    def instrument(self) -> Union[str, None]:
        """
        The name of the instrument or observer that measured the visibilities.
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
        u: apu.Quantity[1 / apu.deg],
        v: apu.Quantity[1 / apu.deg],
        phase_center: apu.Quantity[apu.arcsec] = [0, 0] * apu.arcsec,
        meta: Optional[VisMetaABC] = None,
        uncertainty: Optional[apu.Quantity] = None,
        amplitude: Optional[apu.Quantity] = None,
        amplitude_uncertainty: Optional[apu.Quantity] = None,
        phase: Optional[apu.Quantity[apu.deg]] = None,
        phase_uncertainty: Optional[apu.Quantity[apu.deg]] = None,
    ):
        r"""
        A class for holding visibilities.

        Parameters
        ----------
        visibilities :
            Array of N complex visibilities at coordinates in `uv`.
        u :
            Array of `u` coordinates where visibilities will be evaluated.
        v :
            Array of `v` coordinates where visibilities will be evaluated.
        phase_center : `astropy.units.Quantity` with angular unit.
            The location of the phase center of the visibilities.
            Default = [0, 0] arcsec
        meta :
            Metadata associated with the visibilities.
            In order to use this Visibilities object to make a Map, ``meta``
            must contain a key ``'observer_coordinate'`` which gives a
            `~astropy.coordinates.SkyCoord`, designating the location from which
            the visibilities were measured.
            To give each visibility a label, include a key, ``'vis_labels'``,
            giving an iterable of the same length as the number of visibilities.
        uncertainty:
            The uncertainty of the visibilities.
            Must be same shape and unit as visibilities.
        amplitude :
            The amplitude of the visibilities.  If not given, amplitudes
            be calculated directly from the visibilities.
            Must be same shape and unit as visibilities.
        amplitude_uncertainty :
            The uncertainty of the visibility amplitudes. If not provided,
            amplitude uncertainties will be calculated from visibilities,
            visibility uncertainties, and amplitudes.
            Must be same shape and unit as visibilities.
        phase :
            The phase of the visibilities.  If not given, phases will
            be calculated directly from the visibilities.
            Must be same shape as visibilities.
        phase_uncertainty :
            The uncertainty of the visibility phases. If not provided,
            phase uncertainties will be calculated from visibilities,
            visibility uncertainties, and amplitudes.
            Must be same shape and unit as visibilities.
        """
        # Sanitize inputs.
        if not isinstance(visibilities, apu.Quantity) or visibilities.isscalar:
            raise TypeError("visibilities must all be a non scalar Astropy quantity.")
        nvis = visibilities.shape[-1]
        if len(u) != nvis:
            raise ValueError("u must be the same length as visibilities.")
        if len(v) != nvis:
            raise ValueError("v must be the same length as visibilities.")
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
        self._meta_key = "meta"
        self._uv_key = "uv"
        self._units_key = "units"

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
        if meta is None:
            meta = VisMeta(dict())
        vis_labels = meta.vis_labels
        if vis_labels is not None:
            if len(vis_labels) != nvis:
                raise ValueError(
                    "meta.vis_labels must be same length as number of visibilites. "
                    f"Number of labels = {len(meta.vis_labels)}; "
                    f"Number of visibilities = {nvis}"
                )
            coords[_VIS_LABELS_KEY] = ([self._uv_key], vis_labels)
        attrs = {self._units_key: units, self._meta_key: meta}
        self._data = xarray.Dataset(data, coords=coords, attrs=attrs)

        # Attach phase_center
        self._phase_center = phase_center

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
        return self._phase_center

    @phase_center.setter
    def phase_center(self, value: apu.Quantity[apu.deg]):
        self._phase_center = value

    @property
    def uncertainty(self):
        return self._build_quantity(self._uncert_key, self._vis_key) if self._uncert_key in self._data.keys() else None

    @property
    def meta(self):
        meta, coords = self._data.attrs[self._meta_key], self._data.coords
        if _VIS_LABELS_KEY in coords:
            meta[_VIS_LABELS_KEY] = coords[_VIS_LABELS_KEY].values
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
            vis, uncert, amplitude = self.visibilities, self.uncertainty, self.amplitude
            if uncert is None:
                return None
            else:
                return np.sqrt(
                    (np.real(vis) / amplitude * np.real(uncert)) ** 2
                    + (np.imag(vis) / amplitude * np.imag(uncert)) ** 2
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
            vis, uncert, amplitude = self.visibilities, self.uncertainty, self.amplitude
            if uncert is None:
                return None
            else:
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

    def __eq__(self, other):
        """
        Checks whether two Visibilities objects are equal.

        Does not check whether their metas are equal.
        """
        if not apu.quantity.allclose(self.visibilities, other.visibilities):
            return False
        if not apu.quantity.allclose(self.u, other.u):
            return False
        if not apu.quantity.allclose(self.v, other.v):
            return False
        if not apu.quantity.allclose(self.phase_center, other.phase_center):
            return False
        if not apu.quantity.allclose(self.amplitude, other.amplitude):
            return False
        if not apu.quantity.allclose(self.phase, other.phase):
            return False

        uncerts = (
            (self.uncertainty, other.uncertainty),
            (self.amplitude_uncertainty, other.amplitude_uncertainty),
            (self.phase_uncertainty, other.phase_uncertainty),
        )
        for self_uncert, other_uncert in uncerts:
            if not _attrs_both_none_or_neither(self_uncert, other_uncert) or (
                self_uncert is not None and not apu.quantity.allclose(self_uncert, other_uncert)
            ):
                return False

        return True


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
        # Check controlled/expected inputs are of correct type and units.
        controled_args = (
            (_OBS_COORD_KEY, SkyCoord),
            (_E_RANGE_KEY, apu.Quantity, apu.keV, apu.spectral()),
            (_T_RANGE_KEY, Time),
        )
        for args in controled_args:
            self._check_input_type_and_unit(*args)

    def _check_input_type_and_unit(self, key, key_type, unit=None, equivalencies=None):
        value = self.get(key, None)
        if not isinstance(value, (key_type, type(None))):
            raise KeyError(f"Inputs must include a key, '{key}', that gives a {key_type}.")
        if unit is not None and value is not None and not value.unit.is_equivalent(unit, equivalencies=equivalencies):
            raise ValueError(f"'{key}' must have angular units.")

    @property
    def observer_coordinate(self):
        return self.get(_OBS_COORD_KEY, None)

    @property
    def spectral_range(self):
        return self.get(_E_RANGE_KEY, None)

    @property
    def time_range(self):
        return self.get(_T_RANGE_KEY, None)

    @property
    def vis_labels(self):
        return self.get(_VIS_LABELS_KEY, None)

    @property
    def instrument(self):
        instr = None
        i, n = 0, len(_INSTR_KEYS)
        while not instr and i < n:
            instr = self.get(_INSTR_KEYS[i], None)
            i += 1
        return instr

    @property
    def __eq__(self, other):
        if not _attrs_both_none_or_neither(self.observer_coordinate, other.observer_coordinate):
            return False
        if hasattr(self.observer_coordinate, "__len__"):
            if not hasattr(other.observer_coordinate, "__len__"):
                return False
            elif any(self.observer_coordinate != other.observer_coordinate):
                return False
        elif hasattr(other.observer_coordinate, "__len__"):
            return False
        elif self.observer_coordinate != other.observer_coordinate:
            return False

        if not _attrs_both_none_or_neither(self.vis_labels, other.vis_labels) or tuple(self.vis_labels) != tuple(
            other.vis_labels
        ):
            return False

        if not _attrs_both_none_or_neither(self.instrument, other.instrument) or tuple(self.instrument) != tuple(
            other.instrument
        ):
            return False

        if not (
            _attrs_both_none_or_neither(self.spectral_range, other.spectral_range)
            and apu.quantity.allclose(self.spectral_range, other.spectral_range)
        ):
            return False

        if not (
            _attrs_both_none_or_neither(self.time_range, other.time_range)
            and np.allclose(self.time_range.mjd == other.time_range.mjd)
        ):
            return False
        return True


def _attrs_both_none_or_neither(attr1, attr2):
    if attr1 is None:
        if attr2 is not None:
            return False
    elif attr2 is None:
        return False
    return True


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
