from abc import ABC, abstractmethod
from typing import Callable, Optional
from itertools import chain
from collections import UserList
from dataclasses import dataclass

import numpy as np

__all__ = [
    "circular_gaussian",
    "circular_gaussian_vis",
    "elliptical_gaussian",
    "elliptical_gaussian_vis",
    "GenericSource",
    "Circular",
    "Elliptical",
    "SourceList",
    "SourceFactory",
    "Source",
]


def circular_gaussian(amp, x, y, x0, y0, sigma):
    r"""
    Circular gaussian function sampled at x, y.

    .. math::

        F(x, y) = A \exp{\left(-\frac{(x0-x)^2 + (y0 - y)^2}{2\sigma^2}\right)}


    Parameters
    ----------
    amp :
        Amplitude
    x :
        x coordinates
    y :
        y coordinates
    x0 :
        Center x coordinate
    y0 :
        Center y coordinate
    sigma
        Sigma

    See Also
    --------
    circular_gaussian_vis
    """
    return amp / (2 * np.pi * sigma**2) * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2))


def circular_gaussian_vis(amp, u, v, x0, y0, sigma):
    r"""
    Circular gaussian in Fourier space sampled at u, v.

    .. math::

        F(u, v) = A \exp{\left( -2\pi^2 \sigma^2 (u^2 +v^2 \right)}) \exp( 2\pi i(x0u + y0v))


    Parameters
    ----------
    amp :
        Amplitude
    u :
        u coordinates
    v :
        v coordinates
    x0 :
        Center x coordinate
    y0 :
        Center y coordinate
    sigma
        Sigma

    See Also
    --------
    circular_gaussian
    """
    return amp * np.exp(-2 * np.pi**2 * sigma**2 * (u**2 + v**2)) * np.exp(2j * np.pi * (x0 * u + y0 * v))


def elliptical_gaussian(amp, x, y, x0, y0, sigmax, sigmay, theta):
    r"""
    Elliptical gaussian sampled at x, y.

    .. math::

        x' &= ((x0 - x) \cos(\theta) + ((y0 - y) \sin(\theta)) \\
        y' &= -((x0 - x) \sin(\theta) + ((y0 - y) \cos(\theta)) \\
        F(x, y) &= \frac{A}{(2 \pi \sigma_x \sigma_y)} \exp \left( \frac{x'^2}{2\sigma_x^2} + \frac{y'^2}{\sigma_y^2} \right)


    Parameters
    ----------
    amp :
        Amplitude
    x :
        x coordinates
    y :
        y coordinates
    x0 :
        Center x coordinate
    y0 :
        Center y coordinate
    sigmax :
        Sigma in x direction
    sigma :
        Sigma in y direction
    theta
        Rotation angle in anticlockwise

    See Also
    --------
    elliptical_gaussian_vis
    """
    sint = np.sin(theta)
    cost = np.cos(theta)
    xp = ((x0 - x) * cost) + ((y0 - y) * sint)
    yp = -((x0 - x) * sint) + ((y0 - y) * cost)
    return amp / (2 * np.pi * sigmax * sigmay) * np.exp(-((xp**2 / (2 * sigmax**2)) + (yp**2 / (2 * sigmay**2))))


def elliptical_gaussian_vis(amp, u, v, x0, y0, sigmax, sigmay, theta):
    r"""
    Elliptical gaussian in Fourier space sampled at u, v.

    .. math::

        x' &= u\cos(\theta) +v \sin(\theta) \\
        y' &= -u \sin(\theta) + v \cos(\theta) \\
        F(x, y) &= A \exp \left( -2\pi^2 ((u'^2\sigma_x^2) + (v'^2\sigma_y^2) \right) \exp( 2\pi i(x0u + y0v))

    Parameters
    ----------
    amp :
        Amplitude
    u :
        u coordinates
    v :
        v coordinates
    x0 :
        Center x coordinate
    y0 :
        Center y coordinate
    sigmax :
        Sigma in x direction
    sigma :
        Sigma in y direction
    theta
        Rotation angle in anticlockwise

    See Also
    --------
    elliptical_gaussian
    """
    sint = np.sin(theta)
    cost = np.cos(theta)
    up = cost * u + sint * v
    vp = -sint * u + cost * v
    return (
        amp
        * np.exp(-2 * np.pi**2 * ((up**2 * sigmax**2) + (vp**2 * sigmay**2)))
        * np.exp(2j * np.pi * (x0 * u + y0 * v))
    )


class GenericSource(ABC):
    r"""
    Abstract source class defining the properties and methods.
    """

    _registry: dict[str, Callable] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        key = cls.__name__.lower()
        GenericSource._registry[key] = cls

    @property
    def n_params(self):
        r"""The number of parameters"""
        return len(self.__dict__.keys())

    @property
    @abstractmethod
    def bounds(self) -> list[list[float]]:
        r"""Return the lower and upper bounds of the source."""
        pass

    @property
    @abstractmethod
    def param_list(self) -> list[float]:
        """Return list of parameters if fixed order"""
        pass

    @abstractmethod
    def estimate_bounds(self, *args, **kwargs) -> list[list[float]]:
        """Return estimated bounds"""
        pass


@dataclass()
class Circular(GenericSource):
    amp: float
    x0: float
    y0: float
    sigma: float

    def __init__(self, amp: float, x0: float, y0: float, sigma: float):
        r"""
        Circular gaussian source parameters.

        Parameters
        ----------
        amp :
            Amplitude
        x0 :
            Center x coordinate
        y0 :
            Center y coordinate
        sigma :
            Standard deviation
        """
        self.amp = amp
        self.x0 = x0
        self.y0 = y0
        self.sigma = sigma

    @property
    def bounds(self) -> list[list[float]]:
        return [
            [self.amp / 4, self.x0 - 5 * np.abs(self.sigma), self.x0 - 5 * np.abs(self.sigma), self.sigma / 4],
            [self.amp * 4, self.x0 + 5 * np.abs(self.sigma), self.x0 + 5 * np.abs(self.sigma), self.sigma * 4],
        ]

    @property
    def param_list(self) -> list[float]:
        return [self.amp, self.x0, self.y0, self.sigma]

    def estimate_bounds(self, *args, **kwargs) -> list[list[float]]:
        raise NotImplementedError()


@dataclass
class Elliptical(GenericSource):
    amp: float
    x0: float
    y0: float
    sigmax: float
    sigmay: float
    theta: float

    def __init__(self, amp, x0, y0, sigmax, sigmay, theta):
        r"""
        Elliptical gaussian source parameters.

        ----------
        amp :
            Amplitude
        x0 :
            Center x coordinate
        y0 :
            Center y coordinate
        sigmax :
            Standard deviation in x direction
        sigmay :
            Standard deviation in y direction
        theta :
            Rotation angle in anticlockwise
        """
        self.amp = amp
        self.x0 = x0
        self.y0 = y0
        self.sigmax = sigmax
        self.sigmay = sigmay
        self.theta = theta

    @property
    def bounds(self) -> list[list[float]]:
        return [
            [
                self.amp / 4,
                self.x0 - (5 * np.abs(self.sigmax)),
                self.y0 - (5 * np.abs(self.sigmay)),
                self.sigmax / 4,
                self.sigmay / 4,
                self.theta - 22.5,
            ],
            [
                self.amp * 4,
                self.x0 + (5 * np.abs(self.sigmax)),
                self.y0 + (5 * np.abs(self.sigmay)),
                self.sigmax * 4,
                self.sigmay * 4,
                self.theta + 22.5,
            ],
        ]

    @property
    def param_list(self) -> list[float]:
        return [self.amp, self.x0, self.y0, self.sigmax, self.sigmay, self.theta]

    def estimate_bounds(self, *args, **kwargs) -> list[list[float]]:
        return self.bounds


class SourceList(UserList[GenericSource]):
    r"""
    List of Sources
    """

    def __init__(self, sources: Optional[list[GenericSource]] = None):
        r"""
        List of Sources

        Parameters
        ----------
        sources :
            Sources
        """
        super().__init__(sources)

    @property
    def params(self) -> list[float]:
        r"""Flat list of all parameters for all sources"""
        return list(chain.from_iterable([source.param_list for source in self.data]))

    @property
    def bounds(self) -> list[list[float]]:
        r"""Flat list of upper and lower bounds for all sources"""
        return np.hstack([s.bounds for s in self.data]).tolist()

    @classmethod
    def from_params(cls, sources: "SourceList", params: list[float]) -> "SourceList":
        r"""
        Create a source list from given parameters and sources.

        Parameters
        ----------
        sources :
            List of sources
        params
            Flat list of all parameters for all sources.
        """
        j = 0
        new_sources = cls()
        for i, source in enumerate(sources):
            name = source.__class__.__name__.lower()
            n_params = source.n_params
            new_sources.append(Source(name, *list(params[j : j + n_params])))
            j += n_params

        return new_sources


class SourceFactory:
    r"""
    Source Factory
    """

    def __init__(self, registry: dict[str, Callable]):
        self._registry: dict[str, Callable] = registry

    def __call__(self, shape_type: str, *args, **kwargs) -> GenericSource:
        shape_type = shape_type.lower()
        cls = self._registry.get(shape_type)
        if not cls:
            raise ValueError(f"Unknown shape type: {shape_type}")
        try:
            return cls(*args, **kwargs)
        except TypeError as e:
            raise ValueError(f"Error creating '{shape_type}': {e}")


#: Instance of SourceFactory
Source = SourceFactory(registry=GenericSource._registry)
