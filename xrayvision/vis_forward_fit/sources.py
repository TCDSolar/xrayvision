from abc import ABC, abstractmethod
from typing import Callable, Optional
from itertools import chain
from collections import UserList
from dataclasses import dataclass

import numpy as np
from scipy.special import factorial

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
    sigma :
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
    sigma :
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
    sigmay :
        Sigma in y direction
    theta :
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
    sigmay :
        Sigma in y direction
    theta :
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


def loop(amp, x, y, x0, y0, sigma_max, sigma_min, alpha, beta, max_comps=21):
    r"""
    Loop source sampled at x, y.

    The loop source is approximate with a series of equispaced circular gaussians

    Parameters
    ----------
    amp :

    x :

    y :

    x0 :

    y0 :

    sigma_max :

    sigma_min :

    alpha :

    beta :

    max_comps
            Upper limit to number of ~equispaced circles that will be used to approximate loop.

    Returns
    -------

    """
    sig2fwhm = np.sqrt(8 * np.log(2))

    # Calculate the relative strengths of the  sources to reproduce a gaussian and their collective stddev.
    iseq0 = np.arange(max_comps)
    relflux0 = factorial(max_comps - 1) / (factorial(iseq0) * factorial(max_comps - 1 - iseq0)) / 2 ** (max_comps - 1)
    ok = np.flatnonzero(relflux0 > 0.01)  # Just keep; circles that contain; at least 1 % of flux
    ncirc = ok.size
    relflux = relflux0[ok] / relflux0[ok].sum()
    iseq = np.arange(ncirc)
    reltheta = iseq / (ncirc - 1.0) - 0.5  # locations of circles for arclength=1
    factor = np.sqrt((reltheta**2 * relflux).sum()) * sig2fwhm  # FWHM of binomial distribution for arclength=1

    loopangle = beta / factor
    if np.abs(loopangle) >= 2 * np.pi:
        raise ValueError(f"Internal parameterization error - Loop arc {loopangle} exceeds 2 pi.")

    if loopangle == 0.0:
        loopangle = 0.01  # Avoid problems if loopangle = 0

    theta = np.abs(loopangle) * (iseq / (ncirc - 1.0) - 0.5)  # equispaced between + - loopangle / 2
    xloop = np.sin(theta)  # for unit radius of curvature, R
    yloop = np.cos(theta)  # relative to center of curvature

    if loopangle < 0:
        yloop = -yloop  # Sign of loopangle determines sense of loop curvature

    # Determine the size and location of the equivalent separated components in a coord system where x is an axis
    # parallel to the line joining the footpoints. Note that there are combinations of loop angle, sigminor and
    # sigmajor that cannot occur with radius > 1arcsec. In such a case circle radius is set to 1. Such cases will lead
    # to bad solutions and be flagged as such at the end.

    sigminor = sigma_min / sig2fwhm
    sigmajor = sigma_max / sig2fwhm
    fsumx2 = (xloop**2 * relflux).sum()  # scale - free factors describing loop moments for endpoint separation=1
    fsumy = (yloop * relflux).sum()
    fsumy2 = (yloop**2 * relflux).sum()
    loopradius = np.sqrt((sigmajor**2 - sigminor**2) / (fsumx2 - fsumy2 + fsumy**2))
    term = max(
        (sigmajor**2 - loopradius**2 * fsumx2), 0 * loopradius.unit**2
    )  # > 0 condition avoids problems in next step.
    circfwhm = max(sig2fwhm * np.sqrt(term), 1 * loopradius.unit)  # Set minimum to avoid display problems

    cgshift = loopradius * fsumy  # will enable emission centroid location to be unchanged
    relx = xloop * loopradius  # x is axis joining 'footpoints'
    rely = yloop * loopradius - cgshift

    # Calculate source structures for each circle.
    pasep = alpha
    sinus = np.sin(pasep)
    cosinus = np.cos(pasep)

    data = np.zeros(x.shape)
    pixel = [1, 1]
    for i in range(iseq.size):
        flux_new = amp * relflux[i]  #  Split the flux between components.

        x_loc_new = x0 - relx[i] * sinus + rely[i] * cosinus
        y_loc_new = y0 + relx[i] * cosinus + rely[i] * sinus

        x_tmp = ((x - x_loc_new) * cosinus) + ((y - y_loc_new) * sinus)
        y_tmp = -((x - x_loc_new) * sinus) + ((y - y_loc_new) * cosinus)
        x_tmp = 2.0 * np.sqrt(2.0 * np.log(2.0)) * x_tmp / circfwhm
        y_tmp = 2.0 * np.sqrt(2.0 * np.log(2.0)) * y_tmp / circfwhm
        im_tmp = np.exp(-(x_tmp**2.0 + y_tmp**2.0) / 2.0)
        data += im_tmp / (im_tmp.sum() * pixel[0] * pixel[1]) * flux_new

    return data


def loop_vis(amp, u, v, x0, y0, sigma_max, sigma_min, alpha, beta):
    r"""
    Loop source sampled at u, v in Fourier space.

    Parameters
    ----------
    amp :

    u :

    v :

    x0 :

    y0 :

    sigma_max :

    sigma_min :

    alpha :

    beta :

    Returns
    -------

    """
    n_vis = u.size  # number of visibilities

    ncirc0 = 21  # Upper limit to number of ~equispaced circles that will be used to approximate loop.
    sig2fwhm = np.sqrt(8 * np.log(2.0))

    # Calculate the relative strengths of the sources to reproduce a gaussian and their collective stddev.
    iseq0 = np.arange(ncirc0)
    relflux0 = (
        factorial(ncirc0 - 1) / (factorial(iseq0) * factorial(ncirc0 - 1 - iseq0)) / 2 ** (ncirc0 - 1)
    )  # TOTAL(relflux)=1
    ok = np.flatnonzero(relflux0 > 0.01)  # Just keep circles that contain at least 1% of flux
    ncirc = ok.size
    relflux = relflux0[ok] / (relflux0[ok]).sum()
    iseq = np.arange(ncirc)
    reltheta = iseq / (ncirc - 1.0) - 0.5  # locations of circles for arclength=1
    factor = np.sqrt((reltheta**2 * relflux).sum()) * sig2fwhm  # FWHM of binomial distribution for arclength=1

    loopangle = beta / factor
    if np.abs(loopangle).sum() >= 2 * np.pi:
        raise ValueError(f"Internal parameterization error - Loop arc {loopangle} exceeds 2pi.")

    if loopangle == 0:
        loopangle = 0.01  # Avoids problems if loopangle = 0

    theta = np.abs(loopangle) * (iseq / (ncirc - 1.0) - 0.5)  # equispaced between +- loopangle/2
    xloop = np.sin(theta)  # for unit radius of curvature, R
    yloop = np.cos(theta)  # relative to center of curvature

    if loopangle < 0:
        # Sign of loopangle determines sense of loop curvature # Sign of loopangle determines sense of loop curvature
        yloop = -yloop

    # Determine the size and location of the equivalent separated components in a coord system where...
    # x is an axis parallel to the line joining the footpoints
    # Note that there are combinations of loop angle, sigminor and sigmajor that cannot occur with radius>1arcsec.
    # In such a case circle radius is set to 1.  Such cases will lead to bad solutions and be flagged as such at the end.

    # eccen = np.sqrt(1 - (sigma_min**2 / sigma_max**2))
    # sigminor = sigma_min * (1 - eccen ** 2) ** 0.25 / sig2fwhm
    # sigmajor = sigma_max / (1 - eccen ** 2) ** 0.25 / sig2fwhm

    sigminor = sigma_min / sig2fwhm
    sigmajor = sigma_max / sig2fwhm
    fsumx2 = (xloop**2 * relflux).sum()  # scale-free factors describing loop moments for endpoint separation=1
    fsumy = (yloop * relflux).sum()
    fsumy2 = (yloop**2 * relflux).sum()
    loopradius = np.sqrt((sigmajor**2 - sigminor**2) / (fsumx2 - fsumy2 + fsumy**2))
    term = max(
        (sigmajor**2 - loopradius**2 * fsumx2), 0 * sigmajor.unit**2
    )  # >0 condition avoids problems in next step.
    circfwhm = max(sig2fwhm * np.sqrt(term), 1 * sigmajor.unit)  # Set minimum to avoid display problems

    cgshift = loopradius * fsumy
    relx = xloop * loopradius  # x is axis joining 'footpoints'
    rely = yloop * loopradius - cgshift  # will enable emission centroid location to be unchanged

    # Calculate source structures for each circle.
    pasep = alpha  # position angle of line joining arc endpoints
    x_loc_new = x0 - relx * np.sin(pasep) + rely * np.cos(pasep)
    y_loc_new = y0 + relx * np.cos(pasep) + rely * np.sin(pasep)

    flux_new = amp * relflux  # Split the flux between components.

    arg = (-(np.pi**2) * circfwhm**2) / (4 * np.log(2)) * (u**2 + v**2)
    relvis = np.exp(arg)

    vis = np.zeros(n_vis, dtype=np.complex128)
    for j in range(ncirc):
        vis += flux_new[j] * relvis * np.exp(2j * np.pi * (x_loc_new[j] * u + y_loc_new[j] * v))
    return vis


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
            [self.amp / 4, self.x0 - 5 * np.abs(self.sigma), self.y0 - 5 * np.abs(self.sigma), self.sigma / 4],
            [self.amp * 4, self.x0 + 5 * np.abs(self.sigma), self.y0 + 5 * np.abs(self.sigma), self.sigma * 4],
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

        Parameters
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
