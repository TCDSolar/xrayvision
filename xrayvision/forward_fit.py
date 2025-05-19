from enum import Enum
from typing import Union, Callable, Optional
from itertools import chain
from dataclasses import field, dataclass

import astropy.units as apu
import numpy as np
from astropy.units import Quantity, quantity_input
from hypothesis.extra.numpy import NDArray
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize as moo_minimize
from scipy.optimize import OptimizeResult, minimize
from sunpy.map import Map

from xrayvision.imaging import generate_header
from xrayvision.transform import generate_xy
from xrayvision.visibility import Visibilities

# __all__ = ["vis_forward_fit"]


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


class SourceType(Enum):
    CIRCULAR = "circle"
    ELLIPTICAL = "elliptical"


@dataclass()
class Source:
    source_type: SourceType
    params: list[float]

    def __init__(self, source_type: SourceType, params: list[float]):
        self.source_type = SourceType(source_type)
        self.params = params if params else []


@dataclass()
class SourceList:
    sources: list[Source] = field(default_factory=list)

    @property
    def params(self):
        return list(chain.from_iterable([source.params for source in self.sources]))

    def from_list(self, params):
        j = 0
        for i, source in enumerate(self.sources):
            n_params = len(source.params)
            self.sources[i].params = list(params[j : j + n_params])
            j += n_params


# class SourceFactory:
#     def __init__(self, registry):
#         self._registry = registry
#
#     def __call__(self, shape_type: str, *args, **kwargs):
#         shape_type = shape_type.lower()
#         cls = self._registry.get(shape_type)
#         if not cls:
#             raise ValueError(f"Unknown shape type: {shape_type}")
#         try:
#             return cls(*args, **kwargs)
#         except TypeError as e:
#             raise ValueError(f"Error creating '{shape_type}': {e}")
#
# class GenericSource(ABC):
#     _registry = {}
#
#     def __init_subclass__(cls, **kwargs):
#         super().__init_subclass__(**kwargs)
#         key = cls._name.lower()
#         GenericSource._registry[key] = cls
#
#     @abstractmethod
#     def param_list(self) -> list[float]:
#         """Return list of parameters if fixed order"""
#         pass
#
#     @abstractmethod
#     def estimate_bounds(self, *args, **kwargs) -> list[float]:
#         """Return estimated bounds"""
#         pass
#
# Shape = SourceFactory(registry=GenericSource._registry)
#
# @dataclass()
# class CircularGuassian(GenericSource):
#     _name = 'circular'
#
#     amp : float
#     x0 : float
#     y0 :  float
#     sigma : float
#
#     def __init__(self, amp, x0, y0, sigma):
#         breakpoint()
#         self.amp = amp
#         self.x0 = x0
#         self.y0 = y0
#         self.sigma = sigma
#
#     @property
#     def bounds(self):
#         return ([self.amp/4, self.x0 - 5 * np.abs(self.sigma), self.x0 - 5 *np.abs(self.sigma), self.sigma/4],
#                 [self.amp*4, self.x0 + 5 * np.abs(self.sigma), self.x0 + 5 * np.abs(self.sigma), self.sigma*4])
#
#     def param_list(self) -> list[float]:
#         return [self.amp, self.x0, self.y0, self.sigma]
#
#     def estimate_bounds(self, *args, **kwargs) -> list[float]:
#         pass
#
# @dataclass
# class EllipticalGuassian(GenericSource):
#     _name = 'elliptical'
#
#     def __init__(self, amp, x0, y0, sigmax, sigmay, theta):
#         self.amp = amp
#         self.x0 = x0
#         self.y0 = y0
#         self.sigmax = sigmax
#         self.sigmay = sigmay
#         self.theta = theta
#
#     @property
#     def bounds(self):
#         return ([self.amp / 4, self.x0 - 5 * np.abs(self.sigma), self.x0 - 5 * np.abs(self.sigma), self.sigmax / 4,
#                  self.sigmay / 4, self.theta - 22.5],
#                 [self.amp * 4, self.x0 + 5 * np.abs(self.sigma), self.x0 + 5 * np.abs(self.sigma), self.sigmax * 4,
#                  self.sigmay * 4, self.theta + 22.5])
#
#     def param_list(self) -> list[float]:
#         return [self.amp, self.x0, self.y0, self.sigmax, self.sigmay, self.theta]
#
#     def estimate_bounds(self, *args, **kwargs) -> list[float]:
#         pass

SOURCE_TO_IMAGE: dict[SourceType, Callable] = {
    SourceType.CIRCULAR: circular_gaussian,
    SourceType.ELLIPTICAL: elliptical_gaussian,
}

SOURCE_TO_VIS: dict[SourceType, Callable] = {
    SourceType.CIRCULAR: circular_gaussian_vis,
    SourceType.ELLIPTICAL: elliptical_gaussian_vis,
}


def sources_to_image(
    source_list: SourceList, shape: Quantity[apu.pix], pixel_size: Quantity[apu.arcsec / apu.pix]
) -> np.ndarray[float]:
    r"""
    Create an image from a list of sources.

    Parameters
    ----------
    source_list :
        List of sources and their parameters
    shape :
        Shape of the image create
    pixel_size :
        Size
    Returns
    -------

    """
    image = np.zeros(shape.value.astype(int))
    x = generate_xy(shape[1]).value
    y = generate_xy(shape[0]).value
    x, y = np.meshgrid(x, y)
    for source in source_list.sources:
        try:
            image += SOURCE_TO_IMAGE[source.source_type](*[source.params[0], x, y, *source.params[1:]])
        except KeyError:
            raise KeyError(f"Unknown source type: {source.source_type}")
    return image


def sources_to_vis(source_list: SourceList, u, v) -> np.ndarray[np.complex128]:
    r"""
    Create visibilities from a list of sources.

    Parameters
    ----------
    source_list
    u :
        u coordinates to evaluate sources
    v :
        u coordinates to evaluate sources

    Returns
    -------
    vis :
        Complex visibilities
    """
    vis = np.zeros(u.shape, dtype=np.complex128)
    for source in source_list.sources:
        try:
            vis += SOURCE_TO_VIS[source.source_type](*[source.params[0], u, v, *source.params[1:]])
        except KeyError:
            raise KeyError(f"Unknown source type: {source.source_type}")
    return vis


def _vis_forward_fit_minimise(
    visobs: Visibilities, sources: SourceList, method: str = "Nelder-Mead"
) -> tuple[SourceList, OptimizeResult]:
    r"""

    Parameters
    ----------
    visobs :
        Input Visibilities
    sources :
        List of sources
    method :
        Method to use for the minimisation

    """
    if method == "PSO":
        problem = VisForwardFitProblem(visobs.u, visobs.v, visobs, sources)
        algo = PSO(pop=100)
        res = moo_minimize(problem, algo)
        sources.from_list(res.X)
    else:
        visobs_ri = np.hstack([visobs.visibilities.real, visobs.visibilities.imag])

        def objective(x, u, v, visobs_ri, sources):
            sources.from_list(x)
            vispred = sources_to_vis(sources, u.value, v.value)
            vispred_ri = np.hstack([vispred.real, vispred.imag])
            return np.sum(np.abs(visobs_ri.value - vispred_ri) ** 2)

        flux_estimate = np.max(np.abs(visobs.visibilities)).value
        res = minimize(
            objective,
            sources.params,
            (visobs.u, visobs.v, visobs_ri, sources),
            method=method,
            bounds=[
                (x, y)
                for x, y in zip(
                    [0.1 * flux_estimate, -15, -15, 1] * len(sources.sources),
                    [1.5 * flux_estimate, 15, 15, 15] * len(sources.sources),
                )
            ],
        )
        sources.from_list(res.x)
    return sources, res


@quantity_input()
def vis_forward_fit(
    vis: Visibilities,
    sources: SourceList,
    shape: Quantity[apu.pix],
    pixel_size: Quantity[apu.arcsec / apu.pix],
    map: Optional[bool] = True,
    method: str = "Nelder-Mead",
) -> Union[Quantity, NDArray[np.float64]]:
    r"""
    Visibility forward fit method.

    Parameters
    ----------
    vis :
        Input visibilities
    sources :
        List of sources and their initial parameters
    shape :
        Shape of the image create
    pixel_size :
        Pixel size
    map :
        Return a `Map`
    method :
        Method to use any of those supported methods by `scipy.optimize.minimize` and 'PSO'.
    """
    sources, res = _vis_forward_fit_minimise(vis, sources, method=method)
    image = sources_to_image(sources, shape, pixel_size)
    if map:
        header = generate_header(vis, shape=shape, pixel_size=pixel_size)
        return Map((image, header))
    return image


class VisForwardFitProblem(ElementwiseProblem):
    def __init__(self, u, v, visobs: Visibilities, sources: SourceList):
        self.u = u
        self.v = v
        self.visobs = visobs
        self.visobs_ri = np.hstack([visobs.visibilities.real, visobs.visibilities.imag])
        self.sources = sources
        n_var = len(sources.params)
        flux_estimate = np.max(np.abs(visobs.visibilities)).value
        xl = [0.1 * flux_estimate, -15, -15, 1] * len(sources.sources)
        xu = [1.5 * flux_estimate, 15, 15, 15] * len(sources.sources)
        super().__init__(n_var=n_var, n_obj=1, xl=xl, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        self.sources.from_list(x)
        vispred = sources_to_vis(self.sources, self.u.value, self.v.value)
        vispred_ri = np.hstack([vispred.real, vispred.imag])
        out["F"] = np.sum(np.abs(self.visobs_ri.value - vispred_ri) ** 2)
