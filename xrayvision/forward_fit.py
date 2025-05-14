from enum import Enum
from typing import Callable
from itertools import chain
from dataclasses import field, dataclass

import astropy.units as apu
import numpy as np
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize as moo_minimize
from scipy.optimize import OptimizeResult, minimize

from xrayvision.transform import generate_xy
from xrayvision.visibility import Visibilities

__all__ = ["vis_forward_fit"]


def circular_gaussian(amp, x, y, x0, y0, sigma):
    r"""
    Circulate gaussian function

    Parameters
    ----------
    amp : float
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
        Standard deviation

    Returns
    -------
    np.ndarray[float]
    """
    return amp / (2 * np.pi * sigma**2) * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2)) * 1 * 1


def circular_gaussian_vis(amp, u, v, x0, y0, sigma):
    r"""
    Circular gaussian in Fourier space sampled at u, v.
    Parameters
    ----------
    amp
    u
    v
    x0
    y0
    sigma

    Returns
    -------

    """
    return amp * np.exp(-2 * np.pi**2 * sigma**2 * (u**2 + v**2)) * np.exp(2j * np.pi * (x0 * u + y0 * v))


def elliptical_gaussian(amp, x, y, x0, y0, sigmax, sigma, theta):
    r"""
    Elliptical gaussian in real space sampled at x, y.

    Parameters
    ----------
    amp
    x
    y
    x0
    y0
    sigmax
    sigma
    theta

    Returns
    -------

    """
    pass


def elliptical_gaussian_vis(amp, u, v, x0, y0, sigma, theta):
    r"""
    Elliptical gaussian in Fourier space sampled at u, v.

    Parameters
    ----------
    amp
    u
    v
    x0
    y0
    sigma
    theta

    Returns
    -------

    """


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


SOURCE_TO_IMAGE: dict[SourceType, Callable] = {
    SourceType.CIRCULAR: circular_gaussian,
    SourceType.ELLIPTICAL: elliptical_gaussian,
}

SOURCE_TO_VIS: dict[SourceType, Callable] = {
    SourceType.CIRCULAR: circular_gaussian_vis,
    SourceType.ELLIPTICAL: elliptical_gaussian_vis,
}


def sources_to_image(source_list: SourceList, shape) -> np.ndarray[float]:
    r"""
    Create an image from a list of sources.

    Parameters
    ----------
    source_list :
        List of sources and their parameters
    shape :
        Shape of the image create
    Returns
    -------

    """
    image = np.zeros(shape)
    x = generate_xy(shape[1] * apu.pixel).value
    y = generate_xy(shape[0] * apu.pixel).value
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


def vis_forward_fit(
    visobs: Visibilities, sources: SourceList, method: str = "Nelder-Mead"
) -> tuple[SourceList, OptimizeResult]:
    r"""

    Parameters
    ----------
    visobs
    sources
    method

    Returns
    -------

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
