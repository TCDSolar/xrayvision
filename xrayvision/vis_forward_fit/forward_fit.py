from typing import Union, Callable, Optional

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
from xrayvision.vis_forward_fit.sources import (
    Circular,
    Elliptical,
    SourceList,
    circular_gaussian,
    circular_gaussian_vis,
    elliptical_gaussian,
    elliptical_gaussian_vis,
)
from xrayvision.visibility import Visibilities

__all__ = ["SOURCE_TO_IMAGE", "SOURCE_TO_VIS", "sources_to_image", "sources_to_vis", "vis_forward_fit"]

#: Mapping of sources to image generation functions
SOURCE_TO_IMAGE: dict[str, Callable] = {
    Circular.__name__: circular_gaussian,
    Elliptical.__name__: elliptical_gaussian,
}

#: Mapping of sources to visibility generation functions
SOURCE_TO_VIS: dict[str, Callable] = {
    Circular.__name__: circular_gaussian_vis,
    Elliptical.__name__: elliptical_gaussian_vis,
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
    for source in source_list:
        try:
            image += SOURCE_TO_IMAGE[source.__class__.__name__](*[source.param_list[0], x, y, *source.param_list[1:]])
        except KeyError:
            raise KeyError(f"Unknown source type: {source.__class__.__name__}")
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
    for source in source_list:
        try:
            vis += SOURCE_TO_VIS[source.__class__.__name__](*[source.param_list[0], u, v, *source.param_list[1:]])
        except KeyError:
            raise KeyError(f"Unknown source type: {source.__class__.__name__}")
    return vis


def _vis_forward_fit_minimise(
    visobs: Visibilities, sources: SourceList, method: str
) -> tuple[SourceList, OptimizeResult]:
    r"""
    Internal minimisation function

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
        sources_fit = sources.from_params(sources, res.X)
    else:
        visobs_ri = np.hstack([visobs.visibilities.real, visobs.visibilities.imag])

        def objective(x, u, v, visobs_ri, sources):
            cur_sources = sources.from_params(sources, x)
            vispred = sources_to_vis(cur_sources, u.value, v.value)
            vispred_ri = np.hstack([vispred.real, vispred.imag])
            return np.sum(np.abs(visobs_ri.value - vispred_ri) ** 2)

        res = minimize(
            objective,
            sources.params,
            (visobs.u, visobs.v, visobs_ri, sources),
            method=method,
            bounds=[(x, y) for x, y in zip(*sources.bounds)],
        )
        sources_fit = sources.from_params(sources, res.x)
    return sources_fit, res


@quantity_input()
def vis_forward_fit(
    vis: Visibilities,
    sources: SourceList,
    shape: Quantity[apu.pix],
    pixel_size: Quantity[apu.arcsec / apu.pix],
    map: Optional[bool] = True,
    method: Optional[str] = "Nelder-Mead",
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
    if method is None:
        method = "Nelder-Mead"
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
        xl, xu = sources.bounds
        super().__init__(n_var=n_var, n_obj=1, xl=xl, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        cur_sources = self.sources.from_params(self.sources, x)
        vispred = sources_to_vis(cur_sources, self.u.value, self.v.value)
        vispred_ri = np.hstack([vispred.real, vispred.imag])
        out["F"] = np.sum(np.abs(self.visobs_ri.value - vispred_ri) ** 2)
