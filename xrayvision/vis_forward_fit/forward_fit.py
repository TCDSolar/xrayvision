from typing import Union, Callable, Optional

import astropy.units as apu
import numpy as np
from astropy.units import Quantity, quantity_input
from numpy.typing import NDArray
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
    Loop,
    SourceList,
    circular_gaussian,
    circular_gaussian_vis,
    elliptical_gaussian,
    elliptical_gaussian_vis,
    loop,
    loop_vis,
)
from xrayvision.visibility import Visibilities

__all__ = ["SOURCE_TO_IMAGE", "SOURCE_TO_VIS", "sources_to_image", "sources_to_vis", "vis_forward_fit"]

#: Mapping of sources to image generation functions
SOURCE_TO_IMAGE: dict[str, Callable] = {
    Circular.__name__: circular_gaussian,
    Elliptical.__name__: elliptical_gaussian,
    Loop.__name__: loop,
}

#: Mapping of sources to visibility generation functions
SOURCE_TO_VIS: dict[str, Callable] = {
    Circular.__name__: circular_gaussian_vis,
    Elliptical.__name__: elliptical_gaussian_vis,
    Loop.__name__: loop_vis,
}


def sources_to_image(
    source_list: SourceList,
    shape: Quantity[apu.pix],
    pixel_size: Quantity[apu.arcsec / apu.pix],
    center=(0, 0) * apu.arcsec,
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
    image = None
    x = generate_xy(shape[1], pixel_size=pixel_size[1], phase_center=center[1])
    y = generate_xy(shape[0], pixel_size=pixel_size[0], phase_center=center[0])
    x, y = np.meshgrid(x, y)
    for source in source_list:
        try:
            if image is None:
                image = SOURCE_TO_IMAGE[source.__class__.__name__](
                    *[source.param_list[0], x, y, *source.param_list[1:]]
                )
            else:
                image += SOURCE_TO_IMAGE[source.__class__.__name__](
                    *[source.param_list[0], x, y, *source.param_list[1:]]
                )
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
    if method.casefold() == "pso":
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
            [getattr(p, "value", p) for p in sources.params],
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
        Method to use any of those supported methods by `scipy.optimize.minimize` or 'PSO' for particle swarm optimization
    """
    if method is None:
        method = "Nelder-Mead"
    sources_out, res = _vis_forward_fit_minimise(vis, sources, method=method)
    # add units back
    sources_out = SourceList.from_params(
        sources, [pout * getattr(pin, "unit", 1) for pin, pout in zip(sources.params, sources_out.params)]
    )
    image = sources_to_image(sources_out, shape, pixel_size)
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
        super().__init__(n_var=n_var, n_obj=1, n_ieq_constr=1, xl=xl, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        out["G"] = x[3] - x[4]
        cur_sources = self.sources.from_params(self.sources, x)
        vispred = sources_to_vis(cur_sources, self.u.value, self.v.value)
        vispred_ri = np.hstack([vispred.real, vispred.imag])
        out["F"] = np.sum(np.abs(self.visobs_ri.value - vispred_ri) ** 2)

    # def _evaluate(self, x, out, *args, **kwargs):
    #     # sigmamin < sigmamax -> sigmamin - sigmamax <= 0
    #     out["G"] = x[3] - x[4]
    #     out["F"] = 1e10
    #
    #     # wrap angles
    #     eps = 1e-7
    #     # Alpha (Rotation): Wrap between -pi/2 and pi/2
    #     half_pi = np.pi / 2
    #     x[-2] = ((x[-2] + half_pi) % np.pi) - half_pi
    #     # Beta ("Length"): Clip between 0 and pi
    #     x[-1] = np.clip(x[-1], eps, np.pi)
    #
    #     s_min = min(x[3], x[4])
    #     s_max = max(x[3], x[4])
    #
    #     x[3] = s_min
    #     x[4] = s_max
    #
    #     cur_sources = self.sources.from_params(self.sources, x)
    #
    #     try:
    #         vispred = sources_to_vis(cur_sources, self.u.value, self.v.value)
    #         vispred_ri = np.hstack([vispred.real, vispred.imag])
    #         if not np.isfinite(np.sum(np.abs(self.visobs_ri.value - vispred_ri) ** 2)):
    #             return
    #         out["F"] = np.sum(np.abs(self.visobs_ri.value - vispred_ri) ** 2)
    #     except ValueError:
    #         out["F"] = 1e10
