"""PSO-specific problem definition, imported only when PSO is requested."""

from typing import Any

import numpy as np
from numpy.typing import NDArray
from pymoo.core.problem import Problem

import astropy.units as apu
from astropy.units import Quantity

from xrayvision.vis_forward_fit.fit_data import VisibilityFitData
from xrayvision.vis_forward_fit.forward_fit import SOURCE_TO_VIS, sources_to_vis
from xrayvision.vis_forward_fit.sources import Loop, SourceList
from xrayvision.visibility import Visibilities


def _sources_to_vis_batch(
    source_list: SourceList, u: NDArray[np.float64], v: NDArray[np.float64], population: NDArray[np.float64]
) -> NDArray[np.complex128]:
    r"""Vectorized `sources_to_vis`, used by the PSO problem to evaluate a whole
    PSO population in one batch of broadcast numpy operations rather than one
    `sources_to_vis` call per particle. ``population`` is (n_particles, n_vars); the
    result is (n_particles, n_vis).

    `circular_gaussian_vis`/`elliptical_gaussian_vis` are plain broadcastable numpy
    expressions, so passing them (n_particles, 1)-shaped parameter columns against a
    (1, n_vis)-shaped u, v already produces the right (n_particles, n_vis) result with
    no changes to those functions. `loop_vis`'s geometry calculation branches on
    scalar conditions (e.g. ``if loop_angle == 0``), which isn't safe to evaluate on an
    array of candidate loop angles at once, so any `Loop` source falls back to
    evaluating one particle at a time via the existing scalar `sources_to_vis`.
    """
    if any(isinstance(source, Loop) for source in source_list):
        vis = np.zeros((population.shape[0], u.shape[0]), dtype=np.complex128)
        for i in range(population.shape[0]):
            cur_sources = source_list.from_params(source_list, population[i])
            vis[i] = sources_to_vis(cur_sources, u, v)
        return vis

    u_row, v_row = u[None, :], v[None, :]
    vis = np.zeros((population.shape[0], u.shape[0]), dtype=np.complex128)
    offset = 0
    for source in source_list:
        n_params = source.n_params
        amp = population[:, offset : offset + 1]
        rest = [population[:, offset + i : offset + i + 1] for i in range(1, n_params)]
        try:
            vis += SOURCE_TO_VIS[source.__class__.__name__](amp, u_row, v_row, *rest)
        except KeyError:
            raise KeyError(f"Unknown source type: {source.__class__.__name__}")
        offset += n_params
    return vis


class VisForwardFitProblem(Problem):  # type: ignore[misc]
    def __init__(
        self,
        u: Quantity,
        v: Quantity,
        visobs: Visibilities,
        sources: SourceList,
        *,
        weighting: str = "auto",
        bounds: NDArray[np.float64] | None = None,
    ) -> None:
        self.fit_data = VisibilityFitData(visobs, weighting=weighting)
        self.u = u.to_value(apu.arcsec**-1)
        self.v = v.to_value(apu.arcsec**-1)
        self.sources = sources.to_canonical(self.fit_data.flux_unit)
        n_var = len(sources.params)

        # `Loop` sources need sigma_min <= sigma_max enforced as a constraint, at their own
        # offset within the flat parameter vector (sigma_min, sigma_max are params 3 and 4 of a Loop).
        self._loop_sigma_offsets = []
        offset = 0
        for source in sources:
            if isinstance(source, Loop):
                self._loop_sigma_offsets.append((offset + 3, offset + 4))
            offset += source.n_params

        xl, xu = self.sources.bounds if bounds is None else bounds
        super().__init__(n_var=n_var, n_obj=1, n_ieq_constr=len(self._loop_sigma_offsets), xl=xl, xu=xu)

    def _evaluate(self, x: NDArray[np.float64], out: dict[str, Any], *args: Any, **kwargs: Any) -> None:
        # x is (n_particles, n_var): the whole PSO population is evaluated in one batch
        # of broadcast numpy operations (see _sources_to_vis_batch)
        if self._loop_sigma_offsets:
            out["G"] = np.column_stack([x[:, i_min] - x[:, i_max] for i_min, i_max in self._loop_sigma_offsets])
        vispred = _sources_to_vis_batch(self.sources, self.u, self.v, x)
        out["F"] = self.fit_data.objective(vispred)
